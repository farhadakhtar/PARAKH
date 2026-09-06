"""PARAKH -- supervised multi-label DATA-QUALITY DEFECT channel study (baseline-first).

WHAT THIS IS
    A single self-contained Kaggle script that trains one detector per injected
    data-quality defect channel on the SYNTHETIC development corpus, using the
    generator's defect ledger as exact multi-label supervision.

WHAT THIS IS NOT
    This is not fraud detection. No fraud label exists anywhere in this project.
    Every positive here is an INJECTED DATA-QUALITY DEFECT in a synthetic file
    (a placeholder token, an unparseable date, a sign flip, a cloned identifier).
    Nothing in this script says anything about any real person or organisation.

WHY THE SHAPE OF THIS SCRIPT IS ODD (READ THIS BEFORE CHANGING IT)
    A previous round trained an unsupervised consistency model that plateaued at
    defect ROC-AUC 0.53-0.58 from epoch 10 to epoch 100. This round replaces it
    with direct supervision. Three neural architectures were proposed for that
    supervised round and ALL THREE WERE REJECTED by independent critics who
    measured against this corpus. Their grounds, condensed:

      * Most channels are deterministic single-rule injections. `placeholder` is
        a closed 7-token vocabulary; `format_unparseable` is literally "the
        parser raised"; `negative_amount` is a sign bit; `extreme_value` is the
        literal constant 1e300; `date_order` is a negative date difference;
        `duplicate_work_id` is an exact string repeat. Once the features below
        exist, these are linear predicates. A network cannot earn credit on them.
      * The channels where a network could in principle earn credit
        (extreme_value n=60, duplicate_work_id n=100) have too few positives for
        any paired difference to clear its confidence interval. The study was not
        powered for its own claim.
      * Raw 1e300 fed to a scaler is +inf in float32; the proposed nets would
        have NaN-ed in epoch 1 and the operator would have debugged label
        alignment instead of the input pipeline.

    So: LOGISTIC REGRESSION and GRADIENT BOOSTING are the PRIMARY models here.
    The MLP is a CHALLENGER whose only job is to test the critics' rejection. It
    is adopted for a channel only if a PAIRED bootstrap confidence interval on
    (MLP AUC - LR AUC) excludes zero. It is deliberately NOT tuned to win.
    A run that concludes "logistic regression is sufficient" is a SUCCESS.

DISCIPLINE THIS SCRIPT ENFORCES (each of these is a fix for a measured error)
    * Every metric is printed with its n, its n_pos and a 95% interval. A prior
      round published AUC 0.699 from 171 positives when the true value at
      n=3,382 was 0.573, with non-overlapping intervals. Never again.
    * Every metric is printed beside a majority-class baseline AND a
      hand-written deterministic rule. An accuracy without a baseline is
      uninterpretable; on this corpus the rule is usually the real ceiling.
    * Per-channel only. The aggregate is dominated by the most prevalent channel
      and hides everything else. Any macro number printed is labelled as such.
    * An OBSERVABLE-EVIDENCE CEILING is computed per channel from the corpus.
      The generator runs its missing/placeholder passes LAST and overwrites cells
      that earlier channels already corrupted, while the ledger keeps the earlier
      label. Recall on those rows is unreachable by any model. Reported without
      the ceiling, a recall of 0.85 reads as a modelling shortfall when it is
      irreducible label noise.
    * Two GENERATOR-INVARIANT feature blocks (the work_id year/state decode, and
      the name-vs-district agreement test) are near-perfect separators here
      because the generator never produces a legitimate mismatch. They are kept,
      but the affected channels are reported TWICE -- with and without the block
      -- and the WITHOUT number is the headline.
    * Features that encode the injector's serialisation rather than the domain
      are DELETED, not quarantined: the sanction rounding-granularity bit
      (np.round artifact) and work_id serial-vs-row-position (emission order).
      An assertion below fails the build if such a column reappears.
    * A negative control (a random normal column pushed through the identical
      pipeline) and a shuffled-label refit are reported per channel. Both must
      land at 0.50. They are tripwires for the exact class of pipeline fault --
      fold-boundary leakage, transductive fitting, index misalignment -- that
      produced the 0.699-vs-0.573 error.

TRANSDUCTIVITY, DECLARED ONCE AND HONESTLY
    Cross-row duplicate structure (work_id collision counts, exact work_name
    multiplicity, MinHash near-duplicate neighbourhood) is computed over the
    WHOLE FILE, not per fold. That is deliberate and it matches deployment: a
    submitted register is screened as one batch, so the counts a deployed
    detector sees are whole-file counts. To keep it honest the split is GROUPED:
    a duplicate clique never straddles the train/test boundary, so a test row's
    count means the same thing it meant at training time. Every duplicate-channel
    metric in the output is labelled "whole-file blocking, transductive by
    construction". Everything else -- scalers, vocabularies, peer medians, TF-IDF,
    SVD, frequency tables -- is fitted on TRAIN ROWS ONLY.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import pickle
import re
import sys
import time
import unicodedata
import zlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------------------
# 0. ENVIRONMENT
# --------------------------------------------------------------------------------------

# str.replace(..., regex=True) and modern groupby semantics need pandas >= 1.4. A stale
# Kaggle image would otherwise fail deep inside feature engineering with a confusing
# error, so fail loudly at the top instead.
_PD_MAJOR, _PD_MINOR = (int(x) for x in pd.__version__.split(".")[:2])
if (_PD_MAJOR, _PD_MINOR) < (1, 4):
    raise RuntimeError(f"pandas >= 1.4 required, found {pd.__version__}")

RANDOM_SEED = 20240917
rng_global = np.random.default_rng(RANDOM_SEED)

N_BOOTSTRAP = 1000          # bootstrap resamples for every interval
EPOCH_CEILING = 200         # a CEILING, not a target: early stopping decides the epoch
MIN_POSITIVES = 50          # channel rejection floor
MAX_PREVALENCE = 0.50       # channel rejection ceiling: above this it is a base rate
MIN_PEER_GROUP = 30         # smallest train-fold peer cell that may define a reference
SVD_DIMS = 64

OUT_DIR = "/kaggle/working/mlp_out" if os.path.isdir("/kaggle/working") else "./mlp_out"
os.makedirs(OUT_DIR, exist_ok=True)

try:
    import tensorflow as tf

    HAVE_TF = True
except Exception:  # pragma: no cover - exercised only on images without TF
    tf = None
    HAVE_TF = False

RAW_COLUMNS = [
    "work_id", "work_name", "district", "state",
    "sanction_amount", "amount_spent",
    "date_proposal", "date_approval", "date_completion",
    "implementing_agency", "vendor_name", "status",
]
AMOUNT_COLS = ["sanction_amount", "amount_spent"]
DATE_COLS = ["date_proposal", "date_approval", "date_completion"]
TYPED_COLS = AMOUNT_COLS + DATE_COLS
TEXT_COLS = [c for c in RAW_COLUMNS if c not in TYPED_COLS]

SCHEME_START = dt.date(1993, 1, 1).toordinal()  # domain constant, NOT an empirical quantile

# Feature blocks. Ablations drop whole blocks, so the block name has to travel with the
# column name; every column is emitted as "<block>__<name>".
BLOCKS = ("parse", "amount", "peer", "date", "idcons", "dup", "geo", "freq", "text", "control")

# Channels whose headline number must exclude a generator-invariant block. The critics
# measured these as near-perfect separators that exist only because this generator never
# emits a legitimate mismatch, so the with-block number measures np.random.choice, not
# detection skill.
ABLATIONS = {
    "duplicate_name": ("geo",),      # cloned name carries the SOURCE row's district suffix
    "pre_scheme_date": ("idcons",),  # work_id embeds the pre-shift proposal year
}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def rule_banner(title: str) -> None:
    log("")
    log("=" * 100)
    log(title)
    log("=" * 100)


# --------------------------------------------------------------------------------------
# 1. TEXT NORMALISATION
# --------------------------------------------------------------------------------------

_DEVANAGARI_DIGITS = {ord(c): str(i) for i, c in enumerate("०१२३४५६७८९")}
_ZERO_WIDTH = {0x200C: None, 0x200D: None, 0xFEFF: None}
_NUKTA_FOLD = {
    0x0958: "क़", 0x0959: "ख़", 0x095A: "ग़",
    0x095B: "ज़", 0x095C: "ड़", 0x095D: "ढ़",
    0x095E: "फ़", 0x095F: "य़",
}
_DANDA = {0x0964: ".", 0x0965: "."}
_TRANSLATE = {**_DEVANAGARI_DIGITS, **_ZERO_WIDTH, **_NUKTA_FOLD, **_DANDA}
_WS_RE = re.compile(r"\s+")


def normalise_indic(text: str) -> str:
    """NFC + nukta decomposition + ZWJ/ZWNJ strip + Devanagari digits + danda + casefold.

    Deliberately does NOT map 'null'/'none'/'nan' to the empty string. A normaliser that
    swallows those tokens destroys the `placeholder` channel -- the second largest
    trainable channel -- by folding it into `missing`, and the loss then looks like "the
    model cannot learn placeholder".
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text).translate(_TRANSLATE)
    return _WS_RE.sub(" ", t).strip().casefold()


def normalise_series(s: pd.Series) -> pd.Series:
    return pd.Series([normalise_indic(x) for x in s.astype(str)], index=s.index, dtype=object)


# --------------------------------------------------------------------------------------
# 2. CORPUS LOADING (with a standalone fallback generator)
# --------------------------------------------------------------------------------------

def find_input_dir() -> str | None:
    candidates = [
        os.environ.get("PARAKH_DATA_DIR"),
        "/kaggle/input/parakh-corpus",
        "/kaggle/input/parakh-corpus/data",
        "./data",
        "../data",
        ".",
    ]
    for c in candidates:
        if not c:
            continue
        if os.path.isfile(os.path.join(c, "synthetic_dataset.csv")) and os.path.isfile(
            os.path.join(c, "ground_truth_ledger.json")
        ):
            return c
    return None


def load_corpus(path: str) -> pd.DataFrame:
    """Read every cell as a RAW STRING.

    dtype=str + keep_default_na=False + na_filter=False is load-bearing, not defensive
    style. pandas' default na_values contains 'N/A', 'NA' and 'NULL', which are three of
    the generator's placeholder tokens; the default reader silently converts thousands of
    `placeholder` cells into NaN and merges that channel into `missing` before the first
    feature is computed. It also repairs amounts and dates, destroying the surface form
    that IS the evidence for both format channels.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    missing_cols = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"corpus is missing required columns: {missing_cols}")
    return df[RAW_COLUMNS].reset_index(drop=True)


def generate_fallback_corpus(n_rows: int = 20000, seed: int = 42):
    """Emit a corpus + ledger with the same 12 columns and 11 channels, so the script runs
    standalone when the Kaggle dataset is not attached.

    The injection ORDER matters and is reproduced faithfully: the missing pass runs last
    and overwrites evidence that earlier channels wrote, which is what creates the
    observable-evidence ceilings this script measures and reports.
    """
    rng = np.random.default_rng(seed)
    actions = ["Construction of", "Providing and Fixing of", "Upgradation of", "Repair of",
               "Renovation of", "Widening of", "Strengthening of", "Extension of", "Desilting of"]
    work_types = {
        "Village Road": (4e5, 1.2e6), "Community Hall": (6e5, 2.0e6), "Anganwadi Centre": (3e5, 9e5),
        "Library Building": (5e5, 1.5e6), "Drainage Line": (2e5, 8e5), "Culvert": (3e5, 1.1e6),
        "School Boundary Wall": (2.5e5, 7e5), "Water Tank": (4e5, 1.4e6), "Street Light Network": (1.5e5, 6e5),
        "Public Toilet Block": (2e5, 7e5), "Check Dam": (7e5, 2.4e6), "Bus Shelter": (1e5, 4e5),
        "Panchayat Bhawan": (8e5, 2.6e6), "Playground": (3e5, 1.0e6), "Cremation Shed": (2e5, 6e5),
        "Pond Rejuvenation": (5e5, 1.8e6), "Health Sub Centre": (9e5, 2.8e6), "Market Shed": (4e5, 1.3e6),
        "Foot Bridge": (6e5, 2.1e6), "Solar Pump Unit": (2e5, 8e5),
    }
    states = {"Gujarat": "GJ", "West Bengal": "WB", "Bihar": "BR", "Rajasthan": "RJ",
              "Maharashtra": "MH", "Odisha": "OD", "Assam": "AS", "Kerala": "KL",
              "Punjab": "PB", "Haryana": "HR"}
    districts = {}
    for st in states:
        for k in range(10):
            districts[f"{st.split()[0][:4]}dist{k}"] = st
    dist_names = list(districts)
    localities = [f"Gram Panchayat {n}" for n in
                  ["Kotwali", "Shivpuri", "Rampur", "Bansi", "Naugaon", "Kharia", "Palri", "Tikri",
                   "Baghra", "Sultanpur", "Mahua", "Dhanpur"]]
    vendors = [f"{n} {s}" for n in ["Kulkarni", "Yadav", "Sharma", "Iyer", "Bose", "Patel", "Rao", "Singh"]
               for s in ["Developers", "Builders", "Infra", "Constructions"]]
    statuses = ["completed", "approved", "ongoing", "proposed", "closed"]

    wt_names = list(work_types)
    rows = []
    for i in range(n_rows):
        dname = dist_names[rng.integers(len(dist_names))]
        st = districts[dname]
        wt = wt_names[rng.integers(len(wt_names))]
        lo, hi = work_types[wt]
        sanction = round(float(np.exp(rng.uniform(math.log(lo), math.log(hi)))) / 100.0) * 100.0
        spent = round(sanction * float(rng.uniform(0.35, 1.25)), 2)
        y = int(rng.integers(2015, 2023))
        dp = dt.date(y, int(rng.integers(1, 13)), int(rng.integers(1, 28)))
        da = dp + dt.timedelta(days=int(rng.integers(5, 210)))
        dc = da + dt.timedelta(days=int(rng.integers(30, 720)))
        rows.append({
            "work_id": f"MPL-{states[st]}-{y}-{i + 1:06d}",
            "work_name": f"{actions[rng.integers(len(actions))]} {wt} at "
                         f"{localities[rng.integers(len(localities))]}, {dname}",
            "district": dname, "state": st,
            "sanction_amount": f"{sanction}", "amount_spent": f"{spent}",
            "date_proposal": dp.isoformat(), "date_approval": da.isoformat(),
            "date_completion": dc.isoformat(),
            "implementing_agency": f"Block Development Office, {dname}",
            "vendor_name": vendors[rng.integers(len(vendors))],
            "status": statuses[rng.integers(len(statuses))],
        })
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    ledger = defaultdict(list)

    def mark(i, label):
        ledger[int(i)].append(label)

    def pick(k):
        return rng.choice(n_rows, size=k, replace=False)

    # 1. duplicate_name (40% exact clone, 60% perturbed clone). Only the TARGET is labelled.
    for t in pick(1000):
        s = int(rng.integers(n_rows))
        if s == t:
            continue
        name = df.at[s, "work_name"]
        if rng.random() < 0.4:
            df.at[t, "work_name"] = name
            mark(t, "duplicate_name:exact")
        else:
            v = int(rng.integers(0, 4))
            if v == 0:
                name = name.replace("Construction of", "Constn. of").upper()
            elif v == 1:
                name = name.replace(" at ", "  at  ")
            elif v == 2:
                name = name + " - Phase II"
            else:
                name = name.split(",")[0]
            df.at[t, "work_name"] = name
            mark(t, "duplicate_name:near")

    # 2. duplicate_work_id: the clone imports the source row's state code and year.
    for t in pick(100):
        s = int(rng.integers(n_rows))
        if s == t:
            continue
        df.at[t, "work_id"] = df.at[s, "work_id"]
        mark(t, "duplicate_work_id:exact")

    # 3. cost_outlier: BOTH amounts scale by the SAME factor, so the spend ratio is invariant.
    for t in pick(1000):
        high = rng.random() < 0.75
        f = float(rng.uniform(15, 60)) if high else float(rng.uniform(0.005, 0.05))
        for c in AMOUNT_COLS:
            try:
                df.at[t, c] = f"{round(float(df.at[t, c]) * f, 2)}"
            except ValueError:
                pass
        mark(t, "cost_outlier:high" if high else "cost_outlier:low")

    # 4. negative_amount / extreme_value: exactly ONE amount field is damaged.
    for t in pick(200):
        c = AMOUNT_COLS[int(rng.integers(2))]
        try:
            df.at[t, c] = f"{-abs(float(df.at[t, c]))}"
            mark(t, f"negative_amount:{c}")
        except ValueError:
            pass
    for t in pick(60):
        c = AMOUNT_COLS[int(rng.integers(2))]
        df.at[t, c] = "1e300"
        mark(t, f"extreme_value:{c}")

    # 5. date_order violations.
    for t in pick(1800):
        try:
            if rng.random() < 0.55:
                dp = dt.date.fromisoformat(df.at[t, "date_proposal"])
                df.at[t, "date_approval"] = (dp - dt.timedelta(days=int(rng.integers(1, 400)))).isoformat()
                mark(t, "date_order:approval_before_proposal")
            else:
                da = dt.date.fromisoformat(df.at[t, "date_approval"])
                df.at[t, "date_completion"] = (da - dt.timedelta(days=int(rng.integers(1, 400)))).isoformat()
                mark(t, "date_order:completion_before_approval")
        except ValueError:
            pass

    # 6. pre_scheme_date: all three dates shift back together (lags preserved).
    for t in pick(200):
        shift = int(rng.integers(25, 51)) * 365
        ok = False
        for c in DATE_COLS:
            try:
                d = dt.date.fromisoformat(df.at[t, c])
            except ValueError:
                continue
            df.at[t, c] = (d - dt.timedelta(days=shift)).isoformat()
            ok = True
        if ok:
            mark(t, "pre_scheme_date:shifted")

    # 7. format noise. Recoverable styles parse after cleanup; unparseable ones never do.
    for t in pick(2820):
        c = TYPED_COLS[int(rng.integers(len(TYPED_COLS)))]
        v = df.at[t, c]
        if c in AMOUNT_COLS:
            try:
                x = float(v)
            except ValueError:
                mark(t, f"format_recoverable:{c}")
                continue
            df.at[t, c] = rng.choice([f"Rs {x:,.0f}", f"₹{x:,.2f}", f"{x:,.2f}"])
        else:
            try:
                d = dt.date.fromisoformat(v)
            except ValueError:
                mark(t, f"format_recoverable:{c}")  # cell already damaged; ledger still records it
                continue
            style = int(rng.integers(4))
            df.at[t, c] = [d.strftime("%d-%m-%Y"), d.strftime("%d/%m/%Y"),
                           d.strftime("%d %b %Y"), d.strftime("%B %d, %Y")][style]
        mark(t, f"format_recoverable:{c}")
    for t in pick(1939):
        c = TYPED_COLS[int(rng.integers(len(TYPED_COLS)))]
        if c in AMOUNT_COLS:
            df.at[t, c] = str(rng.choice(["abcd", "as per estimate", "to be decided", "1.2e400", "12-34-56"]))
        else:
            df.at[t, c] = str(rng.choice(["31/02/2020", "2020-13-45", "20200-01-01", "pending",
                                          "date awaited", "not a date"]))
        mark(t, f"format_unparseable:{c}")

    # 8. placeholder: a closed token vocabulary, injected into all 12 columns.
    text_ph, date_ph, num_ph = ["N/A", "unknown", "NULL", "-", "NA"], \
        ["0000-00-00", "N/A", "unknown", "-"], ["N/A", "unknown", "-", "NIL"]
    for t in pick(9454):
        for c in rng.choice(RAW_COLUMNS, size=int(rng.integers(1, 4)), replace=False):
            if c == "work_id":
                continue
            vocab = num_ph if c in AMOUNT_COLS else (date_ph if c in DATE_COLS else text_ph)
            df.at[t, c] = str(rng.choice(vocab))
            mark(t, f"placeholder:{c}")

    # 9. missing: RUNS LAST and overwrites earlier channels' evidence. This is the source of
    #    every observable-evidence ceiling reported later in this script.
    rates = {"work_id": 0.0, "work_name": 0.05, "district": 0.08, "state": 0.06,
             "sanction_amount": 0.12, "amount_spent": 0.19, "date_proposal": 0.10,
             "date_approval": 0.15, "date_completion": 0.24, "implementing_agency": 0.18,
             "vendor_name": 0.26, "status": 0.09}
    for c, r in rates.items():
        if r <= 0:
            continue
        for t in rng.choice(n_rows, size=int(r * n_rows), replace=False):
            df.at[t, c] = ""
            mark(t, f"missing:{c}")

    ledger_obj = {"config": {"n": n_rows, "seed": seed, "source": "fallback generator"},
                  "defects_by_row": {str(k): v for k, v in sorted(ledger.items())}}
    return df, ledger_obj


# --------------------------------------------------------------------------------------
# 3. LABELS
# --------------------------------------------------------------------------------------

def build_label_matrix(ledger: dict, n_rows: int):
    """ledger['defects_by_row'] maps "row index" -> ["channel:sub", ...]. Return the
    row-level multi-hot channel matrix, the sub-label sets, and the raw counts.

    An out-of-range row index means the ledger does not describe THIS corpus. That must
    raise: a mismatched ledger would silently supervise training on the wrong rows, and
    every number downstream would be meaningless in a way no metric can reveal.
    """
    if "defects_by_row" not in ledger:
        raise KeyError("ledger has no 'defects_by_row' key -- wrong ledger file")
    by_row = ledger["defects_by_row"]
    channels = sorted({lab.split(":", 1)[0] for labs in by_row.values() for lab in labs})
    idx = {c: j for j, c in enumerate(channels)}
    Y = np.zeros((n_rows, len(channels)), dtype=np.int8)
    sub_labels = [set() for _ in range(n_rows)]
    for key, labs in by_row.items():
        i = int(key)
        if i < 0 or i >= n_rows:
            raise IndexError(
                f"ledger row index {i} is out of range for a corpus of {n_rows} rows; "
                "this ledger does not describe this corpus and must not supervise it"
            )
        for lab in labs:
            Y[i, idx[lab.split(':', 1)[0]]] = 1
            sub_labels[i].add(lab)
    return Y, channels, sub_labels


def select_channels(Y: np.ndarray, channels: list[str]):
    """Reject channels below MIN_POSITIVES (no usable interval) or above MAX_PREVALENCE
    (a base rate, not a signal), and print exactly why each was dropped."""
    n = Y.shape[0]
    keep, dropped = [], []
    for j, c in enumerate(channels):
        pos = int(Y[:, j].sum())
        prev = pos / n
        if pos < MIN_POSITIVES:
            dropped.append((c, pos, prev, f"only {pos} positives (< {MIN_POSITIVES}); "
                                          "no confidence interval on this channel would be usable"))
        elif prev > MAX_PREVALENCE:
            dropped.append((c, pos, prev, f"prevalence {prev:.1%} (> {MAX_PREVALENCE:.0%}); "
                                          "a target this common is a base rate, not a signal"))
        else:
            keep.append(c)
    return keep, dropped


# --------------------------------------------------------------------------------------
# 4. PARSE LAYER  (shared by features, rules and ceilings)
# --------------------------------------------------------------------------------------

# Parse ladder states. NULLISH is deliberately NOT a state: the generator's unparseable
# date tokens ('pending', 'date awaited', 'not a date') are also placeholder-shaped, and a
# ladder that tests nullish first absorbs over half of format_unparseable:date_* into the
# placeholder state where no model can separate them again. Nullish is an INDEPENDENT flag.
ST_EMPTY, ST_CANONICAL, ST_RECOVERABLE, ST_GARBAGE = 0, 1, 2, 3
STATE_NAMES = {ST_EMPTY: "EMPTY", ST_CANONICAL: "CANONICAL",
               ST_RECOVERABLE: "RECOVERABLE", ST_GARBAGE: "GARBAGE"}

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CURRENCY_PREFIX = re.compile(r"(?i)^\s*(?:rs\.?|inr|₹)\s*")
_DATE_FORMATS = ("%d-%m-%Y", "%d/%m/%Y", "%d %b %Y", "%B %d, %Y", "%d %B %Y", "%b %d, %Y")


def classify_amount(s: str):
    """(state, value). CANONICAL requires float() to succeed AND the result to be finite.

    The finiteness guard is the fix for a measured bug: '1.2e400' is one of the generator's
    unparseable numeric tokens, float() accepts it and returns inf, and without this guard
    a fifth of format_unparseable:sanction_amount cells were encoded as ordinary numbers.
    Note also that pd.to_numeric('1.2e400') returns NaN, not inf -- so an np.isinf() flag
    on the PARSED column is identically zero over the whole corpus and is not built here.
    """
    t = s.strip()
    if t == "":
        return ST_EMPTY, math.nan
    try:
        v = float(t)
        if math.isfinite(v):
            return ST_CANONICAL, v
        return ST_GARBAGE, math.nan
    except ValueError:
        pass
    c = _CURRENCY_PREFIX.sub("", t).replace(",", "").replace(" ", "").replace("₹", "")
    try:
        v = float(c)
        if math.isfinite(v):
            return ST_RECOVERABLE, v
    except ValueError:
        pass
    return ST_GARBAGE, math.nan


def classify_date(s: str):
    """(state, ordinal day number). '31/02/2020' matches a style regex but fails calendar
    validation, so it lands in GARBAGE, which is where format_unparseable lives."""
    t = s.strip()
    if t == "":
        return ST_EMPTY, math.nan
    if _ISO_RE.match(t):
        try:
            return ST_CANONICAL, float(dt.date.fromisoformat(t).toordinal())
        except ValueError:
            return ST_GARBAGE, math.nan
    for f in _DATE_FORMATS:
        try:
            return ST_RECOVERABLE, float(dt.datetime.strptime(t, f).date().toordinal())
        except ValueError:
            continue
    return ST_GARBAGE, math.nan


def parse_layer(df: pd.DataFrame):
    """One pass over the five typed columns producing a state code and a parsed value each."""
    states, values = {}, {}
    for c in TYPED_COLS:
        fn = classify_amount if c in AMOUNT_COLS else classify_date
        st = np.empty(len(df), dtype=np.int8)
        va = np.empty(len(df), dtype=np.float64)
        for i, s in enumerate(df[c].to_numpy()):
            st[i], va[i] = fn(s)
        states[c], values[c] = st, va
    return states, values


def derive_nullish_lexicon(df: pd.DataFrame, train_idx: np.ndarray) -> set[str]:
    """Derive the placeholder/nullish token vocabulary from the TRAIN rows, unsupervised.

    Hard-coding the generator's literal tuple would be an oracle for this corpus and would
    not transfer. The transferable signal is "a short value that appears in several
    unrelated columns": cross-column document frequency. Two fixes over the naive version
    are applied. (1) The digit filter is not absolute -- '0000-00-00' is a placeholder and
    would be excluded by it -- so a value made only of zeros and punctuation is admitted
    provided a lenient numeric parse of it FAILS (which keeps '0.00' out). (2) The
    cross-column threshold is 2, not 3, because 'NIL' occurs in exactly two columns and a
    threshold of 3 silently drops it into the GARBAGE state alongside real corruption.
    """
    cross = Counter()
    total = Counter()
    for c in RAW_COLUMNS:
        vals = normalise_series(df[c].iloc[train_idx])
        seen = set()
        for v in vals:
            if v == "":
                continue
            total[v] += 1
            seen.add(v)
        for v in seen:
            cross[v] += 1
    lex = set()
    for v, cc in cross.items():
        if cc < 2 or total[v] < 10 or len(v) > 14:
            continue
        if not re.search(r"\d", v):
            lex.add(v)
            continue
        if re.fullmatch(r"[0\W_]+", v) and classify_amount(v)[0] == ST_GARBAGE:
            lex.add(v)
    return lex


# --------------------------------------------------------------------------------------
# 5. GROUPED, MULTI-LABEL-BALANCED SPLIT
# --------------------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_groups(df: pd.DataFrame, name_norm: pd.Series, nullish: set[str]) -> np.ndarray:
    """Connected components over exact normalised work_name and exact work_id.

    Purpose: a duplicate clique must never straddle the train/test boundary, or the
    whole-file duplicate counts mean something different at evaluation time than they did
    at training time and both duplicate channels report a fabricated failure.

    Sentinel guard: work_name is blanked or placeholdered on ~5% of rows, which would
    collapse ~1,000 unrelated rows into five atomic mega-blocks -- one of them larger than
    the entire positive set of four channels, which destroys any attempt to balance the
    folds. Rows whose key is empty or a nullish token get a private group.
    """
    n = len(df)
    uf = UnionFind(n)
    for key_series in (name_norm, normalise_series(df["work_id"])):
        buckets = defaultdict(list)
        for i, v in enumerate(key_series):
            if v == "" or v in nullish:
                continue
            buckets[v].append(i)
        for members in buckets.values():
            if len(members) > 1:
                first = members[0]
                for m in members[1:]:
                    uf.union(first, m)
    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int64)
    _, groups = np.unique(roots, return_inverse=True)
    return groups


def grouped_multilabel_split(groups: np.ndarray, Y: np.ndarray, fractions=(0.6, 0.2, 0.2), seed=0):
    """Greedy iterative stratification at GROUP level: assign groups rarest-channel-first
    to whichever split is furthest below its quota for that channel. Random assignment
    would leave a 60-positive channel with 3 test positives in one run and 20 in the next,
    and the reported interval would not describe that variance."""
    rng = np.random.default_rng(seed)
    n_splits = len(fractions)
    order_channels = np.argsort(Y.sum(axis=0))  # rarest first
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    gsize = {g: len(members[g]) for g in uniq}
    gpos = {g: Y[members[g]].sum(axis=0) for g in uniq}

    # Order groups by whether they carry rare channels, then by size, then randomly.
    def group_key(g):
        rare = tuple(-int(gpos[g][j]) for j in order_channels[:4])
        return rare + (-gsize[g], float(rng.random()))

    ordered = sorted(uniq, key=group_key)
    quota_rows = np.array(fractions) * len(groups)
    quota_pos = np.outer(np.array(fractions), Y.sum(axis=0))
    have_rows = np.zeros(n_splits)
    have_pos = np.zeros((n_splits, Y.shape[1]))
    assign = {}
    for g in ordered:
        deficits = []
        for s in range(n_splits):
            d = 0.0
            for j in order_channels:
                if gpos[g][j] > 0 and quota_pos[s, j] > 0:
                    d += (quota_pos[s, j] - have_pos[s, j]) / max(quota_pos[s, j], 1.0)
            d += 0.25 * (quota_rows[s] - have_rows[s]) / max(quota_rows[s], 1.0)
            deficits.append(d)
        s = int(np.argmax(deficits))
        assign[g] = s
        have_rows[s] += gsize[g]
        have_pos[s] += gpos[g]
    split = np.array([assign[g] for g in groups], dtype=np.int8)
    return split


# --------------------------------------------------------------------------------------
# 6. FEATURE ENGINEERING
# --------------------------------------------------------------------------------------

def _finite(a: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Every feature passes through here. An imputed value ALWAYS ships with its own
    was-present mask elsewhere in the block, because zero and unknown are different facts
    and collapsing them makes the least documented rows look the cleanest."""
    a = np.asarray(a, dtype=np.float64)
    return np.nan_to_num(a, nan=fill, posinf=fill, neginf=fill)


def robust_reference(keys: np.ndarray, values: np.ndarray, train_mask: np.ndarray, min_n: int):
    """Per-key median / MAD / sorted train values, FITTED ON TRAIN ROWS ONLY.

    median and MAD rather than mean and std because 5% of every peer cell is contaminated
    by the very outliers the feature exists to detect, and mean/std absorb them.
    """
    ref = {}
    ok = train_mask & np.isfinite(values)
    order = np.argsort(keys[ok], kind="mergesort")
    ks = keys[ok][order]
    vs = values[ok][order]
    if len(ks) == 0:
        return ref
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    ends = np.r_[starts[1:], len(ks)]
    for s, e in zip(starts, ends):
        if e - s < min_n:
            continue
        block = np.sort(vs[s:e])
        med = float(np.median(block))
        mad = float(np.median(np.abs(block - med)))
        ref[ks[s]] = (med, mad, block)
    return ref


def peer_deviation(values, keys_wt, keys_dist, train_mask, prefix, feats):
    """Peer-relative robust z + ECDF rank + envelope escape, with an explicit tier ladder.

    Peer-relative rather than global: the cost injector multiplies both amounts by
    15-60x or 0.005-0.05x, which is +1.2..+1.8 or -1.3..-2.3 dex against a within-work-type
    spread of ~0.25 dex, but only ~0.4 dex against the pooled corpus spread. Held-out
    measurement on this corpus: the peer form recovers roughly 2.7x the positives of the
    global form at the same false-positive rate.

    Uncomputable rows (blank or garbage amount) get z=0 AND tier=UNAVAILABLE, never a
    silent zero on its own -- an imputed 0 sits in the middle of the clean distribution.
    """
    logv = np.where(np.isfinite(values) & (values > 0), np.log10(np.maximum(values, 1e-12)), np.nan)
    ref_wt = robust_reference(keys_wt, logv, train_mask, MIN_PEER_GROUP)
    ref_dist = robust_reference(keys_dist, logv, train_mask, MIN_PEER_GROUP)
    glob = logv[train_mask & np.isfinite(logv)]
    glob_sorted = np.sort(glob) if len(glob) else np.array([0.0])
    glob_med = float(np.median(glob_sorted))
    glob_mad = float(np.median(np.abs(glob_sorted - glob_med))) or 0.1

    n = len(values)
    z = np.zeros(n)
    ecdf = np.full(n, 0.5)
    envelope = np.zeros(n)
    tier = np.zeros(n, dtype=np.int8)  # 0=work-type, 1=district, 2=global, 3=uncomputable
    gsize = np.zeros(n)
    for i in range(n):
        v = logv[i]
        if not np.isfinite(v):
            tier[i] = 3
            continue
        r = ref_wt.get(keys_wt[i])
        t = 0
        if r is None:
            r = ref_dist.get(keys_dist[i])
            t = 1
        if r is None:
            r = (glob_med, glob_mad, glob_sorted)
            t = 2
        med, mad, block = r
        scale = 1.4826 * mad if mad > 0 else 1.4826 * glob_mad
        z[i] = (v - med) / (scale if scale > 0 else 1.0)
        pos = np.searchsorted(block, v)
        ecdf[i] = pos / max(len(block), 1)
        lo = block[max(int(0.01 * len(block)) - 1, 0)]
        hi = block[min(int(0.99 * len(block)), len(block) - 1)]
        envelope[i] = max(0.0, v - hi) + max(0.0, lo - v)
        tier[i] = t
        gsize[i] = len(block)
    z = np.clip(_finite(z), -50, 50)
    feats[f"peer__{prefix}_z"] = z
    feats[f"peer__{prefix}_absz"] = np.abs(z)
    feats[f"peer__{prefix}_ecdf"] = _finite(ecdf, 0.5)
    feats[f"peer__{prefix}_tailness"] = np.abs(_finite(ecdf, 0.5) - 0.5) * 2.0
    feats[f"peer__{prefix}_envelope"] = np.clip(_finite(envelope), 0, 50)
    feats[f"peer__{prefix}_loggroupsize"] = np.log1p(_finite(gsize))
    for t, nm in ((1, "district"), (2, "global"), (3, "uncomputable")):
        feats[f"peer__{prefix}_tier_{nm}"] = (tier == t).astype(np.float64)


def minhash_neighbourhood(name_norm: pd.Series, seed: int):
    """MinHash + banded LSH over character 4-grams of the normalised work_name.

    Three implementation fixes over the naive version, each of which changes the result:
      * shingles are hashed ONCE (crc32, deterministic across processes -- Python's str
        hash is salted per run and would make the whole feature irreproducible) and the 32
        permutations are affine over that single hash, instead of 32 md5 calls per shingle.
      * oversized buckets are NOT discarded. The names here are templated, so bucket
        occupancy grows with n; a size cap silently empties the candidate set at scale
        while still returning a well-formed feature vector. Oversized buckets keep a
        sorted-neighbourhood window instead, which retains exactly the members whose
        signatures agree most.
      * empty candidate sets get explicit sentinels plus a has_candidates flag, so
        "isolated name" is distinguishable from "feature silently degraded".
    """
    n = len(name_norm)
    rng = np.random.default_rng(seed)
    prime = 2147483647
    n_perm, n_bands = 32, 8
    rows_per_band = n_perm // n_bands
    a = rng.integers(1, prime, size=n_perm, dtype=np.int64)
    b = rng.integers(0, prime, size=n_perm, dtype=np.int64)

    flat, offsets, shingle_sets = [], np.zeros(n, dtype=np.int64), []
    cursor = 0
    for i, s in enumerate(name_norm):
        padded = f" {s} "
        grams = {padded[j:j + 4] for j in range(max(len(padded) - 3, 1))} if s else {" "}
        hs = {zlib.crc32(g.encode("utf-8")) % prime for g in grams}
        shingle_sets.append(hs)
        offsets[i] = cursor
        flat.extend(hs)
        cursor += len(hs)
    H = np.asarray(flat, dtype=np.int64)
    sig = np.empty((n, n_perm), dtype=np.int64)
    for j in range(n_perm):
        sig[:, j] = np.minimum.reduceat((a[j] * H + b[j]) % prime, offsets)

    pair_chunks = []
    for band in range(n_bands):
        blk = np.ascontiguousarray(sig[:, band * rows_per_band:(band + 1) * rows_per_band])
        keys = [row.tobytes() for row in blk]
        codes, _ = pd.factorize(pd.Series(keys))
        order = np.argsort(codes, kind="mergesort")
        cs = codes[order]
        starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
        ends = np.r_[starts[1:], len(cs)]
        for s, e in zip(starts, ends):
            members = order[s:e]
            m = len(members)
            if m < 2:
                continue
            if m <= 30:
                ii, jj = np.triu_indices(m, k=1)
                pair_chunks.append(np.column_stack([members[ii], members[jj]]))
            else:
                # sorted-neighbourhood: identical signatures sort adjacently, so a window
                # keeps the highest-agreement candidates without an O(m^2) blow-up.
                srt = members[np.lexsort(tuple(blk[members, k] for k in range(rows_per_band)))]
                w = 10
                for off in range(1, w + 1):
                    pair_chunks.append(np.column_stack([srt[:-off], srt[off:]]))
    if pair_chunks:
        pairs = np.vstack(pair_chunks)
        lo = np.minimum(pairs[:, 0], pairs[:, 1])
        hi = np.maximum(pairs[:, 0], pairs[:, 1])
        pairs = np.unique(np.column_stack([lo, hi]), axis=0)
    else:
        pairs = np.zeros((0, 2), dtype=np.int64)

    # Cheap signature-agreement screen, then exact Jaccard only on plausible candidates.
    if len(pairs):
        agree = (sig[pairs[:, 0]] == sig[pairs[:, 1]]).mean(axis=1)
        keep = pairs[agree >= 0.30]
    else:
        keep = pairs
    best = np.zeros(n)
    near = np.zeros(n)
    blocksz = np.zeros(n)
    sums = np.zeros(n)
    for i, j in keep:
        si, sj = shingle_sets[i], shingle_sets[j]
        inter = len(si & sj)
        union = len(si) + len(sj) - inter
        jac = inter / union if union else 0.0
        for r in (i, j):
            blocksz[r] += 1
            sums[r] += jac
            if jac > best[r]:
                best[r] = jac
        if jac >= 0.95:
            near[i] += 1
            near[j] += 1
    mean_j = np.divide(sums, np.maximum(blocksz, 1), out=np.zeros(n), where=blocksz > 0)
    return {
        "max_jaccard": best,
        "n_near": near,
        "block_size": blocksz,
        "margin": best - mean_j,
        "has_candidates": (blocksz > 0).astype(np.float64),
    }


_ID_RE = re.compile(r"^([A-Z]{2,4})-([A-Z]{2})-(\d{4})-(\d{6})$")


def build_features(df, states, values, nullish, train_mask, name_norm, seed):
    """Assemble the full feature matrix.

    Everything statistical (nullish lexicon, peer references, frequency tables, TF-IDF,
    SVD, state-code map) is fitted on TRAIN ROWS ONLY and applied to all rows. Cross-row
    duplicate structure is whole-file and declared as such in the report.

    Two feature families that a naive reading of this corpus would suggest are ABSENT ON
    PURPOSE and must not be added back:
      * sanction rounding granularity (is_mult_100 / has_paise / trailing zeros). The base
        corpus rounds sanctions to Rs 100 and the outlier injector re-rounds to 2 dp, so
        the bit separates cost_outlier perfectly. It encodes np.round(), not cost anomaly,
        and transfers to zero real registers.
      * work_id serial minus row position. The generator mints the serial from emission
        order, so "serial != row index" is the ledger re-encoded through a side channel.
    An assertion at the end of this function fails the build if either reappears.
    """
    n = len(df)
    F: dict[str, np.ndarray] = {}
    raw = {c: df[c].astype(str).str.strip() for c in RAW_COLUMNS}
    norm = {c: (name_norm if c == "work_name" else normalise_series(df[c])) for c in RAW_COLUMNS}

    # ---- block: parse ------------------------------------------------------------------
    # CANONICAL is the reference state and is not emitted, so the three per-column state
    # columns are not collinear with an intercept (a rank-deficient design makes the
    # required logistic baseline unidentifiable, which is the easiest way to accidentally
    # ship a weak baseline and flatter the challenger).
    for c in TYPED_COLS:
        st = states[c]
        F[f"parse__{c}_empty"] = (st == ST_EMPTY).astype(np.float64)
        F[f"parse__{c}_recoverable"] = (st == ST_RECOVERABLE).astype(np.float64)
        F[f"parse__{c}_garbage"] = (st == ST_GARBAGE).astype(np.float64)
        F[f"parse__{c}_nullish"] = np.array([v in nullish for v in norm[c]], dtype=np.float64)
    for c in TEXT_COLS:
        F[f"parse__{c}_empty"] = (raw[c] == "").to_numpy(dtype=np.float64)
        F[f"parse__{c}_nullish"] = np.array([v in nullish for v in norm[c]], dtype=np.float64)
    st_mat = np.column_stack([states[c] for c in TYPED_COLS])
    F["parse__n_empty"] = (st_mat == ST_EMPTY).sum(axis=1).astype(np.float64)
    F["parse__n_recoverable"] = (st_mat == ST_RECOVERABLE).sum(axis=1).astype(np.float64)
    F["parse__n_garbage"] = (st_mat == ST_GARBAGE).sum(axis=1).astype(np.float64)
    # n_canonical is omitted: it is exactly 5 - (empty + recoverable + garbage).
    F["parse__n_nullish_cells"] = np.sum(
        [[v in nullish for v in norm[c]] for c in RAW_COLUMNS], axis=0).astype(np.float64)

    # placeholder morphology: a vocabulary-free complement to the derived lexicon. It fires
    # on 'TBD', 'XXX', '----' shapes this generator never emits, which is the point -- the
    # lexicon is corpus-specific, this is not.
    morph = np.zeros(n)
    for c in RAW_COLUMNS:
        arr = norm[c].to_numpy()
        hit = np.zeros(n)
        for i, t in enumerate(arr):
            if not t:
                continue
            score = 0
            if len(t) <= 8:
                score += 1
            if re.fullmatch(r"[^\w\s]+", t):
                score += 1
            if len(set(t.replace("-", "").replace("/", ""))) <= 1 and len(t) >= 2:
                score += 1
            if re.fullmatch(r"[0\W_]+", t):
                score += 1
            hit[i] = 1.0 if score >= 2 else 0.0
        morph += hit
    F["parse__n_morph_placeholder"] = morph

    # ---- block: amount -----------------------------------------------------------------
    sa, sp = values["sanction_amount"], values["amount_spent"]
    for c, v in (("sanction", sa), ("spent", sp)):
        fin = np.isfinite(v)
        F[f"amount__{c}_defined"] = fin.astype(np.float64)
        F[f"amount__{c}_log10mag"] = np.where(
            fin, np.log10(np.clip(np.abs(np.where(fin, v, 1.0)), 1.0, 1e308)), -1.0)
        F[f"amount__{c}_negative"] = (fin & (v < 0)).astype(np.float64)
        F[f"amount__{c}_astronomical"] = (fin & (np.abs(v) >= 1e30)).astype(np.float64)
    for c in AMOUNT_COLS:
        r = raw[c]
        # digits of the INTEGER part only: a total digit count silently smuggles the decimal
        # count -- and with it the injector's np.round(x, 2) fingerprint -- back in.
        F[f"amount__{c}_ndigits"] = r.str.split(".").str[0].str.count(r"\d").to_numpy(dtype=np.float64)
        F[f"amount__{c}_nalpha"] = r.str.count(r"[A-Za-z]").to_numpy(dtype=np.float64)
        F[f"amount__{c}_has_exponent"] = r.str.contains(r"\d[eE][+-]?\d", regex=True).to_numpy(dtype=np.float64)
        F[f"amount__{c}_has_grouping"] = r.str.contains(r"\d,\d", regex=True).to_numpy(dtype=np.float64)
        F[f"amount__{c}_has_currency"] = r.str.contains(r"(?i)^(?:rs\.?|inr|₹)",
                                                        regex=True).to_numpy(dtype=np.float64)
        # A minus sign must be followed by a digit. Without that guard the flag fires on the
        # bare '-' placeholder token and entangles negative_amount with placeholder.
        F[f"amount__{c}_raw_minus"] = r.str.contains(r"^\s*(?:rs\.?|inr|₹)?\s*-\s*\d",
                                                     regex=True, case=False).to_numpy(dtype=np.float64)
        # DELIBERATELY ABSENT: a decimal-place count on the amount columns. The base corpus
        # rounds sanctions to Rs 100 and writes them with one decimal; the cost-outlier
        # injector multiplies by a continuous factor and re-rounds to two. A decimal count
        # therefore separates cost_outlier almost perfectly -- and it measures np.round(),
        # not cost anomaly, so it transfers to exactly zero real registers. The same
        # reasoning removed the is_multiple_of_100 / trailing-zeros family entirely.
    both = np.isfinite(sa) & np.isfinite(sp) & (sa != 0)
    ratio = np.divide(sp, sa, out=np.full(n, np.nan), where=both)
    pos_ratio = both & np.isfinite(ratio) & (ratio > 0)
    F["amount__log_ratio"] = np.where(pos_ratio, np.log10(np.where(pos_ratio, ratio, 1.0)), 0.0)
    F["amount__ratio_defined"] = pos_ratio.astype(np.float64)
    F["amount__ratio_nonpositive"] = (both & (ratio <= 0)).astype(np.float64)
    F["amount__ratio_extreme"] = (both & (np.abs(_finite(ratio)) > 1e6)).astype(np.float64)
    F["amount__ratio_uncomputable"] = (~both).astype(np.float64)
    # NOTE the deliberate negative control inside this block: cost_outlier scales BOTH
    # amounts by the same factor, so log_ratio is invariant to it. If a cost_outlier model
    # leans on log_ratio, something upstream is wrong.

    # ---- block: peer -------------------------------------------------------------------
    # Peer key = the work-type phrase ("<action> <type>") that precedes ' at ' in the name.
    # Cost bands are a function of work type only, so this is the true peer cell.
    keys_wt = np.array([s.split(" at ")[0] if " at " in s else "" for s in name_norm], dtype=object)
    ph_key = np.array([(k == "") or (k in nullish) for k in keys_wt])
    keys_wt = np.where(ph_key, np.array([f"__row{i}" for i in range(n)], dtype=object), keys_wt)
    F["peer__key_is_placeholder"] = ph_key.astype(np.float64)
    dist_norm = norm["district"].to_numpy()
    keys_dist = np.array([d if (d and d not in nullish) else f"__row{i}"
                          for i, d in enumerate(dist_norm)], dtype=object)
    peer_deviation(sa, keys_wt, keys_dist, train_mask, "sanction", F)
    peer_deviation(sp, keys_wt, keys_dist, train_mask, "spent", F)

    # ---- block: date -------------------------------------------------------------------
    dp, da, dc = values["date_proposal"], values["date_approval"], values["date_completion"]
    lags = {"pa": da - dp, "ac": dc - da, "pc": dc - dp}
    for k, lag in lags.items():
        ok = np.isfinite(lag)
        # Impute with the TRAIN median of POSITIVE lags, i.e. a value outside the violation
        # region. Imputing 0 would place every uncomputable row inside the defect region:
        # no clean row has a zero lag, so a tree reads 0 as a defect indicator and the
        # required linear baseline is depressed by an encoding choice rather than capacity.
        pos_lags = lag[train_mask & ok & (lag > 0)]
        fill = float(np.median(pos_lags)) if len(pos_lags) else 90.0
        filled = np.where(ok, lag, fill)
        F[f"date__lag_{k}_days"] = np.clip(filled / 365.0, -50, 50)
        F[f"date__lag_{k}_signedlog"] = np.sign(filled) * np.log1p(np.abs(np.clip(filled, -1e6, 1e6)))
        F[f"date__lag_{k}_defined"] = ok.astype(np.float64)
        F[f"date__lag_{k}_x_defined"] = np.clip(filled / 365.0, -50, 50) * ok  # explicit interaction
    viol_pa = (np.isfinite(lags["pa"]) & (lags["pa"] < 0)).astype(np.float64)
    viol_ac = (np.isfinite(lags["ac"]) & (lags["ac"] < 0)).astype(np.float64)
    F["date__viol_pa"] = viol_pa          # targets date_order:approval_before_proposal
    F["date__viol_ac"] = viol_ac          # targets date_order:completion_before_approval
    F["date__viol_any"] = np.maximum(viol_pa, viol_ac)
    depth = np.maximum(np.clip(-_finite(lags["pa"]), 0, None), np.clip(-_finite(lags["ac"]), 0, None))
    F["date__viol_depth_log"] = np.log1p(np.clip(depth, 0, 1e6))
    F["date__n_dates_present"] = np.sum([np.isfinite(x) for x in (dp, da, dc)], axis=0).astype(np.float64)

    for c in DATE_COLS:
        r = raw[c]
        F[f"date__{c}_iso"] = r.str.match(r"^\d{4}-\d{2}-\d{2}$").to_numpy(dtype=np.float64)
        F[f"date__{c}_dmy_dash"] = r.str.match(r"^\d{1,2}-\d{1,2}-\d{4}$").to_numpy(dtype=np.float64)
        F[f"date__{c}_dmy_slash"] = r.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$").to_numpy(dtype=np.float64)
        F[f"date__{c}_textual"] = r.str.contains(r"(?i)[a-z]{3,}", regex=True).to_numpy(dtype=np.float64)
        F[f"date__{c}_yearlen_bad"] = r.str.contains(r"\b\d{5,}\b", regex=True).to_numpy(dtype=np.float64)
    style = np.column_stack([F[f"date__{c}_iso"] for c in DATE_COLS])
    F["date__n_non_iso"] = (3.0 - style.sum(axis=1))
    D3 = np.column_stack([dp, da, dc])
    ecomp = np.isfinite(D3).any(axis=1)
    earliest = np.where(ecomp, np.min(np.where(np.isfinite(D3), D3, np.inf), axis=1), np.nan)
    F["date__earliest_defined"] = ecomp.astype(np.float64)
    F["date__earliest_year"] = np.where(
        ecomp, [dt.date.fromordinal(int(x)).year if np.isfinite(x) else -1 for x in
                np.where(ecomp, earliest, 1)], -1.0)
    # Floor from a ROBUST CENTRE, not a low quantile. Contamination here is ~1%, so any
    # quantile below ~2% sits INSIDE the pre_scheme cluster and the "data-driven" floor
    # turns into a near-constant zero that contributes nothing while a hardcoded constant
    # quietly does all the work.
    et = earliest[train_mask & ecomp]
    if len(et):
        med = float(np.median(et))
        mad = float(np.median(np.abs(et - med))) or 365.0
        floor = med - 6.0 * 1.4826 * mad
    else:
        floor = float(SCHEME_START)
    F["date__years_below_robust_floor"] = np.where(
        ecomp, np.clip((floor - _finite(earliest, floor)) / 365.25, 0, None), 0.0)
    F["date__years_before_scheme_start"] = np.where(
        ecomp, np.clip((SCHEME_START - _finite(earliest, SCHEME_START)) / 365.25, 0, None), 0.0)

    # ---- block: idcons (GENERATOR-INVARIANT -- ablated for pre_scheme_date) -------------
    id_raw = raw["work_id"].to_numpy()
    id_year = np.full(n, np.nan)
    id_state = np.array([""] * n, dtype=object)
    for i, s in enumerate(id_raw):
        m = _ID_RE.match(s)
        if m:
            id_state[i] = m.group(2)
            id_year[i] = float(m.group(3))
    prop_year = np.array([dt.date.fromordinal(int(x)).year if np.isfinite(x) else np.nan for x in dp])
    gap = id_year - prop_year
    gap_ok = np.isfinite(gap)
    F["idcons__year_gap"] = np.clip(_finite(gap), -60, 60)
    F["idcons__year_gap_abs"] = np.abs(np.clip(_finite(gap), -60, 60))
    F["idcons__year_gap_defined"] = gap_ok.astype(np.float64)
    F["idcons__year_gap_large"] = (gap_ok & (np.abs(_finite(gap)) > 2)).astype(np.float64)
    # state-code map learned from TRAIN rows (modal state per code), never hardcoded.
    code_map = {}
    st_norm = norm["state"].to_numpy()
    tmp = defaultdict(Counter)
    for i in np.flatnonzero(train_mask):
        if id_state[i] and st_norm[i] and st_norm[i] not in nullish:
            tmp[id_state[i]][st_norm[i]] += 1
    for k, ctr in tmp.items():
        code_map[k] = ctr.most_common(1)[0][0]
    agree = np.zeros(n)
    disagree = np.zeros(n)
    for i in range(n):
        exp = code_map.get(id_state[i])
        if exp is None or not st_norm[i] or st_norm[i] in nullish:
            continue
        if st_norm[i] == exp:
            agree[i] = 1.0
        else:
            disagree[i] = 1.0
    F["idcons__state_code_disagree"] = disagree
    F["idcons__state_code_undefined"] = 1.0 - agree - disagree

    # ---- block: dup (WHOLE-FILE, transductive by construction -- declared) -------------
    id_key = pd.Series([v.upper() for v in id_raw])
    id_ph = np.array([(v == "") or (normalise_indic(v) in nullish) for v in id_raw])
    id_counts = id_key.map(id_key.value_counts()).to_numpy(dtype=np.float64)
    id_counts = np.where(id_ph, 1.0, id_counts)
    F["dup__id_count_log"] = np.log1p(id_counts)
    F["dup__id_has_twin"] = (id_counts > 1).astype(np.float64)
    F["dup__id_is_placeholder"] = id_ph.astype(np.float64)
    name_ph = np.array([(v == "") or (v in nullish) for v in name_norm])
    nn = pd.Series(list(name_norm))
    name_counts = nn.map(nn.value_counts()).to_numpy(dtype=np.float64)
    name_counts = np.where(name_ph, 1.0, name_counts)
    F["dup__name_multiplicity_log"] = np.log1p(name_counts)
    F["dup__name_has_exact_twin"] = (name_counts > 1).astype(np.float64)
    F["dup__name_is_placeholder"] = name_ph.astype(np.float64)
    mh = minhash_neighbourhood(name_norm, seed)
    for k, v in mh.items():
        F[f"dup__{k}"] = _finite(v)
    # within-id-block field disagreement: a legitimately repeated key agrees on its other
    # fields; an injected clone joins two unrelated works.
    blk = pd.DataFrame({"id": id_key, "district": norm["district"].to_numpy(),
                        "vendor": norm["vendor_name"].to_numpy(),
                        "year": np.where(np.isfinite(prop_year), prop_year, -1)})
    for fld in ("district", "vendor", "year"):
        nuniq = blk.groupby("id")[fld].transform("nunique").to_numpy(dtype=np.float64)
        F[f"dup__id_block_{fld}_nunique"] = np.where(id_ph, 1.0, nuniq)

    # ---- block: geo (GENERATOR-INVARIANT -- ablated for duplicate_name) ----------------
    # The name template embeds the row's own district, so a cloned name carries a FOREIGN
    # district suffix. Measured on this corpus, DISAGREE fires on 781 rows and every one is
    # a duplicate_name positive: precision 1.000. That is a property of the generator (it
    # never emits a legitimate mismatch), not evidence of detection skill, which is why
    # duplicate_name is reported both with and without this block and the WITHOUT number
    # is the headline.
    tails = np.array([s.rsplit(",", 1)[-1].strip() if "," in s else "" for s in name_norm], dtype=object)
    geo_state = np.zeros(n, dtype=np.int8)  # 0=agree 1=disagree 2=no_tail 3=district_undefined
    overlap = np.zeros(n)
    for i in range(n):
        d = dist_norm[i]
        if tails[i] == "":
            geo_state[i] = 2
            continue
        if (not d) or d in nullish:
            geo_state[i] = 3
            continue
        ts, ds = set(tails[i].split()), set(d.split())
        ov = len(ts & ds) / max(len(ts), 1)
        overlap[i] = ov
        geo_state[i] = 0 if (tails[i] == d or ov >= 0.5) else 1
    F["geo__disagree"] = (geo_state == 1).astype(np.float64)
    F["geo__no_tail"] = (geo_state == 2).astype(np.float64)
    F["geo__district_undefined"] = (geo_state == 3).astype(np.float64)
    F["geo__token_overlap"] = overlap

    # ---- block: freq -------------------------------------------------------------------
    # Value-frequency lift, gated by an ABSOLUTE cardinality threshold. A ratio gate
    # (nunique/n) is n-dependent and silently changes the column set between a 20k dev
    # corpus and a 500k register -- including gating out the second best column.
    gated = []
    for c in RAW_COLUMNS:
        vals = norm[c].to_numpy()
        tr = pd.Series(vals[train_mask])
        tr = tr[tr != ""]
        if tr.nunique() < 100:
            continue
        gated.append(c)
        vc = tr.value_counts()
        med = float(np.median(vc.reindex(tr).to_numpy())) or 1.0
        cnt = pd.Series(vals).map(vc).to_numpy(dtype=np.float64)
        unseen = ~np.isfinite(cnt)
        cnt = np.where(unseen, 1.0, cnt)
        F[f"freq__{c}_lift"] = np.where(np.array([v == "" for v in vals]), 0.0, np.log10(cnt / med))
        F[f"freq__{c}_unseen"] = unseen.astype(np.float64)
    F["freq__max_lift"] = (np.max(np.column_stack([F[f"freq__{c}_lift"] for c in gated]), axis=1)
                           if gated else np.zeros(n))

    # Surface-shape signature rarity. The signature PRESERVES LENGTH: collapsing digit runs
    # maps '2019-03-15' and '15-03-2019' to the same shape, which makes one of the four
    # injected recoverable date styles invisible.
    def signature(s: str, drop_fraction: bool) -> str:
        t = s.strip()
        if drop_fraction and "." in t:
            # The fractional part of an amount is dropped before the shape is taken. Keeping
            # it would re-admit the rounding artifact through the back door: '846900.0' and
            # '12703500.52' differ in shape only because np.round(x, 2) ran on the outlier.
            t = t.split(".")[0]
        return "".join("9" if ch.isdigit() else ("A" if ch.isalpha() else ch) for ch in t)

    sig_tbl = {}
    for c in TYPED_COLS:
        sigs = np.array([signature(v, c in AMOUNT_COLS) for v in raw[c]], dtype=object)
        sig_tbl[c] = sigs
        vc = pd.Series(sigs[train_mask]).value_counts()
        cnt = pd.Series(sigs).map(vc).to_numpy(dtype=np.float64)
        cnt = np.where(np.isfinite(cnt), cnt, 1.0)
        F[f"freq__{c}_sig_rarity"] = -np.log10(cnt / max(int(train_mask.sum()), 1))
    F["freq__n_distinct_date_sigs"] = np.array(
        [len({sig_tbl[c][i] for c in DATE_COLS}) for i in range(n)], dtype=np.float64)

    # ---- block: text -------------------------------------------------------------------
    # char_wb n-grams survive the transposition/abbreviation/case perturbations that break
    # word tokens. Vectoriser AND SVD are fitted on TRAIN ROWS ONLY -- fitting on the full
    # corpus leaks test-set text statistics into the representation.
    corpus_text = [s if s else " " for s in name_norm]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                          max_features=20000, lowercase=False)
    Xtr_text = vec.fit_transform([corpus_text[i] for i in np.flatnonzero(train_mask)])
    svd = TruncatedSVD(n_components=min(SVD_DIMS, Xtr_text.shape[1] - 1), random_state=seed)
    svd.fit(Xtr_text)
    Z = svd.transform(vec.transform(corpus_text))
    for k in range(Z.shape[1]):
        F[f"text__svd_{k:03d}"] = Z[:, k]
    F["text__name_defined"] = (~name_ph).astype(np.float64)

    # ---- block: control ----------------------------------------------------------------
    # A random normal column pushed through the identical pipeline. It is null BY
    # CONSTRUCTION, unlike a "vendor peer z" control, which is not null at all (a cost
    # outlier escapes any grouping). Its per-channel univariate AUC is reported in the
    # results table: anything materially above 0.50 means the pipeline, not the data.
    F["control__random_normal"] = np.random.default_rng(seed + 1).normal(size=n)

    X = pd.DataFrame({k: _finite(v).astype(np.float32) for k, v in F.items()})

    banned = [c for c in X.columns if "serial" in c or "rounding" in c or "mult_100" in c]
    if banned:
        raise AssertionError(f"generator-artifact features must not exist: {banned}")
    if not np.isfinite(X.to_numpy()).all():
        raise AssertionError("non-finite value reached the feature matrix")
    return X, {"vectorizer": vec, "svd": svd, "nullish": sorted(nullish),
               "peer_tiers": True, "gated_freq_columns": gated}


def prune_features(X: pd.DataFrame, train_mask: np.ndarray):
    """Drop zero-variance columns, exact duplicates, and linearly dependent columns.

    Constant columns inflate the feature count used to argue for capacity and consume the
    L2 budget of the required logistic baseline while contributing nothing; exact duplicates
    break block ablations (removing one block leaves its twin behind and the measured delta
    reads as zero); and exact linear dependencies -- which this feature set creates on
    purpose, e.g. n_empty is the sum of the five per-column empty flags -- make the
    logistic coefficients unidentifiable and its fit unstable. A weak, unstable baseline is
    the single most likely way a challenger comes out looking better than it is, so the
    dependency screen runs before any model sees the matrix.
    """
    tr_idx = np.flatnonzero(train_mask)
    tr = X.iloc[tr_idx]
    zero_var = [c for c in X.columns if float(tr[c].std()) == 0.0]
    Xp = X.drop(columns=zero_var)
    seen, dup = {}, []
    for c in Xp.columns:
        key = hash(Xp[c].to_numpy().tobytes())
        if key in seen:
            dup.append((c, seen[key]))
        else:
            seen[key] = c
    Xp = Xp.drop(columns=[c for c, _ in dup])

    # Modified Gram-Schmidt over the standardised train block: a column whose residual after
    # projection onto the accepted basis is numerically zero is an exact linear combination
    # of columns already kept, and carries no information the design does not already hold.
    A = Xp.iloc[tr_idx].to_numpy(dtype=np.float64)
    A = (A - A.mean(axis=0)) / np.where(A.std(axis=0) == 0, 1.0, A.std(axis=0))
    basis = np.zeros((A.shape[0], A.shape[1]))
    k = 0
    dependent = []
    for i, col in enumerate(Xp.columns):
        v = A[:, i].copy()
        if k:
            v -= basis[:, :k] @ (basis[:, :k].T @ v)
            v -= basis[:, :k] @ (basis[:, :k].T @ v)  # reorthogonalise once for stability
        nrm = float(np.linalg.norm(v))
        if nrm <= 1e-7 * math.sqrt(A.shape[0]):
            dependent.append(col)
            continue
        basis[:, k] = v / nrm
        k += 1
    Xp = Xp.drop(columns=dependent)
    return Xp, zero_var, dup, dependent


# --------------------------------------------------------------------------------------
# 7. METRICS
# --------------------------------------------------------------------------------------

def _avg_ranks(s: np.ndarray) -> np.ndarray:
    """Tie-averaged ranks, fully vectorised.

    Ties are averaged rather than broken arbitrarily because most of the rule baselines are
    BINARY scores: with arbitrary tie-breaking a binary rule's AUC depends on the row order
    of the file, which is not a property of the rule.
    """
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    n = len(s)
    is_new = np.r_[True, ss[1:] != ss[:-1]]
    grp = np.cumsum(is_new) - 1
    starts = np.flatnonzero(is_new)
    ends = np.r_[starts[1:], n]
    avg = 0.5 * (starts + ends - 1) + 1.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = avg[grp]
    return ranks


def permutation_null_band(y: np.ndarray, score: np.ndarray, n_perm: int, seed: int, alpha: float):
    """Central (1-alpha) band of the AUC under the null that `score` says nothing about `y`.

    This is the correct yardstick for the two tripwire columns, and a bootstrap interval is
    not. A bootstrap describes the uncertainty AROUND an observed value, so a control column
    that lands at 0.30 purely by chance at n_pos=12 gets an interval that excludes 0.50 and
    the tripwire fires on nothing. The null band instead asks the actual question: is this
    AUC further from 0.5 than chance allows AT THIS n_pos?

    `alpha` is BONFERRONI-CORRECTED by the caller for the number of tripwire tests in the
    run. Without that correction a run with ten channels and two controls each fires about
    one false alarm every time, and a tripwire that cries wolf on correct runs is a
    tripwire that gets switched off -- which is how a pipeline fault eventually ships.

    Permuting the labels does not change the score ranks, so the whole band comes from one
    ranking plus n_perm random positive-set draws.
    """
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return (float("nan"), float("nan"))
    r = _avg_ranks(score)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_perm)
    for b in range(n_perm):
        vals[b] = (rng.permutation(r)[:npos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    npos = float(y.sum())
    nneg = float(len(y) - npos)
    if npos == 0 or nneg == 0:
        return float("nan")
    r = _avg_ranks(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def fast_ap(y: np.ndarray, s: np.ndarray) -> float:
    npos = float(y.sum())
    if npos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float(prec[ys == 1].sum() / npos)


def hanley_mcneil_ci(auc: float, npos: int, nneg: int):
    """Analytic AUC interval, printed beside the bootstrap one as a cross-check."""
    if not np.isfinite(auc) or npos == 0 or nneg == 0:
        return (float("nan"), float("nan"))
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (npos - 1) * (q1 - auc ** 2) + (nneg - 1) * (q2 - auc ** 2)) / (npos * nneg)
    se = math.sqrt(max(var, 0.0))
    return (max(0.0, auc - 1.96 * se), min(1.0, auc + 1.96 * se))


def wilson(k: int, n: int):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def make_group_bootstrap(groups_test: np.ndarray, n_boot: int, seed: int):
    """Resample GROUPS, not rows.

    Test rows inside one duplicate clique are not independent -- resampling rows would
    treat a cloned pair as two independent observations and report an interval that is too
    narrow, which is one half of the error this project already made once.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups_test)
    members = [np.flatnonzero(groups_test == g) for g in uniq]
    idx_sets = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), size=len(members))
        idx_sets.append(np.concatenate([members[p] for p in pick]))
    return idx_sets


def bootstrap_curve(y: np.ndarray, s: np.ndarray, idx_sets):
    aucs = np.full(len(idx_sets), np.nan)
    aps = np.full(len(idx_sets), np.nan)
    for b, idx in enumerate(idx_sets):
        yb, sb = y[idx], s[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        aucs[b] = fast_auc(yb, sb)
        aps[b] = fast_ap(yb, sb)
    return aucs, aps


def ci_from(samples: np.ndarray):
    v = samples[np.isfinite(samples)]
    if len(v) < 20:
        return (float("nan"), float("nan"))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


def fmt_ci(point: float, lo: float, hi: float) -> str:
    if not np.isfinite(point):
        return "     n/a        "
    return f"{point:.3f} [{lo:.3f},{hi:.3f}]"


# --------------------------------------------------------------------------------------
# 8. DETERMINISTIC RULE BASELINES
# --------------------------------------------------------------------------------------

def build_rules(df, states, values, nullish, X):
    """Hand-written rules, one per channel.

    These are the real bar on this corpus, not logistic regression. The generator injects
    most channels with a single deterministic edit, so a rule that inverts that edit is the
    ceiling; a model that merely matches it has learned nothing worth shipping. Any channel
    where a model does not clearly beat its rule should ship as the rule.
    """
    n = len(df)
    sa, sp = values["sanction_amount"], values["amount_spent"]
    st = {c: states[c] for c in TYPED_COLS}
    rules = {}

    ph_hits = np.zeros(n)
    for c in RAW_COLUMNS:
        ph_hits += np.array([normalise_indic(v) in nullish for v in df[c].astype(str)], dtype=float)
    rules["placeholder"] = (ph_hits > 0).astype(float)

    # A placeholder token sitting in a numeric or date column ALSO fails every parser, so
    # "any GARBAGE cell" is a poor rule: it fires on thousands of placeholder rows. The
    # primary rule therefore requires a garbage cell that is NOT a nullish token. The naive
    # version is kept as a printed diagnostic so the size of that confusion is visible
    # rather than assumed away.
    nullish_cell = {c: np.array([normalise_indic(v) in nullish for v in df[c].astype(str)])
                    for c in TYPED_COLS}
    garbage = np.sum([(st[c] == ST_GARBAGE) for c in TYPED_COLS], axis=0).astype(float)
    garbage_nn = np.sum([(st[c] == ST_GARBAGE) & (~nullish_cell[c]) for c in TYPED_COLS],
                        axis=0).astype(float)
    rules["format_unparseable"] = (garbage_nn > 0).astype(float)
    rules["format_unparseable (any garbage cell)"] = (garbage > 0).astype(float)
    recov = np.sum([(st[c] == ST_RECOVERABLE) for c in TYPED_COLS], axis=0).astype(float)
    rules["format_recoverable"] = (recov > 0).astype(float)

    rules["negative_amount"] = ((np.isfinite(sa) & (sa < 0)) | (np.isfinite(sp) & (sp < 0))).astype(float)
    # An exponent marker alone is not the rule: '1.2e400' carries one and is an
    # UNPARSEABLE token, not an extreme value. The rule requires either a finite parsed
    # magnitude above 1e30, or an exponent in a cell that still parses.
    expo = ((X["amount__sanction_amount_has_exponent"].to_numpy() > 0) & (st["sanction_amount"] == ST_CANONICAL)) | \
           ((X["amount__amount_spent_has_exponent"].to_numpy() > 0) & (st["amount_spent"] == ST_CANONICAL))
    rules["extreme_value"] = ((np.isfinite(sa) & (np.abs(sa) >= 1e30)) |
                              (np.isfinite(sp) & (np.abs(sp) >= 1e30)) | expo).astype(float)
    rules["date_order"] = X["date__viol_any"].to_numpy().astype(float)
    rules["pre_scheme_date"] = (X["date__years_before_scheme_start"].to_numpy() > 0).astype(float)
    rules["cost_outlier"] = ((np.abs(X["peer__sanction_absz"].to_numpy()) > 4) |
                             (np.abs(X["peer__spent_absz"].to_numpy()) > 4)).astype(float)
    rules["duplicate_work_id"] = X["dup__id_has_twin"].to_numpy().astype(float)
    # For duplicate_name the honest rule is the exact/near collision, NOT the geo test: the
    # geo test is a generator invariant and is reported separately as its own line.
    rules["duplicate_name"] = np.maximum(X["dup__name_has_exact_twin"].to_numpy(),
                                         (X["dup__max_jaccard"].to_numpy() >= 0.95).astype(float))
    rules["duplicate_name (geo invariant)"] = X["geo__disagree"].to_numpy().astype(float)
    return rules


def evidence_ceilings(df, states, values, sub_labels, Y, channels, X, nullish):
    """Per-channel OBSERVABLE-EVIDENCE ceiling.

    The generator's format and missing passes run LAST and overwrite cells that earlier
    channels already corrupted, while the ledger keeps the earlier label. For those rows no
    evidence survives in the emitted corpus, so no model, rule or human can recover them.
    Printed beside every recall, because unstated it reads as a modelling shortfall and the
    next round burns compute chasing irreducible label noise.
    """
    n = len(df)
    sa, sp = values["sanction_amount"], values["amount_spent"]
    obs = {}
    obs["negative_amount"] = ((np.isfinite(sa) & (sa < 0)) | (np.isfinite(sp) & (sp < 0)))
    obs["extreme_value"] = ((np.isfinite(sa) & (np.abs(sa) >= 1e30)) |
                            (np.isfinite(sp) & (np.abs(sp) >= 1e30)) |
                            (X["amount__sanction_amount_has_exponent"].to_numpy() > 0) |
                            (X["amount__amount_spent_has_exponent"].to_numpy() > 0))
    obs["date_order"] = X["date__viol_any"].to_numpy() > 0
    obs["pre_scheme_date"] = X["date__years_before_scheme_start"].to_numpy() > 0
    obs["cost_outlier"] = np.isfinite(sa) | np.isfinite(sp)
    obs["duplicate_name"] = X["text__name_defined"].to_numpy() > 0
    obs["duplicate_work_id"] = X["dup__id_is_placeholder"].to_numpy() == 0

    # format channels: use the sub-label's own column, which is the only honest test.
    for ch in ("format_recoverable", "format_unparseable"):
        want = ST_RECOVERABLE if ch == "format_recoverable" else ST_GARBAGE
        flag = np.zeros(n, dtype=bool)
        for i in range(n):
            for lab in sub_labels[i]:
                if not lab.startswith(ch + ":"):
                    continue
                col = lab.split(":", 1)[1]
                if col in TYPED_COLS and states[col][i] == want:
                    flag[i] = True
        obs[ch] = flag
    # placeholder ceiling from the ledger's own sub-labels: the token must still BE a
    # nullish token in its own column. A cell blanked afterwards by the missing pass, or
    # overwritten by another channel, leaves no placeholder evidence behind.
    ph_flag = np.zeros(n, dtype=bool)
    for i in range(n):
        for lab in sub_labels[i]:
            if lab.startswith("placeholder:"):
                col = lab.split(":", 1)[1]
                if col in RAW_COLUMNS and normalise_indic(str(df[col].iloc[i])) in nullish:
                    ph_flag[i] = True
    obs["placeholder"] = ph_flag

    out = {}
    for j, ch in enumerate(channels):
        pos = np.flatnonzero(Y[:, j] == 1)
        if ch not in obs or len(pos) == 0:
            out[ch] = None
            continue
        k = int(obs[ch][pos].sum())
        lo, hi = wilson(k, len(pos))
        out[ch] = {"n_pos": int(len(pos)), "evidence_present": k,
                   "ceiling": k / len(pos), "ci": [lo, hi]}
    return out


# --------------------------------------------------------------------------------------
# 9. MODELS
# --------------------------------------------------------------------------------------

def fit_logreg(Xtr, ytr, Xva, yva, seed):
    """L2 logistic regression with C chosen on the VALIDATION split.

    The baseline is selected properly on purpose: a weak baseline is exactly what makes a
    challenger look better than it is, and this study's whole point is not to do that.
    """
    best, best_ap, best_c = None, -1.0, None
    for C in (0.05, 0.25, 1.0, 4.0):
        m = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", random_state=seed)
        m.fit(Xtr, ytr)
        ap = fast_ap(yva, m.predict_proba(Xva)[:, 1]) if yva.sum() else -1.0
        if np.isfinite(ap) and ap > best_ap:
            best, best_ap, best_c = m, ap, C
    if best is None:
        best = LogisticRegression(max_iter=3000, random_state=seed).fit(Xtr, ytr)
        best_c = 1.0
    return best, best_c


def fit_gbm(Xtr, ytr, seed):
    """HistGradientBoosting with INTERNAL early stopping carved out of the training rows.

    20k rows of tabular data with engineered interactions is precisely where GBDT wins, so
    omitting it would leave the neural challenger facing a strawman.
    """
    m = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=25, random_state=seed)
    m.fit(Xtr, ytr)
    return m


class MacroAP(  # base class is resolved at import time: TensorFlow may be absent
    tf.keras.callbacks.Callback if HAVE_TF else object
):
    """Writes val_macro_ap into the epoch logs so early stopping can monitor it.

    Monitoring val_loss instead would track the most prevalent channel and stop the run on
    a channel nobody is claiming anything about.
    """

    def __init__(self, Xva, Yva):
        super().__init__()
        self.Xva, self.Yva = Xva, Yva

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        P = self.model.predict(self.Xva, verbose=0)
        aps = [fast_ap(self.Yva[:, j], P[:, j]) for j in range(self.Yva.shape[1])
               if self.Yva[:, j].sum() > 0]
        logs["val_macro_ap"] = float(np.nanmean(aps)) if aps else 0.0


def fit_mlp_tf(Xtr, Ytr, Xva, Yva, seed):
    tf.keras.utils.set_random_seed(seed)
    prev = np.clip(Ytr.mean(axis=0), 1e-4, 1 - 1e-4)
    bias = np.log(prev / (1 - prev)).astype("float32")
    inp = tf.keras.Input(shape=(Xtr.shape[1],))
    x = tf.keras.layers.Dense(128, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    out = tf.keras.layers.Dense(Ytr.shape[1], activation="sigmoid",
                                bias_initializer=tf.keras.initializers.Constant(bias))(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    cbs = [
        MacroAP(Xva, Yva),  # must come first: it writes the metric the others monitor
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_macro_ap", mode="max", factor=0.5,
                                             patience=6, min_lr=1e-5, verbose=0),
        tf.keras.callbacks.EarlyStopping(monitor="val_macro_ap", mode="max", patience=15,
                                         restore_best_weights=True, verbose=0),
    ]
    hist = model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=EPOCH_CEILING,
                     batch_size=256, verbose=0, callbacks=cbs)
    return model, len(hist.history["loss"])


class SklearnMLPBundle:
    """Fallback challenger for images without TensorFlow: one small sklearn MLP per channel,
    with the same early-stopping/plateau-LR/best-weight-restore behaviour (MLPClassifier's
    early_stopping keeps the best-scoring weights and 'adaptive' cuts the LR on plateau)."""

    def __init__(self, seed):
        self.seed = seed
        self.models = []

    def fit(self, Xtr, Ytr):
        from sklearn.neural_network import MLPClassifier
        for j in range(Ytr.shape[1]):
            m = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=EPOCH_CEILING,
                              early_stopping=True, n_iter_no_change=15, validation_fraction=0.15,
                              learning_rate="adaptive", random_state=self.seed)
            m.fit(Xtr, Ytr[:, j])
            self.models.append(m)
        return self

    def predict(self, X, verbose=0):
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.models])


# --------------------------------------------------------------------------------------
# 10. MAIN
# --------------------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    rule_banner("PARAKH -- SUPERVISED DATA-QUALITY DEFECT CHANNELS (baseline-first)")
    log(f"python {sys.version.split()[0]} | numpy {np.__version__} | pandas {pd.__version__} "
        f"| sklearn {sklearn.__version__} | tensorflow {'present ' + tf.__version__ if HAVE_TF else 'ABSENT (sklearn MLP fallback)'}")
    log(f"seed={RANDOM_SEED}  bootstrap={N_BOOTSTRAP} (resampled over TEST GROUPS, not rows)")
    log("PRIMARY models: logistic regression + gradient boosting. The MLP is a CHALLENGER,")
    log("present to test three independent critics' measured rejection of neural architectures")
    log("for this task -- not to headline. It is adopted per channel only if a PAIRED")
    log("bootstrap interval on (MLP AUC - LR AUC) excludes zero.")

    # ---- data -------------------------------------------------------------------------
    src = find_input_dir()
    if src:
        log(f"\ncorpus source: {src}")
        df = load_corpus(os.path.join(src, "synthetic_dataset.csv"))
        with open(os.path.join(src, "ground_truth_ledger.json"), "r", encoding="utf-8") as fh:
            ledger = json.load(fh)
    else:
        log("\ncorpus source: NONE FOUND at /kaggle/input/parakh-corpus -- generating a "
            "fallback synthetic corpus so this script still runs standalone.")
        df, ledger = generate_fallback_corpus()
    n = len(df)
    log(f"rows={n}  columns={len(df.columns)}  (all cells read as RAW STRINGS: dtype=str, "
        "keep_default_na=False -- pandas' default na_values would eat the 'N/A'/'NA'/'NULL' "
        "placeholder tokens and merge that channel into `missing`)")

    Y_all, channels_all, sub_labels = build_label_matrix(ledger, n)
    log(f"\nledger channels found: {channels_all}")
    log(f"rows carrying at least one defect: {int((Y_all.sum(axis=1) > 0).sum())} "
        f"({(Y_all.sum(axis=1) > 0).mean():.1%})")

    keep, dropped = select_channels(Y_all, channels_all)
    rule_banner("CHANNEL SELECTION")
    for ch, pos, prev, why in dropped:
        log(f"  DROPPED  {ch:<22} n_pos={pos:>6}  prevalence={prev:>6.1%}  -- {why}")
    for ch in keep:
        j = channels_all.index(ch)
        pos = int(Y_all[:, j].sum())
        log(f"  KEPT     {ch:<22} n_pos={pos:>6}  prevalence={pos / n:>6.1%}")
    if not keep:
        raise RuntimeError("no channel survived selection")
    keep_idx = [channels_all.index(c) for c in keep]
    Y = Y_all[:, keep_idx]

    # ---- parse + split ----------------------------------------------------------------
    log("\nparsing raw cells (4-state ladder: EMPTY / CANONICAL / RECOVERABLE / GARBAGE;")
    log("nullish is an INDEPENDENT flag, not a ladder state, so an unparseable token that")
    log("also looks placeholder-shaped carries both signals instead of being absorbed)")
    states, values = parse_layer(df)
    name_norm = normalise_series(df["work_name"])

    provisional_train = np.zeros(n, dtype=bool)
    provisional_train[: int(0.6 * n)] = True
    nullish = derive_nullish_lexicon(df, np.flatnonzero(provisional_train))
    groups = build_groups(df, name_norm, nullish)
    gsizes = np.bincount(groups)
    log(f"\ngrouping: {len(gsizes)} components over exact normalised work_name UNION work_id; "
        f"largest component = {gsizes.max()} rows ({gsizes.max() / n:.2%} of corpus)")
    if gsizes.max() > 0.01 * n:
        raise AssertionError(
            f"a group holds {gsizes.max()} rows (>1% of the corpus); sentinel work_name values "
            "have collapsed unrelated rows into one indivisible block and fold balancing is void")

    split = grouped_multilabel_split(groups, Y, (0.6, 0.2, 0.2), seed=RANDOM_SEED)
    tr_mask, va_mask, te_mask = split == 0, split == 1, split == 2
    log(f"split (grouped, greedily balanced on the rarest channels first): "
        f"train={tr_mask.sum()} val={va_mask.sum()} test={te_mask.sum()}")

    # The lexicon was derived on a provisional prefix only to build the groups; refit it on
    # the real train rows now so nothing downstream sees a statistic fitted off-fold.
    nullish = derive_nullish_lexicon(df, np.flatnonzero(tr_mask))
    log(f"nullish/placeholder lexicon derived UNSUPERVISED from train rows "
        f"(cross-column document frequency, no hardcoded token list): {sorted(nullish)}")
    log("  the derivation is honest but imperfect: some UNPARSEABLE tokens are also short and")
    log("  appear in several columns, so they enter the lexicon too. That confusion is not")
    log("  hidden -- it is exactly what the rule false-positive breakdown below quantifies.")

    log("\nper-channel positives by split:")
    log(f"  {'channel':<22}{'train':>8}{'val':>8}{'test':>8}   test prevalence")
    for j, ch in enumerate(keep):
        log(f"  {ch:<22}{int(Y[tr_mask, j].sum()):>8}{int(Y[va_mask, j].sum()):>8}"
            f"{int(Y[te_mask, j].sum()):>8}   {Y[te_mask, j].mean():>6.2%}")

    # ---- features ---------------------------------------------------------------------
    log("\nengineering features (all statistics fitted on TRAIN ROWS ONLY; cross-row duplicate")
    log("structure computed WHOLE-FILE and declared transductive by construction)")
    t_feat = time.time()
    X_full, fitted = build_features(df, states, values, nullish, tr_mask, name_norm, RANDOM_SEED)
    X, zero_var, dupes, dependent = prune_features(X_full, tr_mask)
    log(f"features built in {time.time() - t_feat:.1f}s: {X_full.shape[1]} raw columns -> "
        f"{X.shape[1]} modelling columns (dropped {len(zero_var)} zero-variance, "
        f"{len(dupes)} exact duplicates, {len(dependent)} linearly dependent)")
    if zero_var:
        log(f"  zero-variance dropped: {zero_var}")
    if dupes:
        log(f"  duplicate columns dropped: {[f'{a}=={b}' for a, b in dupes][:10]}")
    if dependent:
        log(f"  linearly dependent dropped: {dependent}")
    block_counts = Counter(c.split("__")[0] for c in X.columns)
    log(f"  columns per block: {dict(block_counts)}")
    rank = int(np.linalg.matrix_rank(
        StandardScaler().fit_transform(X.iloc[np.flatnonzero(tr_mask)].to_numpy(dtype=np.float64))))
    log(f"  numerical rank of the train design matrix: {rank} / {X.shape[1]}")
    if rank != X.shape[1]:
        raise AssertionError(
            "the modelling matrix is still rank deficient after the dependency screen; the "
            "logistic baseline would be unidentifiable and the comparison meaningless")

    # ---- ceilings and rules -----------------------------------------------------------
    # Rules and ceilings read the UNPRUNED matrix: a diagnostic column that was dropped for
    # being collinear with another is still perfectly good as a rule input.
    rules = build_rules(df, states, values, nullish, X_full)
    ceilings = evidence_ceilings(df, states, values, sub_labels, Y_all, channels_all, X_full, nullish)
    rule_banner("OBSERVABLE-EVIDENCE CEILINGS (corpus-wide, n and Wilson interval printed)")
    log("The generator's format and missing passes run LAST and overwrite evidence written by")
    log("earlier channels, while the ledger keeps the earlier label. Recall above the ceiling")
    log("is unreachable by ANY model, rule or human. Read every recall against this column.")
    log(f"  {'channel':<22}{'n_pos':>8}{'evidence':>10}{'ceiling':>10}   95% CI")
    for ch in keep:
        c = ceilings.get(ch)
        if not c:
            log(f"  {ch:<22}{'-':>8}{'-':>10}{'not measured':>14}")
            continue
        log(f"  {ch:<22}{c['n_pos']:>8}{c['evidence_present']:>10}{c['ceiling']:>10.3f}   "
            f"[{c['ci'][0]:.3f},{c['ci'][1]:.3f}]")

    rule_banner("DETERMINISTIC RULE BASELINES (whole corpus) + FALSE-POSITIVE CONFUSION")
    log("Precision/recall of four-line rules that invert the generator's edit. On this corpus")
    log("the rule -- not logistic regression -- is the real bar; a model that only matches it")
    log("has learned nothing worth shipping. The FP breakdown matters: a rule's residual")
    log("errors are usually ANOTHER channel's signature, not noise, so 'FP 0' claims about")
    log("single-channel detectors are silently wrong.")
    rule_report = {}
    for name, score in rules.items():
        base_ch = name.split(" (")[0]
        if base_ch not in channels_all:
            continue
        j = channels_all.index(base_ch)
        y = Y_all[:, j]
        tp = int(((score > 0) & (y == 1)).sum())
        fp = int(((score > 0) & (y == 0)).sum())
        fn = int(((score == 0) & (y == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        pl, ph_ = wilson(tp, max(tp + fp, 1))
        rl, rh = wilson(tp, max(tp + fn, 1))
        conf = Counter()
        fp_rows = np.flatnonzero((score > 0) & (y == 0))
        for i in fp_rows:
            for other_j, other_c in enumerate(channels_all):
                if other_c != base_ch and Y_all[i, other_j] == 1:
                    conf[other_c] += 1
        top = ", ".join(f"{c}:{k}" for c, k in conf.most_common(3)) or "none"
        log(f"  {name:<34} P={prec:.3f}[{pl:.3f},{ph_:.3f}] R={rec:.3f}[{rl:.3f},{rh:.3f}] "
            f"TP={tp} FP={fp} FN={fn}")
        log(f"  {'':<34} FPs also carrying: {top}")
        rule_report[name] = {"precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn,
                             "precision_ci": [pl, ph_], "recall_ci": [rl, rh],
                             "fp_confusion": dict(conf.most_common(5))}
    log("")
    log("Two ceilings that belong beside those numbers, both properties of the LABELS rather")
    log("than of any detector:")
    log("  * the duplicate channels label only the CLONE, never the source row it copied, so")
    log("    a symmetric collision rule necessarily flags both members of every pair and its")
    log("    precision is capped at ~0.50 on duplicate_work_id. On duplicate_name it is lower")
    log("    still, because a templated corpus produces legitimate exact name collisions on")
    log("    top of the injected ones.")
    log("  * every recall above is bounded by the observable-evidence ceiling printed in the")
    log("    previous section, not by 1.0.")

    # ---- matrices ---------------------------------------------------------------------
    cols = list(X.columns)
    Xnp = X.to_numpy(dtype=np.float64)
    tr, va, te = np.flatnonzero(tr_mask), np.flatnonzero(va_mask), np.flatnonzero(te_mask)
    scaler = StandardScaler().fit(Xnp[tr])          # FIT ON TRAIN ONLY
    Xs = scaler.transform(Xnp)
    Xs = np.clip(np.nan_to_num(Xs), -30, 30)        # 1e300 is +inf in float32 without this
    Ytr, Yva, Yte = Y[tr], Y[va], Y[te]
    idx_sets = make_group_bootstrap(groups[te], N_BOOTSTRAP, RANDOM_SEED + 5)
    # Bonferroni across every tripwire test in this run (2 per surviving channel).
    TRIPWIRE_ALPHA = 0.05 / max(2 * len(keep), 1)

    # ---- challenger -------------------------------------------------------------------
    rule_banner("TRAINING")
    if HAVE_TF:
        log("MLP challenger: 128-64 dense trunk, shared across channels, sigmoid heads with")
        log("prior-bias init b_c = logit(train prevalence); early stopping on val macro-AP")
        log(f"(patience 15, restore_best_weights), ReduceLROnPlateau, epoch CEILING={EPOCH_CEILING}.")
        t_mlp = time.time()
        mlp, epochs_run = fit_mlp_tf(Xs[tr].astype("float32"), Ytr.astype("float32"),
                                     Xs[va].astype("float32"), Yva.astype("float32"), RANDOM_SEED)
        P_mlp = mlp.predict(Xs[te].astype("float32"), verbose=0)
        log(f"  MLP stopped after {epochs_run} epochs (ceiling {EPOCH_CEILING}) "
            f"in {time.time() - t_mlp:.1f}s; params={mlp.count_params()}")
    else:
        log("TensorFlow absent: challenger falls back to one sklearn MLPClassifier per channel")
        log(f"(early_stopping=True, adaptive LR, epoch CEILING={EPOCH_CEILING}).")
        t_mlp = time.time()
        mlp = SklearnMLPBundle(RANDOM_SEED).fit(Xs[tr], Ytr)
        P_mlp = mlp.predict(Xs[te])
        epochs_run = EPOCH_CEILING
        log(f"  MLP fallback trained in {time.time() - t_mlp:.1f}s")

    results, models_store = {}, {}
    for j, ch in enumerate(keep):
        ytr, yva, yte = Ytr[:, j], Yva[:, j], Yte[:, j]
        if ytr.sum() == 0 or yte.sum() == 0:
            log(f"  {ch}: skipped (no positives in a split)")
            continue
        t_ch = time.time()
        lr, best_c = fit_logreg(Xs[tr], ytr, Xs[va], yva, RANDOM_SEED)
        gbm = fit_gbm(Xnp[tr], ytr, RANDOM_SEED)
        scores = {
            "majority": np.full(len(te), float(ytr.mean())),   # constant: AUC 0.5 by construction
            "rule": rules.get(ch, np.zeros(n))[te],
            "logreg": lr.predict_proba(Xs[te])[:, 1],
            "gbm": gbm.predict_proba(Xnp[te])[:, 1],
            "mlp": P_mlp[:, j],
        }

        # ---- tripwires -----------------------------------------------------------------
        # (a) the random control column, scored against the REAL labels;
        # (b) a shuffled-label refit: one permutation applied to the labels of ALL rows, so
        #     the model trains on permuted train labels and is scored against the matching
        #     permuted test labels. Permuting only the TRAIN labels would not be a null --
        #     the fitted model is then a quasi-random projection of genuinely predictive
        #     features, which correlates with the true test labels in either direction and
        #     the "tripwire" fires on correct runs, which is how a tripwire gets ignored.
        trip = {}
        s_ctrl = Xs[te][:, cols.index("control__random_normal")]
        auc_ctrl = fast_auc(yte, s_ctrl)
        band_ctrl = permutation_null_band(yte, s_ctrl, 4000, RANDOM_SEED + 31 + j, TRIPWIRE_ALPHA)
        trip["random_control"] = {"auc": auc_ctrl, "null_band": list(band_ctrl),
                                  "inside_null": bool(band_ctrl[0] <= auc_ctrl <= band_ctrl[1])}
        yperm = np.random.default_rng(RANDOM_SEED + 99 + j).permutation(Y[:, j])
        lr_shuf = LogisticRegression(C=best_c, max_iter=3000,
                                     random_state=RANDOM_SEED).fit(Xs[tr], yperm[tr])
        s_shuf = lr_shuf.predict_proba(Xs[te])[:, 1]
        auc_shuf = fast_auc(yperm[te], s_shuf)
        band_shuf = permutation_null_band(yperm[te], s_shuf, 4000, RANDOM_SEED + 61 + j,
                                          TRIPWIRE_ALPHA)
        trip["shuffled_labels"] = {"auc": auc_shuf, "null_band": list(band_shuf),
                                   "inside_null": bool(band_shuf[0] <= auc_shuf <= band_shuf[1])}

        # ablation arm: drop the generator-invariant block for the channels it separates.
        abl = ABLATIONS.get(ch)
        if abl:
            keep_cols = [i for i, c in enumerate(cols) if c.split("__")[0] not in abl]
            lr_a, _ = fit_logreg(Xs[tr][:, keep_cols], ytr, Xs[va][:, keep_cols], yva, RANDOM_SEED)
            gbm_a = fit_gbm(Xnp[tr][:, keep_cols], ytr, RANDOM_SEED)
            scores["logreg_no_invariant"] = lr_a.predict_proba(Xs[te][:, keep_cols])[:, 1]
            scores["gbm_no_invariant"] = gbm_a.predict_proba(Xnp[te][:, keep_cols])[:, 1]

        entry = {"n_test": int(len(te)), "n_pos_test": int(yte.sum()),
                 "prevalence_test": float(yte.mean()), "logreg_C": best_c,
                 "ceiling": ceilings.get(ch), "ablated_blocks": list(abl) if abl else [],
                 "tripwires": trip, "models": {}, "seconds": None}
        boot = {}
        for mname, s in scores.items():
            auc = fast_auc(yte, s)
            ap = fast_ap(yte, s)
            aucs, aps = bootstrap_curve(yte, s, idx_sets)
            boot[mname] = aucs
            alo, ahi = ci_from(aucs)
            plo, phi = ci_from(aps)
            hlo, hhi = hanley_mcneil_ci(auc, int(yte.sum()), int(len(yte) - yte.sum()))
            entry["models"][mname] = {
                "roc_auc": auc, "roc_auc_ci_bootstrap": [alo, ahi],
                "roc_auc_ci_hanley_mcneil": [hlo, hhi],
                "average_precision": ap, "average_precision_ci": [plo, phi],
            }
        for a, b in (("mlp", "logreg"), ("mlp", "gbm"), ("gbm", "logreg")):
            d = boot[a] - boot[b]
            lo, hi = ci_from(d)
            entry["models"].setdefault("_deltas", {})[f"{a}_minus_{b}"] = {
                "point": float(fast_auc(yte, scores[a]) - fast_auc(yte, scores[b])),
                "ci": [lo, hi], "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
            }
        entry["seconds"] = round(time.time() - t_ch, 1)
        results[ch] = entry
        models_store[ch] = {"logreg": lr, "gbm": gbm}
        log(f"  {ch:<22} trained (logreg C={best_c}, gbm iters={gbm.n_iter_}) in {entry['seconds']}s")

    # ---- reporting --------------------------------------------------------------------
    rule_banner("PER-CHANNEL RESULTS  --  ROC-AUC [95% bootstrap CI over test groups]")
    log("Baselines first. 'majority' is the constant train-prevalence predictor: its AUC is 0.5")
    log("by construction and its AP equals the test prevalence, which is the only meaningful")
    log("floor for AP. 'rule' is the deterministic validator from the previous section -- on")
    log("this corpus it is the bar that matters. Intervals are 95% bootstrap over test GROUPS.")
    log("Tripwire columns are reported separately below, against a permutation null band.")
    header = (f"{'channel':<20}{'n_pos':>6}{'prev':>7}  {'majority AP':>12}  {'rule':>18}  "
              f"{'logreg':>18}  {'gbm':>18}  {'mlp':>18}")
    log(header)
    log("-" * len(header))
    for ch, e in results.items():
        m = e["models"]
        log(f"{ch:<20}{e['n_pos_test']:>6}{e['prevalence_test']:>7.2%}  "
            f"{m['majority']['average_precision']:>12.3f}  "
            f"{fmt_ci(m['rule']['roc_auc'], *m['rule']['roc_auc_ci_bootstrap']):>18}  "
            f"{fmt_ci(m['logreg']['roc_auc'], *m['logreg']['roc_auc_ci_bootstrap']):>18}  "
            f"{fmt_ci(m['gbm']['roc_auc'], *m['gbm']['roc_auc_ci_bootstrap']):>18}  "
            f"{fmt_ci(m['mlp']['roc_auc'], *m['mlp']['roc_auc_ci_bootstrap']):>18}")

    rule_banner("PER-CHANNEL AVERAGE PRECISION (primary metric under class imbalance)")
    log("The AP floor is the test prevalence, not 0.5. Read the rule column against the")
    log("models: where a four-line rule's interval touches a model's, the rule is what should")
    log("ship. A rule's AP interval can sit slightly off its point estimate -- a binary score")
    log("is heavily tied and AP is not resample-unbiased when the bootstrap duplicates rows.")
    log(f"{'channel':<20}{'n_pos':>6}  {'prevalence(AP floor)':>21}  {'rule AP':>18}  "
        f"{'logreg AP':>18}  {'gbm AP':>18}  {'mlp AP':>18}")
    for ch, e in results.items():
        m = e["models"]
        log(f"{ch:<20}{e['n_pos_test']:>6}  {e['prevalence_test']:>21.4f}  "
            f"{fmt_ci(m['rule']['average_precision'], *m['rule']['average_precision_ci']):>18}  "
            f"{fmt_ci(m['logreg']['average_precision'], *m['logreg']['average_precision_ci']):>18}  "
            f"{fmt_ci(m['gbm']['average_precision'], *m['gbm']['average_precision_ci']):>18}  "
            f"{fmt_ci(m['mlp']['average_precision'], *m['mlp']['average_precision_ci']):>18}")

    rule_banner("TRIPWIRES -- observed AUC vs its PERMUTATION NULL BAND at this n_pos")
    log("Both columns must fall INSIDE their null band. A value outside it is not a weak")
    log("result, it is evidence of a pipeline fault -- fold-boundary leakage, a transductively")
    log("fitted statistic, or an index misalignment between the feature matrix and the")
    log("vectorised ledger -- and every number in this run should be treated as unusable")
    log("until it is explained. The band is family-wise 5% (Bonferroni over every tripwire")
    log("test in this run) and it is WIDE where n_pos is small -- that is the point: at 12")
    log("positives a null column lands anywhere, and pretending otherwise is the error this")
    log("project already made once.")
    log(f"  {'channel':<22}{'random control':>26}{'shuffled-label refit':>30}")
    tripped = []
    for ch, e in results.items():
        c = e["tripwires"]["random_control"]
        s = e["tripwires"]["shuffled_labels"]
        log(f"  {ch:<22}"
            f"{c['auc']:>10.3f} in [{c['null_band'][0]:.3f},{c['null_band'][1]:.3f}]"
            f"{s['auc']:>14.3f} in [{s['null_band'][0]:.3f},{s['null_band'][1]:.3f}]")
        for nm, d in (("random control", c), ("shuffled labels", s)):
            if not d["inside_null"]:
                tripped.append(f"{ch}/{nm}: AUC {d['auc']:.3f} outside null band "
                               f"[{d['null_band'][0]:.3f},{d['null_band'][1]:.3f}]")
    if tripped:
        log("\n  *** TRIPWIRE FIRED -- treat every number above as unusable until explained ***")
        for t in tripped:
            log(f"    {t}")
    else:
        log("\n  every tripwire sits inside its null band: no evidence of fold-boundary leakage,")
        log("  transductive fitting or index misalignment in this run.")

    rule_banner("GENERATOR-INVARIANT ABLATIONS (the HEADLINE number is the 'without' column)")
    log("These two blocks separate their channel almost perfectly because the generator never")
    log("emits a legitimate mismatch. The with-block number measures a generator invariant and")
    log("does not transfer to a real register.")
    any_abl = False
    for ch, e in results.items():
        if not e["ablated_blocks"]:
            continue
        any_abl = True
        m = e["models"]
        log(f"  {ch} (dropped block(s): {e['ablated_blocks']})")
        log(f"    logreg WITH invariant   : {fmt_ci(m['logreg']['roc_auc'], *m['logreg']['roc_auc_ci_bootstrap'])}")
        log(f"    logreg WITHOUT (HEADLINE): {fmt_ci(m['logreg_no_invariant']['roc_auc'], *m['logreg_no_invariant']['roc_auc_ci_bootstrap'])}")
        log(f"    gbm    WITH invariant   : {fmt_ci(m['gbm']['roc_auc'], *m['gbm']['roc_auc_ci_bootstrap'])}")
        log(f"    gbm    WITHOUT (HEADLINE): {fmt_ci(m['gbm_no_invariant']['roc_auc'], *m['gbm_no_invariant']['roc_auc_ci_bootstrap'])}")
    if not any_abl:
        log("  (no ablated channel survived selection in this run)")

    rule_banner("VERDICT PER CHANNEL -- does the MLP challenger beat logistic regression?")
    log("Test: PAIRED bootstrap over the same test-group resamples. Two overlapping marginal")
    log("intervals are NOT a null result and two non-overlapping ones are NOT automatically a")
    log("real gain, so the paired interval on the DIFFERENCE is what decides.")
    verdicts = {}
    for ch, e in results.items():
        d = e["models"]["_deltas"]["mlp_minus_logreg"]
        dg = e["models"]["_deltas"]["mlp_minus_gbm"]
        lo, hi = d["ci"]
        m = e["models"]
        overlap = not (m["mlp"]["roc_auc_ci_bootstrap"][0] > m["logreg"]["roc_auc_ci_bootstrap"][1]
                       or m["logreg"]["roc_auc_ci_bootstrap"][0] > m["mlp"]["roc_auc_ci_bootstrap"][1])
        if d["excludes_zero"] and d["point"] > 0 and dg["excludes_zero"] and dg["point"] > 0:
            v = "MLP ADOPTED: beats both baselines, paired intervals exclude zero"
        elif d["excludes_zero"] and d["point"] > 0:
            v = "MLP beats logreg but NOT gradient boosting -- gradient boosting is sufficient here"
        elif d["excludes_zero"] and d["point"] < 0:
            v = "MLP is WORSE than logistic regression -- logistic regression is sufficient here"
        else:
            v = "no difference: LOGISTIC REGRESSION IS SUFFICIENT HERE"
        rule_auc = m["rule"]["roc_auc"]
        note = ""
        if np.isfinite(rule_auc) and rule_auc >= m["logreg"]["roc_auc_ci_bootstrap"][0]:
            note = ("  || the four-line RULE is inside the model's own interval: ship the rule, "
                    "not the model")
        if e["n_pos_test"] < 30:
            note += (f"  || n_pos={e['n_pos_test']}: interval half-width exceeds any plausible "
                     "effect; NO claim is supportable at this n")
        verdicts[ch] = {"verdict": v, "delta_mlp_minus_logreg": d,
                        "delta_mlp_minus_gbm": dg, "marginal_intervals_overlap": overlap,
                        "note": note.strip()}
        dgl = e["models"]["_deltas"]["gbm_minus_logreg"]
        log(f"  {ch:<20} delta(MLP-LR)={d['point']:+.4f} [{lo:+.4f},{hi:+.4f}]  "
            f"marginal CIs {'overlap' if overlap else 'do NOT overlap'}")
        log(f"  {'':<20} delta(GBM-LR)={dgl['point']:+.4f} "
            f"[{dgl['ci'][0]:+.4f},{dgl['ci'][1]:+.4f}]  "
            f"{'GBM is measurably better' if (dgl['excludes_zero'] and dgl['point'] > 0) else 'the two baselines are indistinguishable'}")
        log(f"  {'':<20} -> {v}{note}")

    n_suff = sum(1 for v in verdicts.values() if "SUFFICIENT" in v["verdict"].upper())
    rule_banner("SUMMARY")
    log(f"channels evaluated: {len(results)}   MLP adopted on: "
        f"{sum(1 for v in verdicts.values() if v['verdict'].startswith('MLP ADOPTED'))}   "
        f"baseline sufficient on: {n_suff}")
    log("A run that concludes 'logistic regression is sufficient' is a SUCCESSFUL result here:")
    log("it says the engineered features, not the model class, carry the signal -- which is")
    log("exactly what three independent critics predicted from the generator's source.")
    log("Every channel above is a channel of INJECTED DATA-QUALITY DEFECTS. Several are")
    log("near-deterministic because a deterministic rule injected them; those scores measure")
    log("recovery of the injection rule on synthetic data and are NOT evidence of detection")
    log("skill on a real register.")

    # ---- persistence ------------------------------------------------------------------
    payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "task": "multi-label injected data-quality defect channel detection (synthetic corpus)",
        "not_fraud_detection": True,
        "corpus": {"rows": int(n), "source": src or "fallback generator"},
        "seed": RANDOM_SEED,
        "channels_kept": keep,
        "channels_dropped": [{"channel": c, "n_pos": p, "prevalence": pr, "reason": w}
                             for c, p, pr, w in dropped],
        "split": {"train": int(tr_mask.sum()), "val": int(va_mask.sum()), "test": int(te_mask.sum()),
                  "grouping": "connected components over exact normalised work_name UNION work_id",
                  "duplicate_features": "whole-file blocking, transductive by construction"},
        "features": {"n_columns": int(X.shape[1]), "blocks": dict(block_counts),
                     "train_design_rank": rank, "zero_variance_dropped": zero_var,
                     "duplicate_columns_dropped": [[a, b] for a, b in dupes],
                     "linearly_dependent_dropped": dependent,
                     "nullish_lexicon": fitted["nullish"],
                     "gated_frequency_columns": fitted["gated_freq_columns"]},
        "evidence_ceilings": ceilings,
        "rule_baselines": rule_report,
        "results": results,
        "verdicts": verdicts,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "pandas": pd.__version__, "sklearn": sklearn.__version__,
                        "tensorflow": tf.__version__ if HAVE_TF else None},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
    with open(os.path.join(OUT_DIR, "baseline_models.pkl"), "wb") as fh:
        pickle.dump({"scaler": scaler, "columns": cols, "per_channel": models_store}, fh)
    if HAVE_TF:
        mlp.save(os.path.join(OUT_DIR, "mlp_challenger.keras"))
    else:
        with open(os.path.join(OUT_DIR, "mlp_challenger.pkl"), "wb") as fh:
            pickle.dump(mlp, fh)
    log(f"\nwrote {os.path.join(OUT_DIR, 'metrics.json')}, baseline_models.pkl and the MLP "
        f"challenger to {OUT_DIR}")
    log(f"total runtime: {time.time() - t0:.1f}s")

    log("")
    log("NOTE: this model detects INJECTED DATA-QUALITY DEFECTS in a synthetic")
    log("corpus. It is not a fraud detector and has never seen a fraud label.")


if __name__ == "__main__":
    main()
