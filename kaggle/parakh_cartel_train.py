"""PARAKH cartel screen - supervised bid-rigging detection on labelled data.

    !pip -q install pyreadr
    !python parakh_cartel_train.py --parquet /kaggle/input/parakh-cartels/labelled_cartels.parquet

WHY THIS IS DIFFERENT FROM EVERYTHING ELSE IN THIS PROJECT
-----------------------------------------------------------
Every previous PARAKH model was trained on a surrogate: masked-field
reconstruction, injected data-quality defects, synthetic corpora. None had a
fraud label, because none existed.

This one does. 18,807 bids from 73 bid-rigging cartels across 7 European
countries, prosecuted by national competition authorities, with a binary
``is_cartel`` outcome. 8,045 positive, 10,762 negative. It is the first
genuinely supervised fraud task in the project.

Source: Public procurement cartels - a large-sample testing of screens using
machine learning. Zenodo 17595875, CC-BY-NC-3.0. Attribution required;
non-commercial use only.

WHAT NaN MEANS, AND WHY 99.5% OF THE DATA IS NOT HERE
------------------------------------------------------
The source file holds 3,861,477 rows but only 18,807 carry a label. The other
3,842,670 have ``is_cartel = NaN``, and that is **not** "clean" - it is "never
investigated". Treating them as negatives would manufacture 3.8 million false
negatives and produce a model that predicts prosecution coverage rather than
collusion. They are dropped, not defaulted. This is the same rule the rest of
PARAKH runs on and the same error that once put 5,237 bogus labels on a postal
directory.

THE TWO SPLIT RULES THAT DECIDE WHETHER ANY NUMBER HERE IS REAL
---------------------------------------------------------------
**Group by cartel_id.** Bids from one cartel are not independent - they were
placed by the same firms, coordinated by the same agreement, often in the same
month. A random row split puts siblings on both sides and the model recognises
the cartel rather than the collusion. Group sizes here are extreme (ES2 alone
has 2,779 rows, 15% of the data), so this is not a marginal correction: a
random split would inflate every metric substantially.

**Leave one country out.** The question PARAKH actually needs answered is not
"does a screen work in Spain" but "does a screen trained on some jurisdictions
work on one it has never seen". Training on six countries and testing on the
seventh measures exactly that, and it is the closest available proxy for
whether any of this transfers to India. Report it separately from the pooled
number, because it is the honest one and it will be worse.

WHAT IT STILL CANNOT CLAIM
--------------------------
These are European tenders under EU procurement law. A screen validated here is
evidence that the *method* transfers, not that its thresholds do. Any threshold
must be refitted on local data before use, and the model outputs a
prioritisation score for an investigator - never a finding.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

#: Screens from the bid-rigging literature. Each is a within-tender statistic:
#: collusion shows up in the SHAPE of a bid distribution, not in any one bid,
#: so every feature below is computed per tender and broadcast back to rows.
#:
#: The dataset has NO tender identifier, so this key reconstructs one. Getting
#: it wrong is silent and fatal: the first version keyed on
#: (buyer, year, cpv4) over the LABELLED subset only, which produced groups of
#: median size ONE. Every within-tender screen was then computed on a single
#: bid - no spread, no runner-up, no kurtosis - and the model scored 0.52 while
#: appearing to work. The features were measuring nothing.
#:
#: Measured on the full 3.86M-row frame:
#:   buyer+call+cpv4              median 8 bids, 21.1% exact match on
#:                                lot_bidscount, 79.0% of labelled rows in a
#:                                group of >= 2  <- chosen
#:   buyer+call+cpv4+tender_est   median 8, 21.9% exact, 74.8% usable
#:   buyer+call                   median 17, 10.4% exact
#:
#: 21% exact agreement is weak - a buyer can run several tenders on one day in
#: one category - so the screens are APPROXIMATE and that is a stated
#: limitation of every number this script produces, not a detail.
TENDER_KEY = [
    "buyer_id",
    "tender_publications_firstcallfortenderdate",
    "main_cpv_4",
]


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_features(d: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Bid-distribution screens, computed per tender.

    MUST be called on the FULL frame, including unlabelled rows. Only 18,807
    of 3,861,477 rows carry a label, and the sibling bids of a labelled tender
    are overwhelmingly among the unlabelled ones. Computing screens on the
    labelled subset alone leaves each tender with a single bid and silently
    destroys every statistic here. The unlabelled rows supply the tender
    CONTEXT; they never supply an outcome.

    Named after the literature they come from so a reviewer can check each
    against its source rather than against this code:

    * **CV** - coefficient of variation of bids. Cartels coordinate, which
      compresses the spread relative to genuine competition.
    * **SPD** - relative distance between the winning and second bid, scaled
      by the spread of the losers. The classic Imhof screen: cover bids sit
      close to the winner while the losing pack is spread out behind.
    * **DIFFP** - percentage difference between winner and runner-up.
    * **RD** - relative distance normalised by the standard deviation.
    * **KURT / SKEW** - a coordinated bid pack is peaked and asymmetric.
    * **ALTERNATIVE CAPACITY** - the number of distinct bidders and whether
      the tender drew a single bid at all.

    Args:
        d: Labelled bid rows.

    Returns:
        The frame with features attached, and the feature name list.
    """
    d = d.copy()
    d["_tid"] = (
        d[TENDER_KEY].astype(str).agg("|".join, axis=1)
    )

    price = pd.to_numeric(d["bid_price"], errors="coerce")
    d["_price"] = price.where(price > 0)

    g = d.groupby("_tid")["_price"]
    d["t_n_bids"] = g.transform("size")
    d["t_mean"] = g.transform("mean")
    d["t_std"] = g.transform("std")
    d["t_min"] = g.transform("min")
    d["t_max"] = g.transform("max")

    # CV: the single most cited screen. Undefined for one-bid tenders, and
    # left NaN there rather than filled - a tender with no competition has no
    # spread, which is a different fact from a spread of zero.
    d["scr_cv"] = d["t_std"] / d["t_mean"].replace(0, np.nan)

    # Winner/runner-up geometry. Rank within tender, then read off the top two.
    d["_rank"] = d.groupby("_tid")["_price"].rank(method="first")
    first = d[d["_rank"] == 1].set_index("_tid")["_price"]
    second = d[d["_rank"] == 2].set_index("_tid")["_price"]
    d["_p1"] = d["_tid"].map(first)
    d["_p2"] = d["_tid"].map(second)

    d["scr_diffp"] = (d["_p2"] - d["_p1"]) / d["_p1"].replace(0, np.nan)
    d["scr_rd"] = (d["_p2"] - d["_p1"]) / d["t_std"].replace(0, np.nan)

    # SPD: winner-to-runner-up gap against the spread of the LOSING pack only.
    losers_std = (
        d[d["_rank"] > 1].groupby("_tid")["_price"].std().rename("_ls")
    )
    d["_ls"] = d["_tid"].map(losers_std)
    d["scr_spd"] = (d["_p2"] - d["_p1"]) / d["_ls"].replace(0, np.nan)

    d["scr_kurt"] = d["_tid"].map(d.groupby("_tid")["_price"].apply(pd.Series.kurt))
    d["scr_skew"] = d["_tid"].map(d.groupby("_tid")["_price"].apply(pd.Series.skew))

    d["scr_range_ratio"] = (d["t_max"] - d["t_min"]) / d["t_mean"].replace(0, np.nan)
    d["scr_n_bids"] = pd.to_numeric(d["lot_bidscount"], errors="coerce").fillna(
        d["t_n_bids"]
    )
    d["scr_singleb"] = pd.to_numeric(d["singleb"], errors="coerce").fillna(0)

    # Bid against the buyer's own estimate: over-estimate capture is a
    # separate signal from bid-pack shape.
    est = pd.to_numeric(d["lot_estimatedprice"], errors="coerce").replace(0, np.nan)
    d["scr_price_to_est"] = d["_price"] / est

    d["scr_is_winner"] = pd.to_numeric(d["bid_iswinning"], errors="coerce").fillna(0)
    d["scr_consortium"] = pd.to_numeric(d["bid_isconsortium"], errors="coerce").fillna(0)
    d["scr_subcontracted"] = pd.to_numeric(
        d["bid_issubcontracted"], errors="coerce"
    ).fillna(0)

    # Bidder-level experience. Computed on the WHOLE labelled frame, which is
    # a mild transductive leak but a structural feature rather than an outcome
    # one; flagged here so it is not mistaken for a clean design.
    d["scr_bidder_freq"] = d.groupby("bidder_id")["bidder_id"].transform("size")
    d["scr_buyer_freq"] = d.groupby("buyer_id")["buyer_id"].transform("size")

    names = [c for c in d.columns if c.startswith("scr_")]

    # Every screen gets an explicit definedness flag. A NaN screen means the
    # tender could not support it (one bid, no estimate), which is different
    # from a screen that computed to zero - and the difference is exactly what
    # separates "no competition" from "perfectly tied competition".
    for n in list(names):
        d[n + "_ok"] = d[n].notna().astype(int)
    names += [n + "_ok" for n in names]

    d[names] = d[names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return d, names


def auc_ci(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil 95% half-width. Reported with every AUC, always."""
    if n_pos < 2 or n_neg < 2:
        return float("nan")
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (
        auc * (1 - auc)
        + (n_pos - 1) * (q1 - auc ** 2)
        + (n_neg - 1) * (q2 - auc ** 2)
    ) / (n_pos * n_neg)
    return 1.96 * math.sqrt(max(var, 0.0))


def evaluate(y, p) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos < 2 or n_neg < 2:
        return {"n_pos": n_pos, "n_neg": n_neg, "roc_auc": float("nan"),
                "ci95": float("nan"), "pr_auc": float("nan")}
    auc = float(roc_auc_score(y, p))
    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "roc_auc": auc,
        "ci95": auc_ci(auc, n_pos, n_neg),
        "pr_auc": float(average_precision_score(y, p)),
        "base_rate": float(y.mean()),
    }


def models(seed: int) -> Dict[str, object]:
    """Baselines first. The MLP has to beat them, not merely run."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logistic": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, early_stopping=True, random_state=seed
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                early_stopping=True,
                n_iter_no_change=15,
                max_iter=400,
                random_state=seed,
            ),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=str, default=None)
    parser.add_argument("--rdata", type=str, default=None)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--out", type=str, default="cartel_out")
    args = parser.parse_args()

    set_seeds(args.seed)
    from sklearn.model_selection import GroupKFold

    path = args.parquet
    if path is None:
        found = list(Path("/kaggle/input").rglob("labelled_cartels.parquet"))
        if found:
            path = str(found[0])
    if path is None and args.rdata:
        import pyreadr

        d = pyreadr.read_r(args.rdata)["df_full"]
        d = d[d.is_cartel.notna()].copy()
        d["is_cartel"] = d.is_cartel.astype(bool).astype(int)
    elif path:
        d = pd.read_parquet(path)
    else:
        raise SystemExit(
            "No data. Pass --parquet, or --rdata, or attach the "
            "parakh-cartels dataset."
        )

    if "labelled" not in d.columns:
        d["labelled"] = d["is_cartel"].notna()
    n_full = len(d)
    n_lab = int(d["labelled"].sum())
    if n_lab == n_full:
        raise SystemExit(
            "This frame contains ONLY labelled rows. Screens computed on it "
            "would see one bid per tender and measure nothing - see the "
            "TENDER_KEY comment. Pass the full frame (full_cartels.parquet)."
        )

    print("=" * 78)
    print("PARAKH cartel screen - SUPERVISED, real prosecuted outcomes")
    print(f"  full frame     {n_full:,} rows (tender context)")
    print(f"  labelled       {n_lab:,} rows ({n_lab / n_full:.2%})")
    print(f"  unlabelled     {n_full - n_lab:,} - NOT negatives, dropped from "
          "training")
    print("=" * 78, flush=True)

    # Screens over the whole frame, THEN subset. The order is the fix.
    d, feats = build_features(d)
    print(f"\n{len(feats)} screen features built on the full frame")

    sizes = d.groupby("_tid")["_tid"].transform("size")
    true_n = pd.to_numeric(d["lot_bidscount"], errors="coerce")
    print(f"  tender reconstruction: median {sizes.median():.0f} bids/group, "
          f"{float((sizes == true_n).mean()):.1%} exact vs lot_bidscount")

    d = d[d["labelled"]].copy()
    usable = float((sizes[d.index] >= 2).mean()) if len(d) else 0.0
    print(f"  labelled rows kept: {len(d):,}  "
          f"({usable:.1%} sit in a tender with >= 2 bids, so their screens "
          "are defined)")
    print(f"  positives {int(d.is_cartel.sum()):,}   negatives "
          f"{int((1 - d.is_cartel).sum()):,}   cartels "
          f"{d.cartel_id.nunique()}   countries {d.country.nunique()}",
          flush=True)

    X = d[feats].to_numpy(dtype=float)
    y = d["is_cartel"].to_numpy(dtype=int)
    groups = d["cartel_id"].astype(str).to_numpy()
    report: Dict[str, object] = {
        "n_rows_full": int(n_full),
        "n_rows_labelled": int(len(d)),
        "n_features": len(feats),
        "_tender_key": TENDER_KEY,
        "_tender_reconstruction": (
            "No tender_id exists in the source. Screens use a reconstructed "
            "key with ~21% exact agreement against lot_bidscount, so every "
            "within-tender statistic here is approximate."
        ),
        "features": feats,
        "_source": "Zenodo 17595875, CC-BY-NC-3.0",
        "_claim_limit": (
            "European tenders under EU procurement law. Validates that the "
            "METHOD transfers, not that its thresholds do. Output is an "
            "investigator prioritisation score, never a finding."
        ),
    }

    # ---- grouped CV: never split a cartel across the boundary -------------
    print("\n" + "-" * 78)
    print("GROUPED CV (5-fold, grouped by cartel_id)")
    print("-" * 78)
    gkf = GroupKFold(n_splits=5)
    report["grouped_cv"] = {}
    for name, mk in models(args.seed).items():
        oof = np.zeros(len(y))
        for tr, te in gkf.split(X, y, groups):
            m = models(args.seed)[name]
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        r = evaluate(y, oof)
        report["grouped_cv"][name] = r
        print(f"  {name:20s} ROC-AUC {r['roc_auc']:.4f} +/-{r['ci95']:.4f}   "
              f"PR-AUC {r['pr_auc']:.4f}   (n+ {r['n_pos']:,} / n- {r['n_neg']:,})",
              flush=True)

    # ---- leave one country out: the transfer test ------------------------
    print("\n" + "-" * 78)
    print("LEAVE-ONE-COUNTRY-OUT - the honest number, and the India proxy")
    print("-" * 78)
    report["loco"] = {}
    for country in sorted(d.country.unique()):
        te = (d.country == country).to_numpy()
        tr = ~te
        if y[te].sum() < 10 or (1 - y[te]).sum() < 10:
            print(f"  {str(country):4s} SKIPPED - too few of one class in holdout")
            continue
        row = {}
        for name in models(args.seed):
            m = models(args.seed)[name]
            m.fit(X[tr], y[tr])
            row[name] = evaluate(y[te], m.predict_proba(X[te])[:, 1])
        report["loco"][str(country)] = row
        best = max(row, key=lambda k: row[k]["roc_auc"])
        print(f"  {str(country):4s} n={int(te.sum()):6,}  " + "  ".join(
            f"{k[:4]} {v['roc_auc']:.3f}" for k, v in row.items()
        ) + f"   best={best}", flush=True)

    # ---- verdict ---------------------------------------------------------
    gcv = report["grouped_cv"]
    mlp, lr, gb = gcv["mlp"], gcv["logistic"], gcv["gradient_boosting"]
    best_base = max(lr, gb, key=lambda r: r["roc_auc"])
    beats = mlp["roc_auc"] - best_base["roc_auc"] > (mlp["ci95"] + best_base["ci95"])
    verdict = (
        f"MLP {mlp['roc_auc']:.4f} vs best baseline {best_base['roc_auc']:.4f} - "
        + ("MLP WINS beyond the intervals."
           if beats else
           "NOT a significant win. Prefer the baseline: it is interpretable "
           "and an evidentiary system needs to explain its ranking.")
    )
    report["verdict"] = verdict
    print("\n" + "=" * 78)
    print(verdict)

    loco_aucs = [v[k]["roc_auc"] for v in report["loco"].values() for k in v]
    if loco_aucs:
        drop = gcv[max(gcv, key=lambda k: gcv[k]["roc_auc"])]["roc_auc"] - float(
            np.mean(loco_aucs)
        )
        report["transfer_gap"] = drop
        print(f"Transfer gap (pooled CV -> unseen country): {drop:+.4f}")
        print("That gap, not the pooled number, is what a new jurisdiction "
              "should expect.")
    print("=" * 78)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out / "metrics.json", "w"), indent=2, default=str)
    print(f"\nwritten to {out}/metrics.json")
    print("\nNOTE: prosecuted EU cartels. Thresholds must be refitted on local "
          "data before any operational use.")


if __name__ == "__main__":
    main()
