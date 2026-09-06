"""PARAKH consistency network - Kaggle-ready trainer (TensorFlow/Keras).

Self-contained. Imports nothing from the PARAKH repo, so it runs as a Kaggle
notebook cell or `python parakh_consistency_train.py` anywhere.

    !python parakh_consistency_train.py --csv /kaggle/input/parakh/records.csv
    !python parakh_consistency_train.py --demo          # generates its own data

WHAT IT LEARNS
--------------
Mask one field of a record, predict it from the rest. The ground truth is the
field's own value: free, exact, unlimited, and needing no audit. A record
whose fields fail to predict each other is internally inconsistent - the
closest honest analogue to an "illegal move".

WHAT IT DOES NOT LEARN
----------------------
Fraud. Nothing here sees a fraud label because none exists. High surprise
means "this record disagrees with itself", which is a reason to look, not a
finding. Do not report its output as a fraud probability.

FIXES CARRIED OVER FROM THE FAILED FIRST VERSION
------------------------------------------------
The first version scored surprise over categorical heads ONLY, reached
ROC-AUC 0.538 against injected defects, and the post-mortem found two causes.
Both are fixed here:

1. **Numeric and temporal heads now contribute to surprise.** Previously a
   cost outlier or a date-order violation was invisible to the score by
   construction, because the score never looked at those heads.
2. **Evaluation is per defect channel, not against "any defect".** 88% of
   rows in the reference corpus carry some defect, mostly missingness, so
   "any defect" is a near-degenerate target on which AUC measures nothing.

Expect the headline number to be modest. That is the honest range for this
task, and a version of this script reporting 0.95 would be measuring
something other than what it claims.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --- schema ---------------------------------------------------------------

CATEGORICAL_FIELDS = ("state", "district", "implementing_agency", "status")
NUMERIC_FIELDS = ("sanction_amount", "amount_spent")
DATE_FIELDS = ("date_proposal", "date_approval", "date_completion")

#: Derived durations. Modelled explicitly because a date-order violation is a
#: NEGATIVE duration, and a model that only sees raw dates has to rediscover
#: subtraction before it can notice one.
DURATION_FIELDS = ("days_proposal_to_approval", "days_approval_to_completion")

MIN_CATEGORY_COUNT = 20
OTHER, MISSING = "__OTHER__", "__MISSING__"


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


# --- data -----------------------------------------------------------------


def make_demo_frame(n: int = 20000, seed: int = 0) -> pd.DataFrame:
    """A defect-injected demo corpus so the script runs with no input file.

    Structure mirrors an Indian public-works register: districts nest inside
    states, costs are lognormal, and a known fraction of rows are corrupted so
    the evaluation at the end has something to measure. The returned
    ``defect_channel`` column is the ONLY label here and it describes data
    quality, never fraud.
    """
    rng = np.random.default_rng(seed)
    states = [f"STATE_{i:02d}" for i in range(12)]
    # Districts nest inside states, so predicting state from district is
    # nearly a lookup. Reported separately for exactly that reason.
    districts = {s: [f"{s}_D{j:02d}" for j in range(8)] for s in states}
    agencies = [f"AGENCY_{i:02d}" for i in range(6)]

    state = rng.choice(states, n)
    district = np.array([rng.choice(districts[s]) for s in state])
    agency = rng.choice(agencies, n, p=np.array([0.4, 0.2, 0.15, 0.1, 0.1, 0.05]))
    sanction = np.exp(rng.normal(13.5, 1.1, n))
    spent = sanction * rng.uniform(0.6, 1.05, n)
    proposal = pd.Timestamp("2020-01-01") + pd.to_timedelta(
        rng.integers(0, 1000, n), unit="D"
    )
    approval = proposal + pd.to_timedelta(rng.integers(10, 120, n), unit="D")
    completion = approval + pd.to_timedelta(rng.integers(30, 700, n), unit="D")
    status = rng.choice(["COMPLETED", "ONGOING", "SANCTIONED"], n, p=[0.6, 0.3, 0.1])

    frame = pd.DataFrame(
        {
            "state": state,
            "district": district,
            "implementing_agency": agency,
            "status": status,
            "sanction_amount": sanction,
            "amount_spent": spent,
            "date_proposal": proposal,
            "date_approval": approval,
            "date_completion": completion,
        }
    )
    frame["defect_channel"] = "clean"

    def pick(rate: float) -> np.ndarray:
        return rng.choice(n, size=int(n * rate), replace=False)

    rows = pick(0.05)
    frame.loc[rows, "sanction_amount"] *= rng.choice([12.0, 0.08], len(rows))
    frame.loc[rows, "defect_channel"] = "cost_outlier"

    rows = pick(0.04)
    frame.loc[rows, "date_approval"] = frame.loc[rows, "date_proposal"] - pd.to_timedelta(
        rng.integers(5, 90, len(rows)), unit="D"
    )
    frame.loc[rows, "defect_channel"] = "date_order"

    rows = pick(0.03)
    frame.loc[rows, "implementing_agency"] = rng.choice(agencies, len(rows))
    frame.loc[rows, "defect_channel"] = "agency_mismatch"

    rows = pick(0.03)
    frame.loc[rows, "amount_spent"] = frame.loc[rows, "sanction_amount"] * rng.uniform(
        1.5, 4.0, len(rows)
    )
    frame.loc[rows, "defect_channel"] = "overspend"

    return frame


def add_durations(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach signed day gaps. Negative values are the violation signal."""
    out = frame.copy()
    dates = {
        name: pd.to_datetime(out[name], errors="coerce")
        if name in out.columns
        else pd.Series(pd.NaT, index=out.index)
        for name in DATE_FIELDS
    }
    out["days_proposal_to_approval"] = (
        dates["date_approval"] - dates["date_proposal"]
    ).dt.days
    out["days_approval_to_completion"] = (
        dates["date_completion"] - dates["date_approval"]
    ).dt.days
    return out


def encode(
    frame: pd.DataFrame,
    vocabularies: Optional[Dict[str, List[str]]] = None,
    stats: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict:
    """Categorical indices + standardised numerics.

    A holdout must be encoded with the TRAINING vocabulary and statistics.
    Letting it build its own would leak its distribution into its encoding and
    quietly flatter every number computed on it.
    """
    numeric_names = list(NUMERIC_FIELDS) + list(DURATION_FIELDS)
    cats: Dict[str, np.ndarray] = {}
    vocab_out: Dict[str, List[str]] = {}

    for name in CATEGORICAL_FIELDS:
        col = (
            frame[name].astype(str).str.strip().str.upper()
            if name in frame.columns
            else pd.Series([MISSING] * len(frame), index=frame.index)
        )
        col = col.replace({"": MISSING, "NAN": MISSING, "NONE": MISSING})
        if vocabularies and name in vocabularies:
            vocab = list(vocabularies[name])
        else:
            counts = col.value_counts()
            kept = [str(v) for v in counts[counts >= MIN_CATEGORY_COUNT].index]
            vocab = [MISSING, OTHER] + [v for v in kept if v not in (MISSING, OTHER)]
        lookup = {v: i for i, v in enumerate(vocab)}
        cats[name] = col.map(lambda v: lookup.get(v, lookup[OTHER])).to_numpy("int32")
        vocab_out[name] = vocab

    values, present, stats_out = [], [], {}
    for name in numeric_names:
        raw = (
            pd.to_numeric(frame[name], errors="coerce")
            if name in frame.columns
            else pd.Series(np.nan, index=frame.index)
        )
        seen = raw.notna().to_numpy("float32")
        # signed log: preserves the sign of a negative duration, which is
        # exactly the date-order violation the model must be able to see.
        scaled = np.sign(raw.fillna(0.0)) * np.log1p(raw.abs().fillna(0.0))
        scaled = scaled.to_numpy("float64")
        if stats and name in stats:
            mean, std = stats[name]
        else:
            obs = scaled[seen.astype(bool)]
            mean = float(obs.mean()) if obs.size else 0.0
            std = float(obs.std()) or 1.0
        values.append((((scaled - mean) / std) * seen).astype("float32"))
        present.append(seen)
        stats_out[name] = (mean, std)

    return {
        "categorical": cats,
        "numeric": np.stack(values, 1),
        "present": np.stack(present, 1),
        "vocabularies": vocab_out,
        "stats": stats_out,
        "numeric_names": numeric_names,
        "index": frame.index,
        "n": len(frame),
    }


def mask_inputs(enc: Dict, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Hide exactly one field per row behind the MASK token."""
    n_blocks = len(CATEGORICAL_FIELDS) + 1
    choice = rng.integers(0, n_blocks, size=enc["n"])
    out: Dict[str, np.ndarray] = {}
    for i, name in enumerate(CATEGORICAL_FIELDS):
        v = enc["categorical"][name].copy()
        v[choice == i] = len(enc["vocabularies"][name])
        out[f"in_{name}"] = v.reshape(-1, 1)
    numeric, present = enc["numeric"].copy(), enc["present"].copy()
    hit = choice == len(CATEGORICAL_FIELDS)
    numeric[hit] = 0.0
    present[hit] = 0.0
    out["in_numeric"] = numeric
    out["in_present"] = present
    return out


def build_model(enc: Dict, width: int, seed: int):
    from tensorflow import keras
    from tensorflow.keras import layers

    init = keras.initializers.GlorotUniform(seed=seed)
    inputs, parts = {}, []
    for name in CATEGORICAL_FIELDS:
        size = len(enc["vocabularies"][name])
        t = keras.Input(shape=(1,), dtype="int32", name=f"in_{name}")
        inputs[f"in_{name}"] = t
        e = layers.Embedding(size + 1, max(4, min(32, size // 2)),
                             embeddings_initializer=init)(t)
        parts.append(layers.Flatten()(e))

    k = len(enc["numeric_names"])
    n_in = keras.Input(shape=(k,), name="in_numeric")
    p_in = keras.Input(shape=(k,), name="in_present")
    inputs["in_numeric"], inputs["in_present"] = n_in, p_in
    parts += [n_in, p_in]

    h = layers.Concatenate()(parts)
    h = layers.Dense(width, activation="relu", kernel_initializer=init)(h)
    h = layers.Dropout(0.1, seed=seed)(h)
    h = layers.Dense(width, activation="relu", kernel_initializer=init)(h)

    outputs = {
        f"out_{name}": layers.Dense(
            len(enc["vocabularies"][name]), activation="softmax",
            name=f"out_{name}", kernel_initializer=init)(h)
        for name in CATEGORICAL_FIELDS
    }
    outputs["out_numeric"] = layers.Dense(k, name="out_numeric",
                                          kernel_initializer=init)(h)
    return keras.Model(inputs, outputs)


def surprise(model, enc: Dict, rng: np.random.Generator) -> np.ndarray:
    """Per-record surprise across ALL heads.

    The first version summed categorical heads only, which made numeric and
    temporal defects structurally undetectable. Numeric heads contribute
    squared error standardised across the holdout, so the two kinds of head
    are on a comparable scale before being added.
    """
    predictions = model.predict(mask_inputs(enc, rng), verbose=0)
    total = np.zeros(enc["n"])

    for name in CATEGORICAL_FIELDS:
        probability = predictions[f"out_{name}"]
        actual = enc["categorical"][name]
        chosen = probability[np.arange(len(actual)), actual]
        total += -np.log(np.clip(chosen, 1e-9, 1.0))

    error = (predictions["out_numeric"] - enc["numeric"]) ** 2
    error *= enc["present"]
    for column in range(error.shape[1]):
        values = error[:, column]
        std = values.std() or 1.0
        total += (values - values.mean()) / std

    return total


def baseline_metrics(train: Dict, test: Dict) -> Dict[str, float]:
    """Mode baseline per categorical head - the bar the network must clear.

    A head that cannot beat predicting the training mode has learned nothing.
    Accuracy printed without this comparison is not interpretable: 0.91 on a
    field where 90% of rows share one value is a constant predictor.

    Module-level so the long-run trainer can import it rather than keeping a
    second copy that drifts.
    """
    baseline: Dict[str, float] = {}
    for name in CATEGORICAL_FIELDS:
        values, counts = np.unique(train["categorical"][name], return_counts=True)
        mode = values[int(np.argmax(counts))]
        baseline[name] = float((test["categorical"][name] == mode).mean())
    return baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--out", type=str, default="parakh_consistency_out")
    args = parser.parse_args()

    set_seeds(args.seed)
    from tensorflow import keras
    from sklearn.metrics import roc_auc_score

    if args.csv:
        frame = pd.read_csv(args.csv, low_memory=False)
    else:
        if not args.demo:
            print("no --csv given; falling back to --demo")
        frame = make_demo_frame(seed=args.seed)

    frame = add_durations(frame)
    channels = (
        frame["defect_channel"].astype(str)
        if "defect_channel" in frame.columns
        else None
    )

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(frame))
    cut = int(len(frame) * args.holdout)
    test_frame = frame.iloc[order[:cut]]
    train_frame = frame.iloc[order[cut:]]

    train = encode(train_frame)
    test = encode(test_frame, train["vocabularies"], train["stats"])

    model = build_model(train, args.width, args.seed)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={**{f"out_{n}": "sparse_categorical_crossentropy"
                 for n in CATEGORICAL_FIELDS}, "out_numeric": "mse"},
        metrics={f"out_{n}": "accuracy" for n in CATEGORICAL_FIELDS},
    )

    targets = {**{f"out_{n}": train["categorical"][n] for n in CATEGORICAL_FIELDS},
               "out_numeric": train["numeric"]}
    history = model.fit(
        mask_inputs(train, rng), targets,
        validation_split=0.15, epochs=args.epochs, batch_size=args.batch_size,
        shuffle=True, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True)],
    )

    evaluated = model.evaluate(
        mask_inputs(test, rng),
        {**{f"out_{n}": test["categorical"][n] for n in CATEGORICAL_FIELDS},
         "out_numeric": test["numeric"]},
        verbose=0, return_dict=True)

    baseline = baseline_metrics(train, test)

    scores = surprise(model, test, rng)
    report: Dict[str, object] = {
        "manifest": {
            "framework": "tensorflow", "seed": args.seed,
            "n_train": int(len(train_frame)), "n_holdout": int(len(test_frame)),
            "epochs_max": args.epochs, "epochs_run": len(history.history["loss"]),
            "early_stopped": len(history.history["loss"]) < args.epochs,
            "n_parameters": int(model.count_params()),
        },
        "reconstruction": {}, "defect_detection": {},
    }

    print("\nfield                     model   baseline   lift")
    for name in CATEGORICAL_FIELDS:
        accuracy = float(evaluated.get(f"out_{name}_accuracy", float("nan")))
        report["reconstruction"][name] = {
            "accuracy": accuracy, "mode_baseline": baseline[name],
            "lift": accuracy - baseline[name]}
        print(f"  {name:22s} {accuracy:.4f}  {baseline[name]:.4f}  "
              f"{accuracy - baseline[name]:+.4f}")

    # Per channel, never against "any defect": a target carrying 88% positives
    # makes AUC meaningless, which is what sank the first version.
    if channels is not None:
        holdout_channels = channels.loc[test_frame.index]
        print("\ndefect channel        n     ROC-AUC")
        for channel in sorted(set(holdout_channels) - {"clean"}):
            truth = (holdout_channels == channel).to_numpy().astype(int)
            if truth.sum() < 20:
                continue
            keep = (holdout_channels.isin([channel, "clean"])).to_numpy()
            auc = float(roc_auc_score(truth[keep], scores[keep]))
            report["defect_detection"][channel] = {
                "n_positive": int(truth.sum()), "roc_auc_vs_clean": auc}
            print(f"  {channel:20s} {int(truth.sum()):5d}   {auc:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "consistency_model.keras")
    json.dump(report, open(out / "metrics.json", "w"), indent=2)
    pd.DataFrame({"surprise": scores}, index=test_frame.index).to_csv(
        out / "holdout_surprise.csv")
    print(f"\nwritten to {out}/")
    print("\nNOTE: surprise measures INTERNAL INCONSISTENCY, not fraud. "
          "Nothing here has seen a fraud label.")


if __name__ == "__main__":
    main()
