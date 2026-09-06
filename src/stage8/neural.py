"""Stage 8 - self-supervised consistency model (TensorFlow).

What this model claims, and what it does not
--------------------------------------------
It learns the internal grammar of a record: given every field but one,
predict the one held out. A record whose fields fail to predict each other is
*internally inconsistent* - the closest honest analogue to an illegal move,
because the ground truth is free, exact, and unlimited (it is the field's own
value), and no audit is required to obtain it.

It does **not** detect fraud. Nothing here has seen a fraud label, because
none exists. Two separate heads were considered and only one is built:

============================  =============  ==============================
head                          trainable?     validatable?
============================  =============  ==============================
"strange" (density/anomaly)   yes            **no** - needs fraud outcomes
"inconsistent" (this one)     yes            yes - ground truth is the field
============================  =============  ==============================

The anomaly head is deliberately omitted. Stage 4 already computes deviation
from peer norms (``z_cost``, ``z_spend``, ``z_duration``); a neural density
model would compute a better version of the same quantity, and with no labels
there is no way to demonstrate the better version is better *at finding
fraud* rather than merely at reconstructing the corpus. Swapping an
interpretable measure for an opaque one on no evidence is a poor trade for a
system whose thesis is auditability.

Why the synthetic ledger is a legitimate target HERE
-----------------------------------------------------
Everywhere else in PARAKH the generator's defect ledger is refused as a label
source, because the defects it injects are data-quality faults and calling
them fraud would be fabrication.

This model's claim is data-quality consistency. So for once the ledger is the
*matched* target: it records exactly which rows had a field corrupted, and
that is precisely what this model claims to detect. Validating a consistency
detector against injected inconsistencies is sound; validating a fraud
detector against them would not be. The distinction is the whole reason this
module is allowed to report a number at all.

The result therefore transfers to real data only as far as the claim does: it
says the model finds corrupted records, not that it finds corrupt ones.

Baseline discipline
-------------------
Every head is compared against predicting the training mode (categorical) or
median (numeric). A network that cannot beat the mode has learned nothing,
and reporting its accuracy without that comparison would hide it.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.logger import get_logger

LOGGER = get_logger(__name__)

#: Seeds fixed so a reported metric can be reproduced. TensorFlow is not
#: bitwise deterministic on every kernel even so; the run manifest records
#: that rather than claiming a guarantee the framework does not give.
DEFAULT_SEED = 20260906

#: Categorical fields the model reads and predicts.
CATEGORICAL_FIELDS: Tuple[str, ...] = (
    "state",
    "district",
    "implementing_agency",
    "status",
)

#: Numeric fields, log1p-scaled before use because sanctioned amounts span
#: several orders of magnitude and a raw-rupee loss would be dominated by the
#: largest works.
NUMERIC_FIELDS: Tuple[str, ...] = ("sanction_amount", "amount_spent")

#: Rare categories are collapsed rather than given their own embedding row.
#: A category seen twice cannot support a learned vector, and letting it try
#: produces a head that memorises identifiers.
MIN_CATEGORY_COUNT = 20
OTHER_TOKEN = "__OTHER__"
MISSING_TOKEN = "__MISSING__"


@dataclass
class EncodedCorpus:
    """Model-ready arrays plus everything needed to interpret them."""

    categorical: Dict[str, np.ndarray]
    numeric: np.ndarray
    numeric_present: np.ndarray
    vocabularies: Dict[str, List[str]]
    numeric_stats: Dict[str, Tuple[float, float]]
    index: pd.Index

    @property
    def n_records(self) -> int:
        return len(self.index)


def _set_seeds(seed: int) -> None:
    """Seed every generator the training path touches."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


def encode_corpus(
    frame: pd.DataFrame,
    *,
    vocabularies: Optional[Mapping[str, List[str]]] = None,
    numeric_stats: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> EncodedCorpus:
    """Turn a record frame into embedding indices and scaled numerics.

    Args:
        frame: Corpus with PARAKH's Stage 1 columns.
        vocabularies: Reuse a training vocabulary when encoding a holdout.
            Passing None on a holdout would let the split define its own
            categories and leak the holdout's distribution into its encoding.
        numeric_stats: Likewise for numeric centring and scaling.

    Returns:
        An :class:`EncodedCorpus`.
    """
    categorical: Dict[str, np.ndarray] = {}
    vocab_out: Dict[str, List[str]] = {}

    for name in CATEGORICAL_FIELDS:
        column = (
            frame[name].astype(str).str.strip().str.upper()
            if name in frame.columns
            else pd.Series([MISSING_TOKEN] * len(frame), index=frame.index)
        )
        column = column.replace({"": MISSING_TOKEN, "NAN": MISSING_TOKEN})

        if vocabularies is not None and name in vocabularies:
            vocab = list(vocabularies[name])
        else:
            counts = column.value_counts()
            kept = [str(v) for v in counts[counts >= MIN_CATEGORY_COUNT].index]
            vocab = [MISSING_TOKEN, OTHER_TOKEN] + [
                v for v in kept if v not in {MISSING_TOKEN, OTHER_TOKEN}
            ]

        lookup = {value: position for position, value in enumerate(vocab)}
        other = lookup[OTHER_TOKEN]
        categorical[name] = column.map(lambda v: lookup.get(v, other)).to_numpy(
            dtype=np.int32
        )
        vocab_out[name] = vocab

    numeric_columns: List[np.ndarray] = []
    present_columns: List[np.ndarray] = []
    stats_out: Dict[str, Tuple[float, float]] = {}

    for name in NUMERIC_FIELDS:
        raw = (
            pd.to_numeric(frame[name], errors="coerce")
            if name in frame.columns
            else pd.Series(np.nan, index=frame.index)
        )
        present = raw.notna().to_numpy(dtype=np.float32)
        scaled = np.log1p(raw.clip(lower=0).fillna(0.0).to_numpy(dtype=np.float64))

        if numeric_stats is not None and name in numeric_stats:
            mean, std = numeric_stats[name]
        else:
            observed = scaled[present.astype(bool)]
            mean = float(observed.mean()) if observed.size else 0.0
            std = float(observed.std()) or 1.0

        numeric_columns.append(((scaled - mean) / std).astype(np.float32) * present)
        present_columns.append(present)
        stats_out[name] = (mean, std)

    return EncodedCorpus(
        categorical=categorical,
        numeric=np.stack(numeric_columns, axis=1),
        numeric_present=np.stack(present_columns, axis=1),
        vocabularies=vocab_out,
        numeric_stats=stats_out,
        index=frame.index,
    )


def build_model(encoded: EncodedCorpus, *, width: int = 64, seed: int = DEFAULT_SEED):
    """A shared encoder with one prediction head per field.

    Small on purpose. The corpus is 20,000 rows over a handful of low-entropy
    columns; a large network would memorise it and the reconstruction accuracy
    would stop measuring consistency and start measuring capacity.

    Args:
        encoded: Training corpus, for vocabulary sizes.
        width: Shared representation size.
        seed: Weight initialisation seed.

    Returns:
        An uncompiled ``keras.Model`` with one output per maskable field.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    initializer = keras.initializers.GlorotUniform(seed=seed)
    inputs: Dict[str, Any] = {}
    parts: List[Any] = []

    for name in CATEGORICAL_FIELDS:
        size = len(encoded.vocabularies[name])
        # One extra row is the MASK token: the field being predicted is
        # replaced by it, so the head cannot read the answer off its input.
        tensor = keras.Input(shape=(1,), dtype="int32", name=f"in_{name}")
        inputs[f"in_{name}"] = tensor
        dimension = max(4, min(32, size // 2))
        embedded = layers.Embedding(
            size + 1, dimension, embeddings_initializer=initializer
        )(tensor)
        parts.append(layers.Flatten()(embedded))

    numeric_input = keras.Input(shape=(len(NUMERIC_FIELDS),), name="in_numeric")
    present_input = keras.Input(shape=(len(NUMERIC_FIELDS),), name="in_present")
    inputs["in_numeric"] = numeric_input
    inputs["in_present"] = present_input
    parts.extend([numeric_input, present_input])

    shared = layers.Concatenate()(parts)
    shared = layers.Dense(width, activation="relu", kernel_initializer=initializer)(
        shared
    )
    shared = layers.Dropout(0.1, seed=seed)(shared)
    shared = layers.Dense(width, activation="relu", kernel_initializer=initializer)(
        shared
    )

    outputs: Dict[str, Any] = {}
    for name in CATEGORICAL_FIELDS:
        outputs[f"out_{name}"] = layers.Dense(
            len(encoded.vocabularies[name]),
            activation="softmax",
            name=f"out_{name}",
            kernel_initializer=initializer,
        )(shared)
    outputs["out_numeric"] = layers.Dense(
        len(NUMERIC_FIELDS), name="out_numeric", kernel_initializer=initializer
    )(shared)

    return keras.Model(inputs=inputs, outputs=outputs)


def _mask_inputs(
    encoded: EncodedCorpus, rng: np.random.Generator
) -> Dict[str, np.ndarray]:
    """Replace one randomly chosen field per row with the MASK token.

    Masking exactly one field keeps the task well posed: the model always has
    the rest of the record to reason from, so a failure is a statement about
    consistency rather than about how much was hidden.
    """
    n = encoded.n_records
    n_fields = len(CATEGORICAL_FIELDS) + 1  # +1 for the numeric block
    choice = rng.integers(0, n_fields, size=n)

    inputs: Dict[str, np.ndarray] = {}
    for position, name in enumerate(CATEGORICAL_FIELDS):
        values = encoded.categorical[name].copy()
        values[choice == position] = len(encoded.vocabularies[name])  # MASK row
        inputs[f"in_{name}"] = values.reshape(-1, 1)

    numeric = encoded.numeric.copy()
    present = encoded.numeric_present.copy()
    numeric_masked = choice == len(CATEGORICAL_FIELDS)
    numeric[numeric_masked] = 0.0
    present[numeric_masked] = 0.0
    inputs["in_numeric"] = numeric
    inputs["in_present"] = present
    return inputs


def baseline_metrics(train: EncodedCorpus, test: EncodedCorpus) -> Dict[str, float]:
    """Mode/median baselines - the bar the network has to clear.

    Reported alongside every model number. Accuracy on a field where 90% of
    records share one value is 0.90 for a constant predictor, and quoting the
    model's 0.91 without that context would be meaningless.
    """
    metrics: Dict[str, float] = {}
    for name in CATEGORICAL_FIELDS:
        values, counts = np.unique(train.categorical[name], return_counts=True)
        mode = values[int(np.argmax(counts))]
        metrics[f"{name}_accuracy"] = float((test.categorical[name] == mode).mean())

    observed = train.numeric_present.astype(bool)
    for position, name in enumerate(NUMERIC_FIELDS):
        column = train.numeric[:, position][observed[:, position]]
        median = float(np.median(column)) if column.size else 0.0
        mask = test.numeric_present[:, position].astype(bool)
        if mask.any():
            error = np.abs(test.numeric[:, position][mask] - median)
            metrics[f"{name}_mae"] = float(error.mean())
    return metrics


@dataclass
class ConsistencyRun:
    """Everything a reviewer needs to judge or reproduce a training run."""

    metrics: Dict[str, Any]
    baseline: Dict[str, float]
    history: Dict[str, List[float]]
    manifest: Dict[str, Any]
    scores: Optional[pd.Series] = None


def train_consistency_model(
    frame: pd.DataFrame,
    *,
    epochs: int = 30,
    batch_size: int = 256,
    holdout_fraction: float = 0.2,
    seed: int = DEFAULT_SEED,
) -> ConsistencyRun:
    """Train the masked-field model and score every record's consistency.

    Args:
        frame: Corpus to train on.
        epochs: Upper bound. Early stopping decides the actual count - an
            epoch budget chosen for roundness is not a training protocol.
        batch_size: Minibatch size.
        holdout_fraction: Held out before any fitting, and never used for
            weight updates or early-stopping selection... it is used for the
            final reported metric only.
        seed: Governs split, masking, initialisation and shuffling.

    Returns:
        A :class:`ConsistencyRun`. ``scores`` is per-record surprise: the
        model's negative log-likelihood of the values actually observed.
        Higher means the record disagrees with itself more.
    """
    _set_seeds(seed)
    from tensorflow import keras

    rng = np.random.default_rng(seed)
    n = len(frame)
    if n < 100:
        raise ValueError(f"need at least 100 records to train, got {n}")

    shuffled = rng.permutation(n)
    n_holdout = int(n * holdout_fraction)
    holdout_rows = shuffled[:n_holdout]
    train_rows = shuffled[n_holdout:]

    train_frame = frame.iloc[train_rows]
    test_frame = frame.iloc[holdout_rows]

    train = encode_corpus(train_frame)
    test = encode_corpus(
        test_frame, vocabularies=train.vocabularies, numeric_stats=train.numeric_stats
    )

    model = build_model(train, seed=seed)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={
            **{
                f"out_{name}": "sparse_categorical_crossentropy"
                for name in CATEGORICAL_FIELDS
            },
            "out_numeric": "mse",
        },
        metrics={f"out_{name}": "accuracy" for name in CATEGORICAL_FIELDS},
    )

    targets = {
        **{f"out_{name}": train.categorical[name] for name in CATEGORICAL_FIELDS},
        "out_numeric": train.numeric,
    }
    test_targets = {
        **{f"out_{name}": test.categorical[name] for name in CATEGORICAL_FIELDS},
        "out_numeric": test.numeric,
    }

    history = model.fit(
        _mask_inputs(train, rng),
        targets,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
                verbose=0,
            )
        ],
    )

    evaluated = model.evaluate(
        _mask_inputs(test, rng), test_targets, verbose=0, return_dict=True
    )
    metrics = {
        key.replace("out_", "").replace("_accuracy", "_accuracy"): float(value)
        for key, value in evaluated.items()
    }

    predictions = model.predict(_mask_inputs(test, rng), verbose=0)
    surprise = np.zeros(test.n_records, dtype=np.float64)
    for name in CATEGORICAL_FIELDS:
        probabilities = predictions[f"out_{name}"]
        actual = test.categorical[name]
        chosen = probabilities[np.arange(len(actual)), actual]
        surprise += -np.log(np.clip(chosen, 1e-9, 1.0))

    scores = pd.Series(surprise, index=test_frame.index, name="consistency_surprise")

    epochs_run = len(history.history["loss"])
    manifest = {
        "framework": "tensorflow",
        "seed": seed,
        "n_total": int(n),
        "n_train": int(len(train_rows)),
        "n_holdout": int(len(holdout_rows)),
        "epochs_max": epochs,
        "epochs_run": epochs_run,
        "batch_size": batch_size,
        "early_stopped": epochs_run < epochs,
        "n_parameters": int(model.count_params()),
        "_determinism": (
            "Seeds are fixed for split, masking, initialisation and shuffling. "
            "TensorFlow is not bitwise deterministic on every kernel, so "
            "metrics may vary in the last decimal place across runs."
        ),
    }

    LOGGER.info(
        "Stage 8 consistency model: %d params, %d/%d epochs, holdout n=%d",
        model.count_params(),
        epochs_run,
        epochs,
        len(holdout_rows),
    )

    return ConsistencyRun(
        metrics=metrics,
        baseline=baseline_metrics(train, test),
        history={k: [float(x) for x in v] for k, v in history.history.items()},
        manifest=manifest,
        scores=scores,
    )


def evaluate_against_ledger(
    scores: pd.Series, defective_rows: Sequence[int]
) -> Dict[str, Any]:
    """Does surprise actually find the rows the generator corrupted?

    This is the one place PARAKH may use its synthetic defect ledger as a
    target. The ledger records injected *data-quality* faults, and internal
    inconsistency is exactly what this model claims to detect, so the target
    matches the claim. It would be invalid for a fraud model, and that
    distinction is the reason the check lives here and nowhere else.

    Args:
        scores: Per-record surprise, indexed by corpus position.
        defective_rows: Row positions the ledger marks as carrying an
            injected defect.

    Returns:
        AUC and enrichment, plus an explicit statement of what the number
        does and does not license.
    """
    from sklearn.metrics import roc_auc_score

    truth = scores.index.isin(set(defective_rows)).astype(int)
    if truth.sum() == 0 or truth.sum() == len(truth):
        return {
            "status": "NOT_EVALUABLE",
            "reason": f"holdout carries {int(truth.sum())} defective row(s)",
        }

    auc = float(roc_auc_score(truth, scores.to_numpy()))
    top_decile = scores.nlargest(max(1, len(scores) // 10)).index
    precision = float(np.isin(top_decile, list(defective_rows)).mean())
    base_rate = float(truth.mean())

    return {
        "status": "EVALUATED",
        "roc_auc": auc,
        "precision_at_10pct": precision,
        "base_rate": base_rate,
        "enrichment": precision / base_rate if base_rate else None,
        "n_holdout": int(len(scores)),
        "n_defective": int(truth.sum()),
        "_claim": (
            "Measures detection of INJECTED DATA-QUALITY DEFECTS on synthetic "
            "data. It is not a fraud metric, carries no implication for "
            "real-world fraud detection, and must never be reported as one."
        ),
    }
