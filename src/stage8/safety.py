"""Stage 8 - safety gates that do not need outcome labels.

Most of Stage 8 is blocked on labelled outcomes. These three checks are not,
and that makes them the most valuable things in the layer right now: they
interrogate whether PARAKH's score means what it claims, using only the score
and the corpus it was computed on.

Artifact invariance (:func:`artifact_invariance`)
    Asks whether ``risk_score`` is tracking conduct or paperwork. If a
    classifier can recover which FILE a record came from, or how many of its
    fields were blank, from the risk score alone, then risk is substantially a
    measure of reporting practice. A district that files complete returns
    would then score lower than one that files badly - not because it behaves
    better, but because it types better. This is the failure mode that turns a
    fraud system into a penalty on administrative capacity, and it is
    invisible in every accuracy metric.

    Both directions are measured, because they fail differently:
    ``risk -> artifact`` says the score leaks provenance; ``artifact -> risk``
    says provenance drives the score. The second is the more damaging.

Leakage (:func:`leakage_report`)
    A name check, not a statistic. A leak that has to be detected
    statistically has already been trained on. Any feature that is a target,
    or derived from one, is rejected by name before a model sees it.

Stability (:func:`ranking_stability`)
    An investigator works a queue. If dropping 10% of the corpus reshuffles
    the top of that queue, the ordering is an artefact of the sample rather
    than a property of the records, and "top 100 by risk" is not a defensible
    thing to hand anybody. Measured on the records the two runs share, since
    those are the only ones on which the two rankings both have an opinion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import cross_val_predict

from src.core.constants import (
    ARTIFACT_INVARIANCE_MAX_AUC,
    LEAKAGE_FORBIDDEN_FEATURES,
    STABILITY_MIN_RANK_CORRELATION,
    STABILITY_MIN_TOPK_OVERLAP,
    STAGE8_MIN_SUBGROUP_N,
)
from src.core.logger import get_logger

LOGGER = get_logger(__name__)

#: Groupings that are administrative facts about how a record was reported,
#: not facts about the conduct it describes. Risk must not be recoverable
#: from these.
ARTIFACT_CANDIDATES: Tuple[str, ...] = (
    "source_file",
    "file_format",
    "upload_batch",
    "missingness_bucket",
    "cost_scope",
)


@dataclass(frozen=True)
class GateResult:
    """One safety gate's verdict."""

    gate: str
    status: str
    detail: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate,
            "status": self.status,
            "detail": self.detail,
            **self.metrics,
        }


def _missingness_bucket(frame: pd.DataFrame) -> pd.Series:
    """How many fields a record left blank, coarsened into bands.

    A derived artifact rather than a supplied one: no file records "this row
    was badly filled in", but it is exactly the kind of administrative
    property risk must not be reading.
    """
    blanks = frame.isna().sum(axis=1)
    return pd.cut(
        blanks,
        bins=[-1, 0, 2, 5, np.inf],
        labels=["none", "1-2", "3-5", "6+"],
    ).astype(str)


def _multiclass_auc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """One-vs-rest macro AUC, or None when it is not defined.

    Returns None rather than 0.5 on a degenerate input. A gate that cannot be
    evaluated has not passed, and returning a plausible number here would
    quietly convert "we could not check" into "we checked and it was fine".
    """
    classes = np.unique(y_true)
    if len(classes) < 2:
        return None
    try:
        if len(classes) == 2:
            return float(roc_auc_score(y_true, scores[:, 1]))
        return float(
            roc_auc_score(y_true, scores, multi_class="ovr", average="macro")
        )
    except ValueError:
        return None


def artifact_invariance(
    frame: pd.DataFrame,
    *,
    risk_column: str = "risk_score",
    artifacts: Sequence[str] = ARTIFACT_CANDIDATES,
    max_auc: float = ARTIFACT_INVARIANCE_MAX_AUC,
    seed: int = 17,
) -> GateResult:
    """Test whether risk encodes administrative provenance.

    Args:
        frame: Scored corpus.
        risk_column: The score under test.
        artifacts: Candidate administrative groupings. Absent ones are skipped
            and reported as skipped, never silently passed.
        max_auc: Above this, risk reconstructs the artifact too well.
        seed: Fixed so the result is reproducible.

    Returns:
        A :class:`GateResult`. PASS only if every evaluable artifact came in
        under the bound; NOT_EVALUABLE when none could be tested.
    """
    if risk_column not in frame.columns:
        return GateResult(
            "G8_ARTIFACT_INVARIANCE",
            "NOT_EVALUABLE",
            f"{risk_column!r} is not present",
        )

    work = frame.copy()
    work["missingness_bucket"] = _missingness_bucket(frame)

    risk = pd.to_numeric(work[risk_column], errors="coerce")
    defined = risk.notna()
    if int(defined.sum()) < STAGE8_MIN_SUBGROUP_N:
        return GateResult(
            "G8_ARTIFACT_INVARIANCE",
            "NOT_EVALUABLE",
            f"only {int(defined.sum())} record(s) carry a defined risk score",
        )

    results: Dict[str, Any] = {}
    skipped: List[str] = []
    failures: List[str] = []

    for name in artifacts:
        if name not in work.columns:
            skipped.append(f"{name} (absent)")
            continue

        labels = work.loc[defined, name].astype(str)
        counts = labels.value_counts()
        # Classes too small to cross-validate are dropped rather than allowed
        # to produce an unstable AUC that would be read as a measurement.
        keep = counts[counts >= STAGE8_MIN_SUBGROUP_N].index
        mask = labels.isin(keep)
        if mask.sum() < STAGE8_MIN_SUBGROUP_N or len(keep) < 2:
            skipped.append(f"{name} ({len(keep)} usable class(es))")
            continue

        y = labels[mask].to_numpy()
        x = risk.loc[defined][mask].to_numpy().reshape(-1, 1)

        # risk -> artifact: can the score recover where the record came from?
        model = LogisticRegression(max_iter=1000, random_state=seed)
        proba = cross_val_predict(model, x, y, cv=3, method="predict_proba")
        auc = _multiclass_auc(y, proba)

        # artifact -> risk: does provenance drive the score? The more
        # damaging direction, so it is measured explicitly rather than
        # inferred from the first.
        dummies = pd.get_dummies(y, drop_first=True).to_numpy(dtype=float)
        reverse = LinearRegression()
        predicted = cross_val_predict(reverse, dummies, x.ravel(), cv=3)
        r2 = float(r2_score(x.ravel(), predicted))

        results[name] = {
            "n": int(mask.sum()),
            "n_classes": int(len(keep)),
            "risk_reconstructs_artifact_auc": auc,
            "artifact_explains_risk_r2": r2,
        }
        if auc is not None and auc > max_auc:
            failures.append(f"{name}: AUC {auc:.3f} > {max_auc}")

    if not results:
        return GateResult(
            "G8_ARTIFACT_INVARIANCE",
            "NOT_EVALUABLE",
            "no artifact grouping had two sufficiently large classes: "
            + "; ".join(skipped),
            {"skipped": skipped},
        )

    status = "FAIL" if failures else "PASS"
    detail = (
        "; ".join(failures)
        if failures
        else f"{len(results)} artifact(s) tested, all below AUC {max_auc}"
    )
    return GateResult(
        "G8_ARTIFACT_INVARIANCE",
        status,
        detail,
        {"artifacts": results, "skipped": skipped, "max_auc": max_auc},
    )


def leakage_report(
    feature_names: Sequence[str],
    *,
    forbidden: frozenset[str] = LEAKAGE_FORBIDDEN_FEATURES,
) -> GateResult:
    """Reject any feature that is a target or derived from one.

    Deliberately a name check. A statistical leak detector only fires after a
    model has already been trained on the leaked column, and by then the
    reported metric is worthless. Names are checked before a fit happens.

    Args:
        feature_names: Columns a model is about to consume.
        forbidden: Names that may never be features.

    Returns:
        A :class:`GateResult`; FAIL names every offending column.
    """
    offending = sorted({name for name in feature_names if name in forbidden})
    if offending:
        return GateResult(
            "G5_LEAKAGE",
            "FAIL",
            f"target-derived feature(s) present: {', '.join(offending)}",
            {"offending": offending, "n_features": len(feature_names)},
        )
    return GateResult(
        "G5_LEAKAGE",
        "PASS",
        f"{len(feature_names)} feature(s), none target-derived",
        {"n_features": len(feature_names), "checked_against": sorted(forbidden)},
    )


def ranking_stability(
    rankings: Mapping[str, pd.Series],
    *,
    top_k: int = 100,
    min_correlation: float = STABILITY_MIN_RANK_CORRELATION,
    min_overlap: float = STABILITY_MIN_TOPK_OVERLAP,
) -> GateResult:
    """Compare rankings produced under perturbation.

    Args:
        rankings: Named risk series, each indexed by ``record_id``. The first
            is the reference; every other is compared against it.
        top_k: Depth of the queue an investigator would actually work.
        min_correlation: Minimum Spearman correlation on shared records.
        min_overlap: Minimum Jaccard-style overlap of the two top-K sets.

    Returns:
        A :class:`GateResult` carrying per-perturbation numbers.
    """
    names = list(rankings)
    if len(names) < 2:
        return GateResult(
            "G9_STABILITY",
            "NOT_EVALUABLE",
            "need at least two rankings to compare",
        )

    reference = rankings[names[0]].dropna()
    comparisons: Dict[str, Any] = {}
    failures: List[str] = []

    for name in names[1:]:
        other = rankings[name].dropna()
        shared = reference.index.intersection(other.index)
        if len(shared) < STAGE8_MIN_SUBGROUP_N:
            comparisons[name] = {"n_shared": int(len(shared)), "status": "TOO_SMALL"}
            continue

        left = reference.loc[shared]
        right = other.loc[shared]
        correlation = float(stats.spearmanr(left, right).statistic)

        depth = min(top_k, len(shared))
        top_left = set(left.nlargest(depth).index)
        top_right = set(right.nlargest(depth).index)
        overlap = len(top_left & top_right) / depth

        comparisons[name] = {
            "n_shared": int(len(shared)),
            "spearman": correlation,
            f"top{depth}_overlap": overlap,
        }
        if correlation < min_correlation:
            failures.append(f"{name}: rho {correlation:.3f} < {min_correlation}")
        if overlap < min_overlap:
            failures.append(f"{name}: top-{depth} overlap {overlap:.3f} < {min_overlap}")

    if not any("spearman" in value for value in comparisons.values()):
        return GateResult(
            "G9_STABILITY",
            "NOT_EVALUABLE",
            "no perturbation shared enough records with the reference",
            {"comparisons": comparisons},
        )

    status = "FAIL" if failures else "PASS"
    detail = (
        "; ".join(failures)
        if failures
        else f"{len(comparisons)} perturbation(s) within tolerance"
    )
    return GateResult(
        "G9_STABILITY",
        status,
        detail,
        {
            "comparisons": comparisons,
            "min_correlation": min_correlation,
            "min_overlap": min_overlap,
        },
    )
