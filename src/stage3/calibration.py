"""Calibration instrumentation: make every parameter observable and testable.

This module **tunes nothing**. Every default stays exactly where it is. What it
adds is the ability to see what those defaults are doing:

* the full parameter set actually used in a run, saved as a reloadable snapshot
* the distributions those parameters produce - cluster sizes, cell sizes,
  norm coverage, deviation percentiles

Why that matters more than it sounds
------------------------------------
A threshold like ``PEER_CELL_MIN_SIZE = 15`` is a judgement, not a measurement.
Whether it is right depends on a distribution nobody has looked at: if the
median cell holds 200 records the floor is irrelevant, and if it holds 12 the
floor is silently discarding the entire corpus. Until the distribution is on
paper, calibration cannot even begin.

The deviation percentiles are reported for the same reason, and carry a warning
attached to them: they are **descriptive, not thresholds**. Stage 4 must not
lift p99 and call it a flag boundary - that would fit the cut to whatever
happens to be in this corpus, on a corpus whose own defect rates were chosen by
a synthetic generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

import json
import numpy as np
import pandas as pd

from src.core.constants import (
    ARTIFACT_DIR,
    CLUSTER_MIN_RECORDS,
    COST_STRATA_BINS,
    DEVIATION_PERCENTILES,
    DUPLICATE_SIMILARITY_THRESHOLD,
    DUPLICATE_TAU_DAYS,
    HDBSCAN_MIN_CLUSTER_SIZE,
    MAD_SCALE,
    PEER_CELL_MIN_SIZE,
    PEER_STAT_MIN_CONFIDENCE,
    PEER_STAT_MIN_REFERENCE,
    STAGE3_CONFIG_SNAPSHOT_FILE,
    STAGE3_SEED,
    STAGE3_VERSION,
    SVD_COMPONENTS,
)
from src.core.logger import get_logger
from src.utils.helpers import ensure_dir, write_json

if TYPE_CHECKING:  # pragma: no cover
    from src.stage3.pipeline import SemanticResult

LOGGER = get_logger(__name__)

#: Every Stage 3 parameter that can move a result, with what it governs and
#: where it came from. ``source`` is deliberately honest: "default" means
#: nobody estimated it.
CALIBRATION_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "PEER_STAT_MIN_CONFIDENCE": {
        "default": PEER_STAT_MIN_CONFIDENCE,
        "governs": "whether a record may shape the peer norm it is judged against",
        "source": "default",
        "risk_if_wrong": "too low admits garbage into the norm; too high starves "
        "cells of reference members and leaves deviations undefined",
    },
    "PEER_CELL_MIN_SIZE": {
        "default": PEER_CELL_MIN_SIZE,
        "governs": "when a peer cell is large enough to be trusted",
        "source": "Stage3.md sec.8.1",
        "risk_if_wrong": "too low yields norms from a handful of records; too "
        "high discards usable cells and shrinks coverage",
    },
    "PEER_STAT_MIN_REFERENCE": {
        "default": PEER_STAT_MIN_REFERENCE,
        "governs": "high-confidence members needed before a norm is emitted",
        "source": "default",
        "risk_if_wrong": "too low yields an unstable median; too high leaves "
        "cells without norms",
    },
    "DUPLICATE_SIMILARITY_THRESHOLD": {
        "default": DUPLICATE_SIMILARITY_THRESHOLD,
        "governs": "cosine above which two works are near-duplicates",
        "source": "default",
        "risk_if_wrong": "too low floods the queue with same-type works; too "
        "high catches only byte-identical copies",
    },
    "DUPLICATE_TAU_DAYS": {
        "default": DUPLICATE_TAU_DAYS,
        "governs": "temporal decay of the duplicate score",
        "source": "Stage3.md sec.9.2",
        "risk_if_wrong": "sets how far apart two claims can be and still count "
        "as the same work",
    },
    "HDBSCAN_MIN_CLUSTER_SIZE": {
        "default": HDBSCAN_MIN_CLUSTER_SIZE,
        "governs": "distinct texts needed to form a work-type cluster",
        "source": "swept against generator ground truth",
        "risk_if_wrong": "too high merges distinct work types; too low "
        "fragments one type across spellings",
    },
    "CLUSTER_MIN_RECORDS": {
        "default": CLUSTER_MIN_RECORDS,
        "governs": "records below which a cluster is merged into its neighbour",
        "source": "Stage3.md sec.6.4",
        "risk_if_wrong": "too high merges genuine small work types away",
    },
    "SVD_COMPONENTS": {
        "default": SVD_COMPONENTS,
        "governs": "width of the clustering projection",
        "source": "swept against generator ground truth",
        "risk_if_wrong": "too high collapses HDBSCAN into noise; too low loses "
        "the distinction between related work types",
    },
    "COST_STRATA_BINS": {
        "default": COST_STRATA_BINS,
        "governs": "number of cost bands",
        "source": "Stage3.md sec.7.3",
        "risk_if_wrong": "too many thins every cell; too few compares works of "
        "very different scale",
    },
    "MAD_SCALE": {
        "default": MAD_SCALE,
        "governs": "consistency constant making MAD comparable to sigma",
        "source": "statistical constant, not a calibration target",
        "risk_if_wrong": "n/a - fixed by the normal distribution",
    },
    "STAGE3_SEED": {
        "default": STAGE3_SEED,
        "governs": "the SVD solver; nothing else in Stage 3 is random",
        "source": "fixed",
        "risk_if_wrong": "n/a - determinism, not accuracy",
    },
}


def _percentiles(values: np.ndarray, points: Sequence[int]) -> Dict[str, Optional[float]]:
    """Percentiles of the finite entries, or ``None`` when there are none."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {f"p{point}": None for point in points}
    return {
        f"p{point}": round(float(np.percentile(finite, point)), 6) for point in points
    }


def _distribution(series: pd.Series) -> Dict[str, Optional[float]]:
    """Summary of a size distribution, safe on empty input."""
    if not len(series):
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    values = series.to_numpy(dtype="float64")
    return {
        "n": int(values.size),
        "min": round(float(values.min()), 4),
        "p25": round(float(np.percentile(values, 25)), 4),
        "median": round(float(np.median(values)), 4),
        "p75": round(float(np.percentile(values, 75)), 4),
        "max": round(float(values.max()), 4),
        "mean": round(float(values.mean()), 4),
    }


@dataclass(frozen=True)
class ConfigSnapshot:
    """The exact parameter set a run used, reloadable for replay."""

    parameters: Mapping[str, Any]
    stage3_version: str = STAGE3_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "stage3_version": self.stage3_version,
            "parameters": dict(sorted(self.parameters.items())),
        }

    def save(self, artifact_dir: Path = ARTIFACT_DIR) -> Path:
        """Write the snapshot to ``artifacts/stage3_config.json``."""
        directory = ensure_dir(Path(artifact_dir))
        return write_json(self.to_dict(), directory / STAGE3_CONFIG_SNAPSHOT_FILE)

    @classmethod
    def load(cls, artifact_dir: Path = ARTIFACT_DIR) -> Optional["ConfigSnapshot"]:
        """Reload a saved snapshot, or ``None`` if absent."""
        path = Path(artifact_dir) / STAGE3_CONFIG_SNAPSHOT_FILE
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            parameters=dict(payload.get("parameters", {})),
            stage3_version=str(payload.get("stage3_version", STAGE3_VERSION)),
        )

    @classmethod
    def from_config(cls, config: Any) -> "ConfigSnapshot":
        """Capture every field of a :class:`SemanticConfig`."""
        parameters = {
            name: getattr(config, name)
            for name in (
                "seed",
                "n_components",
                "min_cluster_size",
                "cluster_min_records",
                "cost_bins",
                "peer_cell_min_size",
                "min_confidence",
                "min_reference",
                "duplicate_threshold",
                "duplicate_tau_days",
            )
            if hasattr(config, name)
        }
        return cls(parameters=parameters)


def build_calibration_report(result: "SemanticResult") -> Dict[str, Any]:
    """Assemble the calibration report for one Stage 3 run.

    Args:
        result: A completed :class:`~src.stage3.pipeline.SemanticResult`.

    Returns:
        A JSON-serialisable report: the parameters used, the distributions they
        produced, and coverage per feature. Contains no wall-clock value, so
        two runs over one corpus write identical bytes.
    """
    frame = result.frame
    config = result.config

    used = ConfigSnapshot.from_config(config).parameters
    parameters: Dict[str, Any] = {}
    for name, meta in CALIBRATION_PARAMETERS.items():
        entry = dict(meta)
        entry["value_used"] = _resolve_used(name, used, meta["default"])
        entry["is_default"] = entry["value_used"] == meta["default"]
        parameters[name] = entry

    cluster_sizes = (
        frame.loc[~frame["cluster_is_noise"], "cluster_id"].value_counts()
        if len(frame)
        else pd.Series(dtype="int64")
    )
    cell_sizes = (
        frame["peer_cell_id"].value_counts() if len(frame) else pd.Series(dtype="int64")
    )
    stable_cells = (
        frame.loc[frame["peer_cell_stable"], "peer_cell_id"].nunique()
        if len(frame)
        else 0
    )

    coverage: Dict[str, Any] = {}
    for name in result.deviations.names:
        reasons = frame[f"{name}_reason"].value_counts().to_dict()
        defined = frame[name].notna()
        coverage[name] = {
            "defined_pct": round(100.0 * float(defined.mean()), 4) if len(frame) else 0.0,
            "undefined_reasons": {str(k): int(v) for k, v in sorted(reasons.items())},
            "abs_percentiles": _percentiles(
                frame.loc[defined, name].abs().to_numpy(dtype="float64"),
                DEVIATION_PERCENTILES,
            ),
            "signed_percentiles": _percentiles(
                frame.loc[defined, name].to_numpy(dtype="float64"),
                DEVIATION_PERCENTILES,
            ),
        }

    stats = result.statistics.cell_stats
    norm_coverage: Dict[str, Any] = {}
    for feature in ("log_cost", "spend_ratio", "duration_days"):
        column = f"{feature}_median"
        if column not in stats.columns:
            continue
        has_norm = stats[column].notna()
        norm_coverage[feature] = {
            "cells_with_norm": int(has_norm.sum()),
            "cells_total": int(len(stats)),
            "pct": round(100.0 * float(has_norm.mean()), 4) if len(stats) else 0.0,
            "zero_dispersion_cells": int(
                (stats[column].notna() & stats[f"{feature}_mad"].isna()).sum()
            ),
        }

    return {
        "stage3_version": STAGE3_VERSION,
        "n_records": int(len(frame)),
        "_note": (
            "Instrumentation only. No value here has been tuned, and the "
            "deviation percentiles are DESCRIPTIVE - they are not thresholds "
            "and must not be adopted as flag boundaries without calibration "
            "against real outcomes."
        ),
        "parameters": parameters,
        "distributions": {
            "cluster_size": _distribution(cluster_sizes),
            "peer_cell_size": _distribution(cell_sizes),
            "n_clusters": int(cluster_sizes.size),
            "n_cells": int(cell_sizes.size),
            "n_stable_cells": int(stable_cells),
            "stable_cell_pct": round(
                100.0 * stable_cells / cell_sizes.size, 4
            )
            if cell_sizes.size
            else 0.0,
            "stable_record_pct": round(
                100.0 * float(frame["peer_cell_stable"].mean()), 4
            )
            if len(frame)
            else 0.0,
            "noise_record_pct": round(
                100.0 * float(frame["cluster_is_noise"].mean()), 4
            )
            if len(frame)
            else 0.0,
            "reference_record_pct": round(
                100.0 * float(frame["peer_reference"].mean()), 4
            )
            if len(frame)
            else 0.0,
        },
        "norm_coverage": norm_coverage,
        "deviation_coverage": coverage,
        "config_snapshot": ConfigSnapshot.from_config(config).to_dict(),
    }


def _resolve_used(name: str, used: Mapping[str, Any], default: Any) -> Any:
    """Map a constant name onto the config field that carries it."""
    aliases = {
        "PEER_STAT_MIN_CONFIDENCE": "min_confidence",
        "PEER_CELL_MIN_SIZE": "peer_cell_min_size",
        "PEER_STAT_MIN_REFERENCE": "min_reference",
        "DUPLICATE_SIMILARITY_THRESHOLD": "duplicate_threshold",
        "DUPLICATE_TAU_DAYS": "duplicate_tau_days",
        "HDBSCAN_MIN_CLUSTER_SIZE": "min_cluster_size",
        "CLUSTER_MIN_RECORDS": "cluster_min_records",
        "SVD_COMPONENTS": "n_components",
        "COST_STRATA_BINS": "cost_bins",
        "STAGE3_SEED": "seed",
    }
    field_name = aliases.get(name)
    if field_name is None or field_name not in used:
        return default
    return used[field_name]
