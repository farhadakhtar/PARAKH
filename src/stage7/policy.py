"""The refusal Stage 7 makes before it hands anything to a person.

One invariant carries this system's safety: **an escalated record must have a
risk score**. It holds because Stage 4 cannot escalate below its confidence
gate and Stage 5 cannot score below the same number - so ``INVESTIGATE``
implies ``risk_defined``, but only while the two gates are equal.

That equality is not enforced anywhere upstream. A configured divergence,
``RiskConfig(min_confidence=0.80)``, breaks it without changing a single
constant, and was measured to produce **73 escalated records carrying no risk
score**. Those records would reach an investigator as urgent leads with nothing
behind them.

Stage 7 is the right layer to refuse. Stage 6 must remain able to route
contradictory input - its own policy tests construct exactly that combination
on purpose, and a router that cannot be tested on impossible input is a router
whose holes go unnoticed. But nothing may be handed to a **human** from that
state, so the refusal lives at the consumption boundary and fails loudly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.constants import (
    CONFIDENCE_GATE_THRESHOLD,
    ESCALATION_POLICY_VIOLATION_HINT,
    MIN_CONFIDENCE_FOR_RISK,
    STAGE7_VERSION,
)
from src.core.logger import get_logger

LOGGER = get_logger(__name__)

#: How many offending records to name in the error. Enough to diagnose, few
#: enough to read.
_SAMPLE_SIZE = 5


class Stage7PolicyError(RuntimeError):
    """Raised when the input violates a policy Stage 7 will not consume.

    Distinct from ``Stage7ContractError``: the contract is about shape, this
    is about meaning. The columns are all present and well-formed; what they
    say together is unsafe to act on.
    """


def validate_escalation_policy(
    frame: pd.DataFrame,
    stage4_gate: float = CONFIDENCE_GATE_THRESHOLD,
    stage5_gate: float = MIN_CONFIDENCE_FOR_RISK,
) -> None:
    """Refuse to present an escalation that carries no risk score.

    Args:
        frame: Corpus frame with Stage 4 and Stage 5 output.
        stage4_gate: The Stage 4 escalation gate, reported in the error.
        stage5_gate: The Stage 5 scoring gate, reported in the error.

    Raises:
        Stage7PolicyError: If any record is ``INVESTIGATE`` with
            ``risk_defined == False``. The message carries the count, sample
            indices, and both conflicting gate values, because a policy
            failure a caller cannot locate is a policy failure they will
            disable.
    """
    if "decision_class" not in frame.columns or "risk_defined" not in frame.columns:
        # Nothing to check. The contract layer reports missing columns; this
        # function must not duplicate - or contradict - that diagnosis.
        return

    investigate = (
        frame["decision_class"].astype("object") == "INVESTIGATE"
    ).to_numpy(dtype=bool)
    undefined = ~frame["risk_defined"].fillna(False).to_numpy(dtype=bool)
    offending = investigate & undefined

    count = int(offending.sum())
    if not count:
        return

    sample = frame.index[offending][:_SAMPLE_SIZE].tolist()
    detail = [
        {
            "record_id": label,
            "decision_class": str(frame.at[label, "decision_class"]),
            "risk_defined": bool(frame.at[label, "risk_defined"]),
            "risk_flag": str(frame.at[label, "risk_flag"])
            if "risk_flag" in frame.columns
            else None,
            "confidence": float(frame.at[label, "confidence"])
            if "confidence" in frame.columns
            else None,
        }
        for label in sample
    ]

    # The gate values reported here are the CONSTANTS. A per-run override
    # (RiskConfig(min_confidence=...)) is invisible to them, so the two can
    # read as aligned while the run that produced this frame used different
    # numbers. Saying "gates aligned: True" beside 73 violations would send
    # a reader looking in the wrong place, so the constants are labelled as
    # constants and the override is named as the thing to check.
    aligned = stage4_gate == stage5_gate
    alignment_note = (
        "the CONSTANTS agree, so the divergence is a per-run override - "
        "check the RiskConfig(min_confidence=...) this pipeline was built with"
        if aligned
        else "the constants themselves disagree"
    )
    raise Stage7PolicyError(
        f"{count} record(s) are INVESTIGATE with no risk score. An escalation "
        f"with nothing behind it must never reach a reviewer.\n"
        f"  Stage 4 escalation gate (constant) : {stage4_gate}\n"
        f"  Stage 5 scoring gate (constant)    : {stage5_gate}\n"
        f"  constants aligned                  : {aligned}\n"
        f"  -> {alignment_note}\n"
        f"  sample record ids                  : {sample}\n"
        f"  sample detail                      : {detail}\n"
        f"{ESCALATION_POLICY_VIOLATION_HINT}"
    )


def escalation_policy_report(
    frame: pd.DataFrame,
    stage4_gate: float = CONFIDENCE_GATE_THRESHOLD,
    stage5_gate: float = MIN_CONFIDENCE_FOR_RISK,
) -> Dict[str, Any]:
    """The same check, as data rather than an exception.

    Used by the report so a passing run states the invariant it upholds. A
    guarantee that is only visible when it fails is a guarantee nobody knows
    they have.
    """
    if "decision_class" not in frame.columns or "risk_defined" not in frame.columns:
        return {"checked": False, "_note": "Stage 4/5 columns absent"}

    investigate = (
        frame["decision_class"].astype("object") == "INVESTIGATE"
    ).to_numpy(dtype=bool)
    undefined = ~frame["risk_defined"].fillna(False).to_numpy(dtype=bool)
    offending = int((investigate & undefined).sum())
    return {
        "checked": True,
        "stage7_version": STAGE7_VERSION,
        "n_investigate": int(investigate.sum()),
        "n_investigate_without_risk": offending,
        "stage4_gate": float(stage4_gate),
        "stage5_gate": float(stage5_gate),
        "gates_aligned": bool(stage4_gate == stage5_gate),
        "_invariant": (
            "An escalated record must carry a risk score. Enforced here, not "
            "assumed: it holds only while the Stage 4 and Stage 5 confidence "
            "gates are equal, and a configured divergence breaks it without "
            "changing any constant."
        ),
    }
