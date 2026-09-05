"""Stage 7 - Decision Consumption Layer.

Turns Stage 6 decisions into human work, a machine contract, an audit trail
and a feedback notebook. Read-only over Stages 1-6: it attaches nothing to the
corpus and recomputes nothing.

``explanation_payload`` is the source of truth. The human ``explanation`` is
display only and is never parsed.

Public surface:

* :class:`~src.stage7.pipeline.ConsumptionLayer` - the pipeline
* :func:`~src.stage7.pipeline.consume` - run it without mutating anything
"""

from src.stage7.annotations import (
    ANNOTATION_COLUMNS,
    build_annotations,
    build_stage7_explanation,
    build_system_metadata,
    build_transparency_metrics,
    build_work_level_summary,
)
from src.stage7.api import API_FIELDS, build_api_response, build_api_responses, serialise
from src.stage7.audit import (
    AUDIT_FIELDS,
    build_audit_entry,
    build_audit_log,
    compute_input_hash,
    read_audit_log,
    write_audit_log,
)
from src.stage7.feedback import (
    FEEDBACK_FIELDS,
    append_feedback,
    build_feedback_entry,
    read_feedback_log,
    summarise_feedback,
    write_feedback_log,
)
from src.stage7.interface import (
    DECISION_CARD_FIELDS,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    QueueItem,
    Stage7ContractError,
    build_decision_card,
    build_queues,
    decode_payloads,
    require_contract,
)
from src.stage7.policy import (
    Stage7PolicyError,
    escalation_policy_report,
    validate_escalation_policy,
)
from src.stage7.pipeline import (
    ConsumptionLayer,
    ConsumptionResult,
    Stage7InvariantError,
    consume,
)

__all__ = [
    "ANNOTATION_COLUMNS",
    "API_FIELDS",
    "AUDIT_FIELDS",
    "DECISION_CARD_FIELDS",
    "FEEDBACK_FIELDS",
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "ConsumptionLayer",
    "ConsumptionResult",
    "QueueItem",
    "Stage7ContractError",
    "Stage7InvariantError",
    "Stage7PolicyError",
    "append_feedback",
    "build_annotations",
    "build_api_response",
    "build_api_responses",
    "build_audit_entry",
    "build_audit_log",
    "build_decision_card",
    "build_stage7_explanation",
    "build_system_metadata",
    "build_transparency_metrics",
    "build_work_level_summary",
    "build_feedback_entry",
    "build_queues",
    "compute_input_hash",
    "consume",
    "escalation_policy_report",
    "validate_escalation_policy",
    "decode_payloads",
    "read_audit_log",
    "read_feedback_log",
    "require_contract",
    "serialise",
    "summarise_feedback",
    "write_audit_log",
    "write_feedback_log",
]
