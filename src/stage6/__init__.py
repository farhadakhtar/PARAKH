"""Stage 6 - Action & Routing Layer.

Converts Stage 4 decisions and Stage 5 risk into human-actionable work.
Computes nothing. This stage is POLICY, not inference.

Public surface:

* :class:`~src.stage6.pipeline.ActionLayer` - the pipeline
* :func:`~src.stage6.pipeline.attach_actions` - integration with Corpus
* :data:`~src.stage6.pipeline.STAGE6_COLUMNS` - the output contract
* :data:`~src.stage6.routing.POLICY` - the routing table, in precedence order
"""

from src.stage6.explanation import (
    FIELD_ORDER,
    NOT_DEFINED,
    PAYLOAD_FIELDS,
    build_action_explanations,
    build_action_payload,
    build_action_payloads,
    explain_action,
    parse_action_explanation,
    parse_action_payload,
)
from src.stage6.pipeline import (
    STAGE6_COLUMNS,
    STAGE6_DETAIL_COLUMNS,
    STAGE6_SPEC_COLUMNS,
    ActionLayer,
    ActionResult,
    attach_actions,
)
from src.stage6.routing import (
    OPTIONAL_COLUMNS,
    Stage6ConfigError,
    Stage6ContractError,
    Stage6InvariantError,
    assert_gate_alignment,
    require_unique_index,
    validate_stage5_contract,
    POLICY,
    REQUIRED_COLUMNS,
    M1Correction,
    RoutingResult,
    Rule,
    Stage6InputError,
    apply_m1_correction,
    require_contract,
    route,
)

__all__ = [
    "FIELD_ORDER",
    "PAYLOAD_FIELDS",
    "STAGE6_SPEC_COLUMNS",
    "Stage6ConfigError",
    "Stage6ContractError",
    "Stage6InvariantError",
    "assert_gate_alignment",
    "build_action_payload",
    "build_action_payloads",
    "parse_action_payload",
    "require_unique_index",
    "validate_stage5_contract",
    "NOT_DEFINED",
    "OPTIONAL_COLUMNS",
    "POLICY",
    "REQUIRED_COLUMNS",
    "STAGE6_COLUMNS",
    "STAGE6_DETAIL_COLUMNS",
    "ActionLayer",
    "ActionResult",
    "M1Correction",
    "RoutingResult",
    "Rule",
    "Stage6InputError",
    "apply_m1_correction",
    "attach_actions",
    "build_action_explanations",
    "explain_action",
    "parse_action_explanation",
    "require_contract",
    "route",
]
