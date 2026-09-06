"""Stage 8 - statutory compliance checking.

Why this layer exists
---------------------
PARAKH's risk thresholds are judgement constants. ``R > 0.7`` is somebody's
taste, and turning it into a defensible number needs outcome labels that do
not exist. That argument is correct, and it is why Stages 4-7 all carry an
UNCALIBRATED banner.

But it does not apply to every number a procurement system cares about. Some
are fixed by statute. A tender limit is not an estimate anybody has to
calibrate - it is a rule with a number in it, published, dated, and citable.
For those, the right provenance is a **citation**, not a fit, and it is
available today without a single labelled outcome.

What this layer does NOT do
---------------------------
It checks whether a written rule was followed. That is not the same as
detecting fraud, and conflating the two would be the most damaging thing this
module could do:

* **Irregular but honest** - an emergency repair executed without tender and
  justified afterwards breaks the rule and defrauds nobody. Common.
* **Fraudulent but compliant** - a flawless tender whose losing bidders are
  shells owned by the winner passes every check here. That is the
  sophisticated kind, and this layer scores it clean.

So a violation is evidence of irregularity, never a fraud allegation. The
distinction is the same one PARAKH already draws between confidence and risk,
and it is preserved deliberately.

The four-valued outcome
-----------------------
A rule returns COMPLIANT, VIOLATION, NOT_APPLICABLE or UNDETERMINABLE, and the
fourth is the one that matters. A record missing the fields a rule needs is
UNDETERMINABLE - never COMPLIANT. If missing data returned COMPLIANT, the
least documented records would score cleanest, which is precisely backwards.
This is the same "undefined is not zero" rule the rest of the system runs on.

Citation verification, and what it caught
-----------------------------------------
Every rule initially shipped ``citation_verified=False``, gated to
``PENDING_CITATION_VERIFICATION`` - logic testable, no violation emitted.

That gate was not ceremony. On 2026-09-06 the GFR 2017 PDF was obtained from
the Department of Expenditure and three of the five citations turned out to
be **wrong**:

===========================  ==========================================
stated from memory           what GFR 2017 actually says
===========================  ==========================================
Rule 155 = Advertised        Rule 155 = Purchase by purchase committee,
Tender Enquiry               Rs 25,000 to Rs 2,50,000. Advertised
                             Tender Enquiry is **Rule 161**.
Rule 154 = Limited Tender    Rule 154 = Purchase without quotations up
Enquiry                      to Rs 25,000. Limited Tender Enquiry is
                             **Rule 162**.
Rule 230 = Utilisation       Rule 230 is grants-in-aid principles. The
Certificates                 UC timing provision is **Rule 238**, and it
                             runs from the close of the financial year,
                             not from completion.
===========================  ==========================================

Had these fired unverified, five rules would have flagged real records
against a legal basis that does not exist. That is precisely the failure the
gate was built to prevent, and it is the reason the same discipline is worth
keeping for the typology library.

One error was substantive rather than clerical: Rule 162 says the number of
supplier firms "should be **more than three**", so the minimum is four. The
original rule tested ``>= 3``, which would have passed every three-firm
enquiry the rule is meant to catch.

Current state: three rules verified against the primary source and live; two
still gated, each with a ``verification_note`` naming exactly what remains to
be checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from src.core.logger import get_logger

LOGGER = get_logger(__name__)

#: The four verdicts, plus the citation gate.
COMPLIANCE_OUTCOMES: Tuple[str, ...] = (
    "COMPLIANT",
    "VIOLATION",
    "NOT_APPLICABLE",
    "UNDETERMINABLE",
    "PENDING_CITATION_VERIFICATION",
)

#: A field present with this value means "we looked, and it is not there".
#: Distinct from the field being absent, which means "we do not know". The
#: first is a violation; the second is UNDETERMINABLE.
NOT_RECORDED = "NOT_RECORDED"

_NULLISH = {"", "nan", "nat", "none", "null", "<na>", "unknown", "-"}


def _missing(value: Any) -> bool:
    """True when a value asserts nothing.

    ``NOT_RECORDED`` is deliberately NOT missing: it is a positive statement
    that a required justification is absent from the file, which is a finding
    rather than an unknown.
    """
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    text = str(value).strip().lower()
    return text in _NULLISH


def _as_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(result) else result


def _as_date(value: Any) -> Optional[pd.Timestamp]:
    stamp = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(stamp) else stamp


@dataclass(frozen=True)
class ComplianceOutcome:
    """One rule's verdict on one record."""

    rule_id: str
    status: str
    detail: str
    citation: str


@dataclass(frozen=True)
class ComplianceRule:
    """A statutory rule, its source, and the check that implements it.

    Attributes:
        rule_id: Stable internal identifier.
        citation: The clause, as a reviewer would look it up.
        source: The publication the clause lives in.
        source_url: Where that publication is published.
        statement: The rule in plain words.
        required_fields: Fields the check reads. Absent any of them, the
            verdict is UNDETERMINABLE - checked before applicability, so a
            rule can never appear inapplicable merely because data is missing.
        check: Returns ``(status, detail)`` given a complete-enough record.
        threshold: The statutory number, when the rule has one.
        threshold_units: What the threshold is measured in.
        threshold_provenance: Must be ``"STATUTORY"`` when a threshold exists.
            A judgement constant does not belong in a layer whose authority
            comes from citation.
        citation_verified: Whether the clause has been confirmed against the
            primary source. False gates the rule to
            PENDING_CITATION_VERIFICATION.
        verification_note: Exactly what a reviewer must confirm.
        compliant_example: Field overrides producing COMPLIANT.
        violating_example: Field overrides producing a violation.
    """

    rule_id: str
    citation: str
    source: str
    source_url: str
    statement: str
    required_fields: Tuple[str, ...]
    check: Callable[[Mapping[str, Any]], Tuple[str, str]]
    compliant_example: Mapping[str, Any]
    violating_example: Mapping[str, Any]
    verification_note: str
    threshold: Optional[float] = None
    threshold_units: str = ""
    threshold_provenance: str = "STATUTORY"
    citation_verified: bool = False


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------

#: Value above which open advertised tendering is required, in rupees.
OPEN_TENDER_THRESHOLD_INR = 2_500_000.0

#: Minimum supplier firms for a limited tender enquiry. GFR 2017 Rule 162
#: says "more than three", so the minimum is FOUR. Stated as 4 rather than as
#: "3 plus one" because an off-by-one in a legal threshold is a rule that
#: silently permits what it was written to forbid.
LIMITED_TENDER_MIN_BIDDERS = 4

#: Days within which a utilisation certificate is due after completion.
UC_DUE_DAYS = 365


def _check_open_tender(record: Mapping[str, Any]) -> Tuple[str, str]:
    amount = _as_float(record["sanctioned_amount"])
    method = str(record["procurement_method"]).strip().upper()
    if amount is None:
        return "UNDETERMINABLE", "sanctioned_amount is not numeric"
    if amount <= OPEN_TENDER_THRESHOLD_INR:
        return (
            "NOT_APPLICABLE",
            f"value {amount:,.0f} is at or below the "
            f"{OPEN_TENDER_THRESHOLD_INR:,.0f} open-tender threshold",
        )
    if method == "OPEN_TENDER":
        return "COMPLIANT", f"value {amount:,.0f} procured by open tender"
    return (
        "VIOLATION",
        f"value {amount:,.0f} exceeds {OPEN_TENDER_THRESHOLD_INR:,.0f} but was "
        f"procured by {method}",
    )


def _check_limited_tender_bidders(record: Mapping[str, Any]) -> Tuple[str, str]:
    method = str(record["procurement_method"]).strip().upper()
    bidders = _as_float(record["n_bidders"])
    if bidders is None:
        return "UNDETERMINABLE", "n_bidders is not numeric"
    if method != "LIMITED_TENDER":
        return "NOT_APPLICABLE", f"procurement method is {method}"
    if bidders >= LIMITED_TENDER_MIN_BIDDERS:
        return "COMPLIANT", f"{int(bidders)} supplier firm(s)"
    return (
        "VIOLATION",
        f"limited tender involved {int(bidders)} supplier firm(s); Rule 162 "
        f"requires more than three, i.e. {LIMITED_TENDER_MIN_BIDDERS}",
    )


def _check_uc_timeliness(record: Mapping[str, Any]) -> Tuple[str, str]:
    completion = _as_date(record["date_completion"])
    submitted = _as_date(record["uc_submitted_date"])
    if completion is None or submitted is None:
        return "UNDETERMINABLE", "completion or UC submission date unparseable"
    elapsed = (submitted - completion).days
    if elapsed <= UC_DUE_DAYS:
        return "COMPLIANT", f"UC submitted {elapsed} day(s) after completion"
    return (
        "VIOLATION",
        f"UC submitted {elapsed} day(s) after completion; {UC_DUE_DAYS} allowed",
    )


def _check_single_source_justification(record: Mapping[str, Any]) -> Tuple[str, str]:
    method = str(record["procurement_method"]).strip().upper()
    justification = record["single_source_justification"]
    if method != "SINGLE_SOURCE":
        return "NOT_APPLICABLE", f"procurement method is {method}"
    if str(justification).strip().upper() == NOT_RECORDED:
        return "VIOLATION", "single-source award carries no recorded justification"
    return "COMPLIANT", "justification recorded"


def _check_work_before_approval(record: Mapping[str, Any]) -> Tuple[str, str]:
    approval = _as_date(record["date_approval"])
    start = _as_date(record["date_start"])
    if approval is None or start is None:
        return "UNDETERMINABLE", "approval or start date unparseable"
    if start >= approval:
        return "COMPLIANT", "work started on or after approval"
    return (
        "VIOLATION",
        f"work started {(approval - start).days} day(s) before approval",
    )


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
#
# Three entries are verified against Data/legal/GFR_2017_original.pdf (see
# the module docstring for what verification caught). Two remain gated and
# may not emit a violation until a reviewer confirms them.
#
# A rule was deliberately LEFT OUT for the same reason: a cost-overrun limit
# requiring revised administrative approval. Departments set that percentage
# differently and no single figure is safely quotable, so encoding one would
# have meant inventing a statutory number - exactly what this layer exists to
# avoid. It belongs in Stage 5 as a judgement constant or nowhere.

RULE_REGISTRY: Tuple[ComplianceRule, ...] = (
    ComplianceRule(
        rule_id="OPEN_TENDER_ABOVE_THRESHOLD",
        citation="GFR 2017, Rule 161 (Advertised Tender Enquiry)",
        source="General Financial Rules 2017, Government of India",
        source_url="https://doe.gov.in/order-circular/general-financial-rules-2017",
        statement=(
            "Procurement above the prescribed value must use open advertised "
            "tendering rather than a limited or single-source route."
        ),
        required_fields=("sanctioned_amount", "procurement_method"),
        check=_check_open_tender,
        threshold=OPEN_TENDER_THRESHOLD_INR,
        threshold_units="INR",
        citation_verified=True,
        verification_note=(
            "VERIFIED 2026-09-06 against Data/legal/GFR_2017_original.pdf: "
            "Rule 161 requires advertisement for estimated value of "
            "'Rs. 25 lakhs (Rupees Twenty Five Lakh) and above', on CPPP "
            "(eprocure.gov.in) and GeM. The threshold was correct; the rule "
            "number was not - it was stated as 155, which is purchase by "
            "committee. Re-verify against the amendment-updated edition "
            "(GFR_updated_2024.pdf) before production: thresholds move."
        ),
        compliant_example={
            "sanctioned_amount": 5_000_000.0,
            "procurement_method": "OPEN_TENDER",
        },
        violating_example={
            "sanctioned_amount": 5_000_000.0,
            "procurement_method": "LIMITED_TENDER",
        },
    ),
    ComplianceRule(
        rule_id="LIMITED_TENDER_MIN_BIDDERS",
        citation="GFR 2017, Rule 162 (Limited Tender Enquiry)",
        source="General Financial Rules 2017, Government of India",
        source_url="https://doe.gov.in/order-circular/general-financial-rules-2017",
        statement=(
            "A limited tender enquiry must be sent to more than three supplier "
            "firms - that is, at least four - to constitute a competition."
        ),
        required_fields=("procurement_method", "n_bidders"),
        check=_check_limited_tender_bidders,
        threshold=float(LIMITED_TENDER_MIN_BIDDERS),
        threshold_units="supplier firms invited",
        citation_verified=True,
        verification_note=(
            "VERIFIED 2026-09-06 against Data/legal/GFR_2017_original.pdf: "
            "'The number of supplier firms in Limited Tender Enquiry should be "
            "more than three.' Two corrections followed. The rule number was "
            "162, not 154. And 'more than three' means a minimum of FOUR - the "
            "original code tested >= 3 and would have passed every three-firm "
            "enquiry the rule exists to catch. Note the rule counts firms "
            "INVITED, not bids received; where a dataset supplies only bids "
            "received this rule under-reports."
        ),
        compliant_example={"procurement_method": "LIMITED_TENDER", "n_bidders": 4},
        violating_example={"procurement_method": "LIMITED_TENDER", "n_bidders": 3},
    ),
    ComplianceRule(
        rule_id="UC_SUBMITTED_WITHIN_PERIOD",
        citation="GFR 2017, Rule 238 (Utilisation Certificates)",
        source="General Financial Rules 2017, Government of India",
        source_url="https://doe.gov.in/order-circular/general-financial-rules-2017",
        statement=(
            "A utilisation certificate is due within the prescribed period "
            "after the sanctioned work concludes."
        ),
        required_fields=("date_completion", "uc_submitted_date"),
        check=_check_uc_timeliness,
        threshold=float(UC_DUE_DAYS),
        threshold_units="days",
        verification_note=(
            "STILL GATED. Partially checked 2026-09-06: the UC provision is "
            "Rule 238, not Rule 230 (which is grants-in-aid principles), and "
            "the text refers to a period running from 'the close of the "
            "succeeding financial year' - NOT from completion, which is what "
            "this check implements. The extracted text is garbled by the "
            "PDF's two-column layout, so the exact period and its anchor need "
            "a human read of the page before this rule may fire. Changing the "
            "anchor changes every result, so it stays gated."
        ),
        compliant_example={
            "date_completion": "2022-01-01",
            "uc_submitted_date": "2022-06-01",
        },
        violating_example={
            "date_completion": "2022-01-01",
            "uc_submitted_date": "2023-06-01",
        },
    ),
    ComplianceRule(
        rule_id="SINGLE_SOURCE_JUSTIFICATION_RECORDED",
        citation="GFR 2017, Rule 166 (Single Tender Enquiry)",
        source="General Financial Rules 2017, Government of India",
        source_url="https://doe.gov.in/files/inline-documents/GFR2017.pdf",
        statement=(
            "An award made without competition must carry a justification "
            "recorded on the file."
        ),
        required_fields=("procurement_method", "single_source_justification"),
        check=_check_single_source_justification,
        citation_verified=True,
        verification_note=(
            "VERIFIED 2026-09-06 against Data/legal/GFR_2017_original.pdf: "
            "Rule 166 permits single-source procurement in listed "
            "circumstances, and requires that 'the reason for such decision is "
            "to be recorded and approval of competent authority obtained'. "
            "Originally cited to CVC circulars generally, which was the "
            "weakest citation in the registry; GFR carries it directly."
        ),
        compliant_example={
            "procurement_method": "SINGLE_SOURCE",
            "single_source_justification": "recorded",
        },
        violating_example={
            "procurement_method": "SINGLE_SOURCE",
            "single_source_justification": NOT_RECORDED,
        },
    ),
    ComplianceRule(
        rule_id="WORK_NOT_STARTED_BEFORE_APPROVAL",
        citation="CPWD Works Manual - administrative approval precedes execution",
        source="CPWD Works Manual",
        source_url="https://cpwd.gov.in/",
        statement=(
            "Execution of a work must not commence before administrative "
            "approval and expenditure sanction are in place."
        ),
        required_fields=("date_approval", "date_start"),
        check=_check_work_before_approval,
        verification_note=(
            "Confirm the clause and the manual edition. Note that state PWD "
            "manuals govern state works and may differ from CPWD; a record's "
            "executing agency decides which manual applies to it."
        ),
        compliant_example={"date_approval": "2021-05-01", "date_start": "2021-06-01"},
        violating_example={"date_approval": "2021-05-01", "date_start": "2021-04-01"},
    ),
)


def evaluate_rule(rule: ComplianceRule, record: Mapping[str, Any]) -> ComplianceOutcome:
    """Apply one rule to one record.

    Missing inputs are checked before applicability, so a rule can never
    report NOT_APPLICABLE for the wrong reason - "this rule does not concern
    this record" and "we cannot tell whether it does" are different answers.

    Args:
        rule: The rule to apply.
        record: A mapping of field name to value.

    Returns:
        A :class:`ComplianceOutcome`. Never COMPLIANT on incomplete input, and
        never VIOLATION on an unverified citation.
    """
    absent = [name for name in rule.required_fields if _missing(record.get(name))]
    if absent:
        return ComplianceOutcome(
            rule_id=rule.rule_id,
            status="UNDETERMINABLE",
            detail=f"missing: {', '.join(sorted(absent))}",
            citation=rule.citation,
        )

    status, detail = rule.check(record)

    if status == "VIOLATION" and not rule.citation_verified:
        return ComplianceOutcome(
            rule_id=rule.rule_id,
            status="PENDING_CITATION_VERIFICATION",
            detail=f"would flag: {detail}. Citation unverified: {rule.citation}",
            citation=rule.citation,
        )

    return ComplianceOutcome(
        rule_id=rule.rule_id, status=status, detail=detail, citation=rule.citation
    )


@dataclass(frozen=True)
class ComplianceResult:
    """Every rule applied to every record."""

    violations: pd.DataFrame
    outcomes: pd.DataFrame
    summary: Dict[str, Any] = field(default_factory=dict)


def evaluate_compliance(frame: pd.DataFrame) -> ComplianceResult:
    """Apply the whole registry to a frame.

    Args:
        frame: Records to check. Columns the rules need may be absent - the
            realistic case, since the public datasets available are
            state-year aggregates with no procurement fields at all. Those
            records come back UNDETERMINABLE rather than raising.

    Returns:
        A :class:`ComplianceResult` whose ``outcomes`` has one row per
        (record, rule) pair and whose ``violations`` holds only confirmed
        violations - which is empty while every citation is unverified.
    """
    if frame.empty:
        empty = pd.DataFrame(
            columns=["record_id", "rule_id", "status", "detail", "citation"]
        )
        return ComplianceResult(
            violations=empty,
            outcomes=empty,
            summary={"n_records": 0, "n_rules": len(RULE_REGISTRY), "by_status": {}},
        )

    rows: List[Dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        record_id = record.get("record_id", "")
        for rule in RULE_REGISTRY:
            outcome = evaluate_rule(rule, record)
            rows.append(
                {
                    "record_id": record_id,
                    "rule_id": outcome.rule_id,
                    "status": outcome.status,
                    "detail": outcome.detail,
                    "citation": outcome.citation,
                }
            )

    outcomes = pd.DataFrame(rows)
    violations = outcomes[outcomes["status"] == "VIOLATION"].reset_index(drop=True)
    counts = {str(k): int(v) for k, v in outcomes["status"].value_counts().items()}

    LOGGER.info(
        "Stage 8 compliance: %d record(s) x %d rule(s) -> %s",
        len(frame),
        len(RULE_REGISTRY),
        counts,
    )

    return ComplianceResult(
        violations=violations,
        outcomes=outcomes,
        summary={
            "n_records": int(len(frame)),
            "n_rules": len(RULE_REGISTRY),
            "by_status": counts,
            "n_citation_verified": sum(r.citation_verified for r in RULE_REGISTRY),
        },
    )


def registry_report() -> Dict[str, Any]:
    """The registry as data, for the Stage 8 report and for review."""
    verified = sum(rule.citation_verified for rule in RULE_REGISTRY)
    return {
        "n_rules": len(RULE_REGISTRY),
        "n_citation_verified": verified,
        "operational": verified > 0,
        "_status": (
            "INERT - no rule may emit a violation until its citation is "
            "verified against the primary source."
            if verified == 0
            else f"{verified} of {len(RULE_REGISTRY)} rule(s) verified and live."
        ),
        "rules": [
            {
                "rule_id": rule.rule_id,
                "citation": rule.citation,
                "source": rule.source,
                "source_url": rule.source_url,
                "statement": rule.statement,
                "required_fields": list(rule.required_fields),
                "threshold": rule.threshold,
                "threshold_units": rule.threshold_units,
                "threshold_provenance": rule.threshold_provenance,
                "citation_verified": rule.citation_verified,
                "verification_note": rule.verification_note,
            }
            for rule in RULE_REGISTRY
        ],
    }
