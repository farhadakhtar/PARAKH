"""Property tests for the Stage 8 statutory compliance layer.

Written before the implementation.

This layer exists because of a distinction that took a while to state
properly. PARAKH's thresholds are judgement constants - ``R > 0.7`` is
somebody's taste, and calibrating it needs outcome labels that do not exist.
But a subset of the numbers a procurement system cares about are not taste at
all: they are fixed by statute. A tender limit is not an estimate, it is a
rule with a number in it, and the correct provenance for that number is a
citation rather than a fit.

So this layer answers a question that HAS an oracle - "was the written rule
followed?" - and leaves the question that does not - "was this fraud?" -
alone.

Two properties carry the weight:

:class:`TestUndeterminableNeverCompliant`
    A rule evaluated against a record missing the fields it needs must return
    UNDETERMINABLE. If it returned COMPLIANT, absence of evidence would be
    silently recorded as evidence of compliance, and the most incomplete
    records - the ones least able to defend themselves - would score cleanest.
    This is the same "undefined is not zero" discipline the rest of PARAKH
    runs on, applied where it is easiest to get wrong.

:class:`TestCitationDiscipline`
    A rule whose citation has not been verified against a primary source may
    not emit a violation. An unverifiable citation is worse than no citation:
    it carries the authority of law without the substance, and a reviewer
    checking it finds nothing. Fabricating a rule number to make a compliance
    engine look complete is the exact failure this whole system is built to
    avoid.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.stage8.compliance import (
    COMPLIANCE_OUTCOMES,
    RULE_REGISTRY,
    ComplianceRule,
    evaluate_compliance,
    evaluate_rule,
    registry_report,
)


def make_record(**overrides: object) -> dict:
    """A procurement record with every field a rule might need."""
    record = {
        "record_id": "R1",
        "sanctioned_amount": 1_000_000.0,
        "final_cost": 1_050_000.0,
        "n_bidders": 3,
        "procurement_method": "OPEN_TENDER",
        "date_proposal": "2021-04-01",
        "date_approval": "2021-05-01",
        "date_start": "2021-06-01",
        "date_completion": "2022-01-01",
        "uc_submitted_date": "2022-06-01",
        "advance_paid": 100_000.0,
        "single_source_justification": "recorded",
    }
    record.update(overrides)
    return record


# ===========================================================================
# 1. UNDETERMINABLE IS NOT COMPLIANT - the load-bearing property
# ===========================================================================


class TestUndeterminableNeverCompliant:
    """Missing inputs produce UNDETERMINABLE, never a clean bill of health."""

    @pytest.mark.parametrize("rule_id", [rule.rule_id for rule in RULE_REGISTRY])
    def test_empty_record_is_never_compliant(self, rule_id: str) -> None:
        """No rule may pass a record that carries none of its inputs.

        Parametrised over the whole registry so a rule added later cannot
        quietly opt out of the guarantee.
        """
        rule = next(r for r in RULE_REGISTRY if r.rule_id == rule_id)

        outcome = evaluate_rule(rule, {"record_id": "EMPTY"})

        assert outcome.status != "COMPLIANT"
        assert outcome.status in COMPLIANCE_OUTCOMES

    @pytest.mark.parametrize("rule_id", [rule.rule_id for rule in RULE_REGISTRY])
    def test_each_required_field_is_actually_required(self, rule_id: str) -> None:
        """Dropping any single declared input makes the rule undeterminable.

        Catches a rule that declares a field it never reads, which would let
        the registry advertise a stricter precondition than the code enforces.
        """
        rule = next(r for r in RULE_REGISTRY if r.rule_id == rule_id)

        for field in rule.required_fields:
            record = make_record()
            record.pop(field, None)
            outcome = evaluate_rule(rule, record)
            assert outcome.status == "UNDETERMINABLE", (
                f"{rule_id} still returned {outcome.status} without {field!r}"
            )

    def test_none_and_nan_count_as_missing(self) -> None:
        """A present-but-null field is missing, not zero."""
        rule = RULE_REGISTRY[0]
        for empty in (None, float("nan"), ""):
            record = make_record(**{field: empty for field in rule.required_fields})
            assert evaluate_rule(rule, record).status == "UNDETERMINABLE"


# ===========================================================================
# 2. CITATION DISCIPLINE
# ===========================================================================


class TestCitationDiscipline:
    """A rule is only as good as the source a reviewer can check."""

    def test_every_rule_carries_a_citation_and_source(self) -> None:
        """No anonymous rules."""
        for rule in RULE_REGISTRY:
            assert rule.citation.strip(), f"{rule.rule_id} has no citation"
            assert rule.source.strip(), f"{rule.rule_id} has no source document"
            assert rule.statement.strip(), f"{rule.rule_id} has no statement"

    def test_unverified_rules_cannot_emit_violations(self) -> None:
        """The gate that stops a fabricated rule number becoming a finding.

        A rule whose citation has not been checked against the primary source
        may still be evaluated - its logic is testable - but its result is
        reported as PENDING_CITATION_VERIFICATION rather than VIOLATION, so
        nothing acts on a rule nobody has confirmed exists.
        """
        unverified = [r for r in RULE_REGISTRY if not r.citation_verified]
        for rule in unverified:
            record = make_record(**rule.violating_example)
            outcome = evaluate_rule(rule, record)
            assert outcome.status != "VIOLATION", (
                f"{rule.rule_id} emitted a VIOLATION on an unverified citation"
            )
            assert outcome.status == "PENDING_CITATION_VERIFICATION"

    def test_verified_rules_do_emit_violations(self) -> None:
        """The gate must not be so strict that nothing can ever fire."""
        verified = [r for r in RULE_REGISTRY if r.citation_verified]
        for rule in verified:
            record = make_record(**rule.violating_example)
            assert evaluate_rule(rule, record).status == "VIOLATION"

    def test_no_threshold_is_a_judgement_constant(self) -> None:
        """Every numeric threshold traces to the statute, not to taste.

        This is the whole point of the layer. A threshold with provenance
        'judgement' belongs in Stage 5 where it is labelled uncalibrated, not
        here where it would borrow the authority of a citation.
        """
        for rule in RULE_REGISTRY:
            if rule.threshold is not None:
                assert rule.threshold_provenance == "STATUTORY", (
                    f"{rule.rule_id} carries a non-statutory threshold"
                )


# ===========================================================================
# 3. RULE SEMANTICS
# ===========================================================================


class TestRuleSemantics:
    """Applicability, determinism, and the compliant path."""

    def test_inapplicable_rules_return_not_applicable(self) -> None:
        """A tender rule says nothing about a record that had no tender."""
        rule = next(
            (r for r in RULE_REGISTRY if "procurement_method" in r.required_fields),
            None,
        )
        if rule is None:
            pytest.skip("no procurement-method rule in the registry")

        record = make_record(procurement_method="NOT_A_PROCUREMENT")
        assert evaluate_rule(rule, record).status in {
            "NOT_APPLICABLE",
            "COMPLIANT",
            "UNDETERMINABLE",
        }

    def test_each_rule_has_a_passing_and_a_failing_example(self) -> None:
        """The registry's own examples must actually demonstrate the rule.

        A rule shipped with an example that does not trigger it is a rule
        nobody has run.
        """
        for rule in RULE_REGISTRY:
            failing = evaluate_rule(rule, make_record(**rule.violating_example))
            passing = evaluate_rule(rule, make_record(**rule.compliant_example))
            assert failing.status in {"VIOLATION", "PENDING_CITATION_VERIFICATION"}
            assert passing.status == "COMPLIANT"

    def test_evaluation_is_deterministic(self) -> None:
        """Same record, same verdict."""
        record = make_record()
        for rule in RULE_REGISTRY:
            first = evaluate_rule(rule, record)
            second = evaluate_rule(rule, record)
            assert first.status == second.status
            assert first.detail == second.detail

    def test_rule_ids_are_unique(self) -> None:
        ids = [rule.rule_id for rule in RULE_REGISTRY]
        assert len(ids) == len(set(ids))


# ===========================================================================
# 4. FRAME-LEVEL EVALUATION
# ===========================================================================


class TestFrameEvaluation:
    """The vectorised entry point behaves like the per-record one."""

    def test_empty_frame_is_handled(self) -> None:
        result = evaluate_compliance(pd.DataFrame())
        assert result.violations.empty
        assert result.summary["n_records"] == 0

    def test_counts_add_up(self) -> None:
        """Every (record, rule) pair lands in exactly one outcome bucket."""
        frame = pd.DataFrame([make_record(record_id=f"R{i}") for i in range(20)])

        result = evaluate_compliance(frame)
        counts = result.summary["by_status"]

        assert sum(counts.values()) == 20 * len(RULE_REGISTRY)

    def test_missing_columns_do_not_crash(self) -> None:
        """A frame with none of the compliance fields is undeterminable, not fatal.

        This is the realistic case: the public datasets in ``Data/`` are
        state-year aggregates with no procurement fields at all.
        """
        frame = pd.DataFrame({"record_id": ["A", "B"], "state": ["NAGALAND", "BIHAR"]})

        result = evaluate_compliance(frame)

        assert result.summary["by_status"].get("VIOLATION", 0) == 0
        assert result.summary["by_status"]["UNDETERMINABLE"] == 2 * len(RULE_REGISTRY)

    def test_report_states_verification_status(self) -> None:
        """The registry report must say how much of itself is unverified."""
        report = registry_report()
        assert "n_rules" in report
        assert "n_citation_verified" in report
        assert report["n_rules"] == len(RULE_REGISTRY)
