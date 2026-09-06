"""Stage 8 - documented fraud typologies.

The argument for this module
----------------------------
Stage 8's calibration is blocked because Indian fraud *convictions* are scarce
and slow. That is true, but it is not the whole picture, and treating it as
the whole picture was a scoping error: adjudicated procurement-fraud data does
exist, is public, and covers India.

* **World Bank debarment and cross-debarment** - firms sanctioned for fraud,
  collusion, corruption, coercion or obstruction in Bank-financed projects.
  Same domain as PMGSY (donor-financed public works), decided by a Sanctions
  Board that publishes its reasoning, and it names Indian firms.
* **CVC and state PWD blacklists** - Indian, adjudicated, entity-level.
* **OECD bid-rigging guidance** - not case data, but a synthesis of what
  competition authorities have repeatedly proven in enforcement.

Why a typology library rather than a classifier
-----------------------------------------------
Two properties of that data decide the design.

**It labels entities, not works.** A debarment says a firm was sanctioned in a
given year. It does not say which of that firm's contracts were the corrupt
ones. Fitting a record-level model on an entity-level label would spread one
adjudicated fact across hundreds of records - the same error that put 5,237
labels on a postal directory, in a more respectable costume.

**It labels caught fraud.** A model trained on enforcement outcomes learns to
recognise fraud that resembles previously-caught fraud. The sophisticated
uncaught kind is, by construction, absent from the positive class, so recall
against *all* fraud is not merely unmeasured but unmeasurable. More data does
not fix this; it is structural.

Neither is a reason to discard the source. Both are reasons to name the claim
correctly. A typology detector says **"this matches a pattern that has been
proven in adjudicated cases"** - which is checkable, citable, and useful - and
it does not say "this is fraud". The library therefore carries patterns and
their sources, not fitted weights, and each detector's authority comes from
the case that established it.

Verification, as elsewhere in Stage 8
--------------------------------------
Every typology ships ``source_verified=False``. The signatures below are
stated from working knowledge of the public literature and have not been
checked against the primary documents in this environment. Until a reviewer
confirms a typology against its source, :func:`detect_typologies` reports it
as ``PENDING_SOURCE_VERIFICATION`` and it produces no finding - the same gate
the compliance registry uses, for the same reason.

Detection here is also **structural**, not statistical: each signature is a
stated, inspectable condition over vendor/bid data. Nothing is fitted, so
nothing needs labels to run - and nothing can quietly become a probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.core.logger import get_logger

LOGGER = get_logger(__name__)

#: What a typology check can conclude.
TYPOLOGY_OUTCOMES: Tuple[str, ...] = (
    "MATCH",
    "NO_MATCH",
    "UNDETERMINABLE",
    "PENDING_SOURCE_VERIFICATION",
)

#: The unit a typology makes a statement about. Recorded per typology because
#: mixing them is the failure this module exists to prevent: a VENDOR-level
#: pattern must never be reported as a fact about one work.
TYPOLOGY_SUBJECTS: Tuple[str, ...] = ("VENDOR", "TENDER", "WORK", "VENDOR_PAIR")


@dataclass(frozen=True)
class Typology:
    """A documented fraud pattern and the source that establishes it.

    Attributes:
        typology_id: Stable identifier.
        name: The pattern's usual name in the literature.
        subject: What the pattern is a statement about - see
            :data:`TYPOLOGY_SUBJECTS`. A VENDOR typology never labels a work.
        description: What the scheme is, in plain words.
        signature: The observable condition, stated so a reviewer can check
            the code against the sentence.
        required_fields: Data needed to evaluate it. Absent any, the verdict
            is UNDETERMINABLE, never NO_MATCH - a pattern that cannot be
            looked for has not been ruled out.
        source: The document or case body that establishes the pattern.
        source_url: Where to find it.
        source_verified: Whether that has been checked against the primary
            source. False gates the typology to PENDING_SOURCE_VERIFICATION.
        verification_note: What a reviewer must confirm.
        detects_only_known: Always True, and stated per typology rather than
            once in prose, so it travels with the finding into any report.
    """

    typology_id: str
    name: str
    subject: str
    description: str
    signature: str
    required_fields: Tuple[str, ...]
    source: str
    source_url: str
    verification_note: str
    source_verified: bool = False
    detects_only_known: bool = True


TYPOLOGY_LIBRARY: Tuple[Typology, ...] = (
    Typology(
        typology_id="BID_ROTATION",
        name="Bid rotation",
        subject="VENDOR_PAIR",
        description=(
            "A stable group of firms bids on the same tenders, taking turns "
            "to submit the lowest bid so each wins a share over time."
        ),
        signature=(
            "The same set of bidders recurs across tenders while the identity "
            "of the winner cycles, with each member's win rate close to "
            "1/group size and losing bids clustered just above the winner."
        ),
        required_fields=("tender_id", "vendor_id", "bid_amount", "is_winner"),
        source="OECD Guidelines for Fighting Bid Rigging in Public Procurement",
        source_url="https://www.oecd.org/competition/bidrigging.htm",
        verification_note=(
            "Confirm the pattern definition and, more importantly, decide the "
            "recurrence and clustering thresholds from case evidence rather "
            "than by choosing round numbers - an unsourced threshold here "
            "reintroduces exactly the problem this layer avoids."
        ),
    ),
    Typology(
        typology_id="COVER_BIDDING",
        name="Cover or complementary bidding",
        subject="TENDER",
        description=(
            "Competitors submit deliberately uncompetitive bids so a "
            "pre-selected winner appears to have won a genuine competition."
        ),
        signature=(
            "Losing bids are implausibly far above the winner, or cluster at "
            "a near-constant margin, or are submitted by firms that never win "
            "anything in the same category."
        ),
        required_fields=("tender_id", "vendor_id", "bid_amount", "is_winner"),
        source="OECD Guidelines for Fighting Bid Rigging in Public Procurement",
        source_url="https://www.oecd.org/competition/bidrigging.htm",
        verification_note=(
            "Confirm whether the margin test is defined on absolute or "
            "relative spread; the two disagree on large-value tenders."
        ),
    ),
    Typology(
        typology_id="SINGLE_BIDDER_CONCENTRATION",
        name="Persistent single-bidder awards",
        subject="VENDOR",
        description=(
            "A firm repeatedly wins tenders that attracted no other "
            "responsive bid, in a market where competition would be expected."
        ),
        signature=(
            "A vendor's share of single-bid awards is far above the base rate "
            "for the same scheme, district and value band."
        ),
        required_fields=("vendor_id", "tender_id", "n_bidders"),
        source=(
            "World Bank Sanctions Board decisions; EU single-bidding "
            "indicators in public procurement"
        ),
        source_url="https://www.worldbank.org/en/about/unit/sanctions-system",
        verification_note=(
            "Single bidding is a red flag, not proof - remote and specialised "
            "works legitimately attract one bidder. Confirm the peer base "
            "rate is computed within scheme and value band before use."
        ),
    ),
    Typology(
        typology_id="DEBARRED_ENTITY_PARTICIPATION",
        name="Award to a sanctioned entity",
        subject="VENDOR",
        description=(
            "A contract is awarded to a firm that appears on a debarment or "
            "blacklist at the time of award, or to a successor of one."
        ),
        signature=(
            "Vendor identity matches a published debarment list entry whose "
            "sanction period covers the award date."
        ),
        required_fields=("vendor_id", "vendor_name", "date_approval"),
        source=(
            "World Bank listing of debarred and cross-debarred firms; CVC and "
            "state PWD blacklists"
        ),
        source_url="https://www.worldbank.org/en/projects-operations/procurement/debarred-firms",
        verification_note=(
            "This is the only typology here whose evidence is an adjudicated "
            "fact about a named entity rather than an inferred pattern, so it "
            "is the strongest - but matching must be on identity, not name "
            "similarity. A fuzzy name match against a debarment list is a "
            "defamation risk, not a finding."
        ),
    ),
    Typology(
        typology_id="SPLIT_TO_AVOID_THRESHOLD",
        name="Contract splitting",
        subject="TENDER",
        description=(
            "A work is divided into parts each falling just below a "
            "procurement threshold, avoiding open tendering."
        ),
        signature=(
            "Multiple awards to the same vendor, in the same district and "
            "short window, each just below a statutory threshold, whose sum "
            "exceeds it."
        ),
        required_fields=(
            "vendor_id",
            "district",
            "date_approval",
            "sanctioned_amount",
        ),
        source="GFR procurement thresholds; CAG audit findings on splitting",
        source_url="https://cag.gov.in/",
        verification_note=(
            "Depends on the same statutory threshold as the compliance "
            "registry's open-tender rule. Verify once and share the value - "
            "two copies of a legal threshold will drift apart."
        ),
    ),
)


@dataclass(frozen=True)
class TypologyOutcome:
    """One typology's verdict on one subject."""

    typology_id: str
    subject_id: str
    status: str
    detail: str
    source: str


#: Fields that are the same thing under a different column name. Without
#: this, reachability reports a gap that is really a spelling difference:
#: PARAKH's corpus calls it ``sanction_amount`` and this library asked for
#: ``sanctioned_amount``, which would have been reported as missing data and
#: sent somebody looking for a dataset they already had.
FIELD_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "sanctioned_amount": ("sanction_amount", "sanctioned_amount", "budget_allocated"),
    "vendor_id": ("vendor_id", "vendor_name", "contractor_id", "contractor_name"),
    "date_approval": ("date_approval", "approval_date", "date_of_approval"),
    "district": ("district", "district_name"),
    "tender_id": ("tender_id", "tender_no", "tender_reference"),
    "bid_amount": ("bid_amount", "quoted_amount", "bid_value"),
    "is_winner": ("is_winner", "awarded", "award_flag"),
    "n_bidders": ("n_bidders", "num_bidders", "no_of_bidders", "bidders_count"),
    "vendor_name": ("vendor_name", "contractor_name", "firm_name"),
}


def _resolve(field_name: str, columns: set) -> Optional[str]:
    """The column satisfying ``field_name``, allowing for known aliases."""
    for candidate in FIELD_ALIASES.get(field_name, (field_name,)):
        if candidate in columns:
            return candidate
    return None


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null", "unknown", "-"}


def detect_typologies(
    frame: pd.DataFrame, *, library: Sequence[Typology] = TYPOLOGY_LIBRARY
) -> Dict[str, Any]:
    """Report which typologies could even be looked for in this data.

    No typology is *evaluated* yet: every one is unverified, so every one is
    gated. What this does return is the more immediately useful answer - which
    patterns the available columns could support at all, and which are
    unreachable because the data does not carry bids, vendors or tenders.

    On the current corpus that answer is stark and worth having in writing:
    the public datasets are state-year aggregates, so every bid-level typology
    is UNDETERMINABLE for want of a bid table, not for want of a model.

    Args:
        frame: Records to assess.
        library: Typologies to consider.

    Returns:
        A mapping with per-typology reachability and the verification summary.
    """
    columns = set(frame.columns)
    results: Dict[str, Any] = {}

    for typology in library:
        absent = [
            name
            for name in typology.required_fields
            if _resolve(name, columns) is None
        ]
        if absent:
            status = "UNDETERMINABLE"
            detail = f"data lacks {', '.join(absent)}"
        elif not typology.source_verified:
            status = "PENDING_SOURCE_VERIFICATION"
            detail = f"fields present; source unverified: {typology.source}"
        else:  # pragma: no cover - unreachable until a source is verified
            status = "EVALUABLE"
            detail = "fields present and source verified"

        results[typology.typology_id] = {
            "name": typology.name,
            "subject": typology.subject,
            "status": status,
            "detail": detail,
            "missing_fields": absent,
            "source": typology.source,
            "source_verified": typology.source_verified,
            "detects_only_known": typology.detects_only_known,
        }

    verified = sum(t.source_verified for t in library)
    LOGGER.info(
        "Stage 8 typologies: %d defined, %d source-verified, %d reachable "
        "with the columns present",
        len(library),
        verified,
        sum(1 for v in results.values() if v["status"] != "UNDETERMINABLE"),
    )

    return {
        "n_typologies": len(library),
        "n_source_verified": verified,
        "n_reachable": sum(
            1 for value in results.values() if value["status"] != "UNDETERMINABLE"
        ),
        "typologies": results,
        "_claim_limit": (
            "A typology match means the record resembles a pattern proven in "
            "adjudicated cases. It is not a fraud finding. Because "
            "enforcement data records only CAUGHT fraud, recall against "
            "undetected schemes is unmeasurable, and no coverage claim may be "
            "made from these detectors."
        ),
        "_subject_warning": (
            "VENDOR and VENDOR_PAIR typologies make statements about an "
            "entity, never about an individual work. Attributing an "
            "entity-level finding to one record would repeat the labelling "
            "error that invalidated the previous calibration dataset."
        ),
    }


def typology_report() -> Dict[str, Any]:
    """The library as data, for the Stage 8 report and for review."""
    return {
        "n_typologies": len(TYPOLOGY_LIBRARY),
        "n_source_verified": sum(t.source_verified for t in TYPOLOGY_LIBRARY),
        "_status": (
            "INERT - every typology awaits source verification. The library "
            "defines what to look for and where the definition came from; it "
            "asserts nothing about any record yet."
        ),
        "typologies": [
            {
                "typology_id": t.typology_id,
                "name": t.name,
                "subject": t.subject,
                "description": t.description,
                "signature": t.signature,
                "required_fields": list(t.required_fields),
                "source": t.source,
                "source_url": t.source_url,
                "source_verified": t.source_verified,
                "verification_note": t.verification_note,
            }
            for t in TYPOLOGY_LIBRARY
        ],
    }
