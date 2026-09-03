"""Typed schema for PARAKH work records read from Postgres."""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class WorkRecord(BaseModel):
    mp_id: Optional[str] = None
    work_id: str
    location: Optional[Dict[str, Any]] = None
    payment_id: Optional[str] = None
    district_id: Optional[str] = None
    work_status: Optional[str] = None
    contractor_id: Optional[str] = None
    cost_estimate: Optional[float] = None
    sanction_date: Optional[date] = None
    work_category: Optional[str] = None
    completion_date: Optional[date] = None
    constituency_id: Optional[str] = None
    work_description: Optional[str] = None
    sanctioned_amount: Optional[float] = None
    uc_submission_date: Optional[date] = None
    payment_release_date: Optional[date] = None
    implementing_agency_id: Optional[str] = None
    utilization_certificate_status: Optional[str] = None
    confidence_state: Dict[str, str] = Field(default_factory=dict)

    @field_validator("cost_estimate", "sanctioned_amount", mode="before")
    @classmethod
    def blank_number_to_none(cls, value: Any) -> Any:
        if value == "":
            return None
        return value

    @field_validator(
        "sanction_date",
        "completion_date",
        "uc_submission_date",
        "payment_release_date",
        mode="before",
    )
    @classmethod
    def blank_date_to_none(cls, value: Any) -> Any:
        if value == "":
            return None
        return value
