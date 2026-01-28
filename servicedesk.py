from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class TicketIn(BaseModel):
    issue_key: str = Field(..., min_length=1, max_length=128)
    text: str = Field(..., min_length=1)


class TicketsBatchIn(BaseModel):
    source: Optional[str] = Field(default="service_desk", max_length=64)
    tickets: List[TicketIn] = Field(default_factory=list)
