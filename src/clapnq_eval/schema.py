from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


JudgeLabel = Literal["CORRECT", "MINOR_ERROR", "MAJOR_ERROR"]


class JudgeResult(BaseModel):
    """Structured Judge output. Field order is the decoding order."""

    model_config = ConfigDict(extra="forbid")

    label: JudgeLabel
    reason: str = Field(min_length=1, max_length=600)


JUDGE_JSON_SCHEMA = JudgeResult.model_json_schema()
