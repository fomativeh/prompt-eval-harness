from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


Category = Literal["billing", "technical", "account_access", "sales", "other"]
Urgency = Literal["low", "medium", "high"]
ReplyTone = Literal["empathetic", "neutral", "firm"]


class EvalCase(BaseModel):
    id: str
    customer_message: str
    expected_category: Category
    expected_urgency: Urgency
    expected_should_refuse: bool
    expected_reply_tone: ReplyTone
    tags: list[str] = Field(default_factory=list)


class ModelOutput(BaseModel):
    category: Category
    urgency: Urgency
    should_refuse: bool
    reply_tone: ReplyTone
    response: str


class CaseResult(BaseModel):
    case_id: str
    prompt_name: str
    parsed: bool
    output_text: str
    output: ModelOutput | None
    category_correct: bool
    urgency_correct: bool
    tone_correct: bool
    refusal_correct: bool
    format_correct: bool
    passed: bool


class RunMetrics(BaseModel):
    category_accuracy: float
    urgency_accuracy: float
    format_pass_rate: float
    tone_accuracy: float
    refusal_accuracy: float
    overall_score: float


class PromptRunSummary(BaseModel):
    prompt_name: str
    prompt_path: Path
    prompt_file_name: str
    prompt_sha256: str
    prompt_text: str
    metrics: RunMetrics
    case_results: list[CaseResult]


class DatasetMetadata(BaseModel):
    dataset_path: Path
    dataset_file_name: str
    dataset_sha256: str
    case_count: int


class EvalRunArtifact(BaseModel):
    provider: str
    model: str
    dataset: DatasetMetadata
    prompt_summaries: list[PromptRunSummary]
