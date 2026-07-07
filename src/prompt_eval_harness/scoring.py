from __future__ import annotations

import json

from pydantic import ValidationError

from prompt_eval_harness.models import CaseResult, EvalCase, ModelOutput, RunMetrics


def score_case(case: EvalCase, prompt_name: str, output_text: str) -> CaseResult:
    parsed = False
    output = None
    try:
        output = ModelOutput.model_validate(json.loads(output_text))
        parsed = True
    except (json.JSONDecodeError, ValidationError):
        output = None

    category_correct = bool(output and output.category == case.expected_category)
    urgency_correct = bool(output and output.urgency == case.expected_urgency)
    tone_correct = bool(output and output.reply_tone == case.expected_reply_tone)
    refusal_correct = bool(output and output.should_refuse == case.expected_should_refuse)
    format_correct = parsed
    passed = all(
        [category_correct, urgency_correct, tone_correct, refusal_correct, format_correct]
    )

    return CaseResult(
        case_id=case.id,
        prompt_name=prompt_name,
        parsed=parsed,
        output_text=output_text,
        output=output,
        category_correct=category_correct,
        urgency_correct=urgency_correct,
        tone_correct=tone_correct,
        refusal_correct=refusal_correct,
        format_correct=format_correct,
        passed=passed,
    )


def compute_metrics(results: list[CaseResult]) -> RunMetrics:
    total = len(results)
    if total == 0:
        raise ValueError("No results to score")

    category_accuracy = sum(item.category_correct for item in results) / total
    urgency_accuracy = sum(item.urgency_correct for item in results) / total
    format_pass_rate = sum(item.format_correct for item in results) / total
    tone_accuracy = sum(item.tone_correct for item in results) / total
    refusal_accuracy = sum(item.refusal_correct for item in results) / total
    overall_score = (
        category_accuracy
        + urgency_accuracy
        + format_pass_rate
        + tone_accuracy
        + refusal_accuracy
    ) / 5

    return RunMetrics(
        category_accuracy=category_accuracy,
        urgency_accuracy=urgency_accuracy,
        format_pass_rate=format_pass_rate,
        tone_accuracy=tone_accuracy,
        refusal_accuracy=refusal_accuracy,
        overall_score=overall_score,
    )
