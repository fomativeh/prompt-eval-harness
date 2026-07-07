from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from prompt_eval_harness.models import EvalRunArtifact, PromptRunSummary


def print_summary_table(prompt_summaries: list[PromptRunSummary]) -> None:
    table = Table(title="Prompt Eval Summary")
    table.add_column("Prompt")
    table.add_column("Category")
    table.add_column("Urgency")
    table.add_column("Format")
    table.add_column("Tone")
    table.add_column("Refusal")
    table.add_column("Overall")

    for summary in prompt_summaries:
        metrics = summary.metrics
        table.add_row(
            summary.prompt_name,
            f"{metrics.category_accuracy:.0%}",
            f"{metrics.urgency_accuracy:.0%}",
            f"{metrics.format_pass_rate:.0%}",
            f"{metrics.tone_accuracy:.0%}",
            f"{metrics.refusal_accuracy:.0%}",
            f"{metrics.overall_score:.0%}",
        )

    Console().print(table)


def save_run_artifact(artifact: EvalRunArtifact, results_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = run_dir / "artifact.json"
    artifact_payload = artifact.model_dump(mode="json")
    artifact_path.write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")
    return artifact_path
