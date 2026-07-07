from __future__ import annotations

import hashlib
from pathlib import Path

from prompt_eval_harness.dataset import load_dataset
from prompt_eval_harness.models import DatasetMetadata, EvalRunArtifact, PromptRunSummary
from prompt_eval_harness.providers import get_provider
from prompt_eval_harness.reporting import print_summary_table, save_run_artifact
from prompt_eval_harness.scoring import compute_metrics, score_case


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def run_evaluation(
    dataset_path: Path,
    prompts_dir: Path,
    results_dir: Path,
    provider_name: str,
    model_name: str,
) -> Path:
    provider = get_provider(provider_name)
    cases = load_dataset(dataset_path)
    dataset_text = dataset_path.read_text(encoding="utf-8")
    prompt_summaries: list[PromptRunSummary] = []

    for prompt_path in sorted(prompts_dir.glob("*.txt")):
        prompt_text = prompt_path.read_text(encoding="utf-8")
        case_results = []
        for case in cases:
            output_text = provider.generate(prompt_text=prompt_text, case=case, model=model_name)
            case_results.append(score_case(case=case, prompt_name=prompt_path.stem, output_text=output_text))
        prompt_summaries.append(
            PromptRunSummary(
                prompt_name=prompt_path.stem,
                prompt_path=prompt_path,
                prompt_file_name=prompt_path.name,
                prompt_sha256=sha256_text(prompt_text),
                prompt_text=prompt_text,
                metrics=compute_metrics(case_results),
                case_results=case_results,
            )
        )

    artifact = EvalRunArtifact(
        provider=provider_name,
        model=model_name,
        dataset=DatasetMetadata(
            dataset_path=dataset_path,
            dataset_file_name=dataset_path.name,
            dataset_sha256=sha256_text(dataset_text),
            case_count=len(cases),
        ),
        prompt_summaries=prompt_summaries,
    )
    print_summary_table(prompt_summaries)
    return save_run_artifact(artifact, results_dir)
