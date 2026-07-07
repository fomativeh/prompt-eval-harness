from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        default=str(ROOT / "results" / "runs" / "20260707_101729" / "artifact.json"),
    )
    parser.add_argument(
        "--baseline",
        default=str(ROOT / "data" / "baseline_artifact.json"),
    )
    parser.add_argument("--prompt", default="prompt_v2")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_prompt_summary(artifact: dict, prompt_name: str) -> dict:
    try:
        return next(item for item in artifact["prompt_summaries"] if item["prompt_name"] == prompt_name)
    except StopIteration as error:
        raise ValueError(f"Prompt '{prompt_name}' not found in artifact {artifact}") from error


def compare_metrics(baseline_summary: dict, candidate_summary: dict) -> list[str]:
    failures: list[str] = []
    for metric_name, baseline_value in baseline_summary["metrics"].items():
        candidate_value = candidate_summary["metrics"][metric_name]
        if candidate_value < baseline_value:
            failures.append(
                f"metric {metric_name}: candidate={candidate_value:.3f} baseline={baseline_value:.3f}"
            )
    return failures


def compare_case_fields(baseline_summary: dict, candidate_summary: dict) -> list[str]:
    failures: list[str] = []
    candidate_cases = {item["case_id"]: item for item in candidate_summary["case_results"]}

    for baseline_case in baseline_summary["case_results"]:
        case_id = baseline_case["case_id"]
        if case_id not in candidate_cases:
            failures.append(f"missing case result for {case_id}")
            continue
        candidate_case = candidate_cases[case_id]
        for field_name in ["passed", "format_correct", "refusal_correct", "tone_correct"]:
            if candidate_case[field_name] != baseline_case[field_name]:
                failures.append(
                    f"case {case_id} field {field_name}: candidate={candidate_case[field_name]} baseline={baseline_case[field_name]}"
                )

    return failures


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact)
    baseline_path = Path(args.baseline)
    artifact = load_json(artifact_path)
    baseline = load_json(baseline_path)
    baseline_summary = get_prompt_summary(baseline, args.prompt)
    candidate_summary = get_prompt_summary(artifact, args.prompt)

    failures = compare_metrics(baseline_summary, candidate_summary)
    failures.extend(compare_case_fields(baseline_summary, candidate_summary))

    if failures:
        print("Regression detected:")
        for item in failures:
            print(f"- {item}")
        sys.exit(1)

    print(
        f"Regression check passed for {args.prompt} against baseline {baseline_path} using artifact {artifact_path}."
    )


if __name__ == "__main__":
    main()
