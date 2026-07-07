from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prompt_eval_harness.harness import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["mock", "openai"], default="mock")
    parser.add_argument("--model", default="gpt-4o-mini")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_path = run_evaluation(
        dataset_path=ROOT / "data" / "support_eval_dataset.jsonl",
        prompts_dir=ROOT / "prompts",
        results_dir=ROOT / "results",
        provider_name=args.provider,
        model_name=args.model,
    )
    print(f"Saved run artifact to {artifact_path}")


if __name__ == "__main__":
    main()
