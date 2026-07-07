from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prompt_eval_harness.dataset import load_dataset


def main() -> None:
    dataset_path = ROOT / "data" / "support_eval_dataset.jsonl"
    cases = load_dataset(dataset_path)

    id_counts = Counter(case.id for case in cases)
    duplicate_ids = sorted(case_id for case_id, count in id_counts.items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"Duplicate case IDs found: {', '.join(duplicate_ids)}")

    category_counts = Counter(case.expected_category for case in cases)
    urgency_counts = Counter(case.expected_urgency for case in cases)
    tone_counts = Counter(case.expected_reply_tone for case in cases)
    refusal_counts = Counter(case.expected_should_refuse for case in cases)
    tag_counts = Counter(tag for case in cases for tag in case.tags)

    print(f"Dataset path: {dataset_path}")
    print(f"Total cases: {len(cases)}")
    print("Duplicate case IDs: none")
    print(f"Category counts: {dict(sorted(category_counts.items()))}")
    print(f"Urgency counts: {dict(sorted(urgency_counts.items()))}")
    print(f"Tone counts: {dict(sorted(tone_counts.items()))}")
    print(f"Refusal counts: {dict(sorted(refusal_counts.items(), key=lambda item: str(item[0])))}")
    print(f"Tag distribution: {dict(sorted(tag_counts.items()))}")


if __name__ == "__main__":
    main()
