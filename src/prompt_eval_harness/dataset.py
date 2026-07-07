from __future__ import annotations

import json
from pathlib import Path

from prompt_eval_harness.models import EvalCase


def load_dataset(dataset_path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cases.append(EvalCase.model_validate(json.loads(line)))
    return cases
