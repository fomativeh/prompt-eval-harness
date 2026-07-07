# Prompt Eval Harness

Python harness for evaluating prompt variants against a labeled customer-support triage dataset.

It stays focused on the evaluation loop itself: a labeled task, prompt comparison, deterministic scoring, saved artifacts, and a regression check. The repo is runnable with the included mock provider, and there is also an optional OpenAI-backed path for live model calls.

## Task

Given a short customer message, the model must return JSON with:

- `category`: one of `billing`, `technical`, `account_access`, `sales`, `other`
- `urgency`: one of `low`, `medium`, `high`
- `should_refuse`: boolean
- `reply_tone`: one of `empathetic`, `neutral`, `firm`
- `response`: short customer-facing reply

The harness evaluates two prompt variants on the same labeled dataset.

## Scope

This version includes the harness, an offline labeled dataset, prompt files, saved run artifacts, and a regression workflow against a committed baseline artifact.

### Included

- A labeled dataset with 24 customer-support cases covering straightforward requests, ambiguous cases, adversarial attempts, and refusal-required requests
- Two prompt variants evaluated side by side
- Deterministic scoring for category, urgency, format, tone, and refusal behavior
- Saved run artifacts with prompt-level summaries and case-level outputs
- A regression script that compares a selected artifact against a committed baseline artifact

### Non-goals

- A web UI or dashboard
- A database, hosted observability stack, or experiment tracker
- Judge-model scoring
- CI wiring for the regression script
- A large benchmark corpus
- Latency or token-cost analytics

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python scripts\run_eval.py --provider mock
python scripts\validate_dataset.py
python scripts\check_regression.py --artifact results\runs\20260707_101729\artifact.json --baseline data\baseline_artifact.json --prompt prompt_v2
```

To run against an API-backed model, set `OPENAI_API_KEY` in `.env` and use:

```bash
python scripts\run_eval.py --provider openai --model gpt-4o-mini
```

## Project layout

- `data/` labeled dataset and committed baseline artifact
- `prompts/` prompt variants
- `src/prompt_eval_harness/` harness code
- `results/` local eval runs
- `scripts/` runnable entry points

## Regression workflow

- Check a specific saved artifact against the committed baseline with:

```bash
python scripts\check_regression.py --artifact results\runs\20260707_101729\artifact.json --baseline data\baseline_artifact.json --prompt prompt_v2
```

- The regression script compares the selected prompt's aggregate metrics and the case-level `passed`, `format_correct`, `refusal_correct`, and `tone_correct` fields.

## Dataset validation

- Validate dataset structure and coverage sanity with:

```bash
python scripts\validate_dataset.py
```

- The dataset is intentionally small but covers a diverse set of cases.

## Notes

- The included `mock` provider makes the project runnable without paid API access.
- The committed baseline artifact provides a known-good prompt_v2 reference in the repo.
