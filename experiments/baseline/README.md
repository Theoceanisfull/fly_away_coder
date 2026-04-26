# Baseline

This directory sets up a plain single-pass benchmark runner for the same coding benchmarks used elsewhere in the repo:

- `SciCode`
- `LiveCodeBench`

The goal is to provide the cleanest local single-model reference point:

- no tools
- no optimization
- no self-repair loop
- no multi-agent collaboration

The implementation is intentionally aligned with the other experiment directories:

- same benchmark loaders
- same hidden-test evaluator
- same grouped split logic
- same local model aliases
- same sweep/status/analytics layout

## What This Baseline Is

The runner issues one model call per task using a simple DSPy `ChainOfThought` code-generation signature:

- inputs: `task_prompt`, `starter_code`
- output: executable Python 3 code only

This is the standalone version of the repo's plain single-pass code-generation baseline. It is meant to answer the question: how well do the local models perform on these benchmarks before adding agents, tools, reflection, or optimization?

## Files

- `run_baseline.py`: preview and evaluate entrypoint
- `run_full_sweep.py`: matrix sweep wrapper across models and benchmarks
- `requirements.txt`: Python dependencies for this experiment

## Commands

Preview tasks:

```bash
python experiments/baseline/run_baseline.py preview \
  --benchmark scicode \
  --eval-size 2
```

Run a one-task SciCode local-sample smoke test:

```bash
python experiments/baseline/run_baseline.py evaluate \
  --benchmark scicode \
  --model rnj \
  --scicode-source local_sample \
  --allow-shared-group-splits \
  --train-size 0 \
  --val-size 0 \
  --eval-size 1
```

Run a grouped SciCode evaluation slice:

```bash
python experiments/baseline/run_baseline.py evaluate \
  --benchmark scicode \
  --model devstral \
  --train-groups 24 \
  --val-groups 8 \
  --eval-groups 16
```

Run a grouped LiveCodeBench evaluation slice:

```bash
python experiments/baseline/run_baseline.py evaluate \
  --benchmark livecodebench \
  --model rnj \
  --train-groups 40 \
  --val-groups 10 \
  --eval-groups 40
```

Run the full matrix sweep:

```bash
python experiments/baseline/run_full_sweep.py
```

## Defaults

`SciCode` defaults:

- `max_tokens = 3072`
- grouped split defaults `24 / 8 / 16`

`LiveCodeBench` defaults:

- `max_tokens = 2048`
- grouped split defaults `40 / 10 / 40`

Shared runtime defaults:

- `temperature = 0.0`
- `timeout_s = 12.0`
- `num_retries = 3`
- Ollama/OpenAI-compatible endpoint at `http://localhost:11434`

## Outputs

Each `evaluate` run writes:

- `config.json`
- `split_manifest.json`
- `summary.json`
- `task_artifacts/<task_id>.json`

Each task artifact records:

- final solution
- model reasoning text when available
- timing
- hidden-test result

Each sweep writes:

- `status.json`
- per-run logs
- `analytics/summary.csv`
- `analytics/summary.json`
- `analytics/report.md`

## Notes

- The saved train/validation splits exist to keep the partitioning aligned with the other experiments. The baseline runner evaluates only the held-out `eval` split.
- If you want exact apples-to-apples comparison against an older run with different token budgets, override `--max-tokens` explicitly.
