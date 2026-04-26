# Prompt Optimization

This folder sets up a minimal DSPy + GEPA workflow for the two local Ollama models already on the VM:

- `rnj` -> `rnj-1:latest`
- `devstral` / `devestral` -> `devstral-small-2:latest`

The code stays close to the DSPy GEPA tutorial pattern:

1. Build a small `dspy.ChainOfThought` program.
2. Define a benchmark metric that returns `score` and `feedback`.
3. Run `dspy.GEPA(...).compile(program, trainset=..., valset=...)`.
4. Save the optimized program and compare baseline vs optimized scores.

## Files

- `run_gepa.py`: CLI entrypoint for preview, evaluation, and optimization.
- `benchmark_adapters.py`: benchmark loading plus local evaluation code.
- `requirements.txt`: small dependency list for this experiment folder.

## Benchmarks

### LiveCodeBench

- Uses the local `benchmarks/livecodebench/code_generation_lite/` snapshot.
- Decodes the hidden tests from the dataset blob.
- Supports both stdin-style and LeetCode-style functional tasks.

### SciCode

- Uses the full Hugging Face dataset by default: `SciCode1/SciCode`.
- Uses the local `benchmarks/scicode/data/test_data.h5` bundle for hidden targets.
- Optimizes and evaluates at the SciCode sub-step level, not the full end-to-end main-problem harness.

## Important Scope Note

This setup is for local prompt iteration, not contamination-safe leaderboard claims.

- LiveCodeBench uses grouped temporal splits by contest date and contest id, so evaluation stays on newer held-out contests.
- SciCode uses grouped splits by `problem_id`, so train, val, and eval do not share sub-steps from the same parent problem.
- Hidden-test feedback is intentionally redacted to coarse signals such as `wrong_answer`, `timeout`, and `runtime_error`.

If you want quick debugging instead of clean splits, pass `--allow-shared-group-splits`. That flag is for local smoke tests only.

## Quick Start

Preview a few tasks:

```bash
python experiments/prompt_optimization/run_gepa.py preview \
  --benchmark livecodebench \
  --train-size 2 \
  --val-size 1 \
  --eval-size 1
```

Run a small LiveCodeBench GEPA sweep with `rnj`:

```bash
python experiments/prompt_optimization/run_gepa.py optimize \
  --benchmark livecodebench \
  --model rnj \
  --train-size 6 \
  --val-size 2 \
  --eval-size 2 \
  --max-metric-calls 24
```

Run a small SciCode sub-step sweep with `devstral`:

```bash
python experiments/prompt_optimization/run_gepa.py optimize \
  --benchmark scicode \
  --model devstral \
  --train-size 6 \
  --val-size 2 \
  --eval-size 2 \
  --scicode-source huggingface \
  --max-metric-calls 24
```

Evaluate a saved optimized program:

```bash
python experiments/prompt_optimization/run_gepa.py evaluate \
  --benchmark livecodebench \
  --model rnj \
  --train-size 0 \
  --val-size 0 \
  --eval-size 4 \
  --load-program experiments/prompt_optimization/artifacts/livecodebench/rnj/DATE/optimized_program
```

Run the full sequential sweep across both models and both benchmarks:

```bash
python experiments/prompt_optimization/run_full_sweep.py
```

Run an explicit subset with a fixed teacher override:

```bash
python experiments/prompt_optimization/run_full_sweep.py \
  --runs scicode:devstral livecodebench:rnj \
  --reflection-model qwen3-coder:30b
```

Launch the full sweep detached on the VM with `tmux`:

```bash
SWEEP_ROOT=experiments/prompt_optimization/artifacts/sweeps/$(date -u +%Y%m%dT%H%M%SZ)_full_matrix
mkdir -p "$SWEEP_ROOT"
SESSION=prompt_opt_$(basename "$SWEEP_ROOT" | cut -d_ -f1)
tmux new-session -d -s "$SESSION" \
  "cd /home/z4j/fly_away_code && /home/z4j/miniforge3/bin/python -u experiments/prompt_optimization/run_full_sweep.py --sweep-root $SWEEP_ROOT --continue-on-failure > $SWEEP_ROOT/nohup.log 2>&1"
```

The detached sweep writes:

- `status.json`: current run and completed runs
- `runs/...`: per-run GEPA artifacts
- `logs/...`: raw stdout/stderr for each optimize run
- `analytics/summary.csv`
- `analytics/summary.json`
- `analytics/report.md`
- `analytics/*_eval_comparison.csv`

Useful monitor commands:

```bash
cat "$SWEEP_ROOT/status.json"
tail -f "$SWEEP_ROOT/nohup.log"
tail -f "$SWEEP_ROOT/logs/scicode__rnj.log"
tmux attach -t "$SESSION"
```

## DSPy Notes

This setup uses the local Ollama form shown in the DSPy docs:

```python
lm = dspy.LM("ollama_chat/rnj-1:latest", api_base="http://localhost:11434", api_key="")
```

GEPA is configured in the same style as the tutorial:

```python
optimizer = dspy.GEPA(metric=metric, auto="light", ...)
optimized = optimizer.compile(program, trainset=trainset, valset=valset)
```

Use exactly one of `--gepa-auto`, `--max-full-evals`, or `--max-metric-calls`.
If you omit all three, the runner defaults to `--max-metric-calls 48`.

The optimize command now defaults to the other local Ollama model as the reflection LM:

```python
# rnj student -> devstral reflection
# devstral student -> rnj reflection
dspy.LM(
    "devstral" or "rnj",
    api_base="http://localhost:11434",
    api_key="",
)
```

Local cross-reflection through Ollama now uses a longer default request timeout automatically:

```bash
--reflection-request-timeout-s 120
```

If you explicitly point reflection at an OpenAI-compatible remote model, reflection requests fail fast by default with:

```bash
--reflection-request-timeout-s 15 --reflection-num-retries 0
```

For OpenAI-compatible reflection models, `optimize` also runs a tiny preflight chat request before GEPA starts so dead teacher endpoints fail before the benchmark run begins.

If you want to override the local cross-reflection setup, you can still set any teacher model explicitly. For example:

```bash
--reflection-model nemotron-30b-fp8 --reflection-api-base http://earlsinclair.ornl.gov:8201/v1 --reflection-api-key unused
```

The default detached sweep uses these clean held-out settings:

- LiveCodeBench: `train_groups=40`, `val_groups=10`, `eval_groups=40`, `max_full_evals=2`
- SciCode: `train_groups=24`, `val_groups=8`, `eval_groups=16`, `max_full_evals=2`

The detached sweep uses group-count splits rather than flattened task-count splits so the optimization budget is defined in benchmark-native units:

- LiveCodeBench groups are `(contest_date, contest_id)`
- SciCode groups are `problem_id`

Artifacts are written under `experiments/prompt_optimization/artifacts/`.
