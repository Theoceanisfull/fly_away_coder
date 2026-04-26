# CodeAct

This directory sets up a benchmark runner for `dspy.CodeAct` on the same coding benchmarks already used elsewhere in the repo:

- `SciCode`
- `LiveCodeBench`

The runner is intentionally aligned with `experiments/prompt_optimization`:

- same benchmark loaders
- same hidden-test evaluator
- same grouped split logic
- same local model aliases

## Why This Exists

`dspy.CodeAct` is a tool-using code agent. For these benchmarks, the most reasonable use is not to expose hidden tests or external retrieval, but to give the agent:

- a Python scratchpad loop
- a small set of benchmark-aware helper tools
- a benchmark-specific contract note

That lets the agent inspect the task structure, preserve the right interface, and synthesize a final code answer while staying compatible with the existing evaluation setup.

## Official DSPy References

I checked the official DSPy docs before wiring this up:

- DSPy docs home for local/openai-compatible LM setup: https://dspy.ai/
- Official DSPy cheatsheet example for `dspy.CodeAct`: https://github.com/stanfordnlp/dspy/blob/main/docs/docs/cheatsheet.md

The docs currently show a minimal `CodeAct("n->factorial", tools=[factorial])` example, but not a full benchmark-oriented code-generation recipe. This implementation therefore uses:

- the official docs for the public API shape
- the installed DSPy source in this environment for the concrete runtime behavior

## Important Runtime Requirement

DSPy's `CodeAct` uses DSPy's `PythonInterpreter`, which runs through a Deno-backed sandbox.

That means `deno` must be installed and on `PATH` before `evaluate` or `run_full_sweep.py` will work.

Install instructions:

- https://docs.deno.com/runtime/getting_started/installation/

The runner performs a preflight check and fails early with a clear error if `deno` is missing.

## Files

- `run_codeact.py`: preview and evaluate entrypoint
- `run_full_sweep.py`: matrix sweep wrapper across models and benchmarks
- `requirements.txt`: Python dependencies for this experiment

## Benchmark-Aware Setup

The runner feeds `CodeAct` more structure than a plain `task_prompt`.

Each example includes:

- `benchmark`
- `eval_kind`
- `benchmark_notes`
- `task_prompt`
- `starter_code`
- `required_dependencies`

### SciCode

The defaults assume:

- more iterations than LiveCodeBench
- larger token budget
- function-only solving
- dependency-awareness through `required_dependencies`

The agent is explicitly reminded that:

- hidden tests call the requested function directly
- stdin handling is wrong for SciCode
- the provided function header is authoritative

### LiveCodeBench

The runner distinguishes:

- `stdin` tasks: produce a complete program that reads stdin and prints exact output
- `functional` tasks: preserve the provided function interface and avoid extra prints

This matters because a single generic "write Python code" instruction is not enough to reliably preserve the benchmark contract.

## Tooling Strategy

The toolset is deliberately small.

Shared tools:

- `benchmark_playbook`
- `solution_contract`
- `extract_function_header`
- `extract_function_names`
- `preview_solution_shape`
- `strip_markdown_fences_tool`

SciCode adds:

- `extract_required_imports`
- `make_solution_scaffold`

These tools are meant to help the agent inspect the problem contract and self-check solution shape. They do not expose hidden tests.

## Commands

Preview tasks:

```bash
python experiments/CodeAct/run_codeact.py preview \
  --benchmark scicode \
  --eval-size 2
```

Run a one-task local-sample SciCode check after installing `deno`:

```bash
python experiments/CodeAct/run_codeact.py evaluate \
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
python experiments/CodeAct/run_codeact.py evaluate \
  --benchmark scicode \
  --model devstral \
  --train-groups 24 \
  --val-groups 8 \
  --eval-groups 16
```

Run a grouped LiveCodeBench evaluation slice:

```bash
python experiments/CodeAct/run_codeact.py evaluate \
  --benchmark livecodebench \
  --model rnj \
  --train-groups 40 \
  --val-groups 10 \
  --eval-groups 40
```

Run the full matrix sweep:

```bash
python experiments/CodeAct/run_full_sweep.py
```

## Defaults

Benchmark-specific defaults are intentionally different.

`SciCode`:

- `max_iters = 6`
- `max_tokens = 3072`
- grouped split defaults `24 / 8 / 16`

`LiveCodeBench`:

- `max_iters = 4`
- `max_tokens = 2048`
- grouped split defaults `40 / 10 / 40`

These are only starting points. They are not tuned yet.

## Outputs

Each `evaluate` run writes:

- `config.json`
- `split_manifest.json`
- `summary.json`
- `task_artifacts/<task_id>.json`

Each task artifact includes:

- final solution
- solution recovery source
- full trajectory
- per-task timing
- tool usage
- hidden-test result

Each sweep writes:

- `status.json`
- per-run logs
- `analytics/summary.csv`
- `analytics/summary.json`
- `analytics/report.md`

## Current Status

This setup is implemented and wired to the benchmark stack.

What I verified locally:

- `dspy.CodeAct` exists in the installed DSPy package
- the runner shape matches the benchmark adapters already in the repo
- the environment currently does **not** have `deno` installed

What I did **not** verify end-to-end yet:

- an actual `CodeAct` benchmark solve, because the local runtime prerequisite (`deno`) is currently missing

So this is a complete experiment setup, but not yet a completed benchmark run environment.
