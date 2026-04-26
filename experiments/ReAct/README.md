# ReAct

This directory sets up `dspy.ReAct` for the same coding benchmarks used elsewhere in the repo:

- `SciCode`
- `LiveCodeBench`

The runner is aligned with the other experiment dirs:

- same benchmark loaders
- same hidden-test evaluator
- same grouped split logic
- same local model aliases
- a sweep wrapper for both local coding models

## Why ReAct Here

Official DSPy docs present `dspy.ReAct` as a tool-using agent loop that interleaves reasoning and tool calls before producing the final output. The official DSPy docs and cheatsheet show ReAct in exactly that role, even if they do not include a ready-made code-generation benchmark recipe.

Official references used:

- DSPy docs home: https://dspy.ai/
- DSPy cheatsheet `ReAct` example: https://github.com/stanfordnlp/dspy/blob/main/docs/docs/cheatsheet.md
- DSPy optimizer docs showing ReAct as a tool-augmented program: https://github.com/stanfordnlp/dspy/blob/main/docs/docs/index.md

## Benchmark-Guided Tool Design

I checked the official benchmark sources before choosing the toolset.

### LiveCodeBench

The official repository describes:

- code-generation evaluation with hidden tests
- `code_generation_lite` as the fast default
- a separate **self-repair** scenario

That strongly suggests that iterative correction is useful, but in our local setup we do not expose hidden tests during inference. So the toolset focuses on **static self-repair**:

- syntax checking
- contract checking
- I/O mode checking
- candidate-code repair hints

Source:

- https://github.com/LiveCodeBench/LiveCodeBench

### SciCode

The official repository describes SciCode as:

- realistic scientific coding problems
- decomposed into subproblems
- centered on numerical methods, simulation, and scientific calculation
- supplied with optional scientific background and dependency context

That suggests different tools matter:

- preserving the exact function header
- handling scientific imports correctly
- generating a scaffold from the benchmark-provided header and dependency block
- checking the structure of a candidate solution before finalizing it

Source:

- https://github.com/scicode-bench/SciCode

## Tooling Strategy

The toolset is built to target the biggest benchmark failure modes without leaking hidden tests.

Shared tools:

- `benchmark_playbook`
- `summarize_task_contract`
- `solution_contract_json`
- `extract_constraints_json`
- `extract_examples_json`
- `make_solution_scaffold`
- `make_stdio_solution_scaffold`
- `preview_solution_shape_json`
- `syntax_check_json`
- `repair_hints_json`
- `strip_markdown_fences_tool`

SciCode adds:

- `prepend_required_imports`

## Why These Tools

These tools are aimed at the concrete mistakes that frequently kill benchmark scores:

- wrong interface
- wrong execution mode (`stdin` vs function)
- missing use of sample I/O and prompt constraints
- bytes-vs-string bugs in stdin parsing
- syntax errors
- missing scientific imports
- forgetting to preserve the starter code contract
- failing to run a repair pass on a draft

The official LiveCodeBench self-repair support is the main reason the runner includes explicit static repair tools, even though we do not expose hidden tests. The official SciCode emphasis on decomposed scientific subproblems is the reason the runner includes contract and scaffold tools rather than generic retrieval.

## Commands

Preview tasks:

```bash
python experiments/ReAct/run_react.py preview \
  --benchmark scicode \
  --eval-size 2
```

Run a one-task SciCode local-sample smoke test:

```bash
python experiments/ReAct/run_react.py evaluate \
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
python experiments/ReAct/run_react.py evaluate \
  --benchmark scicode \
  --model devstral \
  --train-groups 24 \
  --val-groups 8 \
  --eval-groups 16
```

Run a grouped LiveCodeBench evaluation slice:

```bash
python experiments/ReAct/run_react.py evaluate \
  --benchmark livecodebench \
  --model rnj \
  --train-groups 40 \
  --val-groups 10 \
  --eval-groups 40
```

Run the full matrix sweep:

```bash
python experiments/ReAct/run_full_sweep.py
```

## Defaults

`SciCode` defaults:

- `max_iters = 6`
- `max_tokens = 3072`
- grouped split defaults `24 / 8 / 16`

`LiveCodeBench` defaults:

- `max_iters = 5`
- `max_tokens = 2048`
- grouped split defaults `40 / 10 / 40`

These are intended as reasonable starting points for a tool-using agent, not tuned settings.

## Outputs

Each `evaluate` run writes:

- `config.json`
- `split_manifest.json`
- `summary.json`
- `task_artifacts/<task_id>.json`

Each task artifact records:

- final solution
- full ReAct trajectory
- tool usage
- timing
- hidden-test result

Each sweep writes:

- `status.json`
- per-run logs
- `analytics/summary.csv`
- `analytics/summary.json`
- `analytics/report.md`

## Current Status

This setup is fully local and does **not** require Deno.

It is designed to use deterministic contract/self-repair tools to improve over a plain no-tool baseline, especially on:

- LiveCodeBench interface and format mistakes
- SciCode scientific import and function-signature mistakes

The next step after setup is to run the matrix sweep and compare scores directly against the non-agent baselines already collected elsewhere in the repo.
