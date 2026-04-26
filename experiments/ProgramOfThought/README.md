# ProgramOfThought

This directory sets up `dspy.ProgramOfThought` for the same coding benchmarks used elsewhere in the repo:

- `SciCode`
- `LiveCodeBench`

The setup follows the same basic pattern as the other experiment dirs:

- same benchmark loaders
- same hidden-test evaluator
- same grouped split logic
- same local model aliases
- a sweep wrapper for both models across both benchmarks

## What ProgramOfThought Means Here

`dspy.ProgramOfThought` is built for problems where the model writes Python code, executes it internally, and then uses the execution result to produce the final answer.

For code-generation benchmarks, the adaptation is:

1. The model writes internal scratch Python code.
2. That scratch code constructs a **candidate benchmark solution string**.
3. When ready, the scratch code calls:

```python
SUBMIT({"solution": candidate_solution_string})
```

4. The runner extracts that submitted `solution` and evaluates it on the benchmark hidden tests.

So PoT is not executing the benchmark candidate directly inside the DSPy sandbox. It is using internal execution as reasoning-time scaffolding to synthesize the final candidate program.

## Important Runtime Requirement

DSPy's `ProgramOfThought` uses DSPy's `PythonInterpreter`, which runs through a Deno-backed sandbox.

That means `deno` must be installed and on `PATH` before `evaluate` or `run_full_sweep.py` will work.

Install instructions:

- https://docs.deno.com/runtime/getting_started/installation/

The runner performs a preflight check and fails immediately if `deno` is missing.

## Files

- `run_pot.py`: preview and evaluate entrypoint
- `run_full_sweep.py`: matrix sweep wrapper across models and benchmarks
- `requirements.txt`: Python dependencies for this experiment

## Benchmark-Aware Setup

Each example passed into PoT includes:

- `benchmark`
- `eval_kind`
- `benchmark_notes`
- `tool_guide`
- `task_prompt`
- `starter_code`
- `required_dependencies`

This matters because the benchmark contracts differ.

### SciCode

The runner explicitly tells PoT that:

- hidden tests call the requested function directly
- the submitted solution must preserve the provided function header
- stdin handling is wrong for SciCode
- the dependency block may need to be included

### LiveCodeBench

The runner distinguishes:

- `stdin` tasks: submit a complete program that reads stdin and prints exact output
- `functional` tasks: preserve the starter-code interface exactly and avoid extra prints

## Helper Functions Inside the PoT Sandbox

The scratch Python program can call a small helper set:

- `benchmark_playbook`
- `solution_contract_json`
- `extract_function_header`
- `extract_function_names_json`
- `preview_solution_shape_json`
- `strip_markdown_fences_tool`

SciCode also gets:

- `extract_required_imports_block`
- `make_solution_scaffold`

These helpers are there to let the internal scratch code inspect the contract and shape candidate programs. They do **not** expose hidden tests.

## Commands

Preview tasks:

```bash
python experiments/ProgramOfThought/run_pot.py preview \
  --benchmark scicode \
  --eval-size 2
```

Run a one-task local-sample SciCode check after installing `deno`:

```bash
python experiments/ProgramOfThought/run_pot.py evaluate \
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
python experiments/ProgramOfThought/run_pot.py evaluate \
  --benchmark scicode \
  --model devstral \
  --train-groups 24 \
  --val-groups 8 \
  --eval-groups 16
```

Run a grouped LiveCodeBench evaluation slice:

```bash
python experiments/ProgramOfThought/run_pot.py evaluate \
  --benchmark livecodebench \
  --model rnj \
  --train-groups 40 \
  --val-groups 10 \
  --eval-groups 40
```

Run the full matrix sweep:

```bash
python experiments/ProgramOfThought/run_full_sweep.py
```

## Defaults

`SciCode` defaults:

- `max_iters = 4`
- `max_tokens = 3072`
- grouped split defaults `24 / 8 / 16`

`LiveCodeBench` defaults:

- `max_iters = 3`
- `max_tokens = 2048`
- grouped split defaults `40 / 10 / 40`

Prediction timeout default:

- `prediction_timeout_s = max(request_timeout_s * 3, 360)`

This caps a single PoT reasoning attempt so one hung scratch program cannot stall the whole sweep indefinitely.

These are intended as reasonable starting points, not tuned settings.

## Outputs

Each `evaluate` run writes:

- `config.json`
- `split_manifest.json`
- `summary.json`
- `task_artifacts/<task_id>.json`

Each task artifact records:

- final submitted solution
- solution source
- final scratch code
- sandbox code output
- per-hop attempt history
- timing
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

- `dspy.ProgramOfThought` exists in the installed DSPy package
- the runner compiles
- task preview works
- the Deno preflight fails cleanly when `deno` is missing

What I did **not** verify end-to-end yet:

- a real benchmark solve, because this VM currently does not have `deno` installed

So the experiment harness is ready, but the environment still needs Deno before benchmark runs can start.
