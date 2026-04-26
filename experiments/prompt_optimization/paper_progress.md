# Prompt Optimization Paper Progress

Last updated: 2026-04-22

## Scope

This note tracks the held-out GEPA prompt-optimization sweeps in this repo, with an emphasis on:

- which local teacher/reflection model was used
- how the GEPA budget behaves in this implementation
- what artifacts are available for later paper tables and figures
- how much held-out accuracy changed for each benchmark/model pair

The current focus is the dual benchmark setup used in:

- `scicode:devstral`
- `livecodebench:rnj`

with fixed held-out grouped splits and `max_full_evals=2`.

## Local Reflection Models

On 2026-04-21, the following local Ollama models were confirmed on the VM:

- `qwen3-coder:30b`
- `gpt-oss:20b`
- `rnj-1:latest`
- `devstral-small-2:latest`

For this paper thread, the most relevant teacher/reflection LMs are:

- `qwen3-coder:30b`
- `gpt-oss:20b`

## Important GEPA Mechanics In This Repo

This setup does not behave like a classic population-based evolutionary loop with a small number of named "generations."

Instead:

- DSPy's `GEPA(max_full_evals=2)` is converted into a metric-call budget over `train + val`.
- In this repo, `max_full_evals=2` means:
  - SciCode: `2 * (96 train + 43 val) = 278` metric calls
  - LiveCodeBench: `2 * (159 train + 44 val) = 406` metric calls
- GEPA then runs an iterative loop until that budget is exhausted.
- A normal reflective iteration evaluates:
  - the current candidate on a 3-example train minibatch
  - the proposed candidate on the same 3-example minibatch
- If the proposed candidate strictly improves the minibatch score, GEPA performs a full validation evaluation and adds the candidate to the pool.

This means the meaningful counters to track are:

- loop iterations
- accepted new candidates
- full validation evaluations
- total metric calls

not just `max_full_evals`.

## Pareto Front Interpretation

The current DSPy/GEPA stack is using an instance-level frontier, not a single scalar multi-objective frontier.

Operationally:

- the validation set is the tracking set for the frontier
- for each validation example, GEPA stores the best score seen so far
- if a new candidate beats that score on a validation example, it replaces the frontier winner for that example
- if it ties, it is added to that example's frontier set
- candidate selection then samples from the non-dominated survivors of these per-instance frontier memberships

The final optimized program still collapses to a single returned candidate by highest average validation score, even though the search process keeps a per-instance frontier archive internally.

## Completed Reference Sweep: Qwen Teacher

Reference artifacts:

- `experiments/prompt_optimization/artifacts/sweeps/20260421T161804Z_qwen3_teacher_dual/status.json`
- `experiments/prompt_optimization/artifacts/sweeps/20260421T161804Z_qwen3_teacher_dual/analytics/summary.json`

Teacher/reflection LM:

- `qwen3-coder:30b`

Held-out outcomes:

| Benchmark | Student | Baseline Mean | Optimized Mean | Delta | Baseline Solved | Optimized Solved | Delta Solved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SciCode | `devstral` | 0.2305 | 0.2571 | +0.0266 | 20 | 21 | +1 |
| LiveCodeBench | `rnj` | 0.1373 | 0.1474 | +0.0101 | 22 | 24 | +2 |

Recovered GEPA counters from the saved optimizer state:

| Benchmark | Iterations | Accepted New Candidates | Full Val Evals | Total Metric Calls |
| --- | ---: | ---: | ---: | ---: |
| SciCode | 22 | 3 | 4 | 304 |
| LiveCodeBench | 24 | 5 | 6 | 408 |

Interpretation:

- Qwen produced measurable held-out gains on both benchmarks.
- The gains were larger on SciCode than on LiveCodeBench in this dual-run setup.
- The run also shows why `max_full_evals=2` should not be described as "2 generations" in the paper.

## Completed Replication Sweep: GPT-OSS Teacher

Reference artifacts:

- `experiments/prompt_optimization/artifacts/sweeps/20260421T233100Z_gpt_oss20b_teacher_dual/status.json`
- `experiments/prompt_optimization/artifacts/sweeps/20260421T233100Z_gpt_oss20b_teacher_dual/analytics/summary.json`

Sweep root:
- `experiments/prompt_optimization/artifacts/sweeps/20260421T233100Z_gpt_oss20b_teacher_dual`

Teacher/reflection LM:

- `gpt-oss:20b`

Replicated run set:

- `scicode:devstral`
- `livecodebench:rnj`

Command:

```bash
SWEEP_ROOT=experiments/prompt_optimization/artifacts/sweeps/20260421T233100Z_gpt_oss20b_teacher_dual
SESSION=prompt_opt_20260421T233100Z
tmux new-session -d -s "$SESSION" \
  "cd /home/z4j/fly_away_code && /home/z4j/miniforge3/bin/python -u \
   experiments/prompt_optimization/run_full_sweep.py \
   --sweep-root $SWEEP_ROOT \
   --continue-on-failure \
   --runs scicode:devstral livecodebench:rnj \
   --reflection-model gpt-oss:20b \
   --reflection-api-base http://localhost:11434 \
   --reflection-api-key '' \
   --reflection-num-retries 0 > $SWEEP_ROOT/nohup.log 2>&1"
```

Run notes:

- GPT-OSS is available locally through Ollama on `localhost:11434`.
- An earlier foreground attempt at `20260421T231114Z_gpt_oss20b_teacher_dual` was interrupted and should be ignored for reporting.
- The detached replacement run is the authoritative GPT-OSS sweep for this note.
- The authoritative sweep finished at `2026-04-22T02:11:24Z` with overall status `completed`.

Results table:

| Benchmark | Student | Baseline Mean | Optimized Mean | Delta | Baseline Solved | Optimized Solved | Delta Solved | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SciCode | `devstral` | 0.2305 | 0.2429 | +0.0124 | 20 | 20 | 0 | mean improved; solved count stayed flat |
| LiveCodeBench | `rnj` | 0.1373 | 0.1450 | +0.0077 | 22 | 21 | -1 | mean improved; solved count dropped by 1 |

Recovered GEPA counters from the sweep logs:

| Benchmark | Iterations | Accepted New Candidates | Full Val Evals | Total Metric Calls |
| --- | ---: | ---: | ---: | ---: |
| SciCode | 21 | 3 | 4 | 298 |
| LiveCodeBench | 19 | 6 | 7 | 422 |

Interpretation:

- GPT-OSS improved held-out mean score on both benchmark/model pairs.
- GPT-OSS did not improve solved-count outcomes in this replication sweep: SciCode stayed flat and LiveCodeBench regressed by one solved problem.
- Relative to the Qwen teacher run above, GPT-OSS produced smaller gains on both benchmarks and weaker solved-count movement.
- The recovered metric-call totals again exceed the nominal `max_full_evals=2` budgets (`278` for SciCode and `406` for LiveCodeBench), because accepted candidates trigger additional full validation passes.

## Draft Paper Framing

Candidate framing for the write-up:

1. GEPA improved held-out benchmark performance in the local coding-model setting, but the gains were teacher-dependent.
2. In this implementation, the optimization budget is better described as metric calls or accepted validation passes than as "generations."
3. The search process maintains an instance-level Pareto archive over validation examples, while final reporting still uses a single scalar best program.
4. Teacher quality appears to affect not only final accuracy but also the style of instruction proposals, acceptance rate, and optimization latency.
