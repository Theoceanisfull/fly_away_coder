# Experiment Baseline Results

## Title

Plain single-pass local-model baseline on SciCode and LiveCodeBench using the shared benchmark stack and held-out grouped splits.

## Executive Summary

This report isolates the **baseline-only** results for the four benchmark/model runs evaluated on the repo's standard held-out splits:

- `scicode` + `rnj`
- `scicode` + `devstral`
- `livecodebench` + `rnj`
- `livecodebench` + `devstral`

The objective was to establish a clean reference point before adding optimization, tools, or multi-agent coordination. In this baseline setting, each task receives a **single model call** that returns executable Python code. There is:

- no prompt optimization
- no tool use
- no iterative repair loop
- no candidate reranking
- no multi-agent interaction

All four baseline evaluations completed successfully. The strongest baseline model on both benchmarks was `devstral`:

- **SciCode + `devstral`**: mean score **0.2305**, **20/94** solved
- **LiveCodeBench + `devstral`**: mean score **0.2011**, **34/175** solved

The `rnj` baseline was consistently faster but less accurate:

- **SciCode + `rnj`**: mean score **0.1640**, **13/94** solved
- **LiveCodeBench + `rnj`**: mean score **0.1373**, **22/175** solved

These numbers are the plain baseline metrics that should be used as the direct comparison point for the later CodeAct, ReAct, Program-of-Thought, and twin-agent experiments.

## Experimental Scope

This report uses the baseline artifacts already saved inside the completed sweep:

- `experiments/prompt_optimization/artifacts/sweeps/20260420T011758Z_full_matrix/status.json`
- `experiments/prompt_optimization/artifacts/sweeps/20260420T011758Z_full_matrix/analytics/summary.csv`
- `experiments/prompt_optimization/artifacts/sweeps/20260420T011758Z_full_matrix/analytics/summary.json`
- `experiments/prompt_optimization/artifacts/sweeps/20260420T011758Z_full_matrix/runs/scicode/rnj/baseline_eval.json`
- `experiments/prompt_optimization/artifacts/sweeps/20260420T011758Z_full_matrix/runs/scicode/devstral/baseline_eval.json`
- `experiments/prompt_optimization/artifacts/sweeps/20260420T011758Z_full_matrix/runs/livecodebench/rnj/baseline_eval.json`
- `experiments/prompt_optimization/artifacts/sweeps/20260420T011758Z_full_matrix/runs/livecodebench/devstral/baseline_eval.json`
- the corresponding `run_stats.json` files for timing and split metadata

Although these baseline artifacts were logged during a larger sweep, this report ignores all optimized outputs and discusses only the saved pre-optimization baseline evaluations.

## Methods

### Model and Runtime Setup

The baseline used the same local model aliases as the other experiments:

- `rnj` -> `rnj-1:latest`
- `devstral` -> `devstral-small-2:latest`

The baseline solver is the repo's minimal single-pass code generator:

- DSPy `ChainOfThought` over a simple code-generation signature
- inputs: `task_prompt`, `starter_code`
- output: executable Python 3 code only

Shared runtime settings were:

- `api_base = http://localhost:11434`
- `temperature = 0.0`
- `max_tokens = 2048`
- `timeout_s = 12.0`
- `num_retries = 3`
- `seed = 0`

### What Makes This a Baseline

Each held-out task is solved with exactly one model prediction followed by hidden-test evaluation. There is no:

- optimizer
- reflection model
- tool loop
- scratchpad execution environment
- self-repair stage
- second candidate or selection mechanism

This is therefore the cleanest local single-model reference in the current benchmark stack.

### Benchmarks and Split Construction

The baseline uses the same loaders, hidden-test evaluators, and grouped split construction as the other experiments. Train and validation splits were created only to preserve the same partitioning scheme; the baseline itself evaluated only the held-out `eval` split.

SciCode configuration:

- source: Hugging Face `SciCode1/SciCode`
- split: `test`
- grouped by `problem_id`
- groups: `train=24`, `val=8`, `eval=16`
- resulting evaluation examples: `94`

LiveCodeBench configuration:

- local dataset release: `release_latest`
- grouped by `(contest_date, contest_id)`
- groups: `train=40`, `val=10`, `eval=40`
- resulting evaluation examples: `175`

For both benchmarks, the per-example score is:

- `score = passed_hidden_tests / total_hidden_tests`

## Main Results

### Run-Level Outcomes

| Benchmark | Model | Eval examples | Mean score | Solved | Partial | Zero-score | Avg problem time (s) | Baseline eval duration (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SciCode | `rnj` | 94 | 0.1640 | 13 | 8 | 73 | 7.61 | 715.53 |
| SciCode | `devstral` | 94 | 0.2305 | 20 | 4 | 70 | 19.97 | 1877.63 |
| LiveCodeBench | `rnj` | 175 | 0.1373 | 22 | 13 | 140 | 9.04 | 1581.44 |
| LiveCodeBench | `devstral` | 175 | 0.2011 | 34 | 15 | 126 | 26.84 | 4696.79 |

### Benchmark-Level Averages

| Benchmark | Runs | Avg baseline mean |
| --- | ---: | ---: |
| SciCode | 2 | 0.1973 |
| LiveCodeBench | 2 | 0.1692 |

### Main Observations

1. `devstral` was the stronger baseline model on both benchmarks.
   On SciCode it outscored `rnj` by **0.0665** mean-score points and solved **7** more tasks perfectly. On LiveCodeBench it outscored `rnj` by **0.0639** and solved **12** more tasks perfectly.

2. `rnj` was materially faster.
   Average total time per problem was about **7.61 s** versus **19.97 s** on SciCode, and **9.04 s** versus **26.84 s** on LiveCodeBench.

3. Both benchmarks remained dominated by zero-score examples in the plain baseline setting.
   Even the strongest baseline left **70/94** SciCode examples and **126/175** LiveCodeBench examples at zero score.

4. The strongest overall baseline result in the repo was `devstral` on LiveCodeBench.
   That run reached **0.2011** mean score with **34** fully solved tasks.

## Interpretation

This baseline establishes the reference point for all later comparisons in the repo:

1. A plain single-pass local model can already solve a non-trivial subset of held-out tasks on both benchmarks.
2. `devstral` is the stronger accuracy baseline, while `rnj` is the faster latency baseline.
3. The remaining error mass is large enough that later gains from tools, optimization, or collaboration are meaningful only when reported against these plain baseline numbers.

For paper-style positioning, the concise summary is:

> The repo's baseline is a one-shot code-generation setting with no tools, no optimization, and no collaboration. On the shared held-out splits, `devstral` provides the strongest plain-model baseline on both SciCode and LiveCodeBench, while `rnj` provides a faster but weaker reference point.
