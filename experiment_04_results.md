# Experiment 04 Results

## Title

ReAct benchmark sweep on SciCode and LiveCodeBench using benchmark-aware static analysis and self-repair tools with local Ollama coding models.

## Executive Summary

This report records the completed ReAct sweep in:

`experiments/serial_runs/20260425T203855Z_react_then_pot/react_sweep/`

The sweep started at **2026-04-25 20:38:55 UTC** and finished at **2026-04-26 06:11:50 UTC** (**2026-04-25 04:38:55 PM EDT** to **2026-04-26 02:11:50 AM EDT**). All four benchmark/model runs completed successfully with **exit code 0**.

Held-out outcomes were:

- **SciCode + `rnj`**: mean score **0.1764**, **12/94** solved, **8** partial-credit examples
- **SciCode + `devstral`**: mean score **0.1871**, **15/94** solved, **6** partial-credit examples
- **LiveCodeBench + `rnj`**: mean score **0.1647**, **26/175** solved, **23** partial-credit examples
- **LiveCodeBench + `devstral`**: mean score **0.2119**, **34/175** solved, **26** partial-credit examples

The overall picture is mixed. ReAct produced the strongest completed single-agent result currently recorded for:

- `scicode` + `rnj`
- `livecodebench` + `devstral`

It also stayed close to the earlier CodeAct result on `livecodebench` + `rnj`, but it materially regressed on `scicode` + `devstral` relative to baseline, GEPA, and CodeAct.

Operationally, ReAct was the slowest completed single-agent method in the repo so far. Average per-example total time ranged from **45.39 s** to **95.21 s**, and average agent depth ranged from **2.85** to **4.66** steps.

## Experimental Scope

Authoritative artifacts for this report are:

- `experiments/serial_runs/20260425T203855Z_react_then_pot/react_sweep/status.json`
- `experiments/serial_runs/20260425T203855Z_react_then_pot/react_sweep/sweep_config.json`
- `experiments/serial_runs/20260425T203855Z_react_then_pot/react_sweep/analytics/summary.json`
- `experiments/serial_runs/20260425T203855Z_react_then_pot/react_sweep/analytics/report.md`
- `experiments/serial_runs/20260425T203855Z_react_then_pot/react_sweep/logs/scicode__rnj.log`
- `experiments/serial_runs/20260425T203855Z_react_then_pot/react_sweep/logs/scicode__devstral.log`
- `experiments/serial_runs/20260425T203855Z_react_then_pot/react_sweep/logs/livecodebench__rnj.log`
- `experiments/serial_runs/20260425T203855Z_react_then_pot/react_sweep/logs/livecodebench__devstral.log`

Comparisons in this report use the earlier completed experiment summaries:

- `experiment_baseline_results.md`
- `experiment_01_results.md`
- `experiment_02_results.md`
- `experiment_03_results.md`

## Methods

### Model and Runtime Setup

The ReAct sweep used the same local model aliases as the other experiments:

- `rnj` -> `rnj-1:latest`
- `devstral` -> `devstral-small-2:latest`

Shared runtime settings were:

- `api_base = http://localhost:11434`
- `temperature = 0.0`
- `timeout_s = 12.0`
- `num_retries = 3`

Per the runner defaults for Ollama-backed runs, request timeout falls back to a longer local setting, but the primary benchmark-facing timeout remained the shared **12-second hidden-test evaluation cap**.

### ReAct Setup

The runner uses DSPy's `ReAct` as a tool-using agent loop rather than a plain one-shot code generator. Unlike CodeAct or ProgramOfThought, this setup does **not** require Deno. The ReAct loop uses deterministic helper tools to inspect the benchmark contract and statically repair candidate programs before finalization.

Shared helper tools include:

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

### Benchmarks and Splits

The ReAct runner reuses the same benchmark loaders, grouped split logic, and hidden-test evaluator as the baseline and GEPA experiments.

SciCode configuration:

- source: Hugging Face `SciCode1/SciCode`
- split: `test`
- groups: `train=24`, `val=8`, `eval=16`
- resulting evaluation examples: `94`
- `max_iters = 6`
- `max_tokens = 3072`

LiveCodeBench configuration:

- local dataset release: `release_latest`
- groups: `train=40`, `val=10`, `eval=40`
- resulting evaluation examples: `175`
- `max_iters = 5`
- `max_tokens = 2048`

For both benchmarks, the reported score per example is:

- `score = passed_hidden_tests / total_hidden_tests`

## Main Results

### Run-Level Outcomes

| Benchmark | Model | Eval examples | Mean score | Solved | Partial | Zero-score | Avg problem time (s) | Avg prediction time (s) | Avg agent steps | Max steps | Total run duration (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SciCode | `rnj` | 94 | 0.1764 | 12 | 8 | 74 | 95.21 | 93.80 | 3.63 | 6 | 8966.70 |
| SciCode | `devstral` | 94 | 0.1871 | 15 | 6 | 73 | 62.89 | 61.25 | 4.66 | 6 | 5975.68 |
| LiveCodeBench | `rnj` | 175 | 0.1647 | 26 | 23 | 126 | 45.39 | 44.53 | 2.85 | 5 | 8016.26 |
| LiveCodeBench | `devstral` | 175 | 0.2119 | 34 | 26 | 115 | 64.81 | 63.73 | 4.08 | 5 | 11416.09 |

### Benchmark-Level Averages Within the Sweep

| Benchmark | Runs | Mean of run means | Mean solved count | Mean problem time (s) | Mean agent steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| SciCode | 2 | 0.1817 | 13.50 | 79.05 | 4.14 |
| LiveCodeBench | 2 | 0.1883 | 30.00 | 55.10 | 3.46 |

### Tooling Observations

1. **The `devstral` ReAct runs used a materially richer tool mix than the `rnj` runs.**
   The `devstral` trajectories recorded frequent use of `benchmark_playbook`, `extract_constraints_json`, `extract_examples_json`, `preview_solution_shape_json`, `syntax_check_json`, and `repair_hints_json`.

2. **The `scicode` + `rnj` run relied heavily on contract-summary tools rather than broad tool diversity.**
   Its logged tool usage was dominated by `summarize_task_contract`, `make_solution_scaffold`, and `solution_contract_json`.

3. **More tool activity did not automatically imply a stronger result.**
   The most tool-heavy SciCode run was `devstral`, but its final score was lower than baseline, GEPA, and CodeAct on the same pair.

4. **The LiveCodeBench runs benefited from I/O-aware scaffolding tools.**
   Both LiveCodeBench runs recorded use of `make_stdio_solution_scaffold`, consistent with the repo's focus on interface and formatting failures for `stdin` tasks.

## Comparison To Earlier Experiments

The cleanest direct comparison is against the earlier completed single-agent results on the same split family: the plain baseline, GEPA prompt optimization, and CodeAct where available.

| Pair | ReAct mean | Baseline mean | Delta vs baseline | Best prior completed single-agent | Prior best mean | ReAct delta vs prior best | ReAct solved | Baseline solved | Prior best solved |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `scicode` + `rnj` | 0.1764 | 0.1640 | +0.0124 | `Baseline` / `GEPA` tie | 0.1640 | +0.0124 | 12 | 13 | 13 |
| `scicode` + `devstral` | 0.1871 | 0.2305 | -0.0434 | `GEPA` | 0.2429 | -0.0559 | 15 | 20 | 20 |
| `livecodebench` + `rnj` | 0.1647 | 0.1373 | +0.0274 | `CodeAct` | 0.1689 | -0.0041 | 26 | 22 | 26 |
| `livecodebench` + `devstral` | 0.2119 | 0.2011 | +0.0107 | `Baseline` / `GEPA` tie | 0.2011 | +0.0107 | 34 | 34 | 34 |

Interpretation:

1. **ReAct helped the weaker `rnj` baseline on both benchmarks.**
   Mean score improved on both `scicode/rnj` and `livecodebench/rnj`, with the largest absolute gain on LiveCodeBench.

2. **ReAct produced the strongest completed `livecodebench/devstral` single-agent result currently recorded in the repo.**
   It improved mean score by **0.0107** over the tied baseline and GEPA number while holding solved count constant at **34**.

3. **ReAct was a clear miss on `scicode/devstral`.**
   That pair finished **0.0434** mean-score points below baseline and **0.0559** below the earlier GEPA result, while solving **5** fewer tasks perfectly.

4. **ReAct was competitive but not best on `livecodebench/rnj`.**
   It tied the earlier GEPA solved count and stayed only **0.0041** mean-score points behind CodeAct, but it did so with much higher average per-example latency.

## Interpretation

Three conclusions are supported by the current artifacts.

1. ReAct is viable in this repo's benchmark stack and completes the full four-run matrix cleanly.
2. ReAct helps on interface-sensitive settings, especially LiveCodeBench and the weaker `rnj` model, but it is not a consistent single-agent winner across all pairs.
3. The current ReAct tool loop appears to trade a large amount of latency for selective accuracy gains, rather than broad across-the-board improvement.

For paper-style positioning, the concise summary is:

> ReAct is a credible completed single-agent comparison point in this repo. It improves over baseline on three of the four benchmark/model pairs, sets the strongest recorded completed single-agent result on `scicode/rnj` and `livecodebench/devstral`, but regresses sharply on `scicode/devstral` and is substantially slower than the simpler baselines.
