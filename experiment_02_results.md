# Experiment 02 Results

## Title

CodeAct dual-run benchmark sweep on SciCode and LiveCodeBench using local Ollama coding models and benchmark-aware helper tools.

## Executive Summary

This report records the completed CodeAct sweep in:

`experiments/CodeAct/artifacts/sweeps/20260422T130119Z_codeact_dual/`

The sweep started at **2026-04-22 13:01:19 UTC** and finished at **2026-04-22 14:51:03 UTC** (**2026-04-22 09:01:19 AM EDT** to **2026-04-22 10:51:03 AM EDT**). Both recorded runs completed successfully with **exit code 0**.

This was not a four-way full matrix. The authoritative sweep only ran the two benchmark/model pairs listed in its saved `run_plan`:

- `scicode` + `devstral`
- `livecodebench` + `rnj`

Held-out outcomes were:

- **SciCode + `devstral`**: mean score **0.2367**, **19/94** solved, **8** partial-credit examples, **67** zero-score examples
- **LiveCodeBench + `rnj`**: mean score **0.1689**, **26/175** solved, **26** partial-credit examples, **123** zero-score examples

The most notable operational detail is that both runs reported **`tool_name_counts = {}`**, even though the CodeAct setup exposed benchmark-aware helper tools. Average agent depth also stayed low:

- SciCode + `devstral`: **1.03** average agent steps, **2** max steps
- LiveCodeBench + `rnj`: **1.71** average agent steps, **4** max steps

So the recorded results reflect short CodeAct trajectories rather than heavy observed tool use.

## Experimental Scope

This report covers the latest CodeAct sweep artifacts present in the repo at write time. The authoritative source files are:

- `experiments/CodeAct/artifacts/sweeps/20260422T130119Z_codeact_dual/status.json`
- `experiments/CodeAct/artifacts/sweeps/20260422T130119Z_codeact_dual/sweep_config.json`
- `experiments/CodeAct/artifacts/sweeps/20260422T130119Z_codeact_dual/analytics/summary.json`
- `experiments/CodeAct/artifacts/sweeps/20260422T130119Z_codeact_dual/analytics/report.md`
- `experiments/CodeAct/artifacts/sweeps/20260422T130119Z_codeact_dual/logs/scicode__devstral.log`
- `experiments/CodeAct/artifacts/sweeps/20260422T130119Z_codeact_dual/logs/livecodebench__rnj.log`

The sweep wrapper also wrote a short `nohup.log` confirming orderly completion:

- `Starting scicode__devstral`
- `Starting livecodebench__rnj`
- `Completed sweep: experiments/CodeAct/artifacts/sweeps/20260422T130119Z_codeact_dual`

## Methods

### Model and Runtime Setup

The CodeAct runner uses the same local Ollama model aliases as the prompt-optimization harness:

- `rnj` -> `rnj-1:latest`
- `devstral` -> `devstral-small-2:latest`

The sweep-level runtime settings were:

- `api_base = http://localhost:11434`
- `temperature = 0.0`
- `timeout_s = 12.0`
- `num_retries = 3`

The implementation uses DSPy's `CodeAct` with DSPy's Deno-backed `PythonInterpreter` sandbox. Since this sweep completed end to end, the environment was operational for CodeAct execution at run time.

### Benchmark and Split Construction

Per the CodeAct README and runner implementation, this experiment reuses the same benchmark loaders, hidden-test evaluator, grouped split logic, seed, and local model aliases used by `experiments/prompt_optimization`.

The run used `seed = 0` with grouped disjoint splits and no shuffle.

SciCode configuration:

- source: Hugging Face `SciCode1/SciCode`
- split: `test`
- groups: `train=24`, `val=8`, `eval=16`
- resulting eval examples: `94`
- `max_iters = 6`
- `max_tokens = 3072`

LiveCodeBench configuration:

- local release: `release_latest`
- groups: `train=40`, `val=10`, `eval=40`
- resulting eval examples: `175`
- `max_iters = 4`
- `max_tokens = 2048`

### Tooling Contract

The CodeAct setup exposed a small benchmark-aware toolset rather than hidden tests or external retrieval.

Shared helper tools:

- `benchmark_playbook`
- `solution_contract`
- `extract_function_header`
- `extract_function_names`
- `preview_solution_shape`
- `strip_markdown_fences_tool`

Additional SciCode helpers:

- `extract_required_imports`
- `make_solution_scaffold`

The intended use was to help the agent preserve function signatures, required imports, and output shape while keeping the benchmark contract intact.

## Main Sweep Results

### Run-Level Outcomes

| Benchmark | Model | Eval examples | Mean score | Solved | Partial | Zero-score | Avg problem time (s) | Avg prediction time (s) | Avg agent steps | Max steps | Total run duration (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SciCode | `devstral` | 94 | 0.2367 | 19 | 8 | 67 | 35.36 | 33.52 | 1.03 | 2 | 3339.90 |
| LiveCodeBench | `rnj` | 175 | 0.1689 | 26 | 26 | 123 | 18.16 | 15.85 | 1.71 | 4 | 3244.23 |

### Benchmark-Level Observations

1. **Both benchmark runs completed cleanly.**
   There were no failed sub-runs, non-zero exit codes, or interrupted sweep markers in the authoritative artifacts.

2. **SciCode + `devstral` produced a respectable mean score but modest perfect-solve count.**
   The run reached **0.2367** mean score with **19** fully solved sub-steps out of **94**, leaving most tasks at zero score.

3. **LiveCodeBench + `rnj` produced the stronger held-out result of the two runs on its own benchmark.**
   The run reached **0.1689** mean score with **26** fully solved tasks out of **175**, plus another **26** partial-credit tasks.

4. **Observed CodeAct interaction depth was shallow.**
   The average step counts remained low and the analytics summary recorded no named tool usage in either run. This suggests the current setup behaved more like short agentic drafting passes than like long multi-tool search trajectories.

5. **Total sweep runtime was about 1 hour 49 minutes.**
   From the top-level `status.json`, the sweep lasted from **09:01:19 AM EDT** to **10:51:03 AM EDT** on **2026-04-22**.

## Comparison To Experiment 01 GEPA Results

This comparison is more apples-to-apples than a comparison against the published benchmark baselines, because the CodeAct runner and the GEPA harness share:

- the same model aliases
- the same benchmark loaders
- the same grouped split logic
- the same held-out group counts
- the same default `seed = 0`

The closest matching GEPA reference is the completed sweep in:

- `experiments/prompt_optimization/artifacts/sweeps/20260420T011758Z_full_matrix/`

For the same benchmark/model pairs, the held-out comparison is:

| Benchmark | Model | CodeAct mean | GEPA optimized mean | Delta (CodeAct - GEPA) | CodeAct solved | GEPA solved | Delta solved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SciCode | `devstral` | 0.2367 | 0.2429 | -0.0062 | 19 | 20 | -1 |
| LiveCodeBench | `rnj` | 0.1689 | 0.1601 | +0.0087 | 26 | 26 | 0 |

Interpretation:

1. **CodeAct underperformed the earlier GEPA result on SciCode + `devstral`.**
   It finished **0.0062** mean-score points lower and solved **1** fewer sub-step perfectly.

2. **CodeAct slightly outperformed the earlier GEPA result on LiveCodeBench + `rnj` in mean score, while matching solved count.**
   The mean-score advantage was **0.0087**, with both methods solving **26** tasks.

3. **The mixed outcome argues against a single broad conclusion that CodeAct or GEPA is categorically better in this repo.**
   On the currently recorded matched pairs, CodeAct looks more competitive on LiveCodeBench than on SciCode.

## Practical Takeaways

1. The CodeAct experiment is no longer just a scaffold. It produced a completed benchmark sweep with stable artifacts on both supported benchmarks.

2. The current benchmark-aware CodeAct setup is already viable as a comparison point against prompt optimization on matched held-out splits.

3. The empty `tool_name_counts` fields deserve follow-up. Either the agent rarely needed the provided tools, or the current artifact pipeline is not surfacing tool usage in a useful way.

4. If Experiment 03 is meant to improve CodeAct itself, the clearest next questions are whether the agent can be pushed into more productive multi-step tool use on SciCode and whether LiveCodeBench gains hold across more than the single `rnj` run recorded here.
