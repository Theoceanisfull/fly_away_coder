# Experiment 03 Results

## Title

Twin-agent (`dual_agents`) benchmark sweep on SciCode and LiveCodeBench using a symmetric draft-revise-consensus protocol with local hidden-test reranking.

## Executive Summary

This report records the completed twin-agent sweep in:

`experiments/dual_agents/artifacts/sweeps/20260425T123616Z_rnj_devstral_local_eval/`

The objective was to test whether a minimal symmetric two-model collaboration protocol can improve held-out coding-benchmark performance over the repo's earlier single-agent baselines without adding complex tooling or planner roles. The two local Ollama models, `rnj` and `devstral`, were treated as peers: each drafted a solution, revised after reading the other draft, and then produced a final consensus candidate. The runner then selected the final answer with `local_eval`, which executes all six candidates on the task's hidden tests and keeps the best one.

Both benchmark runs completed and wrote final `summary.json` artifacts. From artifact timestamps, the sweep began at **2026-04-25 12:36:16 UTC** and finished at **2026-04-25 19:14:07 UTC** (**08:36:16 AM EDT** to **03:14:07 PM EDT**), for an end-to-end wall-clock duration of about **6.6 hours**.

Held-out results were:

- **SciCode**: mean score **0.2580**, **20/94** solved, **10** partial-credit examples, **64** zero-score examples
- **LiveCodeBench**: mean score **0.2440**, **40/175** solved, **19** partial-credit examples, **116** zero-score examples

Against the strongest previously recorded single-agent result in this repo on the same split family, the twin setup improved:

- **SciCode**: **0.2429 -> 0.2580** mean score, with solved count unchanged at **20**
- **LiveCodeBench**: **0.2011 -> 0.2440** mean score, with solved count **34 -> 40**

The main caveat is methodological: `local_eval` uses hidden tests to choose among same-task candidates. These numbers are therefore best interpreted as an **oracle-style local reranking result**, not as contamination-safe final benchmark scores.

## Experimental Scope

Authoritative artifacts for this report are:

- `experiments/dual_agents/artifacts/sweeps/20260425T123616Z_rnj_devstral_local_eval/runs/scicode/config.json`
- `experiments/dual_agents/artifacts/sweeps/20260425T123616Z_rnj_devstral_local_eval/runs/scicode/split_manifest.json`
- `experiments/dual_agents/artifacts/sweeps/20260425T123616Z_rnj_devstral_local_eval/runs/scicode/summary.json`
- `experiments/dual_agents/artifacts/sweeps/20260425T123616Z_rnj_devstral_local_eval/logs/scicode.log`
- `experiments/dual_agents/artifacts/sweeps/20260425T123616Z_rnj_devstral_local_eval/runs/livecodebench/config.json`
- `experiments/dual_agents/artifacts/sweeps/20260425T123616Z_rnj_devstral_local_eval/runs/livecodebench/split_manifest.json`
- `experiments/dual_agents/artifacts/sweeps/20260425T123616Z_rnj_devstral_local_eval/runs/livecodebench/summary.json`
- `experiments/dual_agents/artifacts/sweeps/20260425T123616Z_rnj_devstral_local_eval/logs/livecodebench.log`

Comparison points are taken from:

- `experiment_01_results.md`
- `experiment_02_results.md`
- `experiments/prompt_optimization/artifacts/sweeps/20260420T011758Z_full_matrix/analytics/summary.json`
- `experiments/CodeAct/artifacts/sweeps/20260422T130119Z_codeact_dual/analytics/summary.json`

## Methods

### Models and Runtime

The sweep used the same local model aliases as the earlier experiments:

- `rnj` -> `rnj-1:latest`
- `devstral` -> `devstral-small-2:latest`

Shared runtime settings were:

- `api_base = http://localhost:11434`
- `temperature = 0.0`
- `max_tokens = 3072`
- `request_timeout_s = 120.0`
- `timeout_s = 12.0` for hidden-test evaluation
- `num_retries = 0`
- `seed = 0`

### Twin Protocol

For each evaluation task, the runner makes **six sequential model calls**:

1. `Twin A` draft
2. `Twin B` draft
3. `Twin A` revision after reading `Twin B`
4. `Twin B` revision after reading `Twin A`
5. `Twin A` consensus candidate
6. `Twin B` consensus candidate

Each response is required to contain:

- a short `<analysis>` block
- a full executable Python `<solution>` block

The protocol is intentionally symmetric. Neither model is assigned a privileged planner, judge, or teacher role.

### Selection Rule

This sweep used `selection_strategy = local_eval`. All six candidates were executed on the benchmark's hidden tests, and the runner selected the candidate with the best tuple:

1. hidden-test score
2. hidden tests passed
3. stage priority: `consensus > revision > draft`
4. earlier candidate order as the final tiebreak

This setting is useful for measuring the value of candidate diversity plus collaboration, but it is not a clean benchmark protocol because the hidden tests participate in final answer selection.

### Benchmarks and Splits

The twin runner reuses the same benchmark loaders and hidden-test evaluators used by the prompt-optimization experiments. Train and validation splits were saved only to preserve the same clean held-out partitioning scheme; the twin protocol itself evaluated **only the `eval` split**.

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

For both benchmarks, the reported score for each example is:

- `score = passed_hidden_tests / total_hidden_tests`

## Main Results

### Run-Level Outcomes

| Benchmark | Models | Eval examples | Mean score | Solved | Partial | Zero-score | Avg generation time (s) | Avg selection-eval time (s) | Avg total time (s) | Run duration (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SciCode | `rnj + devstral` | 94 | 0.2580 | 20 | 10 | 64 | 68.77 | 8.60 | 77.37 | 7272.94 |
| LiveCodeBench | `rnj + devstral` | 175 | 0.2440 | 40 | 19 | 116 | 90.65 | 3.80 | 94.45 | 16529.19 |

### Selection Behavior

`local_eval` selected consensus-stage candidates most of the time, but not always:

- **SciCode**: consensus selected on **93/94** tasks; one task was best at the draft stage
- **LiveCodeBench**: consensus selected on **167/175** tasks; **6** draft-stage and **2** revision-stage candidates beat both consensus candidates

Across both benchmarks combined:

- consensus candidates were selected on **260/269** tasks (**96.7%**)
- `twin_a` supplied the final selected answer on **249/269** tasks (**92.6%**)

This means the protocol's gains were mostly realized in the final consensus stage, but the local reranker still occasionally recovered stronger earlier-stage candidates.

### Failure Patterns

The main remaining failure mode was still basic program validity rather than subtle partial-credit misses.

On SciCode, the most common failures were:

- **27** syntax-triggered runtime errors on hidden test 1
- **11** `NameError` failures on hidden test 1
- **5** `ModuleNotFoundError` failures on hidden test 1

On LiveCodeBench, the largest single error bucket was:

- **67** parse failures of the form `expected an indented block after function definition on line 2`

So although the twin protocol improved aggregate accuracy, syntax robustness remained a major bottleneck, especially on LiveCodeBench.

## Comparison To Earlier Experiments

The cleanest comparison is against the best previously recorded single-agent result in this repo for each benchmark on the same held-out split family.

| Benchmark | Twin mean | Best prior mean | Delta | Twin solved | Best prior solved | Delta solved | Best prior source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SciCode | 0.2580 | 0.2429 | +0.0151 | 20 | 20 | 0 | GEPA `devstral` |
| LiveCodeBench | 0.2440 | 0.2011 | +0.0429 | 40 | 34 | +6 | GEPA `devstral` |

Relative to the matched CodeAct runs from Experiment 02, the twin setup was also stronger:

- **SciCode**: **0.2367 -> 0.2580**, solved **19 -> 20**
- **LiveCodeBench**: **0.1689 -> 0.2440**, solved **26 -> 40**

These comparisons are informative because the experiments share the same loaders, split logic, hidden-test evaluators, and local model family. But they are not perfectly neutral comparisons, because the twin run is advantaged by six-way hidden-test reranking under `local_eval`.

## Interpretation

Three conclusions are supported by the current artifacts.

1. A very small symmetric twin protocol is already competitive in this repo's coding-benchmark stack.
2. On the recorded held-out splits, the twin setup outperformed the strongest earlier single-agent result on both benchmarks, with the largest gain on LiveCodeBench.
3. Most of that gain should be interpreted as the combination of candidate multiplicity, cross-revision, and oracle-style reranking, not as a clean estimate of collaboration alone.

For paper-style positioning, the most defensible summary is:

> The twin-agent baseline is a strong local upper-bound style protocol in this codebase. It improves held-out mean score on both SciCode and LiveCodeBench under `local_eval`, but the use of hidden tests for final candidate selection means the result should be reported as a reranking-assisted collaboration experiment rather than as a contamination-safe benchmark score.
