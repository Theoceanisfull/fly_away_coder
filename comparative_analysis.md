# Comparative Analysis

## Title

Cross-experiment comparison of the repo's local coding-benchmark techniques on the shared SciCode and LiveCodeBench split family.

## Executive Summary

This document compares the completed experiments that share the repo's standard grouped held-out split setup. Among the **completed full-matrix single-agent sweeps**, the strongest overall average currently belongs to **GEPA prompt optimization**, with average mean score **0.1921** across the four benchmark/model pairs. ReAct follows at **0.1850**, only slightly above the plain baseline at **0.1832**.

By completed single-agent pair, the current winners are:

- `scicode` + `rnj`: **ReAct**, mean score **0.1764**
- `scicode` + `devstral`: **GEPA**, mean score **0.2429**
- `livecodebench` + `rnj`: **CodeAct**, mean score **0.1689**
- `livecodebench` + `devstral`: **ReAct**, mean score **0.2119**

The twin-agent `dual_agents` run still has the strongest absolute benchmark numbers overall:

- **SciCode**: **0.2580**
- **LiveCodeBench**: **0.2440**

But those numbers should be treated as an **oracle-style upper bound**, not a clean benchmark comparison, because the final selection uses hidden-test-based `local_eval`.

ProgramOfThought is excluded from the main comparison tables because its fresh standalone sweep is still running at write time. The completed SciCode sub-runs are:

- `scicode` + `rnj`: **0.1268**
- `scicode` + `devstral`: **0.2376**

Both LiveCodeBench ProgramOfThought runs are still pending.

## Scope And Comparison Rules

This comparison uses the completed artifacts summarized in:

- `experiment_baseline_results.md`
- `experiment_01_results.md`
- `experiment_02_results.md`
- `experiment_03_results.md`
- `experiment_04_results.md`

The comparison is structured in three tiers:

1. **Completed single-agent techniques on matched benchmark/model pairs**
   This includes Baseline, GEPA, CodeAct where available, and ReAct.

2. **Technique-level aggregates**
   These are most meaningful for techniques that completed the full four-pair matrix. CodeAct is included, but only over its two completed pairs.

3. **Collaboration-assisted upper bound**
   The twin-agent run is reported separately because its `local_eval` selector uses hidden tests to choose the final candidate.

For all benchmark rows here, the primary metric remains:

- `score = passed_hidden_tests / total_hidden_tests`

## Coverage

| Technique | Completed pairs or runs | Included in main single-agent table? | Key caveat |
| --- | --- | --- | --- |
| Baseline | 4/4 pairs | Yes | Plain one-shot reference only |
| GEPA prompt optimization | 4/4 pairs | Yes | Optimized prompt, not an agent loop |
| CodeAct | 2/4 pairs | Yes, for completed pairs only | Not yet a full matrix |
| ReAct | 4/4 pairs | Yes | Highest latency among completed single-agent methods |
| Twin-agent `dual_agents` | 2 benchmark-level runs | Reported separately | Hidden tests participate in final selection via `local_eval` |
| ProgramOfThought | 2/4 pairs completed so far | No | Sweep still running |

## Technique-Level Aggregate View

| Technique | Completed comparable pairs | Avg mean score across completed pairs | Avg solved count across completed pairs | Comparison note |
| --- | ---: | ---: | ---: | --- |
| Baseline | 4 | 0.1832 | 22.25 | Full four-pair reference |
| GEPA | 4 | 0.1921 | 23.25 | Best completed full-matrix single-agent average |
| ReAct | 4 | 0.1850 | 21.75 | Slightly above baseline, below GEPA overall |
| CodeAct | 2 | 0.2028 | 22.50 | Promising, but not directly comparable as a full matrix |

The aggregate view is informative but not sufficient by itself. The pairwise tables below show that the winner changes by benchmark and model.

## Completed Single-Agent Pairwise Results

| Benchmark | Model | Technique | Mean score | Solved | Avg problem time (s) | Delta vs baseline mean | Delta vs baseline solved |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| SciCode | `rnj` | Baseline | 0.1640 | 13 | 7.61 | 0.0000 | 0 |
| SciCode | `rnj` | GEPA | 0.1640 | 13 | 1.30 | 0.0000 | 0 |
| SciCode | `rnj` | ReAct | 0.1764 | 12 | 95.21 | +0.0124 | -1 |
| SciCode | `devstral` | Baseline | 0.2305 | 20 | 19.97 | 0.0000 | 0 |
| SciCode | `devstral` | GEPA | 0.2429 | 20 | 22.73 | +0.0124 | 0 |
| SciCode | `devstral` | CodeAct | 0.2367 | 19 | 35.36 | +0.0062 | -1 |
| SciCode | `devstral` | ReAct | 0.1871 | 15 | 62.89 | -0.0434 | -5 |
| LiveCodeBench | `rnj` | Baseline | 0.1373 | 22 | 9.04 | 0.0000 | 0 |
| LiveCodeBench | `rnj` | GEPA | 0.1601 | 26 | 9.87 | +0.0229 | +4 |
| LiveCodeBench | `rnj` | CodeAct | 0.1689 | 26 | 18.16 | +0.0316 | +4 |
| LiveCodeBench | `rnj` | ReAct | 0.1647 | 26 | 45.39 | +0.0274 | +4 |
| LiveCodeBench | `devstral` | Baseline | 0.2011 | 34 | 26.84 | 0.0000 | 0 |
| LiveCodeBench | `devstral` | GEPA | 0.2011 | 34 | 0.74 | 0.0000 | 0 |
| LiveCodeBench | `devstral` | ReAct | 0.2119 | 34 | 64.81 | +0.0107 | 0 |

## Best Completed Single-Agent Result By Pair

| Benchmark | Model | Winner | Mean score | Solved | Runner-up | Gap over runner-up |
| --- | --- | --- | ---: | ---: | --- | ---: |
| SciCode | `rnj` | ReAct | 0.1764 | 12 | Baseline / GEPA tie at 0.1640 | +0.0124 |
| SciCode | `devstral` | GEPA | 0.2429 | 20 | CodeAct at 0.2367 | +0.0062 |
| LiveCodeBench | `rnj` | CodeAct | 0.1689 | 26 | ReAct at 0.1647 | +0.0041 |
| LiveCodeBench | `devstral` | ReAct | 0.2119 | 34 | Baseline / GEPA tie at 0.2011 | +0.0107 |

The pairwise picture is the most important one:

1. **No single completed single-agent technique dominates all four pairs.**
2. **GEPA is strongest on `scicode/devstral`.**
3. **CodeAct is strongest on `livecodebench/rnj`, but only by a narrow margin over ReAct.**
4. **ReAct owns the best completed `scicode/rnj` and `livecodebench/devstral` results.**

## Collaboration-Assisted Upper Bound

| Benchmark | Technique | Mean score | Solved | Avg total time (s) | Best completed single-agent reference | Delta mean | Delta solved | Caveat |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| SciCode | Twin-agent `rnj + devstral` | 0.2580 | 20 | 77.37 | GEPA `devstral` at 0.2429 | +0.0151 | 0 | Final answer selected with hidden-test `local_eval` |
| LiveCodeBench | Twin-agent `rnj + devstral` | 0.2440 | 40 | 94.45 | ReAct `devstral` at 0.2119 | +0.0321 | +6 | Final answer selected with hidden-test `local_eval` |

This is the strongest raw performer in the repo so far, but it is not the cleanest benchmark protocol. Its value is best understood as a collaboration-plus-reranking upper bound.

## ProgramOfThought Status

The current standalone ProgramOfThought sweep is still running in:

- `experiments/ProgramOfThought/artifacts/sweeps/20260426T133412Z_standalone_timeout_guard/`

At write time:

- `scicode` + `rnj`: mean **0.1268**, solved **9/94**, avg total time **3.06 s**
- `scicode` + `devstral`: mean **0.2376**, solved **18/94**, avg total time **3.50 s**
- `livecodebench` runs: not yet complete

Because the matrix is incomplete, ProgramOfThought is omitted from the main comparison tables above. Once both LiveCodeBench runs finish, it should be folded into the same pairwise grid.

## Main Takeaways

1. **GEPA is currently the strongest completed full-matrix single-agent method overall.**
   Its average across the four completed pairs is the best among the techniques that actually completed the full matrix.

2. **ReAct is selective rather than uniformly better.**
   It delivers the best completed result on two of the four pairs, but one of those gains comes with very high latency and one pair regresses badly.

3. **CodeAct looks especially promising on `livecodebench/rnj`, but the current evidence is still incomplete.**
   Until the missing two pairs are run, it should not be treated as the overall single-agent winner.

4. **The twin-agent result is the performance ceiling in the current repo, but not the fairest benchmark number.**
   Its hidden-test-based candidate selection makes it methodologically stronger as an upper-bound experiment than as a headline benchmark claim.

5. **ProgramOfThought remains an open comparison.**
   The timeout-guarded rerun is now progressing, but it should only enter the main comparison after the full benchmark matrix completes.
