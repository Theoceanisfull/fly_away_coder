# Baseline Stats

This directory is the pushable record of benchmark outcomes for the benchmark playground. Raw benchmark datasets, generated model outputs, inspect logs, and local run artifacts should stay local under `SciCode/` and should not be pushed.

The current baseline is from local Ollama runs against SciCode and LiveCodeBench. Treat these as starting points for new coding-agent harness work, not as upstream leaderboard claims.

## Current Baseline

### SciCode

Source artifact: `SciCode/artifacts/scicode_benchmarks/20260403T012955Z/benchmark/reports/`

Run: `20260403T012955Z`

Split: `test`

| Model | Problems | Main problems solved | Main correctness | Passed steps | Subproblem correctness | Timeouts | Duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ollama/devstral-small-2:latest` | 65 | 1 | 1.5% | 51 / 291 | 17.5% | 3 | 8h 50m 31s |
| `ollama/rnj-1:latest` | 65 | 0 | 0.0% | 38 / 291 | 13.1% | 1 | 1h 25m 52s |

Primary target to beat: `ollama/devstral-small-2:latest` at 1 solved main problem and 51 passed steps.

Included files:

- `scicode/20260403T012955Z/summary.md`
- `scicode/20260403T012955Z/summary.json`
- `scicode/20260403T012955Z/overall_metrics.csv`
- `scicode/20260403T012955Z/per_problem.csv`
- `scicode/20260403T012955Z/per_step.csv`
- `scicode/20260403T012955Z/dependency_metrics.csv`
- `scicode/20260403T012955Z/step_bucket_metrics.csv`

### LiveCodeBench

Source artifact: `SciCode/artifacts/livecodebench_benchmarks/20260407T200229Z/benchmark/reports/`

Run: `20260407T200229Z`

Release version: `release_latest`

Samples per problem: `1`

| Model | Problems | pass@1 | Any-correct problems | Avg time / problem | Duration |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ollama/devstral-small-2:latest` | 1055 | 41.2% | 435 | 22s | 6h 29m 19s |
| `ollama/rnj-1:latest` | 1055 | 31.8% | 336 | 3s | 57m 22s |

Primary target to beat: `ollama/devstral-small-2:latest` at 435 solved problems and 41.2% pass@1.

Included files:

- `livecodebench/20260407T200229Z/summary.md`
- `livecodebench/20260407T200229Z/summary.json`
- `livecodebench/20260407T200229Z/overall_metrics.csv`
- `livecodebench/20260407T200229Z/per_problem.csv`

## Repo Hygiene

Keep these local and ignored:

- `SciCode/artifacts/`
- `SciCode/eval/data/test_data.h5`
- `SciCode/livecodebench/code_generation_lite/test*.jsonl`
- generated model outputs such as `generations.json`, `generations.jsonl`, `eval_all.json`, inspect logs, task output, and runner logs

Push these:

- benchmark harness code
- lightweight aggregate reports in `baseline_stats/`
- documentation explaining how a run was produced and what target it is trying to beat

## Updating Baselines

After a new full run finishes, copy only the report outputs into a new timestamped folder under `baseline_stats/<benchmark>/<run_id>/`, then update this README and `manifest.json`.

For SciCode, export from:

```bash
SciCode/artifacts/scicode_benchmarks/<run_id>/benchmark/reports/
```

For LiveCodeBench, export from:

```bash
SciCode/artifacts/livecodebench_benchmarks/<run_id>/benchmark/reports/
```

Do not copy raw benchmark data or generation logs into `baseline_stats/` unless there is a deliberate review decision to publish them.
