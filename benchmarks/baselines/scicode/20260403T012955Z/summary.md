# SciCode Benchmark Summary

Artifacts: `/home/z4j/fly_away_code/SciCode/artifacts/scicode_benchmarks/20260403T012955Z`

## Overall Metrics

| Model | Main Problem Correctness | Subproblem Correctness | Passed Steps | Timeouts | Missing | Duration | Return Code |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| devstral-small-2:latest | 1.5% | 17.5% | 51 | 3 | 0 | 8h 50m 31s | 0 |
| rnj-1:latest | 0.0% | 13.1% | 38 | 1 | 0 | 1h 25m 52s | 0 |

## Head-to-Head

- devstral-small-2:latest unique solved problems: Householder_QR
- rnj-1:latest unique solved problems: none

## Model Notes

### devstral-small-2:latest

- Headline: 1.5% main-problem correctness and 17.5% subproblem correctness.
- Strongest dependency families: cmath (37.5%), mpl_toolkits (28.6%), time (28.6%).
- Weakest dependency families: itertools (22.5%), scipy (18.6%), numpy (17.7%).
- Strongest problems: Householder_QR (100.0%), Tolman_Oppenheimer_Volkoff_star (60.0%), VQE (60.0%).
- Weakest problems: Estimating_Stock_Option_Price (0.0%), dmrg (0.0%), LEG_Dyson_equation_bulk (0.0%).
- Long-chain behavior: Q4 executed-step pass rate drops by 31.4% versus Q1.

### rnj-1:latest

- Headline: 0.0% main-problem correctness and 13.1% subproblem correctness.
- Strongest dependency families: mpl_toolkits (28.6%), pickle (28.6%), time (28.6%).
- Weakest dependency families: numpy (13.2%), scipy (12.4%), cmath (0.0%).
- Strongest problems: Tolman_Oppenheimer_Volkoff_star (60.0%), GCMC (50.0%), phonon_angular_momentum (50.0%).
- Weakest problems: Xray_conversion_I (0.0%), dmrg (0.0%), LEG_Dyson_equation_bulk (0.0%).
- Long-chain behavior: Q4 executed-step pass rate drops by 28.8% versus Q1.

## Charts

- `charts/overall_metrics.png`
- `charts/dependency_pass_rates.png`
- `charts/step_bucket_pass_rates.png`
- `charts/problem_heatmap.png`

## Raw Data

- `overall_metrics.csv`
- `per_problem.csv`
- `per_step.csv`
- `dependency_metrics.csv`
- `step_bucket_metrics.csv`
