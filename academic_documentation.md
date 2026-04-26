# Academic Documentation

## Abstract

This repository is a local research harness for evaluating inference-time methods for code generation on two hidden-test benchmarks: `SciCode` and `LiveCodeBench`. The main contribution is not a new model or a new benchmark. The contribution is a unified experimental stack that keeps the benchmark loaders, grouped split construction, hidden-test evaluators, local model endpoints, and artifact formats fixed while swapping the inference strategy. That makes the repo useful for controlled comparisons between one-shot prompting, prompt optimization, tool-using agents, computation-assisted reasoning, and compact multi-agent collaboration.

As of `2026-04-26`, the completed evidence supports four main conclusions. First, `devstral` is the strongest plain single-pass baseline on both benchmarks. Second, `GEPA` is the strongest completed full-matrix single-agent method overall and the best completed single-agent method on `scicode/devstral`. Third, `ReAct` is a credible completed comparison point that improves three of the four benchmark/model pairs and sets the best completed single-agent result on `scicode/rnj` and `livecodebench/devstral`. Fourth, the best raw benchmark numbers come from the twin-agent protocol with `local_eval`, but those numbers are an upper-bound style result because hidden tests participate in final candidate selection.

`ProgramOfThought` should be treated as provisional at the time of writing. A timeout-guarded standalone rerun has completed the SciCode sub-runs, while the LiveCodeBench sub-runs were still in progress on `2026-04-26`.

## Research Scope

The repo currently implements six method families:

1. `Baseline`: one-shot code generation with a simple DSPy `ChainOfThought` solver.
2. `GEPA prompt optimization`: train/validation-time prompt search over the same base solver.
3. `CodeAct`: short executable-action trajectories with benchmark-aware helper tools.
4. `ReAct`: tool-augmented reasoning focused on contract checking and static repair.
5. `ProgramOfThought`: internal Python execution used as reasoning scaffolding before submitting a final code string.
6. `Dual agents / twins`: a symmetric draft, cross-revise, consensus protocol over two local models.

The repo is therefore best understood as an experimental framework for inference-time adaptation under shared evaluation, not as a single monolithic agent contribution.

## Benchmarks And Evaluation

### Benchmarks

- `SciCode` evaluates scientific programming tasks decomposed into sub-steps.
- `LiveCodeBench` evaluates competitive-programming-style code generation with hidden tests across `stdin` and functional task formats.

### Data Contract

The harness uses a mixed public-plus-local data contract:

- `SciCode` task text is loaded from the Hugging Face dataset `SciCode1/SciCode`.
- `SciCode` hidden targets are loaded locally from `benchmarks/scicode/data/test_data.h5`.
- `LiveCodeBench` tasks and hidden tests are loaded from the local `code_generation_lite` snapshot under `benchmarks/livecodebench/code_generation_lite/`.

This path contract is hard-coded in `experiments/prompt_optimization/benchmark_adapters.py`, which is why the public repo keeps those paths stable even though the large payloads are intentionally excluded from Git.

### Split Protocol

All benchmark-scale comparisons in the experiment reports use grouped held-out splits with `seed = 0`.

SciCode uses:

- `train_groups = 24`
- `val_groups = 8`
- `eval_groups = 16`
- resulting example counts `96 / 43 / 94`

LiveCodeBench uses:

- `train_groups = 40`
- `val_groups = 10`
- `eval_groups = 40`
- resulting example counts `159 / 44 / 175`

Grouping is benchmark-aware:

- SciCode groups by `problem_id`
- LiveCodeBench groups by `(contest_date, contest_id)`

### Metric

For both benchmarks, the per-example score is:

`score = passed_hidden_tests / total_hidden_tests`

Reported aggregates include:

- mean score
- solved count
- partial count
- zero-score count
- timing statistics

That shared metric and shared split family are the main reasons the cross-method comparisons in this repo are meaningful.

## Implemented Methods

### Baseline

The baseline is the cleanest reference point in the repo:

- one model call per task
- no tools
- no optimization
- no repair loop
- no reranking

It isolates the raw capability of the local models under the same hidden-test evaluation stack.

### GEPA

The prompt-optimization runner uses DSPy's `GEPA` to optimize the natural-language instructions of the baseline solver while holding the evaluation stack fixed. Each run:

1. evaluates the unoptimized baseline on the held-out `eval` split
2. optimizes on disjoint `train` and `val` splits
3. re-evaluates the optimized program on the same held-out `eval` split

This is the most controlled prompt-optimization comparison in the repo.

### CodeAct

The CodeAct experiment adapts DSPy's code-execution agent pattern to the benchmark setting. It exposes benchmark-aware helper tools but not hidden tests at inference time. In the completed runs so far, observed trajectories were shallow and tool-usage logs were sparse, so the current CodeAct results should be interpreted as a modest executable-action baseline rather than a long-horizon autonomous coding agent.

### ReAct

The ReAct experiment uses benchmark-aware static tools for:

- syntax checking
- interface preservation
- I/O mode checking
- scaffold construction
- simple repair hints

This method is now fully benchmark-complete on the shared four-pair matrix and is no longer just an infrastructure placeholder.

### ProgramOfThought

The ProgramOfThought experiment adapts DSPy's computation-assisted reasoning pattern to code generation:

1. the model writes scratch Python
2. the scratch code builds a candidate solution string
3. the scratch code submits the final candidate string for benchmark evaluation

This is computation-assisted reasoning, not direct execution of the benchmark candidate inside the DSPy sandbox.

Operationally, this method required a timeout guard in the local harness. Without a wall-clock cap, a single hung Deno-backed scratch execution could stall the whole sweep indefinitely. The current runner now caps each prediction attempt and resets the interpreter on timeout before continuing.

### Dual Agents

The twin-agent protocol is deliberately small and symmetric:

1. Twin A draft
2. Twin B draft
3. Twin A revision
4. Twin B revision
5. Twin A consensus
6. Twin B consensus

The best recorded run uses `local_eval`, which executes all six candidates on the task's hidden tests and selects the best one. That makes the result a strong local upper bound, but not a contamination-safe benchmark headline.

## Results Snapshot

### Strongest Completed Single-Agent Results By Pair

| Benchmark | Model | Winning completed single-agent method | Mean score | Solved |
| --- | --- | --- | ---: | ---: |
| SciCode | `rnj` | ReAct | `0.1764` | `12 / 94` |
| SciCode | `devstral` | GEPA | `0.2429` | `20 / 94` |
| LiveCodeBench | `rnj` | CodeAct | `0.1689` | `26 / 175` |
| LiveCodeBench | `devstral` | ReAct | `0.2119` | `34 / 175` |

### Completed Full-Matrix Single-Agent Aggregate View

| Method | Completed comparable pairs | Average mean score across completed pairs | Average solved count |
| --- | ---: | ---: | ---: |
| Baseline | 4 | `0.1832` | `22.25` |
| GEPA | 4 | `0.1921` | `23.25` |
| ReAct | 4 | `0.1850` | `21.75` |

`CodeAct` is omitted from that table because only two of the four benchmark/model pairs have completed. Its current average over those two completed pairs is `0.2028`, but that is not directly comparable to a full matrix.

### Collaboration-Assisted Upper Bound

| Benchmark | Method | Mean score | Solved | Caveat |
| --- | --- | ---: | ---: | --- |
| SciCode | Twin-agent `rnj + devstral` | `0.2580` | `20 / 94` | final answer selected with hidden-test `local_eval` |
| LiveCodeBench | Twin-agent `rnj + devstral` | `0.2440` | `40 / 175` | final answer selected with hidden-test `local_eval` |

### ProgramOfThought Status

The current timeout-guarded standalone `ProgramOfThought` sweep was still running on `2026-04-26`. The completed sub-runs at that time were:

| Benchmark | Model | Mean score | Solved | Status |
| --- | --- | ---: | ---: | --- |
| SciCode | `rnj` | `0.1268` | `9 / 94` | complete |
| SciCode | `devstral` | `0.2376` | `18 / 94` | complete |
| LiveCodeBench | `rnj` | pending | pending | in progress |
| LiveCodeBench | `devstral` | pending | pending | in progress |

These partial results are informative but should stay outside the main comparison tables until the full matrix completes.

## Interpretation

The current evidence supports the following claims.

1. `devstral` is the strongest plain single-pass baseline model in this repo.
2. `GEPA` is the strongest completed full-matrix single-agent method overall.
3. `ReAct` is selective rather than uniformly dominant: it wins two of the four completed pairs but regresses sharply on `scicode/devstral`.
4. `CodeAct` is promising on `livecodebench/rnj`, but the matrix is still incomplete.
5. The twin-agent result is the current performance ceiling, but it is methodologically different because it uses hidden-test reranking.
6. `ProgramOfThought` is now operationally more robust after the timeout fix, but the benchmark-level conclusion should wait for the LiveCodeBench runs to finish.

## Related Work Positioning

### DSPy And Prompt Optimization

The repo builds directly on `DSPy`, which frames LM systems as programs rather than prompt strings, and on `GEPA`, which treats prompt optimization as reflective text evolution rather than reinforcement learning.

Relevant references:

- DSPy paper: <https://arxiv.org/abs/2310.03714>
- DSPy docs: <https://dspy.ai/>
- DSPy repo: <https://github.com/stanfordnlp/dspy>
- GEPA paper: <https://arxiv.org/abs/2507.19457>
- GEPA OpenReview: <https://openreview.net/forum?id=RQm2KQTM5r>

### Tool-Using And Computation-Assisted Agents

Three papers are especially important for the single-agent methods implemented here:

- ReAct: <https://arxiv.org/abs/2210.03629>
- Program of Thoughts Prompting: <https://arxiv.org/abs/2211.12588>
- CodeAct, "Executable Code Actions Elicit Better LLM Agents": <https://arxiv.org/abs/2402.01030>

These provide the nearest conceptual references for the repo's ReAct, ProgramOfThought, and CodeAct experiment folders, even though the benchmark wiring in this repository is local and custom.

### Benchmark References

The benchmark setup in this repo is grounded in:

- LiveCodeBench paper: <https://arxiv.org/abs/2403.07974>
- LiveCodeBench repo: <https://github.com/livecodebench/livecodebench>
- SciCode paper: <https://arxiv.org/abs/2407.13168>
- SciCode repo: <https://github.com/scicode-bench/SciCode>

### Multi-Agent And Iterative-Refinement Context

The dual-agent protocol in this repo is not novel as a general pattern. The relevant literature includes:

- Multiagent Debate: <https://arxiv.org/abs/2305.14325>
- Reflexion: <https://arxiv.org/abs/2303.11366>
- AgentCoder: <https://arxiv.org/abs/2312.13010>
- MapCoder: <https://arxiv.org/abs/2405.11403>
- MetaGPT: <https://arxiv.org/abs/2308.00352>
- CAMEL: <https://arxiv.org/abs/2303.17760>

The novelty boundary for this repo is therefore modest and defensible:

- the value is the shared local benchmark harness
- the methods are compared under one evaluation contract
- the twin-agent protocol is a compact benchmark-compatible baseline, not a claim to first-in-field multi-agent novelty

## Publication-Oriented Framing

For an IEEE-style paper or workshop report, the safest framing is:

- this repo presents a controlled local benchmark harness for inference-time coding methods
- the primary scientific value is comparability across methods, not the novelty of any single module in isolation
- the strongest publication-quality claims currently supported by completed evidence are:
  - prompt optimization improves some held-out coding settings
  - tool-using agents can help selectively, especially on interface-sensitive tasks
  - compact collaboration plus reranking provides a strong local upper bound

Claims to avoid:

- claiming the twin protocol as the first multi-agent coding method
- presenting `local_eval` twin results as contamination-safe benchmark numbers
- making final ProgramOfThought claims before the LiveCodeBench runs finish

## Bottom Line

This repository now supports a credible experimental story. Under a shared hidden-test benchmark harness, inference-time adaptation matters, but the best method depends on benchmark type, model, and evaluation protocol. `GEPA` is the strongest completed full-matrix single-agent method overall. `ReAct` is a real completed comparison point with selective wins. The twin-agent `local_eval` protocol is the strongest raw local upper bound. `ProgramOfThought` is now robust enough to run cleanly, but its final comparative position still depends on the unfinished LiveCodeBench portion of the sweep.
