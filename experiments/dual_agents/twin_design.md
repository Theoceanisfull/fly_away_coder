# Twin Design

## Purpose

This document describes the initial "twin" multi-agent benchmark setup in `experiments/dual_agents/`.

The goal is to let two coding LLMs collaborate on the same benchmark task as peers rather than as a strict teacher-student pair. The design is intentionally simple:

- both models see the same task
- both produce an independent first attempt
- both revise after seeing the other agent's work
- both produce a final consensus attempt
- the runner selects a final candidate

This is meant to be a usable baseline for benchmarking collaborative problem solving, not a final research design.

## Design Goals

The current system optimizes for a few concrete properties.

### 1. Symmetry

The two models should be treated as twins, not as planner and executor or judge and worker. That means:

- both agents get equivalent roles
- both agents get the same number of turns
- both agents can influence the final answer
- neither agent has privileged authority by default

This keeps the setup simple and makes model-pair comparisons easier to interpret.

### 2. Reuse Existing Benchmark Infrastructure

The repository already has stable benchmark loading and hidden-test evaluation logic in `experiments/prompt_optimization/benchmark_adapters.py`.

The twin system reuses that code instead of introducing a second evaluation stack. This reduces drift between single-agent and multi-agent experiments.

### 3. Small Surface Area

The first version should be easy to inspect, run, and modify. The point is to get a real benchmark harness into the repo quickly, not to prematurely build a large orchestration framework.

### 4. Artifact Quality

A benchmark run should leave enough trace data behind to inspect what each agent did. This is important for later analysis, failure review, and paper write-up work.

## Scope

The current implementation supports:

- `scicode`
- `livecodebench`
- local Ollama-hosted models through the OpenAI-compatible API
- grouped train/val/eval slicing logic consistent with the prompt-optimization experiments
- two final-selection strategies

It does not yet try to solve broader multi-agent research questions such as:

- long-horizon planning
- explicit tool routing
- role specialization
- debate trees or branching search
- learned arbitration between candidates
- cross-task memory
- adaptive turn budgets

## Current Files

The directory currently contains:

- `run_twins.py`: benchmark runner and twin protocol implementation
- `README.md`: short operational documentation
- `requirements.txt`: minimal dependencies
- `artifacts/`: outputs from runs

This design note is intended to sit beside the implementation and explain why the current protocol looks the way it does.

## Protocol Overview

The protocol uses six model calls per task.

### Stage 1: Independent Drafts

1. `Twin A` receives the benchmark prompt and produces:
   - short analysis
   - full executable Python solution
2. `Twin B` does the same independently

The purpose of this stage is to create diversity. If both agents begin from the same prompt and respond independently, they can produce different hypotheses, edge-case handling, or algorithm choices.

### Stage 2: Cross Revision

3. `Twin A` sees:
   - the task
   - its own prior draft
   - `Twin B`'s draft

   It must critique both and return a stronger revised solution.

4. `Twin B` does the symmetric operation.

The purpose of this stage is to let each model:

- salvage useful ideas from the other agent
- identify bugs in the other draft
- repair mistakes in its own draft

### Stage 3: Consensus

5. `Twin A` sees the two revised solutions and produces a final merged answer.
6. `Twin B` does the same.

The consensus stage is not a vote. Each model must commit to one concrete implementation. The prompt explicitly discourages averaging or vague synthesis.

## Why This Protocol

The protocol is designed to answer a narrow question:

Can two strong coding models improve over independent single-shot solving if they are allowed one round of mutual inspection and one final merge pass?

This setup is useful because it isolates a simple collaboration mechanism without confounding it with:

- external judges
- separate planner models
- search trees
- hidden coordination state

If the protocol helps, it is easier to attribute the gains to peer interaction rather than orchestration complexity.

If it does not help, that is also informative, because the failure mode is easier to diagnose.

## Prompt Format

Each agent is required to return:

- `<analysis>...</analysis>`
- `<solution>...</solution>`

The implementation extracts those tags, then falls back to code-fence stripping if the model does not follow the requested format exactly.

This structure is useful for two reasons:

1. It preserves a readable reasoning summary for artifact inspection.
2. It gives the later twin stage a compact record of what the earlier agent was trying to do.

The analysis is not used by the evaluator. Only the extracted solution code is executed.

## Benchmark Integration

The runner reuses benchmark loading and evaluation from `experiments/prompt_optimization/benchmark_adapters.py`.

That means:

- `SciCode` tasks are loaded through the same sub-step extraction logic
- `LiveCodeBench` tasks are loaded through the same release-file logic
- hidden-test execution is shared with the existing benchmark pipeline

This matters because it keeps the multi-agent experiments comparable to the single-agent prompt-optimization experiments.

## Split Logic

The twin runner copies the same basic split semantics already used in the prompt-optimization work.

### Clean Grouped Splits

By default, tasks are split by benchmark-specific groups:

- `SciCode`: grouped by `problem_id`
- `LiveCodeBench`: grouped by contest date and contest id

This prevents near-duplicate leakage between train, validation, and evaluation slices.

### Shared Group Splits

`--allow-shared-group-splits` exists for debugging only. It allows flat slicing without group isolation. This is mainly useful for smoke tests on tiny local samples.

## Selection Strategies

Two selection strategies are implemented.

### 1. `local_eval`

The runner evaluates every candidate attempt on the task's local hidden tests and selects the best one. Tie-breaking prefers:

- higher score
- more passed tests
- later stage
- earlier appearance among equally ranked candidates

This is the strongest local problem-solving setup, because it uses the benchmark evaluator as a reranker.

It is also methodologically impure if treated as a strict benchmark evaluation protocol, because the hidden tests are being used to choose among same-task candidates. For research reporting, this should be described carefully as an oracle-style local reranking setup.

### 2. `consensus_first`

The runner takes the first valid consensus-stage answer and evaluates only that final choice.

This is a cleaner collaboration protocol if the objective is to measure the effect of peer interaction without hidden-test reranking.

## Artifact Design

Each run writes:

- `config.json`
- `split_manifest.json`
- `summary.json`
- `task_artifacts/<task_id>.json`

The per-task artifact is important. It records:

- every attempt
- which stage produced it
- which twin produced it
- extracted analysis
- extracted solution
- raw model response
- generation time
- evaluation time
- per-candidate evaluation result when applicable
- which attempt was selected

This is enough to inspect:

- whether one model dominates the pair
- whether revision actually improves anything
- whether consensus introduces regressions
- which tasks benefit from collaboration

## Model and Runtime Assumptions

The current defaults assume:

- local Ollama server
- OpenAI-compatible endpoint
- `http://localhost:11434`

The runner automatically normalizes the API base to append `/v1` if needed, since the OpenAI client expects the chat endpoint under that path.

Current model aliases match the prompt-optimization setup:

- `rnj` -> `rnj-1:latest`
- `devstral` -> `devstral-small-2:latest`

Additional models can be passed directly by full model name without editing the alias map.

## Smoke Test Result

The harness was validated with a one-task SciCode smoke test using:

- `Twin A = rnj-1:latest`
- `Twin B = devstral-small-2:latest`
- local hidden-test reranking via `local_eval`

Run artifact:

- `experiments/dual_agents/artifacts/scicode/rnj-1_latest__devstral-small-2_latest/20260421T235511Z/`

Observed result:

- `mean_score = 1.0`
- `solved_count = 1 / 1`
- all six candidate attempts passed the hidden tests for that sample task

This does not prove the protocol is strong. It only verifies that:

- the models are wired correctly
- the protocol runs end to end
- artifact writing works
- benchmark evaluation works

## Strengths of the Current Design

### Simple and Interpretable

The protocol is small enough that failures can be inspected directly from artifacts.

### Symmetric

The two models are treated as peers, which matches the intended "twin" framing.

### Easy to Extend

The runner is a good starting point for more structured experiments:

- adding more rounds
- asymmetric roles
- judge models
- tool use
- candidate ensembling
- self-consistency

### Benchmark-Compatible

The setup uses the same benchmark interfaces as the existing prompt-optimization experiments, which reduces duplicated infrastructure.

## Weaknesses and Known Limitations

### 1. `local_eval` Is Not a Clean Final Benchmark Protocol

If hidden tests are used to pick the best candidate on the same task, the result is no longer a pure single-shot task evaluation. This is acceptable for development and local ablations, but it needs to be reported honestly.

### 2. No True Parallelism

The current implementation runs the stages sequentially. The two agents are conceptually twins, but the runtime does not execute both model calls concurrently.

That means:

- collaboration latency is higher than necessary
- there is no scheduler or async batching

### 3. No Role Differentiation

The symmetry is intentional, but it also means the system cannot exploit specialization such as:

- one model as planner
- one model as bug finder
- one model as synthesizer

### 4. Limited Search Depth

There is only one revision round and one consensus round. Hard tasks may need:

- multiple rebuttal cycles
- explicit test-driven repair loops
- broader candidate branching

### 5. No External Memory or Scratchpad State

The system only passes the current task context and prior attempts. There is no persistent shared memory across tasks or runs.

### 6. Prompt-Only Coordination

The twins collaborate only through exchanged natural-language analyses and code snippets. There is no structured error channel, no patch-based merge, and no explicit disagreement resolution mechanism.

## Recommended Next Steps

The most useful next experiments are likely:

### 1. Run Clean Held-Out Benchmark Slices

Start with:

- `SciCode` grouped eval slices
- `LiveCodeBench` grouped eval slices

Measure:

- final score
- solved count
- latency overhead
- per-stage improvement rates

### 2. Compare `local_eval` vs `consensus_first`

This separates two different questions:

- does collaboration improve final proposals
- does oracle-style hidden-test reranking improve outcomes further

Those should not be conflated.

### 3. Add Stage-Level Analysis

Useful derived metrics would include:

- draft-to-revision gain rate
- revision-to-consensus gain rate
- win rate of `Twin A` vs `Twin B`
- frequency of consensus regressions

### 4. Add Optional Parallel Execution

The draft and revision stages are structurally parallelizable. Running them concurrently would make the system faster without changing the protocol semantics.

### 5. Add More Controlled Twin Variants

Candidate ablations:

- same model paired with itself
- strong + weak model pair
- planner/reviewer asymmetric prompts
- bug-fix-only second pass

These would help determine whether gains come from diversity, raw model strength, or the protocol shape itself.

## Suggested Research Framing

If this enters a paper write-up, it should probably be framed conservatively:

"Twin" is a lightweight peer-collaboration baseline for code generation benchmarks. Two coding agents independently draft, cross-revise, and merge solutions. The setup is intended to measure the value of simple peer interaction before adding heavier orchestration mechanisms.

That framing is accurate and does not overclaim.

## Operational Notes

Example smoke-test command:

```bash
python experiments/dual_agents/run_twins.py run \
  --benchmark scicode \
  --scicode-source local_sample \
  --allow-shared-group-splits \
  --train-size 0 \
  --val-size 0 \
  --eval-size 1 \
  --agent-a-model rnj \
  --agent-b-model devstral \
  --selection-strategy local_eval
```

Example grouped SciCode run:

```bash
python experiments/dual_agents/run_twins.py run \
  --benchmark scicode \
  --train-groups 24 \
  --val-groups 8 \
  --eval-groups 16 \
  --agent-a-model rnj \
  --agent-b-model devstral \
  --selection-strategy local_eval
```

Example grouped LiveCodeBench run:

```bash
python experiments/dual_agents/run_twins.py run \
  --benchmark livecodebench \
  --train-groups 40 \
  --val-groups 10 \
  --eval-groups 40 \
  --agent-a-model rnj \
  --agent-b-model devstral \
  --selection-strategy local_eval
```

## Bottom Line

The current twin system is a deliberately small baseline:

- symmetric
- benchmark-compatible
- locally runnable
- artifact-rich

It is good enough to support immediate experiments and paper-progress notes. It is not yet the final form of a multi-agent coding protocol, but it creates a clean starting point for that work.
