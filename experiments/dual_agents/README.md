# Dual Agents

This experiment folder sets up a simple "twin" protocol for two coding LLMs to solve the benchmark tasks together.

The current implementation is intentionally small and symmetric:

1. `Twin A` drafts a solution.
2. `Twin B` drafts a solution.
3. Each twin revises after reading the other twin's draft.
4. Each twin performs a final consensus pass over the two revised solutions.
5. A selector chooses the final candidate.

The runner reuses the benchmark loaders and hidden-test evaluator from `experiments/prompt_optimization/benchmark_adapters.py`.

## Files

- `run_twins.py`: CLI entrypoint for previewing tasks or running the twin protocol on a benchmark split.
- `requirements.txt`: minimal dependencies for this experiment.

## Twin Protocol

The protocol is symmetric on purpose. Neither model is a teacher or judge by default. Both are treated as peers:

- independent draft
- cross-review and revision
- final merge / consensus

Two selection strategies are supported:

- `local_eval`: evaluate every candidate on the local hidden tests and choose the best candidate.
  This is the strongest setting for local problem solving, but it is not a contamination-safe evaluation protocol because it uses the hidden tests to choose among candidates on the same task.
- `consensus_first`: return the first consensus candidate without hidden-test reranking.
  This is a cleaner collaboration protocol when you do not want oracle-style candidate selection.

## Quick Start

Preview tasks:

```bash
python experiments/dual_agents/run_twins.py preview \
  --benchmark scicode \
  --eval-size 2
```

Run a one-task smoke test on the local SciCode sample:

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

Run the twin protocol on held-out grouped SciCode slices:

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

Run the twin protocol on held-out grouped LiveCodeBench slices:

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

## Outputs

Each run writes artifacts under `experiments/dual_agents/artifacts/`:

- `config.json`
- `split_manifest.json`
- `summary.json`
- `task_artifacts/<task_id>.json`

Each task artifact records:

- every draft / revision / consensus attempt
- per-candidate hidden-test score when `local_eval` is used
- which candidate was selected

## Notes

- The default setup assumes local Ollama models on `http://localhost:11434`.
- Model aliases currently match the prompt-optimization runner:
  - `rnj` -> `rnj-1:latest`
  - `devstral` -> `devstral-small-2:latest`
- This is a simple baseline protocol, not a full multi-agent planner. The point is to get a working twin-agent benchmark harness in place first.
