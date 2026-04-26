# Local Coding Benchmark Harness

This repository is a local DSPy-based experiment harness for comparing inference-time coding methods on two hidden-test benchmarks:

- `SciCode`
- `LiveCodeBench`

The design goal is controlled comparison. The repo keeps the benchmark loaders, grouped split construction, hidden-test evaluators, local model wiring, and artifact schema fixed while swapping the inference strategy.

This public-facing cleanup keeps:

- code
- documentation
- lightweight benchmark metadata and sample fixtures
- aggregate baseline reports

And it intentionally keeps out of Git:

- large benchmark payloads
- hidden-test data files
- generated run artifacts
- local logs

## What Is Implemented

| Method | Entry point | Core DSPy or runtime idea | Needs Deno? | Status on the shared split family |
| --- | --- | --- | --- | --- |
| Baseline | `experiments/baseline/run_baseline.py` | `dspy.ChainOfThought` | No | Complete |
| GEPA | `experiments/prompt_optimization/run_gepa.py` | `dspy.GEPA` over the baseline solver | No | Complete |
| CodeAct | `experiments/CodeAct/run_codeact.py` | `dspy.CodeAct` with helper tools | Yes | Partial matrix complete |
| ReAct | `experiments/ReAct/run_react.py` | `dspy.ReAct` with benchmark-aware static tools | No | Complete |
| ProgramOfThought | `experiments/ProgramOfThought/run_pot.py` | `dspy.ProgramOfThought` with internal Python execution | Yes | SciCode complete, LiveCodeBench in progress as of `2026-04-26` |
| Dual agents | `experiments/dual_agents/run_twins.py` | symmetric two-model draft / revise / consensus protocol | No | Complete benchmark-level run |

## Current Results Snapshot

As of `2026-04-26`, the strongest completed single-agent result on each benchmark/model pair is:

| Benchmark | Model | Best completed single-agent method | Mean score | Solved |
| --- | --- | --- | ---: | ---: |
| SciCode | `rnj` | ReAct | `0.1764` | `12 / 94` |
| SciCode | `devstral` | GEPA | `0.2429` | `20 / 94` |
| LiveCodeBench | `rnj` | CodeAct | `0.1689` | `26 / 175` |
| LiveCodeBench | `devstral` | ReAct | `0.2119` | `34 / 175` |

The strongest overall raw numbers in the repo come from the twin-agent run with `local_eval` reranking:

- SciCode: `0.2580`, `20 / 94`
- LiveCodeBench: `0.2440`, `40 / 175`

That twin result is useful as a local upper bound, but it is not a clean pass-at-one comparison because hidden tests participate in final candidate selection.

ProgramOfThought is still in flight. The timeout-guarded standalone rerun has already completed three sub-runs:

- SciCode + `rnj`: `0.1268`, `9 / 94`
- SciCode + `devstral`: `0.2376`, `18 / 94`
- LiveCodeBench + `rnj`: `0.0376`, `6 / 175`

The remaining run, `livecodebench + devstral`, was still running when this README was updated on `2026-04-26`.

For the full writeups, see:

- [Baseline results](experiment_baseline_results.md)
- [Experiment 01: GEPA](experiment_01_results.md)
- [Experiment 02: CodeAct](experiment_02_results.md)
- [Experiment 03: Dual agents](experiment_03_results.md)
- [Experiment 04: ReAct](experiment_04_results.md)
- [Comparative analysis](comparative_analysis.md)
- [Academic documentation](academic_documentation.md)

## Hardware And Runtime

The runs summarized in this repo were executed on the local benchmark VM configuration visible on `2026-04-26`:

| Component | Observed configuration |
| --- | --- |
| GPU | `NVIDIA L40S` |
| GPU memory | `46068 MiB` VRAM reported by `nvidia-smi` |
| NVIDIA driver | `580.126.20` |
| CPU visible to the VM | `15` vCPUs on an `AMD EPYC 7713 64-Core Processor` host |
| System memory visible to the VM | `59 GiB` RAM |
| Model serving | Ollama at `http://localhost:11434` |

The most useful wall-clock numbers for the shared held-out split experiments are:

| Run | Scope | Wall-clock duration |
| --- | --- | ---: |
| GEPA full sweep | 4 benchmark/model pairs | `9h 03m 21s` |
| CodeAct sweep | 2 benchmark/model pairs | `1h 49m 44s` |
| Dual-agent sweep | SciCode + LiveCodeBench benchmark runs | `6h 37m 51s` |
| ReAct sweep | 4 benchmark/model pairs | `9h 32m 55s` |
| ProgramOfThought `scicode__rnj` | 94 examples | `5m 03s` |
| ProgramOfThought `scicode__devstral` | 94 examples | `5m 45s` |
| ProgramOfThought `livecodebench__rnj` | 175 examples | `6m 39s` |
| ProgramOfThought `livecodebench__devstral` | 175 examples | still running as of `2026-04-26` |

Two notes matter when reading these durations:

- The shared split family in this repo uses `94` held-out SciCode sub-steps and `175` held-out LiveCodeBench tasks, not the full upstream benchmark sizes.
- The published benchmark baselines under `benchmarks/baselines/` come from larger full-benchmark runs and therefore have materially longer wall-clock times than the held-out split experiments summarized above.

## Repository Layout

```text
benchmarks/
  baselines/                  Lightweight aggregate benchmark reports
  livecodebench/              Local dataset snapshot contract and docs
  scicode/                    Local hidden-target contract and sample fixture
experiments/
  baseline/                   Plain one-shot solver
  prompt_optimization/        GEPA prompt-optimization harness
  CodeAct/                    CodeAct experiment
  ReAct/                      ReAct experiment
  ProgramOfThought/           Program-of-Thought experiment
  dual_agents/                Twin-agent protocol
  run_serial_sweeps.py        ReAct -> ProgramOfThought serial runner
```

The benchmark adapter constants live in [experiments/prompt_optimization/benchmark_adapters.py](experiments/prompt_optimization/benchmark_adapters.py). The repo expects the benchmark payloads to appear at those exact paths.

## Setup

### 1. Python Environment

The repo now includes a top-level dependency file that covers the shared runtime across the experiment folders:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer, you can still install per-experiment requirement files from each subdirectory.

### 2. Install And Run Ollama

Official references:

- Ollama docs: <https://docs.ollama.com/>
- Linux install guide: <https://docs.ollama.com/linux>

This repo assumes an Ollama-compatible endpoint at:

- `http://localhost:11434`

Start Ollama and download the exact model tags used in the experiments:

```bash
ollama serve
ollama pull rnj-1:latest
ollama pull devstral-small-2:latest
```

Official model pages:

- `rnj-1`: <https://ollama.com/library/rnj-1>
- `devstral-small-2`: <https://ollama.com/library/devstral-small-2>

The local aliases used throughout the codebase are:

- `rnj` -> `rnj-1:latest`
- `devstral` -> `devstral-small-2:latest`

### 3. Install Deno For CodeAct And ProgramOfThought

Only the CodeAct and ProgramOfThought runners require Deno, because they use DSPy's Deno-backed Python execution sandbox.

Official Deno install docs:

- <https://docs.deno.com/runtime/manual/getting_started/installation>

After installation:

```bash
deno --version
```

### 4. Populate The Benchmark Data

The repo keeps large benchmark payloads out of Git, but the runtime paths are intentionally stable. Do not rename these directories unless you also update the adapter constants.

Create the expected directories if needed:

```bash
mkdir -p benchmarks/scicode/data
mkdir -p benchmarks/scicode/test_data
mkdir -p benchmarks/livecodebench/code_generation_lite
```

Expected local files:

- `benchmarks/scicode/data/test_data.h5`
- `benchmarks/scicode/test_data/first_problem.jsonl`
- `benchmarks/livecodebench/code_generation_lite/test.jsonl`
- `benchmarks/livecodebench/code_generation_lite/test2.jsonl`
- `benchmarks/livecodebench/code_generation_lite/test3.jsonl`
- `benchmarks/livecodebench/code_generation_lite/test4.jsonl`
- `benchmarks/livecodebench/code_generation_lite/test5.jsonl`
- `benchmarks/livecodebench/code_generation_lite/test6.jsonl`

SciCode notes:

- task text is loaded from the Hugging Face dataset `SciCode1/SciCode`
- this repo still expects the hidden-target HDF5 file locally at `benchmarks/scicode/data/test_data.h5`
- the small `first_problem.jsonl` fixture is used for smoke tests and examples

LiveCodeBench notes:

- the repo expects a local snapshot of the `code_generation_lite` release files
- the default release tag in the harness is `release_latest`

Source references:

- SciCode repo: <https://github.com/scicode-bench/SciCode>
- SciCode paper: <https://arxiv.org/abs/2407.13168>
- SciCode Hugging Face dataset: <https://huggingface.co/datasets/SciCode1/SciCode>
- LiveCodeBench repo: <https://github.com/livecodebench/livecodebench>
- LiveCodeBench paper: <https://arxiv.org/abs/2403.07974>
- LiveCodeBench Hugging Face dataset: <https://huggingface.co/datasets/lighteval/code_generation_lite>

## Quickstart

Preview a couple of tasks:

```bash
python experiments/baseline/run_baseline.py preview \
  --benchmark scicode \
  --eval-size 2
```

Run the plain baseline on SciCode:

```bash
python experiments/baseline/run_baseline.py evaluate \
  --benchmark scicode \
  --model devstral
```

Run the GEPA sweep:

```bash
python experiments/prompt_optimization/run_full_sweep.py
```

Run the ReAct sweep:

```bash
python experiments/ReAct/run_full_sweep.py
```

Run ProgramOfThought:

```bash
python experiments/ProgramOfThought/run_full_sweep.py
```

Run the twin-agent benchmark:

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

Run the serial `ReAct -> ProgramOfThought` pipeline:

```bash
python experiments/run_serial_sweeps.py
```

Method-specific notes live in:

- [Baseline README](experiments/baseline/README.md)
- [Prompt optimization README](experiments/prompt_optimization/README.md)
- [CodeAct README](experiments/CodeAct/README.md)
- [ReAct README](experiments/ReAct/README.md)
- [ProgramOfThought README](experiments/ProgramOfThought/README.md)
- [Dual agents README](experiments/dual_agents/README.md)

## Related Work And References

The most relevant external references for this repo are:

- DSPy paper: <https://arxiv.org/abs/2310.03714>
- DSPy docs: <https://dspy.ai/>
- DSPy repo: <https://github.com/stanfordnlp/dspy>
- GEPA paper: <https://arxiv.org/abs/2507.19457>
- GEPA OpenReview entry: <https://openreview.net/forum?id=RQm2KQTM5r>
- ReAct: <https://arxiv.org/abs/2210.03629>
- Program of Thoughts Prompting: <https://arxiv.org/abs/2211.12588>
- CodeAct, "Executable Code Actions Elicit Better LLM Agents": <https://arxiv.org/abs/2402.01030>
- LiveCodeBench: <https://arxiv.org/abs/2403.07974>
- SciCode: <https://arxiv.org/abs/2407.13168>
- Multiagent Debate: <https://arxiv.org/abs/2305.14325>
- Reflexion: <https://arxiv.org/abs/2303.11366>
- AgentCoder: <https://arxiv.org/abs/2312.13010>
- MapCoder: <https://arxiv.org/abs/2405.11403>

These are especially relevant for interpreting the implemented method families:

- GEPA for prompt optimization
- ReAct for tool-using reasoning loops
- Program of Thoughts for computation-assisted reasoning
- CodeAct for executable action spaces
- Multiagent Debate, Reflexion, AgentCoder, and MapCoder for the twin-agent and iterative-refinement context

## Repo Hygiene

The `.gitignore` now keeps the following local-only by default:

- large benchmark payloads such as `test_data.h5` and `code_generation_lite/test*.jsonl`
- experiment artifact trees under `experiments/**/artifacts/`
- serial run outputs under `experiments/serial_runs/`
- Python caches and local scratch directories

That split is intentional. The public repo stays small and inspectable, while the local benchmark payloads and run outputs remain usable on disk at the exact paths the runners expect.
