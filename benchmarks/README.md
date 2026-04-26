# Benchmarks

This directory holds the benchmark-facing material that the experiment runners expect, but it does not try to publish every large local payload.

Tracked here:

- `baselines/`: lightweight aggregate benchmark reports and manifests
- `scicode/`: local path contract, small sample fixture, and small support files
- `livecodebench/`: local path contract, dataset README, loader snapshot, and metadata

Kept local and gitignored:

- `benchmarks/scicode/data/test_data.h5`
- `benchmarks/livecodebench/code_generation_lite/test*.jsonl`
- local dataset caches and generated run outputs

The runtime path contract is defined in `experiments/prompt_optimization/benchmark_adapters.py`:

- `benchmarks/scicode/data/test_data.h5`
- `benchmarks/scicode/test_data/first_problem.jsonl`
- `benchmarks/livecodebench/code_generation_lite/test.jsonl` through `test6.jsonl`

Do not rename those locations unless you also update the adapter constants.

Official references:

- SciCode repo: <https://github.com/scicode-bench/SciCode>
- SciCode paper: <https://arxiv.org/abs/2407.13168>
- SciCode dataset: <https://huggingface.co/datasets/SciCode1/SciCode>
- LiveCodeBench repo: <https://github.com/livecodebench/livecodebench>
- LiveCodeBench paper: <https://arxiv.org/abs/2403.07974>
- LiveCodeBench dataset: <https://huggingface.co/datasets/lighteval/code_generation_lite>
