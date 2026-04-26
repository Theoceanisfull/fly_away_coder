# LiveCodeBench

This repo expects a local `code_generation_lite` snapshot under:

- `benchmarks/livecodebench/code_generation_lite/`

The benchmark adapter reads:

- `test.jsonl`
- `test2.jsonl`
- `test3.jsonl`
- `test4.jsonl`
- `test5.jsonl`
- `test6.jsonl`

Those release files are large and are intentionally gitignored in the public repo.

Tracked here:

- `code_generation_lite/README.md`
- `code_generation_lite/code_generation_lite.py`
- `code_generation_lite/.gitattributes`
- any lightweight metadata or images already bundled with the local snapshot

Kept local and gitignored:

- `code_generation_lite/test*.jsonl`
- local cache directories

The default harness setting is `release_latest`, which expands to all six release files above.

Official references:

- repo: <https://github.com/livecodebench/livecodebench>
- paper: <https://arxiv.org/abs/2403.07974>
- Hugging Face dataset: <https://huggingface.co/datasets/lighteval/code_generation_lite>
