# SciCode

This repo uses SciCode in a mixed public-plus-local way:

- task text is loaded from the public Hugging Face dataset `SciCode1/SciCode`
- hidden targets are expected locally at `benchmarks/scicode/data/test_data.h5`
- the small smoke-test fixture lives at `benchmarks/scicode/test_data/first_problem.jsonl`

Tracked here:

- `data/*.txt` support files from the local benchmark snapshot
- `test_data/first_problem.jsonl`
- this README

Kept local and gitignored:

- `data/test_data.h5`

That HDF5 file is the hidden-target payload used by the benchmark adapter. The public repo intentionally does not ship it.

If you are preparing a fresh clone, keep these exact paths:

- `benchmarks/scicode/data/test_data.h5`
- `benchmarks/scicode/test_data/first_problem.jsonl`

Official references:

- repo: <https://github.com/scicode-bench/SciCode>
- paper: <https://arxiv.org/abs/2407.13168>
- Hugging Face dataset: <https://huggingface.co/datasets/SciCode1/SciCode>
