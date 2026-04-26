from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import dspy

PROMPT_OPT_ROOT = Path(__file__).resolve().parents[1] / "prompt_optimization"
if str(PROMPT_OPT_ROOT) not in sys.path:
    sys.path.append(str(PROMPT_OPT_ROOT))

from benchmark_adapters import (  # noqa: E402
    BenchmarkTask,
    EvalResult,
    evaluate_task,
    load_livecodebench_tasks,
    load_scicode_tasks,
    strip_code_fences,
)


ARTIFACTS_ROOT = Path(__file__).resolve().parent / "artifacts"

MODEL_ALIASES = {
    "rnj": "rnj-1:latest",
    "rnj-1": "rnj-1:latest",
    "rnj-1:latest": "rnj-1:latest",
    "devstral": "devstral-small-2:latest",
    "devestral": "devstral-small-2:latest",
    "devstral-small-2": "devstral-small-2:latest",
    "devstral-small-2:latest": "devstral-small-2:latest",
}

BENCHMARK_DEFAULTS = {
    "scicode": {
        "max_tokens": 3072,
        "train_groups": 24,
        "val_groups": 8,
        "eval_groups": 16,
    },
    "livecodebench": {
        "max_tokens": 2048,
        "train_groups": 40,
        "val_groups": 10,
        "eval_groups": 40,
    },
}


class SolveCodeTask(dspy.Signature):
    """Solve the coding task and return only executable Python 3 code with no markdown fences."""

    task_prompt: str = dspy.InputField(desc="Full benchmark task statement and instructions.")
    starter_code: str = dspy.InputField(desc="Starter code or required function header to preserve when present.")
    solution: str = dspy.OutputField(desc="Executable Python 3 code only.")


class BaselineSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.solve = dspy.ChainOfThought(SolveCodeTask)

    def forward(self, task_prompt: str, starter_code: str = ""):
        try:
            return self.solve(task_prompt=task_prompt, starter_code=starter_code)
        except Exception as exc:
            return dspy.Prediction(
                reasoning=f"solver_error: {type(exc).__name__}: {exc}",
                solution="",
            )


def main() -> None:
    args = parse_args()
    apply_benchmark_runtime_defaults(args)

    if args.command == "preview":
        tasks = load_benchmark_tasks(args)
        print_preview(tasks)
        return

    tasks = load_benchmark_tasks(args)
    splits = build_splits(
        tasks,
        benchmark=args.benchmark,
        train_size=args.train_size,
        val_size=args.val_size,
        eval_size=args.eval_size,
        train_groups=args.train_groups,
        val_groups=args.val_groups,
        eval_groups=args.eval_groups,
        seed=args.seed,
        offset=args.offset,
        shuffle=args.shuffle,
        allow_shared_group_splits=args.allow_shared_group_splits,
    )

    run_dir = make_run_dir(args)
    save_json(run_dir / "config.json", vars(args))
    save_json(run_dir / "split_manifest.json", summarize_splits(splits))

    lm = make_lm(
        args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        request_timeout_s=args.request_timeout_s,
        num_retries=args.num_retries,
    )
    dspy.configure(lm=lm)

    program = BaselineSolver()
    program.solve.set_lm(lm)
    summary = evaluate_program(
        program,
        splits["eval"],
        timeout_s=args.timeout_s,
        run_dir=run_dir,
        model=args.model,
        benchmark=args.benchmark,
    )
    save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=json_default))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a plain single-pass baseline on SciCode and LiveCodeBench.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser("preview")
    add_shared_arguments(preview)

    evaluate = subparsers.add_parser("evaluate")
    add_shared_arguments(evaluate)
    evaluate.add_argument("--model", required=True, help="Model alias or full DSPy model string.")
    evaluate.add_argument("--api-base", default="http://localhost:11434")
    evaluate.add_argument("--api-key", default="")
    evaluate.add_argument("--temperature", type=float, default=0.0)
    evaluate.add_argument("--max-tokens", type=int)
    evaluate.add_argument("--request-timeout-s", type=float)
    evaluate.add_argument("--num-retries", type=int, default=3)
    evaluate.add_argument("--timeout-s", type=float, default=12.0, help="Hidden-test eval timeout per candidate.")
    evaluate.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", choices=("livecodebench", "scicode"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0, help="Skip this many grouped slices before splitting.")
    parser.add_argument("--train-size", type=int, default=8)
    parser.add_argument("--val-size", type=int, default=4)
    parser.add_argument("--eval-size", type=int, default=4)
    parser.add_argument("--train-groups", type=int)
    parser.add_argument("--val-groups", type=int)
    parser.add_argument("--eval-groups", type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--livecodebench-release", default="release_latest")
    parser.add_argument("--scicode-source", choices=("huggingface", "local_sample"), default="huggingface")
    parser.add_argument("--scicode-split", default="test")
    parser.add_argument("--allow-shared-group-splits", action="store_true")


def apply_benchmark_runtime_defaults(args: argparse.Namespace) -> None:
    defaults = BENCHMARK_DEFAULTS[args.benchmark]
    if args.command == "evaluate":
        if args.max_tokens is None:
            args.max_tokens = defaults["max_tokens"]
        if args.request_timeout_s is None:
            args.request_timeout_s = 120.0 if is_ollama_api_base(args.api_base) else 30.0

    if args.train_groups is None and args.val_groups is None and args.eval_groups is None and not args.allow_shared_group_splits:
        args.train_groups = defaults["train_groups"]
        args.val_groups = defaults["val_groups"]
        args.eval_groups = defaults["eval_groups"]


def load_benchmark_tasks(args: argparse.Namespace) -> list[BenchmarkTask]:
    if args.benchmark == "livecodebench":
        tasks = load_livecodebench_tasks(
            release_tag=args.livecodebench_release,
            offset=0,
            limit=None if args.command != "preview" else max(args.train_size + args.val_size + args.eval_size, 3),
        )
    else:
        tasks = load_scicode_tasks(
            source=args.scicode_source,
            split=args.scicode_split,
            offset=0,
            limit=None if args.command != "preview" else max(args.train_size + args.val_size + args.eval_size, 3),
        )

    if args.command == "preview":
        tasks = tasks[args.offset:]
        if args.shuffle:
            random.Random(args.seed).shuffle(tasks)
    return tasks


def build_splits(
    tasks: list[BenchmarkTask],
    *,
    benchmark: str,
    train_size: int,
    val_size: int,
    eval_size: int,
    train_groups: int | None = None,
    val_groups: int | None = None,
    eval_groups: int | None = None,
    seed: int,
    offset: int = 0,
    shuffle: bool = False,
    allow_shared_group_splits: bool = False,
) -> dict[str, list[dspy.Example]]:
    if allow_shared_group_splits:
        if any(value is not None for value in (train_groups, val_groups, eval_groups)):
            raise ValueError("Group-count splits cannot be combined with --allow-shared-group-splits.")
        flat_tasks = tasks[offset:]
        if shuffle:
            random.Random(seed).shuffle(flat_tasks)
        return _build_flat_slices(flat_tasks, train_size=train_size, val_size=val_size, eval_size=eval_size)

    grouped = _ordered_task_groups(tasks, benchmark=benchmark, seed=seed, shuffle=shuffle)
    grouped = grouped[offset:]
    if not grouped:
        raise ValueError("No grouped tasks remain after applying the offset.")

    if any(value is not None for value in (train_groups, val_groups, eval_groups)):
        if None in (train_groups, val_groups, eval_groups):
            raise ValueError("Provide all of --train-groups, --val-groups, and --eval-groups together.")
        return _build_group_count_slices(
            grouped,
            train_groups=train_groups,
            val_groups=val_groups,
            eval_groups=eval_groups,
        )

    eval_groups_slice, remaining = _take_groups_from_end(grouped, eval_size)
    val_groups_slice, remaining = _take_groups_from_end(remaining, val_size)
    train_groups_slice, _ = _take_groups_from_start(remaining, train_size)

    if (
        len(_flatten_groups(train_groups_slice)) < train_size
        or len(_flatten_groups(val_groups_slice)) < val_size
        or len(_flatten_groups(eval_groups_slice)) < eval_size
    ):
        raise ValueError(
            "Not enough disjoint benchmark groups for clean train/val/eval splits. "
            "Reduce the split sizes or pass --allow-shared-group-splits for debugging only."
        )

    return {
        "train": [make_example(task) for task in _flatten_groups(train_groups_slice)],
        "val": [make_example(task) for task in _flatten_groups(val_groups_slice)],
        "eval": [make_example(task) for task in _flatten_groups(eval_groups_slice)],
    }


def _build_flat_slices(tasks: list[BenchmarkTask], *, train_size: int, val_size: int, eval_size: int) -> dict[str, list[dspy.Example]]:
    total_needed = train_size + val_size + eval_size
    if len(tasks) < total_needed:
        raise ValueError(f"Requested {total_needed} tasks, but only loaded {len(tasks)}.")

    cursor = 0
    splits: dict[str, list[dspy.Example]] = {}
    for name, size in (("train", train_size), ("val", val_size), ("eval", eval_size)):
        subset = tasks[cursor : cursor + size]
        splits[name] = [make_example(task) for task in subset]
        cursor += size
    return splits


def _build_group_count_slices(
    grouped: list[list[BenchmarkTask]],
    *,
    train_groups: int,
    val_groups: int,
    eval_groups: int,
) -> dict[str, list[dspy.Example]]:
    total_needed = train_groups + val_groups + eval_groups
    if len(grouped) < total_needed:
        raise ValueError(
            f"Requested {total_needed} benchmark groups, but only {len(grouped)} groups are available."
        )

    eval_groups_slice = grouped[-eval_groups:] if eval_groups else []
    remaining = grouped[:-eval_groups] if eval_groups else grouped[:]
    val_groups_slice = remaining[-val_groups:] if val_groups else []
    remaining = remaining[:-val_groups] if val_groups else remaining
    train_groups_slice = remaining[:train_groups]

    return {
        "train": [make_example(task) for task in _flatten_groups(train_groups_slice)],
        "val": [make_example(task) for task in _flatten_groups(val_groups_slice)],
        "eval": [make_example(task) for task in _flatten_groups(eval_groups_slice)],
    }


def _ordered_task_groups(tasks: list[BenchmarkTask], *, benchmark: str, seed: int, shuffle: bool) -> list[list[BenchmarkTask]]:
    grouped: dict[Any, list[BenchmarkTask]] = defaultdict(list)
    for task in tasks:
        grouped[_task_group_key(task)].append(task)

    groups = list(grouped.values())
    if benchmark == "livecodebench":
        if shuffle:
            raise ValueError("Shuffle is disabled for LiveCodeBench clean splits because eval should stay temporally held out.")
        groups.sort(key=lambda group: (_livecodebench_group_sort_key(group[0]), group[0].task_id))
    else:
        groups.sort(key=lambda group: group[0].metadata.get("problem_id", group[0].task_id))
        if shuffle:
            random.Random(seed).shuffle(groups)
    return groups


def _task_group_key(task: BenchmarkTask) -> Any:
    if task.benchmark == "livecodebench":
        return (task.metadata.get("contest_date"), task.metadata.get("contest_id"))
    return task.metadata.get("problem_id")


def _livecodebench_group_sort_key(task: BenchmarkTask) -> tuple[str, str]:
    return (task.metadata.get("contest_date", ""), str(task.metadata.get("contest_id", "")))


def _take_groups_from_start(groups: list[list[BenchmarkTask]], min_tasks: int) -> tuple[list[list[BenchmarkTask]], list[list[BenchmarkTask]]]:
    taken: list[list[BenchmarkTask]] = []
    total = 0
    index = 0
    while index < len(groups) and total < min_tasks:
        taken.append(groups[index])
        total += len(groups[index])
        index += 1
    return taken, groups[index:]


def _take_groups_from_end(groups: list[list[BenchmarkTask]], min_tasks: int) -> tuple[list[list[BenchmarkTask]], list[list[BenchmarkTask]]]:
    taken: list[list[BenchmarkTask]] = []
    total = 0
    index = len(groups)
    while index > 0 and total < min_tasks:
        index -= 1
        taken.insert(0, groups[index])
        total += len(groups[index])
    return taken, groups[:index]


def _flatten_groups(groups: list[list[BenchmarkTask]]) -> list[BenchmarkTask]:
    return [task for group in groups for task in group]


def make_example(task: BenchmarkTask) -> dspy.Example:
    return dspy.Example(
        task=task,
        task_id=task.task_id,
        task_prompt=task.prompt,
        starter_code=task.starter_code,
    ).with_inputs("task_prompt", "starter_code")


def make_lm(
    model_name: str,
    *,
    api_base: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    request_timeout_s: float | None = None,
    num_retries: int = 3,
) -> dspy.LM:
    if "/" in model_name:
        resolved = model_name
    elif model_name in MODEL_ALIASES or is_ollama_api_base(api_base):
        resolved_name = MODEL_ALIASES.get(model_name, model_name)
        resolved = f"ollama_chat/{resolved_name}"
    else:
        resolved = model_name
    return dspy.LM(
        resolved,
        api_base=api_base,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=request_timeout_s,
        num_retries=num_retries,
    )


def is_ollama_api_base(api_base: str) -> bool:
    parsed = urlparse(api_base)
    return parsed.port == 11434


def evaluate_program(
    program: dspy.Module,
    examples: list[dspy.Example],
    *,
    timeout_s: float,
    run_dir: Path,
    model: str,
    benchmark: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_score = 0.0
    prediction_times: list[float] = []
    evaluation_times: list[float] = []
    total_times: list[float] = []
    solved_count = 0
    partial_count = 0
    task_artifacts_dir = run_dir / "task_artifacts"
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)

    started_at_utc = utc_timestamp()
    started_perf = time.perf_counter()

    for index, example in enumerate(examples, start=1):
        print(f"[task {index}/{len(examples)}] {example.task_id}: running baseline", flush=True)
        prediction_started = time.perf_counter()
        prediction = program(task_prompt=example.task_prompt, starter_code=example.starter_code)
        prediction_time_s = time.perf_counter() - prediction_started

        reasoning = str(getattr(prediction, "reasoning", "") or "")
        solution = strip_code_fences(str(getattr(prediction, "solution", "") or ""))

        evaluation_started = time.perf_counter()
        result = evaluate_task(example.task, solution, timeout_s=timeout_s)
        evaluation_time_s = time.perf_counter() - evaluation_started
        total_time_s = prediction_time_s + evaluation_time_s
        score = result.score
        if score >= 1.0:
            solved_count += 1
        elif score > 0.0:
            partial_count += 1

        prediction_times.append(prediction_time_s)
        evaluation_times.append(evaluation_time_s)
        total_times.append(total_time_s)

        row = {
            "task_id": example.task_id,
            "score": score,
            "passed": result.passed,
            "total_tests": result.total,
            "feedback": result.feedback,
            "first_failure": result.first_failure,
            "prediction_time_s": prediction_time_s,
            "evaluation_time_s": evaluation_time_s,
            "total_time_s": total_time_s,
            "solution_preview": solution[:400],
            "reasoning_preview": reasoning[:400],
        }
        rows.append(row)
        total_score += score

        save_json(
            task_artifacts_dir / f"{sanitize_slug(example.task_id)}.json",
            {
                "task_id": example.task_id,
                "benchmark": example.task.benchmark,
                "model": model,
                "task_prompt": example.task_prompt,
                "starter_code": example.starter_code,
                "required_dependencies": example.task.required_dependencies,
                "eval_result": result,
                "prediction_time_s": prediction_time_s,
                "evaluation_time_s": evaluation_time_s,
                "total_time_s": total_time_s,
                "reasoning": reasoning,
                "solution": solution,
            },
        )
        print(f"[task {index}/{len(examples)}] {example.task_id}: score={score:.3f}", flush=True)

    mean_score = total_score / len(examples) if examples else 0.0
    return {
        "benchmark": benchmark,
        "model": model,
        "run_started_at_utc": started_at_utc,
        "run_finished_at_utc": utc_timestamp(),
        "wall_time_s": time.perf_counter() - started_perf,
        "mean_score": mean_score,
        "num_examples": len(examples),
        "solved_count": solved_count,
        "partial_count": partial_count,
        "zero_score_count": len(examples) - solved_count - partial_count,
        "time_stats_s": {
            "prediction": summarize_numeric_series(prediction_times),
            "evaluation": summarize_numeric_series(evaluation_times),
            "total": summarize_numeric_series(total_times),
        },
        "rows": rows,
    }


def make_run_dir(args: argparse.Namespace) -> Path:
    if getattr(args, "output_dir", None):
        run_dir = Path(args.output_dir)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    model_slug = sanitize_slug(args.model)
    run_dir = ARTIFACTS_ROOT / args.benchmark / model_slug / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BenchmarkTask):
        return asdict(value)
    if isinstance(value, EvalResult):
        return asdict(value)
    raise TypeError(f"Unsupported JSON type: {type(value)!r}")


def summarize_splits(splits: dict[str, list[dspy.Example]]) -> dict[str, Any]:
    return {
        name: [
            {
                "task_id": example.task.task_id,
                "benchmark": example.task.benchmark,
                "metadata": example.task.metadata,
            }
            for example in examples
        ]
        for name, examples in splits.items()
    }


def print_preview(tasks: list[BenchmarkTask]) -> None:
    print(f"Loaded {len(tasks)} tasks.")
    for task in tasks[:3]:
        print("=" * 80)
        print(task.task_id, task.metadata)
        print(task.prompt[:1200])
        print()


def sanitize_slug(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def summarize_numeric_series(values: list[float]) -> dict[str, float]:
    if not values:
        return {"sum": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "sum": sum(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    main()
