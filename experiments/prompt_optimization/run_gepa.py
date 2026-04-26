from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import dspy

try:
    from .benchmark_adapters import BenchmarkTask, EvalResult, evaluate_task, load_livecodebench_tasks, load_scicode_tasks
except ImportError:
    from benchmark_adapters import BenchmarkTask, EvalResult, evaluate_task, load_livecodebench_tasks, load_scicode_tasks


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

DEFAULT_REFLECTION_PAIRS = {
    "rnj-1:latest": "devstral",
    "devstral-small-2:latest": "rnj",
}


class SolveCodeTask(dspy.Signature):
    """Solve the coding task and return only executable Python 3 code with no markdown fences."""

    task_prompt: str = dspy.InputField(desc="Full benchmark task statement and instructions.")
    starter_code: str = dspy.InputField(desc="Starter code or required function header to preserve when present.")
    solution: str = dspy.OutputField(desc="Executable Python 3 code only.")


class CodeSolver(dspy.Module):
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
    apply_default_reflection_settings(args)
    run_started_at_utc = utc_timestamp()
    run_started_perf = time.perf_counter()

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

    run_dir = None
    if args.command == "optimize":
        run_dir = make_run_dir(args)
        save_json(run_dir / "config.json", vars(args))
        save_json(run_dir / "split_manifest.json", summarize_splits(splits))

    student_lm = make_lm(
        args.model,
        api_base=args.api_base,
        api_key="",
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    reflection_lm = None
    reflection_model = getattr(args, "reflection_model", None)
    if reflection_model:
        reflection_lm = make_lm(
            reflection_model,
            api_base=args.reflection_api_base,
            api_key=args.reflection_api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            request_timeout_s=args.reflection_request_timeout_s,
            num_retries=args.reflection_num_retries,
        )
        preflight_reflection_lm(
            reflection_model,
            api_base=args.reflection_api_base,
            api_key=args.reflection_api_key,
            request_timeout_s=args.reflection_request_timeout_s,
            num_retries=args.reflection_num_retries,
        )

    dspy.configure(lm=student_lm)
    metric = build_metric(timeout_s=args.timeout_s)

    if args.command == "evaluate":
        program = CodeSolver()
        program.solve.set_lm(student_lm)
        if args.load_program:
            program.load(args.load_program, allow_pickle=True)
        summary = evaluate_program(program, splits["eval"], timeout_s=args.timeout_s)
        print(json.dumps(summary, indent=2))
        return

    baseline_program = CodeSolver()
    baseline_program.solve.set_lm(student_lm)
    baseline_eval_started = time.perf_counter()
    baseline_summary = evaluate_program(baseline_program, splits["eval"], timeout_s=args.timeout_s)
    baseline_eval_duration_s = time.perf_counter() - baseline_eval_started
    print("Baseline summary:")
    print(json.dumps(baseline_summary, indent=2))
    save_json(run_dir / "baseline_eval.json", baseline_summary)

    gepa_kwargs = {
        "metric": metric,
        "reflection_lm": reflection_lm,
        "num_threads": args.num_threads,
        "seed": args.seed,
        "log_dir": str(run_dir / "gepa_logs"),
    }
    if args.gepa_auto:
        gepa_kwargs["auto"] = args.gepa_auto
    elif args.max_full_evals is not None:
        gepa_kwargs["max_full_evals"] = args.max_full_evals
    elif args.max_metric_calls is not None:
        gepa_kwargs["max_metric_calls"] = args.max_metric_calls
    else:
        gepa_kwargs["max_metric_calls"] = 48

    optimizer = dspy.GEPA(**gepa_kwargs)

    optimization_started = time.perf_counter()
    try:
        optimized_program = optimizer.compile(
            CodeSolver(),
            trainset=splits["train"],
            valset=splits["val"],
        )
    except Exception as exc:
        if reflection_model:
            raise RuntimeError(
                "GEPA reflection failed using "
                f"{reflection_model} at {args.reflection_api_base}. "
                "Check that the teacher endpoint is reachable, or tune "
                "--reflection-request-timeout-s / --reflection-num-retries."
            ) from exc
        raise
    optimization_duration_s = time.perf_counter() - optimization_started
    optimized_program.save(run_dir / "optimized_program", save_program=True)

    optimized_eval_started = time.perf_counter()
    optimized_summary = evaluate_program(optimized_program, splits["eval"], timeout_s=args.timeout_s)
    optimized_eval_duration_s = time.perf_counter() - optimized_eval_started
    print("Optimized summary:")
    print(json.dumps(optimized_summary, indent=2))
    save_json(run_dir / "optimized_eval.json", optimized_summary)

    run_finished_at_utc = utc_timestamp()
    run_stats = build_run_stats(
        args=args,
        splits=splits,
        baseline_summary=baseline_summary,
        optimized_summary=optimized_summary,
        baseline_eval_duration_s=baseline_eval_duration_s,
        optimization_duration_s=optimization_duration_s,
        optimized_eval_duration_s=optimized_eval_duration_s,
        run_started_at_utc=run_started_at_utc,
        run_finished_at_utc=run_finished_at_utc,
        total_run_duration_s=time.perf_counter() - run_started_perf,
    )
    save_json(run_dir / "run_stats.json", run_stats)
    print("Run stats:")
    print(json.dumps(run_stats, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal DSPy GEPA runner for LiveCodeBench and SciCode.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("preview", "evaluate", "optimize"):
        subparser = subparsers.add_parser(command)
        add_shared_arguments(subparser)
        if command == "preview":
            continue
        subparser.add_argument("--model", required=True, help="Model alias or full DSPy model string.")
        subparser.add_argument("--api-base", default="http://localhost:11434")
        subparser.add_argument("--temperature", type=float, default=0.0)
        subparser.add_argument("--max-tokens", type=int, default=2048)
        subparser.add_argument("--timeout-s", type=float, default=12.0)
        subparser.add_argument("--load-program", type=Path)
        subparser.add_argument("--output-dir", type=Path)
        if command == "optimize":
            subparser.add_argument("--reflection-model", help="Reflection model alias or full model string.")
            subparser.add_argument("--reflection-api-base", default="http://localhost:11434")
            subparser.add_argument("--reflection-api-key", default="")
            subparser.add_argument("--reflection-request-timeout-s", type=float)
            subparser.add_argument("--reflection-num-retries", type=int, default=0)
            subparser.add_argument("--gepa-auto", choices=("light", "medium", "heavy"))
            subparser.add_argument("--max-full-evals", type=int)
            subparser.add_argument("--max-metric-calls", type=int)
            subparser.add_argument("--num-threads", type=int, default=1)
    return parser.parse_args()


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", choices=("livecodebench", "scicode"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0, help="Skip this many flattened tasks before loading.")
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


def apply_default_reflection_settings(args: argparse.Namespace) -> None:
    if args.command != "optimize":
        return

    if not args.reflection_model:
        resolved_student = MODEL_ALIASES.get(args.model, args.model)
        args.reflection_model = DEFAULT_REFLECTION_PAIRS.get(resolved_student)

    if args.reflection_request_timeout_s is None:
        args.reflection_request_timeout_s = 120.0 if is_ollama_api_base(args.reflection_api_base) else 15.0


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

    eval_groups, remaining = _take_groups_from_end(grouped, eval_size)
    val_groups, remaining = _take_groups_from_end(remaining, val_size)
    train_groups, remaining = _take_groups_from_start(remaining, train_size)

    if len(_flatten_groups(train_groups)) < train_size or len(_flatten_groups(val_groups)) < val_size or len(_flatten_groups(eval_groups)) < eval_size:
        raise ValueError(
            "Not enough disjoint benchmark groups for clean train/val/eval splits. "
            "Reduce the split sizes or pass --allow-shared-group-splits for debugging only."
        )

    return {
        "train": [make_example(task) for task in _flatten_groups(train_groups)],
        "val": [make_example(task) for task in _flatten_groups(val_groups)],
        "eval": [make_example(task) for task in _flatten_groups(eval_groups)],
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


def preflight_reflection_lm(
    model_name: str,
    *,
    api_base: str,
    api_key: str,
    request_timeout_s: float | None,
    num_retries: int,
) -> None:
    if is_ollama_api_base(api_base) and "/" not in model_name:
        return

    from openai import OpenAI

    try:
        client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=request_timeout_s,
            max_retries=num_retries,
        )
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
    except Exception as exc:
        raise RuntimeError(
            "Reflection LM preflight failed for "
            f"{model_name} at {api_base}. "
            "Check that the teacher endpoint is reachable, or tune "
            "--reflection-request-timeout-s / --reflection-num-retries."
        ) from exc


def build_metric(*, timeout_s: float):
    def metric(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None):
        result = evaluate_task(gold.task, getattr(pred, "solution", ""), timeout_s=timeout_s)
        if pred_name is not None or pred_trace is not None:
            return {"score": result.score, "feedback": result.feedback}
        return result.score

    return metric


def evaluate_program(program: dspy.Module, examples: list[dspy.Example], *, timeout_s: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_score = 0.0
    prediction_times: list[float] = []
    evaluation_times: list[float] = []
    total_times: list[float] = []
    solved_count = 0
    partial_count = 0

    for example in examples:
        prediction_started = time.perf_counter()
        prediction = program(task_prompt=example.task_prompt, starter_code=example.starter_code)
        prediction_time_s = time.perf_counter() - prediction_started
        evaluation_started = time.perf_counter()
        result = evaluate_task(example.task, getattr(prediction, "solution", ""), timeout_s=timeout_s)
        evaluation_time_s = time.perf_counter() - evaluation_started
        total_time_s = prediction_time_s + evaluation_time_s
        score = result.score
        feedback = result.feedback
        if score >= 1.0:
            solved_count += 1
        elif score > 0.0:
            partial_count += 1

        prediction_times.append(prediction_time_s)
        evaluation_times.append(evaluation_time_s)
        total_times.append(total_time_s)

        rows.append(
            {
                "task_id": example.task_id,
                "score": score,
                "passed": result.passed,
                "total_tests": result.total,
                "feedback": feedback,
                "first_failure": result.first_failure,
                "prediction_time_s": prediction_time_s,
                "evaluation_time_s": evaluation_time_s,
                "total_time_s": total_time_s,
                "reasoning": getattr(prediction, "reasoning", None),
                "solution_preview": getattr(prediction, "solution", "")[:400],
            }
        )
        total_score += score

    mean_score = total_score / len(examples) if examples else 0.0
    return {
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
    path.write_text(json.dumps(payload, indent=2, default=json_default))


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


def build_run_stats(
    *,
    args: argparse.Namespace,
    splits: dict[str, list[dspy.Example]],
    baseline_summary: dict[str, Any],
    optimized_summary: dict[str, Any],
    baseline_eval_duration_s: float,
    optimization_duration_s: float,
    optimized_eval_duration_s: float,
    run_started_at_utc: str,
    run_finished_at_utc: str,
    total_run_duration_s: float,
) -> dict[str, Any]:
    return {
        "benchmark": args.benchmark,
        "model": args.model,
        "reflection_model": args.reflection_model,
        "run_started_at_utc": run_started_at_utc,
        "run_finished_at_utc": run_finished_at_utc,
        "durations_s": {
            "baseline_eval": baseline_eval_duration_s,
            "optimization": optimization_duration_s,
            "optimized_eval": optimized_eval_duration_s,
            "total": total_run_duration_s,
        },
        "split_counts": {
            "train_examples": len(splits["train"]),
            "val_examples": len(splits["val"]),
            "eval_examples": len(splits["eval"]),
            "train_groups": count_example_groups(splits["train"]),
            "val_groups": count_example_groups(splits["val"]),
            "eval_groups": count_example_groups(splits["eval"]),
        },
        "baseline": summarize_eval_for_run_stats(baseline_summary),
        "optimized": summarize_eval_for_run_stats(optimized_summary),
        "improvement": {
            "mean_score_delta": optimized_summary["mean_score"] - baseline_summary["mean_score"],
            "solved_count_delta": optimized_summary["solved_count"] - baseline_summary["solved_count"],
            "partial_count_delta": optimized_summary["partial_count"] - baseline_summary["partial_count"],
            "row_deltas": compare_eval_rows(baseline_summary["rows"], optimized_summary["rows"]),
        },
    }


def summarize_eval_for_run_stats(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_score": summary["mean_score"],
        "num_examples": summary["num_examples"],
        "solved_count": summary["solved_count"],
        "partial_count": summary["partial_count"],
        "zero_score_count": summary["zero_score_count"],
        "time_stats_s": summary["time_stats_s"],
    }


def compare_eval_rows(baseline_rows: list[dict[str, Any]], optimized_rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_by_id = {row["task_id"]: row for row in baseline_rows}
    improved = 0
    regressed = 0
    unchanged = 0
    for optimized_row in optimized_rows:
        baseline_row = baseline_by_id.get(optimized_row["task_id"])
        if baseline_row is None:
            continue
        if optimized_row["score"] > baseline_row["score"]:
            improved += 1
        elif optimized_row["score"] < baseline_row["score"]:
            regressed += 1
        else:
            unchanged += 1
    return {"improved": improved, "regressed": regressed, "unchanged": unchanged}


def count_example_groups(examples: list[dspy.Example]) -> int:
    return len({_task_group_key(example.task) for example in examples})


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    main()
