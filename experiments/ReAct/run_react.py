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
        "max_iters": 6,
        "max_tokens": 3072,
        "train_groups": 24,
        "val_groups": 8,
        "eval_groups": 16,
    },
    "livecodebench": {
        "max_iters": 5,
        "max_tokens": 2048,
        "train_groups": 40,
        "val_groups": 10,
        "eval_groups": 40,
    },
}


class SolveWithReAct(dspy.Signature):
    """Solve the benchmark task by using tools to inspect the contract and self-repair candidate code.

    The final `solution` must be executable Python 3 code only.
    Preserve the required interface from starter_code when present.
    Do not include markdown fences or any explanation outside the submitted code.
    """

    benchmark: str = dspy.InputField(desc="Benchmark name: scicode or livecodebench.")
    eval_kind: str = dspy.InputField(desc="Evaluation kind such as scicode_step, stdin, or functional.")
    benchmark_notes: str = dspy.InputField(desc="Benchmark-specific notes and evaluation contract.")
    task_prompt: str = dspy.InputField(desc="Full task statement.")
    starter_code: str = dspy.InputField(desc="Starter code or required function header to preserve when present.")
    required_dependencies: str = dspy.InputField(desc="Required dependency block when provided by the benchmark.")
    solution: str = dspy.OutputField(desc="Executable Python 3 code only, with no markdown fences.")


class ReActSolver(dspy.Module):
    def __init__(self, *, benchmark: str, max_iters: int):
        super().__init__()
        self.benchmark = benchmark
        self.solve = dspy.ReAct(
            SolveWithReAct,
            tools=build_tools(benchmark),
            max_iters=max_iters,
        )
        self.fallback = dspy.ChainOfThought(SolveWithReAct)

    def forward(
        self,
        *,
        benchmark: str,
        eval_kind: str,
        benchmark_notes: str,
        task_prompt: str,
        starter_code: str,
        required_dependencies: str,
    ):
        fallback_reason: str | None = None
        try:
            pred = self.solve(
                benchmark=benchmark,
                eval_kind=eval_kind,
                benchmark_notes=benchmark_notes,
                task_prompt=task_prompt,
                starter_code=starter_code,
                required_dependencies=required_dependencies,
            )
        except Exception as exc:
            fallback_reason = f"react_error: {type(exc).__name__}: {exc}"
            pred = self._run_fallback(
                benchmark=benchmark,
                eval_kind=eval_kind,
                benchmark_notes=benchmark_notes,
                task_prompt=task_prompt,
                starter_code=starter_code,
                required_dependencies=required_dependencies,
                fallback_reason=fallback_reason,
            )

        solution = extract_solution_from_prediction(pred)
        reasoning = getattr(pred, "reasoning", None)
        trajectory = getattr(pred, "trajectory", {}) or {}

        if not solution:
            fallback_reason = fallback_reason or "react_empty_solution"
            pred = self._run_fallback(
                benchmark=benchmark,
                eval_kind=eval_kind,
                benchmark_notes=benchmark_notes,
                task_prompt=task_prompt,
                starter_code=starter_code,
                required_dependencies=required_dependencies,
                fallback_reason=fallback_reason,
            )
            solution = extract_solution_from_prediction(pred)
            reasoning = getattr(pred, "reasoning", reasoning)
            fallback_trajectory = getattr(pred, "trajectory", {}) or {}
            trajectory = {
                "fallback_used": True,
                "fallback_reason": fallback_reason,
                "react_trajectory": trajectory,
                **fallback_trajectory,
            }
        elif fallback_reason:
            trajectory = {
                "fallback_used": True,
                "fallback_reason": fallback_reason,
                **trajectory,
            }

        return dspy.Prediction(
            solution=solution,
            reasoning=reasoning,
            trajectory=trajectory,
        )

    def _run_fallback(
        self,
        *,
        benchmark: str,
        eval_kind: str,
        benchmark_notes: str,
        task_prompt: str,
        starter_code: str,
        required_dependencies: str,
        fallback_reason: str,
    ):
        derived_scaffold = (
            make_stdio_solution_scaffold(task_prompt)
            if benchmark == "livecodebench" and eval_kind == "stdin"
            else make_solution_scaffold(starter_code, required_dependencies)
        )
        tool_context = [
            "Fallback mode: the ReAct tool loop failed or returned no code. Use the deterministic context below directly.",
            f"fallback_reason={fallback_reason}",
            "playbook:",
            benchmark_playbook(benchmark, eval_kind, starter_code, required_dependencies),
            "task_contract:",
            summarize_task_contract(benchmark, eval_kind, task_prompt, starter_code, required_dependencies),
            "solution_contract_json:",
            solution_contract_json(benchmark, eval_kind, starter_code),
            "constraints_json:",
            extract_constraints_json(task_prompt),
            "examples_json:",
            extract_examples_json(task_prompt),
        ]
        if derived_scaffold:
            tool_context.extend(["suggested_scaffold:", derived_scaffold])

        return self.fallback(
            benchmark=benchmark,
            eval_kind=eval_kind,
            benchmark_notes=benchmark_notes + "\n\n" + "\n".join(tool_context),
            task_prompt=task_prompt,
            starter_code=starter_code,
            required_dependencies=required_dependencies,
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

    program = ReActSolver(benchmark=args.benchmark, max_iters=args.max_iters)
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
    parser = argparse.ArgumentParser(description="Run DSPy ReAct on SciCode and LiveCodeBench.")
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
    evaluate.add_argument("--max-iters", type=int)
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
        if args.max_iters is None:
            args.max_iters = defaults["max_iters"]
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
    benchmark_notes = build_benchmark_notes(task)
    return dspy.Example(
        task=task,
        task_id=task.task_id,
        benchmark=task.benchmark,
        eval_kind=task.eval_kind,
        benchmark_notes=benchmark_notes,
        task_prompt=task.prompt,
        starter_code=task.starter_code,
        required_dependencies=task.required_dependencies,
    ).with_inputs(
        "benchmark",
        "eval_kind",
        "benchmark_notes",
        "task_prompt",
        "starter_code",
        "required_dependencies",
    )


def build_benchmark_notes(task: BenchmarkTask) -> str:
    lines: list[str] = [
        "Use the tools to inspect the contract before finalizing code.",
        "Use the static analysis tools on your candidate solution before finishing.",
        "The final answer must be executable Python 3 code only, with no markdown fences.",
    ]
    if task.benchmark == "scicode":
        lines.extend(
            [
                "SciCode hidden tests call the required function directly.",
                "Preserve the exact function header and argument order from starter_code.",
                "Do not submit a stdin program for SciCode.",
                "Required scientific imports may matter.",
            ]
        )
    elif task.eval_kind == "stdin":
        lines.extend(
            [
                "This is a LiveCodeBench stdin task.",
                "Submit a complete program that reads stdin and prints exact output with no extra text.",
            ]
        )
    else:
        lines.extend(
            [
                "This is a LiveCodeBench functional task.",
                "Preserve the required interface from starter_code and avoid extra prints.",
            ]
        )
    return "\n".join(lines)


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


def build_tools(benchmark: str) -> list:
    tools = [
        benchmark_playbook,
        summarize_task_contract,
        solution_contract_json,
        extract_constraints_json,
        extract_examples_json,
        make_solution_scaffold,
        make_stdio_solution_scaffold,
        preview_solution_shape_json,
        syntax_check_json,
        repair_hints_json,
        strip_markdown_fences_tool,
    ]
    if benchmark == "scicode":
        tools.append(prepend_required_imports)
    return tools


def benchmark_playbook(
    benchmark: str = "",
    eval_kind: str = "",
    starter_code: str = "",
    required_dependencies: str = "",
) -> str:
    """Return benchmark-specific solving advice based on the official task contract."""
    lines = []
    if benchmark == "scicode":
        lines.append("SciCode subproblems are function-writing tasks with hidden direct function calls.")
        lines.append("Preserve the exact function header from starter_code.")
        lines.append("Scientific imports from required_dependencies may be needed in the final code.")
        lines.append("Avoid stdin handling and avoid extra print statements.")
    elif eval_kind == "stdin":
        lines.append("LiveCodeBench stdin tasks require a complete script that reads stdin and prints exact output.")
        lines.append("Avoid prompts, extra newlines, debug prints, and commentary.")
    else:
        lines.append("LiveCodeBench functional tasks import and call the required function directly.")
        lines.append("Preserve the starter-code interface exactly and avoid extra prints or examples.")
    if starter_code.strip():
        lines.append("Starter code is authoritative for the callable interface.")
    if required_dependencies.strip():
        lines.append("A dependency block is available and can be merged if imports are missing.")
    return "\n".join(lines)


def summarize_task_contract(
    benchmark: str = "",
    eval_kind: str = "",
    task_prompt: str = "",
    starter_code: str = "",
    required_dependencies: str = "",
) -> str:
    """Return a compact contract-focused digest of the task for the agent to reason over."""
    lines = [
        f"benchmark={benchmark}",
        f"eval_kind={eval_kind}",
    ]
    if benchmark == "scicode":
        sections = extract_scicode_sections(task_prompt)
        lines.append(f"sub_step={sections.get('sub_step', '')}")
        lines.append(f"function_header={extract_function_header(starter_code)}")
        if sections.get("required_dependencies"):
            lines.append("required_dependencies_present=yes")
    else:
        lines.append(f"function_header={extract_function_header(starter_code)}")
        lines.append(f"has_starter_code={'yes' if starter_code.strip() else 'no'}")
        lines.append(f"mode={'stdin' if eval_kind == 'stdin' else 'function'}")
    if required_dependencies.strip():
        lines.append("dependency_block_present=yes")
    return "\n".join(lines)


def solution_contract_json(benchmark: str = "", eval_kind: str = "", starter_code: str = "") -> str:
    """Return the expected I/O contract and required function names as JSON."""
    import re

    payload = {
        "benchmark": benchmark,
        "eval_kind": eval_kind,
        "mode": "stdin" if eval_kind == "stdin" else "function",
        "required_function_names": re.findall(r"def\s+([A-Za-z_]\w*)\s*\(", starter_code),
        "must_preserve_interface": bool(starter_code.strip()),
        "should_print_output": eval_kind == "stdin",
        "should_return_values": eval_kind != "stdin",
    }
    return json.dumps(payload, sort_keys=True)


def extract_constraints_json(task_prompt: str = "") -> str:
    """Extract explicit constraint lines and a coarse size hint from the prompt."""
    import re

    constraint_lines: list[str] = []
    for line in task_prompt.splitlines():
        compact = " ".join(line.split())
        if not compact:
            continue
        if re.search(r"[<>]=?|≤|≥", compact) or "sum of" in compact.lower():
            constraint_lines.append(compact)

    large_input_hint = "unknown"
    if re.search(r"10\^5|2 \* 10\^5|200000|100000", task_prompt):
        large_input_hint = "linear_or_nlogn_expected"
    elif re.search(r"10\^4|10000", task_prompt):
        large_input_hint = "moderate_input_size"
    elif re.search(r"10\^2|100", task_prompt):
        large_input_hint = "small_input_size"

    payload = {
        "constraint_lines": constraint_lines,
        "large_input_hint": large_input_hint,
        "num_constraint_lines": len(constraint_lines),
    }
    return json.dumps(payload, sort_keys=True)


def extract_examples_json(task_prompt: str = "") -> str:
    """Extract sample input/output blocks from competitive-programming style prompts."""
    import re

    normalized = task_prompt.replace("\r\n", "\n")
    lines = normalized.splitlines()

    examples: list[dict[str, str]] = []
    sample_indices = [index for index, line in enumerate(lines) if re.match(r"^\s*Sample Input\b", line)]
    for start in sample_indices:
        input_lines: list[str] = []
        output_lines: list[str] = []
        index = start + 1
        while index < len(lines) and not re.match(r"^\s*Sample Output\b", lines[index]):
            input_lines.append(lines[index])
            index += 1
        if index < len(lines) and re.match(r"^\s*Sample Output\b", lines[index]):
            index += 1
            while index < len(lines):
                current = lines[index]
                if re.match(r"^\s*Sample Input\b", current) or re.match(r"^\s*Note\b", current):
                    break
                output_lines.append(current)
                index += 1
        examples.append(
            {
                "sample_input": "\n".join(input_lines).strip(),
                "sample_output": "\n".join(output_lines).strip(),
            }
        )

    payload = {
        "num_examples": len(examples),
        "examples": examples[:3],
    }
    return json.dumps(payload, sort_keys=True)


def make_solution_scaffold(starter_code: str = "", required_dependencies: str = "") -> str:
    """Create a starter scaffold by combining unique import lines with the benchmark-provided header."""
    imports = unique_import_lines(required_dependencies)
    parts = []
    if imports:
        parts.append("\n".join(imports))
    if starter_code.strip():
        if parts:
            parts.append("")
        parts.append(starter_code.rstrip())
    return "\n".join(parts).strip()


def make_stdio_solution_scaffold(task_prompt: str = "") -> str:
    """Create a minimal stdin-oriented Python scaffold for LiveCodeBench IO tasks."""
    del task_prompt
    return (
        "import sys\n\n"
        "def solve() -> None:\n"
        "    data = sys.stdin.read().strip().split()\n"
        "    if not data:\n"
        "        return\n"
        "\n"
        "if __name__ == \"__main__\":\n"
        "    solve()\n"
    )


def preview_solution_shape_json(
    solution: str = "",
    benchmark: str = "",
    eval_kind: str = "",
    starter_code: str = "",
) -> str:
    """Return structural checks on the current candidate solution without running hidden tests."""
    import re

    cleaned = strip_code_fences(solution).strip()
    required_names = re.findall(r"def\s+([A-Za-z_]\w*)\s*\(", starter_code)
    defined_names = re.findall(r"def\s+([A-Za-z_]\w*)\s*\(", cleaned)
    payload = {
        "benchmark": benchmark,
        "eval_kind": eval_kind,
        "mode": "stdin" if eval_kind == "stdin" else "function",
        "required_function_names": required_names,
        "defined_function_names": defined_names,
        "missing_function_names": [name for name in required_names if name not in defined_names],
        "has_main_guard": "__name__ == '__main__'" in cleaned or '__name__ == "__main__"' in cleaned,
        "has_input_call": "input(" in cleaned or "sys.stdin" in cleaned,
        "has_print_call": "print(" in cleaned,
        "line_count": len(cleaned.splitlines()) if cleaned else 0,
    }
    return json.dumps(payload, sort_keys=True)


def syntax_check_json(solution: str = "") -> str:
    """Compile the candidate solution and return syntax diagnostics as JSON."""
    cleaned = strip_code_fences(solution).strip()
    payload: dict[str, Any] = {"ok": False, "error_type": None, "error_message": None, "line": None}
    try:
        compile(cleaned, "<candidate>", "exec")
    except SyntaxError as exc:
        payload["error_type"] = "SyntaxError"
        payload["error_message"] = exc.msg
        payload["line"] = exc.lineno
        return json.dumps(payload, sort_keys=True)
    except Exception as exc:
        payload["error_type"] = type(exc).__name__
        payload["error_message"] = str(exc)
        return json.dumps(payload, sort_keys=True)

    payload["ok"] = True
    return json.dumps(payload, sort_keys=True)


def repair_hints_json(
    solution: str = "",
    benchmark: str = "",
    eval_kind: str = "",
    starter_code: str = "",
    required_dependencies: str = "",
) -> str:
    """Return deterministic repair hints for common benchmark failures."""
    import re

    cleaned = strip_code_fences(solution).strip()
    issues: list[str] = []
    suggestions: list[str] = []

    try:
        compile(cleaned, "<candidate>", "exec")
    except SyntaxError as exc:
        issues.append(f"syntax_error_line_{exc.lineno}: {exc.msg}")
        suggestions.append("Fix syntax before finalizing the solution.")

    required_names = re.findall(r"def\s+([A-Za-z_]\w*)\s*\(", starter_code)
    defined_names = re.findall(r"def\s+([A-Za-z_]\w*)\s*\(", cleaned)
    missing = [name for name in required_names if name not in defined_names]
    if missing:
        issues.append(f"missing_required_functions: {missing}")
        suggestions.append("Preserve the exact required function names from starter_code.")

    if eval_kind == "stdin":
        if "input(" not in cleaned and "sys.stdin" not in cleaned:
            issues.append("stdin_task_without_input_handling")
            suggestions.append("Read from stdin in the submitted program.")
        if "print(" not in cleaned:
            issues.append("stdin_task_without_print")
            suggestions.append("Print the final answer for stdin tasks.")
        if "buffer.read().split()" in cleaned:
            issues.append("stdin_buffer_split_produces_bytes")
            suggestions.append("Prefer sys.stdin.read().split() or decode buffer input before comparing tokens to strings.")
    else:
        if "input(" in cleaned or "sys.stdin" in cleaned:
            issues.append("function_task_with_stdin_logic")
            suggestions.append("Function tasks should not read from stdin.")
        if "print(" in cleaned and benchmark == "scicode":
            issues.append("scicode_solution_has_prints")
            suggestions.append("SciCode functions should usually return values instead of printing debug output.")

    if benchmark == "scicode":
        imports = unique_import_lines(required_dependencies)
        if imports and not any(line in cleaned for line in imports):
            issues.append("required_dependency_block_not_present")
            suggestions.append("Consider prepending the required scientific import block.")

    payload = {"issues": issues, "suggestions": suggestions}
    return json.dumps(payload, sort_keys=True)


def strip_markdown_fences_tool(text: str = "") -> str:
    """Strip markdown code fences from a candidate solution string."""
    return strip_code_fences(text).strip()


def prepend_required_imports(solution: str = "", required_dependencies: str = "") -> str:
    """Prepend missing benchmark-provided import lines to the current candidate solution."""
    cleaned = strip_code_fences(solution).strip()
    imports = unique_import_lines(required_dependencies)
    missing = [line for line in imports if line not in cleaned]
    if not missing:
        return cleaned
    if not cleaned:
        return "\n".join(missing)
    return "\n".join(missing) + "\n\n" + cleaned


def unique_import_lines(required_dependencies: str) -> list[str]:
    imports: list[str] = []
    seen: set[str] = set()
    for raw_line in required_dependencies.splitlines():
        line = raw_line.strip()
        if not line or (not line.startswith("import ") and not line.startswith("from ")):
            continue
        if line in seen:
            continue
        seen.add(line)
        imports.append(line)
    return imports


def extract_scicode_sections(task_prompt: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    lines = [line.rstrip() for line in task_prompt.splitlines()]
    for index, line in enumerate(lines):
        if line.startswith("Sub-step "):
            sections["sub_step"] = line.replace("Sub-step ", "").strip(":")
        elif line == "Required dependencies:":
            collected = []
            cursor = index + 1
            while cursor < len(lines) and lines[cursor].strip():
                collected.append(lines[cursor])
                cursor += 1
            sections["required_dependencies"] = "\n".join(collected).strip()
    return sections


def extract_function_header(starter_code: str) -> str:
    import re

    text = starter_code.strip()
    if not text:
        return ""
    match = re.search(r"(def\s+[A-Za-z_]\w*\s*\(.*?\)\s*:)", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.splitlines()[0].strip()


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
    step_counts: list[int] = []
    tool_name_counts: dict[str, int] = {}
    solved_count = 0
    partial_count = 0
    task_artifacts_dir = run_dir / "task_artifacts"
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)

    started_at_utc = utc_timestamp()
    started_perf = time.perf_counter()

    for index, example in enumerate(examples, start=1):
        print(f"[task {index}/{len(examples)}] {example.task_id}: running ReAct", flush=True)
        prediction_started = time.perf_counter()
        prediction = program(
            benchmark=example.benchmark,
            eval_kind=example.eval_kind,
            benchmark_notes=example.benchmark_notes,
            task_prompt=example.task_prompt,
            starter_code=example.starter_code,
            required_dependencies=example.required_dependencies,
        )
        prediction_time_s = time.perf_counter() - prediction_started

        trajectory = getattr(prediction, "trajectory", {}) or {}
        step_tool_names = [str(value) for key, value in sorted(trajectory.items()) if key.startswith("tool_name_")]
        for tool_name in step_tool_names:
            tool_name_counts[tool_name] = tool_name_counts.get(tool_name, 0) + 1

        solution = str(getattr(prediction, "solution", "") or "")
        evaluation_started = time.perf_counter()
        result = evaluate_task(example.task, solution, timeout_s=timeout_s)
        evaluation_time_s = time.perf_counter() - evaluation_started
        total_time_s = prediction_time_s + evaluation_time_s
        score = result.score
        if score >= 1.0:
            solved_count += 1
        elif score > 0.0:
            partial_count += 1

        num_steps = len(step_tool_names)
        step_counts.append(num_steps)
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
            "num_agent_steps": num_steps,
            "tool_names": step_tool_names,
            "solution_preview": solution[:400],
        }
        rows.append(row)
        total_score += score

        save_json(
            task_artifacts_dir / f"{sanitize_slug(example.task_id)}.json",
            {
                "task_id": example.task_id,
                "benchmark": example.task.benchmark,
                "model": model,
                "benchmark_notes": example.benchmark_notes,
                "eval_result": result,
                "prediction_time_s": prediction_time_s,
                "evaluation_time_s": evaluation_time_s,
                "total_time_s": total_time_s,
                "num_agent_steps": num_steps,
                "tool_names": step_tool_names,
                "solution": solution,
                "trajectory": trajectory,
            },
        )
        print(
            f"[task {index}/{len(examples)}] {example.task_id}: "
            f"score={score:.3f} steps={num_steps}",
            flush=True,
        )

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
        "tool_name_counts": tool_name_counts,
        "time_stats_s": {
            "prediction": summarize_numeric_series(prediction_times),
            "evaluation": summarize_numeric_series(evaluation_times),
            "total": summarize_numeric_series(total_times),
        },
        "agent_step_stats": summarize_numeric_series(step_counts),
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
                "eval_kind": example.task.eval_kind,
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


def summarize_numeric_series(values: list[float | int]) -> dict[str, float]:
    if not values:
        return {"sum": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    numeric = [float(value) for value in values]
    return {
        "sum": sum(numeric),
        "mean": sum(numeric) / len(numeric),
        "median": statistics.median(numeric),
        "max": max(numeric),
    }


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def extract_solution_from_prediction(pred: dspy.Prediction) -> str:
    return strip_code_fences(str(getattr(pred, "solution", "") or "")).strip()


if __name__ == "__main__":
    main()
