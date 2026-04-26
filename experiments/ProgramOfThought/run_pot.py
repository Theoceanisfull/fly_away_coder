from __future__ import annotations

import argparse
import json
import random
import signal
import shutil
import subprocess
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import dspy
from dspy.primitives.python_interpreter import PythonInterpreter

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
        "max_iters": 4,
        "max_tokens": 3072,
        "train_groups": 24,
        "val_groups": 8,
        "eval_groups": 16,
    },
    "livecodebench": {
        "max_iters": 3,
        "max_tokens": 2048,
        "train_groups": 40,
        "val_groups": 10,
        "eval_groups": 40,
    },
}


class PredictionTimeoutError(TimeoutError):
    """Raised when a single ProgramOfThought prediction exceeds its wall-clock budget."""


@contextmanager
def prediction_timeout(seconds: float | None):
    if not seconds or seconds <= 0 or not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)

    def handle_timeout(signum, frame):  # type: ignore[unused-argument]
        raise PredictionTimeoutError(f"ProgramOfThought prediction timed out after {seconds:.1f}s.")

    signal.signal(signal.SIGALRM, handle_timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def reset_python_interpreter(interpreter: PythonInterpreter | None, *, force: bool) -> None:
    if interpreter is None:
        return

    process = getattr(interpreter, "deno_process", None)
    try:
        if process is None:
            return
        if process.poll() is None:
            if not force and getattr(process, "stdin", None):
                process.stdin.write(json.dumps({"shutdown": True}) + "\n")
                process.stdin.flush()
                process.stdin.close()
                process.wait(timeout=5)
            else:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
    except Exception:
        if process is not None and process.poll() is None:
            try:
                process.kill()
                process.wait(timeout=2)
            except Exception:
                pass
    finally:
        for stream_name in ("stdin", "stdout", "stderr"):
            stream = getattr(process, stream_name, None)
            if stream is None:
                continue
            try:
                stream.close()
            except Exception:
                pass
        interpreter.deno_process = None
        if hasattr(interpreter, "_mounted_files"):
            interpreter._mounted_files = False
        if hasattr(interpreter, "_tools_registered"):
            interpreter._tools_registered = False


def make_error_prediction(reason: str) -> dspy.Prediction:
    return dspy.Prediction(
        solution="",
        reasoning=reason,
        solution_source="error",
        final_generated_code="",
        code_output=None,
        num_hops=0,
        attempts=[],
    )


class SolveWithProgramOfThought(dspy.Signature):
    benchmark: str = dspy.InputField(desc="Benchmark name: scicode or livecodebench.")
    eval_kind: str = dspy.InputField(desc="Evaluation kind such as scicode_step, stdin, or functional.")
    benchmark_notes: str = dspy.InputField(desc="Benchmark-specific solution contract.")
    tool_guide: str = dspy.InputField(desc="Helper functions available inside the PoT execution sandbox.")
    task_prompt: str = dspy.InputField(desc="Full benchmark task statement.")
    starter_code: str = dspy.InputField(desc="Starter code or required function header to preserve when present.")
    required_dependencies: str = dspy.InputField(desc="Required dependency block when provided by the benchmark.")
    solution: str = dspy.OutputField(desc="Executable Python 3 code only with no markdown fences.")


class BenchmarkProgramOfThought(dspy.ProgramOfThought):
    def forward(self, **kwargs):
        input_kwargs = {field_name: kwargs[field_name] for field_name in self.input_fields}
        attempts: list[dict[str, Any]] = []

        code_data = self.code_generate(**input_kwargs)
        parsed_code, parse_error = self._parse_code(code_data)
        execution_output = None
        execution_error = None
        attempts.append(
            {
                "hop": 1,
                "mode": "generate",
                "raw_generated_code": getattr(code_data, "generated_code", ""),
                "parsed_code": parsed_code,
                "parse_error": parse_error,
                "execution_output": None,
                "execution_error": None,
            }
        )

        if not parse_error:
            execution_output, execution_error = self._execute_code(parsed_code)
            attempts[-1]["execution_output"] = execution_output
            attempts[-1]["execution_error"] = execution_error
        else:
            execution_error = parse_error

        hop = 1
        while execution_error is not None:
            if hop == self.max_iters:
                self.interpreter.shutdown()
                raise RuntimeError(f"Max hops reached. Failed to run ProgramOfThought: {execution_error}")

            input_kwargs.update({"previous_code": parsed_code, "error": execution_error})
            code_data = self.code_regenerate(**input_kwargs)
            hop += 1
            parsed_code, parse_error = self._parse_code(code_data)
            execution_output = None
            execution_error = None

            attempts.append(
                {
                    "hop": hop,
                    "mode": "regenerate",
                    "raw_generated_code": getattr(code_data, "generated_code", ""),
                    "parsed_code": parsed_code,
                    "parse_error": parse_error,
                    "execution_output": None,
                    "execution_error": None,
                }
            )

            if not parse_error:
                execution_output, execution_error = self._execute_code(parsed_code)
                attempts[-1]["execution_output"] = execution_output
                attempts[-1]["execution_error"] = execution_error
            else:
                execution_error = parse_error

        input_kwargs.update({"final_generated_code": parsed_code, "code_output": execution_output})
        final_prediction = self.generate_output(**input_kwargs)

        submitted_solution = extract_solution_from_code_output(execution_output)
        lm_solution = strip_code_fences(str(getattr(final_prediction, "solution", "") or "")).strip()
        if looks_like_code(submitted_solution):
            solution = submitted_solution
            solution_source = "submitted_json"
        elif looks_like_code(lm_solution):
            solution = lm_solution
            solution_source = "lm_answer"
        elif submitted_solution:
            solution = submitted_solution
            solution_source = "submitted_raw"
        else:
            solution = lm_solution
            solution_source = "lm_answer"

        reasoning = getattr(final_prediction, "reasoning", None)
        self.interpreter.shutdown()
        return dspy.Prediction(
            solution=solution,
            reasoning=reasoning,
            solution_source=solution_source,
            final_generated_code=parsed_code,
            code_output=execution_output,
            num_hops=hop,
            attempts=attempts,
        )


class PoTSolver(dspy.Module):
    def __init__(self, *, benchmark: str, max_iters: int, prediction_timeout_s: float | None):
        super().__init__()
        self.prediction_timeout_s = prediction_timeout_s
        interpreter = PythonInterpreter(
            tools=build_tools(benchmark),
            output_fields=[{"name": "solution", "type": "str"}],
        )
        self.solve = BenchmarkProgramOfThought(
            SolveWithProgramOfThought,
            max_iters=max_iters,
            interpreter=interpreter,
        )

    def forward(
        self,
        *,
        benchmark: str,
        eval_kind: str,
        benchmark_notes: str,
        tool_guide: str,
        task_prompt: str,
        starter_code: str,
        required_dependencies: str,
    ):
        try:
            with prediction_timeout(self.prediction_timeout_s):
                return self.solve(
                    benchmark=benchmark,
                    eval_kind=eval_kind,
                    benchmark_notes=benchmark_notes,
                    tool_guide=tool_guide,
                    task_prompt=task_prompt,
                    starter_code=starter_code,
                    required_dependencies=required_dependencies,
                )
        except PredictionTimeoutError as exc:
            reset_python_interpreter(self.solve.interpreter, force=True)
            return make_error_prediction(f"solver_timeout: {exc}")
        except Exception as exc:
            reset_python_interpreter(self.solve.interpreter, force=True)
            return make_error_prediction(f"solver_error: {type(exc).__name__}: {exc}")


def main() -> None:
    args = parse_args()
    apply_benchmark_runtime_defaults(args)

    if args.command == "preview":
        tasks = load_benchmark_tasks(args)
        print_preview(tasks)
        return

    ensure_deno_available()
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

    program = PoTSolver(
        benchmark=args.benchmark,
        max_iters=args.max_iters,
        prediction_timeout_s=args.prediction_timeout_s,
    )
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
    parser = argparse.ArgumentParser(description="Run DSPy ProgramOfThought on SciCode and LiveCodeBench.")
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
    evaluate.add_argument(
        "--prediction-timeout-s",
        type=float,
        help="Wall-clock cap for a single ProgramOfThought prediction, including scratch-code execution.",
    )
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
        if args.prediction_timeout_s is None:
            args.prediction_timeout_s = max(args.request_timeout_s * 3.0, 360.0)

    if args.train_groups is None and args.val_groups is None and args.eval_groups is None and not args.allow_shared_group_splits:
        args.train_groups = defaults["train_groups"]
        args.val_groups = defaults["val_groups"]
        args.eval_groups = defaults["eval_groups"]


def ensure_deno_available() -> None:
    if shutil.which("deno"):
        return
    raise RuntimeError(
        "DSPy ProgramOfThought requires Deno because it uses DSPy's PythonInterpreter sandbox. "
        "No `deno` executable was found on PATH. Install it first: "
        "https://docs.deno.com/runtime/getting_started/installation/"
    )


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
    tool_guide = build_tool_guide(task.benchmark)
    return dspy.Example(
        task=task,
        task_id=task.task_id,
        benchmark=task.benchmark,
        eval_kind=task.eval_kind,
        benchmark_notes=benchmark_notes,
        tool_guide=tool_guide,
        task_prompt=task.prompt,
        starter_code=task.starter_code,
        required_dependencies=task.required_dependencies,
    ).with_inputs(
        "benchmark",
        "eval_kind",
        "benchmark_notes",
        "tool_guide",
        "task_prompt",
        "starter_code",
        "required_dependencies",
    )


def build_benchmark_notes(task: BenchmarkTask) -> str:
    lines: list[str] = [
        "Your internal scratch program should build a final candidate solution string.",
        "When the candidate is ready, call SUBMIT({'solution': candidate_solution_string}).",
        "The submitted solution itself must be executable Python 3 code with no markdown fences.",
    ]
    if task.benchmark == "scicode":
        lines.extend(
            [
                "SciCode task: hidden tests call the requested function directly.",
                "Preserve the provided function header and required argument order exactly.",
                "Do not read from stdin or print debugging output in the submitted solution.",
                "Use the required dependency block when those imports are relevant.",
            ]
        )
    elif task.eval_kind == "stdin":
        lines.extend(
            [
                "LiveCodeBench stdin task: the submitted solution should be a full program.",
                "The submitted solution must read stdin and print exact output with no extras.",
            ]
        )
    else:
        lines.extend(
            [
                "LiveCodeBench functional task: hidden tests import and call the required function directly.",
                "The submitted solution must preserve the starter-code interface exactly.",
                "Avoid extra prints or example usage code in the submitted solution.",
            ]
        )
    if task.starter_code.strip():
        lines.append("Starter code is authoritative for interface and naming.")
    if task.required_dependencies.strip():
        lines.append("A required dependency block is available and may need to be included in the submitted solution.")
    return "\n".join(lines)


def build_tool_guide(benchmark: str) -> str:
    lines = [
        "Helper functions available inside the PoT Python sandbox:",
        "- benchmark_playbook(benchmark, eval_kind, starter_code, required_dependencies) -> str",
        "- solution_contract_json(benchmark, eval_kind, starter_code) -> str",
        "- extract_function_header(starter_code) -> str",
        "- extract_function_names_json(starter_code) -> str",
        "- preview_solution_shape_json(solution, benchmark, eval_kind, starter_code) -> str",
        "- strip_markdown_fences_tool(text) -> str",
    ]
    if benchmark == "scicode":
        lines.extend(
            [
                "- extract_required_imports_block(required_dependencies) -> str",
                "- make_solution_scaffold(starter_code, required_dependencies) -> str",
            ]
        )
    lines.extend(
        [
            "",
            "Use these helpers if useful while constructing the final submitted solution string.",
            "The final scratch program should call SUBMIT({'solution': candidate_solution_string}).",
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


def build_tools(benchmark: str) -> dict[str, Any]:
    tools: dict[str, Any] = {
        "benchmark_playbook": benchmark_playbook,
        "solution_contract_json": solution_contract_json,
        "extract_function_header": extract_function_header,
        "extract_function_names_json": extract_function_names_json,
        "preview_solution_shape_json": preview_solution_shape_json,
        "strip_markdown_fences_tool": strip_markdown_fences_tool,
    }
    if benchmark == "scicode":
        tools["extract_required_imports_block"] = extract_required_imports_block
        tools["make_solution_scaffold"] = make_solution_scaffold
    return tools


def benchmark_playbook(benchmark: str, eval_kind: str, starter_code: str, required_dependencies: str) -> str:
    lines = []
    if benchmark == "scicode":
        lines.append("SciCode: implement the requested function directly and preserve the function header.")
        lines.append("SciCode hidden tests call the function, so the submitted solution should not be a stdin program.")
        lines.append("Use required_dependencies when imports are needed.")
    elif eval_kind == "stdin":
        lines.append("LiveCodeBench stdin: submit a complete script that reads stdin and prints exact output.")
        lines.append("Avoid prompts, debug text, or extra formatting.")
    else:
        lines.append("LiveCodeBench functional: preserve the starter-code function interface exactly.")
        lines.append("Return values from the required function and avoid extra prints.")
    if starter_code.strip():
        lines.append("Starter code is the source of truth for interface and naming.")
    if required_dependencies.strip():
        lines.append("A dependency block is available if imports are needed.")
    return "\n".join(lines)


def solution_contract_json(benchmark: str, eval_kind: str, starter_code: str) -> str:
    import re

    contract = {
        "benchmark": benchmark,
        "eval_kind": eval_kind,
        "mode": "stdin" if eval_kind == "stdin" else "function",
        "must_preserve_interface": bool(starter_code.strip()),
        "should_print_output": eval_kind == "stdin",
        "should_return_values": eval_kind != "stdin",
        "required_function_names": re.findall(r"def\s+([A-Za-z_]\w*)\s*\(", starter_code),
    }
    return json.dumps(contract, sort_keys=True)


def extract_function_header(starter_code: str) -> str:
    import re

    text = starter_code.strip()
    if not text:
        return ""
    match = re.search(r"(def\s+[A-Za-z_]\w*\s*\(.*?\)\s*:)", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.splitlines()[0].strip()


def extract_function_names_json(starter_code: str) -> str:
    import re

    names = re.findall(r"def\s+([A-Za-z_]\w*)\s*\(", starter_code)
    return json.dumps(names)


def preview_solution_shape_json(solution: str, benchmark: str, eval_kind: str, starter_code: str) -> str:
    import re

    cleaned = strip_code_fences(solution).strip()
    required_names = re.findall(r"def\s+([A-Za-z_]\w*)\s*\(", starter_code)
    defined_names = re.findall(r"def\s+([A-Za-z_]\w*)\s*\(", cleaned)
    missing_names = [name for name in required_names if name not in defined_names]
    payload = {
        "benchmark": benchmark,
        "eval_kind": eval_kind,
        "mode": "stdin" if eval_kind == "stdin" else "function",
        "required_function_names": required_names,
        "defined_function_names": defined_names,
        "missing_function_names": missing_names,
        "has_main_guard": "__name__ == '__main__'" in cleaned or '__name__ == "__main__"' in cleaned,
        "has_input_call": "input(" in cleaned or "sys.stdin" in cleaned,
        "has_print_call": "print(" in cleaned,
        "line_count": len(cleaned.splitlines()) if cleaned else 0,
    }
    return json.dumps(payload, sort_keys=True)


def extract_required_imports_block(required_dependencies: str) -> str:
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
    return "\n".join(imports)


def make_solution_scaffold(starter_code: str, required_dependencies: str) -> str:
    imports_block = extract_required_imports_block(required_dependencies)
    parts = []
    if imports_block:
        parts.append(imports_block)
    if starter_code.strip():
        if parts:
            parts.append("")
        parts.append(starter_code.rstrip())
    return "\n".join(parts).strip()


def strip_markdown_fences_tool(text: str) -> str:
    return strip_code_fences(text)


def extract_solution_from_code_output(code_output: str | None) -> str:
    if not code_output:
        return ""
    try:
        parsed = json.loads(code_output)
    except json.JSONDecodeError:
        return strip_code_fences(code_output).strip()

    candidate = ""
    if isinstance(parsed, dict):
        if "solution" in parsed:
            candidate = str(parsed["solution"] or "")
        elif len(parsed) == 1:
            candidate = str(next(iter(parsed.values())) or "")
    elif isinstance(parsed, str):
        candidate = parsed
    else:
        candidate = json.dumps(parsed)
    return strip_code_fences(candidate).strip()


def looks_like_code(text: str) -> bool:
    if not text:
        return False
    code_markers = ("def ", "class ", "import ", "from ", "if __name__", "sys.stdin", "print(", "return ")
    return any(marker in text for marker in code_markers)


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
    hop_counts: list[int] = []
    solved_count = 0
    partial_count = 0
    task_artifacts_dir = run_dir / "task_artifacts"
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)

    started_at_utc = utc_timestamp()
    started_perf = time.perf_counter()

    for index, example in enumerate(examples, start=1):
        print(f"[task {index}/{len(examples)}] {example.task_id}: running ProgramOfThought", flush=True)
        prediction_started = time.perf_counter()
        prediction = program(
            benchmark=example.benchmark,
            eval_kind=example.eval_kind,
            benchmark_notes=example.benchmark_notes,
            tool_guide=example.tool_guide,
            task_prompt=example.task_prompt,
            starter_code=example.starter_code,
            required_dependencies=example.required_dependencies,
        )
        prediction_time_s = time.perf_counter() - prediction_started

        solution = str(getattr(prediction, "solution", "") or "")
        solution_source = str(getattr(prediction, "solution_source", "unknown") or "unknown")
        reasoning = getattr(prediction, "reasoning", None)
        num_hops = int(getattr(prediction, "num_hops", 0) or 0)
        final_generated_code = str(getattr(prediction, "final_generated_code", "") or "")
        code_output = getattr(prediction, "code_output", None)
        attempts = list(getattr(prediction, "attempts", []) or [])

        evaluation_started = time.perf_counter()
        result = evaluate_task(example.task, solution, timeout_s=timeout_s)
        evaluation_time_s = time.perf_counter() - evaluation_started
        total_time_s = prediction_time_s + evaluation_time_s
        score = result.score
        if score >= 1.0:
            solved_count += 1
        elif score > 0.0:
            partial_count += 1

        hop_counts.append(num_hops)
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
            "num_hops": num_hops,
            "solution_source": solution_source,
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
                "tool_guide": example.tool_guide,
                "eval_result": result,
                "prediction_time_s": prediction_time_s,
                "evaluation_time_s": evaluation_time_s,
                "total_time_s": total_time_s,
                "num_hops": num_hops,
                "solution_source": solution_source,
                "reasoning": reasoning,
                "solution": solution,
                "final_generated_code": final_generated_code,
                "code_output": code_output,
                "attempts": attempts,
            },
        )
        print(
            f"[task {index}/{len(examples)}] {example.task_id}: "
            f"score={score:.3f} hops={num_hops} solution_source={solution_source}",
            flush=True,
        )

    mean_score = total_score / len(examples) if examples else 0.0
    finished_at_utc = utc_timestamp()
    return {
        "benchmark": benchmark,
        "model": model,
        "run_started_at_utc": started_at_utc,
        "run_finished_at_utc": finished_at_utc,
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
        "hop_stats": summarize_numeric_series(hop_counts),
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


if __name__ == "__main__":
    main()
