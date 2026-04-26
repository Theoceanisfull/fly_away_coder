from __future__ import annotations

import ast
import base64
import json
import pickle
import re
import subprocess
import sys
import tempfile
import textwrap
import traceback
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_ROOT = REPO_ROOT / "benchmarks"
LIVECODEBENCH_ROOT = BENCHMARKS_ROOT / "livecodebench" / "code_generation_lite"
SCICODE_SAMPLE_PATH = BENCHMARKS_ROOT / "scicode" / "test_data" / "first_problem.jsonl"
SCICODE_H5_PATH = BENCHMARKS_ROOT / "scicode" / "data" / "test_data.h5"

LIVECODEBENCH_RELEASE_FILES = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
    "release_latest": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
}

for idx in range(1, 7):
    key = f"v{idx}"
    LIVECODEBENCH_RELEASE_FILES[key] = [f"test{idx}.jsonl" if idx > 1 else "test.jsonl"]


@dataclass(frozen=True)
class BenchmarkTask:
    benchmark: str
    task_id: str
    prompt: str
    starter_code: str
    eval_kind: str
    tests: list[dict[str, Any]]
    required_dependencies: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalResult:
    score: float
    passed: int
    total: int
    feedback: str
    first_failure: str | None = None


def strip_code_fences(text: str) -> str:
    text = text.strip()
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[0].strip()
    return text


def load_livecodebench_tasks(
    *,
    release_tag: str = "release_latest",
    offset: int = 0,
    limit: int | None = None,
) -> list[BenchmarkTask]:
    file_names = LIVECODEBENCH_RELEASE_FILES.get(release_tag)
    if file_names is None:
        raise ValueError(f"Unsupported LiveCodeBench release tag: {release_tag}")

    tasks: list[BenchmarkTask] = []
    global_index = 0

    for file_name in file_names:
        path = LIVECODEBENCH_ROOT / file_name
        with path.open() as handle:
            for line in handle:
                if global_index < offset:
                    global_index += 1
                    continue

                row = json.loads(line)
                tests = json.loads(_decode_lcb_private_test_blob(row["private_test_cases"]))
                prompt = _build_livecodebench_prompt(row)
                eval_kind = tests[0].get("testtype", "stdin")

                tasks.append(
                    BenchmarkTask(
                        benchmark="livecodebench",
                        task_id=row["question_id"],
                        prompt=prompt,
                        starter_code=row.get("starter_code", ""),
                        eval_kind=eval_kind,
                        tests=tests,
                        metadata={
                            "question_title": row["question_title"],
                            "platform": row["platform"],
                            "difficulty": row["difficulty"],
                            "contest_id": row["contest_id"],
                            "contest_date": row["contest_date"],
                            "release_tag": release_tag,
                        },
                    )
                )
                global_index += 1

                if limit is not None and len(tasks) >= limit:
                    return tasks

    return tasks


def load_scicode_tasks(
    *,
    source: str = "huggingface",
    split: str = "test",
    offset: int = 0,
    limit: int | None = None,
) -> list[BenchmarkTask]:
    if source == "local_sample":
        rows = [json.loads(SCICODE_SAMPLE_PATH.read_text().splitlines()[0])]
    elif source == "huggingface":
        rows = list(load_dataset("SciCode1/SciCode", split=split))
    else:
        raise ValueError(f"Unsupported SciCode source: {source}")

    h5_targets = _load_scicode_h5_targets()
    tasks: list[BenchmarkTask] = []
    global_index = 0

    for row in rows:
        for step in row["sub_steps"]:
            if global_index < offset:
                global_index += 1
                continue

            tests: list[dict[str, Any]] = []
            step_targets = h5_targets.get(step["step_number"], {})
            for test_index, case_code in enumerate(step["test_cases"], start=1):
                tests.append(
                    {
                        "name": f"test{test_index}",
                        "case_code": case_code,
                        "targets": step_targets.get(f"test{test_index}", []),
                    }
                )

            prompt = _build_scicode_prompt(row, step)
            tasks.append(
                BenchmarkTask(
                    benchmark="scicode",
                    task_id=step["step_number"],
                    prompt=prompt,
                    starter_code=step["function_header"],
                    eval_kind="scicode_step",
                    tests=tests,
                    required_dependencies=row.get("required_dependencies", ""),
                    metadata={
                        "problem_id": row["problem_id"],
                        "problem_name": row["problem_name"],
                        "source": source,
                        "split": split,
                    },
                )
            )
            global_index += 1

            if limit is not None and len(tasks) >= limit:
                return tasks

    return tasks


def evaluate_task(task: BenchmarkTask, raw_solution: str, *, timeout_s: float) -> EvalResult:
    solution = strip_code_fences(raw_solution)
    if not solution:
        return EvalResult(
            score=0.0,
            passed=0,
            total=max(len(task.tests), 1),
            feedback="The model did not return any executable Python code.",
            first_failure="empty solution",
        )

    if task.benchmark == "livecodebench":
        return _evaluate_livecodebench(task, solution=solution, timeout_s=timeout_s)
    if task.benchmark == "scicode":
        return _evaluate_scicode(task, solution=solution, timeout_s=timeout_s)

    raise ValueError(f"Unsupported benchmark: {task.benchmark}")


def _build_livecodebench_prompt(row: dict[str, Any]) -> str:
    parts = [row["question_content"].strip()]
    if row.get("starter_code"):
        parts.extend(
            [
                "",
                "Starter code. Preserve the required interface:",
                row["starter_code"].rstrip(),
            ]
        )
    return "\n".join(parts).strip()


def _build_scicode_prompt(problem: dict[str, Any], step: dict[str, Any]) -> str:
    parts = [
        "Scientific coding task.",
        "",
        "Main problem description:",
        problem["problem_description_main"].strip(),
    ]
    background = (problem.get("problem_background_main") or "").strip()
    if background:
        parts.extend(["", "Background:", background])

    parts.extend(
        [
            "",
            f"Sub-step {step['step_number']}:",
            step["step_description_prompt"].strip(),
            "",
            "Required dependencies:",
            problem.get("required_dependencies", "").strip(),
            "",
            "Function header to implement:",
            step["function_header"].rstrip(),
        ]
    )
    return "\n".join(parts).strip()


def _decode_lcb_private_test_blob(blob: str) -> str:
    return pickle.loads(zlib.decompress(base64.b64decode(blob)))


def _evaluate_livecodebench(task: BenchmarkTask, *, solution: str, timeout_s: float) -> EvalResult:
    if task.eval_kind == "stdin":
        return _evaluate_livecodebench_stdin(task, solution=solution, timeout_s=timeout_s)
    if task.eval_kind == "functional":
        return _evaluate_livecodebench_functional(task, solution=solution, timeout_s=timeout_s)

    raise ValueError(f"Unsupported LiveCodeBench test type: {task.eval_kind}")


def _evaluate_livecodebench_stdin(task: BenchmarkTask, *, solution: str, timeout_s: float) -> EvalResult:
    passed = 0
    first_failure: str | None = None

    with tempfile.TemporaryDirectory(prefix="lcb-stdin-") as tmp_dir:
        candidate_path = Path(tmp_dir) / "candidate.py"
        candidate_path.write_text(solution)

        for index, test in enumerate(task.tests, start=1):
            try:
                completed = subprocess.run(
                    [sys.executable, str(candidate_path)],
                    input=test["input"],
                    text=True,
                    capture_output=True,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                first_failure = _hidden_test_failure("timeout", index)
                break

            if completed.returncode != 0:
                runtime_label = _extract_runtime_label(completed.stderr) or "runtime_error"
                first_failure = _hidden_test_failure(runtime_label, index)
                break

            expected = _normalize_text_output(test["output"])
            actual = _normalize_text_output(completed.stdout)
            if actual != expected:
                first_failure = _hidden_test_failure("wrong_answer", index)
                break

            passed += 1

    return _finalize_eval_result(passed=passed, total=len(task.tests), first_failure=first_failure)


def _evaluate_livecodebench_functional(task: BenchmarkTask, *, solution: str, timeout_s: float) -> EvalResult:
    try:
        entry_kind, entry_name = _extract_functional_entrypoint(task.starter_code or solution)
    except Exception as exc:
        return EvalResult(
            score=0.0,
            passed=0,
            total=len(task.tests),
            feedback=f"Unable to determine the callable entrypoint for the solution: {exc}",
            first_failure=str(exc),
        )

    passed = 0
    first_failure: str | None = None

    for index, test in enumerate(task.tests, start=1):
        script = _render_livecodebench_functional_script(
            solution=solution,
            entry_kind=entry_kind,
            entry_name=entry_name,
            raw_input=test["input"],
            raw_expected=test["output"],
        )
        outcome = _run_wrapper_script(script=script, timeout_s=timeout_s)
        if not outcome["passed"]:
            first_failure = _hidden_test_failure(outcome["message"], index)
            break
        passed += 1

    return _finalize_eval_result(passed=passed, total=len(task.tests), first_failure=first_failure)


def _extract_functional_entrypoint(source: str) -> tuple[str, str]:
    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name != "__init__":
                    return ("method", item.name)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return ("function", node.name)

    raise ValueError("no top-level function or Solution method was found")


def _render_livecodebench_functional_script(
    *,
    solution: str,
    entry_kind: str,
    entry_name: str,
    raw_input: str,
    raw_expected: str,
) -> str:
    return textwrap.dedent(
        f"""
        import json
        import math
        import sys
        from typing import *

        def _normalize(value):
            if hasattr(value, "tolist"):
                return _normalize(value.tolist())
            if isinstance(value, tuple):
                return [_normalize(item) for item in value]
            if isinstance(value, list):
                return [_normalize(item) for item in value]
            if isinstance(value, dict):
                return {{str(key): _normalize(val) for key, val in value.items()}}
            if isinstance(value, (set, frozenset)):
                return sorted(_normalize(item) for item in value)
            if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
                try:
                    return value.item()
                except Exception:
                    pass
            return value

        def _close(a, b, atol=1e-6, rtol=1e-5):
            if isinstance(a, dict) and isinstance(b, dict):
                if set(a.keys()) != set(b.keys()):
                    return False
                return all(_close(a[key], b[key], atol=atol, rtol=rtol) for key in a)
            if isinstance(a, list) and isinstance(b, list):
                if len(a) != len(b):
                    return False
                return all(_close(x, y, atol=atol, rtol=rtol) for x, y in zip(a, b))
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)
            return a == b

        namespace = {{}}
        candidate_code = {solution!r}
        try:
            exec("from typing import *\\n" + candidate_code, namespace)
            args = json.loads({raw_input!r})
            if not isinstance(args, list):
                args = [args]

            if {entry_kind!r} == "method":
                result = namespace["Solution"]().{entry_name}(*args)
            else:
                result = namespace[{entry_name!r}](*args)

            expected = json.loads({raw_expected!r})
            actual = _normalize(result)
            ok = _close(actual, expected)
            if ok:
                print(json.dumps({{"passed": True, "message": "ok"}}))
            else:
                print(
                    json.dumps(
                        {{
                            "passed": False,
                            "message": "wrong_answer",
                        }}
                    )
                )
        except Exception:
            print(
                json.dumps(
                    {{
                        "passed": False,
                        "message": f"runtime_error: {{type(sys.exc_info()[1]).__name__}}",
                    }}
                )
            )
        """
    )


def _evaluate_scicode(task: BenchmarkTask, *, solution: str, timeout_s: float) -> EvalResult:
    passed = 0
    first_failure: str | None = None

    for index, test in enumerate(task.tests, start=1):
        script = _render_scicode_test_script(
            solution=solution,
            required_dependencies=task.required_dependencies,
            test_case=test["case_code"],
            raw_targets=test["targets"],
        )
        outcome = _run_wrapper_script(script=script, timeout_s=timeout_s)
        if not outcome["passed"]:
            first_failure = _hidden_test_failure(outcome["message"], index)
            break
        passed += 1

    return _finalize_eval_result(passed=passed, total=len(task.tests), first_failure=first_failure)


def _render_scicode_test_script(
    *,
    solution: str,
    required_dependencies: str,
    test_case: str,
    raw_targets: list[Any],
) -> str:
    target_payload = raw_targets[0] if len(raw_targets) == 1 else tuple(raw_targets)
    encoded_target = base64.b64encode(pickle.dumps(target_payload)).decode("ascii")
    sanitized_case = re.sub(r"from\\s+scicode\\.compare\\.cmp\\s+import\\s+[^\\n]+\\n", "", test_case)

    return textwrap.dedent(
        f"""
        import base64
        import json
        import math
        import pickle
        from typing import *

        import numpy as np
        import pandas as pd
        import scipy as sp

        def _normalize(value):
            if hasattr(value, "tolist"):
                return _normalize(value.tolist())
            if isinstance(value, tuple):
                return [_normalize(item) for item in value]
            if isinstance(value, list):
                return [_normalize(item) for item in value]
            if isinstance(value, dict):
                return {{key: _normalize(val) for key, val in value.items()}}
            if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
                try:
                    return value.item()
                except Exception:
                    pass
            return value

        def _values_close(a, b, atol=1e-6, rtol=1e-5):
            if isinstance(a, dict) and isinstance(b, dict):
                return are_dicts_close(a, b, atol=atol, rtol=rtol)
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                return cmp_tuple_or_list(a, b, atol=atol, rtol=rtol)
            try:
                return bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True))
            except Exception:
                return _normalize(a) == _normalize(b)

        def cmp_tuple_or_list(actual, expected, atol=1e-6, rtol=1e-5):
            actual = _normalize(actual)
            expected = _normalize(expected)
            if not isinstance(actual, (list, tuple)) or not isinstance(expected, (list, tuple)):
                return _values_close(actual, expected, atol=atol, rtol=rtol)
            if len(actual) != len(expected):
                return False
            return all(_values_close(a, b, atol=atol, rtol=rtol) for a, b in zip(actual, expected))

        def are_dicts_close(actual, expected, atol=1e-6, rtol=1e-5):
            actual = _normalize(actual)
            expected = _normalize(expected)
            if set(actual.keys()) != set(expected.keys()):
                return False
            return all(_values_close(actual[key], expected[key], atol=atol, rtol=rtol) for key in actual)

        namespace = {{}}
        target = pickle.loads(base64.b64decode({encoded_target!r}))
        namespace.update(
            {{
                "np": np,
                "pd": pd,
                "sp": sp,
                "cmp_tuple_or_list": cmp_tuple_or_list,
                "are_dicts_close": are_dicts_close,
                "target": target,
            }}
        )

        try:
            prelude = "from typing import *\\n" + {required_dependencies!r} + "\\n"
            exec(prelude + {solution!r}, namespace)
            exec({sanitized_case!r}, namespace)
            print(json.dumps({{"passed": True, "message": "ok"}}))
        except AssertionError:
            print(json.dumps({{"passed": False, "message": "wrong_answer"}}))
        except Exception:
            print(
                json.dumps(
                    {{
                        "passed": False,
                        "message": f"runtime_error: {{type(sys.exc_info()[1]).__name__}}",
                    }}
                )
            )
        """
    )


def _run_wrapper_script(*, script: str, timeout_s: float) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="prompt-opt-") as tmp_dir:
        runner_path = Path(tmp_dir) / "runner.py"
        runner_path.write_text(script)
        try:
            completed = subprocess.run(
                [sys.executable, str(runner_path)],
                text=True,
                capture_output=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "message": "timeout"}

        if completed.returncode != 0:
            return {
                "passed": False,
                "message": _extract_runtime_label(completed.stderr) or "runtime_error",
            }

        stdout = completed.stdout.strip()
        if not stdout:
            return {"passed": False, "message": "runtime_error"}

        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return {"passed": False, "message": "runtime_error"}


def _finalize_eval_result(*, passed: int, total: int, first_failure: str | None) -> EvalResult:
    score = passed / total if total else 0.0
    if first_failure is None:
        feedback = f"Passed all {passed}/{total} hidden tests."
    else:
        feedback = f"Passed {passed}/{total} hidden tests. First failure: {first_failure}"
    return EvalResult(
        score=score,
        passed=passed,
        total=total,
        feedback=feedback,
        first_failure=first_failure,
    )


def _normalize_text_output(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def _hidden_test_failure(message: str, index: int) -> str:
    if message == "wrong_answer":
        return f"wrong_answer on hidden test {index}"
    if message == "timeout":
        return f"timeout on hidden test {index}"
    if message.startswith("runtime_error:"):
        return f"{message} on hidden test {index}"
    if message == "runtime_error":
        return f"runtime_error on hidden test {index}"
    return f"evaluation_failure on hidden test {index}"


def _extract_runtime_label(text: str) -> str | None:
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))", text or "")
    if match:
        return f"runtime_error: {match.group(1)}"
    return None


def _load_scicode_h5_targets() -> dict[str, dict[str, list[Any]]]:
    targets: dict[str, dict[str, list[Any]]] = {}
    with h5py.File(SCICODE_H5_PATH, "r") as handle:
        for step_name in handle.keys():
            step_targets: dict[str, list[Any]] = {}
            step_group = handle[step_name]
            for test_name in step_group.keys():
                test_group = step_group[test_name]
                values = [_h5_to_python(test_group[var_name]) for var_name in sorted(test_group.keys())]
                step_targets[test_name] = values
            targets[step_name] = step_targets
    return targets


def _h5_to_python(node: h5py.Group | h5py.Dataset) -> Any:
    if isinstance(node, h5py.Dataset):
        value = node[()]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if getattr(value, "shape", ()) == ():
            try:
                return value.item()
            except Exception:
                return value
        return value

    return {name: _h5_to_python(node[name]) for name in sorted(node.keys())}
