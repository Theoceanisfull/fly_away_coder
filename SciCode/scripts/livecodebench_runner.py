#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
DEFAULT_MODELS = [
    "ollama/rnj-1:latest",
    "ollama/devstral-small-2:latest",
]
DEFAULT_LCB_BASE = PROJECT_ROOT.parent / "benchmarks" / "livecodebench"
DEFAULT_LCB_REPO_DIR = DEFAULT_LCB_BASE / "LiveCodeBench"
DEFAULT_LCB_DATASET_DIR = DEFAULT_LCB_BASE / "code_generation_lite"
DEFAULT_RELEASE_VERSION = "release_latest"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_DATASET_REPO_ID = "livecodebench/code_generation_lite"
DEFAULT_REPO_URL = "https://github.com/LiveCodeBench/LiveCodeBench"

COLOR_RESET = "\033[0m"
COLOR_INFO = "\033[36m"
COLOR_WARN = "\033[33m"

SYSTEM_MESSAGE_GENERIC = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program "
    "that matches the specification and passes all tests."
)
FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution "
    "to the problem and enclose your code within delimiters."
)
FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within "
    "delimiters as follows. Ensure that when the python program runs, it reads "
    "the inputs, runs the algorithm and writes output to STDOUT."
)

ALLOWED_FILES: dict[str, list[str]] = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
    "release_latest": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
}
_VERSION_NAMES = ["v1", "v2", "v3", "v4", "v5", "v6"]
for version_name in _VERSION_NAMES:
    ALLOWED_FILES[version_name] = [f"test{version_name[1:]}.jsonl" if version_name != "v1" else "test.jsonl"]
for start_idx in range(1, len(_VERSION_NAMES) + 1):
    for end_idx in range(start_idx + 1, len(_VERSION_NAMES) + 1):
        key = f"{_VERSION_NAMES[start_idx - 1]}_{_VERSION_NAMES[end_idx - 1]}"
        ALLOWED_FILES[key] = [
            f"test{idx}.jsonl" if idx != 1 else "test.jsonl"
            for idx in range(start_idx, end_idx + 1)
        ]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def bare_ollama_model_name(model: str) -> str:
    return model.split("/", 1)[-1] if model.startswith("ollama/") else model


def model_display_name(model: str) -> str:
    return bare_ollama_model_name(model)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl_append(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def shell_output(cmd: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, text=True)


def format_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "n/a"
    total = int(round(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def log_info(message: str) -> None:
    print(f"{COLOR_INFO}[info]{COLOR_RESET} {message}", flush=True)


def log_warn(message: str) -> None:
    print(f"{COLOR_WARN}[warn]{COLOR_RESET} {message}", flush=True)


def ensure_local_ollama_models(models: list[str]) -> dict[str, Any]:
    output = shell_output(["ollama", "list"], cwd=PROJECT_ROOT)
    lines = [line for line in output.splitlines() if line.strip()]
    available = {line.split()[0] for line in lines[1:]}
    expected = [bare_ollama_model_name(model) for model in models]
    missing = [model for model in expected if model not in available]
    if missing:
        raise RuntimeError(f"Missing Ollama models: {', '.join(missing)}")
    return {
        "checked_at": utc_now(),
        "available_models": sorted(available),
        "raw_output": output,
    }


def ensure_livecodebench_repo(repo_dir: Path, repo_url: str) -> dict[str, Any]:
    if not (repo_dir / "pyproject.toml").is_file():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        if repo_dir.exists():
            raise RuntimeError(f"LiveCodeBench repo dir exists but is incomplete: {repo_dir}")
        log_info(f"Cloning LiveCodeBench into {repo_dir}")
        subprocess.check_call(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], cwd=str(repo_dir.parent))

    git_head = None
    try:
        git_head = shell_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"]).strip()
    except Exception:
        pass
    return {
        "repo_dir": str(repo_dir),
        "repo_url": repo_url,
        "git_head": git_head,
    }


def ensure_livecodebench_dataset(dataset_dir: Path, dataset_repo_id: str) -> dict[str, Any]:
    required = {"README.md", "code_generation_lite.py", *ALLOWED_FILES["release_latest"]}
    missing = sorted(name for name in required if not (dataset_dir / name).is_file())
    downloaded = False

    if missing:
        log_info(f"Downloading LiveCodeBench dataset snapshot into {dataset_dir}")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=dataset_repo_id,
            repo_type="dataset",
            local_dir=str(dataset_dir),
        )
        downloaded = True

    files = sorted(path.name for path in dataset_dir.glob("test*.jsonl"))
    total_size_bytes = sum(path.stat().st_size for path in dataset_dir.glob("test*.jsonl"))
    return {
        "dataset_dir": str(dataset_dir),
        "dataset_repo_id": dataset_repo_id,
        "downloaded": downloaded,
        "files": files,
        "total_size_bytes": total_size_bytes,
    }


def import_livecodebench_modules(repo_dir: Path) -> dict[str, Any]:
    repo_path = str(repo_dir)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
    from lcb_runner.evaluation.pass_k_utils import estimate_pass_at_k, extract_instance_results

    return {
        "CodeGenerationProblem": CodeGenerationProblem,
        "codegen_metrics": codegen_metrics,
        "estimate_pass_at_k": estimate_pass_at_k,
        "extract_instance_results": extract_instance_results,
    }


def parse_release_files(release_version: str) -> list[str]:
    try:
        return ALLOWED_FILES[release_version]
    except KeyError as exc:
        supported = ", ".join(sorted(ALLOWED_FILES))
        raise ValueError(f"Unsupported release version {release_version!r}. Expected one of: {supported}") from exc


def build_prompt(problem: Any) -> tuple[str, str]:
    prompt = f"### Question:\n{problem.question_content}\n\n"
    if problem.starter_code:
        prompt += f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{problem.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return SYSTEM_MESSAGE_GENERIC, prompt


def extract_code_from_response(text: str) -> str:
    fenced = re.findall(r"```(?:python|Python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return fenced[-1].strip()
    answer_tag = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if answer_tag:
        return answer_tag[-1].strip()
    return text.strip()


def ollama_chat(
    *,
    ollama_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    request_timeout: int,
    seed: int | None,
) -> tuple[str, dict[str, Any]]:
    options: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "num_predict": max_tokens,
    }
    if seed is not None:
        options["seed"] = seed

    payload = {
        "model": bare_ollama_model_name(model),
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": options,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{ollama_url.rstrip('/')}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=request_timeout) as response:
        raw_payload = json.loads(response.read().decode("utf-8"))
    duration_seconds = time.perf_counter() - started
    message = raw_payload.get("message", {})
    content = str(message.get("content", ""))
    metadata = {
        "created_at": raw_payload.get("created_at"),
        "done_reason": raw_payload.get("done_reason"),
        "total_duration": raw_payload.get("total_duration"),
        "load_duration": raw_payload.get("load_duration"),
        "prompt_eval_count": raw_payload.get("prompt_eval_count"),
        "prompt_eval_duration": raw_payload.get("prompt_eval_duration"),
        "eval_count": raw_payload.get("eval_count"),
        "eval_duration": raw_payload.get("eval_duration"),
        "request_duration_seconds": duration_seconds,
    }
    return content, metadata


def request_completion_with_retries(
    *,
    ollama_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    request_timeout: int,
    request_retries: int,
    retry_backoff_seconds: int,
    seed: int | None,
) -> tuple[str, dict[str, Any]]:
    last_error: Exception | None = None
    attempts = max(1, request_retries)
    for attempt in range(1, attempts + 1):
        try:
            return ollama_chat(
                ollama_url=ollama_url,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                request_timeout=request_timeout,
                seed=seed,
            )
        except (TimeoutError, urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
            log_warn(
                f"{model_display_name(model)} request attempt {attempt}/{attempts} failed: {type(exc).__name__}: {exc}"
            )
            if attempt < attempts:
                time.sleep(retry_backoff_seconds)
    assert last_error is not None
    raise last_error


def load_generation_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            question_id = str(payload["question_id"])
            records[question_id] = payload
    return records


def load_benchmark(
    *,
    dataset_dir: Path,
    release_version: str,
    start_date: str | None,
    end_date: str | None,
    CodeGenerationProblem: Any,
) -> list[Any]:
    release_files = parse_release_files(release_version)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    problems: list[Any] = []
    for file_name in release_files:
        path = dataset_dir / file_name
        if not path.is_file():
            raise RuntimeError(f"LiveCodeBench dataset file not found: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                problem = CodeGenerationProblem(**row)
                if start_dt and problem.contest_date < start_dt:
                    continue
                if end_dt and problem.contest_date > end_dt:
                    continue
                problems.append(problem)
    problems.sort(key=lambda item: item.question_id)
    return problems


def build_problem_index(problems: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "question_id": problem.question_id,
            "question_title": problem.question_title,
            "contest_date": problem.contest_date.isoformat(),
            "platform": problem.platform.value,
            "difficulty": problem.difficulty.value,
            "has_starter_code": bool(problem.starter_code),
            "public_tests": len(problem.public_test_cases),
            "private_tests": len(problem.private_test_cases),
        }
        for problem in problems
    ]


def select_problems(problems: list[Any], limit: int | None) -> list[Any]:
    return problems if limit is None else problems[:limit]


def generate_for_problem(
    *,
    problem: Any,
    model: str,
    ollama_url: str,
    num_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    request_timeout: int,
    request_retries: int,
    retry_backoff_seconds: int,
    seed_base: int | None,
    existing_record: dict[str, Any] | None,
) -> dict[str, Any]:
    outputs = list(existing_record.get("output_list", [])) if existing_record else []
    codes = list(existing_record.get("code_list", [])) if existing_record else []
    sample_metadata = list(existing_record.get("sample_metadata", [])) if existing_record else []

    outputs = outputs[:num_samples]
    codes = codes[:num_samples]
    sample_metadata = sample_metadata[:num_samples]

    system_prompt, user_prompt = build_prompt(problem)
    started_at = utc_now()
    started_perf = time.perf_counter()

    for sample_index in range(len(codes), num_samples):
        seed = None if seed_base is None else seed_base + sample_index
        raw_output, metadata = request_completion_with_retries(
            ollama_url=ollama_url,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            request_retries=request_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            seed=seed,
        )
        outputs.append(raw_output)
        codes.append(extract_code_from_response(raw_output))
        sample_metadata.append(metadata)

    duration_seconds = time.perf_counter() - started_perf
    return {
        "question_id": problem.question_id,
        "question_title": problem.question_title,
        "contest_date": problem.contest_date.isoformat(),
        "platform": problem.platform.value,
        "difficulty": problem.difficulty.value,
        "started_at": started_at,
        "finished_at": utc_now(),
        "generation_duration_seconds": duration_seconds,
        "output_list": outputs,
        "code_list": codes,
        "sample_metadata": sample_metadata,
    }


def run_phase_for_model(
    *,
    phase_name: str,
    model: str,
    problems: list[Any],
    artifact_root: Path,
    ollama_url: str,
    num_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    request_timeout: int,
    request_retries: int,
    retry_backoff_seconds: int,
    seed_base: int | None,
) -> dict[str, Any]:
    model_slug = slugify(model)
    phase_dir = artifact_root / phase_name / model_slug
    generations_jsonl = phase_dir / "generations.jsonl"
    generations_json = phase_dir / "generations.json"
    result_path = phase_dir / "result.json"

    existing = load_generation_records(generations_jsonl)
    write_text(
        phase_dir / "command.sh",
        " ".join(
            [
                shlex.quote(str(DEFAULT_PYTHON)),
                shlex.quote(str(Path(__file__).resolve())),
                shlex.quote("--artifact-root"),
                shlex.quote(str(artifact_root)),
            ]
        )
        + "\n",
    )

    started_at = utc_now()
    started_perf = time.perf_counter()
    for index, problem in enumerate(problems, start=1):
        record = existing.get(problem.question_id)
        existing_samples = len(record.get("code_list", [])) if record else 0
        if existing_samples >= num_samples:
            log_info(
                f"{phase_name} {model_display_name(model)} [{index}/{len(problems)}] "
                f"reusing {problem.question_id} with {existing_samples} sample(s)"
            )
            continue

        log_info(
            f"{phase_name} {model_display_name(model)} [{index}/{len(problems)}] "
            f"generating {problem.question_id} ({problem.difficulty.value}, {problem.platform.value})"
        )
        updated_record = generate_for_problem(
            problem=problem,
            model=model,
            ollama_url=ollama_url,
            num_samples=num_samples,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            request_retries=request_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            seed_base=seed_base,
            existing_record=record,
        )
        existing[problem.question_id] = updated_record
        write_jsonl_append(generations_jsonl, updated_record)

    duration_seconds = time.perf_counter() - started_perf
    ordered_records = [existing[problem.question_id] for problem in problems]
    write_json(generations_json, ordered_records)

    result = {
        "phase": phase_name,
        "model": model,
        "model_slug": model_slug,
        "problems": len(problems),
        "num_samples": num_samples,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": duration_seconds,
        "generations_jsonl": str(generations_jsonl),
        "generations_json": str(generations_json),
    }
    write_json(result_path, result)
    return result


def estimate_problem_pass_at_k(estimate_pass_at_k: Any, samples: int, correct: int, k: int) -> float | None:
    if samples < k:
        return None
    return float(estimate_pass_at_k(samples, [correct], k)[0])


def evaluate_phase_for_model(
    *,
    phase_name: str,
    model: str,
    problems: list[Any],
    artifact_root: Path,
    num_samples: int,
    num_process_evaluate: int,
    timeout: int,
    codegen_metrics: Any,
    estimate_pass_at_k: Any,
    extract_instance_results: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_slug = slugify(model)
    phase_dir = artifact_root / phase_name / model_slug
    generations_jsonl = phase_dir / "generations.jsonl"
    generation_map = load_generation_records(generations_jsonl)

    ordered_records: list[dict[str, Any]] = []
    for problem in problems:
        record = generation_map.get(problem.question_id)
        if record is None:
            raise RuntimeError(f"Missing generation for {model} {phase_name} question {problem.question_id}")
        if len(record.get("code_list", [])) < num_samples:
            raise RuntimeError(
                f"Incomplete generations for {model} {phase_name} question {problem.question_id}: "
                f"{len(record.get('code_list', []))} < {num_samples}"
            )
        trimmed = dict(record)
        trimmed["output_list"] = list(record["output_list"][:num_samples])
        trimmed["code_list"] = list(record["code_list"][:num_samples])
        trimmed["sample_metadata"] = list(record.get("sample_metadata", [])[:num_samples])
        ordered_records.append(trimmed)

    eval_samples = [problem.get_evaluation_sample() for problem in problems]
    generations = [record["code_list"] for record in ordered_records]
    metrics, results, metadata = codegen_metrics(
        eval_samples,
        generations,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )
    graded = extract_instance_results(results)

    eval_all: list[dict[str, Any]] = []
    per_problem_rows: list[dict[str, Any]] = []
    for problem, record, graded_list, metadata_list in zip(problems, ordered_records, graded, metadata):
        correct_samples = sum(bool(item) for item in graded_list)
        samples = len(graded_list)
        pass_at_1 = estimate_problem_pass_at_k(estimate_pass_at_k, samples, correct_samples, 1)
        pass_at_5 = estimate_problem_pass_at_k(estimate_pass_at_k, samples, correct_samples, 5)
        eval_all.append(
            problem.insert_output_evaluation(
                record["output_list"],
                record["code_list"],
                graded_list,
                metadata=metadata_list,
            )
        )
        per_problem_rows.append(
            {
                "phase": phase_name,
                "model": model,
                "model_slug": model_slug,
                "question_id": problem.question_id,
                "question_title": problem.question_title,
                "contest_date": problem.contest_date.isoformat(),
                "platform": problem.platform.value,
                "difficulty": problem.difficulty.value,
                "samples": samples,
                "correct_samples": correct_samples,
                "questions_with_any_correct_sample": int(correct_samples > 0),
                "pass_at_1": pass_at_1,
                "pass_at_5": pass_at_5,
                "generation_duration_seconds": record.get("generation_duration_seconds"),
            }
        )

    eval_dir = phase_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    write_json(eval_dir / "metrics.json", metrics)
    write_json(eval_dir / "eval_all.json", eval_all)
    write_json(eval_dir / "results.json", results)

    generation_duration_seconds = sum(
        float(record.get("generation_duration_seconds", 0.0)) for record in ordered_records
    )
    overall_row = {
        "phase": phase_name,
        "model": model,
        "model_slug": model_slug,
        "problems": len(problems),
        "samples_per_problem": num_samples,
        "questions_with_any_correct_sample": sum(row["questions_with_any_correct_sample"] for row in per_problem_rows),
        "generation_duration_seconds": generation_duration_seconds,
        "average_seconds_per_problem": (
            generation_duration_seconds / len(problems) if problems else None
        ),
    }
    for key, value in metrics.items():
        if key == "detail":
            continue
        overall_row[key.replace("@", "_at_")] = value
    write_json(eval_dir / "overall_row.json", overall_row)
    return overall_row, per_problem_rows


def write_csv(path: Path, rows: list[dict[str, Any]], field_order: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(field_order) + "\n")
        for row in rows:
            values: list[str] = []
            for field in field_order:
                value = row.get(field)
                if value is None:
                    values.append("")
                else:
                    text = str(value).replace('"', '""')
                    if any(ch in text for ch in [",", "\n", '"']):
                        text = f'"{text}"'
                    values.append(text)
            handle.write(",".join(values) + "\n")


def compute_head_to_head(per_problem_rows: list[dict[str, Any]]) -> dict[str, Any]:
    models = sorted({row["model"] for row in per_problem_rows})
    if len(models) != 2:
        return {}

    left_rows = {
        row["question_id"]: row
        for row in per_problem_rows
        if row["model"] == models[0]
    }
    right_rows = {
        row["question_id"]: row
        for row in per_problem_rows
        if row["model"] == models[1]
    }

    margins: list[dict[str, Any]] = []
    unique_left: list[str] = []
    unique_right: list[str] = []
    for question_id in sorted(set(left_rows) & set(right_rows)):
        left = left_rows[question_id]
        right = right_rows[question_id]
        left_pass = float(left.get("pass_at_1") or 0.0)
        right_pass = float(right.get("pass_at_1") or 0.0)
        if left_pass > 0 and right_pass == 0:
            unique_left.append(left["question_title"])
        if right_pass > 0 and left_pass == 0:
            unique_right.append(right["question_title"])
        margins.append(
            {
                "question_title": left["question_title"],
                "margin": left_pass - right_pass,
            }
        )

    biggest_left = sorted(margins, key=lambda item: item["margin"], reverse=True)[:10]
    biggest_right = sorted(margins, key=lambda item: item["margin"])[:10]
    return {
        "model_a": models[0],
        "model_b": models[1],
        "unique_problem_wins": {
            models[0]: unique_left,
            models[1]: unique_right,
        },
        "largest_positive_margins": biggest_left,
        "largest_negative_margins": biggest_right,
    }


def build_summary_markdown(
    *,
    phase_name: str,
    artifact_root: Path,
    release_version: str,
    overall_rows: list[dict[str, Any]],
    head_to_head: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append(f"# LiveCodeBench {phase_name.title()} Summary")
    lines.append("")
    lines.append(f"Artifacts: `{artifact_root}`")
    lines.append(f"Release version: `{release_version}`")
    if overall_rows:
        lines.append(f"Samples per problem: `{overall_rows[0]['samples_per_problem']}`")
    lines.append("")
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append(
        "| Model | Problems | pass@1 | pass@5 | Any-Correct Problems | Avg Time / Problem | Duration |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(overall_rows, key=lambda item: item["model"]):
        lines.append(
            "| "
            + " | ".join(
                [
                    model_display_name(row["model"]),
                    str(int(row["problems"])),
                    format_pct(row.get("pass_at_1")),
                    format_pct(row.get("pass_at_5")),
                    str(int(row["questions_with_any_correct_sample"])),
                    format_seconds(row.get("average_seconds_per_problem")),
                    format_seconds(row.get("generation_duration_seconds")),
                ]
            )
            + " |"
        )

    if head_to_head:
        lines.append("")
        lines.append("## Head-to-Head")
        lines.append("")
        for model, wins in head_to_head["unique_problem_wins"].items():
            display = model_display_name(model)
            if wins:
                lines.append(f"- {display} unique wins: {', '.join(wins[:10])}")
            else:
                lines.append(f"- {display} unique wins: none")

    lines.append("")
    lines.append("## Raw Data")
    lines.append("")
    lines.append("- `overall_metrics.csv`")
    lines.append("- `per_problem.csv`")
    lines.append("- `summary.json`")
    return "\n".join(lines).rstrip() + "\n"


def save_phase_reports(
    *,
    phase_name: str,
    artifact_root: Path,
    release_version: str,
    overall_rows: list[dict[str, Any]],
    per_problem_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reports_dir = artifact_root / phase_name / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    head_to_head = compute_head_to_head(per_problem_rows)
    summary = {
        "phase": phase_name,
        "generated_at": utc_now(),
        "release_version": release_version,
        "overall_metrics": overall_rows,
        "head_to_head": head_to_head,
    }

    write_csv(
        reports_dir / "overall_metrics.csv",
        overall_rows,
        [
            "phase",
            "model",
            "model_slug",
            "problems",
            "samples_per_problem",
            "pass_at_1",
            "pass_at_5",
            "questions_with_any_correct_sample",
            "generation_duration_seconds",
            "average_seconds_per_problem",
        ],
    )
    write_csv(
        reports_dir / "per_problem.csv",
        per_problem_rows,
        [
            "phase",
            "model",
            "model_slug",
            "question_id",
            "question_title",
            "contest_date",
            "platform",
            "difficulty",
            "samples",
            "correct_samples",
            "questions_with_any_correct_sample",
            "pass_at_1",
            "pass_at_5",
            "generation_duration_seconds",
        ],
    )
    write_json(reports_dir / "summary.json", summary)
    write_text(
        reports_dir / "summary.md",
        build_summary_markdown(
            phase_name=phase_name,
            artifact_root=artifact_root,
            release_version=release_version,
            overall_rows=overall_rows,
            head_to_head=head_to_head,
        ),
    )
    return summary


def record_environment(
    *,
    python_executable: Path,
    repo_dir: Path,
    dataset_dir: Path,
    artifact_root: Path,
) -> None:
    payload = {
        "captured_at": utc_now(),
        "python_executable": str(python_executable),
        "python_version": shell_output([str(python_executable), "--version"], cwd=PROJECT_ROOT).strip(),
        "working_directory": str(PROJECT_ROOT),
        "livecodebench_repo_dir": str(repo_dir),
        "livecodebench_dataset_dir": str(dataset_dir),
    }
    try:
        payload["ollama_version"] = shell_output(["ollama", "--version"], cwd=PROJECT_ROOT).strip()
    except Exception:
        payload["ollama_version"] = None
    try:
        payload["livecodebench_git_head"] = shell_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"]).strip()
    except Exception:
        payload["livecodebench_git_head"] = None
    write_json(artifact_root / "metadata" / "environment.json", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run LiveCodeBench code-generation evaluations for local Ollama models. "
            "Defaults are tuned for practical local pass@1 runs; "
            "use --num-samples 10 --temperature 0.2 for a more official-style setup."
        )
    )
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--python-executable", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--release-version", default=DEFAULT_RELEASE_VERSION)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--smoke-limit", type=int, default=1)
    parser.add_argument("--benchmark-limit", type=int, default=None)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--num-process-evaluate", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--livecodebench-repo-dir", type=Path, default=DEFAULT_LCB_REPO_DIR)
    parser.add_argument("--livecodebench-dataset-dir", type=Path, default=DEFAULT_LCB_DATASET_DIR)
    parser.add_argument("--livecodebench-repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--livecodebench-dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "started_at": utc_now(),
        "status": "running",
        "artifact_root": str(artifact_root),
        "models": args.models,
        "release_version": args.release_version,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "smoke_limit": args.smoke_limit,
        "benchmark_limit": args.benchmark_limit,
        "skip_benchmark": args.skip_benchmark,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "num_process_evaluate": args.num_process_evaluate,
        "timeout": args.timeout,
        "ollama_url": args.ollama_url,
        "livecodebench_repo_dir": str(args.livecodebench_repo_dir),
        "livecodebench_dataset_dir": str(args.livecodebench_dataset_dir),
    }
    write_json(artifact_root / "run_manifest.json", manifest)

    try:
        repo_info = ensure_livecodebench_repo(args.livecodebench_repo_dir, args.livecodebench_repo_url)
        dataset_info = ensure_livecodebench_dataset(
            args.livecodebench_dataset_dir,
            args.livecodebench_dataset_repo_id,
        )
        modules = import_livecodebench_modules(args.livecodebench_repo_dir)
        CodeGenerationProblem = modules["CodeGenerationProblem"]
        codegen_metrics = modules["codegen_metrics"]
        estimate_pass_at_k = modules["estimate_pass_at_k"]
        extract_instance_results = modules["extract_instance_results"]

        record_environment(
            python_executable=args.python_executable,
            repo_dir=args.livecodebench_repo_dir,
            dataset_dir=args.livecodebench_dataset_dir,
            artifact_root=artifact_root,
        )
        write_json(artifact_root / "metadata" / "livecodebench_repo.json", repo_info)
        write_json(artifact_root / "metadata" / "livecodebench_dataset.json", dataset_info)
        ollama_info = ensure_local_ollama_models(args.models)
        write_json(artifact_root / "metadata" / "ollama_models.json", ollama_info)

        all_problems = load_benchmark(
            dataset_dir=args.livecodebench_dataset_dir,
            release_version=args.release_version,
            start_date=args.start_date,
            end_date=args.end_date,
            CodeGenerationProblem=CodeGenerationProblem,
        )
        if not all_problems:
            raise RuntimeError("LiveCodeBench filter produced zero problems.")

        smoke_problems = select_problems(all_problems, args.smoke_limit)
        benchmark_problems = select_problems(all_problems, args.benchmark_limit)

        write_json(artifact_root / "metadata" / "smoke_problem_index.json", build_problem_index(smoke_problems))
        if not args.skip_benchmark:
            write_json(
                artifact_root / "metadata" / "benchmark_problem_index.json",
                build_problem_index(benchmark_problems),
            )

        smoke_results: list[dict[str, Any]] = []
        smoke_overall_rows: list[dict[str, Any]] = []
        smoke_per_problem_rows: list[dict[str, Any]] = []
        for model in args.models:
            smoke_results.append(
                run_phase_for_model(
                    phase_name="smoke",
                    model=model,
                    problems=smoke_problems,
                    artifact_root=artifact_root,
                    ollama_url=args.ollama_url,
                    num_samples=args.num_samples,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    request_timeout=args.request_timeout,
                    request_retries=args.request_retries,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                    seed_base=args.seed_base,
                )
            )
            overall_row, per_problem_rows = evaluate_phase_for_model(
                phase_name="smoke",
                model=model,
                problems=smoke_problems,
                artifact_root=artifact_root,
                num_samples=args.num_samples,
                num_process_evaluate=args.num_process_evaluate,
                timeout=args.timeout,
                codegen_metrics=codegen_metrics,
                estimate_pass_at_k=estimate_pass_at_k,
                extract_instance_results=extract_instance_results,
            )
            smoke_overall_rows.append(overall_row)
            smoke_per_problem_rows.extend(per_problem_rows)

        smoke_summary = save_phase_reports(
            phase_name="smoke",
            artifact_root=artifact_root,
            release_version=args.release_version,
            overall_rows=smoke_overall_rows,
            per_problem_rows=smoke_per_problem_rows,
        )

        benchmark_results: list[dict[str, Any]] = []
        benchmark_summary: dict[str, Any] | None = None
        if not args.skip_benchmark:
            benchmark_overall_rows: list[dict[str, Any]] = []
            benchmark_per_problem_rows: list[dict[str, Any]] = []
            for model in args.models:
                benchmark_results.append(
                    run_phase_for_model(
                        phase_name="benchmark",
                        model=model,
                        problems=benchmark_problems,
                        artifact_root=artifact_root,
                        ollama_url=args.ollama_url,
                        num_samples=args.num_samples,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        request_timeout=args.request_timeout,
                        request_retries=args.request_retries,
                        retry_backoff_seconds=args.retry_backoff_seconds,
                        seed_base=args.seed_base,
                    )
                )
                overall_row, per_problem_rows = evaluate_phase_for_model(
                    phase_name="benchmark",
                    model=model,
                    problems=benchmark_problems,
                    artifact_root=artifact_root,
                    num_samples=args.num_samples,
                    num_process_evaluate=args.num_process_evaluate,
                    timeout=args.timeout,
                    codegen_metrics=codegen_metrics,
                    estimate_pass_at_k=estimate_pass_at_k,
                    extract_instance_results=extract_instance_results,
                )
                benchmark_overall_rows.append(overall_row)
                benchmark_per_problem_rows.extend(per_problem_rows)

            benchmark_summary = save_phase_reports(
                phase_name="benchmark",
                artifact_root=artifact_root,
                release_version=args.release_version,
                overall_rows=benchmark_overall_rows,
                per_problem_rows=benchmark_per_problem_rows,
            )

        manifest["finished_at"] = utc_now()
        manifest["status"] = "completed"
        manifest["smoke_results"] = smoke_results
        manifest["smoke_summary"] = smoke_summary
        manifest["benchmark_results"] = benchmark_results
        manifest["benchmark_summary"] = benchmark_summary
        write_json(artifact_root / "run_manifest.json", manifest)
        log_info(f"Completed LiveCodeBench run in {artifact_root}")
        return 0
    except Exception as exc:
        manifest["finished_at"] = utc_now()
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["traceback"] = traceback.format_exc()
        write_json(artifact_root / "run_manifest.json", manifest)
        failed_path = artifact_root / "FAILED.txt"
        write_text(failed_path, f"{manifest['error']}\n")
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
