#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
TASK_FILE = PROJECT_ROOT / "eval" / "inspect_ai" / "scicode.py"
TASK_FILE_ARGUMENT = str(TASK_FILE.relative_to(PROJECT_ROOT))
TEST_DATA_FILE = PROJECT_ROOT / "eval" / "data" / "test_data.h5"
DEFAULT_DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR?usp=drive_link"
)
DEFAULT_MODELS = [
    "ollama/rnj-1:latest",
    "ollama/devstral-small-2:latest",
]
SKIPPED_STEP_INDEXES = {
    ("13", 5),
    ("62", 0),
    ("76", 2),
}
COLOR_PALETTE = ["#275DAD", "#D16014", "#1E8A5A", "#A239CA"]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def task_model_dir_name(model: str) -> str:
    return model.replace("/", "-")


def model_display_name(model: str) -> str:
    return model.split("/", 1)[-1]


def bare_ollama_model_name(model: str) -> str:
    return model.split("/", 1)[-1] if model.startswith("ollama/") else model


def phase_background_dir(with_background: bool) -> str:
    return "with_background" if with_background else "without_background"


def quartile_bucket(idx: int, total: int) -> str:
    if total <= 0:
        return "Q1"
    return f"Q{min(4, max(1, math.ceil(((idx + 1) / total) * 4)))}"


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_status_text(path: Path) -> str:
    if not path.is_file():
        return "missing"
    content = path.read_text(encoding="utf-8").strip().splitlines()
    return content[0].strip() if content else "empty"


def parse_dependency_modules(required_dependencies: str) -> list[str]:
    modules: list[str] = []
    for line in required_dependencies.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("import "):
            imports = stripped.removeprefix("import ").split(",")
            for item in imports:
                module = item.strip().split(" as ")[0].strip().split(".")[0]
                if module:
                    modules.append(module)
        elif stripped.startswith("from "):
            lhs = stripped.removeprefix("from ").split(" import ", 1)[0].strip()
            module = lhs.split(".")[0]
            if module:
                modules.append(module)
    return sorted(set(modules)) or ["none"]


def run_and_log(cmd: list[str], cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] $ {shlex.join(cmd)}\n")
        handle.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            text=True,
        )
        return process.wait()


def shell_output(cmd: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, text=True)


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


def ensure_test_data(
    *,
    python_executable: Path,
    test_data_file: Path,
    drive_folder_url: str,
    artifact_root: Path,
) -> dict[str, Any]:
    download_log = artifact_root / "metadata" / "download_test_data.log"
    if test_data_file.is_file():
        return {
            "path": str(test_data_file),
            "size_bytes": test_data_file.stat().st_size,
            "downloaded": False,
        }

    download_dir = artifact_root / "metadata" / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python_executable),
        "-m",
        "gdown",
        "--folder",
        drive_folder_url,
        "--remaining-ok",
        "-O",
        str(download_dir),
    ]
    rc = run_and_log(cmd, PROJECT_ROOT, download_log)
    if rc != 0:
        raise RuntimeError(f"Failed to download SciCode test data, see {download_log}")

    downloaded_file = download_dir / "test_data.h5"
    if not downloaded_file.is_file():
        raise RuntimeError(f"Downloaded test data not found at {downloaded_file}")

    test_data_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded_file, test_data_file)
    return {
        "path": str(test_data_file),
        "size_bytes": test_data_file.stat().st_size,
        "downloaded": True,
        "download_log": str(download_log),
    }


def load_records(split: str) -> list[dict[str, Any]]:
    dataset = load_dataset("SciCode1/SciCode", split=split)
    return [dict(row) for row in dataset]


def select_records(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    return records if limit is None else records[:limit]


def select_smoke_records(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda row: (len(row["sub_steps"]), int(row["problem_id"])))
    return ordered if limit is None else ordered[:limit]


def build_record_index(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed: list[dict[str, Any]] = []
    for row in records:
        modules = parse_dependency_modules(row["required_dependencies"])
        indexed.append(
            {
                "problem_id": str(row["problem_id"]),
                "problem_name": row["problem_name"],
                "num_steps": len(row["sub_steps"]),
                "dependency_modules": modules,
                "required_dependencies": row["required_dependencies"],
                "step_numbers": [str(step["step_number"]) for step in row["sub_steps"]],
            }
        )
    return indexed


def run_eval_for_model(
    *,
    phase_name: str,
    model: str,
    split: str,
    limit: int | None,
    sample_ids: list[str] | None,
    artifact_root: Path,
    python_executable: Path,
    test_data_file: Path,
    with_background: bool,
    max_connections: int,
    max_samples: int,
    max_subprocesses: int,
) -> dict[str, Any]:
    model_slug = slugify(model)
    phase_dir = artifact_root / phase_name / model_slug
    inspect_log_dir = phase_dir / "inspect_logs"
    task_output_dir = phase_dir / "task_output"
    run_log = phase_dir / "run.log"
    result_path = phase_dir / "result.json"

    command = [
        str(python_executable),
        "-m",
        "inspect_ai",
        "eval",
        TASK_FILE_ARGUMENT,
        "--model",
        model,
        "--temperature",
        "0",
        "--max-connections",
        str(max_connections),
        "--max-samples",
        str(max_samples),
        "--max-subprocesses",
        str(max_subprocesses),
        "--log-format",
        "json",
        "--log-dir",
        str(inspect_log_dir),
        "--display",
        "plain",
        "-T",
        f"split={split}",
        "-T",
        f"output_dir={task_output_dir}",
        "-T",
        f"h5py_file={test_data_file}",
    ]
    if sample_ids:
        command.extend(["--sample-id", ",".join(sample_ids)])
    elif limit is not None:
        command.extend(["--limit", str(limit)])
    if with_background:
        command.extend(["-T", "with_background=True"])

    command_text = shlex.join(command)
    write_text(phase_dir / "command.sh", command_text + "\n")

    started_at = utc_now()
    started_perf = time.perf_counter()
    rc = run_and_log(command, PROJECT_ROOT, run_log, env=os.environ.copy())
    duration_seconds = time.perf_counter() - started_perf
    result = {
        "phase": phase_name,
        "model": model,
        "model_slug": model_slug,
        "split": split,
        "limit": limit,
        "sample_ids": sample_ids,
        "command": command,
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": duration_seconds,
        "returncode": rc,
        "run_log": str(run_log),
        "inspect_log_dir": str(inspect_log_dir),
        "task_output_dir": str(task_output_dir),
    }
    write_json(result_path, result)
    return result


def collect_phase_frames(
    *,
    phase_name: str,
    artifact_root: Path,
    models: list[str],
    records: list[dict[str, Any]],
    with_background: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_step_rows: list[dict[str, Any]] = []
    per_problem_rows: list[dict[str, Any]] = []

    for model in models:
        model_slug = slugify(model)
        phase_dir = artifact_root / phase_name / model_slug
        result_path = phase_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        output_dir = Path(result.get("task_output_dir", phase_dir / "task_output"))
        model_dir = output_dir / task_model_dir_name(model)
        logs_dir = model_dir / "evaluation_logs" / phase_background_dir(with_background)
        code_dir = model_dir / "generated_code" / phase_background_dir(with_background)

        for row in records:
            problem_id = str(row["problem_id"])
            problem_name = row["problem_name"]
            dependency_modules = parse_dependency_modules(row["required_dependencies"])
            nominal_total_steps = len(row["sub_steps"])
            executed_steps = 0
            passed_steps = 0
            failed_steps = 0
            timeout_steps = 0
            missing_steps = 0

            for idx, step in enumerate(row["sub_steps"]):
                skipped = (problem_id, idx) in SKIPPED_STEP_INDEXES
                if skipped:
                    continue
                executed_steps += 1
                step_number = str(step["step_number"])
                status = read_status_text(logs_dir / f"{step_number}.log")
                generated_code_path = code_dir / f"{problem_id}.{idx + 1}.py"
                passed = status == "pass"
                timed_out = status == "time out"
                missing = status == "missing"
                failed = status not in {"pass", "time out", "missing"}

                passed_steps += int(passed)
                timeout_steps += int(timed_out)
                missing_steps += int(missing)
                failed_steps += int(failed)

                per_step_rows.append(
                    {
                        "phase": phase_name,
                        "model": model,
                        "model_slug": model_slug,
                        "problem_id": problem_id,
                        "problem_name": problem_name,
                        "step_index": idx + 1,
                        "step_number": step_number,
                        "nominal_total_steps": nominal_total_steps,
                        "executed_total_steps": nominal_total_steps - sum(
                            1 for i in range(nominal_total_steps) if (problem_id, i) in SKIPPED_STEP_INDEXES
                        ),
                        "status": status,
                        "passed": int(passed),
                        "timed_out": int(timed_out),
                        "missing": int(missing),
                        "generated_code_exists": int(generated_code_path.is_file()),
                        "step_bucket": quartile_bucket(idx, nominal_total_steps),
                        "dependency_modules": dependency_modules,
                    }
                )

            per_problem_rows.append(
                {
                    "phase": phase_name,
                    "model": model,
                    "model_slug": model_slug,
                    "problem_id": problem_id,
                    "problem_name": problem_name,
                    "nominal_total_steps": nominal_total_steps,
                    "executed_total_steps": executed_steps,
                    "passed_steps": passed_steps,
                    "failed_steps": failed_steps,
                    "timeout_steps": timeout_steps,
                    "missing_steps": missing_steps,
                    "benchmark_subproblem_pass_rate": (
                        passed_steps / nominal_total_steps if nominal_total_steps else 0.0
                    ),
                    "executed_subproblem_pass_rate": (
                        passed_steps / executed_steps if executed_steps else 0.0
                    ),
                    "benchmark_problem_solved": int(passed_steps == nominal_total_steps and nominal_total_steps > 0),
                    "executed_problem_solved": int(passed_steps == executed_steps and executed_steps > 0),
                    "dependency_modules": dependency_modules,
                    "returncode": result.get("returncode"),
                    "duration_seconds": result.get("duration_seconds"),
                }
            )

    return pd.DataFrame(per_problem_rows), pd.DataFrame(per_step_rows)


def compute_overall_metrics(per_problem_df: pd.DataFrame, per_step_df: pd.DataFrame) -> pd.DataFrame:
    problem_counts = per_problem_df.groupby("model").agg(
        problems=("problem_id", "count"),
        solved_problems=("benchmark_problem_solved", "sum"),
        passed_steps=("passed_steps", "sum"),
        nominal_total_steps=("nominal_total_steps", "sum"),
        timeout_steps=("timeout_steps", "sum"),
        missing_steps=("missing_steps", "sum"),
        duration_seconds=("duration_seconds", "max"),
    )
    generated = per_step_df.groupby("model").agg(
        executed_total_steps=("step_number", "count"),
        generated_code_files=("generated_code_exists", "sum"),
    )
    overall = problem_counts.join(generated, how="left").reset_index()
    overall["main_problem_correctness"] = overall["solved_problems"] / overall["problems"]
    overall["subproblem_correctness"] = overall["passed_steps"] / overall["nominal_total_steps"]
    overall["timeout_rate"] = overall["timeout_steps"] / overall["nominal_total_steps"]
    overall["missing_rate"] = overall["missing_steps"] / overall["nominal_total_steps"]
    return overall


def compute_dependency_metrics(per_step_df: pd.DataFrame) -> pd.DataFrame:
    dep_rows = per_step_df.copy()
    dep_rows = dep_rows.explode("dependency_modules")
    dep_rows = dep_rows.rename(columns={"dependency_modules": "dependency_module"})
    metrics = dep_rows.groupby(["model", "dependency_module"]).agg(
        steps=("step_number", "count"),
        passed=("passed", "sum"),
        timeouts=("timed_out", "sum"),
        missing=("missing", "sum"),
    )
    metrics = metrics.reset_index()
    metrics["pass_rate"] = metrics["passed"] / metrics["steps"]
    return metrics


def compute_step_bucket_metrics(per_step_df: pd.DataFrame) -> pd.DataFrame:
    metrics = per_step_df.groupby(["model", "step_bucket"]).agg(
        steps=("step_number", "count"),
        passed=("passed", "sum"),
    )
    metrics = metrics.reset_index()
    metrics["pass_rate"] = metrics["passed"] / metrics["steps"]
    metrics["bucket_order"] = metrics["step_bucket"].str.removeprefix("Q").astype(int)
    return metrics.sort_values(["model", "bucket_order"])


def compute_head_to_head(per_problem_df: pd.DataFrame) -> dict[str, Any]:
    if per_problem_df["model"].nunique() != 2:
        return {}

    model_a, model_b = sorted(per_problem_df["model"].unique())
    left = per_problem_df[per_problem_df["model"] == model_a][
        ["problem_id", "problem_name", "benchmark_problem_solved", "executed_subproblem_pass_rate"]
    ].rename(
        columns={
            "benchmark_problem_solved": "a_solved",
            "executed_subproblem_pass_rate": "a_pass_rate",
        }
    )
    right = per_problem_df[per_problem_df["model"] == model_b][
        ["problem_id", "benchmark_problem_solved", "executed_subproblem_pass_rate"]
    ].rename(
        columns={
            "benchmark_problem_solved": "b_solved",
            "executed_subproblem_pass_rate": "b_pass_rate",
        }
    )
    merged = left.merge(right, on="problem_id", how="inner")
    unique_a = merged[(merged["a_solved"] == 1) & (merged["b_solved"] == 0)]["problem_name"].tolist()
    unique_b = merged[(merged["b_solved"] == 1) & (merged["a_solved"] == 0)]["problem_name"].tolist()
    merged["pass_rate_margin"] = merged["a_pass_rate"] - merged["b_pass_rate"]
    biggest_a = merged.sort_values("pass_rate_margin", ascending=False).head(5)[
        ["problem_name", "pass_rate_margin"]
    ]
    biggest_b = merged.sort_values("pass_rate_margin", ascending=True).head(5)[
        ["problem_name", "pass_rate_margin"]
    ]
    return {
        "model_a": model_a,
        "model_b": model_b,
        "unique_problem_wins": {
            model_a: unique_a,
            model_b: unique_b,
        },
        "largest_positive_margins": [
            {"problem_name": row["problem_name"], "margin": row["pass_rate_margin"]}
            for _, row in biggest_a.iterrows()
        ],
        "largest_negative_margins": [
            {"problem_name": row["problem_name"], "margin": row["pass_rate_margin"]}
            for _, row in biggest_b.iterrows()
        ],
    }


def save_overall_chart(overall_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(overall_df))
    width = 0.35
    ax.bar(
        [i - width / 2 for i in x],
        overall_df["main_problem_correctness"] * 100,
        width=width,
        label="Main Problem Correctness",
        color=COLOR_PALETTE[0],
    )
    ax.bar(
        [i + width / 2 for i in x],
        overall_df["subproblem_correctness"] * 100,
        width=width,
        label="Subproblem Correctness",
        color=COLOR_PALETTE[1],
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([model_display_name(model) for model in overall_df["model"]], rotation=0)
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 100)
    ax.set_title("SciCode Benchmark Overview")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_dependency_chart(dependency_df: pd.DataFrame, path: Path) -> None:
    shared = (
        dependency_df.groupby("dependency_module")["steps"]
        .sum()
        .sort_values(ascending=False)
        .head(8)
        .index.tolist()
    )
    chart_df = dependency_df[dependency_df["dependency_module"].isin(shared)].copy()
    if chart_df.empty:
        return
    pivot = chart_df.pivot(index="dependency_module", columns="model", values="pass_rate").fillna(0.0)
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", ax=ax, color=COLOR_PALETTE[: len(pivot.columns)])
    ax.set_ylabel("Pass Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Executed Substep Pass Rate by Dependency Family")
    ax.legend([model_display_name(model) for model in pivot.columns], title="Model")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_step_bucket_chart(step_bucket_df: pd.DataFrame, path: Path) -> None:
    if step_bucket_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (model, model_df) in enumerate(step_bucket_df.groupby("model")):
        model_df = model_df.sort_values("bucket_order")
        ax.plot(
            model_df["step_bucket"],
            model_df["pass_rate"],
            marker="o",
            linewidth=2,
            color=COLOR_PALETTE[idx % len(COLOR_PALETTE)],
            label=model_display_name(model),
        )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pass Rate")
    ax.set_title("Executed Substep Pass Rate by Step Quartile")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_problem_heatmap(per_problem_df: pd.DataFrame, path: Path) -> None:
    if per_problem_df.empty:
        return
    heatmap_df = per_problem_df.pivot(
        index="model",
        columns="problem_name",
        values="executed_subproblem_pass_rate",
    ).fillna(0.0)
    fig, ax = plt.subplots(figsize=(max(12, heatmap_df.shape[1] * 0.22), 3 + heatmap_df.shape[0] * 0.8))
    image = ax.imshow(heatmap_df.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels([model_display_name(model) for model in heatmap_df.index])
    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns, rotation=90, fontsize=8)
    ax.set_title("Executed Substep Pass Rate by Problem")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def model_strengths_and_weaknesses(
    model: str,
    overall_df: pd.DataFrame,
    dependency_df: pd.DataFrame,
    per_problem_df: pd.DataFrame,
    step_bucket_df: pd.DataFrame,
) -> dict[str, Any]:
    overall_row = overall_df[overall_df["model"] == model].iloc[0]
    dep_rows = dependency_df[
        (dependency_df["model"] == model) & (dependency_df["steps"] >= 5)
    ].sort_values("pass_rate", ascending=False)
    problem_rows = per_problem_df[per_problem_df["model"] == model].copy()
    problem_rows = problem_rows.sort_values("executed_subproblem_pass_rate", ascending=False)
    bucket_rows = step_bucket_df[step_bucket_df["model"] == model].sort_values("bucket_order")

    strongest_dependencies = dep_rows.head(3)[["dependency_module", "pass_rate", "steps"]].to_dict("records")
    weakest_dependencies = dep_rows.tail(3)[["dependency_module", "pass_rate", "steps"]].to_dict("records")
    strongest_problems = problem_rows.head(5)[
        ["problem_name", "executed_subproblem_pass_rate", "benchmark_problem_solved"]
    ].to_dict("records")
    weakest_problems = problem_rows.tail(5)[
        ["problem_name", "executed_subproblem_pass_rate", "benchmark_problem_solved"]
    ].to_dict("records")

    q1 = bucket_rows[bucket_rows["step_bucket"] == "Q1"]["pass_rate"]
    q4 = bucket_rows[bucket_rows["step_bucket"] == "Q4"]["pass_rate"]
    late_drop = float(q4.iloc[0] - q1.iloc[0]) if not q1.empty and not q4.empty else None

    return {
        "model": model,
        "headline_metrics": {
            "main_problem_correctness": float(overall_row["main_problem_correctness"]),
            "subproblem_correctness": float(overall_row["subproblem_correctness"]),
            "timeout_rate": float(overall_row["timeout_rate"]),
            "missing_rate": float(overall_row["missing_rate"]),
        },
        "strongest_dependencies": strongest_dependencies,
        "weakest_dependencies": weakest_dependencies,
        "strongest_problems": strongest_problems,
        "weakest_problems": weakest_problems,
        "late_step_vs_early_step_delta": late_drop,
    }


def build_summary_markdown(
    *,
    phase_name: str,
    artifact_root: Path,
    overall_df: pd.DataFrame,
    dependency_df: pd.DataFrame,
    per_problem_df: pd.DataFrame,
    step_bucket_df: pd.DataFrame,
    head_to_head: dict[str, Any],
    run_results: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append(f"# SciCode {phase_name.title()} Summary")
    lines.append("")
    lines.append(f"Artifacts: `{artifact_root}`")
    lines.append("")
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append(
        "| Model | Main Problem Correctness | Subproblem Correctness | Passed Steps | Timeouts | Missing | Duration | Return Code |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    result_map = {item["model"]: item for item in run_results}
    for _, row in overall_df.sort_values("model").iterrows():
        result = result_map[row["model"]]
        lines.append(
            "| "
            + " | ".join(
                [
                    model_display_name(row["model"]),
                    format_pct(float(row["main_problem_correctness"])),
                    format_pct(float(row["subproblem_correctness"])),
                    str(int(row["passed_steps"])),
                    str(int(row["timeout_steps"])),
                    str(int(row["missing_steps"])),
                    format_seconds(result.get("duration_seconds")),
                    str(result.get("returncode")),
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
                lines.append(f"- {display} unique solved problems: {', '.join(wins[:10])}")
            else:
                lines.append(f"- {display} unique solved problems: none")

    lines.append("")
    lines.append("## Model Notes")
    lines.append("")
    for model in sorted(overall_df["model"].unique()):
        detail = model_strengths_and_weaknesses(
            model, overall_df, dependency_df, per_problem_df, step_bucket_df
        )
        lines.append(f"### {model_display_name(model)}")
        lines.append("")
        strongest_deps = ", ".join(
            f"{item['dependency_module']} ({format_pct(item['pass_rate'])})"
            for item in detail["strongest_dependencies"]
        ) or "n/a"
        weakest_deps = ", ".join(
            f"{item['dependency_module']} ({format_pct(item['pass_rate'])})"
            for item in detail["weakest_dependencies"]
        ) or "n/a"
        strongest_probs = ", ".join(
            f"{item['problem_name']} ({format_pct(item['executed_subproblem_pass_rate'])})"
            for item in detail["strongest_problems"][:3]
        ) or "n/a"
        weakest_probs = ", ".join(
            f"{item['problem_name']} ({format_pct(item['executed_subproblem_pass_rate'])})"
            for item in detail["weakest_problems"][:3]
        ) or "n/a"
        lines.append(
            f"- Headline: {format_pct(detail['headline_metrics']['main_problem_correctness'])} main-problem correctness and {format_pct(detail['headline_metrics']['subproblem_correctness'])} subproblem correctness."
        )
        lines.append(f"- Strongest dependency families: {strongest_deps}.")
        lines.append(f"- Weakest dependency families: {weakest_deps}.")
        lines.append(f"- Strongest problems: {strongest_probs}.")
        lines.append(f"- Weakest problems: {weakest_probs}.")
        delta = detail["late_step_vs_early_step_delta"]
        if delta is not None:
            trend = "improves" if delta > 0 else "drops"
            lines.append(
                f"- Long-chain behavior: Q4 executed-step pass rate {trend} by {format_pct(abs(delta))} versus Q1."
            )
        lines.append("")

    lines.append("## Charts")
    lines.append("")
    lines.append("- `charts/overall_metrics.png`")
    lines.append("- `charts/dependency_pass_rates.png`")
    lines.append("- `charts/step_bucket_pass_rates.png`")
    lines.append("- `charts/problem_heatmap.png`")
    lines.append("")
    lines.append("## Raw Data")
    lines.append("")
    lines.append("- `overall_metrics.csv`")
    lines.append("- `per_problem.csv`")
    lines.append("- `per_step.csv`")
    lines.append("- `dependency_metrics.csv`")
    lines.append("- `step_bucket_metrics.csv`")
    return "\n".join(lines).rstrip() + "\n"


def save_phase_reports(
    *,
    phase_name: str,
    artifact_root: Path,
    models: list[str],
    records: list[dict[str, Any]],
    with_background: bool,
    run_results: list[dict[str, Any]],
) -> dict[str, Any]:
    per_problem_df, per_step_df = collect_phase_frames(
        phase_name=phase_name,
        artifact_root=artifact_root,
        models=models,
        records=records,
        with_background=with_background,
    )
    reports_dir = artifact_root / phase_name / "reports"
    charts_dir = reports_dir / "charts"
    reports_dir.mkdir(parents=True, exist_ok=True)

    overall_df = compute_overall_metrics(per_problem_df, per_step_df)
    dependency_df = compute_dependency_metrics(per_step_df)
    step_bucket_df = compute_step_bucket_metrics(per_step_df)
    head_to_head = compute_head_to_head(per_problem_df)

    overall_df.to_csv(reports_dir / "overall_metrics.csv", index=False)
    dependency_df.to_csv(reports_dir / "dependency_metrics.csv", index=False)
    step_bucket_df.to_csv(reports_dir / "step_bucket_metrics.csv", index=False)

    per_problem_export = per_problem_df.copy()
    per_problem_export["dependency_modules"] = per_problem_export["dependency_modules"].apply(
        lambda items: ",".join(items)
    )
    per_problem_export.to_csv(reports_dir / "per_problem.csv", index=False)

    per_step_export = per_step_df.copy()
    per_step_export["dependency_modules"] = per_step_export["dependency_modules"].apply(
        lambda items: ",".join(items)
    )
    per_step_export.to_csv(reports_dir / "per_step.csv", index=False)

    save_overall_chart(overall_df, charts_dir / "overall_metrics.png")
    save_dependency_chart(dependency_df, charts_dir / "dependency_pass_rates.png")
    save_step_bucket_chart(step_bucket_df, charts_dir / "step_bucket_pass_rates.png")
    save_problem_heatmap(per_problem_df, charts_dir / "problem_heatmap.png")

    summary = {
        "phase": phase_name,
        "generated_at": utc_now(),
        "overall_metrics": overall_df.to_dict("records"),
        "head_to_head": head_to_head,
    }
    write_json(reports_dir / "summary.json", summary)
    write_text(
        reports_dir / "summary.md",
        build_summary_markdown(
            phase_name=phase_name,
            artifact_root=artifact_root,
            overall_df=overall_df,
            dependency_df=dependency_df,
            per_problem_df=per_problem_df,
            step_bucket_df=step_bucket_df,
            head_to_head=head_to_head,
            run_results=run_results,
        ),
    )
    return summary


def record_environment(python_executable: Path, artifact_root: Path) -> None:
    payload = {
        "captured_at": utc_now(),
        "python_executable": str(python_executable),
        "python_version": shell_output([str(python_executable), "--version"], cwd=PROJECT_ROOT).strip(),
        "inspect_ai_version": shell_output(
            [str(python_executable), "-m", "inspect_ai", "info", "version"], cwd=PROJECT_ROOT
        ).strip(),
        "working_directory": str(PROJECT_ROOT),
    }
    write_json(artifact_root / "metadata" / "environment.json", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SciCode benchmarks for local Ollama models.")
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--python-executable", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--test-data-file", type=Path, default=TEST_DATA_FILE)
    parser.add_argument("--test-data-url", default=DEFAULT_DRIVE_FOLDER_URL)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--smoke-split", default="validation")
    parser.add_argument("--smoke-limit", type=int, default=1)
    parser.add_argument("--benchmark-split", default="test")
    parser.add_argument("--benchmark-limit", type=int, default=None)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--max-connections", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-subprocesses", type=int, default=1)
    parser.add_argument("--with-background", action="store_true")
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
        "smoke_split": args.smoke_split,
        "smoke_limit": args.smoke_limit,
        "benchmark_split": args.benchmark_split,
        "benchmark_limit": args.benchmark_limit,
        "skip_benchmark": args.skip_benchmark,
        "with_background": args.with_background,
        "task_file": str(TASK_FILE),
    }
    write_json(artifact_root / "run_manifest.json", manifest)

    try:
        record_environment(args.python_executable, artifact_root)
        ollama_info = ensure_local_ollama_models(args.models)
        write_json(artifact_root / "metadata" / "ollama_models.json", ollama_info)

        test_data_info = ensure_test_data(
            python_executable=args.python_executable,
            test_data_file=args.test_data_file,
            drive_folder_url=args.test_data_url,
            artifact_root=artifact_root,
        )
        write_json(artifact_root / "metadata" / "test_data.json", test_data_info)

        smoke_records = select_smoke_records(load_records(args.smoke_split), args.smoke_limit)
        benchmark_records = select_records(load_records(args.benchmark_split), args.benchmark_limit)
        write_json(artifact_root / "metadata" / "smoke_problem_index.json", build_record_index(smoke_records))
        if not args.skip_benchmark:
            write_json(
                artifact_root / "metadata" / "benchmark_problem_index.json",
                build_record_index(benchmark_records),
            )

        smoke_results: list[dict[str, Any]] = []
        for model in args.models:
            smoke_results.append(
                run_eval_for_model(
                    phase_name="smoke",
                    model=model,
                    split=args.smoke_split,
                    limit=args.smoke_limit,
                    sample_ids=[str(row["problem_id"]) for row in smoke_records],
                    artifact_root=artifact_root,
                    python_executable=args.python_executable,
                    test_data_file=args.test_data_file,
                    with_background=args.with_background,
                    max_connections=args.max_connections,
                    max_samples=args.max_samples,
                    max_subprocesses=args.max_subprocesses,
                )
            )
        if any(result["returncode"] != 0 for result in smoke_results):
            raise RuntimeError("Smoke test failed. See smoke/<model>/run.log in the artifact directory.")
        save_phase_reports(
            phase_name="smoke",
            artifact_root=artifact_root,
            models=args.models,
            records=smoke_records,
            with_background=args.with_background,
            run_results=smoke_results,
        )

        benchmark_results: list[dict[str, Any]] = []
        if not args.skip_benchmark:
            for model in args.models:
                benchmark_results.append(
                    run_eval_for_model(
                        phase_name="benchmark",
                        model=model,
                        split=args.benchmark_split,
                        limit=args.benchmark_limit,
                        sample_ids=None,
                        artifact_root=artifact_root,
                        python_executable=args.python_executable,
                        test_data_file=args.test_data_file,
                        with_background=args.with_background,
                        max_connections=args.max_connections,
                        max_samples=args.max_samples,
                        max_subprocesses=args.max_subprocesses,
                    )
                )
            save_phase_reports(
                phase_name="benchmark",
                artifact_root=artifact_root,
                models=args.models,
                records=benchmark_records,
                with_background=args.with_background,
                run_results=benchmark_results,
            )

        manifest["finished_at"] = utc_now()
        manifest["status"] = "completed"
        manifest["smoke_results"] = smoke_results
        manifest["benchmark_results"] = benchmark_results
        write_json(artifact_root / "run_manifest.json", manifest)
        return 0
    except Exception as exc:
        manifest["finished_at"] = utc_now()
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        manifest["traceback"] = traceback.format_exc()
        write_json(artifact_root / "run_manifest.json", manifest)
        write_text(artifact_root / "FAILED.txt", manifest["traceback"])
        print(manifest["traceback"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
