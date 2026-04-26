from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = EXPERIMENTS_ROOT / "artifacts"
SWEEPS_ROOT = ARTIFACTS_ROOT / "sweeps"
RUN_GEPA_PATH = EXPERIMENTS_ROOT / "run_gepa.py"

DEFAULT_MODELS = ["rnj", "devstral"]
DEFAULT_BENCHMARKS = ["scicode", "livecodebench"]


def main() -> None:
    args = parse_args()
    sweep_root = prepare_sweep_root(args)
    logs_root = sweep_root / "logs"
    runs_root = sweep_root / "runs"
    analytics_root = sweep_root / "analytics"
    logs_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    analytics_root.mkdir(parents=True, exist_ok=True)

    run_plan = build_run_plan(args, runs_root, logs_root)
    save_json(
        sweep_root / "sweep_config.json",
        {
            "models": args.models,
            "benchmarks": args.benchmarks,
            "timeout_s": args.timeout_s,
            "num_threads": args.num_threads,
            "continue_on_failure": args.continue_on_failure,
            "run_plan": run_plan,
        },
    )

    status: dict[str, Any] = {
        "state": "running",
        "started_at_utc": utc_timestamp(),
        "current_run": None,
        "runs": [],
    }
    save_json(sweep_root / "status.json", status)

    print(f"Sweep root: {sweep_root}")
    for plan in run_plan:
        run_name = plan["run_name"]
        print(f"Starting {run_name}")
        status["current_run"] = run_name
        save_json(sweep_root / "status.json", status)

        run_started_perf = time.perf_counter()
        run_started_at = utc_timestamp()
        exit_code = run_single_plan(plan)
        run_finished_at = utc_timestamp()
        wall_time_s = time.perf_counter() - run_started_perf

        run_record: dict[str, Any] = {
            "run_name": run_name,
            "benchmark": plan["benchmark"],
            "model": plan["model"],
            "run_dir": plan["run_dir"],
            "log_path": plan["log_path"],
            "started_at_utc": run_started_at,
            "finished_at_utc": run_finished_at,
            "wall_time_s": wall_time_s,
            "exit_code": exit_code,
            "status": "completed" if exit_code == 0 else "failed",
        }

        run_stats_path = Path(plan["run_dir"]) / "run_stats.json"
        if run_stats_path.exists():
            run_record["run_stats"] = load_json(run_stats_path)

        status["runs"].append(run_record)
        status["current_run"] = None
        save_json(sweep_root / "status.json", status)

        if exit_code != 0 and not args.continue_on_failure:
            status["state"] = "failed"
            status["finished_at_utc"] = utc_timestamp()
            save_json(sweep_root / "status.json", status)
            build_analytics(sweep_root, status["runs"])
            raise SystemExit(exit_code)

    status["state"] = "completed"
    status["finished_at_utc"] = utc_timestamp()
    save_json(sweep_root / "status.json", status)
    build_analytics(sweep_root, status["runs"])
    print(f"Completed sweep: {sweep_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a detached prompt-optimization sweep across models and benchmarks.")
    parser.add_argument("--sweep-root", type=Path)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS)
    parser.add_argument(
        "--runs",
        nargs="+",
        help="Explicit benchmark:model pairs to run, for example scicode:devstral livecodebench:rnj.",
    )
    parser.add_argument("--timeout-s", type=float, default=12.0)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--reflection-model")
    parser.add_argument("--reflection-api-base", default="http://localhost:11434")
    parser.add_argument("--reflection-api-key", default="")
    parser.add_argument("--reflection-request-timeout-s", type=float)
    parser.add_argument("--reflection-num-retries", type=int, default=0)

    parser.add_argument("--livecodebench-train-groups", type=int, default=40)
    parser.add_argument("--livecodebench-val-groups", type=int, default=10)
    parser.add_argument("--livecodebench-eval-groups", type=int, default=40)
    parser.add_argument("--livecodebench-max-full-evals", type=int, default=2)
    parser.add_argument("--livecodebench-release", default="release_latest")

    parser.add_argument("--scicode-train-groups", type=int, default=24)
    parser.add_argument("--scicode-val-groups", type=int, default=8)
    parser.add_argument("--scicode-eval-groups", type=int, default=16)
    parser.add_argument("--scicode-max-full-evals", type=int, default=2)
    parser.add_argument("--scicode-source", choices=("huggingface", "local_sample"), default="huggingface")
    parser.add_argument("--scicode-split", default="test")
    return parser.parse_args()


def prepare_sweep_root(args: argparse.Namespace) -> Path:
    if args.sweep_root:
        sweep_root = args.sweep_root
    else:
        sweep_root = SWEEPS_ROOT / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_full_matrix"
    sweep_root.mkdir(parents=True, exist_ok=True)
    return sweep_root


def build_run_plan(args: argparse.Namespace, runs_root: Path, logs_root: Path) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    run_specs = parse_run_specs(args.runs) if args.runs else [(benchmark, model) for benchmark in args.benchmarks for model in args.models]
    for benchmark, model in run_specs:
        benchmark_cfg = benchmark_config(args, benchmark)
        run_name = f"{benchmark}__{model}"
        run_dir = runs_root / benchmark / model
        log_path = logs_root / f"{run_name}.log"
        cmd = [
            sys.executable,
            str(RUN_GEPA_PATH),
            "optimize",
            "--benchmark",
            benchmark,
            "--model",
            model,
            "--train-groups",
            str(benchmark_cfg["train_groups"]),
            "--val-groups",
            str(benchmark_cfg["val_groups"]),
            "--eval-groups",
            str(benchmark_cfg["eval_groups"]),
            "--timeout-s",
            str(args.timeout_s),
            "--num-threads",
            str(args.num_threads),
            "--max-full-evals",
            str(benchmark_cfg["max_full_evals"]),
            "--output-dir",
            str(run_dir),
        ]
        if args.reflection_model:
            cmd.extend(
                [
                    "--reflection-model",
                    args.reflection_model,
                    "--reflection-api-base",
                    args.reflection_api_base,
                    "--reflection-api-key",
                    args.reflection_api_key,
                    "--reflection-num-retries",
                    str(args.reflection_num_retries),
                ]
            )
            if args.reflection_request_timeout_s is not None:
                cmd.extend(["--reflection-request-timeout-s", str(args.reflection_request_timeout_s)])
        if benchmark == "livecodebench":
            cmd.extend(["--livecodebench-release", benchmark_cfg["release"]])
        else:
            cmd.extend(["--scicode-source", benchmark_cfg["source"], "--scicode-split", benchmark_cfg["split"]])
        plan.append(
            {
                "run_name": run_name,
                "benchmark": benchmark,
                "model": model,
                "cmd": cmd,
                "run_dir": str(run_dir),
                "log_path": str(log_path),
                "config": benchmark_cfg,
            }
        )
    return plan


def parse_run_specs(run_specs: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for spec in run_specs:
        benchmark, separator, model = spec.partition(":")
        if not separator or not benchmark or not model:
            raise ValueError(f"Invalid run spec {spec!r}. Expected benchmark:model.")
        parsed.append((benchmark, model))
    return parsed


def benchmark_config(args: argparse.Namespace, benchmark: str) -> dict[str, Any]:
    if benchmark == "livecodebench":
        return {
            "train_groups": args.livecodebench_train_groups,
            "val_groups": args.livecodebench_val_groups,
            "eval_groups": args.livecodebench_eval_groups,
            "max_full_evals": args.livecodebench_max_full_evals,
            "release": args.livecodebench_release,
        }
    if benchmark == "scicode":
        return {
            "train_groups": args.scicode_train_groups,
            "val_groups": args.scicode_val_groups,
            "eval_groups": args.scicode_eval_groups,
            "max_full_evals": args.scicode_max_full_evals,
            "source": args.scicode_source,
            "split": args.scicode_split,
        }
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def run_single_plan(plan: dict[str, Any]) -> int:
    log_path = Path(plan["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_handle:
        log_handle.write(f"# Command\n{' '.join(plan['cmd'])}\n\n")
        log_handle.flush()
        completed = subprocess.run(
            plan["cmd"],
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return completed.returncode


def build_analytics(sweep_root: Path, run_records: list[dict[str, Any]]) -> None:
    analytics_root = sweep_root / "analytics"
    analytics_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    combined_row_comparisons: list[dict[str, Any]] = []

    for run_record in run_records:
        run_stats = run_record.get("run_stats")
        if not run_stats:
            summary_rows.append(failed_summary_row(run_record))
            continue

        run_dir = Path(run_record["run_dir"])
        baseline_eval = load_json(run_dir / "baseline_eval.json")
        optimized_eval = load_json(run_dir / "optimized_eval.json")
        row_comparison = build_row_comparison(run_record, baseline_eval, optimized_eval)
        combined_row_comparisons.extend(row_comparison)
        comparison_path = analytics_root / f"{run_record['run_name']}_eval_comparison.csv"
        write_csv(comparison_path, row_comparison)
        summary_rows.append(success_summary_row(run_record, run_stats))

    summary_path = analytics_root / "summary.csv"
    write_csv(summary_path, summary_rows)
    if combined_row_comparisons:
        write_csv(analytics_root / "combined_eval_comparison.csv", combined_row_comparisons)

    summary_payload = {
        "generated_at_utc": utc_timestamp(),
        "runs": summary_rows,
        "benchmark_aggregates": aggregate_by_key(summary_rows, "benchmark"),
        "model_aggregates": aggregate_by_key(summary_rows, "model"),
    }
    save_json(analytics_root / "summary.json", summary_payload)
    (analytics_root / "report.md").write_text(render_report(summary_rows, summary_payload), encoding="utf-8")


def success_summary_row(run_record: dict[str, Any], run_stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": run_record["status"],
        "benchmark": run_record["benchmark"],
        "model": run_record["model"],
        "reflection_model": run_stats["reflection_model"],
        "run_dir": run_record["run_dir"],
        "log_path": run_record["log_path"],
        "wall_time_s": run_record["wall_time_s"],
        "train_examples": run_stats["split_counts"]["train_examples"],
        "val_examples": run_stats["split_counts"]["val_examples"],
        "eval_examples": run_stats["split_counts"]["eval_examples"],
        "baseline_mean_score": run_stats["baseline"]["mean_score"],
        "optimized_mean_score": run_stats["optimized"]["mean_score"],
        "mean_score_delta": run_stats["improvement"]["mean_score_delta"],
        "baseline_solved_count": run_stats["baseline"]["solved_count"],
        "optimized_solved_count": run_stats["optimized"]["solved_count"],
        "solved_count_delta": run_stats["improvement"]["solved_count_delta"],
        "baseline_avg_problem_time_s": run_stats["baseline"]["time_stats_s"]["total"]["mean"],
        "optimized_avg_problem_time_s": run_stats["optimized"]["time_stats_s"]["total"]["mean"],
        "optimization_duration_s": run_stats["durations_s"]["optimization"],
        "total_duration_s": run_stats["durations_s"]["total"],
        "improved_rows": run_stats["improvement"]["row_deltas"]["improved"],
        "regressed_rows": run_stats["improvement"]["row_deltas"]["regressed"],
        "unchanged_rows": run_stats["improvement"]["row_deltas"]["unchanged"],
    }


def failed_summary_row(run_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": run_record["status"],
        "benchmark": run_record["benchmark"],
        "model": run_record["model"],
        "reflection_model": "",
        "run_dir": run_record["run_dir"],
        "log_path": run_record["log_path"],
        "wall_time_s": run_record["wall_time_s"],
        "train_examples": "",
        "val_examples": "",
        "eval_examples": "",
        "baseline_mean_score": "",
        "optimized_mean_score": "",
        "mean_score_delta": "",
        "baseline_solved_count": "",
        "optimized_solved_count": "",
        "solved_count_delta": "",
        "baseline_avg_problem_time_s": "",
        "optimized_avg_problem_time_s": "",
        "optimization_duration_s": "",
        "total_duration_s": "",
        "improved_rows": "",
        "regressed_rows": "",
        "unchanged_rows": "",
    }


def build_row_comparison(run_record: dict[str, Any], baseline_eval: dict[str, Any], optimized_eval: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_by_id = {row["task_id"]: row for row in baseline_eval["rows"]}
    rows: list[dict[str, Any]] = []
    for optimized_row in optimized_eval["rows"]:
        baseline_row = baseline_by_id.get(optimized_row["task_id"], {})
        rows.append(
            {
                "run_name": run_record["run_name"],
                "benchmark": run_record["benchmark"],
                "model": run_record["model"],
                "task_id": optimized_row["task_id"],
                "baseline_score": baseline_row.get("score"),
                "optimized_score": optimized_row.get("score"),
                "score_delta": (optimized_row.get("score") or 0.0) - (baseline_row.get("score") or 0.0),
                "baseline_total_time_s": baseline_row.get("total_time_s"),
                "optimized_total_time_s": optimized_row.get("total_time_s"),
                "baseline_feedback": baseline_row.get("feedback"),
                "optimized_feedback": optimized_row.get("feedback"),
            }
        )
    return rows


def aggregate_by_key(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] != "completed":
            continue
        grouped.setdefault(str(row[key]), []).append(row)

    aggregates: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(grouped.items()):
        aggregates.append(
            {
                key: group_key,
                "runs": len(group_rows),
                "avg_baseline_mean_score": mean_of(group_rows, "baseline_mean_score"),
                "avg_optimized_mean_score": mean_of(group_rows, "optimized_mean_score"),
                "avg_mean_score_delta": mean_of(group_rows, "mean_score_delta"),
                "avg_total_duration_s": mean_of(group_rows, "total_duration_s"),
            }
        )
    return aggregates


def mean_of(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) not in ("", None)]
    return sum(values) / len(values) if values else 0.0


def render_report(summary_rows: list[dict[str, Any]], summary_payload: dict[str, Any]) -> str:
    lines = [
        "# Prompt Optimization Sweep",
        "",
        f"Generated at: {summary_payload['generated_at_utc']}",
        "",
        "## Runs",
        "",
        "| Benchmark | Model | Status | Baseline | Optimized | Delta | Baseline Solved | Optimized Solved | Total Duration (s) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {benchmark} | {model} | {status} | {baseline_mean_score} | {optimized_mean_score} | {mean_score_delta} | "
            "{baseline_solved_count} | {optimized_solved_count} | {total_duration_s} |".format(**row)
        )

    for section_name, aggregates in (("Benchmark aggregates", summary_payload["benchmark_aggregates"]), ("Model aggregates", summary_payload["model_aggregates"])):
        lines.extend(
            [
                "",
                f"## {section_name}",
                "",
                f"| Key | Runs | Avg Baseline | Avg Optimized | Avg Delta | Avg Total Duration (s) |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in aggregates:
            key_name = next(key for key in row.keys() if key not in {"runs", "avg_baseline_mean_score", "avg_optimized_mean_score", "avg_mean_score_delta", "avg_total_duration_s"})
            lines.append(
                f"| {row[key_name]} | {row['runs']} | {row['avg_baseline_mean_score']:.4f} | {row['avg_optimized_mean_score']:.4f} | {row['avg_mean_score_delta']:.4f} | {row['avg_total_duration_s']:.2f} |"
            )

    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    main()
