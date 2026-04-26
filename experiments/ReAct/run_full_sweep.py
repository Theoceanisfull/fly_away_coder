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
RUN_REACT_PATH = EXPERIMENTS_ROOT / "run_react.py"

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
            "temperature": args.temperature,
            "api_base": args.api_base,
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

        summary_path = Path(plan["run_dir"]) / "summary.json"
        if summary_path.exists():
            run_record["summary"] = load_json(summary_path)

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
    parser = argparse.ArgumentParser(description="Run a ReAct sweep across models and benchmarks.")
    parser.add_argument("--sweep-root", type=Path)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS)
    parser.add_argument(
        "--runs",
        nargs="+",
        help="Explicit benchmark:model pairs to run, for example scicode:devstral livecodebench:rnj.",
    )
    parser.add_argument("--api-base", default="http://localhost:11434")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=12.0)
    parser.add_argument("--request-timeout-s", type=float)
    parser.add_argument("--num-retries", type=int, default=3)
    parser.add_argument("--continue-on-failure", action="store_true")

    parser.add_argument("--livecodebench-train-groups", type=int, default=40)
    parser.add_argument("--livecodebench-val-groups", type=int, default=10)
    parser.add_argument("--livecodebench-eval-groups", type=int, default=40)
    parser.add_argument("--livecodebench-max-iters", type=int, default=5)
    parser.add_argument("--livecodebench-max-tokens", type=int, default=2048)
    parser.add_argument("--livecodebench-release", default="release_latest")

    parser.add_argument("--scicode-train-groups", type=int, default=24)
    parser.add_argument("--scicode-val-groups", type=int, default=8)
    parser.add_argument("--scicode-eval-groups", type=int, default=16)
    parser.add_argument("--scicode-max-iters", type=int, default=6)
    parser.add_argument("--scicode-max-tokens", type=int, default=3072)
    parser.add_argument("--scicode-source", choices=("huggingface", "local_sample"), default="huggingface")
    parser.add_argument("--scicode-split", default="test")
    return parser.parse_args()


def prepare_sweep_root(args: argparse.Namespace) -> Path:
    if args.sweep_root:
        sweep_root = args.sweep_root
    else:
        sweep_root = SWEEPS_ROOT / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_react_matrix"
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
            str(RUN_REACT_PATH),
            "evaluate",
            "--benchmark",
            benchmark,
            "--model",
            model,
            "--api-base",
            args.api_base,
            "--api-key",
            args.api_key,
            "--temperature",
            str(args.temperature),
            "--timeout-s",
            str(args.timeout_s),
            "--num-retries",
            str(args.num_retries),
            "--train-groups",
            str(benchmark_cfg["train_groups"]),
            "--val-groups",
            str(benchmark_cfg["val_groups"]),
            "--eval-groups",
            str(benchmark_cfg["eval_groups"]),
            "--max-iters",
            str(benchmark_cfg["max_iters"]),
            "--max-tokens",
            str(benchmark_cfg["max_tokens"]),
            "--output-dir",
            str(run_dir),
        ]
        if args.request_timeout_s is not None:
            cmd.extend(["--request-timeout-s", str(args.request_timeout_s)])
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
            "max_iters": args.livecodebench_max_iters,
            "max_tokens": args.livecodebench_max_tokens,
            "release": args.livecodebench_release,
        }
    if benchmark == "scicode":
        return {
            "train_groups": args.scicode_train_groups,
            "val_groups": args.scicode_val_groups,
            "eval_groups": args.scicode_eval_groups,
            "max_iters": args.scicode_max_iters,
            "max_tokens": args.scicode_max_tokens,
            "source": args.scicode_source,
            "split": args.scicode_split,
        }
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def run_single_plan(plan: dict[str, Any]) -> int:
    log_path = Path(plan["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
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
    for run_record in run_records:
        summary = run_record.get("summary")
        if not summary:
            summary_rows.append(failed_summary_row(run_record))
            continue
        summary_rows.append(success_summary_row(run_record, summary))

    write_csv(analytics_root / "summary.csv", summary_rows)
    summary_payload = {
        "generated_at_utc": utc_timestamp(),
        "runs": summary_rows,
        "benchmark_aggregates": aggregate_by_key(summary_rows, "benchmark"),
        "model_aggregates": aggregate_by_key(summary_rows, "model"),
    }
    save_json(analytics_root / "summary.json", summary_payload)
    (analytics_root / "report.md").write_text(render_report(summary_rows, summary_payload), encoding="utf-8")


def success_summary_row(run_record: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": run_record["status"],
        "benchmark": run_record["benchmark"],
        "model": run_record["model"],
        "run_dir": run_record["run_dir"],
        "log_path": run_record["log_path"],
        "wall_time_s": run_record["wall_time_s"],
        "mean_score": summary["mean_score"],
        "solved_count": summary["solved_count"],
        "partial_count": summary["partial_count"],
        "zero_score_count": summary["zero_score_count"],
        "num_examples": summary["num_examples"],
        "avg_problem_time_s": summary["time_stats_s"]["total"]["mean"],
        "avg_prediction_time_s": summary["time_stats_s"]["prediction"]["mean"],
        "avg_agent_steps": summary["agent_step_stats"]["mean"],
        "max_agent_steps": summary["agent_step_stats"]["max"],
        "tool_name_counts": json.dumps(summary.get("tool_name_counts", {}), sort_keys=True),
    }


def failed_summary_row(run_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": run_record["status"],
        "benchmark": run_record["benchmark"],
        "model": run_record["model"],
        "run_dir": run_record["run_dir"],
        "log_path": run_record["log_path"],
        "wall_time_s": run_record["wall_time_s"],
        "mean_score": None,
        "solved_count": None,
        "partial_count": None,
        "zero_score_count": None,
        "num_examples": None,
        "avg_problem_time_s": None,
        "avg_prediction_time_s": None,
        "avg_agent_steps": None,
        "max_agent_steps": None,
        "tool_name_counts": "{}",
    }


def aggregate_by_key(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] != "completed":
            continue
        groups.setdefault(str(row[key]), []).append(row)

    aggregates: list[dict[str, Any]] = []
    for group_key, items in sorted(groups.items()):
        aggregates.append(
            {
                key: group_key,
                "runs": len(items),
                "mean_of_mean_scores": average([float(item["mean_score"]) for item in items]),
                "mean_solved_count": average([float(item["solved_count"]) for item in items]),
                "mean_problem_time_s": average([float(item["avg_problem_time_s"]) for item in items]),
                "mean_agent_steps": average([float(item["avg_agent_steps"]) for item in items]),
            }
        )
    return aggregates


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary_rows: list[dict[str, Any]], summary_payload: dict[str, Any]) -> str:
    lines = [
        "# ReAct Sweep Report",
        "",
        f"Generated at: {summary_payload['generated_at_utc']}",
        "",
        "## Runs",
        "",
    ]

    for row in summary_rows:
        lines.append(f"- `{row['benchmark']} + {row['model']}`: status `{row['status']}`, mean score `{row['mean_score']}`")

    if summary_payload["benchmark_aggregates"]:
        lines.extend(["", "## Benchmark Aggregates", ""])
        for row in summary_payload["benchmark_aggregates"]:
            lines.append(
                f"- `{row['benchmark']}`: mean score `{row['mean_of_mean_scores']:.4f}`, "
                f"mean solved `{row['mean_solved_count']:.2f}`, "
                f"mean steps `{row['mean_agent_steps']:.2f}`"
            )

    if summary_payload["model_aggregates"]:
        lines.extend(["", "## Model Aggregates", ""])
        for row in summary_payload["model_aggregates"]:
            lines.append(
                f"- `{row['model']}`: mean score `{row['mean_of_mean_scores']:.4f}`, "
                f"mean solved `{row['mean_solved_count']:.2f}`, "
                f"mean steps `{row['mean_agent_steps']:.2f}`"
            )

    return "\n".join(lines) + "\n"


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    main()
