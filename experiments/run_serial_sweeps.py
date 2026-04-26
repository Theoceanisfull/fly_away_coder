from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
SERIAL_RUNS_ROOT = EXPERIMENTS_ROOT / "serial_runs"

SWEEP_SCRIPTS = {
    "react": EXPERIMENTS_ROOT / "ReAct" / "run_full_sweep.py",
    "pot": EXPERIMENTS_ROOT / "ProgramOfThought" / "run_full_sweep.py",
}

DEFAULT_SEQUENCE = ["react", "pot"]
DEFAULT_MODELS = ["rnj", "devstral"]
DEFAULT_BENCHMARKS = ["scicode", "livecodebench"]


def main() -> None:
    args = parse_args()
    if args.command == "launch":
        launch_detached(args)
        return
    run_serial_sweeps(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch ReAct and ProgramOfThought benchmark sweeps sequentially, with optional detached execution."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch = subparsers.add_parser("launch", help="Spawn the serial sweep runner as a detached background process.")
    add_shared_arguments(launch)

    run = subparsers.add_parser("run", help="Run the serial sweep runner in the current process.")
    add_shared_arguments(run)

    return parser.parse_args()


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, help="Top-level root for the serial run metadata and logs.")
    parser.add_argument("--sequence", nargs="+", choices=tuple(SWEEP_SCRIPTS.keys()), default=DEFAULT_SEQUENCE)
    parser.add_argument("--dry-run", action="store_true", help="Print the planned commands without executing them.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS)
    parser.add_argument(
        "--runs",
        nargs="+",
        help="Explicit benchmark:model pairs to run inside each sweep, for example scicode:devstral livecodebench:rnj.",
    )
    parser.add_argument("--api-base", default="http://localhost:11434")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=12.0)
    parser.add_argument("--request-timeout-s", type=float)
    parser.add_argument("--prediction-timeout-s", type=float)
    parser.add_argument("--num-retries", type=int, default=3)
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue to later phases even if one sweep exits non-zero, and forward the same behavior to child sweeps.",
    )

    parser.add_argument("--livecodebench-train-groups", type=int, default=40)
    parser.add_argument("--livecodebench-val-groups", type=int, default=10)
    parser.add_argument("--livecodebench-eval-groups", type=int, default=40)
    parser.add_argument("--livecodebench-max-iters", type=int)
    parser.add_argument("--livecodebench-max-tokens", type=int)
    parser.add_argument("--livecodebench-release", default="release_latest")

    parser.add_argument("--scicode-train-groups", type=int, default=24)
    parser.add_argument("--scicode-val-groups", type=int, default=8)
    parser.add_argument("--scicode-eval-groups", type=int, default=16)
    parser.add_argument("--scicode-max-iters", type=int)
    parser.add_argument("--scicode-max-tokens", type=int)
    parser.add_argument("--scicode-source", choices=("huggingface", "local_sample"), default="huggingface")
    parser.add_argument("--scicode-split", default="test")


def launch_detached(args: argparse.Namespace) -> None:
    root = prepare_root(args.root, args.sequence)
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "launcher.log"

    child_cmd = build_self_run_command(args, root)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "root": str(root),
                    "log_path": str(log_path),
                    "cmd": child_cmd,
                },
                indent=2,
            )
        )
        return
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            child_cmd,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    launch_record = {
        "pid": process.pid,
        "launched_at_utc": utc_timestamp(),
        "root": str(root),
        "log_path": str(log_path),
        "cmd": child_cmd,
    }
    save_json(root / "launch.json", launch_record)
    print(json.dumps(launch_record, indent=2))


def run_serial_sweeps(args: argparse.Namespace) -> None:
    root = prepare_root(args.root, args.sequence)
    root.mkdir(parents=True, exist_ok=True)
    logs_root = root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    status: dict[str, Any] = {
        "state": "running",
        "started_at_utc": utc_timestamp(),
        "sequence": list(args.sequence),
        "current_phase": None,
        "phases": [],
    }
    save_json(root / "status.json", status)

    print(f"Serial run root: {root}")
    if args.dry_run:
        planned = []
        for phase_name in args.sequence:
            phase_root = root / f"{phase_name}_sweep"
            log_path = logs_root / f"{phase_name}.log"
            planned.append(
                {
                    "phase": phase_name,
                    "sweep_root": str(phase_root),
                    "log_path": str(log_path),
                    "cmd": build_sweep_command(phase_name, args, phase_root),
                }
            )
        status["state"] = "planned"
        status["finished_at_utc"] = utc_timestamp()
        status["phases"] = planned
        save_json(root / "status.json", status)
        print(json.dumps({"root": str(root), "phases": planned}, indent=2))
        return
    for phase_name in args.sequence:
        phase_root = root / f"{phase_name}_sweep"
        log_path = logs_root / f"{phase_name}.log"
        cmd = build_sweep_command(phase_name, args, phase_root)

        print(f"Starting phase: {phase_name}")
        status["current_phase"] = phase_name
        save_json(root / "status.json", status)

        started_perf = time.perf_counter()
        started_at_utc = utc_timestamp()
        exit_code = run_command(cmd, log_path)
        finished_at_utc = utc_timestamp()
        wall_time_s = time.perf_counter() - started_perf

        phase_record = {
            "phase": phase_name,
            "started_at_utc": started_at_utc,
            "finished_at_utc": finished_at_utc,
            "wall_time_s": wall_time_s,
            "exit_code": exit_code,
            "status": "completed" if exit_code == 0 else "failed",
            "sweep_root": str(phase_root),
            "log_path": str(log_path),
            "cmd": cmd,
        }
        child_status_path = phase_root / "status.json"
        if child_status_path.exists():
            phase_record["child_status"] = load_json(child_status_path)

        status["phases"].append(phase_record)
        status["current_phase"] = None
        save_json(root / "status.json", status)

        if exit_code != 0 and not args.continue_on_failure:
            status["state"] = "failed"
            status["finished_at_utc"] = utc_timestamp()
            save_json(root / "status.json", status)
            raise SystemExit(exit_code)

    status["state"] = "completed"
    status["finished_at_utc"] = utc_timestamp()
    save_json(root / "status.json", status)
    print(f"Completed serial run: {root}")


def build_self_run_command(args: argparse.Namespace, root: Path) -> list[str]:
    cmd = [sys.executable, str(Path(__file__)), "run", "--root", str(root)]
    cmd.extend(["--sequence", *args.sequence])
    if args.dry_run:
        cmd.append("--dry-run")
    cmd.extend(["--models", *args.models])
    cmd.extend(["--benchmarks", *args.benchmarks])
    if args.runs:
        cmd.extend(["--runs", *args.runs])
    cmd.extend(common_flag_args(args))
    return cmd


def build_sweep_command(phase_name: str, args: argparse.Namespace, phase_root: Path) -> list[str]:
    cmd = [sys.executable, str(SWEEP_SCRIPTS[phase_name]), "--sweep-root", str(phase_root)]
    cmd.extend(["--models", *args.models])
    cmd.extend(["--benchmarks", *args.benchmarks])
    if args.runs:
        cmd.extend(["--runs", *args.runs])
    cmd.extend(common_flag_args(args))
    return cmd


def common_flag_args(args: argparse.Namespace) -> list[str]:
    cmd = [
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
        "--livecodebench-train-groups",
        str(args.livecodebench_train_groups),
        "--livecodebench-val-groups",
        str(args.livecodebench_val_groups),
        "--livecodebench-eval-groups",
        str(args.livecodebench_eval_groups),
        "--livecodebench-release",
        args.livecodebench_release,
        "--scicode-train-groups",
        str(args.scicode_train_groups),
        "--scicode-val-groups",
        str(args.scicode_val_groups),
        "--scicode-eval-groups",
        str(args.scicode_eval_groups),
        "--scicode-source",
        args.scicode_source,
        "--scicode-split",
        args.scicode_split,
    ]
    if args.request_timeout_s is not None:
        cmd.extend(["--request-timeout-s", str(args.request_timeout_s)])
    if args.prediction_timeout_s is not None:
        cmd.extend(["--prediction-timeout-s", str(args.prediction_timeout_s)])
    if args.continue_on_failure:
        cmd.append("--continue-on-failure")
    if args.livecodebench_max_iters is not None:
        cmd.extend(["--livecodebench-max-iters", str(args.livecodebench_max_iters)])
    if args.livecodebench_max_tokens is not None:
        cmd.extend(["--livecodebench-max-tokens", str(args.livecodebench_max_tokens)])
    if args.scicode_max_iters is not None:
        cmd.extend(["--scicode-max-iters", str(args.scicode_max_iters)])
    if args.scicode_max_tokens is not None:
        cmd.extend(["--scicode-max-tokens", str(args.scicode_max_tokens)])
    return cmd


def run_command(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"# Command\n{' '.join(cmd)}\n\n")
        log_handle.flush()
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return completed.returncode


def prepare_root(root: Path | None, sequence: list[str]) -> Path:
    if root is not None:
        return root
    slug = "_then_".join(sequence)
    return SERIAL_RUNS_ROOT / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{slug}"


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    main()
