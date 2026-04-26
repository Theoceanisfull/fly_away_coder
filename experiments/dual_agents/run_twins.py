from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

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

SYSTEM_PROMPT = """You are one half of a twin coding pair solving benchmark tasks together.

Rules:
- Return only plain text, never markdown explanations outside the requested tags.
- Put concise reasoning inside <analysis>...</analysis>.
- Put the full executable Python 3 solution inside <solution>...</solution>.
- Preserve the required function signature or starter code interface when present.
- Do not include markdown fences in the solution.
- Prefer robust, runnable code over stylistic polish.
"""


@dataclass(frozen=True)
class TaskExample:
    task: BenchmarkTask
    task_id: str
    task_prompt: str
    starter_code: str


@dataclass
class CandidateAttempt:
    label: str
    stage: str
    agent: str
    analysis: str
    solution: str
    raw_response: str
    generation_time_s: float
    evaluation_time_s: float = 0.0
    eval_result: EvalResult | None = None


@dataclass
class TaskRecord:
    task_id: str
    benchmark: str
    selected_label: str
    selected_via: str
    selected_stage: str
    selected_agent: str
    final_result: EvalResult
    attempts: list[CandidateAttempt]
    timings_s: dict[str, float]


def main() -> None:
    args = parse_args()
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

    agent_a_model = resolve_model_alias(args.agent_a_model)
    agent_b_model = resolve_model_alias(args.agent_b_model)
    base_url = normalize_api_base(args.api_base)
    clients = {
        "twin_a": OpenAI(
            base_url=base_url,
            api_key=args.api_key,
            timeout=args.request_timeout_s,
            max_retries=args.num_retries,
        ),
        "twin_b": OpenAI(
            base_url=base_url,
            api_key=args.api_key,
            timeout=args.request_timeout_s,
            max_retries=args.num_retries,
        ),
    }
    models = {"twin_a": agent_a_model, "twin_b": agent_b_model}

    summary = run_protocol(
        examples=splits["eval"],
        clients=clients,
        models=models,
        timeout_s=args.timeout_s,
        selection_strategy=args.selection_strategy,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        run_dir=run_dir,
    )
    save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=json_default))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Twin-agent coding benchmark runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser("preview")
    add_shared_arguments(preview)

    run = subparsers.add_parser("run")
    add_shared_arguments(run)
    run.add_argument("--agent-a-model", default="rnj", help="First twin model alias or full model name.")
    run.add_argument("--agent-b-model", default="devstral", help="Second twin model alias or full model name.")
    run.add_argument("--api-base", default="http://localhost:11434")
    run.add_argument("--api-key", default="")
    run.add_argument("--request-timeout-s", type=float, default=120.0)
    run.add_argument("--num-retries", type=int, default=0)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--max-tokens", type=int, default=3072)
    run.add_argument("--timeout-s", type=float, default=12.0, help="Hidden-test eval timeout per candidate.")
    run.add_argument(
        "--selection-strategy",
        choices=("local_eval", "consensus_first"),
        default="local_eval",
        help="How to pick the final candidate after the twin protocol.",
    )
    run.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", choices=("livecodebench", "scicode"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
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


def resolve_model_alias(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name, model_name)


def normalize_api_base(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def load_benchmark_tasks(args: argparse.Namespace) -> list[BenchmarkTask]:
    if args.benchmark == "livecodebench":
        tasks = load_livecodebench_tasks(release_tag=args.livecodebench_release)
    else:
        tasks = load_scicode_tasks(source=args.scicode_source, split=args.scicode_split)

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
) -> dict[str, list[TaskExample]]:
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


def _build_flat_slices(
    tasks: list[BenchmarkTask],
    *,
    train_size: int,
    val_size: int,
    eval_size: int,
) -> dict[str, list[TaskExample]]:
    total_needed = train_size + val_size + eval_size
    if len(tasks) < total_needed:
        raise ValueError(f"Requested {total_needed} tasks, but only loaded {len(tasks)}.")

    cursor = 0
    splits: dict[str, list[TaskExample]] = {}
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
) -> dict[str, list[TaskExample]]:
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
    grouped: dict[Any, list[BenchmarkTask]] = {}
    for task in tasks:
        grouped.setdefault(_task_group_key(task), []).append(task)

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


def make_example(task: BenchmarkTask) -> TaskExample:
    return TaskExample(task=task, task_id=task.task_id, task_prompt=task.prompt, starter_code=task.starter_code)


def run_protocol(
    *,
    examples: list[TaskExample],
    clients: dict[str, OpenAI],
    models: dict[str, str],
    timeout_s: float,
    selection_strategy: str,
    temperature: float,
    max_tokens: int,
    run_dir: Path,
) -> dict[str, Any]:
    task_artifacts_dir = run_dir / "task_artifacts"
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    protocol_times: list[float] = []
    selection_eval_times: list[float] = []
    total_times: list[float] = []
    solved_count = 0
    partial_count = 0
    selection_counts: dict[str, int] = {}

    total_examples = len(examples)
    for index, example in enumerate(examples, start=1):
        print(f"[task {index}/{total_examples}] {example.task_id}: starting twin protocol", flush=True)
        started = time.perf_counter()
        record = solve_task_with_twins(
            task=example.task,
            clients=clients,
            models=models,
            timeout_s=timeout_s,
            selection_strategy=selection_strategy,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        total_time_s = time.perf_counter() - started
        protocol_time_s = record.timings_s["generation"]
        selection_eval_time_s = record.timings_s["selection_eval"]
        result = record.final_result
        if result.score >= 1.0:
            solved_count += 1
        elif result.score > 0.0:
            partial_count += 1

        selection_counts[record.selected_label] = selection_counts.get(record.selected_label, 0) + 1
        protocol_times.append(protocol_time_s)
        selection_eval_times.append(selection_eval_time_s)
        total_times.append(total_time_s)
        print(
            f"[task {index}/{total_examples}] {example.task_id}: "
            f"selected {record.selected_label} score={result.score:.3f} total_time_s={total_time_s:.2f}",
            flush=True,
        )

        candidate_scores = {
            attempt.label: (attempt.eval_result.score if attempt.eval_result is not None else None)
            for attempt in record.attempts
        }
        rows.append(
            {
                "task_id": record.task_id,
                "score": result.score,
                "passed": result.passed,
                "total_tests": result.total,
                "feedback": result.feedback,
                "first_failure": result.first_failure,
                "selected_label": record.selected_label,
                "selected_stage": record.selected_stage,
                "selected_agent": record.selected_agent,
                "selected_via": record.selected_via,
                "protocol_time_s": protocol_time_s,
                "selection_eval_time_s": selection_eval_time_s,
                "total_time_s": total_time_s,
                "candidate_scores": candidate_scores,
            }
        )
        save_json(task_artifacts_dir / f"{sanitize_slug(record.task_id)}.json", record)

    mean_score = sum(row["score"] for row in rows) / len(rows) if rows else 0.0
    return {
        "mean_score": mean_score,
        "num_examples": len(rows),
        "solved_count": solved_count,
        "partial_count": partial_count,
        "zero_score_count": len(rows) - solved_count - partial_count,
        "selection_counts": selection_counts,
        "time_stats_s": {
            "generation": summarize_numeric_series(protocol_times),
            "selection_eval": summarize_numeric_series(selection_eval_times),
            "total": summarize_numeric_series(total_times),
        },
        "rows": rows,
    }


def solve_task_with_twins(
    *,
    task: BenchmarkTask,
    clients: dict[str, OpenAI],
    models: dict[str, str],
    timeout_s: float,
    selection_strategy: str,
    temperature: float,
    max_tokens: int,
) -> TaskRecord:
    attempts: list[CandidateAttempt] = []
    generation_started = time.perf_counter()
    print(f"  - {task.task_id}: twin_a draft", flush=True)

    draft_a = run_stage(
        client=clients["twin_a"],
        model=models["twin_a"],
        stage="draft",
        agent="twin_a",
        label="draft_twin_a",
        messages=build_draft_messages(task, agent_name="Twin A", twin_name="Twin B"),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    attempts.append(draft_a)
    print(f"  - {task.task_id}: twin_b draft", flush=True)

    draft_b = run_stage(
        client=clients["twin_b"],
        model=models["twin_b"],
        stage="draft",
        agent="twin_b",
        label="draft_twin_b",
        messages=build_draft_messages(task, agent_name="Twin B", twin_name="Twin A"),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    attempts.append(draft_b)
    print(f"  - {task.task_id}: twin_a revision", flush=True)

    revision_a = run_stage(
        client=clients["twin_a"],
        model=models["twin_a"],
        stage="revision",
        agent="twin_a",
        label="revision_twin_a",
        messages=build_revision_messages(
            task,
            agent_name="Twin A",
            twin_name="Twin B",
            own_attempt=draft_a,
            twin_attempt=draft_b,
        ),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    attempts.append(revision_a)
    print(f"  - {task.task_id}: twin_b revision", flush=True)

    revision_b = run_stage(
        client=clients["twin_b"],
        model=models["twin_b"],
        stage="revision",
        agent="twin_b",
        label="revision_twin_b",
        messages=build_revision_messages(
            task,
            agent_name="Twin B",
            twin_name="Twin A",
            own_attempt=draft_b,
            twin_attempt=draft_a,
        ),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    attempts.append(revision_b)
    print(f"  - {task.task_id}: twin_a consensus", flush=True)

    consensus_a = run_stage(
        client=clients["twin_a"],
        model=models["twin_a"],
        stage="consensus",
        agent="twin_a",
        label="consensus_twin_a",
        messages=build_consensus_messages(
            task,
            agent_name="Twin A",
            twin_name="Twin B",
            own_attempt=revision_a,
            twin_attempt=revision_b,
        ),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    attempts.append(consensus_a)
    print(f"  - {task.task_id}: twin_b consensus", flush=True)

    consensus_b = run_stage(
        client=clients["twin_b"],
        model=models["twin_b"],
        stage="consensus",
        agent="twin_b",
        label="consensus_twin_b",
        messages=build_consensus_messages(
            task,
            agent_name="Twin B",
            twin_name="Twin A",
            own_attempt=revision_b,
            twin_attempt=revision_a,
        ),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    attempts.append(consensus_b)

    generation_time_s = time.perf_counter() - generation_started
    selection_started = time.perf_counter()
    print(f"  - {task.task_id}: selecting final candidate via {selection_strategy}", flush=True)
    selected = select_attempt(attempts, task=task, timeout_s=timeout_s, selection_strategy=selection_strategy)
    selection_eval_time_s = time.perf_counter() - selection_started
    final_result = selected.eval_result if selected.eval_result is not None else evaluate_task(task, selected.solution, timeout_s=timeout_s)

    return TaskRecord(
        task_id=task.task_id,
        benchmark=task.benchmark,
        selected_label=selected.label,
        selected_via=selection_strategy,
        selected_stage=selected.stage,
        selected_agent=selected.agent,
        final_result=final_result,
        attempts=attempts,
        timings_s={
            "generation": generation_time_s,
            "selection_eval": selection_eval_time_s,
            "total": generation_time_s + selection_eval_time_s,
        },
    )


def run_stage(
    *,
    client: OpenAI,
    model: str,
    stage: str,
    agent: str,
    label: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> CandidateAttempt:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw_text = response.choices[0].message.content or ""
    parsed = parse_twin_response(raw_text)
    return CandidateAttempt(
        label=label,
        stage=stage,
        agent=agent,
        analysis=parsed["analysis"],
        solution=parsed["solution"],
        raw_response=raw_text,
        generation_time_s=time.perf_counter() - started,
    )


def parse_twin_response(raw_text: str) -> dict[str, str]:
    analysis = extract_tag(raw_text, "analysis").strip()
    solution = extract_tag(raw_text, "solution").strip()
    if not solution:
        solution = strip_code_fences(raw_text)
    else:
        solution = strip_code_fences(solution)
    return {"analysis": analysis, "solution": solution}


def extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else ""


def build_draft_messages(task: BenchmarkTask, *, agent_name: str, twin_name: str) -> list[dict[str, str]]:
    user_prompt = f"""You are {agent_name}. {twin_name} will solve the same task independently.

Solve the task from scratch.

Return exactly:
<analysis>
3-8 short lines on the key algorithm, interface constraints, and likely failure modes.
</analysis>
<solution>
full executable Python 3 code only
</solution>

Task:
{task.prompt}

Starter code:
{task.starter_code or "(none)"}
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_revision_messages(
    task: BenchmarkTask,
    *,
    agent_name: str,
    twin_name: str,
    own_attempt: CandidateAttempt,
    twin_attempt: CandidateAttempt,
) -> list[dict[str, str]]:
    user_prompt = f"""You are {agent_name}. Revise your draft after reviewing {twin_name}'s draft.

Your job:
- identify what {twin_name} got right or wrong
- preserve anything useful from either draft
- return a stronger full solution than your first draft

Return exactly:
<analysis>
Brief critique of both drafts and the main corrections in your revision.
</analysis>
<solution>
full executable Python 3 code only
</solution>

Task:
{task.prompt}

Starter code:
{task.starter_code or "(none)"}

Your prior analysis:
{own_attempt.analysis or "(none)"}

Your prior solution:
{own_attempt.solution or "(empty)"}

{twin_name}'s analysis:
{twin_attempt.analysis or "(none)"}

{twin_name}'s solution:
{twin_attempt.solution or "(empty)"}
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_consensus_messages(
    task: BenchmarkTask,
    *,
    agent_name: str,
    twin_name: str,
    own_attempt: CandidateAttempt,
    twin_attempt: CandidateAttempt,
) -> list[dict[str, str]]:
    user_prompt = f"""You are {agent_name}. This is the twin consensus step.

Merge the best parts of both revised solutions into the strongest final answer you can produce.
Do not average the drafts. Commit to one concrete implementation.

Return exactly:
<analysis>
Short merge rationale: what you kept, what you discarded, and the main correctness risks.
</analysis>
<solution>
full executable Python 3 code only
</solution>

Task:
{task.prompt}

Starter code:
{task.starter_code or "(none)"}

Your revised analysis:
{own_attempt.analysis or "(none)"}

Your revised solution:
{own_attempt.solution or "(empty)"}

{twin_name}'s revised analysis:
{twin_attempt.analysis or "(none)"}

{twin_name}'s revised solution:
{twin_attempt.solution or "(empty)"}
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def select_attempt(
    attempts: list[CandidateAttempt],
    *,
    task: BenchmarkTask,
    timeout_s: float,
    selection_strategy: str,
) -> CandidateAttempt:
    if selection_strategy == "consensus_first":
        for attempt in attempts:
            if attempt.stage == "consensus" and attempt.solution.strip():
                attempt.eval_result = evaluate_task(task, attempt.solution, timeout_s=timeout_s)
                return attempt
        fallback = next(attempt for attempt in attempts if attempt.solution.strip())
        fallback.eval_result = evaluate_task(task, fallback.solution, timeout_s=timeout_s)
        return fallback

    stage_priority = {"draft": 0, "revision": 1, "consensus": 2}
    best_attempt: CandidateAttempt | None = None
    best_key: tuple[float, int, int, int] | None = None

    for index, attempt in enumerate(attempts):
        started = time.perf_counter()
        attempt.eval_result = evaluate_task(task, attempt.solution, timeout_s=timeout_s)
        attempt.evaluation_time_s = time.perf_counter() - started
        key = (
            attempt.eval_result.score,
            attempt.eval_result.passed,
            stage_priority.get(attempt.stage, -1),
            -index,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_attempt = attempt

    if best_attempt is None:
        raise RuntimeError("No candidate attempts were produced.")
    return best_attempt


def make_run_dir(args: argparse.Namespace) -> Path:
    if getattr(args, "output_dir", None):
        run_dir = Path(args.output_dir)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    pair_slug = f"{sanitize_slug(resolve_model_alias(args.agent_a_model))}__{sanitize_slug(resolve_model_alias(args.agent_b_model))}"
    run_dir = ARTIFACTS_ROOT / args.benchmark / pair_slug / timestamp
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
    if isinstance(value, TaskExample):
        return asdict(value)
    if isinstance(value, CandidateAttempt):
        return {
            **asdict(value),
            "eval_result": asdict(value.eval_result) if value.eval_result is not None else None,
        }
    if isinstance(value, TaskRecord):
        return {
            **asdict(value),
            "final_result": asdict(value.final_result),
            "attempts": [
                {
                    **asdict(attempt),
                    "eval_result": asdict(attempt.eval_result) if attempt.eval_result is not None else None,
                }
                for attempt in value.attempts
            ],
        }
    raise TypeError(f"Unsupported JSON type: {type(value)!r}")


def summarize_splits(splits: dict[str, list[TaskExample]]) -> dict[str, Any]:
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


if __name__ == "__main__":
    main()
