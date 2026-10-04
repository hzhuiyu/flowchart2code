#!/usr/bin/env python3
"""
Backfill reflection iteration code into original samples.jsonl and run HumanEval evaluation.

Rules:
1) If a task first appears with passed=True in iteration 1~max_iterations, use that iteration's generated_code to replace the original completion.
2) Otherwise, if the task appears with passed=False in the max_iterations-th iteration, use that iteration's generated_code to replace the original completion.
3) All other tasks keep their original completion unchanged.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_HUMAN_EVAL_PATH = Path(__file__).resolve().parent.parent / "human-eval"
sys.path.insert(0, str(SRC_HUMAN_EVAL_PATH))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge reflection iterations into samples.jsonl and run HumanEval."
    )
    parser.add_argument(
        "--samples-file",
        type=str,
        default=str(
            PROJECT_ROOT
            / "output"
            / "Algorithm"
            / "qwen3-vl-plus-qwen3-vl-plus"
            / "samples.jsonl"
        ),
        help="Path to the original samples.jsonl",
    )
    parser.add_argument(
        "--reflection-dir",
        type=str,
        default=None,
        help="Reflection directory path (default: reflection directory under the same directory as samples-file)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Merged output samples.jsonl path (default: reflection/samples_reflection_merged.jsonl)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of iterations (default: 3)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Data root directory (default: PROJECT_ROOT/data)",
    )
    parser.add_argument(
        "--problem-file",
        type=str,
        default=None,
        help="Explicit problem_file path (default: auto-inferred)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of HumanEval concurrent workers",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Per-sample test timeout (seconds)",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def estimate_pass_at_k_python(num_samples: List[int], num_correct: List[int], k: int) -> float:
    """
    Pure Python version of pass@k estimation (for environments without numpy).
    """

    def estimator(n: int, c: int, kk: int) -> float:
        if n - c < kk:
            return 1.0
        prod = 1.0
        for i in range(n - c + 1, n + 1):
            prod *= 1.0 - kk / i
        return 1.0 - prod

    if not num_samples:
        return 0.0
    vals = [estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct)]
    return sum(vals) / len(vals)


def evaluate_functional_correctness_fallback(
    sample_file: str,
    k: List[int],
    n_workers: int,
    timeout: float,
    problem_file: str,
) -> Dict[str, float]:
    """
    HumanEval-compatible evaluation without numpy dependency.
    """
    from human_eval.data import read_problems, stream_jsonl, write_jsonl  # type: ignore
    from human_eval.execution import check_correctness  # type: ignore

    problems = read_problems(problem_file)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            completion = sample["completion"]
            if task_id not in problems:
                continue
            args = (problems[task_id], completion, timeout, completion_id[task_id])
            futures.append(executor.submit(check_correctness, *args))
            completion_id[task_id] += 1
            n_samples += 1

        for future in as_completed(futures):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    total: List[int] = []
    correct: List[int] = []
    for result_list in results.values():
        result_list.sort()
        passed = [r[1]["passed"] for r in result_list]
        total.append(len(passed))
        correct.append(sum(1 for p in passed if p))

    pass_at_k: Dict[str, float] = {}
    for kk in k:
        if total and all(t >= kk for t in total):
            pass_at_k[f"pass@{kk}"] = estimate_pass_at_k_python(total, correct, kk)

    def combine_results():
        used = defaultdict(list)
        for task_id, lst in results.items():
            used[task_id] = [x for x in lst]
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            if task_id not in used or not used[task_id]:
                continue
            result = used[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample

    out_file = sample_file + "_results.jsonl"
    write_jsonl(out_file, combine_results())
    return pass_at_k


def run_humaneval(
    sample_file: str,
    k: List[int],
    n_workers: int,
    timeout: float,
    problem_file: str,
) -> Dict[str, float]:
    """
    Prefer using the official evaluate_functional_correctness; fall back to pure Python compatible implementation on failure.
    """
    try:
        from human_eval.evaluation import evaluate_functional_correctness  # type: ignore

        return evaluate_functional_correctness(
            sample_file=sample_file,
            k=k,
            n_workers=n_workers,
            timeout=timeout,
            problem_file=problem_file,
        )
    except ModuleNotFoundError as e:
        print(f"  - Official evaluation dependency missing, falling back to compatible implementation: {e}")
        return evaluate_functional_correctness_fallback(
            sample_file=sample_file,
            k=k,
            n_workers=n_workers,
            timeout=timeout,
            problem_file=problem_file,
        )


def infer_problem_file(samples_file: Path, data_root: Path) -> Path:
    parts = set(samples_file.parts)
    if "HumanEval-V" in parts:
        return data_root / "HumanEval-V" / "HumanEval.jsonl"
    if "Algorithm" in parts:
        return data_root / "Algorithm" / "Algorithm.jsonl"
    if "MATH" in parts:
        return data_root / "MATH" / "MATH.jsonl"

    # Default fallback to HumanEval-V
    return data_root / "HumanEval-V" / "HumanEval.jsonl"


def normalize_difficulty(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text in {"easy", "medium", "hard"}:
        return text.capitalize()
    return None


def extract_difficulty(problem_row: Dict[str, Any]) -> Optional[str]:
    # MATH: difficulty; Algorithm: meta.difficulty
    diff = normalize_difficulty(problem_row.get("difficulty"))
    if diff is not None:
        return diff

    meta = problem_row.get("meta")
    if isinstance(meta, dict):
        diff = normalize_difficulty(meta.get("difficulty"))
        if diff is not None:
            return diff

    # Some data may have easy/medium/hard directly as the first line
    prompt = problem_row.get("prompt")
    if isinstance(prompt, str):
        stripped = prompt.strip()
        if stripped:
            first_line = stripped.splitlines()[0]
            diff = normalize_difficulty(first_line)
            if diff is not None:
                return diff

    return None


def build_task_difficulty_map(problem_file: Path) -> Dict[str, str]:
    task_to_difficulty: Dict[str, str] = {}
    for row in read_jsonl(problem_file):
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            continue
        diff = extract_difficulty(row)
        if diff is not None:
            task_to_difficulty[task_id] = diff
    return task_to_difficulty


def compute_difficulty_stats(
    results_rows: List[Dict[str, Any]],
    task_to_difficulty: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {
        "Easy": {"passed": 0.0, "total": 0.0, "rate": 0.0},
        "Medium": {"passed": 0.0, "total": 0.0, "rate": 0.0},
        "Hard": {"passed": 0.0, "total": 0.0, "rate": 0.0},
    }

    for row in results_rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            continue
        difficulty = task_to_difficulty.get(task_id)
        if difficulty not in stats:
            continue
        stats[difficulty]["total"] += 1
        if bool(row.get("passed", False)):
            stats[difficulty]["passed"] += 1

    for level in ("Easy", "Medium", "Hard"):
        total = int(stats[level]["total"])
        passed = int(stats[level]["passed"])
        stats[level]["rate"] = (passed / total) if total else 0.0

    return stats


def format_results_txt(
    eval_result: Dict[str, float],
    difficulty_stats: Dict[str, Dict[str, float]],
) -> str:
    pass_at_1 = float(eval_result.get("pass@1", 0.0))
    lines = [str({"pass@1": pass_at_1}), ""]
    for level in ("Easy", "Medium", "Hard"):
        passed = int(difficulty_stats[level]["passed"])
        total = int(difficulty_stats[level]["total"])
        rate = float(difficulty_stats[level]["rate"])
        lines.append(f"{level}: {passed}/{total} = {rate}")
    return "\n".join(lines) + "\n"


def build_iteration_maps(
    reflection_dir: Path, max_iterations: int
) -> Tuple[Dict[int, Dict[str, Dict[str, Any]]], Dict[int, Dict[str, Dict[str, Any]]]]:
    sample_maps: Dict[int, Dict[str, Dict[str, Any]]] = {}
    result_maps: Dict[int, Dict[str, Dict[str, Any]]] = {}

    for i in range(1, max_iterations + 1):
        samples_i = read_jsonl(reflection_dir / f"samples{i}.jsonl")
        results_i = read_jsonl(reflection_dir / f"samples.jsonl_results{i}.jsonl")

        sample_maps[i] = {}
        for row in samples_i:
            task_id = row.get("task_id")
            if task_id:
                sample_maps[i][task_id] = row

        result_maps[i] = {}
        for row in results_i:
            task_id = row.get("task_id")
            if task_id:
                result_maps[i][task_id] = row

    return sample_maps, result_maps


def select_iteration(
    task_id: str,
    sample_maps: Dict[int, Dict[str, Dict[str, Any]]],
    result_maps: Dict[int, Dict[str, Dict[str, Any]]],
    max_iterations: int,
) -> Tuple[Optional[int], str]:
    # First, find the first passed=True
    for i in range(1, max_iterations + 1):
        res = result_maps.get(i, {}).get(task_id)
        if res is not None and bool(res.get("passed", False)):
            return i, "first_passed"

    # Otherwise, find the max_iterations-th iteration with passed=False
    res_last = result_maps.get(max_iterations, {}).get(task_id)
    if res_last is not None and not bool(res_last.get("passed", False)):
        return max_iterations, f"iter_{max_iterations}_failed"

    return None, "keep_original"


def merge_samples(
    original_rows: List[Dict[str, Any]],
    sample_maps: Dict[int, Dict[str, Dict[str, Any]]],
    result_maps: Dict[int, Dict[str, Dict[str, Any]]],
    max_iterations: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    merged_rows: List[Dict[str, Any]] = []
    replaced = 0
    replaced_by_passed = 0
    replaced_by_last_failed = 0
    missing_generated_code = 0
    selection_stats: Dict[str, int] = {}

    for row in original_rows:
        task_id = row.get("task_id")
        if not task_id:
            merged_rows.append(row)
            continue

        selected_iter, reason = select_iteration(
            task_id=task_id,
            sample_maps=sample_maps,
            result_maps=result_maps,
            max_iterations=max_iterations,
        )
        selection_stats[reason] = selection_stats.get(reason, 0) + 1

        if selected_iter is None:
            merged_rows.append(row)
            continue

        iter_sample = sample_maps.get(selected_iter, {}).get(task_id, {})
        new_code = iter_sample.get("generated_code")

        if not isinstance(new_code, str) or not new_code.strip():
            missing_generated_code += 1
            merged_rows.append(row)
            continue

        new_row = dict(row)
        new_row["completion"] = new_code
        new_row["reflection_selected_iteration"] = selected_iter
        new_row["reflection_selected_reason"] = reason
        merged_rows.append(new_row)

        replaced += 1
        if reason == "first_passed":
            replaced_by_passed += 1
        else:
            replaced_by_last_failed += 1

    summary = {
        "total_rows": len(original_rows),
        "replaced_rows": replaced,
        "replaced_by_first_passed": replaced_by_passed,
        "replaced_by_last_failed": replaced_by_last_failed,
        "missing_generated_code": missing_generated_code,
        "selection_stats": selection_stats,
    }
    return merged_rows, summary


def main() -> None:
    args = parse_args()

    samples_file = Path(args.samples_file).resolve()
    if not samples_file.exists():
        raise FileNotFoundError(f"Samples file does not exist: {samples_file}")

    reflection_dir = (
        Path(args.reflection_dir).resolve()
        if args.reflection_dir
        else samples_file.parent / "reflection"
    )
    if not reflection_dir.exists():
        raise FileNotFoundError(f"Reflection directory does not exist: {reflection_dir}")

    output_file = (
        Path(args.output_file).resolve()
        if args.output_file
        else reflection_dir / "samples_reflection_merged.jsonl"
    )
    data_root = Path(args.data_root).resolve()
    problem_file = (
        Path(args.problem_file).resolve()
        if args.problem_file
        else infer_problem_file(samples_file=samples_file, data_root=data_root)
    )
    if not problem_file.exists():
        raise FileNotFoundError(f"Problem file does not exist: {problem_file}")

    print(f"[1/4] Reading original samples: {samples_file}")
    original_rows = read_jsonl(samples_file)
    print(f"  - Original sample count: {len(original_rows)}")

    print(f"[2/4] Reading reflection iteration files: {reflection_dir}")
    sample_maps, result_maps = build_iteration_maps(
        reflection_dir=reflection_dir,
        max_iterations=args.max_iterations,
    )
    for i in range(1, args.max_iterations + 1):
        print(
            f"  - iteration {i}: samples={len(sample_maps.get(i, {}))}, "
            f"results={len(result_maps.get(i, {}))}"
        )

    print("[3/4] Merging and backfilling completions ...")
    merged_rows, merge_summary = merge_samples(
        original_rows=original_rows,
        sample_maps=sample_maps,
        result_maps=result_maps,
        max_iterations=args.max_iterations,
    )
    write_jsonl(output_file, merged_rows)
    print(f"  - Merged output: {output_file}")
    print(f"  - Backfilled count: {merge_summary['replaced_rows']}/{merge_summary['total_rows']}")
    print(
        "  - Backfilled by first pass: "
        f"{merge_summary['replaced_by_first_passed']}, "
        "Backfilled by last iteration failure: "
        f"{merge_summary['replaced_by_last_failed']}"
    )
    print(f"  - Skipped due to missing generated_code: {merge_summary['missing_generated_code']}")

    print("[4/4] Running HumanEval ...")
    eval_result = run_humaneval(
        sample_file=str(output_file),
        k=[1],
        n_workers=args.n_workers,
        timeout=args.timeout,
        problem_file=str(problem_file),
    )

    merged_results_file = Path(str(output_file) + "_results.jsonl")
    results_rows = read_jsonl(merged_results_file)
    task_to_difficulty = build_task_difficulty_map(problem_file)
    difficulty_stats = compute_difficulty_stats(results_rows, task_to_difficulty)

    results_txt_path = output_file.parent / "results.txt"
    results_txt = format_results_txt(eval_result, difficulty_stats)
    with results_txt_path.open("w", encoding="utf-8") as f:
        f.write(results_txt)

    summary_path = output_file.parent / "reflection_merge_eval_summary.json"
    summary = {
        "samples_file": str(samples_file),
        "reflection_dir": str(reflection_dir),
        "output_file": str(output_file),
        "merged_results_file": str(merged_results_file),
        "results_txt": str(results_txt_path),
        "problem_file": str(problem_file),
        "merge_summary": merge_summary,
        "eval_result": eval_result,
        "difficulty_summary": difficulty_stats,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Overall Result ===")
    for key, value in eval_result.items():
        print(f"{key}: {value:.6f}")
    for level in ("Easy", "Medium", "Hard"):
        passed = int(difficulty_stats[level]["passed"])
        total = int(difficulty_stats[level]["total"])
        rate = float(difficulty_stats[level]["rate"])
        print(f"{level}: {passed}/{total} = {rate}")
    print(f"results_file: {merged_results_file}")
    print(f"results_txt: {results_txt_path}")
    print(f"summary_file: {summary_path}")


if __name__ == "__main__":
    main()
