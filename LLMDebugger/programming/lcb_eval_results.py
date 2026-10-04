"""Official LiveCodeBench re-eval after an LDB / simple run.

Reads the per-problem log jsonl, converts `solution` into flowchart2code
samples (`task_id` + `completion`), grades with `evaluate_lcb_samples` against
`data/LiveCodeBench/LiveCodeBench_merged.jsonl` (same path as
`src/scripts/evaluate_all_datasets.py`), and writes:

  LLMDebugger/results/<dataset>/<model>/results.json
  LLMDebugger/results/<dataset>/<model>/results.txt
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_LDB_ROOT = _THIS_DIR.parent
_F2C_ROOT = _LDB_ROOT.parent
_SRC_ROOT = _F2C_ROOT / "src"
_MERGED_PROBLEMS = _F2C_ROOT / "data" / "LiveCodeBench" / "LiveCodeBench_merged.jsonl"

if str(_SRC_ROOT) not in sys.path:
    sys.path.append(str(_SRC_ROOT))


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_lcb_log(rows: Iterable[dict], dataset_path: str = "", log_path: str = "") -> bool:
    haystack = f"{dataset_path} {log_path}".lower().replace("\\", "/")
    if "livecodebench" in haystack:
        return True
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if task_id.startswith("LiveCodeBench/") or task_id.startswith("v"):
            return True
        break
    return False


def to_f2c_task_id(task_id: str) -> str:
    """`LiveCodeBench/v1_2757` -> `v1_2757` (flowchart2code sample id)."""
    task_id = str(task_id or "").strip()
    if "/" in task_id:
        task_id = task_id.split("/", 1)[1]
    return task_id


def ldb_log_to_samples(rows: List[dict]) -> List[dict]:
    samples = []
    seen = set()
    for row in rows:
        task_id = to_f2c_task_id(row.get("task_id", ""))
        completion = row.get("solution") or row.get("completion") or ""
        if not task_id:
            continue
        # Keep the last write for a task (resume / retries).
        if task_id in seen:
            samples = [s for s in samples if s["task_id"] != task_id]
        seen.add(task_id)
        samples.append({"task_id": task_id, "completion": completion})
    return samples


def infer_dataset_and_model(
    log_path: str,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
) -> Tuple[str, str]:
    parts = Path(log_path).resolve().parts
    lowered = [p.lower() for p in parts]
    dataset_name = dataset or "LiveCodeBench"
    model_name = model or ""
    if "livecodebench" in lowered:
        dataset_name = "LiveCodeBench"
        idx = lowered.index("livecodebench")
        if not model_name and idx + 1 < len(parts) - 1:
            model_name = parts[idx + 1]
    if not model_name:
        model_name = "unknown"
    return dataset_name, model_name


def results_dir(dataset: str, model: str) -> Path:
    return _LDB_ROOT / "results" / dataset / model


def save_official_results(
    out_dir: Path,
    pass_at_1: float,
    results: List[dict],
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))

    payload = {
        "pass@1": pass_at_1,
        "passed": passed,
        "total": total,
        "results": results,
    }
    results_json = out_dir / "results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    results_txt = out_dir / "results.txt"
    with open(results_txt, "w", encoding="utf-8") as f:
        f.write(f"{{'pass@1': {pass_at_1}}}\n")
        f.write(f"通过率: {pass_at_1:.4f} ({pass_at_1 * 100:.2f}%)\n")
        f.write(f"通过数/总数: {passed}/{total}\n")

    return results_json, results_txt


def evaluate_ldb_log(
    log_path: str,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    problem_file: Optional[str] = None,
    timeout: int = 6,
) -> Optional[dict]:
    """Grade every problem in an LDB/simple log with the official LCB evaluator."""
    log_file = Path(log_path)
    if not log_file.is_file():
        print(f"[lcb-eval] log not found: {log_file}")
        return None

    rows = _read_jsonl(log_file)
    if not rows:
        print(f"[lcb-eval] empty log: {log_file}")
        return None
    if not _is_lcb_log(rows, log_path=str(log_file)):
        print("[lcb-eval] skip (not a LiveCodeBench log)")
        return None

    samples = ldb_log_to_samples(rows)
    if not samples:
        print("[lcb-eval] no samples with task_id/solution")
        return None

    problems = problem_file or str(_MERGED_PROBLEMS)
    if not Path(problems).is_file():
        print(f"[lcb-eval] problem file not found: {problems}")
        return None

    from evaluation.evaluate_lcb import evaluate_lcb_samples  # noqa: E402

    dataset_name, model_name = infer_dataset_and_model(str(log_file), model, dataset)
    out_dir = results_dir(dataset_name, model_name)

    print(f"[lcb-eval] official re-eval of {len(samples)} samples")
    print(f"[lcb-eval] problems: {problems}")
    print(f"[lcb-eval] output:   {out_dir}")

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        prefix="ldb_lcb_samples_",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        for sample in samples:
            tmp.write(json.dumps(sample, ensure_ascii=False) + "\n")
        tmp_path = tmp.name

    try:
        pass_at_1, detailed = evaluate_lcb_samples(tmp_path, problems, timeout=timeout)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    results_json, results_txt = save_official_results(out_dir, pass_at_1, detailed)
    passed = sum(1 for r in detailed if r.get("passed"))
    total = len(detailed)
    print(f"[lcb-eval] pass@1={pass_at_1:.4f}  通过数/总数={passed}/{total}")
    print(f"[lcb-eval] wrote {results_json}")
    print(f"[lcb-eval] wrote {results_txt}")
    return {
        "pass@1": pass_at_1,
        "passed": passed,
        "total": total,
        "results_json": str(results_json),
        "results_txt": str(results_txt),
    }


def maybe_evaluate_lcb(
    log_path: str,
    model: Optional[str] = None,
    dataset_path: str = "",
) -> Optional[dict]:
    """No-op unless the finished run is LiveCodeBench."""
    if os.getenv("LDB_SKIP_INLINE_EVAL"):
        return None
    if not log_path or not Path(log_path).is_file():
        print(f"[lcb-eval] skip (log missing): {log_path}")
        return None
    rows = _read_jsonl(Path(log_path))
    if not _is_lcb_log(rows, dataset_path=dataset_path, log_path=log_path):
        return None
    return evaluate_ldb_log(log_path, model=model)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Official LCB re-eval of an LDB log")
    parser.add_argument("--log-path", required=True, help="LDB/simple output jsonl")
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", default="LiveCodeBench")
    parser.add_argument("--problems", default=None, help="Override merged problem file")
    parser.add_argument("--timeout", type=int, default=6)
    args = parser.parse_args()
    evaluate_ldb_log(
        log_path=args.log_path,
        model=args.model,
        dataset=args.dataset,
        problem_file=args.problems,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
