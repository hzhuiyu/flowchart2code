"""Sequential LDB runner for flowchart2code's four datasets.

Order: HumanEval-V -> Algorithm -> MATH -> LiveCodeBench.

For each dataset:
  1. Convert to LDB input (if needed; LCB reuses existing conversion)
  2. Skip if this run's results/<dataset>/<model>/{results.json,results.txt}
     exist AND the max_iters=3 ldb log already has every problem
  3. simple seed (resumable) then ldb --max_iters 3 (resumable)
  4. Official flowchart2code re-eval, then the next dataset

Results:
  LLMDebugger/results/<dataset>/<model>/results.json
  LLMDebugger/results/<dataset>/<model>/results.txt

  With --use-image, last-level folder is <model>-image (flowchart attached at
  seed + first ldb iteration). Text-only dumps under <model>/ are untouched.

Usage:
  LDB_CONFIG_PATH=../../src/configs/gpt_api_key_config.json python run_ldb_f2c_datasets.py
  python run_ldb_f2c_datasets.py --config ../../src/configs/gpt_api_key_config.json --use-image
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

_THIS_DIR = Path(__file__).resolve().parent
_LDB_ROOT = _THIS_DIR.parent
_F2C_ROOT = _LDB_ROOT.parent
_SRC_ROOT = _F2C_ROOT / "src"
_HUMAN_EVAL = _SRC_ROOT / "human-eval"

if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from f2c_convert import convert_one  # noqa: E402
from lcb_eval_results import ldb_log_to_samples, save_official_results, to_f2c_task_id  # noqa: E402

IMPORT_HEADER = (
    "from typing import *\nimport math\nfrom heapq import *\nimport itertools\n"
    "import re\nimport typing\nimport heapq\n_str=str\nimport re\n"
    "from sortedcontainers import SortedList, SortedDict, SortedSet\n"
    "import sortedcontainers\n"
)

MAX_ITERS = 3
PASS_AT_K = 1
RUN_NAME = "max3"
CLI_MODEL = "config"

DATASETS = [
    {
        "name": "HumanEval-V",
        "ldb_dir": "humaneval-v",
        "kind": "humaneval",
        "problem_file": _F2C_ROOT / "data" / "HumanEval-V" / "HumanEval.jsonl",
        "needs_convert": True,
    },
    {
        "name": "Algorithm",
        "ldb_dir": "algorithm",
        "kind": "humaneval",
        "problem_file": _F2C_ROOT / "data" / "Algorithm" / "Algorithm.jsonl",
        "needs_convert": True,
    },
    {
        "name": "MATH",
        "ldb_dir": "math",
        "kind": "humaneval",
        "problem_file": _F2C_ROOT / "data" / "MATH" / "MATH.jsonl",
        "needs_convert": True,
    },
    {
        "name": "LiveCodeBench",
        "ldb_dir": "livecodebench",
        "kind": "lcb",
        "problem_file": _F2C_ROOT / "data" / "LiveCodeBench" / "LiveCodeBench_merged.jsonl",
        "needs_convert": False,
    },
]


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid config: {path}")
    return cfg


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _log_path(root_dir: Path, run_name: str, dataset_path: Path, strategy: str,
              max_iters: int, model: str, pass_at_k: int, seedfile: str) -> Path:
    dataset_name = os.path.basename(str(dataset_path)).replace("jsonl", "")
    log_dir = root_dir / run_name
    seed_name = os.path.basename(seedfile).replace("jsonl", "") if seedfile else ""
    return log_dir / (
        f"{dataset_name}_{strategy}_{max_iters}_{model}_pass_at_{pass_at_k}_seed_{seed_name}.jsonl"
    )


def _results_dir(dataset: str, model: str) -> Path:
    return _LDB_ROOT / "results" / dataset / model


def _progress_path(model: str) -> Path:
    return _LDB_ROOT / "output_data" / "progress" / f"{model}_ldb_f2c.json"


def _load_progress(model: str) -> dict:
    path = _progress_path(model)
    if not path.is_file():
        return {"model": model, "max_iters": MAX_ITERS, "datasets": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("datasets", {})
            return data
    except Exception:
        pass
    return {"model": model, "max_iters": MAX_ITERS, "datasets": {}}


def _save_progress(model: str, progress: dict) -> None:
    path = _progress_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def _strip_humaneval_v_completion(solution: str, prompt: str) -> str:
    text = solution or ""
    if text.startswith(IMPORT_HEADER):
        text = text[len(IMPORT_HEADER):]
    text = text.lstrip("\n")
    if prompt and text.startswith(prompt):
        text = text[len(prompt):]
        if text.startswith("\n"):
            text = text[1:]
        return text
    return solution or ""


def _samples_from_log(dataset_name: str, log_path: Path, problem_file: Path) -> List[dict]:
    rows = _read_jsonl(log_path)
    if dataset_name != "HumanEval-V":
        return ldb_log_to_samples(rows)

    prompts = {}
    if problem_file.is_file():
        for item in _read_jsonl(problem_file):
            prompts[item.get("task_id", "")] = item.get("prompt", "")

    samples = []
    seen = set()
    for row in rows:
        task_id = to_f2c_task_id(row.get("task_id", ""))
        if not task_id:
            continue
        prompt = prompts.get(task_id) or row.get("prompt") or ""
        completion = _strip_humaneval_v_completion(
            row.get("solution") or row.get("completion") or "",
            prompt,
        )
        if task_id in seen:
            samples = [s for s in samples if s["task_id"] != task_id]
        seen.add(task_id)
        samples.append({"task_id": task_id, "completion": completion})
    return samples


def _eval_humaneval(samples_path: Path, problem_file: Path) -> tuple:
    if str(_HUMAN_EVAL) not in sys.path:
        sys.path.insert(0, str(_HUMAN_EVAL))
    from human_eval.evaluation import evaluate_functional_correctness  # noqa: E402

    pass_at_k = evaluate_functional_correctness(
        sample_file=str(samples_path),
        k=[1],
        n_workers=4,
        timeout=3.0,
        problem_file=str(problem_file),
    )
    pass_at_1 = float(pass_at_k.get("pass@1", 0.0))
    detailed = []
    results_file = Path(str(samples_path) + "_results.jsonl")
    if results_file.is_file():
        for row in _read_jsonl(results_file):
            detailed.append({
                "task_id": row.get("task_id"),
                "passed": bool(row.get("passed")),
                "test_results": row.get("result"),
                "metadata": {},
                "difficulty": row.get("difficulty"),
            })
    return pass_at_1, detailed


def _eval_lcb(samples_path: Path, problem_file: Path) -> tuple:
    from evaluation.evaluate_lcb import evaluate_lcb_samples  # noqa: E402
    return evaluate_lcb_samples(str(samples_path), str(problem_file), timeout=6)


def official_eval(spec: dict, log_path: Path, out_dir: Path) -> dict:
    samples = _samples_from_log(spec["name"], log_path, spec["problem_file"])
    if not samples:
        raise RuntimeError(f"no samples in {log_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.jsonl"
    _write_jsonl(samples_path, samples)

    print(f"[eval] {spec['name']}: official re-eval of {len(samples)} samples")
    if spec["kind"] == "lcb":
        pass_at_1, detailed = _eval_lcb(samples_path, spec["problem_file"])
    else:
        pass_at_1, detailed = _eval_humaneval(samples_path, spec["problem_file"])

    results_json, results_txt = save_official_results(out_dir, float(pass_at_1), detailed)
    passed = sum(1 for r in detailed if r.get("passed"))
    total = len(detailed)
    print(f"[eval] {spec['name']}: pass@1={pass_at_1:.4f}  通过数/总数={passed}/{total}")
    print(f"[eval] wrote {results_json}")
    print(f"[eval] wrote {results_txt}")
    return {
        "pass@1": float(pass_at_1),
        "passed": passed,
        "total": total,
        "results_json": str(results_json),
        "results_txt": str(results_txt),
    }


def _run_main(env: dict, args: List[str]) -> None:
    cmd = [sys.executable, "-u", str(_THIS_DIR / "main.py"), *args]
    print("[run]", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(_THIS_DIR), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"main.py exited with {result.returncode}")


def _dataset_complete(spec: dict, model_folder: str, n_problems: int, log_path: Path) -> bool:
    out_dir = _results_dir(spec["name"], model_folder)
    if not (out_dir / "results.json").is_file() or not (out_dir / "results.txt").is_file():
        return False
    return _count_jsonl(log_path) >= n_problems


def run_dataset(spec: dict, env: dict, model_folder: str, progress: dict) -> None:
    name = spec["name"]
    ldb_dir = spec["ldb_dir"]
    probs = _LDB_ROOT / "input_data" / ldb_dir / "dataset" / "probs.jsonl"
    tests = _LDB_ROOT / "input_data" / ldb_dir / "test" / "tests.jsonl"

    if spec["needs_convert"]:
        convert_one(name)
    elif not probs.is_file() or not tests.is_file():
        raise FileNotFoundError(
            f"{name} LDB input missing: {probs} / {tests}. Run lcb_convert.py first."
        )

    n_problems = _count_jsonl(probs)
    if n_problems == 0:
        raise RuntimeError(f"{name}: empty {probs}")

    simple_root = _LDB_ROOT / "output_data" / "simple" / ldb_dir / model_folder
    ldb_root = _LDB_ROOT / "output_data" / "ldb" / ldb_dir / model_folder
    simple_log = _log_path(simple_root, RUN_NAME, probs, "simple", MAX_ITERS, CLI_MODEL, PASS_AT_K, "")
    ldb_log = _log_path(
        ldb_root, RUN_NAME, probs, "ldb", MAX_ITERS, CLI_MODEL, PASS_AT_K, str(simple_log)
    )
    out_dir = _results_dir(name, model_folder)

    ds_progress = progress.setdefault("datasets", {}).setdefault(name, {})
    ds_progress["n_problems"] = n_problems
    ds_progress["simple_log"] = str(simple_log)
    ds_progress["ldb_log"] = str(ldb_log)
    ds_progress["results_dir"] = str(out_dir)

    if _dataset_complete(spec, model_folder, n_problems, ldb_log):
        print(f"[skip] {name}: 已完成 ({out_dir})")
        ds_progress["status"] = "skipped"
        _save_progress(model_folder, progress)
        return

    print(f"\n======== {name} ({n_problems} problems, max_iters={MAX_ITERS}) ========")

    if _count_jsonl(simple_log) < n_problems:
        ds_progress["status"] = "simple"
        _save_progress(model_folder, progress)
        print(f"[simple] {name}: {_count_jsonl(simple_log)}/{n_problems} -> resume/start")
        _run_main(env, [
            "--run_name", RUN_NAME,
            "--root_dir", str(simple_root),
            "--dataset_path", str(probs),
            "--strategy", "simple",
            "--model", CLI_MODEL,
            "--n_proc", "1",
            "--pass_at_k", str(PASS_AT_K),
            "--max_iters", str(MAX_ITERS),
            "--testfile", str(tests),
            "--verbose",
            "--port", "8000",
        ])
        if _count_jsonl(simple_log) < n_problems:
            raise RuntimeError(
                f"{name} simple 未跑完: {_count_jsonl(simple_log)}/{n_problems} ({simple_log})"
            )
    else:
        print(f"[simple] {name}: seed 已齐 ({n_problems}), 跳过")

    if _count_jsonl(ldb_log) < n_problems:
        ds_progress["status"] = "ldb"
        _save_progress(model_folder, progress)
        print(f"[ldb] {name}: {_count_jsonl(ldb_log)}/{n_problems} -> resume/start")
        _run_main(env, [
            "--run_name", RUN_NAME,
            "--root_dir", str(ldb_root),
            "--dataset_path", str(probs),
            "--strategy", "ldb",
            "--model", CLI_MODEL,
            "--n_proc", "1",
            "--pass_at_k", str(PASS_AT_K),
            "--max_iters", str(MAX_ITERS),
            "--seedfile", str(simple_log),
            "--testfile", str(tests),
            "--verbose",
            "--port", "8000",
        ])
        if _count_jsonl(ldb_log) < n_problems:
            raise RuntimeError(
                f"{name} ldb 未跑完: {_count_jsonl(ldb_log)}/{n_problems} ({ldb_log})"
            )
    else:
        print(f"[ldb] {name}: log 已齐 ({n_problems}), 跳过生成")

    ds_progress["status"] = "eval"
    _save_progress(model_folder, progress)
    summary = official_eval(spec, ldb_log, out_dir)
    ds_progress["status"] = "completed"
    ds_progress["pass@1"] = summary["pass@1"]
    ds_progress["passed"] = summary["passed"]
    ds_progress["total"] = summary["total"]
    _save_progress(model_folder, progress)


def main():
    parser = argparse.ArgumentParser(description="Sequential LDB on flowchart2code datasets")
    parser.add_argument(
        "--config",
        default=os.getenv("LDB_CONFIG_PATH", str(_SRC_ROOT / "configs" / "gpt_api_key_config.json")),
        help="flowchart2code API config (model/api_key/base_url)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Subset of: HumanEval-V Algorithm MATH LiveCodeBench",
    )
    parser.add_argument(
        "--use-image",
        action="store_true",
        help=(
            "Attach flowchart images at seed generation and the first LDB iteration. "
            "Results/logs go under <model>-image so text-only dumps are not overwritten."
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    cfg = _load_config(config_path)
    model_folder = str(cfg.get("model") or "gpt-4o-mini").replace("/", "-")

    env = os.environ.copy()
    env["LDB_CONFIG_PATH"] = str(config_path)
    env["LDB_SKIP_INLINE_EVAL"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if args.use_image:
        env["LDB_USE_IMAGE"] = "1"
        model_folder = f"{model_folder}-image"
        if cfg.get("timeout_sec") and not env.get("LDB_API_TIMEOUT_SEC"):
            env["LDB_API_TIMEOUT_SEC"] = str(cfg["timeout_sec"])

    wanted = set(args.datasets) if args.datasets else None
    if wanted:
        known = {d["name"] for d in DATASETS}
        unknown = sorted(wanted - known)
        if unknown:
            raise ValueError(f"unknown datasets: {unknown}")

    progress = _load_progress(model_folder)
    progress["model"] = model_folder
    progress["max_iters"] = MAX_ITERS
    progress["config"] = str(config_path)
    _save_progress(model_folder, progress)

    print(f"model={model_folder}  max_iters={MAX_ITERS}  config={config_path}")
    print(f"use_image={bool(args.use_image)}  LDB_USE_IMAGE={env.get('LDB_USE_IMAGE', '')}")
    print(f"results -> {_LDB_ROOT / 'results' / '<dataset>' / model_folder}")

    for spec in DATASETS:
        if wanted and spec["name"] not in wanted:
            continue
        run_dataset(spec, env, model_folder, progress)

    print("\n全部数据集处理完毕。")
    for spec in DATASETS:
        if wanted and spec["name"] not in wanted:
            continue
        info = progress.get("datasets", {}).get(spec["name"], {})
        status = info.get("status", "?")
        extra = ""
        if "passed" in info and "total" in info:
            extra = f"  {info['passed']}/{info['total']}  pass@1={info.get('pass@1')}"
        print(f"  {spec['name']}: {status}{extra}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
