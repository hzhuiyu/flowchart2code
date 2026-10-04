"""Convert flowchart2code HumanEval-V / Algorithm / MATH into LDB input files.

Writes, for each dataset:
  input_data/<ldb_dir>/dataset/probs.jsonl
  input_data/<ldb_dir>/test/tests.jsonl

tests.jsonl is one {"task_id", "given_tests"} object per line (same as LCB).

Usage:
  python f2c_convert.py --all
  python f2c_convert.py --dataset HumanEval-V
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_LDB_ROOT = _THIS_DIR.parent
_F2C_ROOT = _LDB_ROOT.parent

MAX_GIVEN_TESTS = 8

DATASETS = {
    "HumanEval-V": {
        "ldb_dir": "humaneval-v",
        "src": _F2C_ROOT / "data" / "HumanEval-V" / "HumanEval.jsonl",
        "prefix": "HumanEval-V",
    },
    "Algorithm": {
        "ldb_dir": "algorithm",
        "src": _F2C_ROOT / "data" / "Algorithm" / "Algorithm.jsonl",
        "prefix": "Algorithm",
    },
    "MATH": {
        "ldb_dir": "math",
        "src": _F2C_ROOT / "data" / "MATH" / "MATH.jsonl",
        "prefix": "MATH",
    },
}


def read_jsonl(path: Path) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_entry_point(starter_code: str) -> str:
    match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", starter_code or "")
    return match.group(1) if match else ""


def extract_doctest_asserts(prompt: str) -> List[str]:
    """Turn `>>> call` / result pairs in a HumanEval-style docstring into asserts."""
    tests: List[str] = []
    lines = (prompt or "").splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith(">>>"):
            call = stripped[3:].strip()
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i >= len(lines):
                break
            result = lines[i].strip()
            if (
                call
                and result
                and not result.startswith(">>>")
                and not result.startswith('"""')
                and not result.startswith("'''")
            ):
                tests.append(f"assert {call} == {result}")
                i += 1
                continue
        i += 1
    return tests


def extract_assert_lines(test: str, limit: int = MAX_GIVEN_TESTS) -> List[str]:
    tests: List[str] = []
    for line in (test or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("assert "):
            tests.append(stripped)
            if len(tests) >= limit:
                break
    return tests


def _parse_literal(text: str):
    return ast.literal_eval(text)


def extract_algorithm_given_tests(test: str, method: str, limit: int = MAX_GIVEN_TESTS) -> List[str]:
    """Rewrite hidden-test pairs into standalone `assert Solution().method(**{...}) == expected`."""
    tests: List[str] = []
    lines = (test or "").splitlines()
    i = 0
    n = len(lines)
    while i < n and len(tests) < limit:
        stripped = lines[i].strip()
        if stripped.startswith("test_input"):
            buf = stripped
            j = i
            payload = None
            while j < n:
                try:
                    rhs = buf.split("=", 1)[1].strip()
                    payload = _parse_literal(rhs)
                    break
                except Exception:
                    j += 1
                    if j >= n:
                        break
                    buf += " " + lines[j].strip()
            if payload is None:
                i += 1
                continue
            k = j + 1
            while k < n and not lines[k].strip().startswith("assert "):
                k += 1
            if k < n:
                assert_line = lines[k].strip()
                match = re.search(r"==\s*(.+)$", assert_line)
                expected = match.group(1).strip() if match else "True"
                if "  #" in expected:
                    expected = expected.split("  #", 1)[0].strip()
                tests.append(f"assert Solution().{method}(**{repr(payload)}) == {expected}")
                i = k + 1
                continue
        i += 1

    if tests:
        return tests

    # Fallback: `assert my_solution.method(...)` -> `assert Solution().method(...)`
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("assert my_solution."):
            tests.append(stripped.replace("assert my_solution.", "assert Solution().", 1))
            if len(tests) >= limit:
                break
    return tests


def _math_prompt(prompt: str, starter_code: str) -> str:
    starter = (starter_code or "").rstrip()
    instruction = (
        "Write a Python solution. Implement the function with EXACTLY the signature above.\n"
        "Return the final answer. Do NOT add extra top-level test code or calls.\n"
    )
    body = (prompt or "").strip()
    if starter:
        return f"{body}\n\n```python\n{starter}\n```\n\n{instruction}"
    return f"{body}\n\n{instruction}"


def convert_humaneval_v(rows: List[dict], prefix: str) -> Tuple[List[dict], List[dict]]:
    problems, tests = [], []
    for item in rows:
        orig_id = item["task_id"]
        task_id = f"{prefix}/{orig_id}"
        given = extract_doctest_asserts(item.get("prompt", ""))
        record = {
            "task_id": task_id,
            "prompt": item.get("prompt", ""),
            "entry_point": item.get("entry_point") or extract_entry_point(item.get("starter_code", "")),
            "test": item.get("test", ""),
            "canonical_solution": item.get("canonical_solution", ""),
            "given_tests": given,
        }
        problems.append(record)
        tests.append({"task_id": task_id, "given_tests": given})
    return problems, tests


def convert_algorithm(rows: List[dict], prefix: str) -> Tuple[List[dict], List[dict]]:
    problems, tests = [], []
    for item in rows:
        orig_id = item["task_id"]
        task_id = f"{prefix}/{orig_id}"
        method = extract_entry_point(item.get("starter_code", "") or item.get("prompt", ""))
        given = extract_algorithm_given_tests(item.get("test", ""), method)
        record = {
            "task_id": task_id,
            "prompt": item.get("prompt", ""),
            "entry_point": method,
            "test": item.get("test", ""),
            "canonical_solution": item.get("canonical_solution", ""),
            "given_tests": given,
        }
        problems.append(record)
        tests.append({"task_id": task_id, "given_tests": given})
    return problems, tests


def convert_math(rows: List[dict], prefix: str) -> Tuple[List[dict], List[dict]]:
    problems, tests = [], []
    for item in rows:
        orig_id = item["task_id"]
        task_id = f"{prefix}/{orig_id}"
        func_name = extract_entry_point(item.get("starter_code", ""))
        given = extract_assert_lines(item.get("test", ""))
        record = {
            "task_id": task_id,
            "prompt": _math_prompt(item.get("prompt", ""), item.get("starter_code", "")),
            "entry_point": func_name,
            "test": item.get("test", ""),
            "canonical_solution": item.get("canonical_solution", ""),
            "given_tests": given,
            "difficulty": item.get("difficulty", ""),
        }
        problems.append(record)
        tests.append({"task_id": task_id, "given_tests": given})
    return problems, tests


_CONVERT = {
    "HumanEval-V": convert_humaneval_v,
    "Algorithm": convert_algorithm,
    "MATH": convert_math,
}


def output_paths(ldb_dir: str) -> Tuple[Path, Path]:
    base = _LDB_ROOT / "input_data" / ldb_dir
    return base / "dataset" / "probs.jsonl", base / "test" / "tests.jsonl"


def already_converted(probs_path: Path, tests_path: Path, expected: int) -> bool:
    if expected <= 0 or not probs_path.is_file() or not tests_path.is_file():
        return False
    return _count_jsonl(probs_path) == expected and _count_jsonl(tests_path) == expected


def _count_jsonl(path: Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def convert_one(name: str, force: bool = False) -> Tuple[Path, Path, int]:
    spec = DATASETS[name]
    src = spec["src"]
    if not src.is_file():
        raise FileNotFoundError(f"source jsonl missing: {src}")
    rows = read_jsonl(src)
    probs_path, tests_path = output_paths(spec["ldb_dir"])
    if not force and already_converted(probs_path, tests_path, len(rows)):
        print(f"[convert] skip {name}: {len(rows)} problems already at {probs_path}")
        return probs_path, tests_path, len(rows)

    problems, tests = _CONVERT[name](rows, spec["prefix"])
    write_jsonl(probs_path, problems)
    write_jsonl(tests_path, tests)
    n_with_tests = sum(1 for t in tests if t.get("given_tests"))
    print(f"[convert] {name}: wrote {len(problems)} problems -> {probs_path}")
    print(f"[convert] {name}: wrote {len(tests)} tests ({n_with_tests} with given_tests) -> {tests_path}")
    return probs_path, tests_path, len(problems)


def convert_all(force: bool = False, names: Optional[List[str]] = None) -> None:
    for name in names or list(DATASETS):
        convert_one(name, force=force)


def main():
    parser = argparse.ArgumentParser(description="Convert flowchart2code datasets for LDB")
    parser.add_argument("--all", action="store_true", help="Convert HumanEval-V, Algorithm, MATH")
    parser.add_argument("--dataset", choices=list(DATASETS), help="Convert one dataset")
    parser.add_argument("--force", action="store_true", help="Overwrite existing LDB input files")
    args = parser.parse_args()
    if args.dataset:
        convert_one(args.dataset, force=args.force)
    else:
        convert_all(force=args.force)


if __name__ == "__main__":
    main()
