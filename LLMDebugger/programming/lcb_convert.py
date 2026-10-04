"""Convert flowchart2code's LiveCodeBench data into LDB input files.

flowchart2code stores LCB as two views of the same 148 problems:
  - data/LiveCodeBench/LiveCodeBench.jsonl        HumanEval-style records (task_id `v1_*`, prompt, starter_code)
  - data/LiveCodeBench/LiveCodeBench_merged.jsonl official metadata (question_content, func_name, public_test_cases, reference_code)

76 problems are converted stdio tasks (`def solve(arg0, ...)`, tests rewritten as
`solve('DD??S')` call expressions) and 72 are call-based (`class Solution`,
tests stored as raw stdin lines that json-decode into typed args).

This script emits LDB files:
  input_data/livecodebench/dataset/probs.jsonl   task_id/prompt/entry_point/test(canonical LDB record)
  input_data/livecodebench/test/tests.jsonl      per-task given_tests (assert-based unit tests)
and can also convert a flowchart2code samples.jsonl into an LDB seed file.

Usage:
  python lcb_convert.py --f2c-root ../.. --out-dir ../input_data/livecodebench
  python lcb_convert.py --f2c-root ../.. --from-f2c-samples ../../output/LiveCodeBench/gpt-5-nano_text_image/samples.jsonl --seed-out ../input_data/livecodebench/seed/gpt-5-nano/seed.jsonl
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# flowchart2code/src (three levels above programming/) for the canonical evaluator.
_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.append(str(_SRC_ROOT))

from evaluation.evaluate_lcb import (  # noqa: E402
    _decode_lcb_call_input,
    _parse_expected_value,
    build_lcb_input_output,
    import_string,
)

DATASET_TYPE = "LiveCodeBench"


def read_jsonl(path: Path):
    items = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def detect_mode(starter_code: str) -> str:
    starter = (starter_code or "").lstrip()
    if starter.startswith("def solve(") and "class Solution" not in starter:
        return "stdio"
    return "call_based"


def _run_subprocess(code: str, stdin_text: str = "", timeout: int = 10):
    return subprocess.run(
        [sys.executable, "-c", code],
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _decode_args(raw_input: str):
    """Decode one test input into typed positional args (official semantics)."""
    args = []
    for line in str(raw_input).split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            args.append(json.loads(line))
        except (ValueError, TypeError):
            return None
    return args


def _build_problem_prompt(mode: str, question_content: str, starter_code: str) -> str:
    if mode == "stdio":
        instruction = """Write a Python solution. Implement the `solve` function with EXACTLY the signature above.

- Each parameter arg0, arg1, ... receives one already-parsed input value; the type annotation tells you its type.
- Return the final answer from solve (printing it is also accepted).
- Do NOT call solve() yourself and do not add a standalone input loop.
- Reading stdin inside solve (input()/sys.stdin) is also allowed; in that case ignore the parameters.
"""
    else:
        instruction = """Write a Python solution. Implement the method inside `class Solution` with EXACTLY the signature above.

- Return the final answer from the method (printing it is also accepted).
- Do NOT add extra top-level test code or calls.
"""
    return f"{question_content.strip()}\n\n```python\n{starter_code.rstrip()}\n```\n\n{instruction}"


def _build_assert(mode: str, entry_point: str, args, expected) -> str:
    if mode == "stdio":
        call = f"{entry_point}(" + ", ".join(repr(a) for a in args) + ")"
    else:
        call = f"Solution().{entry_point}(" + ", ".join(repr(a) for a in args) + ")"
    return f"assert {call} == {repr(expected)}"


def _unescape_reference(code: str) -> str:
    """Some merged reference_code entries are double-escaped (`float(\\'inf\\')`)."""
    if "\\'" in code or '\\"' in code:
        try:
            compile(code, "<ref>", "exec")
            return code
        except SyntaxError:
            return code.replace("\\'", "'").replace('\\"', '"')
    return code


def _reference_compiles(reference_code: str) -> bool:
    try:
        compile(reference_code, "<ref>", "exec")
        return True
    except SyntaxError:
        return False


# Extra imports commonly used by LCB reference solutions but not covered by
# evaluate_lcb.import_string; without them validation wrongly drops asserts.
_EXTRA_IMPORTS = (
    "from sortedcontainers import SortedList, SortedDict, SortedSet\n"
    "import sortedcontainers\n"
    "import numpy\n"
    "import string\n"
    "import re\n"
    "import heapq\n"
    "import bisect\n"
    "import collections\n"
    "import itertools\n"
    "import math\n"
    "import sys\n"
)


def _validation_prelude(reference_code: str) -> str:
    """Imports + reference code, retrying with extra imports on NameError."""
    prelude = import_string + "\n\n" + reference_code
    probe = prelude + '\nprint("LDB_REF_OK")\n'
    try:
        result = _run_subprocess(probe, timeout=15)
    except subprocess.TimeoutExpired:
        return prelude
    if result.returncode == 0 and "LDB_REF_OK" in result.stdout:
        return prelude
    patched = import_string + "\n" + _EXTRA_IMPORTS + "\n" + reference_code
    probe = patched + '\nprint("LDB_REF_OK")\n'
    try:
        result = _run_subprocess(probe, timeout=15)
    except subprocess.TimeoutExpired:
        return prelude
    if result.returncode == 0 and "LDB_REF_OK" in result.stdout:
        return patched
    return prelude


def _validate_call_based_asserts(reference_code: str, asserts, timeout: int = 10):
    """Keep asserts that pass against the reference solution.

    Returns (kept_asserts, notes).  A broken reference (fragments, double
    escapes that survive unescaping, timeouts) keeps the asserts unvalidated
    instead of dropping them: argument decoding follows official semantics.
    """
    if not reference_code.strip():
        return asserts, ["no reference code; asserts unvalidated"]
    if not _reference_compiles(reference_code):
        return asserts, ["reference does not compile; asserts unvalidated"]
    prelude = _validation_prelude(reference_code)
    probe = prelude + '\nprint("LDB_REF_OK")\n'
    try:
        result = _run_subprocess(probe, timeout=timeout)
    except subprocess.TimeoutExpired:
        return asserts, ["reference timeout; asserts unvalidated"]
    if result.returncode != 0 or "LDB_REF_OK" not in result.stdout:
        return asserts, ["reference crashes on import; asserts unvalidated"]

    script = prelude + "\n\n" + "\n".join(asserts) + '\nprint("LDB_VALIDATE_OK")\n'
    try:
        result = _run_subprocess(script, timeout=timeout)
    except subprocess.TimeoutExpired:
        return [], ["reference timeout during asserts"]
    if result.returncode == 0 and "LDB_VALIDATE_OK" in result.stdout:
        return asserts, []
    kept = []
    for line in asserts:
        one = prelude + "\n\n" + line + '\nprint("LDB_VALIDATE_OK")\n'
        try:
            r = _run_subprocess(one, timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
        if r.returncode == 0 and "LDB_VALIDATE_OK" in r.stdout:
            kept.append(line)
    return kept, [f"kept {len(kept)}/{len(asserts)} after per-assert check"]


def _validate_stdio_case(reference_code: str, stdin_text: str, expected_text: str, timeout: int = 10):
    """Check that the reference program really maps stdin_text -> expected_text."""
    if not reference_code.strip():
        return False
    try:
        result = _run_subprocess(reference_code, stdin_text=stdin_text, timeout=timeout)
        if result.returncode != 0:
            # Reference may rely on imports beyond the stdlib set; retry with
            # the common competition imports prepended.
            patched = _EXTRA_IMPORTS + "\n" + reference_code
            result = _run_subprocess(patched, stdin_text=stdin_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    if result.returncode != 0:
        return False
    # Converted stdio `output` fields hold the repr of solve()'s return value
    # (`'BCAC'`); compare against the parsed value, like the official grader.
    return result.stdout.strip() == str(expected_text).strip()


def build_record(problem_f2c: dict, problem_merged: dict, validate: bool = True):
    """Build one LDB probs.jsonl record plus its given_tests."""
    task_id = problem_f2c.get("task_id") or f"v1_{problem_merged.get('question_id', '')}"
    question_content = problem_merged.get("question_content") or problem_f2c.get("prompt", "")
    starter_code = problem_merged.get("starter_code") or problem_f2c.get("starter_code", "")
    func_name = problem_merged.get("func_name") or ""
    cases = problem_merged.get("public_test_cases") or []
    if isinstance(cases, str):
        try:
            cases = json.loads(cases)
        except (ValueError, TypeError):
            cases = []
    reference_code = _unescape_reference(problem_merged.get("reference_code") or "")

    mode = detect_mode(starter_code)
    if not func_name:
        func_name = "solve" if mode == "stdio" else ""

    # Canonical input_output structure consumed by LCBPyExecutor.evaluate.
    problem_info = {
        "lcb_mode": mode,
        "func_name": func_name,
        "entry_point": func_name,
        "starter_code": starter_code,
        "public_test_cases": cases,
    }
    io_structure = build_lcb_input_output(problem_info, cases)

    # Collect assert candidates first; validation may then prune them.
    candidates = []  # (assert_line, stdin_text, raw_output)
    notes = []
    for case in cases:
        if not isinstance(case, dict) or str(case.get("testtype", "functional")).lower() != "functional":
            continue
        raw_input = case.get("input", "")
        raw_output = case.get("output", "")
        expected = _parse_expected_value(raw_output)
        args = None
        stdin_text = str(raw_input)
        if mode == "stdio":
            decoded_args, stdin_text = _decode_lcb_call_input(raw_input, func_name)
            args = decoded_args
        else:
            args = _decode_args(raw_input)
        if args is None:
            notes.append("skip: undecodable input")
            continue
        candidates.append((_build_assert(mode, func_name, args, expected), stdin_text, raw_output))

    given_tests = []
    if mode == "call_based":
        if validate:
            kept, info = _validate_call_based_asserts(reference_code, [c[0] for c in candidates])
            notes.extend(info)
            if not kept and candidates:
                # Broken reference (e.g. C++-isms in the Python reference) —
                # keep the decoded asserts unvalidated instead of dropping
                # them, so the problem still participates in unit-test
                # feedback (final grading always uses the official tests).
                notes.append("reference unusable; keeping asserts unvalidated")
                kept_set = {c[0] for c in candidates}
            else:
                kept_set = set(kept)
        else:
            kept_set = {c[0] for c in candidates}
        for line, _stdin, _out in candidates:
            if line in kept_set:
                given_tests.append(line)
    else:
        ref_usable = _reference_compiles(reference_code)
        ref_working = None
        if validate and ref_usable and candidates:
            # Probe once: a broken reference (e.g. `mp.upper_bound`) fails
            # every case; then keep the asserts unvalidated instead.
            c0 = candidates[0]
            ref_working = _validate_stdio_case(reference_code, c0[1], str(_parse_expected_value(c0[2])))
            if not ref_working:
                notes.append("reference unusable; keeping asserts unvalidated")
        for line, stdin_text, raw_output in candidates:
            if validate and ref_usable and ref_working:
                if not _validate_stdio_case(reference_code, stdin_text, str(_parse_expected_value(raw_output))):
                    notes.append("skip: reference mismatch on stdio case")
                    continue
            given_tests.append(line)

    # Keep order, drop exact duplicates.
    seen = set()
    ordered = []
    for line in given_tests:
        if line not in seen:
            seen.add(line)
            ordered.append(line)
    given_tests = ordered

    record = {
        "task_id": f"{DATASET_TYPE}/{task_id}",
        "prompt": _build_problem_prompt(mode, question_content, starter_code),
        "entry_point": func_name,
        # JSON string of the canonical input_output structure; LCBPyExecutor
        # feeds it to the official mixed-I/O run_test.
        "test": json.dumps(io_structure, ensure_ascii=False),
        "canonical_solution": reference_code,
        "given_tests": given_tests,
        "lcb_mode": mode,
    }
    return record, given_tests, notes


def _normalize_lcb_id(value: str) -> str:
    """Strip the release prefix (`v1_2757` -> `2757`, as problem_extractor does)."""
    return re.sub(r"^v\d+_", "", str(value or ""))


def convert_dataset(f2c_root: Path, out_dir: Path, limit=None, validate=True):
    f2c_file = f2c_root / "data" / "LiveCodeBench" / "LiveCodeBench.jsonl"
    merged_file = f2c_root / "data" / "LiveCodeBench" / "LiveCodeBench_merged.jsonl"

    f2c_rows = {row["task_id"]: row for row in read_jsonl(f2c_file)}
    merged_rows = {row.get("question_id"): row for row in read_jsonl(merged_file)}
    # Match `v1_2757`/`v6_abc399_f` -> `2757`/`abc399_f` (release prefix stripped).
    merged_by_f2c_id = {_normalize_lcb_id(qid): row for qid, row in merged_rows.items() if qid}

    dataset_dir = out_dir / "dataset"
    test_dir = out_dir / "test"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    stats = {"stdio": 0, "call_based": 0, "no_tests": 0}
    problems, tests = [], []
    for index, (task_id, f2c_row) in enumerate(sorted(f2c_rows.items())):
        if limit is not None and index >= limit:
            break
        merged_row = merged_by_f2c_id.get(_normalize_lcb_id(task_id))
        if merged_row is None:
            print(f"[warn] no merged metadata for {task_id}, skipping")
            continue
        record, given_tests, notes = build_record(f2c_row, merged_row, validate=validate)
        problems.append(record)
        tests.append({"task_id": record["task_id"], "given_tests": given_tests})
        stats[record["lcb_mode"]] += 1
        if not given_tests:
            stats["no_tests"] += 1
        flag = " | ".join(notes[:2])
        print(f"[{index + 1}/{len(f2c_rows)}] {task_id} mode={record['lcb_mode']} "
              f"tests={len(given_tests)} {flag}")

    probs_path = dataset_dir / "probs.jsonl"
    tests_path = test_dir / "tests.jsonl"
    with open(probs_path, "w", encoding="utf-8", newline="\n") as f:
        for record in problems:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(tests_path, "w", encoding="utf-8", newline="\n") as f:
        for item in tests:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(problems)} problems to {probs_path}")
    print(f"Wrote {len(tests)} test entries to {tests_path}")
    print(f"Stats: {stats}")


def convert_f2c_samples_to_seed(samples_file: Path, seed_out: Path):
    """Turn a flowchart2code samples.jsonl (task_id `v1_*`, completion) into an LDB seed file."""
    rows = read_jsonl(samples_file)
    seed_out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(seed_out, "w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            task_id = row.get("task_id", "")
            solution = row.get("completion") or row.get("solution") or ""
            if not task_id or not str(solution).strip():
                continue
            out.write(json.dumps(
                {"task_id": f"{DATASET_TYPE}/{task_id}", "solution": solution},
                ensure_ascii=False,
            ) + "\n")
            written += 1
    print(f"Wrote {written} seeds to {seed_out}")


def main():
    parser = argparse.ArgumentParser(description="Convert flowchart2code LCB data for LDB")
    parser.add_argument("--f2c-root", default="../..", help="flowchart2code repo root")
    parser.add_argument("--out-dir", default="../input_data/livecodebench", help="LDB input_data output dir")
    parser.add_argument("--limit", type=int, default=None, help="Only convert the first N problems")
    parser.add_argument("--no-validate", action="store_true", help="Skip reference-solution validation of asserts")
    parser.add_argument("--from-f2c-samples", default=None, help="flowchart2code samples.jsonl to convert into a seed file")
    parser.add_argument("--seed-out", default=None, help="Output path for the seed file")
    args = parser.parse_args()

    f2c_root = Path(args.f2c_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    if args.from_f2c_samples:
        if not args.seed_out:
            parser.error("--seed-out is required with --from-f2c-samples")
        convert_f2c_samples_to_seed(Path(args.from_f2c_samples), Path(args.seed_out))
        return

    convert_dataset(
        f2c_root=f2c_root,
        out_dir=out_dir,
        limit=args.limit,
        validate=not args.no_validate,
    )


if __name__ == "__main__":
    main()
