"""Local verification of class-Solution code against a sample's unittest-style
`test` field. Splits the test code into per-case assert blocks and counts
passes/failures (mirrors the harness unittest suite, without multiprocessing).

Usage: python check_code.py <sample.json> <code.py> [expected_fail_min]
"""
import json
import re
import sys
from pathlib import Path

sample = json.load(open(sys.argv[1], encoding="utf-8"))
code = Path(sys.argv[2]).read_text(encoding="utf-8")
test_code = sample["test"]

ns = {}
exec(compile(code, "<code>", "exec"), ns)

# Split into blocks: each 'test_input = {...}' ... 'assert ...' pair
parts = re.split(r"(?=test_input = )", test_code)
prefix, blocks = parts[0], parts[1:]
exec(compile(prefix, "<setup>", "exec"), ns)
passed = failed = 0
first_fail = None
for b in blocks:
    if "assert" not in b:
        continue
    local_ns = dict(ns)
    try:
        exec(compile(b, "<test>", "exec"), local_ns)
        passed += 1
    except AssertionError:
        failed += 1
        if first_fail is None:
            first_fail = b.strip().splitlines()[0:2]
    except Exception as e:
        failed += 1
        if first_fail is None:
            first_fail = [f"EXC {type(e).__name__}: {e}"]
print(f"CODE (local unittest replication): passed={passed} failed={failed}")
if first_fail:
    print("first failure:", first_fail)
