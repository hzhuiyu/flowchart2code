"""Run untrusted solution code in a worker subprocess with hard-kill timeouts.

Model-generated solutions can contain infinite loops that survive every
in-process timeout mechanism on Windows (Timer-based async exception injection
is swallowed by `except Exception` inside the solution, or never delivered
while stuck in a C call).  The only reliable isolation is a separate process
that is terminated from the outside.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
_PROG_ROOT = Path(__file__).resolve().parents[1]

_WORKER_PROGRAM = r'''
import io, json, os, sys

payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
mode = payload["mode"]
timeout = payload["timeout"]

sys.path.append(payload["src_root"])
if payload.get("prog_root"):
    sys.path.append(payload["prog_root"])

# Real stdin is never available in the worker; solutions reading stdin must
# rely on the evaluator's mocks (call_method patches sys.stdin per test).
sys.stdin = io.StringIO("")

results = None
metadata = None
error = None
try:
    if mode == "evaluate":
        from evaluation.evaluate_lcb import run_test
        results, metadata = run_test(
            {"input_output": payload["io"]},
            test=payload["func"],
            timeout=timeout,
        )
    elif mode == "eval_code":
        # Execute a raw Python code string (HumanEval/Algorithm/MATH style).
        # Returns True if exec succeeds without exception, False otherwise.
        from executors.executor_utils import function_with_timeout
        try:
            function_with_timeout(exec, (payload["code"], globals()), timeout)
            results = True
        except Exception:
            results = False
    elif mode == "execute":
        from executors.py_executor import PyExecutor
        exe_result = PyExecutor().execute(payload["func"], payload["tests"], timeout=timeout)
        results = {
            "is_passing": exe_result.is_passing,
            "feedback": exe_result.feedback,
            "state": list(exe_result.state),
        }
    else:
        error = f"unknown mode {mode}"
except BaseException as e:  # worker must always answer, even on crash
    error = f"{type(e).__name__}: {e}"

def _jsonable(obj):
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        return str(obj)

try:
    print(json.dumps({"results": results,
                      "metadata": None if metadata is None else _jsonable(metadata),
                      "error": error}))
except BaseException as e:
    print(json.dumps({"results": None, "metadata": None,
                      "error": f"SERIALIZE_FAIL {type(e).__name__}: {e}"}))
# Failed timed-out asserts leave zombie PropagatingThreads (non-daemon) that
# would keep the worker alive until the outer kill; the result is already
# printed, so exit the interpreter directly.
sys.stdout.flush()
os._exit(0)
'''

_worker_path_cache = None


def _get_worker_path() -> str:
    global _worker_path_cache
    if _worker_path_cache is None:
        f = tempfile.NamedTemporaryFile(
            "w", suffix="_ldb_worker.py", delete=False, encoding="utf-8"
        )
        f.write(_WORKER_PROGRAM)
        f.close()
        _worker_path_cache = f.name
    return _worker_path_cache


def run_worker(payload: dict, timeout: int):
    """Run one payload in a fresh worker process; return parsed result dict.

    On timeout the process is killed; `error` is set to "TIMEOUT_KILLED".
    """
    import subprocess

    payload = dict(payload)
    payload["src_root"] = str(_SRC_ROOT)
    payload["prog_root"] = str(_PROG_ROOT)
    payload["timeout"] = timeout

    payload_file = None
    proc = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix="_ldb_payload.json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(payload, f, ensure_ascii=False)
            payload_file = f.name
        proc = subprocess.Popen(
            [sys.executable, "-u", _get_worker_path(), payload_file],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass
            return {
                "results": None,
                "metadata": None,
                "error": "TIMEOUT_KILLED",
            }
        if proc.returncode != 0 or not stdout.strip():
            return {
                "results": None,
                "metadata": None,
                "error": f"WORKER_CRASH rc={proc.returncode} {stderr[-2000:]}",
            }
        # The worker prints exactly one JSON object on the last stdout line.
        for line in reversed(stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except ValueError:
                    break
        return {"results": None, "metadata": None,
                "error": f"WORKER_BAD_OUTPUT {stdout[-2000:]}"}
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
        if payload_file is not None:
            try:
                os.unlink(payload_file)
            except OSError:
                pass
