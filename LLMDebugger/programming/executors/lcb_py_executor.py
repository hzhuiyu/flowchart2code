"""LiveCodeBench executor for the LDB pipeline.

All solution execution (unit-test feedback in ``execute`` and official mixed
I/O grading in ``evaluate``) runs in an isolated worker subprocess that is
hard-killed on timeout.  Model-generated solutions regularly contain infinite
loops that in-process timers cannot interrupt on Windows (async exception
injection is swallowed by the solution's own ``except`` blocks), so process
isolation is the only reliable guard.
"""

import json
import sys
from pathlib import Path
from typing import NamedTuple, Tuple, List

# flowchart2code/src lives three levels above this file (LLMDebugger/programming/executors).
# Append (never insert) so LDB's own top-level `utils` module keeps priority.
_SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.append(str(_SRC_ROOT))

from .subprocess_runner import run_worker  # noqa: E402

from .py_executor import PyExecutor  # noqa: E402


class LCBPyExecutor(PyExecutor):
    """PyExecutor with subprocess isolation and LCB official grading."""

    def execute(self, func: str, tests, timeout: int = 10):
        # LCB solutions routinely need more than the 1s HumanEval default.
        result = run_worker(
            {"mode": "execute", "func": func, "tests": list(tests)},
            # Worker runs each assert at `timeout`; outer budget is the
            # hard-kill backstop (solutions can swallow the worker's TimeoutError).
            timeout=max(timeout, 10) * max(1, len(tests)) + 15,
        )
        if result["results"] is not None:
            data = result["results"]
            return ExecuteResult(
                data["is_passing"], data["feedback"], tuple(data["state"])
            )
        # Worker crash/timeout: fail all tests with the reason attached, so the
        # debugger prompt shows e.g. "# worker error: TIMEOUT_KILLED".
        return ExecuteResult(
            False,
            [f"{t} # worker error: {result['error']}" for t in tests],
            (),
        )
    def evaluate(self, name: str, func: str, test: str, timeout: int = 10) -> bool:
        # `test` is the JSON string produced by build_lcb_input_output in
        # lcb_convert.py: {"inputs": [...], "outputs": [...], "fn_name": ..., "stdins": ...}.
        io = None
        if isinstance(test, str) and test.strip().startswith("{"):
            try:
                parsed = json.loads(test)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, dict) and "inputs" in parsed:
                io = parsed

        if io is not None:
            result = run_worker(
                {"mode": "evaluate", "func": func, "io": json.dumps(io, ensure_ascii=False)},
                # Worker alarm fires per test at `timeout`; outer budget is the
                # hard-kill backstop for solutions that swallow the alarm.
                timeout=timeout * max(1, len(io.get("inputs", [1]))) + 15,
            )
            results = result["results"]
            return bool(results) and all(r is True for r in results)

        # Fallback: HumanEval-style `def check(candidate)` test code.
        return super().evaluate(name, func, test, timeout=timeout)


class ExecuteResult(NamedTuple):
    """Same shape as executors.executor_types.ExecuteResult (unpackable)."""

    is_passing: bool
    feedback: List[str]
    state: Tuple[str]
