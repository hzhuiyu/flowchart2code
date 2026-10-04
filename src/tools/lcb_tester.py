"""
LCB (LiveCodeBench) call_based tester.

Tests generated code against LCB-style public_test_cases where:
- input:  a function call expression string, e.g. "solve([3,3], [2,1,1])"
- output: the expected return value as a Python literal string, e.g. "6"

This is used for the converted LCB problems that are now in call_based format.

Execution model: generated code is NEVER exec'd in the parent process. Each
test case runs in a fresh python subprocess (temp-file job/result IPC, hard
timeout, stdin=DEVNULL), so module-level ``input()`` in generated code fails
fast with EOFError instead of blocking the tester forever, and infinite loops
are killed at the per-case timeout instead of hanging the whole run.
"""

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, Any, Optional


class _WholeTestAbort(Exception):
    """Internal: module-level failure — abort test() with this message."""


# Runner script executed as a separate python process. It reads the job file
# path from argv[1] (a Python literal written with repr(), so tuples in the
# arguments survive the round trip) and writes a JSON result to argv[2], then
# hard-exits so lingering non-daemon threads in generated code cannot hang it.
_RUNNER_SOURCE = '''\
import ast
import json
import os
import sys
import time


def _find_callable(exec_globals, func_name):
    candidate = exec_globals.get(func_name)
    if callable(candidate):
        return candidate
    if "Solution" in exec_globals:
        try:
            method = getattr(exec_globals["Solution"](), func_name, None)
            if callable(method):
                return method
        except Exception:
            pass
    for obj in exec_globals.values():
        if isinstance(obj, type) and callable(getattr(obj, func_name, None)):
            try:
                return getattr(obj(), func_name)
            except Exception:
                continue
    for name, obj in exec_globals.items():
        if callable(obj) and name == func_name:
            return obj
    return None


def main():
    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        job = ast.literal_eval(fh.read())
    result = None
    try:
        exec_globals = {}
        exec(compile(job["code"], "<lcb_solution>", "exec"), exec_globals)
    except BaseException as exc:
        result = {"status": "exec_error",
                  "error": "%s: %s" % (type(exc).__name__, exc)}
    if result is None:
        callable_obj = _find_callable(exec_globals, job["func_name"])
        if callable_obj is None:
            result = {"status": "no_callable",
                      "error": "Function '%s' not found" % job["func_name"]}
        else:
            try:
                start = time.time()
                args, kwargs = job.get("args") or [], job.get("kwargs") or {}
                if kwargs:
                    actual = callable_obj(**kwargs)
                elif args:
                    actual = callable_obj(*args)
                else:
                    actual = callable_obj()
                try:
                    actual_repr = repr(actual)
                except Exception:
                    actual_repr = str(actual)
                result = {"status": "ok", "actual_repr": actual_repr,
                          "elapsed": time.time() - start}
            except BaseException as exc:
                result = {"status": "call_error",
                          "error": "%s: %s" % (type(exc).__name__, exc)}
    with open(sys.argv[2], "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    sys.stdout.flush()
    os._exit(0)


main()
'''

_RUNNER_PATH: Optional[str] = None


def _get_runner_path() -> str:
    """Materialize the runner script once per process (temp dir cleanup is
    left to the OS; one small file per run is harmless)."""
    global _RUNNER_PATH
    if _RUNNER_PATH is None or not os.path.exists(_RUNNER_PATH):
        fd, _RUNNER_PATH = tempfile.mkstemp(prefix="lcb_runner_", suffix=".py")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(_RUNNER_SOURCE)
    return _RUNNER_PATH


class LCBCallBasedTester:
    """Tester for LCB call_based problems."""

    # Standard imports injected into the execution namespace
    _IMPORTS = """from typing import List, Tuple, Dict, Any, Optional
import math
from collections import defaultdict, deque, Counter
from functools import cache
import collections
import heapq
import bisect
import itertools
"""

    def test(
        self,
        generated_code: str,
        problem_info: Dict[str, Any],
        timeout: float = 6.0,
    ) -> Dict[str, Any]:
        """
        Test generated code against LCB public_test_cases.

        Args:
            generated_code: The generated Python code.
            problem_info: Must contain:
                - starter_code: function/class skeleton
                - func_name: entry function name
                - public_test_cases: list of {"input": "func(args)", "output": "expected"}
            timeout: Per-test-case timeout in seconds (covers module exec + call).

        Returns:
            Standard test result dict.
        """
        starter_code = problem_info.get("starter_code", "")
        func_name = problem_info.get("func_name", "")
        test_cases = problem_info.get("public_test_cases", [])

        if not test_cases:
            return {
                "success": False,
                "passed": 0,
                "total": 0,
                "pass_rate": 0,
                "results": [],
                "error": "No public test cases",
                "test_method": "lcb_call_based",
            }

        if not func_name:
            # Try to extract from starter_code
            func_name = self._extract_func_name(starter_code) or self._extract_func_name(generated_code)

        # Build the complete program: imports + starter_code + generated_code
        # Strategy: if generated_code already contains the class/function definition,
        # use it directly. Otherwise wrap it.
        full_code = self._build_full_code(starter_code, generated_code, func_name)

        # Compile once in the parent (SyntaxError is deterministic and needs
        # no subprocess); actual execution happens in the runner subprocess.
        try:
            compile(full_code, "<lcb_solution>", "exec")
        except SyntaxError as e:
            return {
                "success": False,
                "passed": 0,
                "total": len(test_cases),
                "pass_rate": 0,
                "results": [],
                "error": f"Syntax error: {e}",
                "test_method": "lcb_call_based",
            }

        runner_path = _get_runner_path()

        # Run each test case
        results = []
        passed_count = 0

        try:
            for idx, tc in enumerate(test_cases):
                result = self._run_single_test(
                    runner_path, full_code, func_name,
                    tc.get("input", ""), tc.get("output", ""), idx + 1, timeout,
                )
                results.append(result)
                if result["passed"]:
                    passed_count += 1
        except _WholeTestAbort as exc:
            # Module-level failure (exec error / missing callable): the same
            # failure would repeat for every case, so report it once.
            return {
                "success": False,
                "passed": 0,
                "total": len(test_cases),
                "pass_rate": 0,
                "results": [],
                "error": str(exc),
                "test_method": "lcb_call_based",
            }

        total = len(results)
        return {
            "success": True,
            "passed": passed_count,
            "total": total,
            "pass_rate": (passed_count / total * 100) if total else 0,
            "results": results,
            "error": None,
            "test_method": "lcb_call_based",
        }

    def _build_full_code(self, starter_code: str, generated_code: str, func_name: str) -> str:
        """Build complete executable code from starter and generated code."""
        # If generated_code already defines the function/class, use it
        # Otherwise, we need to combine starter_code + generated_code

        # Check if generated_code is standalone (has class/def)
        has_definition = False
        try:
            tree = ast.parse(generated_code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    has_definition = True
                    break
        except SyntaxError:
            pass

        if has_definition:
            # Generated code is self-contained
            return self._IMPORTS + "\n" + generated_code

        # Generated code might be just the function body; combine with starter
        if starter_code.strip():
            return self._IMPORTS + "\n" + starter_code + "\n" + generated_code

        # Fallback: wrap generated code in a function
        return self._IMPORTS + f"\ndef {func_name}():\n    pass\n" + generated_code

    def _run_single_test(
        self,
        runner_path: str,
        full_code: str,
        func_name: str,
        tc_input: str,
        tc_output: str,
        test_num: int,
        timeout: float,
    ) -> Dict[str, Any]:
        """Run a single LCB test case in a fresh subprocess."""
        try:
            # Parse the input call expression
            # tc_input is like: "solve([3, 3], [2, 1, 1])"
            # or: "\"FFF\"" (for single string arg)
            # We need to extract arguments and call the function

            call_expr = tc_input.strip()

            # Try to parse as a function call
            args, kwargs = self._parse_call_expression(call_expr, func_name)

            # Parse expected output
            expected = self._parse_expected(tc_output)

            # Execute in a fresh subprocess (hard timeout, stdin=DEVNULL)
            completed, payload = self._execute_in_subprocess(
                runner_path, full_code, func_name, args, kwargs, timeout,
            )

            if not completed:
                # payload is a failure reason string
                error = "Time Limit Exceeded" if payload == "timeout" else f"Runner error: {payload}"
                return {
                    "test_num": test_num,
                    "passed": False,
                    "input": self._truncate(call_expr),
                    "expected": self._truncate(repr(expected)),
                    "actual": "",
                    "error": error,
                }

            status = payload.get("status")
            if status == "exec_error":
                raise _WholeTestAbort(f"Execution error: {payload.get('error', '')}")
            if status == "no_callable":
                raise _WholeTestAbort(
                    f"Function '{func_name}' not found in generated code"
                )
            if status == "call_error":
                return {
                    "test_num": test_num,
                    "passed": False,
                    "input": self._truncate(call_expr),
                    "expected": self._truncate(repr(expected)),
                    "actual": "",
                    "error": f"Runtime Error: {payload.get('error', '')}",
                }

            # status == "ok": reconstruct the return value from its repr
            actual_repr = payload.get("actual_repr", "")
            try:
                actual = ast.literal_eval(actual_repr)
            except (ValueError, SyntaxError, MemoryError):
                actual = actual_repr
            elapsed = payload.get("elapsed", 0.0)

            # Compare actual vs expected
            passed = self._compare_values(actual, expected)

            return {
                "test_num": test_num,
                "passed": passed,
                "input": self._truncate(call_expr),
                "expected": self._truncate(repr(expected)),
                "actual": self._truncate(repr(actual)),
                "error": None if passed else f"Expected {expected}, got {actual}",
                "execution_time": round(elapsed, 4),
            }

        except _WholeTestAbort:
            raise
        except Exception as e:
            return {
                "test_num": test_num,
                "passed": False,
                "input": self._truncate(str(tc_input)),
                "expected": self._truncate(str(tc_output)),
                "actual": "",
                "error": f"Test execution error: {e}",
            }

    def _execute_in_subprocess(
        self,
        runner_path: str,
        full_code: str,
        func_name: str,
        args: list,
        kwargs: dict,
        timeout: float,
    ) -> tuple:
        """Run code + call in a fresh python subprocess via temp-file IPC.

        stdin is /dev/null so module-level ``input()`` raises EOFError instead
        of blocking; stdout/stderr are discarded so endless printing cannot
        fill a pipe and deadlock. Returns (True, result_dict) when the runner
        completed, or (False, reason) when it timed out or died silently.
        """
        tmpdir = tempfile.mkdtemp(prefix="lcb_case_")
        try:
            job_path = os.path.join(tmpdir, "job.txt")
            out_path = os.path.join(tmpdir, "result.json")
            with open(job_path, "w", encoding="utf-8") as fh:
                # repr() round-trips tuples that JSON would flatten to lists
                fh.write(repr({"code": full_code, "func_name": func_name,
                               "args": args, "kwargs": kwargs}))

            proc = subprocess.Popen(
                [sys.executable, runner_path, job_path, out_path],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=tmpdir,
            )
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                return False, "timeout"

            if not os.path.exists(out_path):
                return False, f"no result file (exit code {proc.returncode})"
            try:
                with open(out_path, "r", encoding="utf-8") as fh:
                    return True, json.load(fh)
            except Exception as exc:
                return False, f"unreadable result file ({exc})"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _parse_call_expression(self, call_expr: str, func_name: str) -> tuple:
        """
        Parse a call expression like "solve([3, 3], [2, 1, 1])" into args and kwargs.
        Also supports raw stdin-style multi-line input (for original call_based LCB problems).
        Returns (args_list, kwargs_dict).
        """
        call_expr = call_expr.strip()

        # Check if it's already a function call expression
        if call_expr.startswith(func_name + "(") and call_expr.endswith(")"):
            inner = call_expr[len(func_name) + 1 : -1]
            return self._parse_arg_string(inner)
        elif "(" in call_expr and call_expr.endswith(")"):
            # Extract everything inside the outermost parentheses
            inner = self._extract_inner_args(call_expr)
            return self._parse_arg_string(inner)
        elif "\n" in call_expr:
            # Multi-line input: treat each line as a separate argument
            # This handles original call_based LCB problems where input is stdin-style
            return self._parse_multiline_input(call_expr)
        else:
            # Single value, treat as single argument
            return self._parse_arg_string(call_expr)

    def _parse_arg_string(self, arg_str: str) -> tuple:
        """Parse an argument string into a list of args."""
        arg_str = arg_str.strip()
        if not arg_str:
            return [], {}

        # Try to parse as Python expression tuple
        try:
            parsed = ast.literal_eval(f"({arg_str},)")
            if isinstance(parsed, tuple):
                return list(parsed), {}
        except (ValueError, SyntaxError):
            pass

        # Try to parse the whole thing as a single value
        try:
            parsed = ast.literal_eval(arg_str)
            return [parsed], {}
        except (ValueError, SyntaxError):
            pass

        # Fallback: return as raw string
        return [arg_str], {}

    def _parse_multiline_input(self, input_text: str) -> tuple:
        """
        Parse multi-line stdin-style input into arguments.
        Each line becomes one argument. Try to parse each line as a Python value.
        """
        lines = input_text.strip().split("\n")
        args = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Try to parse as Python literal
            try:
                parsed = ast.literal_eval(line)
                args.append(parsed)
            except (ValueError, SyntaxError):
                # If can't parse, use raw string
                args.append(line)
        return args, {}

    def _extract_inner_args(self, call_expr: str) -> str:
        """Extract inner arguments from a call expression."""
        # Find the matching parentheses
        start = call_expr.find("(")
        if start == -1:
            return call_expr

        depth = 1
        i = start + 1
        while i < len(call_expr) and depth > 0:
            if call_expr[i] == "(" and (i == 0 or call_expr[i - 1] != "\\"):
                depth += 1
            elif call_expr[i] == ")" and (i == 0 or call_expr[i - 1] != "\\"):
                depth -= 1
            i += 1

        if depth == 0:
            return call_expr[start + 1 : i - 1]
        return call_expr[start + 1 :]

    def _parse_expected(self, expected_str: str) -> Any:
        """Parse expected output string into Python value."""
        expected_str = expected_str.strip()
        if not expected_str:
            return ""

        try:
            return ast.literal_eval(expected_str)
        except (ValueError, SyntaxError):
            return expected_str

    def _compare_values(self, actual: Any, expected: Any) -> bool:
        """Compare actual and expected values with type coercion."""
        # Direct equality
        if actual == expected:
            return True

        # Tuple vs list equivalence
        if isinstance(actual, tuple) and isinstance(expected, list):
            return list(actual) == expected
        if isinstance(actual, list) and isinstance(expected, tuple):
            return actual == list(expected)

        # Numeric comparison
        try:
            if float(actual) == float(expected):
                return True
        except (ValueError, TypeError):
            pass

        # String comparison
        return str(actual).strip() == str(expected).strip()

    def _extract_func_name(self, code: str) -> Optional[str]:
        """Extract function name from code."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except SyntaxError:
            pass
        return None

    @staticmethod
    def _truncate(s, length: int = 300) -> str:
        if not isinstance(s, str):
            s = str(s)
        if len(s) <= length:
            return s
        return s[: length // 2] + "...(truncated)..." + s[-length // 2:]


# ------------------------------------------------------------------
#  Integration with CodeTester
# ------------------------------------------------------------------

def is_lcb_call_based_problem(problem_info: Dict[str, Any]) -> bool:
    """Check if a problem is in LCB call_based format."""
    # Has public_test_cases with functional testtype
    test_cases = problem_info.get("public_test_cases", [])
    if test_cases and isinstance(test_cases, list):
        first_tc = test_cases[0]
        if isinstance(first_tc, dict):
            testtype = first_tc.get("testtype", "")
            if testtype == "functional":
                return True
    # Or has func_name + public_test_cases (converted LCB)
    if problem_info.get("func_name") and problem_info.get("public_test_cases"):
        return True
    return False


def test_lcb_call_based(
    generated_code: str,
    problem_info: Dict[str, Any],
    timeout: float = 6.0,
) -> Dict[str, Any]:
    """Convenience function to test LCB call_based problems."""
    tester = LCBCallBasedTester()
    return tester.test(generated_code, problem_info, timeout)
