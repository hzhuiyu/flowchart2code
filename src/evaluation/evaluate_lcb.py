#!/usr/bin/env python3
"""
LiveCodeBench (LCB) Evaluation Module
======================================
Standalone evaluation for LCB dataset, integrated into flowchart2code project.
"""

import ast
import json
import sys
import os
import time
import faulthandler
import platform
import threading
import ctypes
import multiprocessing
from datetime import datetime
from io import StringIO
from unittest.mock import patch, mock_open
from types import ModuleType
from enum import Enum
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

_IS_WINDOWS = platform.system() == "Windows"

import_string = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nfrom sortedcontainers import SortedList, SortedDict, SortedSet\nimport sortedcontainers\nsys.setrecursionlimit(1000)\n"


# ==================== Utility Functions ====================

def _strip_ufffd(code: str) -> str:
    """Remove Unicode replacement characters (U+FFFD) that cause SyntaxError."""
    if code is None:
        return ""
    return code.replace('\ufffd', '')


def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s
    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


class TimeoutException(Exception):
    pass


# Cross-platform alarm mechanism
_alarm_timer = None


def _win_async_raise(tid, exc):
    if not isinstance(tid, int):
        return
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid), ctypes.py_object(exc)
    )
    if res == 0:
        return
    elif res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)


def _win_timeout_fire():
    main_tid = threading.main_thread().ident
    _win_async_raise(main_tid, TimeoutException)


def set_alarm(timeout: int):
    global _alarm_timer
    clear_alarm()
    if timeout <= 0:
        return
    if _IS_WINDOWS:
        _alarm_timer = threading.Timer(timeout, _win_timeout_fire)
        _alarm_timer.daemon = True
        _alarm_timer.start()
    else:
        import signal
        signal.alarm(timeout)


def clear_alarm():
    global _alarm_timer
    if _IS_WINDOWS:
        if _alarm_timer is not None:
            _alarm_timer.cancel()
            _alarm_timer = None
    else:
        import signal
        signal.alarm(0)


def init_alarm_handler():
    if not _IS_WINDOWS:
        import signal
        signal.signal(signal.SIGALRM, lambda signum, frame: (_ for _ in ()).throw(TimeoutException))


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


class MockStdinWithBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs
        self._stringio = StringIO(inputs)
        self.buffer = MockBuffer(inputs)

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self.inputs.split("\n")

    def __getattr__(self, name):
        return getattr(self._stringio, name)


class MockBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs.encode("utf-8")

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self.inputs.split(b"\n")[0] + b"\n"


def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = (
                    ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)
                )
    except:
        pass
    return code


def make_function(code: str) -> str:
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        main_code = (
            import_string
            + "\n"
            + ast.unparse(import_stmts)
            + "\n"
            + ast.unparse(function_ast)
        )
        return main_code
    except Exception as e:
        return code


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)
    inputs_line_iterator = iter(inputs.split("\n"))
    mock_stdin = MockStdinWithBuffer(inputs)

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", mock_stdin)
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner_call_method(_method):
        # Also patch sys.stdin.buffer by replacing the attribute directly
        original_buffer = getattr(sys.stdin, 'buffer', None)
        try:
            sys.stdin.buffer = mock_stdin.buffer
        except AttributeError:
            # If buffer is read-only, try to use object.__setattr__
            try:
                object.__setattr__(sys.stdin, 'buffer', mock_stdin.buffer)
            except:
                pass
        try:
            return _method()
        except SystemExit as e:
            pass
        finally:
            # Restore original buffer
            if original_buffer is not None:
                try:
                    sys.stdin.buffer = original_buffer
                except AttributeError:
                    try:
                        object.__setattr__(sys.stdin, 'buffer', original_buffer)
                    except:
                        pass
    return _inner_call_method(method)


def get_function(compiled_sol, fn_name: str):
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception as e:
        return


def compile_code(code: str, timeout: int):
    code = _strip_ufffd(code)
    set_alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if "class Solution" in code:
            compiled_sol = tmp_sol.Solution()
        else:
            compiled_sol = tmp_sol
        assert compiled_sol is not None
    finally:
        clear_alarm()
    return compiled_sol


def convert_line_to_decimals(line: str) -> tuple:
    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except:
        return False, []
    return True, decimal_line


def get_stripped_lines(val: str):
    val = val.strip()
    return [val_line.strip() for val_line in val.split("\n")]


def _parse_expected_value(value: Any) -> Any:
    """Parse a test value while preserving non-literal strings."""
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped:
        return ""

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(stripped)
        except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return stripped


def _normalize_output_text(value: Any) -> str:
    """Normalize presentation-only whitespace without changing content."""
    return "\n".join(" ".join(line.split()) for line in str(value).strip().splitlines())


def _nested_list_as_stdout(expected: Any) -> Optional[str]:
    """Render a nested list/tuple as AtCoder multi-line stdout.

    ``[['Yes'], [0, 0, 6]]`` becomes ``Yes\\n0 0 6``.  One-level lists stay
    with the existing space-join path so ``[1, 2, 3]`` is unchanged.
    """
    if not isinstance(expected, (list, tuple)) or not expected:
        return None
    if not any(isinstance(item, (list, tuple)) for item in expected):
        return None
    lines: List[str] = []
    for item in expected:
        if isinstance(item, (list, tuple)):
            lines.append(" ".join(str(value) for value in item))
        else:
            lines.append(str(item))
    return "\n".join(lines)


def _compare_call_based_output(prediction: Any, expected: Any) -> bool:
    """Compare return/stdout values across functional and stdio formats."""
    if prediction == expected:
        return True
    if isinstance(prediction, tuple) and isinstance(expected, list):
        return list(prediction) == expected
    if isinstance(prediction, list) and isinstance(expected, tuple):
        return prediction == list(expected)

    if isinstance(prediction, str):
        parsed_prediction = _parse_expected_value(prediction)
        if parsed_prediction != prediction and _compare_call_based_output(parsed_prediction, expected):
            return True

        if isinstance(expected, (list, tuple)) and all(
            not isinstance(item, (list, tuple, dict)) for item in expected
        ):
            expected_text = " ".join(str(item) for item in expected)
            if _normalize_output_text(prediction) == _normalize_output_text(expected_text):
                return True

        nested_text = _nested_list_as_stdout(expected)
        if nested_text is not None and _normalize_output_text(prediction) == _normalize_output_text(
            nested_text
        ):
            return True

    if isinstance(expected, str):
        if _normalize_output_text(prediction) == _normalize_output_text(expected):
            return True
        if isinstance(prediction, bool) and prediction == (expected.strip().lower() == "true"):
            return expected.strip().lower() in {"true", "false"}

    try:
        return Decimal(str(prediction)) == Decimal(str(expected))
    except Exception:
        return False


def _validate_special_output(validator: Optional[str], prediction: Any, inputs: list) -> bool:
    """Validate non-unique outputs for the small set of tasks that need it."""
    if validator != "abc363_f" or not inputs:
        return False

    text = str(prediction).strip()
    if not text or text == "-1" or len(text) > 1000 or text != text[::-1]:
        return False
    if text[0] not in "123456789":
        return False

    factors = text.split("*")
    if any(not factor or any(char not in "123456789" for char in factor) for factor in factors):
        return False

    product = 1
    for factor in factors:
        product *= int(factor)
    return product == inputs[0]


def _invoke_call_based(method, args: list):
    """Call with functional args, or without them for a stdin-only solution."""
    import inspect

    try:
        signature = inspect.signature(method)
        signature.bind(*args)
    except (TypeError, ValueError):
        try:
            signature.bind()
        except (TypeError, ValueError, UnboundLocalError):
            return method(*args)
        return method()
    return method(*args)


# ==================== Grading Functions ====================

def grade_call_based(
    code: str,
    all_inputs: list,
    all_outputs: list,
    fn_name: str,
    timeout: int,
    all_stdins: Optional[list] = None,
    output_validator: Optional[str] = None,
):
    code = _strip_ufffd(code)
    code = import_string + "\n\n" + code
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return

    method = get_function(compiled_sol, fn_name)
    if method is None:
        return

    all_inputs = [
        [json.loads(line) for line in inputs.split("\n") if line.strip()] for inputs in all_inputs
    ]
    all_outputs = [_parse_expected_value(output) for output in all_outputs]
    if all_stdins is None:
        all_stdins = [None] * len(all_inputs)

    total_execution = 0
    all_results = []
    for idx, (gt_inp, gt_out, gt_stdin) in enumerate(
        zip(all_inputs, all_outputs, all_stdins)
    ):
        set_alarm(timeout)
        faulthandler.enable()
        try:
            start = time.time()
            
            # Capture stdout to handle print-based output
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            try:
                if gt_stdin is None:
                    prediction = _invoke_call_based(method, gt_inp)
                else:
                    prediction = call_method(
                        lambda: _invoke_call_based(method, gt_inp), gt_stdin
                    )
            finally:
                sys.stdout = old_stdout
            
            total_execution += time.time() - start
            clear_alarm()
            
            # If function returned None but printed output, use the printed output
            if prediction is None:
                printed_output = captured_output.getvalue()
                if printed_output:
                    prediction = _parse_expected_value(printed_output)
            
            # Handle case where function prints but also returns
            elif captured_output.getvalue().strip():
                # Function both printed and returned something
                # Prefer return value, but if it's None, use printed output
                pass

            if isinstance(prediction, tuple):
                prediction = list(prediction)

            tmp_result = _compare_call_based_output(
                prediction, gt_out
            ) or _validate_special_output(output_validator, prediction, gt_inp)
            all_results.append(tmp_result)

            if not tmp_result:
                return all_results, {
                    "output": truncatefn(prediction),
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                    "error_code": -2,
                    "error_message": "Wrong Answer",
                }
        except Exception as e:
            clear_alarm()
            if "timeoutexception" in repr(e).lower():
                all_results.append(-3)
                return all_results, {
                    "error": repr(e),
                    "error_code": -3,
                    "error_message": "Time Limit Exceeded",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }
            else:
                all_results.append(-4)
                return all_results, {
                    "error": repr(e),
                    "error_code": -4,
                    "error_message": "Runtime Error",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }
        finally:
            clear_alarm()
            faulthandler.disable()

    return all_results, {"execution time": total_execution}


def grade_stdio(code: str, all_inputs: list, all_outputs: list, timeout: int):
    code = clean_if_name(code)
    code = make_function(code)
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return

    method = get_function(compiled_sol, "wrapped_function")
    if method is None:
        return

    all_results = []
    total_execution_time = 0
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        set_alarm(timeout)
        faulthandler.enable()
        set_alarm(timeout)

        with Capturing() as captured_output:
            try:
                start = time.time()
                call_method(method, gt_inp)
                total_execution_time += time.time() - start
                clear_alarm()
            except Exception as e:
                clear_alarm()
                if "timeoutexception" in repr(e).lower():
                    all_results.append(-3)
                    return all_results, {
                        "error": repr(e),
                        "error_code": -3,
                        "error_message": "Time Limit Exceeded",
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                    }
                else:
                    all_results.append(-4)
                    return all_results, {
                        "error": repr(e),
                        "error_code": -4,
                        "error_message": "Runtime Error",
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                    }
            finally:
                clear_alarm()
                faulthandler.disable()

        prediction = captured_output[0]
        stripped_prediction_lines = get_stripped_lines(prediction)
        stripped_gt_out_lines = get_stripped_lines(gt_out)

        WA_send_args = {
            "output": truncatefn(prediction),
            "inputs": truncatefn(gt_inp),
            "expected": truncatefn(gt_out),
            "error_code": -2,
        }

        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            all_results.append(-2)
            WA_send_args["error_message"] = "Wrong answer: mismatched output length"
            return all_results, WA_send_args

        for output_line_idx, (
            stripped_prediction_line,
            stripped_gt_out_line,
        ) in enumerate(zip(stripped_prediction_lines, stripped_gt_out_lines)):
            WA_send_args["error_message"] = (
                f"Wrong answer at output_line_idx={output_line_idx}: {truncatefn(stripped_prediction_line)} != {truncatefn(stripped_gt_out_line)}"
            )

            if stripped_prediction_line == stripped_gt_out_line:
                continue

            success, decimal_prediction_line = convert_line_to_decimals(stripped_prediction_line)
            if not success:
                all_results.append(-2)
                return all_results, WA_send_args
            success, decimal_gtout_line = convert_line_to_decimals(stripped_gt_out_line)
            if not success:
                all_results.append(-2)
                return all_results, WA_send_args

            if decimal_prediction_line == decimal_gtout_line:
                continue

            all_results.append(-2)
            return all_results, WA_send_args
        all_results.append(True)

    return all_results, {"execution time": total_execution_time}


def run_test(sample, test=None, debug=False, timeout=6):
    init_alarm_handler()

    try:
        in_outs = json.loads(sample["input_output"])
    except ValueError as e:
        in_outs = None

    if in_outs:
        if in_outs.get("fn_name") is None:
            which_type = CODE_TYPE.standard_input
            method_name = None
        else:
            which_type = CODE_TYPE.call_based
            method_name = in_outs["fn_name"]

    if test is None:
        return in_outs, {"error": "no test code provided"}

    # Guard: never let evaluated code read the real console stdin.  A blocking
    # console read cannot be interrupted by the async timeout exception, so a
    # solution that calls input() outside the mocked stdin would hang the
    # whole in-process evaluation when run from a terminal.  Reads against the
    # empty mock raise EOFError immediately and are recorded as runtime errors.
    # MockStdinWithBuffer is used (not a plain StringIO) because call_method
    # patches sys.stdin.readline, which requires a settable attribute.
    real_stdin = sys.stdin
    sys.stdin = MockStdinWithBuffer("")
    try:
        if which_type == CODE_TYPE.call_based:
            set_alarm(timeout)
            try:
                results, metadata = grade_call_based(
                    code=test,
                    all_inputs=in_outs["inputs"],
                    all_outputs=in_outs["outputs"],
                    fn_name=method_name,
                    timeout=timeout,
                    all_stdins=in_outs.get("stdins"),
                    output_validator=in_outs.get("output_validator"),
                )
                return results, metadata
            except Exception as e:
                return [-4], {
                    "error_code": -4,
                    "error_message": f"Error during testing: {e}",
                }
            finally:
                clear_alarm()
        elif which_type == CODE_TYPE.standard_input:
            set_alarm(timeout)
            try:
                results, metadata = grade_stdio(
                    code=test,
                    all_inputs=in_outs["inputs"],
                    all_outputs=in_outs["outputs"],
                    timeout=timeout,
                )
                return results, metadata
            except Exception as e:
                return [-4], {
                    "error_code": -4,
                    "error_message": f"Error during testing: {e}",
                }
            finally:
                clear_alarm()
    finally:
        sys.stdin = real_stdin


def _digit_text(value: Any) -> Optional[str]:
    """Return a decimal digit string, or None if ``value`` is not one."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        if value < 0:
            return None
        return str(value)
    if isinstance(value, str) and value.isdigit():
        return value
    return None


def _pad_digit_arg_width(n: int, digits: str) -> Optional[int]:
    """Recover a leading-zero width lost when S was stored as an int.

    ``solve(N, S)`` conversions drop zeros: ``S="010"`` becomes ``10``, and
    majority-vote bitstrings of length ``3**N`` become a shorter int.  Prefer
    width ``N`` when the token is already shorter than ``N``; only then try
    ``3**N`` for 0/1 strings (abc391_e).  Do not pad when ``len(S) == N``
    (abc379_e ``379``).
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        return None
    length = len(digits)
    if length < n:
        return n
    # Majority-vote bitstrings have length 3^N (abc391_e). Require the
    # remaining digits to already look truncated, not a packed one-element
    # list such as ``solve(1, 1)``.
    if 1 <= n <= 13 and set(digits) <= {"0", "1"}:
        width = 3 ** n
        if length < width and (length > n or digits == "0"):
            return width
    return None


def _restore_converted_call_args(args: List[Any]) -> List[Any]:
    """Undo packing artifacts from ``solve(...)`` conversion.

    Leading zeros / bitstring width are restored on two-argument ``(N, S)``
    calls.  A lone numeric scalar is wrapped as a one-element list only when
    it is a packed length-1 array: the first argument is a header list whose
    leading ``N`` is 1, and the scalar is not a trailing query after two or
    more lists (``solve([1, 10], 0)`` → ``A = [0]``).  This leaves
    ``solve(N, A)``, trailing query ints such as abc355_e, and mixed int/list
    signatures like abc380_f unchanged.
    """
    restored = list(args)

    if len(restored) == 2:
        digits = _digit_text(restored[1])
        width = _pad_digit_arg_width(restored[0], digits) if digits is not None else None
        if width is not None:
            restored[1] = digits.zfill(width)

    scalar_indexes = [
        i
        for i, arg in enumerate(restored)
        if not isinstance(arg, (list, tuple, dict, bool))
        and arg is not None
        and isinstance(arg, (int, float))
    ]
    header = restored[0] if restored else None
    index = scalar_indexes[0] if len(scalar_indexes) == 1 else None
    n_lists = sum(isinstance(arg, (list, tuple)) for arg in restored)
    if (
        index is not None
        and index > 0
        and isinstance(header, (list, tuple))
        and header
        and header[0] == 1
        and not (index == len(restored) - 1 and n_lists >= 2)
    ):
        restored[index] = [restored[index]]

    return restored


def _args_to_stdin_text(args: List[Any]) -> str:
    stdin_lines: List[str] = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            stdin_lines.append(" ".join(str(item) for item in arg))
        else:
            stdin_lines.append(str(arg))
    return "\n".join(stdin_lines)


def _decode_lcb_call_input(value: Any, fn_name: str) -> Tuple[Optional[List[Any]], str]:
    """Decode a converted ``solve(...)`` test input.

    Returns the positional arguments and their equivalent stdin text.  A
    malformed expression returns ``(None, original_text)`` so callers can
    still evaluate it as a regular stdin case.
    """
    if not isinstance(value, str):
        return None, str(value)
    text = value.strip()
    prefix = f"{fn_name}("
    if not (text.startswith(prefix) and text.endswith(")")):
        return None, value
    args_text = text[len(prefix):-1].strip()
    try:
        if not args_text:
            args: List[Any] = []
        else:
            parsed = ast.parse(args_text, mode="eval").body
            if isinstance(parsed, ast.Tuple):
                args = [ast.literal_eval(element) for element in parsed.elts]
            else:
                args = [ast.literal_eval(parsed)]
    except Exception:
        return None, value

    args = _restore_converted_call_args(args)
    return args, _args_to_stdin_text(args)


def build_lcb_input_output(
    problem: Dict[str, Any],
    test_cases: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the evaluator's canonical LCB ``input_output`` structure.

    Explicit ``lcb_mode`` takes precedence.  Converted ``solve(...)`` cases
    retain a function name and equivalent stdin values so hybrid solutions
    (which read stdin inside ``solve``) continue to work.  Raw stdio cases do
    not receive ``fn_name`` and are graded through stdin/stdout.
    """
    problem = problem or {}
    cases = test_cases if test_cases is not None else problem.get("public_test_cases")
    if not cases:
        cases = problem.get("test_cases") or []
    if isinstance(cases, str):
        try:
            cases = json.loads(cases)
        except Exception:
            cases = []

    metadata = problem.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}
    mode = (problem.get("lcb_mode") or metadata.get("lcb_mode") or "").strip().lower()
    fn_name = problem.get("func_name") or problem.get("entry_point") or metadata.get("func_name")
    fn_name = fn_name or ""

    first_input = cases[0].get("input", "") if cases and isinstance(cases[0], dict) else ""
    converted = bool(fn_name and isinstance(first_input, str)
                    and first_input.strip().startswith(f"{fn_name}(")
                    and first_input.strip().endswith(")"))
    if not mode:
        if problem.get("testtype") == "stdin" or problem.get("test_type") == "stdin":
            mode = "stdio"
        elif converted:
            mode = "call_based"
        elif fn_name:
            mode = "call_based"
        else:
            mode = "stdio"

    inputs: List[str] = []
    outputs: List[str] = []
    stdins: List[str] = []
    use_function = mode == "call_based" or converted

    for case in cases:
        if not isinstance(case, dict):
            continue
        raw_input = case.get("input", "")
        raw_output = case.get("output", "")
        if not isinstance(raw_output, str):
            raw_output = json.dumps(raw_output, ensure_ascii=False)
        args, stdin_text = _decode_lcb_call_input(raw_input, fn_name) if fn_name else (None, str(raw_input))
        if use_function and args is not None:
            inputs.append("\n".join(json.dumps(arg, ensure_ascii=False) for arg in args))
            stdins.append(stdin_text)
        elif use_function:
            # Native functional LCB records already use one JSON value per line.
            if isinstance(raw_input, (list, tuple)):
                input_text = "\n".join(json.dumps(item, ensure_ascii=False) for item in raw_input)
            elif isinstance(raw_input, dict):
                input_text = json.dumps(raw_input, ensure_ascii=False)
            else:
                input_text = str(raw_input)
            inputs.append(input_text)
            stdins.append(input_text)
        else:
            if isinstance(raw_input, (list, tuple)):
                input_text = "\n".join(str(item) for item in raw_input)
            elif isinstance(raw_input, dict):
                input_text = json.dumps(raw_input, ensure_ascii=False)
            else:
                input_text = str(raw_input)
            inputs.append(input_text)
        outputs.append(raw_output)

    result: Dict[str, Any] = {"inputs": inputs, "outputs": outputs}
    if use_function and fn_name:
        result["fn_name"] = fn_name
        result["stdins"] = stdins
    if any(str(case.get("testtype", "")).lower() == "functional" for case in cases if isinstance(case, dict)):
        result.setdefault("lcb_mode", "call_based")
    return result


def reliability_guard(maximum_memory_bytes=None):
    if maximum_memory_bytes is not None and not _IS_WINDOWS:
        import resource
        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )

    faulthandler.disable()

    import builtins
    builtins.exit = None
    builtins.quit = None

    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    if hasattr(os, "setuid"):
        os.setuid = None
    if hasattr(os, "fork"):
        os.fork = None
    if hasattr(os, "forkpty"):
        os.forkpty = None
    if hasattr(os, "killpg"):
        os.killpg = None
    os.rename = None
    os.renames = None
    if hasattr(os, "truncate"):
        os.truncate = None
    os.replace = None
    os.unlink = None
    if hasattr(os, "fchmod"):
        os.fchmod = None
    if hasattr(os, "fchown"):
        os.fchown = None
    if hasattr(os, "chmod"):
        os.chmod = None
    if hasattr(os, "chown"):
        os.chown = None
    if hasattr(os, "chroot"):
        os.chroot = None
    os.fchdir = None
    if hasattr(os, "lchflags"):
        os.lchflags = None
    if hasattr(os, "lchmod"):
        os.lchmod = None
    if hasattr(os, "lchown"):
        os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil
    shutil.rmtree = None
    shutil.move = None
    if hasattr(shutil, "chown"):
        shutil.chown = None

    import subprocess
    subprocess.Popen = None

    __builtins__["help"] = None

    import sys
    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


# ==================== Main Evaluation Function ====================

def _infer_param_types_from_reference(reference_code: str, param_count: int, starter_types: list = None) -> list:
    """Infer parameter types by analyzing how reference code reads input.
    
    Returns a list of types like ['int', 'str', 'list', None, ...] for each parameter.
    
    Key heuristics:
    - Skip lambda/function definitions (e.g. `input = lambda: sys.stdin.readline()`)
    - If `a, b = map(int, input().split())` but starter_code has List type, map to one list
    - Handle `data = list(map(int, sys.stdin.buffer.read().split()))` for multi-value
    """
    if not reference_code or param_count == 0:
        return [None] * param_count
    
    try:
        lines = reference_code.replace('\r\n', '\n').replace('\r', '\n').split('\n')
        
        # Collect all input-reading patterns, skipping lambda definitions
        input_patterns = []
        for line in lines:
            line = line.strip()
            # Skip lambda definitions and function aliases
            if 'lambda' in line and ('input' in line.split('=')[0] if '=' in line else False):
                continue
            if line.startswith('input =') or line.startswith('input='):
                continue
            if line.startswith('ii =') or line.startswith('ii=') or \
               line.startswith('mi =') or line.startswith('mi=') or \
               line.startswith('mmi') or line.startswith('li =') or line.startswith('li='):
                continue
            # Only count actual input reads
            if ('input()' in line or 'sys.stdin' in line) and '=' in line:
                input_patterns.append(line)
        
        result = [None] * param_count
        param_idx = 0
        
        for pattern in input_patterns:
            if param_idx >= param_count:
                break
            
            eq_idx = pattern.index('=')
            var_part = pattern[:eq_idx].strip()
            rhs = pattern[eq_idx+1:].strip()
            is_tuple_unpack = ',' in var_part
            
            # Handle sys.stdin.buffer.read().split() - reads ALL input as one list
            if 'read()' in rhs and 'split()' in rhs:
                # All input read at once - if model has 2 params, split into int + list
                # or list + list depending on starter types
                if param_count == 1:
                    result[param_idx] = 'list'
                    param_idx += 1
                elif param_count == 2 and param_idx == 0:
                    # First param is typically a single int (N), second is the list
                    # But the data list contains everything, so we need to handle differently
                    # Actually, for sys.stdin.buffer.read().split(), model code typically
                    # accesses data[0] as N and data[1:] as the list
                    # So we read first token as int, rest as list
                    result[param_idx] = 'int'  # will be overridden by annotation if needed
                    param_idx += 1
                else:
                    result[param_idx] = 'list'
                    param_idx += 1
                continue
            
            if 'list(map(int' in rhs or 'list(map(lambda' in rhs:
                result[param_idx] = 'list'
                param_idx += 1
            elif 'map(int' in rhs and is_tuple_unpack:
                num_vars = len([v.strip() for v in var_part.split(',') if v.strip()])
                remaining_params = param_count - param_idx
                
                # Check if starter_types says this should be a list
                # If starter has List type for the current param, treat as list
                if starter_types and param_idx < len(starter_types):
                    current_starter_type = starter_types[param_idx]
                    if current_starter_type and ('List' in current_starter_type or 'list' in current_starter_type):
                        # Starter says List, so map tuple unpack to one list
                        result[param_idx] = 'list'
                        param_idx += 1
                        continue
                
                if remaining_params >= num_vars and remaining_params > 1:
                    # Enough params to map each variable to its own int
                    for _ in range(num_vars):
                        if param_idx < param_count:
                            result[param_idx] = 'int'
                            param_idx += 1
                else:
                    # Not enough params or single param - map to one list
                    result[param_idx] = 'list'
                    param_idx += 1
            elif 'map(int' in rhs and not is_tuple_unpack:
                # Single var = map(int, input().split()) -> list
                result[param_idx] = 'list'
                param_idx += 1
            elif 'int(' in rhs:
                result[param_idx] = 'int'
                param_idx += 1
            elif 'float(' in rhs:
                result[param_idx] = 'float'
                param_idx += 1
            elif 'input()' in rhs:
                result[param_idx] = 'str'
                param_idx += 1
            elif 'sys.stdin' in rhs and 'readline' in rhs:
                if 'int(' in rhs:
                    result[param_idx] = 'int'
                elif 'list(map' in rhs:
                    result[param_idx] = 'list'
                else:
                    result[param_idx] = 'str'
                param_idx += 1
            elif 'read()' in rhs or 'readlines' in rhs:
                if 'split()' in rhs or 'split(' in rhs:
                    if 'int' in rhs:
                        result[param_idx] = 'list'
                    else:
                        result[param_idx] = 'str'
                else:
                    result[param_idx] = 'str'
                param_idx += 1
        
        return result
    except:
        return [None] * param_count


def transform_atcoder_code(code: str, fn_name: str = "solve", sample_args: list = None, reference_code: str = None, starter_code: str = None) -> str:
    """Transform AtCoder solve() function code for stdio evaluation.
    
    Model generates code in two styles:
    
    Style 1 (uses function args):
        def solve(arg0: int, arg1: str):
            N = arg0
            S = arg1
            print(ans)
    
    Style 2 (uses stdin/input()):
        def solve(arg0: int, arg1: str):
            N = int(sys.stdin.readline())
            S = input().strip()
            print(ans)
    
    For Style 1: generate wrapper code that reads from stdin, parses based on type annotations,
                 and calls solve() with the parsed arguments.
    For Style 2: remove all parameters from solve() signature and add solve() call at the end.
                 The function body already reads from stdin, so no wrapper is needed.
    
    sample_args: if provided, use these values to infer parameter types when annotations are missing.
    """
    # Clean encoding issues first
    code = _strip_ufffd(code)
    
    try:
        tree = ast.parse(code)
        solve_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                solve_func = node
                break
        
        if solve_func is None:
            return code
        
        # Get parameter info
        params = solve_func.args.args
        param_count = len(params)
        
        if param_count == 0:
            # Already no params, just add call if not present
            already_calls = f"{fn_name}()" in code.replace(f"def {fn_name}()", "")
            if not already_calls:
                code = code + f"\n\n{fn_name}()\n"
            return code
        
        # Get actual parameter names (model may have renamed arg0, arg1 to a, b, etc.)
        param_names = [p.arg for p in params]
        
        # Check if the function body uses stdin/input() to read input
        # If so, we should strip parameters and just call solve()
        func_body_str = ast.unparse(ast.Module(body=solve_func.body, type_ignores=[]))
        uses_stdin = ('sys.stdin' in func_body_str or 'input(' in func_body_str or
                      'sys.stdin' in code and 'read()' in code)
        
        # Check if function body references any of its own parameter names
        # Use AST Name nodes for precise detection (avoids false positives from annotations)
        uses_args = False
        func_body_module = ast.Module(body=solve_func.body, type_ignores=[])
        for node in ast.walk(func_body_module):
            if isinstance(node, ast.Name) and node.id in param_names:
                uses_args = True
                break
        # Fallback: also check string for short renamed params (e.g. N, A, a, b)
        if not uses_args:
            uses_args = any(pname in func_body_str for pname in param_names if len(pname) <= 3)
        
        # Remove the if __name__ == '__main__' block from original code
        new_body = []
        for stmt in tree.body:
            if isinstance(stmt, ast.If):
                try:
                    if ast.unparse(stmt.test).strip() == "__name__ == '__main__'":
                        continue
                except:
                    pass
            new_body.append(stmt)
        
        if uses_stdin and not uses_args:
            # Style 2: function uses stdin directly, doesn't use parameters
            # Strip parameters from solve() signature and add solve() call
            for node in ast.walk(ast.Module(body=new_body, type_ignores=[])):
                if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                    node.args.args = []
                    node.args.posonlyargs = []
                    node.args.kwonlyargs = []
                    node.args.kw_defaults = []
                    node.args.defaults = []
            base_code = ast.unparse(ast.Module(body=new_body, type_ignores=[]))
            already_calls = f"{fn_name}()" in base_code.replace(f"def {fn_name}()", "")
            if not already_calls:
                base_code = base_code + f"\n\n{fn_name}()\n"
            return base_code
        
        # Style 1: function uses parameters (arg0, arg1, ... or renamed a, b, ...)
        # Generate wrapper code that reads from stdin and calls solve() with parsed args
        
        # Step 1: Get type annotations from function signature
        param_types = []
        for p in params:
            ann = p.annotation
            if ann is not None:
                try:
                    ann_str = ast.unparse(ann)
                    param_types.append(ann_str)
                except:
                    param_types.append(None)
            else:
                param_types.append(None)
        
        # Step 2: If reference_code provided, use it to infer correct types
        # This is the most reliable source because it shows how the problem actually reads input
        # Pass the starter_code type annotations so it can respect List vs int distinctions
        # Extract types from starter_code if model code doesn't have annotations
        starter_types = list(param_types)  # Start with model's own annotations
        if starter_code:
            try:
                starter_tree = ast.parse(starter_code)
                for node in ast.walk(starter_tree):
                    if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                        for i, p in enumerate(node.args.args):
                            if i < len(starter_types) and starter_types[i] is None:
                                ann = p.annotation
                                if ann is not None:
                                    try:
                                        starter_types[i] = ast.unparse(ann)
                                    except:
                                        pass
                        break
            except:
                pass
        
        ref_types = _infer_param_types_from_reference(reference_code, param_count, starter_types=starter_types) if reference_code else [None] * param_count
        for i in range(min(len(ref_types), len(param_types))):
            if ref_types[i] is not None:
                param_types[i] = ref_types[i]
        
        # Step 3: If sample_args provided, infer types from them when annotations are missing
        # BUT: be careful - starter_code type annotations can be wrong (e.g. int instead of str)
        # So we prefer sample_args types over annotations when they disagree
        if sample_args:
            for i, arg_val in enumerate(sample_args):
                if i < len(param_types):
                    inferred_type = None
                    if isinstance(arg_val, list):
                        inferred_type = 'list'
                    elif isinstance(arg_val, bool):
                        inferred_type = 'int'
                    elif isinstance(arg_val, int):
                        inferred_type = 'int'
                    elif isinstance(arg_val, float):
                        inferred_type = 'float'
                    elif isinstance(arg_val, str):
                        inferred_type = 'str'
                    
                    # Override annotation if sample_args type disagrees
                    # This handles cases like starter_code says int but actual value is str
                    if inferred_type and param_types[i] and param_types[i] != inferred_type:
                        # If annotation says int but sample is str, trust the sample
                        # because the test input will produce str-type values
                        if param_types[i] == 'int' and inferred_type == 'str':
                            param_types[i] = 'str'
                        elif param_types[i] == 'float' and inferred_type == 'str':
                            param_types[i] = 'str'
                        # If annotation says List[int] but sample is a single int, trust the sample
                        # because the test input will read a single integer, not a space-separated list
                        elif ('List' in param_types[i] or 'list' in param_types[i]) and inferred_type == 'int':
                            param_types[i] = 'int'
                    elif inferred_type and not param_types[i]:
                        param_types[i] = inferred_type
        
        # Generate wrapper code
        import sys as _sys
        print(f"[DEBUG transform] param_types={param_types}, sample_args={sample_args}", file=_sys.stderr)
        wrapper_lines = [
            "import sys",
            f"def _main():",
            "    _input = sys.stdin.read().strip().split('\\n')",
            "    _idx = 0",
            "    def _read_line():",
            "        nonlocal _idx",
            "        if _idx < len(_input):",
            "            line = _input[_idx]",
            "            _idx += 1",
            "            return line",
            "        return ''",
        ]
        
        call_args = []
        for i, ptype in enumerate(param_types):
            pname = f"_arg{i}"
            if ptype and ('List' in ptype or 'list' in ptype):
                wrapper_lines.append(f"    {pname} = list(map(int, _read_line().split()))")
            elif ptype and ptype == 'int':
                wrapper_lines.append(f"    {pname} = int(_read_line())")
            elif ptype and ptype == 'str':
                wrapper_lines.append(f"    {pname} = _read_line()")
            elif ptype and ptype == 'float':
                wrapper_lines.append(f"    {pname} = float(_read_line())")
            else:
                # Unknown type, try to read as string (safest default)
                wrapper_lines.append(f"    {pname} = _read_line()")
            call_args.append(pname)
        
        call_str = ", ".join(call_args)
        wrapper_lines.append(f"    _result = {fn_name}({call_str})")
        wrapper_lines.append("    if _result is not None:")
        wrapper_lines.append("        print(_result)")
        wrapper_lines.append("")
        wrapper_lines.append("_main()")
        wrapper_lines.append("")
        
        base_code = ast.unparse(ast.Module(body=new_body, type_ignores=[]))
        return base_code + "\n" + "\n".join(wrapper_lines)
        
    except Exception as e:
        # Fallback: just add solve() call at the end
        try:
            already_calls = f"{fn_name}(" in code.replace(f"def {fn_name}(", "")
            if not already_calls:
                return code + f"\n\n{fn_name}()\n"
        except:
            pass
        return code


def _isolated_run_test_worker(sample, test, timeout, conn) -> None:
    """Child-process entry: run one sample and send ``(status, payload)``."""
    try:
        result = run_test(sample, test=test, timeout=timeout)
        conn.send(("ok", result))
    except Exception as e:
        conn.send(("err", repr(e)))
    finally:
        conn.close()


def run_test_with_process_timeout(sample, test, timeout: int = 6):
    """Run ``run_test`` in a subprocess so infinite loops can be killed.

    Windows cannot reliably interrupt a tight Python loop with
    ``PyThreadState_SetAsyncExc``.  A child process can be terminated, after
    which the sample is recorded as TLE and evaluation continues.
    """
    n_tests = 1
    try:
        in_outs = json.loads(sample["input_output"])
        n_tests = max(1, len(in_outs.get("inputs") or []))
    except Exception:
        pass
    wait_s = timeout * n_tests + 5

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_isolated_run_test_worker,
        args=(sample, test, timeout, child_conn),
    )
    proc.start()
    child_conn.close()
    proc.join(wait_s)

    if proc.is_alive():
        proc.terminate()
        proc.join(3)
        if proc.is_alive():
            proc.kill()
            proc.join(1)
        parent_conn.close()
        return [-3], {
            "error_code": -3,
            "error_message": f"Time Limit Exceeded after {wait_s}s",
        }

    if parent_conn.poll(1):
        status, payload = parent_conn.recv()
        parent_conn.close()
        if status == "ok":
            if not payload or payload[0] is None:
                return [-4], {
                    "error_code": -4,
                    "error_message": "No test results",
                }
            return payload
        return [-4], {
            "error_code": -4,
            "error_message": str(payload),
        }

    parent_conn.close()
    return [-3], {
        "error_code": -3,
        "error_message": "Time Limit Exceeded",
    }


def evaluate_lcb_samples(samples_file: str, problem_file: str, timeout: int = 6) -> Tuple[float, List[Dict]]:
    """
    Evaluate LCB samples and return pass@1 and detailed results.
    
    Args:
        samples_file: Path to samples.jsonl file
        problem_file: Path to LiveCodeBench.jsonl problem file (can be flowchart2code format or original LCB format)
        timeout: Timeout per test case in seconds
    
    Returns:
        (pass_at_1, detailed_results)
    """
    # Load problems
    problems = {}
    with open(problem_file, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            try:
                problem = json.loads(line.strip())
                # LCB uses task_id as the unique identifier
                problem_id = problem.get("task_id", problem.get("question_id", ""))
                problems[problem_id] = problem
                # Also store with platform prefix (e.g., leetcode_2757)
                platform = problem.get("platform", "")
                if platform and problem_id:
                    problems[f"{platform}_{problem_id}"] = problem
                # Also store with numeric ID only
                if str(problem_id).isdigit():
                    problems[str(problem_id)] = problem
            except:
                continue

    # Load samples
    samples = []
    with open(samples_file, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            try:
                samples.append(json.loads(line.strip()))
            except:
                continue

    results = []
    passed_count = 0

    total_samples = len(samples)
    for sample_idx, sample in enumerate(samples, 1):
        task_id = sample.get("task_id", "")
        completion = sample.get("completion", "")
        print(f"Evaluating sample {sample_idx}/{total_samples}: {task_id}...", flush=True)
        problem = problems.get(task_id, {})
        if not problem:
            # Try extracting ID from task_id (e.g., v1_2757 -> 2757, v1_abc301_f -> abc301_f)
            parts = task_id.split("_")
            if len(parts) > 1:
                # For numeric IDs like v1_2757 -> 2757
                numeric_id = parts[-1]
                problem = problems.get(numeric_id, {})
                
                # For abc format like v1_abc301_f -> abc301_f
                if not problem and len(parts) >= 3 and parts[1].startswith("abc"):
                    abc_id = "_".join(parts[1:])  # abc301_f
                    problem = problems.get(abc_id, {})
                
                # For arc format like v6_arc196_a -> arc196_a
                if not problem and len(parts) >= 3 and parts[1].startswith("arc"):
                    arc_id = "_".join(parts[1:])  # arc196_a
                    problem = problems.get(arc_id, {})
        
        if not problem or not completion:
            result = {
                "task_id": task_id,
                "passed": False,
                "error_code": -1,
                "error_message": "Missing problem or completion",
            }
            results.append(result)
            continue

        # Build input_output from raw test cases
        raw_public_tests = problem.get("raw_public_tests", [])
        raw_private_tests = problem.get("raw_private_tests", [])
        if isinstance(raw_public_tests, str):
            try:
                raw_public_tests = json.loads(raw_public_tests)
            except Exception:
                raw_public_tests = []
        if isinstance(raw_private_tests, str):
            try:
                raw_private_tests = json.loads(raw_private_tests)
            except Exception:
                raw_private_tests = []
        if not isinstance(raw_public_tests, list):
            raw_public_tests = []
        if not isinstance(raw_private_tests, list):
            raw_private_tests = []
        all_tests = raw_public_tests + raw_private_tests
        
        # If no raw tests, try to parse from original LCB format
        if not all_tests:
            public_tests = problem.get("public_test_cases", [])
            private_tests = problem.get("private_test_cases", [])
            
            # Parse public_test_cases
            if isinstance(public_tests, str):
                try:
                    public_tests = json.loads(public_tests)
                except:
                    public_tests = []
            
            # Parse private_test_cases (might be compressed)
            if isinstance(private_tests, str):
                try:
                    private_tests = json.loads(private_tests)
                except:
                    try:
                        import zlib, pickle, base64
                        private_tests = json.loads(
                            pickle.loads(
                                zlib.decompress(
                                    base64.b64decode(private_tests.encode("utf-8"))
                                )
                            )
                        )
                    except:
                        private_tests = []
            
            all_tests = public_tests + private_tests
        
        if not all_tests:
            result = {
                "task_id": task_id,
                "passed": False,
                "error_code": -1,
                "error_message": "No test cases found",
            }
            results.append(result)
            continue

        input_output = build_lcb_input_output(problem, all_tests)

        if task_id.endswith("_abc363_f"):
            input_output["output_validator"] = "abc363_f"
        
        # Create a sample dict compatible with run_test
        test_sample = {
            "input_output": json.dumps(input_output)
        }
        
        # Run test in a subprocess so a hung solution can be killed on timeout.
        test_results, metadata = run_test_with_process_timeout(
            test_sample, completion, timeout=timeout
        )
        if isinstance(metadata, dict) and metadata.get("error_code") == -3:
            print(
                f"  [TIMEOUT] {task_id}: {metadata.get('error_message', 'TLE')}",
                flush=True,
            )

        # Check if passed (all True)
        passed = all(r is True for r in test_results) if test_results else False
        
        if passed:
            passed_count += 1

        result = {
            "task_id": task_id,
            "passed": passed,
            "test_results": test_results,
            "metadata": metadata,
            "difficulty": problem.get("difficulty", ""),
        }
        results.append(result)

    pass_at_1 = passed_count / len(samples) if samples else 0.0
    
    return pass_at_1, results


def save_lcb_results(output_dir: Path, pass_at_1: float, results: List[Dict]) -> None:
    """
    Save LCB evaluation results to files.
    
    Args:
        output_dir: Directory to save results
        pass_at_1: Overall pass@1 score
        results: Detailed results for each sample
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results as JSONL
    results_file = output_dir / "samples.jsonl_results.jsonl"
    with open(results_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # Save only the overall metric; LCB results no longer include difficulty sections.
    results_txt = output_dir / "results.txt"
    with open(results_txt, 'w', encoding='utf-8') as f:
        f.write(f"{{'pass@1': {pass_at_1}}}\n")
    
    print(f"  Results saved: {results_file}")
    print(f"  Summary saved: {results_txt}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate LCB samples")
    parser.add_argument("--samples", type=str, required=True, help="Path to samples.jsonl")
    parser.add_argument("--problems", type=str, required=True, help="Path to problems.jsonl")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--timeout", type=int, default=6, help="Timeout per test case")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.samples).parent
    
    print("Evaluating LCB samples...")
    pass_at_1, results = evaluate_lcb_samples(args.samples, args.problems, args.timeout)
    print(f"Pass@1: {pass_at_1:.4f}")
    
    save_lcb_results(output_dir, pass_at_1, results)
