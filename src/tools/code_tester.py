"""
Code testing tool
Uses test cases from problem_extractor to test generated code
Supports two testing modes:
1. test_with_simple_cases: Uses simplified test cases (original method)
2. test_with_unittest_suite: Uses complete unittest test suite (same as HumanEval)
"""

import ast
import json
import subprocess
import sys
import tempfile
import os
import contextlib
import io
import traceback
import platform
import threading
import ctypes
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


# ---- Cross-platform timeout ----
_IS_WINDOWS = platform.system() == "Windows"


class _TimeoutException(Exception):
    pass


_win_alarm_timer = None


def _win_async_raise(tid, exc):
    """Raise exception in a thread (Windows fallback for signal.alarm)."""
    if not isinstance(tid, int):
        return
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid), ctypes.py_object(exc)
    )
    if res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)


def _win_timeout_fire():
    main_tid = threading.main_thread().ident
    _win_async_raise(main_tid, _TimeoutException)


def _sigalrm_handler(signum, frame):
    raise _TimeoutException("Timed out")


def _set_alarm(timeout):
    global _win_alarm_timer
    _clear_alarm()
    if timeout <= 0:
        return
    if _IS_WINDOWS:
        _win_alarm_timer = threading.Timer(timeout, _win_timeout_fire)
        _win_alarm_timer.daemon = True
        _win_alarm_timer.start()
    else:
        import signal
        # Without a handler, SIGALRM's default action terminates the whole
        # process; register one so a timeout raises an exception instead.
        signal.signal(signal.SIGALRM, _sigalrm_handler)
        signal.alarm(int(timeout))


def _clear_alarm():
    global _win_alarm_timer
    if _IS_WINDOWS:
        if _win_alarm_timer is not None:
            _win_alarm_timer.cancel()
            _win_alarm_timer = None
    else:
        import signal
        signal.alarm(0)


# ---- stdio helpers (adapted from LiveCodeBench testing_util) ----

class _MockBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs.encode("utf-8")

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self.inputs.split(b"\n")[0] + b"\n"


class _MockStdin:
    """Mock stdin with buffer support, compatible with input() and sys.stdin.read()."""
    def __init__(self, inputs: str):
        self.inputs = inputs
        self._stringio = io.StringIO(inputs)
        self.buffer = _MockBuffer(inputs)

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self.inputs.split("\n")

    def __getattr__(self, name):
        return getattr(self._stringio, name)


class _Capturing(list):
    """Context manager that captures stdout into a list."""
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = io.StringIO()
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


class CodeTester:
    """Code tester

    Supports three testing modes:
    1. unittest suite  – HumanEval / Algorithm style (assert-based, call_based)
    2. stdin/stdout    – LiveCodeBench style (input()/print() based, standard_input)
    3. simple cases    – dict-based test cases (legacy)

    Additionally supports optional **code instrumentation** that captures
    variable snapshots and execution traces for the reflection agent.
    """

    def __init__(self, data_root: str = "../data"):
        """
        Initialize code tester

        Args:
            data_root: Data root directory
        """
        self.data_root = Path(data_root)
        
    def test_generated_code(
        self,
        generated_code: str,
        problem_info: Dict[str, Any],
        language: str = "python",
        use_unittest: bool = True,
        timeout: float = 3.0
    ) -> Dict[str, Any]:
        """
        Test generated code

        Args:
            generated_code: Generated code
            problem_info: Problem information, containing test cases
            language: Programming language, currently only supports python
            use_unittest: Whether to use unittest test suite (same as HumanEval).
                         True=use complete unittest test suite (recommended)
                         False=use simplified test_cases testing
            timeout: Timeout duration (seconds)

        Returns:
            Test result dictionary
        """
        if language != "python":
            return {
                'success': False,
                'passed': 0,
                'total': 0,
                'results': [],
                'error': f"Unsupported programming language: {language}"
            }

        # Auto-detect stdin/stdout mode (LiveCodeBench style)
        if problem_info.get('input_output') and not problem_info.get('test'):
            input_output = problem_info['input_output']
            if isinstance(input_output, str):
                input_output = json.loads(input_output)
            return self.test_with_stdio(generated_code, input_output, timeout)

        # Prefer unittest test suite (same as HumanEval)
        if use_unittest and problem_info.get('test'):
            return self.test_with_unittest_suite(generated_code, problem_info, timeout)

        # Fall back to simplified test_cases testing
        try:
            # Validate code syntax
            syntax_valid = self._validate_python_syntax(generated_code)
            if not syntax_valid:
                return {
                    'success': False,
                    'passed': 0,
                    'total': 0,
                    'results': [],
                    'error': 'Generated code has syntax errors'
                }

            # Clean generated code, remove top-level statements that may interfere with testing (such as print calls)
            cleaned_code = self._clean_code(generated_code)

            # Extract test cases
            test_cases = problem_info.get('test_cases', [])
            if not test_cases:
                return {
                    'success': False,
                    'passed': 0,
                    'total': 0,
                    'results': [],
                    'error': 'No available test cases'
                }

            # Run tests
            results = []
            passed_count = 0

            # Extract function name, prefer entry_point
            function_name = problem_info.get('entry_point')
            if not function_name:
                function_name = self._extract_function_name(generated_code)

            for i, test_case in enumerate(test_cases):
                result = self._run_single_test(cleaned_code, test_case, i + 1, function_name)
                results.append(result)

                if result['passed']:
                    passed_count += 1

            return {
                'success': True,
                'passed': passed_count,
                'total': len(test_cases),
                'pass_rate': passed_count / len(test_cases) * 100 if test_cases else 0,
                'results': results,
                'function_name': function_name,
                'error': None,
                'test_method': 'simple_cases'  # Mark the test method used
            }

        except Exception as e:
            return {
                'success': False,
                'passed': 0,
                'total': 0,
                'results': [],
                'error': f"Error during testing: {str(e)}"
            }
    
    def _validate_python_syntax(self, code: str) -> bool:
        """Validate Python code syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            print(f"Syntax error: {e}")
            return False
        except Exception as e:
            print(f"Error validating syntax: {e}")
            return False
    
    def _clean_code(self, code: str) -> str:
        """
        Clean generated code, keep only imports, functions and class definitions, remove other top-level statements

        Args:
            code: Original code

        Returns:
            Cleaned code
        """
        try:
            import ast

            tree = ast.parse(code)

            # Collect nodes to keep
            keep_nodes = []
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
                    keep_nodes.append(node)
                elif isinstance(node, ast.Assign):
                    # Keep assignment statements (may be used for constants or configuration)
                    keep_nodes.append(node)
                elif isinstance(node, ast.Expr):
                    # Expression statements (like print calls) should be removed
                    # But if it's a docstring (string constant), keep it
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        # String constant may be a docstring, keep it
                        keep_nodes.append(node)
                    # Other expressions (like print) ignore
                    continue
                else:
                    # Other types of nodes (like if, while, for, etc.) remove
                    continue

            # If no nodes are kept, return original code
            if not keep_nodes:
                return code

            # Rebuild AST
            new_tree = ast.Module(body=keep_nodes, type_ignores=[])

            # Convert to code string
            # Prefer ast.unparse (Python 3.9+)
            if hasattr(ast, 'unparse'):
                cleaned_code = ast.unparse(new_tree)
            else:
                # Fall back to astor
                try:
                    import astor
                    cleaned_code = astor.to_source(new_tree)
                except ImportError:
                    # Final fallback: manually build simple code
                    # This is just a basic fallback, may not be perfect
                    cleaned_code = self._simple_ast_to_code(new_tree)

            # Add necessary typing imports
            typing_imports = []
            for typing_name in ['List', 'Tuple', 'Dict', 'Any', 'Optional']:
                if typing_name in cleaned_code and f'from typing import {typing_name}' not in cleaned_code:
                    typing_imports.append(typing_name)
            if typing_imports:
                import_line = f'from typing import {", ".join(typing_imports)}'
                # Ensure import statement is at the beginning of the code
                cleaned_code = import_line + '\n' + cleaned_code

            return cleaned_code

        except Exception as e:
            print(f"Error cleaning code: {e}")
            # Return original code on error
            return code
    
    def _simple_ast_to_code(self, tree) -> str:
        """
        Simple implementation of converting AST tree to code string
        Only handles import, function, class and assignment statements

        Args:
            tree: AST tree

        Returns:
            Code string
        """
        import ast

        lines = []

        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    lines.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                names = ', '.join([alias.name for alias in node.names])
                lines.append(f"from {module} import {names}")
            elif isinstance(node, ast.FunctionDef):
                # Simple function definition (ignore parameter details)
                lines.append(f"def {node.name}():")
                lines.append("    pass")  # Placeholder
            elif isinstance(node, ast.ClassDef):
                lines.append(f"class {node.name}:")
                lines.append("    pass")  # Placeholder
            elif isinstance(node, ast.Assign):
                # Simple assignment (ignore complex expressions)
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    target = node.targets[0].id
                    lines.append(f"{target} = ...")

        return '\n'.join(lines)
    
    def _run_single_test(self, code: str, test_case: Dict[str, Any], test_num: int, function_name: Optional[str] = None) -> Dict[str, Any]:
        """Run a single test"""
        try:
            input_str = test_case.get('input', '')
            expected_output = test_case.get('output', '')
            explanation = test_case.get('explanation', '')

            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                # Write code
                f.write(code)
                f.write('\n\n')

                # Add test code
                test_code = self._create_test_code(code, input_str, expected_output, function_name)
                f.write(test_code)

                temp_file = f.name

            try:
                # Run test
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    timeout=5  # 5 second timeout
                )

                # Check output
                actual_output = result.stdout.strip() if result.stdout is not None else ''
                error_output = result.stderr.strip() if result.stderr is not None else ''

                # Determine if passed
                passed = False
                error_message = None

                if result.returncode != 0:
                    error_message = f"Program exited abnormally, return code: {result.returncode}"
                    if error_output:
                        error_message += f"\nError message: {error_output}"
                else:
                    # Compare output
                    passed = self._compare_output(actual_output, expected_output)
                    if not passed:
                        error_message = f"Output mismatch. Expected: {expected_output}, Actual: {actual_output}"

                return {
                    'test_num': test_num,
                    'input': input_str,
                    'expected_output': expected_output,
                    'actual_output': actual_output,
                    'passed': passed,
                    'error': error_message,
                    'explanation': explanation
                }

            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file)
                except:
                    pass

        except subprocess.TimeoutExpired:
            return {
                'test_num': test_num,
                'input': input_str,
                'expected_output': expected_output,
                'actual_output': '',
                'passed': False,
                'error': 'Test timed out (exceeded 5 seconds)',
                'explanation': explanation
            }
        except Exception as e:
            return {
                'test_num': test_num,
                'input': input_str,
                'expected_output': expected_output,
                'actual_output': '',
                'passed': False,
                'error': f"Test execution error: {str(e)}",
                'explanation': explanation
            }
    
    def _extract_solution_info(self, code: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract solution class name and method name (or standalone function name) from generated code

        Args:
            code: Generated code

        Returns:
            (class_name, method_name) tuple. If code contains a class, return class name and first method name;
            If only contains standalone function, return (None, function_name); if neither found, return (None, None)
        """
        try:
            import ast

            tree = ast.parse(code)

            # First look for class definition
            class_name = None
            method_name = None

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    # Find first method definition
                    for subnode in node.body:
                        if isinstance(subnode, ast.FunctionDef):
                            method_name = subnode.name
                            break
                    if method_name:
                        return class_name, method_name

            # If no class found, look for standalone function
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Standalone function
                    return None, node.name

            return None, None

        except Exception as e:
            print(f"Error extracting solution info: {e}")
            return None, None
        
    def _get_function_arg_count(self, code: str, class_name: Optional[str], method_name: str) -> int:
                """
                Get the number of function parameters

                Args:
                    code: Complete code
                    class_name: Class name, None if standalone function
                    method_name: Method name or function name

                Returns:
                    Number of parameters, return -1 if unable to determine
                """
                try:
                    import ast
                    tree = ast.parse(code)

                    # Find target function
                    for node in ast.walk(tree):
                        if class_name and isinstance(node, ast.ClassDef) and node.name == class_name:
                            # Find method in class
                            for subnode in node.body:
                                if isinstance(subnode, ast.FunctionDef) and subnode.name == method_name:
                                    # Calculate parameter count (excluding self)
                                    args = subnode.args
                                    arg_count = len(args.args)
                                    if args.vararg:
                                        arg_count += 1  # *args
                                    if args.kwarg:
                                        arg_count += 1  # **kwargs
                                    # If it's a method, subtract self parameter
                                    if arg_count > 0 and 'self' in [arg.arg for arg in args.args[:1]]:
                                        arg_count -= 1
                                    return arg_count
                        elif not class_name and isinstance(node, ast.FunctionDef) and node.name == method_name:
                            # Standalone function
                            args = node.args
                            arg_count = len(args.args)
                            if args.vararg:
                                arg_count += 1  # *args
                            if args.kwarg:
                                arg_count += 1  # **kwargs
                            return arg_count
                except Exception as e:
                    print(f"Error getting function parameter count: {e}")

                return -1
        
    def _create_test_code(self, code: str, input_data: Any, expected_output: str, function_name: Optional[str] = None) -> str:
        """
        Create test code
        
        Args:
            code: Generated code
            input_data: Input data, may be a string or dictionary
            expected_output: Expected output (not used)
            
        Returns:
            Test code string
        """
        try:
            # Extract solution info
            if function_name is not None:
                class_name = None
                method_name = function_name
            else:
                class_name, method_name = self._extract_solution_info(code)
            
            # If no method name found, use fallback method
            if not method_name:
                return self._create_fallback_test_code(code, input_data)
            
            # Build call string
            if class_name:
                # Class method call
                call_prefix = f"{class_name}().{method_name}"
            else:
                # Standalone function call
                call_prefix = method_name
            
            # Determine input data type and generate appropriate call
            if isinstance(input_data, dict):
                # Check if it's HumanEval format test case (contains args key)
                if 'args' in input_data:
                    # Get function parameter count
                    arg_count = self._get_function_arg_count(code, class_name, method_name)
                    
                    import json
                    args_str = json.dumps(input_data['args'], ensure_ascii=False)
                    
                    if arg_count == 1:
                        # Function has only one parameter, pass args as single argument
                        call_code = f"{call_prefix}({args_str})"
                    else:
                        # Function has multiple parameters, use *args unpacking
                        call_code = f"{call_prefix}(*{args_str})"
                elif 'kwargs' in input_data:
                    # Use **kwargs as keyword arguments
                    import json
                    kwargs_str = json.dumps(input_data['kwargs'], ensure_ascii=False)
                    call_code = f"{call_prefix}(**{kwargs_str})"
                else:
                    # Input is a dictionary, use ** unpacking
                    import json
                    input_dict_str = json.dumps(input_data, ensure_ascii=False)
                    call_code = f"{call_prefix}(**{input_dict_str})"
            elif isinstance(input_data, str):
                # Input is a string, may be an expression or simple value
                # Try to evaluate to determine type
                try:
                    import ast
                    parsed = ast.literal_eval(input_data)
                    if isinstance(parsed, dict):
                        # String represents a dictionary
                        import json
                        input_dict_str = json.dumps(parsed, ensure_ascii=False)
                        call_code = f"{call_prefix}(**{input_dict_str})"
                    else:
                        # Other types (number, string, list, etc.)
                        call_code = f"{call_prefix}({input_data})"
                except:
                    # Cannot parse, use string directly
                    call_code = f"{call_prefix}({input_data})"
            else:
                # Other types, use directly
                call_code = f"{call_prefix}({input_data})"
            
            # Generate test code
            test_code = f'''
if __name__ == "__main__":
    try:
        result = {call_code}
        print(str(result))
    except Exception as e:
        print(f"Test execution error: {{e}}")
'''
            return test_code
            
        except Exception as e:
            print(f"Error creating test code: {e}")
            # Return simple test code
            return '''
if __name__ == "__main__":
    print("Test code generation failed")
'''
    
    def _create_fallback_test_code(self, code: str, input_data: Any) -> str:
        """Create fallback test code (when solution info cannot be extracted)"""
        # Try to directly execute the input (if it's a string)
        if isinstance(input_data, str):
            return f'''
if __name__ == "__main__":
    try:
        {code}
        result = eval("{input_data}")
        print(str(result))
    except Exception as e:
        print(f"Test execution error: {{e}}")
'''
        else:
            return '''
if __name__ == "__main__":
    print("Cannot generate test code: input data is not a string and no solution class found")
'''
    
    def _compare_output(self, actual: str, expected: Any) -> bool:
        """Compare actual output with expected output"""
        # Ensure actual is a string
        if not isinstance(actual, str):
            actual = str(actual)
        
        # Clean actual (remove whitespace)
        actual_clean = actual.strip()
        
        # Convert expected to string for comparison
        if not isinstance(expected, str):
            expected_str = str(expected)
        else:
            expected_str = expected
        
        expected_clean = expected_str.strip()
        
        # First try to parse both as Python objects for comparison (handles lists, dicts, etc.)
        try:
            import ast
            actual_parsed = ast.literal_eval(actual_clean)
            expected_parsed = ast.literal_eval(expected_clean)
            return actual_parsed == expected_parsed
        except:
            pass
        
        # Then try numeric comparison (if both are numbers)
        try:
            actual_num = float(actual_clean)
            expected_num = float(expected_clean)
            return abs(actual_num - expected_num) < 1e-9
        except:
            pass
        
        # Try integer comparison (to avoid floating point errors)
        try:
            actual_int = int(actual_clean)
            expected_int = int(expected_clean)
            return actual_int == expected_int
        except:
            pass
        
        # Finally perform string comparison (remove all spaces for lenient comparison)
        # Remove all spaces in string for comparison
        actual_no_spaces = ''.join(actual_clean.split())
        expected_no_spaces = ''.join(expected_clean.split())
        if actual_no_spaces == expected_no_spaces:
            return True
        
        # Original string comparison
        return actual_clean == expected_clean
    
    def _extract_function_name(self, code: str) -> Optional[str]:
        """Extract function name from code"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except:
            pass
        
        return None
    
    def test_with_specific_inputs(self, generated_code: str, inputs: List[str],
                                expected_outputs: List[str]) -> Dict[str, Any]:
        """
        Test code with specific inputs

        Args:
            generated_code: Generated code
            inputs: Input list
            expected_outputs: Expected output list

        Returns:
            Test results
        """
        if len(inputs) != len(expected_outputs):
            return {
                'success': False,
                'error': 'Input and output list lengths do not match'
            }

        try:
            results = []
            passed_count = 0

            for i, (input_str, expected_output) in enumerate(zip(inputs, expected_outputs)):
                test_case = {'input': input_str, 'output': expected_output}
                result = self._run_single_test(generated_code, test_case, i + 1)
                results.append(result)

                if result['passed']:
                    passed_count += 1

            return {
                'success': True,
                'passed': passed_count,
                'total': len(inputs),
                'pass_rate': passed_count / len(inputs) * 100 if inputs else 0,
                'results': results,
                'error': None
            }

        except Exception as e:
            return {
                'success': False,
                'error': f"Error during testing: {str(e)}"
            }

    # ------------------------------------------------------------------
    #  stdin / stdout testing  (LiveCodeBench style)
    # ------------------------------------------------------------------

    _LCB_IMPORT_STRING = (
        "from string import *\nfrom re import *\nfrom datetime import *\n"
        "from collections import *\nfrom heapq import *\nfrom bisect import *\n"
        "from copy import *\nfrom math import *\nfrom random import *\n"
        "from statistics import *\nfrom itertools import *\nfrom functools import *\n"
        "from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\n"
        "from builtins import *\nfrom typing import *\n"
        "import string\nimport re\nimport datetime\nimport collections\nimport heapq\n"
        "import bisect\nimport copy\nimport math\nimport random\nimport statistics\n"
        "import itertools\nimport functools\nimport operator\nimport io\nimport sys\n"
        "import json\nsys.setrecursionlimit(50000)\n"
    )

    @staticmethod
    def _clean_if_name(code: str) -> str:
        """Remove trailing ``if __name__ == '__main__':`` wrapper."""
        try:
            astree = ast.parse(code)
            last_block = astree.body[-1]
            if isinstance(last_block, ast.If):
                condition = last_block.test
                if ast.unparse(condition).strip() == "__name__ == '__main__'":
                    code = (
                        ast.unparse(astree.body[:-1]) + "\n"
                        + ast.unparse(last_block.body)  # type: ignore
                    )
        except Exception:
            pass
        return code

    @classmethod
    def _wrap_as_function(cls, code: str) -> str:
        """Wrap free-standing code into ``wrapped_function()`` for stdio testing."""
        try:
            import_stmts: List[ast.stmt] = []
            other_stmts: List[ast.stmt] = []
            astree = ast.parse(code)
            for stmt in astree.body:
                if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    import_stmts.append(stmt)
                else:
                    other_stmts.append(stmt)

            func_ast = ast.FunctionDef(
                name="wrapped_function",
                args=ast.arguments(
                    posonlyargs=[], args=[], kwonlyargs=[],
                    kw_defaults=[], defaults=[]
                ),
                body=other_stmts or [ast.Pass()],
                decorator_list=[],
                lineno=-1,
            )
            return (
                cls._LCB_IMPORT_STRING + "\n"
                + ast.unparse(import_stmts) + "\n"
                + ast.unparse(func_ast)
            )
        except Exception:
            return code

    @staticmethod
    def _get_stripped_lines(val: str) -> List[str]:
        val = val.strip()
        return [line.strip() for line in val.split("\n")]

    @staticmethod
    def _convert_line_to_decimals(line: str) -> Tuple[bool, list]:
        from decimal import Decimal
        try:
            return True, [Decimal(elem) for elem in line.split()]
        except Exception:
            return False, []

    @staticmethod
    def _truncate(s, length: int = 300) -> str:
        if not isinstance(s, str):
            s = str(s)
        if len(s) <= length:
            return s
        return s[: length // 2] + "...(truncated) ..." + s[-length // 2:]

    def test_with_stdio(
        self,
        generated_code: str,
        input_output: Dict[str, Any],
        timeout: float = 6.0,
    ) -> Dict[str, Any]:
        """Test code that reads from *stdin* and writes to *stdout*.

        Args:
            generated_code: The generated Python source code.
            input_output: A dict with keys ``inputs`` (list[str]) and
                ``outputs`` (list[str]).  Each *input* is fed to ``input()``
                / ``sys.stdin.read()`` and each *output* is the expected
                stdout.  ``fn_name`` (optional) switches to call_based mode.
            timeout: Per-test-case timeout in seconds.

        Returns:
            Standard test-result dict (same shape as other methods).
        """
        all_inputs: List[str] = input_output.get("inputs", [])
        all_outputs: List[str] = input_output.get("outputs", [])
        fn_name: Optional[str] = input_output.get("fn_name")

        if not all_inputs:
            return {
                'success': False, 'passed': 0, 'total': 0,
                'pass_rate': 0, 'results': [],
                'error': 'No stdin test cases provided',
            }

        # Decide mode
        if fn_name:
            # call_based: code has a function, we call it with parsed args
            return self._grade_call_based_stdio(
                generated_code, all_inputs, all_outputs, fn_name, timeout
            )
        else:
            # standard_input: code reads from input() / sys.stdin
            return self._grade_stdio(
                generated_code, all_inputs, all_outputs, timeout
            )

    def _grade_call_based_stdio(
        self, code: str, all_inputs: List[str],
        all_outputs: List[str], fn_name: str, timeout: float,
    ) -> Dict[str, Any]:
        """Grade call_based LCB problems (function takes args, returns value)."""
        from types import ModuleType
        from unittest.mock import patch, mock_open

        full_code = self._LCB_IMPORT_STRING + "\n\n" + code
        try:
            _set_alarm(timeout)
            tmp_sol = ModuleType("tmp_sol", "")
            exec(full_code, tmp_sol.__dict__)
            if "class Solution" in code:
                compiled_sol = tmp_sol.Solution()
            else:
                compiled_sol = tmp_sol
            assert compiled_sol is not None
            _clear_alarm()
        except _TimeoutException:
            _clear_alarm()
            return self._stdio_fail("Compile Time Limit Exceeded", -3, timeout)
        except Exception as e:
            _clear_alarm()
            return self._stdio_fail(f"Compile Error: {e}", -4, timeout)

        method = getattr(compiled_sol, fn_name, None)
        if method is None:
            return self._stdio_fail(f"Function '{fn_name}' not found", -4, timeout)

        parsed_inputs = [
            [json.loads(line) for line in inp.split("\n")] for inp in all_inputs
        ]
        parsed_outputs = [json.loads(out) for out in all_outputs]

        results: List[Dict[str, Any]] = []
        passed_count = 0

        for idx, (gt_inp, gt_out) in enumerate(zip(parsed_inputs, parsed_outputs)):
            _set_alarm(timeout)
            try:
                start = time.time()
                prediction = method(*gt_inp)
                elapsed = time.time() - start
                _clear_alarm()

                if isinstance(prediction, tuple):
                    prediction = list(prediction)

                if prediction == gt_out:
                    results.append({
                        'test_num': idx + 1,
                        'passed': True,
                        'input': self._truncate(gt_inp),
                        'expected': self._truncate(gt_out),
                        'actual': self._truncate(prediction),
                        'error': None,
                        'execution_time': round(elapsed, 4),
                    })
                    passed_count += 1
                else:
                    results.append({
                        'test_num': idx + 1,
                        'passed': False,
                        'input': self._truncate(gt_inp),
                        'expected': self._truncate(gt_out),
                        'actual': self._truncate(prediction),
                        'error': 'Wrong Answer',
                        'execution_time': round(elapsed, 4),
                    })
            except _TimeoutException:
                _clear_alarm()
                results.append({
                    'test_num': idx + 1, 'passed': False,
                    'input': self._truncate(gt_inp), 'expected': self._truncate(gt_out),
                    'actual': '', 'error': 'Time Limit Exceeded',
                })
            except Exception as e:
                _clear_alarm()
                results.append({
                    'test_num': idx + 1, 'passed': False,
                    'input': self._truncate(gt_inp), 'expected': self._truncate(gt_out),
                    'actual': '', 'error': f'Runtime Error: {e}',
                })
            finally:
                _clear_alarm()

        total = len(results)
        return {
            'success': True,
            'passed': passed_count,
            'total': total,
            'pass_rate': (passed_count / total * 100) if total else 0,
            'results': results,
            'error': None,
            'test_method': 'stdio_call_based',
        }

    def _grade_stdio(
        self, code: str, all_inputs: List[str],
        all_outputs: List[str], timeout: float,
    ) -> Dict[str, Any]:
        """Grade standard_input LCB problems (input()/print() based)."""
        from types import ModuleType
        from unittest.mock import patch, mock_open

        # Prepare code: remove if __name__ guard, wrap into function
        code = self._clean_if_name(code)
        code = self._wrap_as_function(code)

        try:
            _set_alarm(timeout)
            tmp_sol = ModuleType("tmp_sol", "")
            exec(code, tmp_sol.__dict__)
            compiled_sol = tmp_sol
            assert compiled_sol is not None
            _clear_alarm()
        except _TimeoutException:
            _clear_alarm()
            return self._stdio_fail("Compile Time Limit Exceeded", -3, timeout)
        except Exception as e:
            _clear_alarm()
            return self._stdio_fail(f"Compile Error: {e}", -4, timeout)

        method = getattr(compiled_sol, "wrapped_function", None)
        if method is None:
            return self._stdio_fail("wrapped_function not found", -4, timeout)

        results: List[Dict[str, Any]] = []
        passed_count = 0

        for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
            _set_alarm(timeout)
            captured_output = ""
            try:
                inputs_iter = iter(gt_inp.split("\n"))
                mock_stdin = _MockStdin(gt_inp)

                with _Capturing() as captured:
                    with patch("builtins.open", mock_open(read_data=gt_inp)):
                        with patch("sys.stdin", mock_stdin):
                            with patch("sys.stdin.readline", lambda *a: next(inputs_iter)):
                                with patch("sys.stdin.readlines", lambda *a: gt_inp.split("\n")):
                                    with patch("sys.stdin.read", lambda *a: gt_inp):
                                        start = time.time()
                                        method()
                                        elapsed = time.time() - start

                _clear_alarm()
                captured_output = captured[0] if captured else ""

                pred_lines = self._get_stripped_lines(captured_output)
                gt_lines = self._get_stripped_lines(gt_out)

                passed = False
                error_msg = None

                if len(pred_lines) != len(gt_lines):
                    error_msg = (
                        f"Wrong answer: mismatched output length "
                        f"(expected {len(gt_lines)} lines, got {len(pred_lines)})"
                    )
                else:
                    passed = True
                    for line_idx, (pred_line, gt_line) in enumerate(zip(pred_lines, gt_lines)):
                        if pred_line == gt_line:
                            continue
                        # Try decimal comparison for floats
                        ok1, dec_pred = self._convert_line_to_decimals(pred_line)
                        ok2, dec_gt = self._convert_line_to_decimals(gt_line)
                        if ok1 and ok2 and dec_pred == dec_gt:
                            continue
                        passed = False
                        error_msg = (
                            f"Wrong answer at line {line_idx}: "
                            f"{self._truncate(pred_line)} != {self._truncate(gt_line)}"
                        )
                        break

                results.append({
                    'test_num': idx + 1,
                    'passed': passed,
                    'input': self._truncate(gt_inp),
                    'expected': self._truncate(gt_out),
                    'actual': self._truncate(captured_output),
                    'error': error_msg,
                    'execution_time': round(elapsed, 4),
                })
                if passed:
                    passed_count += 1

            except _TimeoutException:
                _clear_alarm()
                results.append({
                    'test_num': idx + 1, 'passed': False,
                    'input': self._truncate(gt_inp), 'expected': self._truncate(gt_out),
                    'actual': '', 'error': 'Time Limit Exceeded',
                })
            except Exception as e:
                _clear_alarm()
                results.append({
                    'test_num': idx + 1, 'passed': False,
                    'input': self._truncate(gt_inp), 'expected': self._truncate(gt_out),
                    'actual': '', 'error': f'Runtime Error: {e}',
                })
            finally:
                _clear_alarm()

        total = len(results)
        return {
            'success': True,
            'passed': passed_count,
            'total': total,
            'pass_rate': (passed_count / total * 100) if total else 0,
            'results': results,
            'error': None,
            'test_method': 'stdio_standard_input',
        }

    @staticmethod
    def _stdio_fail(message: str, error_code: int, timeout: float) -> Dict[str, Any]:
        return {
            'success': False,
            'passed': 0,
            'total': 0,
            'pass_rate': 0,
            'results': [{
                'test_num': 1,
                'passed': False,
                'input': 'N/A',
                'expected': 'N/A',
                'actual': '',
                'error': message,
            }],
            'error': message,
            'test_method': 'stdio',
        }

    # ------------------------------------------------------------------
    #  Code instrumentation
    # ------------------------------------------------------------------

    def instrument_code(
        self,
        code: str,
        watch_vars: Optional[List[str]] = None,
        max_trace_entries: int = 200,
    ) -> str:
        """Insert tracing hooks into *code*.

        After every executable statement a ``__trace__`` call is inserted
        that records the line number, local variables (filtered by
        *watch_vars* if given), and the current output buffer.

        The instrumented code, when executed, populates a global
        ``__trace_log__`` list which can be retrieved afterwards.

        Args:
            code: Original Python source.
            watch_vars: If provided, only these variable names are recorded.
            max_trace_entries: Safety limit to avoid memory explosion.

        Returns:
            Instrumented Python source code.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code

        trace_preamble = (
            "import json as __trace_json\n"
            "import sys as __trace_sys\n"
            "from io import StringIO as __trace_StringIO\n"
            "__trace_log__ = []\n"
            "__trace_max__ = " + str(max_trace_entries) + "\n"
            "__trace_watch__ = " + repr(watch_vars) + "\n"
            "def __trace__(__line__):\n"
            "    if len(__trace_log__) >= __trace_max__:\n"
            "        return\n"
            "    __frame__ = __trace_sys._getframe(1)\n"
            "    __locals__ = {}\n"
            "    for __k__, __v__ in __frame__.f_locals.items():\n"
            "        if __k__.startswith('__') or __k__ in ('__trace__', '__trace_log__', '__trace_max__', '__trace_watch__', '__trace_json', '__trace_sys', '__trace_StringIO'):\n"
            "            continue\n"
            "        if __trace_watch__ is not None and __k__ not in __trace_watch__:\n"
            "            continue\n"
            "        try:\n"
            "            repr(__v__)\n"
            "            __locals__[__k__] = __v__\n"
            "        except Exception:\n"
            "            __locals__[__k__] = '<unreprable>'\n"
            "    __trace_log__.append({'line': __line__, 'variables': __locals__})\n"
        )

        # Walk the tree and insert trace calls
        new_body: List[ast.stmt] = []

        def _make_trace_call(lineno: int) -> ast.Expr:
            return ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="__trace__", ctx=ast.Load()),
                    args=[ast.Constant(value=lineno)],
                    keywords=[],
                )
            )

        def _instrument_stmts(stmts: List[ast.stmt]) -> List[ast.stmt]:
            result: List[ast.stmt] = []
            for stmt in stmts:
                # Insert trace before the statement
                result.append(_make_trace_call(stmt.lineno))
                # Recursively instrument bodies of compound statements
                if isinstance(stmt, (ast.For, ast.While)):
                    stmt.body = _instrument_stmts(stmt.body)
                    if stmt.orelse:
                        stmt.orelse = _instrument_stmts(stmt.orelse)
                elif isinstance(stmt, ast.If):
                    stmt.body = _instrument_stmts(stmt.body)
                    if stmt.orelse:
                        stmt.orelse = _instrument_stmts(stmt.orelse)
                elif isinstance(stmt, (ast.With,)):
                    stmt.body = _instrument_stmts(stmt.body)
                elif isinstance(stmt, ast.Try):
                    stmt.body = _instrument_stmts(stmt.body)
                    for handler in stmt.handlers:
                        handler.body = _instrument_stmts(handler.body)
                    if stmt.orelse:
                        stmt.orelse = _instrument_stmts(stmt.orelse)
                    if stmt.finalbody:
                        stmt.finalbody = _instrument_stmts(stmt.finalbody)
                elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    stmt.body = _instrument_stmts(stmt.body)
                elif isinstance(stmt, ast.ClassDef):
                    stmt.body = _instrument_stmts(stmt.body)
                result.append(stmt)
            return result

        new_body = _instrument_stmts(tree.body)

        instrumented_tree = ast.Module(body=new_body, type_ignores=[])
        try:
            instrumented_code = ast.unparse(instrumented_tree)
        except Exception:
            return code

        return trace_preamble + "\n" + instrumented_code

    def test_with_instrumentation(
        self,
        generated_code: str,
        problem_info: Dict[str, Any],
        watch_vars: Optional[List[str]] = None,
        timeout: float = 6.0,
    ) -> Dict[str, Any]:
        """Run a test with code instrumentation, returning execution traces.

        This is designed for the **reflection agent**: when a test fails,
        the agent can examine the trace to understand *where* the code
        diverged from expected behavior.

        The method auto-detects the test type (unittest / stdio / simple)
        and runs the instrumented version.  The returned dict includes
        a ``trace`` key containing the execution log.

        Args:
            generated_code: Generated code to test.
            problem_info: Standard problem info dict.
            watch_vars: Optional list of variable names to watch.
            timeout: Timeout in seconds.

        Returns:
            Test result dict with an added ``trace`` key.
        """
        import multiprocessing

        # Instrument the code
        instrumented_code = self.instrument_code(
            generated_code, watch_vars=watch_vars
        )

        # We need to run the instrumented code in a subprocess so we
        # can retrieve ``__trace_log__`` from the exec globals.
        test_type = self._detect_test_type(problem_info)

        def _worker(result_list, trace_list, code, problem_info, test_type, timeout):
            """Run in subprocess."""
            try:
                if test_type == "unittest":
                    prompt = problem_info.get('prompt', '')
                    test_code = problem_info.get('test', '')
                    entry_point = problem_info.get('entry_point', '')

                    package_import = (
                        "from typing import List, Tuple, Optional, Any\n"
                        "import math, collections, heapq, bisect, time\n"
                        "from collections import defaultdict, deque, Counter\n"
                        "from itertools import accumulate\n"
                        "from functools import cache\n"
                    )

                    if entry_point:
                        check_program = (
                            package_import + "\n" +
                            prompt + code + "\n" +
                            test_code + "\n" +
                            f"check({entry_point})"
                        )
                    else:
                        check_program = (
                            package_import + "\n" +
                            code + "\n" + test_code
                        )

                    exec_globals: Dict[str, Any] = {}
                    old_stdout = sys.stdout
                    sys.stdout = io.StringIO()
                    try:
                        exec(check_program, exec_globals)
                        result_list.append("passed")
                    except AssertionError as e:
                        result_list.append(f"failed: {e}")
                    except Exception as e:
                        result_list.append(f"failed: {e}")
                    finally:
                        sys.stdout = old_stdout

                    trace_list.append(exec_globals.get("__trace_log__", []))

                elif test_type == "stdio":
                    input_output = problem_info.get('input_output', {})
                    if isinstance(input_output, str):
                        input_output = json.loads(input_output)
                    fn_name = input_output.get("fn_name")
                    all_inputs = input_output.get("inputs", [])
                    all_outputs = input_output.get("outputs", [])

                    if fn_name:
                        # call_based with instrumentation
                        full_code = self._LCB_IMPORT_STRING + "\n\n" + code
                        exec_globals = {}
                        exec(full_code, exec_globals)
                        compiled_sol = exec_globals
                        if "class Solution" in code:
                            compiled_sol = compiled_sol["Solution"]()
                        method = getattr(compiled_sol, fn_name, None)

                        results = []
                        if method:
                            for inp, out in zip(all_inputs, all_outputs):
                                parsed_inp = [json.loads(l) for l in inp.split("\n")]
                                parsed_out = json.loads(out)
                                try:
                                    pred = method(*parsed_inp)
                                    if isinstance(pred, tuple):
                                        pred = list(pred)
                                    results.append(pred == parsed_out)
                                except Exception:
                                    results.append(False)
                        result_list.append("passed" if all(results) else "failed")
                    else:
                        # standard_input with instrumentation
                        clean_code = self._clean_if_name(code)
                        wrapped = self._wrap_as_function(clean_code)
                        exec_globals = {}
                        exec(wrapped, exec_globals)
                        method = exec_globals.get("wrapped_function")

                        results = []
                        if method:
                            from unittest.mock import patch, mock_open
                            for inp, out in zip(all_inputs, all_outputs):
                                inputs_iter = iter(inp.split("\n"))
                                mock_stdin = _MockStdin(inp)
                                with _Capturing() as captured:
                                    with patch("builtins.open", mock_open(read_data=inp)):
                                        with patch("sys.stdin", mock_stdin):
                                            with patch("sys.stdin.readline", lambda *a: next(inputs_iter)):
                                                with patch("sys.stdin.readlines", lambda *a: inp.split("\n")):
                                                    with patch("sys.stdin.read", lambda *a: inp):
                                                        try:
                                                            method()
                                                        except Exception:
                                                            pass
                                pred = captured[0] if captured else ""
                                pred_lines = self._get_stripped_lines(pred)
                                gt_lines = self._get_stripped_lines(out)
                                results.append(pred_lines == gt_lines)
                        result_list.append("passed" if all(results) else "failed")

                    trace_list.append(exec_globals.get("__trace_log__", []))

                elif test_type == "lcb_call_based":
                    # LCB call_based with instrumentation
                    from tools.lcb_tester import LCBCallBasedTester
                    lcb_tester = LCBCallBasedTester()
                    # Use the non-multiprocessing test to get trace
                    result = lcb_tester.test(code, problem_info, timeout)
                    result_list.append("passed" if result.get('passed', 0) == result.get('total', 0) else "failed")
                    trace_list.append([])

                else:
                    # simple cases
                    test_cases = problem_info.get('test_cases', [])
                    cleaned_code = self._clean_code(code)
                    function_name = problem_info.get('entry_point') or self._extract_function_name(code)
                    results = []
                    for tc in test_cases:
                        r = self._run_single_test(cleaned_code, tc, 0, function_name)
                        results.append(r.get('passed', False))
                    result_list.append("passed" if all(results) else "failed")
                    trace_list.append([])

            except Exception as e:
                result_list.append(f"error: {e}")
                trace_list.append([])

        manager = multiprocessing.Manager()
        result = manager.list()
        trace = manager.list()

        p = multiprocessing.Process(
            target=_worker,
            args=(result, trace, instrumented_code, problem_info, test_type, timeout)
        )
        p.start()
        p.join(timeout=timeout + 2)

        if p.is_alive():
            p.kill()
            p.join()
            return {
                'success': False,
                'passed': 0,
                'total': 0,
                'pass_rate': 0,
                'results': [],
                'error': 'Instrumented test timed out',
                'trace': [],
                'test_method': 'instrumented',
            }

        result_str = result[0] if result else "unknown"
        trace_data = list(trace[0]) if trace else []

        # Also run the normal test to get detailed results
        normal_result = self.test_single_problem(
            generated_code, problem_info, timeout=timeout
        )

        return {
            **normal_result,
            'trace': trace_data,
            'instrumented': True,
        }

    # ------------------------------------------------------------------
    #  Unified single-problem entry point
    # ------------------------------------------------------------------

    def _detect_test_type(self, problem_info: Dict[str, Any]) -> str:
        """Auto-detect the test type from problem_info.

        Returns one of:
        - ``"unittest"``        – has a ``test`` field (HumanEval/Algorithm style)
        - ``"lcb_call_based"``  – has ``public_test_cases`` with ``testtype=="functional"`` (LCB converted)
        - ``"stdio"``           – has ``input_output`` field (LiveCodeBench style)
        - ``"simple"``          – only has ``test_cases`` (legacy)
        """
        # Explicit LCB mode is authoritative.  This prevents a ``solve``
        # stdio problem from being reclassified merely because public cases
        # also carry a function name.
        lcb_mode = str(problem_info.get('lcb_mode', '') or '').strip().lower()
        if lcb_mode == 'stdio':
            return 'stdio'
        if lcb_mode == 'call_based' and problem_info.get('input_output'):
            return 'stdio'  # input_output may include fn_name for mixed I/O

        # Check for LCB call_based format first (converted stdin problems)
        test_cases = problem_info.get('public_test_cases', [])
        if test_cases and isinstance(test_cases, list):
            first_tc = test_cases[0]
            if isinstance(first_tc, dict):
                testtype = first_tc.get('testtype', '')
                if testtype == 'functional':
                    return 'lcb_call_based'
        # Also detect by func_name + public_test_cases (merged LCB files)
        if problem_info.get('func_name') and problem_info.get('public_test_cases'):
            return 'lcb_call_based'

        if problem_info.get('test'):
            return "unittest"
        if problem_info.get('input_output'):
            return "stdio"
        return "simple"

    def test_single_problem(
        self,
        generated_code: str,
        problem_info: Dict[str, Any],
        timeout: float = 6.0,
        instrument: bool = False,
        watch_vars: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Unified single-problem testing entry point.

        Auto-detects the test type and dispatches to the appropriate
        testing method.  Optionally runs with code instrumentation.

        Args:
            generated_code: Generated Python code.
            problem_info: Problem info dict.  Supports the following keys:
                - ``test``: HumanEval-style assert test suite
                - ``input_output``: LiveCodeBench-style JSON string or dict
                  with ``inputs``, ``outputs``, and optional ``fn_name``
                - ``public_test_cases``: LCB call_based test cases (converted)
                - ``test_cases``: Legacy list of ``{input, output}`` dicts
                - ``entry_point``: Function name (for unittest mode)
            timeout: Timeout per test case in seconds.
            instrument: If True, also run with code instrumentation and
                include the execution trace in the result.
            watch_vars: Variables to watch during instrumentation.

        Returns:
            Standard test result dict.  If ``instrument=True``, an
            additional ``trace`` key is included.
        """
        if instrument:
            return self.test_with_instrumentation(
                generated_code, problem_info,
                watch_vars=watch_vars, timeout=timeout
            )

        test_type = self._detect_test_type(problem_info)

        if test_type == "unittest":
            return self.test_with_unittest_suite(
                generated_code, problem_info, timeout
            )
        elif test_type == "lcb_call_based":
            from tools.lcb_tester import LCBCallBasedTester
            tester = LCBCallBasedTester()
            return tester.test(generated_code, problem_info, timeout)
        elif test_type == "stdio":
            input_output = problem_info.get('input_output')
            if input_output is None and problem_info.get('public_test_cases'):
                from evaluation.evaluate_lcb import build_lcb_input_output
                input_output = build_lcb_input_output(problem_info, problem_info.get('public_test_cases'))
            if isinstance(input_output, str):
                input_output = json.loads(input_output)
            return self.test_with_stdio(
                generated_code, input_output, timeout
            )
        else:
            # simple cases
            use_unittest = bool(problem_info.get('test'))
            return self.test_generated_code(
                generated_code, problem_info,
                use_unittest=use_unittest, timeout=timeout
            )

    def test_with_unittest_suite(
        self,
        generated_code: str,
        problem_info: Dict[str, Any],
        timeout: float = 3.0
    ) -> Dict[str, Any]:
        """
        Test using the complete unittest test suite (same as HumanEval)
        This method executes the complete test in problem['test'] and captures detailed info for each test case

        Args:
            generated_code: Generated code
            problem_info: Problem info, should contain test field; prompt can be empty
            timeout: Timeout duration (seconds)

        Returns:
            Test result dictionary, containing:
            - passed: Whether all tests passed
            - total: Total number of tests
            - results: Detailed info for each test case
            - error: Error message (if any)
        """
        prompt = problem_info.get('prompt', '')
        test_code = problem_info.get('test', '')
        entry_point = problem_info.get('entry_point', '')

        if not test_code:
            return {
                'success': False,
                'passed': 0,
                'total': 0,
                'pass_rate': 0,
                'results': [],
                'error': 'No test suite available (missing test field in problem_info)'
            }

        # Use unittest test suite executor
        return self._run_unittest_suite(generated_code, prompt, test_code, entry_point, timeout)

    def _run_unittest_suite(
        self,
        completion: str,
        prompt: str,
        test: str,
        entry_point: str,
        timeout: float = 3.0
    ) -> Dict[str, Any]:
        """
        Run unittest test suite, same as HumanEval execution
        but captures detailed info for each test case

        Args:
            completion: Generated code
            prompt: Additional prefix code, usually empty
            test: Test suite code
            entry_point: Entry function name
            timeout: Timeout duration

        Returns:
            Test results
        """
        # Standard package imports (same as HumanEval)
        package_import = """from typing import List
from typing import List, Tuple
from typing import List, Optional
from typing import List, Any
import math
from collections import defaultdict
from functools import cache
from collections import deque
import collections
import heapq
from sortedcontainers import SortedList
from sortedcontainers import SortedDict
from itertools import accumulate
from collections import Counter
from math import floor
import bisect
import time
from datetime import datetime
"""

        # Build complete test program
        if entry_point:
            check_program = (
                package_import + "\n" +
                prompt + completion + "\n" +
                test + "\n" +
                f"check({entry_point})"
            )
        else:
            check_program = (
                package_import + "\n" +
                completion + "\n" +
                test
            )

        # Use TestResult capturer to get detailed test results
        test_results = self._execute_with_test_capture(check_program, timeout)

        return test_results

    def _execute_with_test_capture(self, program: str, timeout: float) -> Dict[str, Any]:
        """
        Execute test program and capture detailed info for each test case

        Args:
            program: Complete test program
            timeout: Timeout duration

        Returns:
            Test result dictionary
        """
        import multiprocessing
        import signal
        from contextlib import contextmanager

        class TimeoutException(Exception):
            pass

        @contextmanager
        def time_limit(seconds: float):
            def signal_handler(signum, frame):
                raise TimeoutException("Timed out!")
            signal.setitimer(signal.ITIMER_REAL, seconds)
            signal.signal(signal.SIGALRM, signal_handler)
            try:
                yield
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)

        class TestCapture:
            """Class to capture unittest test results"""
            def __init__(self):
                self.results = []

            def add_result(self, test_name: str, passed: bool, error: str = None,
                          input_args: str = None, expected: str = None, actual: str = None):
                self.results.append({
                    'test_name': test_name,
                    'passed': passed,
                    'error': error,
                    'input_args': input_args,
                    'expected': expected,
                    'actual': actual
                })

        def unsafe_execute(result_list, capture_list):
            """Execute tests in subprocess"""
            import sys
            from io import StringIO
            import contextlib

            # Redirect output
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()

            try:
                # Create custom TestResult
                import unittest

                class DetailedTestResult(unittest.TestResult):
                    def __init__(self, capture):
                        super().__init__()
                        self.capture = capture

                    def addSuccess(self, test):
                        # Try to extract parameter info from test name
                        test_name = str(test)
                        self.capture.add_result(
                            test_name=test_name,
                            passed=True,
                            input_args=self._extract_test_info(test),
                            expected="N/A",
                            actual="N/A"
                        )

                    def addFailure(self, test, err):
                        test_name = str(test)
                        error_msg = ''.join(traceback.format_exception(*err))
                        input_args, expected, actual = self._parse_assertion_error(error_msg)
                        self.capture.add_result(
                            test_name=test_name,
                            passed=False,
                            error=error_msg,
                            input_args=input_args,
                            expected=expected,
                            actual=actual
                        )

                    def addError(self, test, err):
                        test_name = str(test)
                        error_msg = ''.join(traceback.format_exception(*err))
                        self.capture.add_result(
                            test_name=test_name,
                            passed=False,
                            error=f"Test error: {error_msg}",
                            input_args=None,
                            expected=None,
                            actual=None
                        )

                    def _extract_test_info(self, test):
                        """Try to extract parameter info from test object"""
                        try:
                            test_str = str(test)
                            # Try to extract test method name
                            if '(' in test_str:
                                return test_str
                            return test_str
                        except:
                            return "N/A"

                    def _parse_assertion_error(self, error_msg):
                        """Parse AssertionError, try to extract expected and actual values"""
                        import re
                        expected = "N/A"
                        actual = "N/A"

                        # Try to match common assert patterns
                        # assert actual == expected
                        pattern1 = r"assert (.+) == (.+)"
                        match1 = re.search(pattern1, error_msg)
                        if match1:
                            actual = match1.group(1).strip()
                            expected = match1.group(2).strip()

                        # AssertionError: expected != actual
                        pattern2 = r"AssertionError: (.+) != (.+)"
                        match2 = re.search(pattern2, error_msg)
                        if match2:
                            expected = match2.group(1).strip()
                            actual = match2.group(2).strip()

                        return "N/A", expected, actual

                capture = TestCapture()

                # Execute test program
                exec_globals = {}
                with contextlib.redirect_stdout(io.StringIO()):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with time_limit(timeout):
                            exec(program, exec_globals)

                # If execution reaches here, tests passed
                result_list.append("passed")
                capture_list.append([r for r in capture.results])

            except TimeoutException:
                result_list.append("timed out")
                capture_list.append([])
            except AssertionError as e:
                result_list.append(f"failed: {e}")
                # Try to parse assertion error
                capture_list.append([{
                    'test_name': 'main',
                    'passed': False,
                    'error': str(e),
                    'input_args': 'N/A',
                    'expected': 'N/A',
                    'actual': 'N/A'
                }])
            except Exception as e:
                result_list.append(f"failed: {e}")
                capture_list.append([{
                    'test_name': 'execution',
                    'passed': False,
                    'error': f"Execution error: {str(e)}\n{traceback.format_exc()}",
                    'input_args': 'N/A',
                    'expected': 'N/A',
                    'actual': 'N/A'
                }])
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        # Execute using multiprocessing (same as HumanEval)
        manager = multiprocessing.Manager()
        result = manager.list()
        capture_results = manager.list()

        p = multiprocessing.Process(target=unsafe_execute, args=(result, capture_results))
        p.start()
        p.join(timeout=timeout + 1)

        if p.is_alive():
            p.kill()
            p.join()

        if not result:
            return {
                'success': False,
                'passed': 0,
                'total': 0,
                'pass_rate': 0,
                'results': [{
                    'test_name': 'timeout',
                    'passed': False,
                    'error': 'Test timed out',
                    'input_args': None,
                    'expected': None,
                    'actual': None
                }],
                'error': 'Test timed out'
            }

        # Parse results
        result_str = result[0]
        test_results_list = list(capture_results[0]) if capture_results else []

        # Calculate pass count
        total = len(test_results_list) if test_results_list else 1
        passed = sum(1 for r in test_results_list if r.get('passed', False))

        # If no detailed test results (e.g. simple assert error), create a default result
        if not test_results_list:
            passed = 1 if result_str == "passed" else 0
            test_results_list = [{
                'test_name': 'check',
                'passed': (result_str == "passed"),
                'error': None if result_str == "passed" else result_str,
                'input_args': 'N/A',
                'expected': 'N/A',
                'actual': 'N/A'
            }]
            total = 1

        return {
            'success': True,
            'passed': passed,
            'total': total,
            'pass_rate': (passed / total * 100) if total > 0 else 0,
            'results': test_results_list,
            'error': None if result_str == "passed" else result_str
        }


def test_code_tester():
    """Test code tester"""
    tester = CodeTester()

    # Test code
    test_code = '''
def add(a, b):
    return a + b
'''

    # Create test cases
    test_problem_info = {
        'test_cases': [
            {'input': 'add(2, 3)', 'output': '5', 'explanation': '2+3=5'},
            {'input': 'add(-1, 1)', 'output': '0', 'explanation': '-1+1=0'},
            {'input': 'add(0, 0)', 'output': '0', 'explanation': '0+0=0'}
        ]
    }

    print("=== Test Code Tester (Simple Mode) ===")

    # Test syntax validation
    print(f"Syntax validation: {tester._validate_python_syntax(test_code)}")

    # Test code (using simplified test_cases)
    result = tester.test_generated_code(test_code, test_problem_info, use_unittest=False)

    print(f"\nTest Results:")
    print(f"Success: {result['success']}")
    print(f"Passed: {result['passed']}/{result['total']}")
    print(f"Pass Rate: {result['pass_rate']:.1f}%")

    if result['error']:
        print(f"Error: {result['error']}")

    if result['results']:
        print("\nDetailed Results:")
        for r in result['results']:
            status = "PASS" if r['passed'] else "FAIL"
            print(f"Test {r['test_num']}: {status}")
            if not r['passed'] and r['error']:
                print(f"  Error: {r['error']}")


def test_unittest_mode():
    """Test unittest mode (same as HumanEval)"""
    tester = CodeTester()

    # Test code - correct implementation
    correct_code = '''
def add(a, b):
    return a + b
'''

    # Test code - wrong implementation
    wrong_code = '''
def add(a, b):
    return a - b  # Error: should be addition
'''

    # HumanEval-style problem info
    problem_info = {
        'prompt': 'def add(a, b):\n',
        'test': '''
def check(a):
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
    assert add(10, 20) == 30
''',
        'entry_point': 'add'
    }

    print("\n=== Test Unittest Mode (same as HumanEval) ===")

    # Test correct code
    print("\n--- Testing Correct Implementation ---")
    result = tester.test_generated_code(correct_code, problem_info, use_unittest=True)
    print(f"Passed: {result['passed']}/{result['total']}")
    print(f"Pass Rate: {result['pass_rate']:.1f}%")

    if result['results']:
        print("\nDetailed Results:")
        for r in result['results']:
            status = "✓" if r['passed'] else "✗"
            print(f"  {status} {r['test_name']}")
            if not r['passed'] and r.get('error'):
                print(f"    Error: {r['error'][:100]}...")

    # Test wrong code
    print("\n--- Testing Wrong Implementation ---")
    result = tester.test_generated_code(wrong_code, problem_info, use_unittest=True)
    print(f"Passed: {result['passed']}/{result['total']}")
    print(f"Pass Rate: {result['pass_rate']:.1f}%")

    if result['results']:
        print("\nDetailed Results:")
        for r in result['results']:
            status = "✓" if r['passed'] else "✗"
            print(f"  {status} {r['test_name']}")
            if not r['passed']:
                if r.get('error'):
                    print(f"    Error: {r['error'][:200]}...")


def test_stdio_mode():
    """Test stdin/stdout mode (LiveCodeBench style)"""
    tester = CodeTester()

    # --- standard_input (no fn_name) ---
    stdio_code = """
n = int(input())
arr = list(map(int, input().split()))
print(sum(arr))
"""

    stdio_problem = {
        'input_output': {
            'inputs': ['3\n1 2 3', '5\n10 20 30 40 50', '1\n42'],
            'outputs': ['6', '150', '42'],
        }
    }

    print("\n=== Test Stdio Mode (standard_input) ===")
    result = tester.test_single_problem(stdio_code, stdio_problem)
    print(f"Method: {result.get('test_method', 'N/A')}")
    print(f"Passed: {result['passed']}/{result['total']}")
    print(f"Pass Rate: {result['pass_rate']:.1f}%")
    for r in result.get('results', []):
        status = "PASS" if r['passed'] else "FAIL"
        print(f"  Test {r['test_num']}: {status}")
        if not r['passed']:
            print(f"    Input:    {r['input']}")
            print(f"    Expected: {r['expected']}")
            print(f"    Actual:   {r['actual']}")
            print(f"    Error:    {r.get('error', '')}")

    # --- call_based (with fn_name) ---
    call_based_code = """
class Solution:
    def twoSum(self, nums, target):
        seen = {}
        for i, n in enumerate(nums):
            if target - n in seen:
                return [seen[target - n], i]
            seen[n] = i
"""

    call_based_problem = {
        'input_output': {
            'fn_name': 'twoSum',
            'inputs': ['[2,7,11,15]\n9', '[3,2,4]\n6', '[3,3]\n6'],
            'outputs': ['[0,1]', '[1,2]', '[0,1]'],
        }
    }

    print("\n=== Test Stdio Mode (call_based) ===")
    result = tester.test_single_problem(call_based_code, call_based_problem)
    print(f"Method: {result.get('test_method', 'N/A')}")
    print(f"Passed: {result['passed']}/{result['total']}")
    print(f"Pass Rate: {result['pass_rate']:.1f}%")
    for r in result.get('results', []):
        status = "PASS" if r['passed'] else "FAIL"
        print(f"  Test {r['test_num']}: {status}")
        if not r['passed']:
            print(f"    Expected: {r['expected']}")
            print(f"    Actual:   {r['actual']}")


def test_instrumentation():
    """Test code instrumentation mode"""
    tester = CodeTester()

    # Simple code with a bug
    buggy_code = """
def multiply(a, b):
    result = a + b  # Bug: should be a * b
    return result
"""

    problem_info = {
        'test': '''
def check(candidate):
    assert candidate(2, 3) == 6
    assert candidate(4, 5) == 20
    assert candidate(0, 100) == 0
''',
        'entry_point': 'multiply'
    }

    print("\n=== Test Instrumentation Mode ===")

    # First show the instrumented code
    instrumented = tester.instrument_code(buggy_code, watch_vars=['a', 'b', 'result'])
    print("\n--- Instrumented Code (preview) ---")
    print(instrumented[:500] + "..." if len(instrumented) > 500 else instrumented)

    # Run with instrumentation
    result = tester.test_single_problem(
        buggy_code, problem_info,
        instrument=True,
        watch_vars=['a', 'b', 'result']
    )

    print(f"\nPassed: {result['passed']}/{result['total']}")
    print(f"Pass Rate: {result['pass_rate']:.1f}%")

    trace = result.get('trace', [])
    print(f"\nExecution Trace ({len(trace)} entries):")
    for i, entry in enumerate(trace[:10]):  # Show first 10 entries
        print(f"  [{i}] Line {entry['line']}: vars = {entry['variables']}")
    if len(trace) > 10:
        print(f"  ... ({len(trace) - 10} more entries)")


if __name__ == "__main__":
    test_code_tester()
    test_unittest_mode()
    test_stdio_mode()
    test_instrumentation()
