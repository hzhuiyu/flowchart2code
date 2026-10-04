import ast
import json
import signal
import astunparse
from .executor_utils import function_with_timeout
from .subprocess_runner import run_worker
from typing import List
from .executor_types import ExecuteResult, Executor


class PyExecutor(Executor):
    def execute(self, func: str, tests: List[str], timeout: int = 1) -> ExecuteResult:
        print("|| Begin Executing...")
        # Combine function code and assert statement
        imports = 'from typing import *'
        func_test_list = [f'{imports}\n{func}\n{test}' for test in tests]

        # Run the tests and collect the results
        success_tests = []
        failed_tests = []
        is_passing = True
        num_tests = len(func_test_list)
        for i in range(num_tests):
            try:
                function_with_timeout(exec, (func_test_list[i], globals()), timeout)
                success_tests += [tests[i]]
            except Exception:
                output = get_output(func, tests[i], timeout=timeout)
                failed_tests += [f"{tests[i]} # Real Execution Output: {output}"]
                is_passing = False

        state = []
        print("|| End Executing...")
        return ExecuteResult(is_passing, failed_tests, state)

    def evaluate(self, name: str, func: str, test: str, timeout: int = 1) -> bool:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """
        if "def check(" in (test or ""):
            code = f"""{func}

{test}

check({name})
    """
        else:
            # Algorithm / MATH (and similar) ship raw asserts, not a check().
            code = f"""{func}

{test}
    """
        try:

            function_with_timeout(exec, (code, globals()), timeout)

            return True
        except Exception:
            return False


class SubprocessPyExecutor(PyExecutor):
    """PyExecutor with subprocess isolation for hard-kill timeouts.

    Model-generated solutions can contain infinite loops that survive every
    in-process timeout mechanism on Windows (thread terminate / signal alarm).
    This executor delegates execution to a worker subprocess that is killed
    from the outside when it exceeds the time budget.
    """

    def execute(self, func: str, tests: List[str], timeout: int = 10) -> ExecuteResult:
        print("|| Begin Executing...")
        result = run_worker(
            {"mode": "execute", "func": func, "tests": list(tests)},
            timeout=max(timeout, 10) * max(1, len(tests)) + 15,
        )
        if result["results"] is not None:
            data = result["results"]
            print("|| End Executing...")
            return ExecuteResult(
                data["is_passing"], data["feedback"], tuple(data["state"])
            )
        print("|| End Executing (worker error)...")
        return ExecuteResult(
            False,
            [f"{t} # worker error: {result['error']}" for t in tests],
            (),
        )

    def evaluate(self, name: str, func: str, test: str, timeout: int = 10) -> bool:
        # Build the evaluation code the same way PyExecutor does, then ship it
        # to the worker as an "eval_code" payload.
        if "def check(" in (test or ""):
            code = f"""{func}

{test}

check({name})
    """
        else:
            code = f"""{func}

{test}
    """
        result = run_worker(
            {"mode": "eval_code", "code": code},
            timeout=timeout + 15,
        )
        if result["results"] is not None:
            return bool(result["results"])
        return False


def get_call_str(assert_statement: str) -> str:
    ast_parsed = ast.parse(assert_statement)
    try:
        call_str = ast_parsed.body[0].test.left # type: ignore
    except:
        call_str = ast_parsed.body[0].test # type: ignore

    return astunparse.unparse(call_str).strip()

def get_output(func: str, assert_statement: str, timeout: int = 1) -> str:
    try:
        exec(f"from typing import *\n{func}", globals())
        func_call = get_call_str(assert_statement)
        output = function_with_timeout(eval, (func_call, globals()), timeout)
        return output
    except TimeoutError:
        return "TIMEOUT"
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    pass
    # Test the function
    func = "def add(a, b):\n    while True:\n        x = 1\n    return a + b"
    tests = ["assert add(1, 2) == 3", "assert add(1, 2) == 4"]
    print(PyExecutor().execute(func, tests, timeout=1))
