from .py_executor import PyExecutor, SubprocessPyExecutor
from .lcb_py_executor import LCBPyExecutor
from .executor_types import Executor

def executor_factory(lang: str, is_leet: bool = False, dataset_type: str = "") -> Executor:
    if lang == "py" or lang == "python":
        if dataset_type == "LiveCodeBench":
            return LCBPyExecutor()
        # Use subprocess isolation for all datasets so infinite-loop
        # solutions are hard-killed instead of hanging the process.
        return SubprocessPyExecutor()
    else:
        raise ValueError(f"Invalid language for executor: {lang}")
