from typing import Optional, Callable, Dict, List
import ast
import contextlib
import ctypes
import faulthandler
import io
import os
import multiprocessing
import platform
import signal
import tempfile
import threading


_INPROCESS_LOCK = threading.Lock()
_WARNED_LINUX_FALLBACK = False
_FORCE_INPROCESS = False
_FORCE_INPROCESS_LOCK = threading.Lock()


def _is_linux_isolation_failure(exc: BaseException) -> bool:
    """True when the official Process+SIGALRM path cannot start (typical on Windows)."""
    if isinstance(exc, (NotImplementedError, OSError)):
        return True
    msg = str(exc).lower()
    if isinstance(exc, AttributeError):
        return any(
            token in msg
            for token in ("pickle", "local object", "setitimer", "sigalrm", "itimer")
        )
    if isinstance(exc, RuntimeError):
        return any(
            token in msg
            for token in ("pickle", "multiprocessing", "start a new process")
        )
    return False


def _warn_linux_fallback(exc: BaseException) -> None:
    global _WARNED_LINUX_FALLBACK
    if _WARNED_LINUX_FALLBACK:
        return
    _WARNED_LINUX_FALLBACK = True
    print(
        f"[human_eval] Linux process isolation failed ({type(exc).__name__}: {exc}); "
        "falling back to in-process Windows timeout"
    )


def _async_raise_timeout(tid: int) -> None:
    if not isinstance(tid, int):
        return
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid), ctypes.py_object(TimeoutException)
    )
    if res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)


def _build_check_program(problem: Dict, completion: str) -> str:
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
from sympy import *
import hashlib"""

    if "entry_point" in problem:
        return (
            package_import + "\n" +
            problem["prompt"] + completion + "\n" +
            problem["test"] + "\n" +
            f"check({problem['entry_point']})"
        )
    return (
        package_import + "\n" +
        completion + "\n" +
        problem["test"]
    )


def _snapshot_reliability_guard() -> dict:
    """Capture interpreter state that reliability_guard() mutates.

    Linux isolation runs in a child process, so leaks do not matter there.
    The Windows in-process fallback must restore this, or later evals and
    subprocess.run() in the sequential runner break.
    """
    import builtins
    import shutil
    import subprocess
    import sys

    os_keys = (
        "kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir",
        "setuid", "fork", "forkpty", "killpg", "rename", "renames", "truncate",
        "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot",
        "lchflags", "lchmod", "lchown", "getcwd", "chdir",
    )
    help_fn = (
        __builtins__.get("help") if isinstance(__builtins__, dict)
        else getattr(__builtins__, "help", None)
    )
    return {
        "os": {k: getattr(os, k, None) for k in os_keys},
        "shutil": {k: getattr(shutil, k, None) for k in ("rmtree", "move", "chown")},
        "subprocess.Popen": subprocess.Popen,
        "builtins.exit": getattr(builtins, "exit", None),
        "builtins.quit": getattr(builtins, "quit", None),
        "help": help_fn,
        "modules": {
            k: sys.modules.get(k)
            for k in ("ipdb", "joblib", "resource", "psutil", "tkinter")
        },
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "faulthandler": faulthandler.is_enabled(),
    }


def _restore_reliability_guard(snap: dict) -> None:
    import builtins
    import shutil
    import subprocess
    import sys

    for k, v in snap["os"].items():
        setattr(os, k, v)
    for k, v in snap["shutil"].items():
        setattr(shutil, k, v)
    subprocess.Popen = snap["subprocess.Popen"]
    builtins.exit = snap["builtins.exit"]
    builtins.quit = snap["builtins.quit"]
    if isinstance(__builtins__, dict):
        __builtins__["help"] = snap["help"]
    else:
        try:
            __builtins__.help = snap["help"]
        except Exception:
            pass
    for k, v in snap["modules"].items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v
    if snap["OMP_NUM_THREADS"] is None:
        os.environ.pop("OMP_NUM_THREADS", None)
    else:
        os.environ["OMP_NUM_THREADS"] = snap["OMP_NUM_THREADS"]
    if snap["faulthandler"] and not faulthandler.is_enabled():
        try:
            faulthandler.enable()
        except Exception:
            pass


def _execute_check(problem: Dict, completion: str, timeout: float, result: List) -> None:
    with create_tempdir():
        import shutil

        snap = _snapshot_reliability_guard()
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        try:
            # Disable functionalities that can make destructive changes to the test.
            reliability_guard()

            check_program = _build_check_program(problem, completion)

            try:
                exec_globals = {}
                with swallow_io():
                    with time_limit(timeout):
# WARNING
# This program exists to execute untrusted model-generated code. Although
# it is highly unlikely that model-generated code will do something overtly
# malicious in response to this test suite, model-generated code may act
# destructively due to a lack of model capability or alignment.
# Users are strongly encouraged to sandbox this evaluation suite so that it
# does not perform destructive actions on their host or network. For more
# information on how OpenAI sandboxes its code, see the accompanying paper.
# Once you have read this disclaimer and taken appropriate precautions,
# uncomment the following line and proceed at your own risk:
                        exec(check_program, exec_globals)
                result.append("passed")
            except TimeoutException:
                result.append("timed out")
            except AssertionError as e:
                result.append(f"failed: {e}")
            except Exception as e:
                result.append(f"failed: {e}")
            except BaseException as e:
                result.append(f"failed: {e}")
        finally:
            # Needed for cleaning up, and to keep the parent interpreter usable
            # when this ran in-process (Windows fallback).
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir
            _restore_reliability_guard(snap)


def _finish_check_result(problem: Dict, result: List, completion_id: Optional[int]) -> Dict:
    if not result:
        result.append("timed out")
    return dict(
        task_id=problem["task_id"],
        passed=result[0] == "passed",
        result=result[0],
        completion_id=completion_id,
    )


def _check_correctness_inprocess(problem: Dict, completion: str, timeout: float,
                                 completion_id: Optional[int] = None) -> Dict:
    """Windows fallback: in-process exec + thread timeout (same idea as evaluate_lcb.py)."""
    result: List = []
    # reliability_guard() mutates process-global os/shutil; serialize workers.
    with _INPROCESS_LOCK:
        try:
            _execute_check(problem, completion, timeout, result)
        except Exception as e:
            if not result:
                result.append(f"failed: {e}")
    return _finish_check_result(problem, result, completion_id)


def _unsafe_execute_worker(problem: Dict, completion: str, timeout: float, result: List) -> None:
    """Module-level Linux worker so multiprocessing spawn can pickle it."""
    _execute_check(problem, completion, timeout, result)


def check_correctness(problem: Dict, completion: str, timeout: float,
                      completion_id: Optional[int] = None) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.

    Prefer the original Linux method: multiprocessing.Process isolation +
    SIGALRM/setitimer timeout. If that cannot start (Windows spawn cannot
    pickle the nested worker, or setitimer is missing), fall back to the
    in-process Windows timeout used by flowchart2code's LCB evaluator.

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """

    global _FORCE_INPROCESS
    if _FORCE_INPROCESS:
        return _check_correctness_inprocess(problem, completion, timeout, completion_id)

    manager = None
    try:
        manager = multiprocessing.Manager()
        result = manager.list()
        p = multiprocessing.Process(
            target=_unsafe_execute_worker,
            args=(problem, completion, timeout, result),
        )
        p.start()
    except Exception as exc:
        if manager is not None:
            try:
                manager.shutdown()
            except Exception:
                pass
        if not _is_linux_isolation_failure(exc):
            raise
        with _FORCE_INPROCESS_LOCK:
            _FORCE_INPROCESS = True
        _warn_linux_fallback(exc)
        return _check_correctness_inprocess(problem, completion, timeout, completion_id)

    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    return _finish_check_result(problem, result, completion_id)


@contextlib.contextmanager
def time_limit(seconds: float):
    """Prefer Linux SIGALRM/setitimer; fall back to interrupting this thread."""
    can_itimer = (
        hasattr(signal, "setitimer")
        and hasattr(signal, "ITIMER_REAL")
        and hasattr(signal, "SIGALRM")
    )
    if can_itimer:
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        try:
            signal.setitimer(signal.ITIMER_REAL, seconds)
            signal.signal(signal.SIGALRM, signal_handler)
        except (AttributeError, ValueError, OSError):
            can_itimer = False
        else:
            try:
                yield
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            return

    tid = threading.get_ident()
    timer = threading.Timer(seconds, lambda: _async_raise_timeout(tid))
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """ StringIO that throws an exception when it's read from """

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """ Returns True if the IO object can be read. """
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the 
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == 'Darwin':
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins
    builtins.exit = None
    builtins.quit = None

    import os
    os.environ['OMP_NUM_THREADS'] = '1'

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess
    subprocess.Popen = None  # type: ignore

    __builtins__['help'] = None

    import sys
    sys.modules['ipdb'] = None
    sys.modules['joblib'] = None
    sys.modules['resource'] = None
    sys.modules['psutil'] = None
    sys.modules['tkinter'] = None
