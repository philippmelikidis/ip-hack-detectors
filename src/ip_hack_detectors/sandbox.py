"""Runs generated code in a subprocess (best effort, no Docker).

`run_test` runs the code with one test. `run_calls` (D2) calls one function of the code on many
inputs in one subprocess: every call gets a deep copy of its input and its own timer, results
are streamed back line by line under a random token, values cross as literals, and inputs
without a result after the process died or was killed count as timeouts (PREREGISTRATION.md
addendum A3).

Protection: timeout that kills the whole process group, CPU, memory and file-size limits
(the memory limit only applies on Linux), temporary working directory, empty environment,
blocked sockets. A test only counts as passed if the wrapper prints a random token after the
assert; an early os._exit(0) therefore counts as a crash. The code runs in the same
interpreter as the wrapper, so code that deliberately inspects the wrapper could reach it;
this guards against accidents, not against an adversary.
"""

from __future__ import annotations

import ast
import math
import os
import secrets
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

from .config import SandboxConfig

TestStatus = Literal["passed", "failed", "error", "timeout", "crash"]

_WRAPPER = """\
import sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
    resource.setrlimit(resource.RLIMIT_FSIZE, (10_000_000, 10_000_000))
    try:
        resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
    except (ValueError, OSError):
        pass
except ImportError:
    pass
import socket as _socket
def _no_network(*args, **kwargs):
    raise OSError("network disabled in sandbox")
_socket.socket = _no_network
_socket.create_connection = _no_network
_globals = {{"__name__": "__main__"}}
try:
    exec(compile({program!r}, "solution.py", "exec"), _globals)
except AssertionError:
    sys.exit(1)
except BaseException:
    sys.exit(2)
sys.stdout.write("\\n{nonce}\\n")
sys.stdout.flush()
"""


def run_test(code: str, imports: tuple[str, ...], test: str, cfg: SandboxConfig) -> TestStatus:
    program = "\n".join([*imports, code, test]) + "\n"
    nonce = secrets.token_hex(16)
    script = _WRAPPER.format(
        cpu=math.ceil(cfg.timeout_s) + 1,
        mem=cfg.memory_mb * 1024 * 1024,
        program=program,
        nonce=nonce,
    )
    with tempfile.TemporaryDirectory(prefix="ipd-sbx-") as tmp:
        path = Path(tmp) / "main.py"
        path.write_text(script, encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", str(path)],
            cwd=tmp,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "HOME": tmp},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            out, _ = proc.communicate(timeout=cfg.timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            return "timeout"
    if proc.returncode == 0 and out.rstrip().endswith(nonce.encode()):
        return "passed"
    if proc.returncode == 1:
        return "failed"
    if proc.returncode == 2:
        return "error"
    if proc.returncode == -signal.SIGXCPU:
        return "timeout"
    return "crash"


_CALLS_WRAPPER = """\
import sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
    resource.setrlimit(resource.RLIMIT_FSIZE, (10_000_000, 10_000_000))
    try:
        resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
    except (ValueError, OSError):
        pass
except ImportError:
    pass
import ast, copy, os, signal
import socket as _socket
def _no_network(*args, **kwargs):
    raise OSError("network disabled in sandbox")
_socket.socket = _no_network
_socket.create_connection = _no_network
_out = os.fdopen(os.dup(1), "w")
_null = open(os.devnull, "w")
os.dup2(_null.fileno(), 1)
sys.stdout = _null
class _Timeout(BaseException):
    pass
def _alarm(signum, frame):
    raise _Timeout()
signal.signal(signal.SIGALRM, _alarm)
def _emit(i, kind, text=""):
    _out.write({nonce!r} + "\\t" + str(i) + "\\t" + kind + "\\t" + text + "\\n")
    _out.flush()
_inputs = ast.literal_eval({inputs!r})
_globals = {{"__name__": "__main__"}}
_func = None
try:
    signal.setitimer(signal.ITIMER_REAL, {exec_timeout})
    exec(compile({program!r}, "solution.py", "exec"), _globals)
    signal.setitimer(signal.ITIMER_REAL, 0)
    _func = _globals[{func_name!r}]
except BaseException:
    signal.setitimer(signal.ITIMER_REAL, 0)
for _i, _args in enumerate(_inputs):
    if _func is None:
        _emit(_i, "raised")
        continue
    _kind, _text = "raised", ""
    try:
        _call_args = copy.deepcopy(_args)
        signal.setitimer(signal.ITIMER_REAL, {call_timeout})
        try:
            _value = _func(*_call_args)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        _text = repr(_value)
        if len(_text) <= {max_repr}:
            _parsed = ast.literal_eval(_text)
            _same = type(_parsed) is type(_value) and _parsed == _value
            if _same and len(repr(_parsed)) == len(_text):
                _kind = "ok"
    except _Timeout:
        _kind = "timeout"
    except BaseException:
        pass
    _emit(_i, _kind, _text if _kind == "ok" else "")
"""

CallResult = tuple[Literal["ok", "raised", "timeout"], Any]


def run_calls(
    code: str,
    imports: tuple[str, ...],
    func_name: str,
    inputs: list[tuple],
    cfg: SandboxConfig,
    *,
    setup: str = "",
    call_timeout_s: float = 2.0,
    exec_timeout_s: float = 10.0,
    batch_timeout_s: float | None = None,
    max_repr: int = 100_000,
) -> list[CallResult]:
    """Call `func_name` of `code` on every input in ONE subprocess.

    Memory, file-size and network limits are those of `run_test`; the time limits come from
    `call_timeout_s` and `batch_timeout_s` (the CPU limit follows the batch timeout), not from
    `cfg.timeout_s`.

    Every call gets a deep copy of its input and its own `call_timeout_s`, measured inside the
    process with a timer signal. Results are streamed line by line under a random token, so that
    prints of the code cannot forge them and a hard kill keeps what was already computed.
    The whole process is killed after `batch_timeout_s` (default: all calls at their full
    timeout plus the time to define the code); inputs without a result then count as "timeout".

    Per input: ("ok", value) where value was transported as repr and `ast.literal_eval` and gives
    an equal value of the same type; ("raised", None) when the call raised, the function is
    missing, or the value cannot be transported; ("timeout", None).
    """
    program = "\n".join([*imports, setup, code]) + "\n"
    nonce = secrets.token_hex(16)
    if batch_timeout_s is None:
        batch_timeout_s = len(inputs) * call_timeout_s + exec_timeout_s + 5
    script = _CALLS_WRAPPER.format(
        cpu=math.ceil(batch_timeout_s) + 1,
        mem=cfg.memory_mb * 1024 * 1024,
        nonce=nonce,
        inputs=repr(inputs),
        program=program,
        func_name=func_name,
        exec_timeout=exec_timeout_s,
        call_timeout=call_timeout_s,
        max_repr=max_repr,
    )
    with tempfile.TemporaryDirectory(prefix="ipd-sbx-") as tmp:
        path = Path(tmp) / "main.py"
        path.write_text(script, encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", str(path)],
            cwd=tmp,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "HOME": tmp},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            out, _ = proc.communicate(timeout=batch_timeout_s)
        except subprocess.TimeoutExpired:
            out = _kill_group(proc)
    results: list[CallResult] = [("timeout", None)] * len(inputs)
    prefix = nonce + "\t"
    for line in out.decode("utf-8", errors="replace").splitlines():
        if not line.startswith(prefix):
            continue
        parts = line[len(prefix) :].split("\t", 2)
        index = parts[0] if len(parts) == 3 else ""
        if not (index.isascii() and index.isdigit()) or int(index) >= len(inputs):
            continue
        idx, kind, text = int(parts[0]), parts[1], parts[2]
        if kind == "ok":
            try:
                results[idx] = ("ok", ast.literal_eval(text))
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                results[idx] = ("raised", None)
        elif kind in ("raised", "timeout"):
            results[idx] = (kind, None)
    return results


def _kill_group(proc: subprocess.Popen) -> bytes:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    out, _ = proc.communicate()
    return out
