"""Runs generated code with one test in its own subprocess (best effort, no Docker).

Protection: timeout that kills the whole process group, CPU, memory and file-size limits
(the memory limit only applies on Linux), temporary working directory, empty environment,
blocked sockets. A test only counts as passed if the wrapper prints a random token after the
assert; an early os._exit(0) therefore counts as a crash.
"""

from __future__ import annotations

import math
import os
import secrets
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

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


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.communicate()
