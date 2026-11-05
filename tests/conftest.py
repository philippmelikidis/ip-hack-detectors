from pathlib import Path

import pytest

from ip_hack_detectors.config import SandboxConfig
from ip_hack_detectors.tasks import load_jsonl_tasks

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def tasks():
    return {t.task_id: t for t in load_jsonl_tasks(FIXTURES / "tasks.jsonl")}


@pytest.fixture
def sandbox():
    return SandboxConfig(timeout_s=2.0, memory_mb=1024)
