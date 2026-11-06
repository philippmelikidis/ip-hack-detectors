import os
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


def pytest_collection_modifyitems(config, items):
    """Slow tests (real small model) run only with IPD_SLOW=1, gpu tests only with IPD_GPU=1."""
    for marker, var in (("slow", "IPD_SLOW"), ("gpu", "IPD_GPU")):
        if os.environ.get(var) == "1":
            continue
        skip = pytest.mark.skip(reason=f"{marker} test: set {var}=1")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
