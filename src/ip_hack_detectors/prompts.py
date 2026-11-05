"""Prompt nach dem Template des Papers (Wichers et al., App. C)."""

from __future__ import annotations

import hashlib
import json

from .tasks import Task

TEMPLATE = (
    "Write a Python function to solve this problem. {prefix}Return only the code, no other text:"
    "\n\n{prompt}\n\n## Test Case:\n```python\n{first_test_case}\n```"
)


def user_message(task: Task, prefix: str) -> str:
    prefix = prefix.strip()
    return TEMPLATE.format(
        prefix=f"{prefix} " if prefix else "",
        prompt=task.prompt,
        first_test_case=task.first_test,
    )


def build_messages(task: Task, prefix: str) -> list[dict[str, str]]:
    """Chat-Nachrichten; das Chat-Template des Modells wendet das Backend an."""
    return [{"role": "user", "content": user_message(task, prefix)}]


def prompt_hash(messages: list[dict[str, str]]) -> str:
    blob = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
