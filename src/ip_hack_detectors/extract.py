"""Code-Extraktion wie im Paper: erster Python-Block, sonst der rohe Text."""

from __future__ import annotations

import re

_PY_BLOCK = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    m = _PY_BLOCK.search(text)
    return m.group(1) if m else text
