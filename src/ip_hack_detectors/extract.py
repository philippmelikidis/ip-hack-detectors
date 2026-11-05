"""Code extraction as in the paper: first Python block, otherwise the raw text."""

from __future__ import annotations

import re

_PY_BLOCK = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    m = _PY_BLOCK.search(text)
    return m.group(1) if m else text
