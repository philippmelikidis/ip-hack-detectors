"""Hand-written D2 input strategies per task (PREREGISTRATION.md section 7).

Maps a task id to a zero-argument function that returns a Hypothesis strategy producing the
whole argument tuple of the tested function, for example
`lambda: st.tuples(st.lists(st.integers(0, 9), min_size=1), st.integers(1, 5))`.
Strategies are written from task text, tests and reference solution only, never from model
outputs. Every drawn value must survive `repr` and `ast.literal_eval` (no `inf`, `nan`,
frozensets or objects): the inputs travel to the sandbox that way, and one value that does not
makes the whole task not covered. The file is frozen with the rest of the detector code at the
tag `detectors-v1`.
"""

from __future__ import annotations

from collections.abc import Callable

from hypothesis.strategies import SearchStrategy

OVERRIDES: dict[int, Callable[[], SearchStrategy]] = {}
