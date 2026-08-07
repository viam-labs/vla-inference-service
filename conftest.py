"""Root conftest: keep test_differential_guard.py's hard import from
breaking collection for runs that were never going to select it anyway.

`tests/controller/test_differential_guard.py` does a deliberate, unguarded
`import lerobot` / `import torch` at module level (no `importorskip`) so
that `-m differential` fails LOUDLY -- a collection error, not a silent
skip -- if lerobot/torch are missing or broken. Its own docstring explains
why: every other differential test uses `importorskip`, so without this
file, a broken install would make the whole differential suite report
"0 selected, exit code 5" today and silently "1 selected, 1 passed" the
moment a second lerobot-independent differential test exists -- a green CI
job for a comparison that never ran.

The problem: pytest's marker deselection (`-m "not differential"`) happens
*after* collection -- every test module is imported first, mark filtering
applied second. So this file's hard import also breaks `mise run test`
(`-m "not integration and not differential"`) and the Task 21 no-torch
check (`uv sync` with no extra, then that same `-m` filter) even though
neither run wants a single differential test to execute. Verified: with
`lerobot`/`torch` uninstalled, plain `uv run pytest -m "not integration and
not differential"` fails at collection with
`ModuleNotFoundError: No module named 'lerobot'`, before a single test
runs -- the exact failure this hook exists to prevent.

This hook restores both properties: if lerobot/torch import cleanly, this
file collects and runs exactly as before (nothing changes for CI, which
always `uv sync --extra lerobot`s first). If they do not import, the file
is collected only when the invocation's marker expression could plausibly
select a `differential`-marked test -- so `-m differential` (or no `-m` at
all) still fails hard on a broken install, while a filter that positively
excludes differential tests (`not differential` as a token) skips this
file's collection instead of crashing the entire run over a dependency
none of the selected tests need.
"""

from __future__ import annotations

import re


def pytest_ignore_collect(collection_path, config):
    if collection_path.name != "test_differential_guard.py":
        return None

    try:
        import lerobot  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        return None  # both present -- collect normally, nothing to guard against

    markexpr = config.getoption("markexpr", default="") or ""
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", markexpr)
    if "differential" not in tokens:
        # No mention of "differential" at all: either no -m filter (every
        # test, including this file's, would run) or a filter naming only
        # unrelated markers. The former genuinely wants this file; the
        # latter is a narrower corner case this project's real invocations
        # (mise.toml, .github/workflows/differential.yml) never produce --
        # accepting it keeps this hook simple and its false-negative side
        # (a spurious hard failure) safe rather than silent.
        return None

    idx = tokens.index("differential")
    negated = idx > 0 and tokens[idx - 1] == "not"
    if negated:
        return True  # explicitly excluded: skip collection, let the run proceed
    return None  # explicitly or ambiguously included: collect, and fail loudly if broken
