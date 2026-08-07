"""Guard against a broken/missing lerobot silently vanishing from the
differential suite's coverage.

Every test in `test_action_queue_differential.py` goes through
`pytest.importorskip("torch")` / `pytest.importorskip("lerobot...")` at
module level. Verified empirically (rename `torch` out of site-packages,
run `uv run pytest -m differential -v`): today that produces
"collected 350 items / 350 deselected / 1 skipped / 0 selected" and pytest
exit code 5 ("no tests collected"), which happens to fail the CI job -- but
only because zero tests currently match `-m differential` without it. The
moment a second differential-marked test exists that does not itself need
lerobot (this file, for instance, if it also skipped), that "0 selected"
becomes "1 selected, 1 passed", exit code 0: a broken install would then
report the differential job as green while the actual upstream comparison
silently never ran.

This module does a real, unguarded `import lerobot` / `import torch` --
no `importorskip` -- so a missing or broken install fails collection
outright (pytest exit code 2, "error during collection") regardless of
what any sibling differential test file does.
"""

import importlib.metadata
import json

import pytest

pytestmark = pytest.mark.differential

import lerobot  # noqa: E402 -- deliberately a hard import, not importorskip
import torch  # noqa: E402


def test_lerobot_and_torch_report_a_version():
    assert lerobot.__version__, "lerobot imported but reports no __version__"
    assert torch.__version__, "torch imported but reports no __version__"


def test_lerobot_is_installed_from_a_git_ref_not_pypi():
    """Both CI legs (.github/workflows/differential.yml) install lerobot
    via `git+https://github.com/huggingface/lerobot@<ref>` -- pinned SHA
    or `main`. Either way, pip/uv record the resolved commit in
    direct_url.json. A plain PyPI or path install -- e.g. from a
    misconfigured lockfile -- has no such record, and would mean this
    suite is not actually comparing against the intended upstream source.
    Deliberately does not assert the exact pinned SHA: the 'main' leg is
    supposed to resolve to a different, moving commit.
    """
    dist = importlib.metadata.distribution("lerobot")
    raw = dist.read_text("direct_url.json")
    assert raw, (
        "lerobot has no direct_url.json -- it wasn't installed from a git "
        "ref (see pyproject.toml's lerobot extra, or the 'main' leg of "
        ".github/workflows/differential.yml)"
    )
    direct_url = json.loads(raw)
    commit_id = direct_url.get("vcs_info", {}).get("commit_id")
    assert commit_id, f"lerobot's direct_url.json has no vcs_info.commit_id: {direct_url!r}"
    assert len(commit_id) == 40, f"commit_id doesn't look like a full git SHA: {commit_id!r}"
