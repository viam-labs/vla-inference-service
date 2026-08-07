import logging
import os
import sys
import tempfile
from pathlib import Path

import pytest

from vla.policy.config import PolicyConfig
from vla.policy.resolver import resolve_checkpoint, ResolveError

REQUIRED = ["config.json", "model.safetensors"]


def _make_checkpoint(tmp_path):
    for name in REQUIRED:
        (tmp_path / name).write_text("{}")
    return tmp_path


# ---------------------------------------------------------------------------
# Local path resolution
# ---------------------------------------------------------------------------


def test_resolves_local_path(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    cfg = PolicyConfig.parse({"model_path": str(ckpt)})
    assert resolve_checkpoint(cfg) == str(ckpt)


def test_missing_local_path_errors(tmp_path):
    cfg = PolicyConfig.parse({"model_path": str(tmp_path / "nope")})
    with pytest.raises(ResolveError, match="does not exist"):
        resolve_checkpoint(cfg)


def test_local_path_missing_config_json_errors(tmp_path):
    (tmp_path / "model.safetensors").write_text("{}")
    cfg = PolicyConfig.parse({"model_path": str(tmp_path)})
    with pytest.raises(ResolveError, match="config.json"):
        resolve_checkpoint(cfg)


def test_local_path_missing_model_safetensors_errors(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    cfg = PolicyConfig.parse({"model_path": str(tmp_path)})
    with pytest.raises(ResolveError, match="model.safetensors"):
        resolve_checkpoint(cfg)


def test_local_path_missing_both_required_files_names_both(tmp_path):
    # An operator who fixes only the first reported error should not have
    # to re-run and discover a second one.
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    cfg = PolicyConfig.parse({"model_path": str(empty_dir)})
    with pytest.raises(ResolveError) as excinfo:
        resolve_checkpoint(cfg)
    assert "config.json" in str(excinfo.value)
    assert "model.safetensors" in str(excinfo.value)


def test_local_path_that_is_a_file_gets_distinct_error(tmp_path):
    # A file (not a directory) at model_path currently falls through
    # is_dir() -> False and would misreport "does not exist" -- it exists,
    # it's just the wrong kind of thing.
    a_file = tmp_path / "checkpoint.bin"
    a_file.write_text("not a directory")
    cfg = PolicyConfig.parse({"model_path": str(a_file)})
    with pytest.raises(ResolveError) as excinfo:
        resolve_checkpoint(cfg)
    message = str(excinfo.value)
    assert "not a directory" in message or "is a file" in message
    assert "does not exist" not in message


# ---------------------------------------------------------------------------
# Hugging Face hub resolution
# ---------------------------------------------------------------------------


def test_hub_download_uses_module_data_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(_make_checkpoint(tmp_path / "dl"))

    cfg = PolicyConfig.parse(
        {"model_hub_id": "lerobot/smolvla_base", "model_revision": "abc123"}
    )
    (tmp_path / "dl").mkdir()
    out = resolve_checkpoint(cfg, snapshot_download=fake_snapshot_download)

    assert captured["repo_id"] == "lerobot/smolvla_base"
    assert captured["revision"] == "abc123"
    assert str(tmp_path) in captured["cache_dir"]
    assert out.endswith("dl")


def test_hub_cache_dir_is_exactly_module_data_slash_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(_make_checkpoint(tmp_path / "dl"))

    cfg = PolicyConfig.parse({"model_hub_id": "a/b"})
    (tmp_path / "dl").mkdir()
    resolve_checkpoint(cfg, snapshot_download=fake_snapshot_download)

    expected = str(Path(tmp_path) / "checkpoints")
    assert captured["cache_dir"] == expected
    assert Path(captured["cache_dir"]).is_dir()


def test_hub_download_default_revision_is_main(tmp_path, monkeypatch):
    # Nothing in the base test list asserts the default revision reaches
    # snapshot_download -- it could silently become a hardcoded constant
    # and the suite would stay green.
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(_make_checkpoint(tmp_path / "dl"))

    cfg = PolicyConfig.parse({"model_hub_id": "a/b"})
    (tmp_path / "dl").mkdir()
    resolve_checkpoint(cfg, snapshot_download=fake_snapshot_download)

    assert captured["revision"] == "main"


def test_hub_download_repo_id_matches_hub_id(tmp_path, monkeypatch):
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(_make_checkpoint(tmp_path / "dl"))

    cfg = PolicyConfig.parse({"model_hub_id": "some/other-repo"})
    (tmp_path / "dl").mkdir()
    resolve_checkpoint(cfg, snapshot_download=fake_snapshot_download)

    assert captured["repo_id"] == "some/other-repo"


def test_hub_download_result_still_verified(tmp_path, monkeypatch):
    # A hub download that returns a directory missing required files must
    # still raise -- _verify must not be skipped on the hub path.
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))
    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()

    def fake_snapshot_download(**kwargs):
        return str(broken_dir)

    cfg = PolicyConfig.parse({"model_hub_id": "a/b"})
    with pytest.raises(ResolveError, match="config.json"):
        resolve_checkpoint(cfg, snapshot_download=fake_snapshot_download)


def test_missing_huggingface_hub_dependency_gives_actionable_error(tmp_path, monkeypatch):
    # huggingface_hub is not a base dependency of this module -- it only
    # arrives transitively via the optional `lerobot` extra. A wheel
    # installed without that extra must fail with an actionable
    # ResolveError, not a bare ModuleNotFoundError surfacing from inside
    # the function during first boot. No snapshot_download is injected
    # here specifically to exercise the real import branch.
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    cfg = PolicyConfig.parse({"model_hub_id": "a/b"})
    with pytest.raises(ResolveError, match="lerobot"):
        resolve_checkpoint(cfg)


def test_hub_token_read_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))
    monkeypatch.setenv("MY_HF_TOKEN", "secret-value")
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(_make_checkpoint(tmp_path / "dl"))

    (tmp_path / "dl").mkdir()
    cfg = PolicyConfig.parse(
        {"model_hub_id": "a/b", "hf_token_env": "MY_HF_TOKEN"}
    )
    resolve_checkpoint(cfg, snapshot_download=fake_snapshot_download)
    assert captured["token"] == "secret-value"


def test_hub_token_is_none_when_hf_token_env_unset(tmp_path, monkeypatch):
    # The counterpart to the "token read from env" test: nothing in the
    # base suite proves token defaults to None rather than some other
    # placeholder when hf_token_env is not configured.
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(_make_checkpoint(tmp_path / "dl"))

    (tmp_path / "dl").mkdir()
    cfg = PolicyConfig.parse({"model_hub_id": "a/b"})
    resolve_checkpoint(cfg, snapshot_download=fake_snapshot_download)
    assert captured["token"] is None


def test_missing_token_env_var_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))
    monkeypatch.delenv("ABSENT_TOKEN", raising=False)
    cfg = PolicyConfig.parse({"model_hub_id": "a/b", "hf_token_env": "ABSENT_TOKEN"})
    with pytest.raises(ResolveError, match="ABSENT_TOKEN"):
        resolve_checkpoint(cfg, snapshot_download=lambda **k: "")


def test_falls_back_to_tempdir_without_module_data(tmp_path, monkeypatch):
    monkeypatch.delenv("VIAM_MODULE_DATA", raising=False)
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(_make_checkpoint(tmp_path / "dl2"))

    (tmp_path / "dl2").mkdir()
    cfg = PolicyConfig.parse({"model_hub_id": "a/b"})
    resolve_checkpoint(cfg, snapshot_download=fake_snapshot_download)
    assert captured["cache_dir"]  # some writable fallback was chosen


def test_fallback_tempdir_is_under_system_temp_and_created(tmp_path, monkeypatch):
    monkeypatch.delenv("VIAM_MODULE_DATA", raising=False)
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(_make_checkpoint(tmp_path / "dl3"))

    (tmp_path / "dl3").mkdir()
    cfg = PolicyConfig.parse({"model_hub_id": "a/b"})
    resolve_checkpoint(cfg, snapshot_download=fake_snapshot_download)

    assert captured["cache_dir"].startswith(tempfile.gettempdir())
    assert Path(captured["cache_dir"]).is_dir()


# ---------------------------------------------------------------------------
# Secret handling
# ---------------------------------------------------------------------------


def test_token_never_appears_in_logs(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))
    monkeypatch.setenv("MY_HF_TOKEN", "super-secret-do-not-log")

    def fake_snapshot_download(**kwargs):
        return str(_make_checkpoint(tmp_path / "dl"))

    (tmp_path / "dl").mkdir()
    cfg = PolicyConfig.parse({"model_hub_id": "a/b", "hf_token_env": "MY_HF_TOKEN"})

    with caplog.at_level(logging.DEBUG):
        resolve_checkpoint(cfg, snapshot_download=fake_snapshot_download)

    assert "super-secret-do-not-log" not in caplog.text


def test_hf_token_env_with_model_path_is_ignored_with_warning(tmp_path, caplog):
    # A local checkpoint doesn't need a hub token. Rather than silently
    # discarding a configured credential, warn and proceed -- do not raise.
    ckpt = _make_checkpoint(tmp_path)
    cfg = PolicyConfig.parse(
        {"model_path": str(ckpt), "hf_token_env": "SOME_TOKEN_VAR"}
    )
    with caplog.at_level(logging.WARNING):
        result = resolve_checkpoint(cfg)

    assert result == str(ckpt)
    assert "hf_token_env" in caplog.text
    assert "SOME_TOKEN_VAR" in caplog.text
