import logging
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

from vla.config_util import VLAError
from vla.policy.config import PolicyConfig
from vla.policy.resolver import resolve_checkpoint, ResolveError

# True on a genuine POSIX permission-enforcement environment: `os.chmod`
# tests are meaningless on Windows, and root bypasses permission checks
# entirely, so both must be skipped rather than silently passing for the
# wrong reason.
_SKIP_PERMISSION_TESTS = os.name == "nt" or (hasattr(os, "getuid") and os.getuid() == 0)
skip_without_permission_enforcement = pytest.mark.skipif(
    _SKIP_PERMISSION_TESTS,
    reason="requires POSIX permission enforcement as a non-root user",
)

REQUIRED = ["config.json", "model.safetensors"]


def _make_checkpoint(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        (tmp_path / name).write_text("{}")
    return tmp_path


# ---------------------------------------------------------------------------
# Local path resolution
# ---------------------------------------------------------------------------


def test_resolves_local_path(tmp_path):
    ckpt = _make_checkpoint(tmp_path)
    cfg = PolicyConfig.parse({"model_path": str(ckpt)})
    # .resolve() rather than a bare string comparison: on macOS tmp_path
    # commonly lives under a symlinked /var, and the resolver is now
    # required (item 5) to return a canonical absolute path.
    assert resolve_checkpoint(cfg) == str(ckpt.resolve())


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
    #
    # A dummy module is registered first and then overwritten with None so
    # this forces the same "import halted" ImportError regardless of
    # whether huggingface_hub actually happens to be installed in whatever
    # environment runs this test.
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.ModuleType("huggingface_hub"))
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


def test_hub_token_is_false_when_hf_token_env_unset(tmp_path, monkeypatch):
    # `token=False` -- not `None` -- must reach snapshot_download when no
    # hf_token_env is configured. huggingface_hub treats `None` as "defer to
    # whatever is cached on this machine" and `False` as "anonymous, no
    # matter what is cached." A module that never asked for a token must not
    # let some unrelated (and possibly stale/invalid) ambient login get sent
    # on its behalf -- that previously turned into a `401 Unauthorized` on a
    # fully public repo, which reads like "model_hub_id is wrong" and is not.
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(_make_checkpoint(tmp_path / "dl"))

    (tmp_path / "dl").mkdir()
    cfg = PolicyConfig.parse({"model_hub_id": "a/b"})
    resolve_checkpoint(cfg, snapshot_download=fake_snapshot_download)
    assert captured["token"] is False


def test_hub_token_is_the_real_string_not_true_when_hf_token_env_set(tmp_path, monkeypatch):
    # The counterpart to the anonymous-path test above: when a token *is*
    # configured, the literal secret string must reach snapshot_download --
    # not `True` (which would tell huggingface_hub to go pull some other
    # token from the cache instead of the one actually configured here).
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))
    monkeypatch.setenv("MY_HF_TOKEN", "secret-value")
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(_make_checkpoint(tmp_path / "dl"))

    (tmp_path / "dl").mkdir()
    cfg = PolicyConfig.parse({"model_hub_id": "a/b", "hf_token_env": "MY_HF_TOKEN"})
    resolve_checkpoint(cfg, snapshot_download=fake_snapshot_download)
    assert captured["token"] == "secret-value"
    assert captured["token"] is not True


def test_missing_token_env_var_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))
    monkeypatch.delenv("ABSENT_TOKEN", raising=False)
    cfg = PolicyConfig.parse({"model_hub_id": "a/b", "hf_token_env": "ABSENT_TOKEN"})
    with pytest.raises(ResolveError) as excinfo:
        resolve_checkpoint(cfg, snapshot_download=lambda **k: "")
    # The env var *name* lands in this message too, so it is redacted the
    # same as the token value would be -- belt and braces, since a short
    # pasted secret could itself look like a plausible env var name.
    message = str(excinfo.value)
    assert "ABSENT_TOKEN" not in message
    assert "ABSE" in message


def test_falls_back_to_tempdir_without_module_data(tmp_path, monkeypatch):
    # Monkeypatch tempfile.gettempdir so this never touches the real system
    # temp dir -- without this, the fallback path was creating a real
    # directory outside tmp_path (and thus outside pytest's cleanup) on
    # every run of this test.
    monkeypatch.delenv("VIAM_MODULE_DATA", raising=False)
    fake_system_tmp = tmp_path / "faketmp"
    fake_system_tmp.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_system_tmp))
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(_make_checkpoint(tmp_path / "dl2"))

    (tmp_path / "dl2").mkdir()
    cfg = PolicyConfig.parse({"model_hub_id": "a/b"})
    resolve_checkpoint(cfg, snapshot_download=fake_snapshot_download)
    assert captured["cache_dir"]  # some writable fallback was chosen
    assert captured["cache_dir"].startswith(str(fake_system_tmp))


def test_fallback_tempdir_is_under_system_temp_and_created(tmp_path, monkeypatch):
    monkeypatch.delenv("VIAM_MODULE_DATA", raising=False)
    fake_system_tmp = tmp_path / "faketmp2"
    fake_system_tmp.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_system_tmp))
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(_make_checkpoint(tmp_path / "dl3"))

    (tmp_path / "dl3").mkdir()
    cfg = PolicyConfig.parse({"model_hub_id": "a/b"})
    resolve_checkpoint(cfg, snapshot_download=fake_snapshot_download)

    assert captured["cache_dir"].startswith(str(fake_system_tmp))
    assert Path(captured["cache_dir"]).is_dir()


def test_fallback_cache_dir_is_unique_per_call(tmp_path, monkeypatch):
    # A fixed fallback directory name under a shared /tmp is a symlink-attack
    # vector from another local user; mkdtemp()'s unique, unpredictable name
    # (mode 0700) is the fix. Two resolutions in the same process must not
    # collide on the same fallback directory.
    monkeypatch.delenv("VIAM_MODULE_DATA", raising=False)
    fake_system_tmp = tmp_path / "faketmp3"
    fake_system_tmp.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_system_tmp))

    seen = []

    def fake_snapshot_download(**kwargs):
        seen.append(kwargs["cache_dir"])
        target = Path(kwargs["cache_dir"]) / "dl"
        target.mkdir()
        return str(_make_checkpoint(target))

    cfg = PolicyConfig.parse({"model_hub_id": "a/b"})
    resolve_checkpoint(cfg, snapshot_download=fake_snapshot_download)
    resolve_checkpoint(cfg, snapshot_download=fake_snapshot_download)

    assert seen[0] != seen[1]


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

    assert result == str(ckpt.resolve())
    assert "hf_token_env" in caplog.text
    # The field's value is redacted even in the warning -- never the whole
    # value, only enough to recognize it (first 4 chars + a length).
    assert "SOME_TOKEN_VAR" not in caplog.text
    assert "SOME" in caplog.text


def test_ignored_hf_token_env_warning_redacts_a_longer_value(tmp_path, caplog):
    fake_name = "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"  # 32 chars, valid env-var shape
    ckpt = _make_checkpoint(tmp_path)
    cfg = PolicyConfig.parse({"model_path": str(ckpt), "hf_token_env": fake_name})
    with caplog.at_level(logging.WARNING):
        resolve_checkpoint(cfg)

    assert fake_name not in caplog.text
    assert "ABCD" in caplog.text


def test_missing_token_env_var_error_redacts_a_longer_value(tmp_path, monkeypatch):
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))
    fake_name = "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
    monkeypatch.delenv(fake_name, raising=False)
    cfg = PolicyConfig.parse({"model_hub_id": "a/b", "hf_token_env": fake_name})
    with pytest.raises(ResolveError) as excinfo:
        resolve_checkpoint(cfg, snapshot_download=lambda **k: "")

    message = str(excinfo.value)
    assert fake_name not in message
    assert "ABCD" in message
    assert "32" in message


# ---------------------------------------------------------------------------
# Shared error base
# ---------------------------------------------------------------------------


def test_resolve_error_is_a_vla_error():
    assert issubclass(ResolveError, VLAError)


def test_resolve_error_is_still_a_runtime_error():
    assert issubclass(ResolveError, RuntimeError)


# ---------------------------------------------------------------------------
# Filesystem errors must not escape as bare OSError-family exceptions
# ---------------------------------------------------------------------------


@skip_without_permission_enforcement
def test_unreadable_model_path_directory_gives_resolve_error(tmp_path):
    # viam-server commonly runs the module as a non-root user; an scp'd or
    # root-extracted checkpoint directory lacking the execute bit is a
    # realistic deployment failure, not a hypothetical one.
    ckpt = _make_checkpoint(tmp_path / "ckpt")
    os.chmod(ckpt, 0o000)
    try:
        cfg = PolicyConfig.parse({"model_path": str(ckpt)})
        with pytest.raises(ResolveError, match="could not read"):
            resolve_checkpoint(cfg)
    finally:
        os.chmod(ckpt, 0o700)  # restore so pytest can clean up tmp_path


@skip_without_permission_enforcement
def test_read_only_module_data_parent_gives_resolve_error(tmp_path, monkeypatch):
    parent = tmp_path / "readonly_parent"
    parent.mkdir()
    os.chmod(parent, 0o500)  # r-x: cannot create the "checkpoints" subdir
    monkeypatch.setenv("VIAM_MODULE_DATA", str(parent))
    try:
        cfg = PolicyConfig.parse({"model_hub_id": "a/b"})
        with pytest.raises(ResolveError, match="could not create"):
            resolve_checkpoint(cfg, snapshot_download=lambda **k: "unused")
    finally:
        os.chmod(parent, 0o700)


def test_snapshot_download_failure_wrapped_as_resolve_error(tmp_path, monkeypatch):
    # huggingface_hub's HfHubHTTPError (401/404/connection refused/disk
    # full) descends from requests.exceptions.RequestException, itself an
    # OSError subclass -- this simulates that without depending on the
    # package being installed.
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))

    class FakeHubHTTPError(OSError):
        pass

    def fake_snapshot_download(**kwargs):
        raise FakeHubHTTPError("401 Client Error: Unauthorized")

    cfg = PolicyConfig.parse({"model_hub_id": "a/b"})
    with pytest.raises(ResolveError) as excinfo:
        resolve_checkpoint(cfg, snapshot_download=fake_snapshot_download)

    assert "a/b" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, FakeHubHTTPError)


# ---------------------------------------------------------------------------
# Bounding the download
# ---------------------------------------------------------------------------


def test_hub_download_passes_etag_timeout(tmp_path, monkeypatch):
    # An unreachable hub endpoint must not hang forever. etag_timeout
    # bounds the initial metadata request; HF_HUB_DOWNLOAD_TIMEOUT (an env
    # var read by huggingface_hub itself, not passed here) bounds the
    # transfer itself.
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path))
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(_make_checkpoint(tmp_path / "dl"))

    cfg = PolicyConfig.parse({"model_hub_id": "a/b"})
    (tmp_path / "dl").mkdir()
    resolve_checkpoint(cfg, snapshot_download=fake_snapshot_download)

    assert captured["etag_timeout"] == 10


# ---------------------------------------------------------------------------
# Path handling
# ---------------------------------------------------------------------------


def test_uninterpolated_package_reference_in_model_path_errors(tmp_path):
    # If viam-server failed to substitute ${packages.ml_model.name}, the
    # module sees the literal placeholder string. That is a different, more
    # actionable fault than "directory does not exist" and deserves its own
    # message.
    cfg = PolicyConfig.parse({"model_path": "${packages.ml_model.name}"})
    with pytest.raises(ResolveError) as excinfo:
        resolve_checkpoint(cfg)
    message = str(excinfo.value)
    assert "${" in message
    assert "does not exist" not in message


def test_resolves_local_path_returns_absolute_path(tmp_path, monkeypatch):
    # A relative model_path resolves against an unpredictable module cwd;
    # the resolved checkpoint path must be absolute so it means the same
    # thing regardless of what the cwd is by the time backend.load runs.
    _make_checkpoint(tmp_path / "sub")
    monkeypatch.chdir(tmp_path)
    cfg = PolicyConfig.parse({"model_path": "sub"})

    result = resolve_checkpoint(cfg)

    assert os.path.isabs(result)
    assert result == str((tmp_path / "sub").resolve())


def test_expands_home_directory_in_model_path(tmp_path, monkeypatch):
    _make_checkpoint(tmp_path / "ckpt")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # harmless on POSIX

    cfg = PolicyConfig.parse({"model_path": "~/ckpt"})
    result = resolve_checkpoint(cfg)

    assert result == str((tmp_path / "ckpt").resolve())


def test_dangling_symlink_gets_distinct_error(tmp_path):
    # A dangling symlink exists (as a link) but its target does not -- a
    # different, more specific fault than the generic "does not exist" one,
    # same misdiagnosis class as the file-vs-directory case.
    target = tmp_path / "nonexistent_target"
    link = tmp_path / "broken_link"
    link.symlink_to(target)
    cfg = PolicyConfig.parse({"model_path": str(link)})

    with pytest.raises(ResolveError) as excinfo:
        resolve_checkpoint(cfg)

    message = str(excinfo.value)
    assert "symlink" in message.lower()
    assert "checkpoint directory does not exist" not in message
