import asyncio
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from vla.policy import installer as ins


def _req_dir(tmp_path: Path) -> Path:
    d = tmp_path / "requirements"
    d.mkdir()
    (d / "base.txt").write_text(
        "torch==2.11.0 ; sys_platform != 'linux'\n"
        "torch==2.11.0+cu128 ; sys_platform == 'linux'\n"
        "torchvision==0.26.0+cu128 ; sys_platform == 'linux'\n"
        "nvidia-cublas-cu12==12.8.4.1 ; sys_platform == 'linux'\n"
        "triton==3.6.0 ; sys_platform == 'linux'\n"
        "lerobot @ git+https://github.com/huggingface/lerobot@abc\n"
        "einops==0.8.1\n"
    )
    for stem in ("smolvla", "evo1", "diffusion", "pi", "cosmos3"):
        (d / f"{stem}.txt").write_text(f"{stem}-dep==1.0\n")
    return d


@pytest.mark.parametrize(
    "policy_type,extra",
    [
        ("smolvla", "smolvla.txt"),
        ("evo1", "evo1.txt"),
        ("diffusion", "diffusion.txt"),
        ("pi0", "pi.txt"),
        ("pi0_fast", "pi.txt"),
        ("pi05", "pi.txt"),
        ("cosmos3_edge", "cosmos3.txt"),
        ("cosmos3_nano", "cosmos3.txt"),
    ],
)
def test_type_maps_to_base_and_its_extra(tmp_path, policy_type, extra):
    d = _req_dir(tmp_path)
    assert ins.requirement_files(d, policy_type) == [d / "base.txt", d / extra]


@pytest.mark.parametrize("policy_type", ["act", "tdmpc", "vqbet"])
def test_base_only_types(tmp_path, policy_type):
    d = _req_dir(tmp_path)
    assert ins.requirement_files(d, policy_type) == [d / "base.txt"]


def test_unlisted_type_is_refused_with_the_supported_list(tmp_path):
    with pytest.raises(ins.InstallError, match="'groot'.*smolvla"):
        ins.requirement_files(_req_dir(tmp_path), "groot")


def test_digest_changes_with_file_contents_and_jetson_settings(tmp_path):
    d = _req_dir(tmp_path)
    files = ins.requirement_files(d, "smolvla")
    a = ins.marker_digest(files, None)
    assert a == ins.marker_digest(files, None)
    assert a != ins.marker_digest(files, {"JETSON_TORCH_VERSION": "2.8.0"})
    (d / "smolvla.txt").write_text("smolvla-dep==2.0\n")
    assert a != ins.marker_digest(files, None)


def test_jetson_filter_drops_torch_and_its_cuda_dependencies():
    text = (
        "torch==2.11.0+cu128 ; sys_platform == 'linux'\n"
        "torchvision==0.26.0+cu128 ; sys_platform == 'linux'\n"
        "nvidia-cublas-cu12==12.8.4.1 ; sys_platform == 'linux'\n"
        "triton==3.6.0 ; sys_platform == 'linux'\n"
        "cuda-toolkit==12.8.1 ; sys_platform == 'linux'\n"
        "cuda-bindings==12.9.4 ; sys_platform == 'linux'\n"
        "lerobot @ git+https://x@abc\n"
        "einops==0.8.1\n"
    )
    assert ins.filter_for_jetson(text) == "lerobot @ git+https://x@abc\neinops==0.8.1\n"


def test_commands_without_jetson_install_all_files_in_one_call(tmp_path):
    d = _req_dir(tmp_path)
    files = ins.requirement_files(d, "smolvla")
    cmds = ins.install_commands("uv", "/py", files, None, tmp_path)
    assert cmds == [["uv", "pip", "install", "--python", "/py", "-r", str(d / "base.txt"), "-r", str(d / "smolvla.txt")]]


def test_commands_on_jetson_install_jetson_torch_first_then_filtered_base(tmp_path):
    d = _req_dir(tmp_path)
    files = ins.requirement_files(d, "smolvla")
    jetson = {"JETSON_TORCH_INDEX": "https://j/idx", "JETSON_TORCH_VERSION": "2.8.0", "JETSON_TORCHVISION_VERSION": "0.23.0"}
    cmds = ins.install_commands("uv", "/py", files, jetson, tmp_path)
    assert cmds[0] == [
        "uv", "pip", "install", "--python", "/py", "--no-deps",
        "--reinstall-package", "torch", "--reinstall-package", "torchvision",
        "--index-url", "https://j/idx", "torch==2.8.0", "torchvision==0.23.0",
    ]
    filtered = tmp_path / "base.jetson.txt"
    assert cmds[1] == ["uv", "pip", "install", "--python", "/py", "-r", str(filtered), "-r", str(d / "smolvla.txt")]
    text = filtered.read_text()
    assert "torch" not in text.replace("lerobot", "") and "nvidia" not in text
    assert "triton" not in text and "cuda" not in text


def test_jetson_settings_none_off_tegra(monkeypatch):
    monkeypatch.setattr(ins, "is_jetson", lambda: False)
    assert ins.jetson_settings({}) is None


def test_jetson_settings_defaults_and_overrides(monkeypatch):
    monkeypatch.setattr(ins, "is_jetson", lambda: True)
    monkeypatch.setattr(ins, "glibc_version", lambda: (2, 39))
    s = ins.jetson_settings({"JETSON_TORCH_VERSION": "2.9.0"})
    assert s == {
        "JETSON_TORCH_INDEX": "https://pypi.jetson-ai-lab.io/jp6/cu129",
        "JETSON_TORCH_VERSION": "2.9.0",
        "JETSON_TORCHVISION_VERSION": "0.23.0",
    }


def test_jetson_with_old_glibc_falls_back_to_generic_with_a_warning(monkeypatch, caplog):
    monkeypatch.setattr(ins, "is_jetson", lambda: True)
    monkeypatch.setattr(ins, "glibc_version", lambda: (2, 35))
    assert ins.jetson_settings({}) is None
    assert "glibc is 2.35" in caplog.text
    assert "JetPack 7" in caplog.text


def test_glibc_version_parses_major_minor_ignoring_suffix(monkeypatch):
    monkeypatch.setattr(ins.platform, "libc_ver", lambda: ("glibc", "2.38a"))
    assert ins.glibc_version() == (2, 38)


@pytest.mark.parametrize("libc", [("glibc", ""), ("", "")])
def test_glibc_version_none_when_unknown(monkeypatch, libc):
    monkeypatch.setattr(ins.platform, "libc_ver", lambda: libc)
    assert ins.glibc_version() is None


def test_jetson_with_unknown_glibc_takes_the_sm87_path(monkeypatch):
    monkeypatch.setattr(ins, "is_jetson", lambda: True)
    monkeypatch.setattr(ins, "glibc_version", lambda: None)
    assert ins.jetson_settings({}) == ins.JETSON_DEFAULTS


def _fake_uv(prefix: Path, body: str = "") -> Path:
    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    uv = bin_dir / "uv"
    uv.write_text(f'#!/bin/sh\necho "$@" >> "$UV_LOG"\n{body}\n')
    uv.chmod(uv.stat().st_mode | stat.S_IXUSR)
    return uv


@pytest.fixture
def env(tmp_path, monkeypatch):
    log = tmp_path / "uv.log"
    monkeypatch.setenv("UV_LOG", str(log))
    monkeypatch.setattr(ins, "is_jetson", lambda: False)
    return log


async def test_no_requirements_dir_is_a_noop(tmp_path, monkeypatch, env):
    monkeypatch.delenv("VLA_REQUIREMENTS_DIR", raising=False)
    await ins.ensure_installed("smolvla", prefix=tmp_path / "venv")
    assert not env.exists()


async def test_installs_then_skips_on_matching_marker(tmp_path, env):
    d, prefix = _req_dir(tmp_path), tmp_path / "venv"
    _fake_uv(prefix)
    await ins.ensure_installed("smolvla", req_dir=d, prefix=prefix)
    assert env.read_text().count("pip install") == 1
    assert (prefix / ".vla-installed-smolvla.sha256").exists()
    await ins.ensure_installed("smolvla", req_dir=d, prefix=prefix)
    assert env.read_text().count("pip install") == 1


async def test_changed_requirements_reinstall(tmp_path, env):
    d, prefix = _req_dir(tmp_path), tmp_path / "venv"
    _fake_uv(prefix)
    await ins.ensure_installed("smolvla", req_dir=d, prefix=prefix)
    (d / "base.txt").write_text("einops==0.8.2\n")
    await ins.ensure_installed("smolvla", req_dir=d, prefix=prefix)
    assert env.read_text().count("pip install") == 2


async def test_failure_raises_with_uv_output_and_writes_no_marker(tmp_path, env):
    d, prefix = _req_dir(tmp_path), tmp_path / "venv"
    _fake_uv(prefix, 'echo "error: no space left on device"\nexit 2')
    with pytest.raises(ins.InstallError, match=r"(?s)exit 2.*no space left"):
        await ins.ensure_installed("smolvla", req_dir=d, prefix=prefix)
    assert not (prefix / ".vla-installed-smolvla.sha256").exists()


async def test_missing_git_is_refused(tmp_path, env, monkeypatch):
    d, prefix = _req_dir(tmp_path), tmp_path / "venv"
    _fake_uv(prefix)
    monkeypatch.setattr(ins.shutil, "which", lambda name: None)
    with pytest.raises(ins.InstallError, match="git"):
        await ins.ensure_installed("smolvla", req_dir=d, prefix=prefix)


async def test_linux_sets_cu128_backend_unless_set(tmp_path, env, monkeypatch):
    d, prefix = _req_dir(tmp_path), tmp_path / "venv"
    _fake_uv(prefix, 'echo "backend=$UV_TORCH_BACKEND" >> "$UV_LOG"')
    monkeypatch.setattr(ins.sys, "platform", "linux")
    monkeypatch.delenv("UV_TORCH_BACKEND", raising=False)
    await ins.ensure_installed("act", req_dir=d, prefix=prefix)
    assert "backend=cu128" in env.read_text()


async def test_jetson_drops_the_cu128_backend(tmp_path, env, monkeypatch):
    d, prefix = _req_dir(tmp_path), tmp_path / "venv"
    _fake_uv(prefix, 'echo "backend=$UV_TORCH_BACKEND" >> "$UV_LOG"')
    monkeypatch.setattr(ins, "jetson_settings", lambda env: dict(ins.JETSON_DEFAULTS))
    monkeypatch.setattr(ins.sys, "platform", "linux")
    monkeypatch.setenv("UV_TORCH_BACKEND", "cu128")
    await ins.ensure_installed("act", req_dir=d, prefix=prefix)
    log = env.read_text()
    assert "backend=cu128" not in log
    assert "jetson-ai-lab" in log


async def test_cancellation_terminates_the_subprocess(tmp_path, env):
    d, prefix = _req_dir(tmp_path), tmp_path / "venv"
    pid_file = tmp_path / "pid"
    _fake_uv(prefix, f'echo $$ > "{pid_file}"\nsleep 30')
    task = asyncio.create_task(ins.ensure_installed("smolvla", req_dir=d, prefix=prefix))
    for _ in range(100):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    pid = int(pid_file.read_text())
    # The whole process group is gone, the shell's `sleep` child included
    # (a killed child can be an unreaped zombie for a moment).
    for _ in range(20):
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, 0)
    assert not (prefix / ".vla-installed-smolvla.sha256").exists()


async def test_lock_serializes_and_second_caller_skips(tmp_path, env):
    d, prefix = _req_dir(tmp_path), tmp_path / "venv"
    _fake_uv(prefix, "sleep 0.5")
    await asyncio.gather(
        ins.ensure_installed("smolvla", req_dir=d, prefix=prefix),
        ins.ensure_installed("smolvla", req_dir=d, prefix=prefix),
    )
    assert env.read_text().count("pip install") == 1


async def test_lock_waiter_can_be_cancelled(tmp_path, env):
    d, prefix = _req_dir(tmp_path), tmp_path / "venv"
    _fake_uv(prefix, "sleep 1")
    first = asyncio.create_task(ins.ensure_installed("smolvla", req_dir=d, prefix=prefix))
    # Wait until the first install's uv is running (it holds the lock), not a fixed
    # delay: under full-suite load the subprocess can take longer to start.
    for _ in range(100):
        if env.exists() and "smolvla.txt" in env.read_text():
            break
        await asyncio.sleep(0.05)
    waiter = asyncio.create_task(ins.ensure_installed("evo1", req_dir=d, prefix=prefix))
    await asyncio.sleep(0.2)
    assert "evo1.txt" not in env.read_text()  # blocked on the lock
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await first


async def test_for_checkpoint_reads_the_type(tmp_path, env, monkeypatch):
    d, prefix = _req_dir(tmp_path), tmp_path / "venv"
    _fake_uv(prefix)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text('{"type": "diffusion"}')
    monkeypatch.setenv("VLA_REQUIREMENTS_DIR", str(d))
    await ins.ensure_installed_for_checkpoint(str(ckpt), prefix=prefix)
    assert "diffusion.txt" in env.read_text()


async def test_for_checkpoint_without_config_json_fails_clearly(tmp_path, env, monkeypatch):
    monkeypatch.setenv("VLA_REQUIREMENTS_DIR", str(_req_dir(tmp_path)))
    with pytest.raises(ins.InstallError, match="config.json"):
        await ins.ensure_installed_for_checkpoint(str(tmp_path), prefix=tmp_path / "venv")


def test_cli_without_any_requirements_dir_fails(tmp_path):
    e = {k: v for k, v in os.environ.items() if k != "VLA_REQUIREMENTS_DIR"}
    r = subprocess.run(
        [sys.executable, "-m", "vla.policy.installer", "smolvla"], env=e, capture_output=True, text=True
    )
    assert r.returncode != 0
    assert "VLA_REQUIREMENTS_DIR is unset" in r.stderr


def test_requirements_dir_unset_or_empty_is_none(monkeypatch):
    monkeypatch.delenv("VLA_REQUIREMENTS_DIR", raising=False)
    assert ins.requirements_dir() is None
    monkeypatch.setenv("VLA_REQUIREMENTS_DIR", "")
    assert ins.requirements_dir() is None


def test_requirements_dir_set_but_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("VLA_REQUIREMENTS_DIR", str(tmp_path / "nope"))
    with pytest.raises(ins.InstallError, match="is not a directory"):
        ins.requirements_dir()


async def test_very_long_output_line_does_not_break_the_install(tmp_path, env):
    d, prefix = _req_dir(tmp_path), tmp_path / "venv"
    _fake_uv(prefix, "head -c 200000 /dev/zero | tr '\\0' x\necho")
    await ins.ensure_installed("smolvla", req_dir=d, prefix=prefix)
    assert (prefix / ".vla-installed-smolvla.sha256").exists()


async def test_unwritable_marker_raises_install_error(tmp_path, env):
    d, prefix = _req_dir(tmp_path), tmp_path / "venv"
    _fake_uv(prefix)
    (prefix / ".vla-installed-smolvla.sha256").mkdir()  # os.replace onto a directory fails
    with pytest.raises(ins.InstallError, match="installed, but could not write"):
        await ins.ensure_installed("smolvla", req_dir=d, prefix=prefix)
