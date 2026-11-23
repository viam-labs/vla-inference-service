"""Install the heavy policy stack (torch, LeRobot, a policy type's extras) lazily.

setup.sh installs only the core. The policy service calls `ensure_installed`
during its background load, before the first `import torch`. Requirement files
are exported from uv.lock at build time (tools/export_requirements.py) and
found through VLA_REQUIREMENTS_DIR, which run.sh sets; without it (a source
checkout, tests) installing is a no-op. See
docs/superpowers/specs/2026-11-22-lazy-policy-install-design.md.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import importlib
import json
import logging
import os
import platform
import re
import shutil
import signal
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import AsyncIterator, Mapping

from vla.config_util import VLAError

LOGGER = logging.getLogger(__name__)

# Policy type -> its extra's requirement file (None: base only). A type is added
# here once its extra is verified; anything else is refused.
TYPE_FILES: dict[str, str | None] = {
    "smolvla": "smolvla.txt",
    "evo1": "evo1.txt",
    "diffusion": "diffusion.txt",
    "pi0": "pi.txt",
    "pi0_fast": "pi.txt",
    "pi05": "pi.txt",
    "cosmos3_edge": "cosmos3.txt",
    "cosmos3_nano": "cosmos3.txt",
    "act": None,
    "tdmpc": None,
    "vqbet": None,
}

# ponytail: jp6/cu129 defaults, overridable -- a different JetPack ships a
# different CUDA, and these env vars are the knob for it. NVIDIA's
# jetson-ai-lab index is the only source of sm_87 cp312 builds.
JETSON_DEFAULTS = {
    "JETSON_TORCH_INDEX": "https://pypi.jetson-ai-lab.io/jp6/cu129",
    "JETSON_TORCH_VERSION": "2.8.0",
    "JETSON_TORCHVISION_VERSION": "0.23.0",
}

# Torch 2.11+cu128 and the packages only it depends on; Jetson's torch 2.8.0 brings its own.
_JETSON_DROP = re.compile(
    r"^\s*(torch|torchvision|triton|nvidia[-_.][A-Za-z0-9._-]*|cuda[-_.](?:toolkit|bindings|pathfinder))"
    r"\s*(==|@|;|$)",
    re.IGNORECASE,
)


class InstallError(VLAError):
    """The lazy install could not complete; the message is meant for `status`."""


def requirement_files(req_dir: Path, policy_type: str) -> list[Path]:
    if policy_type not in TYPE_FILES:
        raise InstallError(
            f"policy type {policy_type!r} is not supported by the installer; "
            f"supported: {', '.join(sorted(TYPE_FILES))}"
        )
    extra = TYPE_FILES[policy_type]
    return [req_dir / "base.txt"] + ([req_dir / extra] if extra else [])


def marker_digest(files: list[Path], jetson: Mapping[str, str] | None) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    h.update(repr(sorted((jetson or {}).items())).encode())
    return h.hexdigest()


def is_jetson() -> bool:
    if Path("/etc/nv_tegra_release").exists():
        return True
    try:
        return b"tegra" in Path("/proc/device-tree/compatible").read_bytes().lower()
    except OSError:
        return False


def glibc_version() -> tuple[int, ...] | None:
    name, version = platform.libc_ver()
    m = re.match(r"(\d+)\.(\d+)", version)
    if name != "glibc" or not m:
        return None
    return int(m.group(1)), int(m.group(2))


def jetson_settings(env: Mapping[str, str]) -> dict[str, str] | None:
    """The Jetson torch overrides, or None to install the generic (base.txt) torch."""
    if not is_jetson():
        return None
    glibc = glibc_version()
    # The cp312 sm_87 wheels are built against Ubuntu 24.04 and need glibc 2.38;
    # on JetPack 6 (22.04, glibc 2.35) they install and then fail to import.
    if glibc is not None and glibc < (2, 38):
        LOGGER.warning(
            "Tegra detected, but glibc is %s and the cp312 sm_87 wheels need >= 2.38. "
            "Installing the generic aarch64 torch, which cannot run kernels on this GPU -- "
            "the policy will fall back to CPU. JetPack 7 (Ubuntu 24.04) unblocks GPU inference.",
            ".".join(map(str, glibc)),
        )
        return None
    return {k: env.get(k, v) for k, v in JETSON_DEFAULTS.items()}


def filter_for_jetson(text: str) -> str:
    return "".join(line for line in text.splitlines(keepends=True) if not _JETSON_DROP.match(line))


def install_commands(
    uv: str, python: str, files: list[Path], jetson: Mapping[str, str] | None, tmp: Path
) -> list[list[str]]:
    """The `uv pip install` invocations, in order.

    On Jetson the sm_87 torch goes in first, --no-deps: LeRobot requires
    torch>=2.7,<2.16, so installing the filtered base first would pull the
    generic torch back in. With 2.8.0 already present and in range, uv keeps it.
    """
    pip = [uv, "pip", "install", "--python", python]
    if jetson is None:
        return [pip + [arg for f in files for arg in ("-r", str(f))]]
    filtered = tmp / "base.jetson.txt"
    filtered.write_text(filter_for_jetson(files[0].read_text()))
    # --reinstall-package: a broken sm_87 torch of the same version would otherwise
    # count as satisfied. This only runs when the marker is missing or stale.
    torch_cmd = pip + [
        "--no-deps",
        "--reinstall-package", "torch",
        "--reinstall-package", "torchvision",
        "--index-url", jetson["JETSON_TORCH_INDEX"],
        f"torch=={jetson['JETSON_TORCH_VERSION']}",
        f"torchvision=={jetson['JETSON_TORCHVISION_VERSION']}",
    ]
    rest = [filtered, *files[1:]]
    return [torch_cmd, pip + [arg for f in rest for arg in ("-r", str(f))]]

_LOCK_POLL_S = 0.5
_TERMINATE_GRACE_S = 5.0
_TAIL_LINES = 20


def requirements_dir() -> Path | None:
    value = os.environ.get("VLA_REQUIREMENTS_DIR")
    if not value:
        return None
    if not Path(value).is_dir():
        raise InstallError(f"VLA_REQUIREMENTS_DIR={value} is not a directory; the module package is incomplete")
    return Path(value)


def read_policy_type(checkpoint_dir: str) -> str:
    path = Path(checkpoint_dir) / "config.json"
    try:
        return str(json.loads(path.read_text())["type"])
    except (OSError, ValueError, KeyError) as exc:
        raise InstallError(f"cannot read the policy type from {path}: {exc}") from exc


@contextlib.asynccontextmanager
async def _install_lock(path: Path) -> AsyncIterator[None]:
    # LOCK_NB + sleep rather than a blocking flock, so a waiter stays cancellable.
    # flock locks belong to the open file description, so two opens in one
    # process still exclude each other.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(_LOCK_POLL_S)
        yield
    finally:
        os.close(fd)  # releases the lock


async def _stop(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), _TERMINATE_GRACE_S)
    except asyncio.TimeoutError:
        pass
    finally:
        # Also runs if _stop itself is cancelled mid-wait: the group must not survive.
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
    await proc.wait()


async def _run(cmd: list[str], env: dict[str, str]) -> None:
    LOGGER.info("installing: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
        # A long uv output line must not raise ValueError mid-install.
        limit=1 << 20,
        # Own process group, so _stop also reaches uv's children (git, build backends).
        # If viam-server SIGKILLs the module before cancellation reaches _stop, uv is
        # orphaned; the flock is released, so a restarted module may run a second uv
        # in the same venv, and uv's own environment lock is what serializes them.
        start_new_session=True,
    )
    tail: deque[str] = deque(maxlen=_TAIL_LINES)
    try:
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            tail.append(line)
            LOGGER.info("uv: %s", line)
        code = await proc.wait()
    except BaseException:
        # Cancellation (reconfigure, close(), load_timeout_s) or any other error:
        # a multi-GB install must not run on after the load that wanted it is gone.
        await _stop(proc)
        raise
    if code != 0:
        raise InstallError(f"`uv pip install` failed (exit {code}):\n" + "\n".join(tail))


def _marker_matches(marker: Path, digest: str) -> bool:
    try:
        return marker.read_text().strip() == digest
    except OSError:
        return False


async def ensure_installed(
    policy_type: str, *, req_dir: Path | None = None, prefix: Path | None = None
) -> None:
    """Install base + `policy_type`'s extra into this venv, once per lock content."""
    req_dir = req_dir if req_dir is not None else requirements_dir()
    if req_dir is None:
        LOGGER.debug("VLA_REQUIREMENTS_DIR unset; not installing (dev checkout)")
        return
    prefix = prefix if prefix is not None else Path(sys.prefix)
    files = requirement_files(req_dir, policy_type)
    jetson = jetson_settings(os.environ)
    digest = marker_digest(files, jetson)
    # Inside the venv: a recreated venv loses the marker along with the packages.
    marker = prefix / f".vla-installed-{policy_type}.sha256"
    if _marker_matches(marker, digest):
        return
    async with _install_lock(prefix / ".vla-install.lock"):
        if _marker_matches(marker, digest):
            return
        if shutil.which("git") is None:
            raise InstallError("git is required to install LeRobot (a git+https requirement); install git")
        env = dict(os.environ)
        if jetson is not None:
            # UV_TORCH_BACKEND overrides --index-url for torch; the Jetson index must win.
            env.pop("UV_TORCH_BACKEND", None)
        elif sys.platform.startswith("linux"):
            # Without this, Linux resolves PyPI's CUDA 13 torch, which falls back
            # to CPU on cu12-driver hosts. The export has no index lines on purpose.
            env.setdefault("UV_TORCH_BACKEND", "cu128")
        LOGGER.info("installing the %s policy stack (first load; this can take minutes)", policy_type)
        with tempfile.TemporaryDirectory() as tmp:
            for cmd in install_commands(str(prefix / "bin" / "uv"), sys.executable, files, jetson, Path(tmp)):
                await _run(cmd, env)
        importlib.invalidate_caches()
        tmp_marker = marker.with_suffix(".tmp")
        try:
            tmp_marker.write_text(digest)
            os.replace(tmp_marker, marker)
        except OSError as exc:
            raise InstallError(f"installed, but could not write {marker}: {exc}") from exc
        LOGGER.info("%s policy stack installed", policy_type)


async def ensure_installed_for_checkpoint(checkpoint_dir: str, *, prefix: Path | None = None) -> None:
    if requirements_dir() is None:
        return
    await ensure_installed(read_policy_type(checkpoint_dir), prefix=prefix)


if __name__ == "__main__":
    # Pre-provisioning: python -m vla.policy.installer <type> [REQUIREMENTS_DIR]
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: python -m vla.policy.installer <policy_type> [requirements_dir]")
    req = Path(sys.argv[2]) if len(sys.argv) == 3 else requirements_dir()
    if req is None:
        sys.exit("VLA_REQUIREMENTS_DIR is unset and no requirements_dir was given")
    asyncio.run(ensure_installed(sys.argv[1], req_dir=req))
