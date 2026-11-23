"""Install the heavy policy stack (torch, LeRobot, a policy type's extras) lazily.

setup.sh installs only the core. The policy service calls `ensure_installed`
during its background load, before the first `import torch`. Requirement files
are exported from uv.lock at build time (tools/export_requirements.py) and
found through VLA_REQUIREMENTS_DIR, which run.sh sets; without it (a source
checkout, tests) installing is a no-op. See
docs/superpowers/specs/2026-11-22-lazy-policy-install-design.md.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import re
from pathlib import Path
from typing import Mapping

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
