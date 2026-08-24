"""Resolve a configured model reference to a local checkpoint directory.

A `PolicyConfig` names a checkpoint one of two ways:

- `model_path` — a plain filesystem path. This also covers Viam registry
  packages, because viam-server interpolates `${packages.ml_model.name}`
  into a real path before the module ever sees the config.
- `model_hub_id` (+ `model_revision`) — a Hugging Face hub repo, downloaded
  and cached locally.

`PolicyConfig.parse` already guarantees exactly one of the two is set, so
this module only has to dispatch on which one it is.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Callable

from vla.config_util import VLAError, redact_secret

from .config import PolicyConfig

LOGGER = logging.getLogger(__name__)

REQUIRED_FILES = ("config.json", "model.safetensors")

# Bounds the initial hub metadata request (etag lookup), so an unreachable
# endpoint fails fast instead of hanging the caller forever. The transfer
# itself is governed separately by huggingface_hub reading the
# HF_HUB_DOWNLOAD_TIMEOUT env var -- there is no per-call parameter for that
# half of the request, so this module has nothing to pass for it.
_ETAG_TIMEOUT_SECONDS = 10

# A denylist, deliberately, not an allowlist. `lerobot-train` writes its
# intermediate checkpoints to `output_dir/checkpoints/NNNNNN/` -- each one a
# full `pretrained_model/` copy plus a `training_state/` with an optimizer
# snapshot roughly half the size of the weights again -- and `push_to_hub`
# uploads the lot. Measured on viamrobotics/smolvla-box-bot: 9.2 GB in the
# repo for a checkpoint whose root files are 868 MB, i.e. ~10x, all of it
# training bookkeeping no inference path ever opens.
#
# An allowlist would download less still, but this module is generic over
# any LeRobot-registered policy (`policy_type` comes from the checkpoint,
# never from config), so an allowlist missing one file a policy we have not
# inspected actually needs turns into a load failure on someone's robot.
# These three patterns are lerobot's own output-directory convention rather
# than a guess about file naming, and getting them wrong costs only
# bandwidth. Note `model.safetensors` at the repo root is never matched by
# `checkpoints/**`, so the flat single-directory layout is untouched.
_IGNORE_PATTERNS = (
    "checkpoints/**",
    "**/training_state/**",
    "*optimizer*",
)


class ResolveError(VLAError, RuntimeError):
    """Raised when a checkpoint cannot be resolved."""


def _cache_dir() -> str:
    module_data = os.environ.get("VIAM_MODULE_DATA")
    if module_data:
        path = Path(module_data) / "checkpoints"
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ResolveError(
                f"could not create checkpoint cache directory {path}: {exc}"
            ) from exc
        return str(path)

    # Viam sets VIAM_MODULE_DATA in deployment; on a dev workstation it is
    # absent. Fall back to a fresh, uniquely-named temp directory rather
    # than a fixed path under a shared /tmp: a predictable name there is a
    # symlink-attack vector from another local user, and mkdtemp() also
    # creates the directory mode 0700 so only this user can read into it.
    # Downloads made through this fallback are not cached across restarts.
    LOGGER.warning(
        "VIAM_MODULE_DATA unset; downloading into a fresh temp directory "
        "this run instead of a persistent cache"
    )
    try:
        return tempfile.mkdtemp(prefix="viam-vla-checkpoints-")
    except OSError as exc:
        raise ResolveError(f"could not create a temp checkpoint cache directory: {exc}") from exc


def _verify(directory: str) -> str:
    if "${" in directory:
        raise ResolveError(
            "model_path contains an uninterpolated package reference -- "
            f"viam-server did not substitute it before this module saw the config: {directory}"
        )

    d = Path(os.path.expanduser(directory))

    try:
        if d.is_symlink() and not d.exists():
            raise ResolveError(
                f"model_path is a symlink whose target is missing: {directory}"
            )
        if d.is_file():
            raise ResolveError(f"model_path is a file, not a directory: {directory}")
        if not d.is_dir():
            raise ResolveError(f"checkpoint directory does not exist: {directory}")
        missing = [name for name in REQUIRED_FILES if not (d / name).is_file()]
    except OSError as exc:
        raise ResolveError(f"could not read checkpoint directory {directory}: {exc}") from exc

    if missing:
        raise ResolveError(
            f"checkpoint at {directory} is missing required file(s): "
            f"{', '.join(missing)}"
        )
    # Absolute + canonical: a relative model_path would otherwise resolve
    # against an unpredictable module cwd, and this value is handed on to
    # backend.load as-is.
    return str(d.resolve())


def resolve_checkpoint(
    cfg: PolicyConfig,
    *,
    snapshot_download: Callable[..., str] | None = None,
) -> str:
    """Return a local directory containing the checkpoint.

    `snapshot_download` is injectable so tests never touch the network.

    When no `hf_token_env` is configured, the hub download deliberately
    passes `token=False` (forcing anonymous access) rather than `token=None`
    (which would silently defer to huggingface_hub's own cached-login
    lookup). Without this, a stale or unrelated token cached on the machine
    running the module -- something this module never asked for and has no
    way to see -- would still be sent, and a fully public repo would fail
    with a misleading `401 Unauthorized` / "Repository Not Found" instead of
    downloading anonymously as configured.
    """
    if cfg.model_path:
        if cfg.hf_token_env:
            # A local checkpoint has no use for a hub token. Warn rather
            # than raise: a configured-but-unused credential is confusing,
            # but not itself invalid config. The value is redacted even
            # here -- config-time validation already restricts the shape of
            # hf_token_env, but a short pasted secret could still pass that
            # check, so this is belt and braces.
            LOGGER.warning(
                "hf_token_env=%s is set but ignored because model_path is configured",
                redact_secret(cfg.hf_token_env),
            )
        return _verify(cfg.model_path)

    # `False`, not `None`. huggingface_hub treats `token=None` as "use the
    # ambient cached token if one exists" (huggingface_hub.utils.
    # get_token_to_send: None falls through to get_token(), which reads
    # ~/.cache/huggingface/token or HF_TOKEN). Only `token=False` actually
    # means anonymous. When no hf_token_env is configured this module has
    # explicitly decided not to authenticate, but `None` would still let a
    # stale or unrelated cached login on the machine get sent -- and if that
    # cached token happens to be invalid, a fully public repo comes back as
    # `401 Unauthorized` / "Repository Not Found", which sends whoever is
    # debugging it looking for a typo in `model_hub_id` that was never there.
    token: str | bool = False
    if cfg.hf_token_env:
        token = os.environ.get(cfg.hf_token_env)
        if not token:
            raise ResolveError(
                f"hf_token_env names {redact_secret(cfg.hf_token_env)} "
                "but that variable is unset or empty"
            )

    if snapshot_download is None:
        try:
            from huggingface_hub import snapshot_download as _sd
        except ImportError as exc:
            raise ResolveError(
                "downloading from the Hugging Face hub requires huggingface_hub, which "
                "ships with the 'lerobot' extra; install it or use model_path instead"
            ) from exc

        snapshot_download = _sd

    cache_dir = _cache_dir()

    # Never interpolate `token` here (or anywhere else in this module) --
    # hf_token_env exists specifically to keep the secret out of config and
    # logs alike.
    LOGGER.info("downloading checkpoint %s@%s", cfg.model_hub_id, cfg.model_revision)
    try:
        local = snapshot_download(
            repo_id=cfg.model_hub_id,
            revision=cfg.model_revision,
            cache_dir=cache_dir,
            token=token,
            etag_timeout=_ETAG_TIMEOUT_SECONDS,
            ignore_patterns=_IGNORE_PATTERNS,
        )
    except OSError as exc:
        raise ResolveError(
            f"failed to download checkpoint {cfg.model_hub_id}@{cfg.model_revision} "
            f"from the Hugging Face hub: {exc}"
        ) from exc
    return _verify(local)
