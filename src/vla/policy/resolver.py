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

from .config import PolicyConfig

LOGGER = logging.getLogger(__name__)

REQUIRED_FILES = ("config.json", "model.safetensors")


class ResolveError(RuntimeError):
    """Raised when a checkpoint cannot be resolved."""


def _cache_dir() -> str:
    module_data = os.environ.get("VIAM_MODULE_DATA")
    if module_data:
        path = Path(module_data) / "checkpoints"
    else:
        # Viam sets VIAM_MODULE_DATA in deployment; on a dev workstation it
        # is absent, so fall back to somewhere writable.
        LOGGER.warning(
            "VIAM_MODULE_DATA unset; caching checkpoints under the system temp dir"
        )
        path = Path(tempfile.gettempdir()) / "viam-vla-checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _verify(directory: str) -> str:
    d = Path(directory)
    if d.is_file():
        raise ResolveError(f"model_path is a file, not a directory: {directory}")
    if not d.is_dir():
        raise ResolveError(f"checkpoint directory does not exist: {directory}")

    missing = [name for name in REQUIRED_FILES if not (d / name).is_file()]
    if missing:
        raise ResolveError(
            f"checkpoint at {directory} is missing required file(s): "
            f"{', '.join(missing)}"
        )
    return str(d)


def resolve_checkpoint(
    cfg: PolicyConfig,
    *,
    snapshot_download: Callable[..., str] | None = None,
) -> str:
    """Return a local directory containing the checkpoint.

    `snapshot_download` is injectable so tests never touch the network.
    """
    if cfg.model_path:
        if cfg.hf_token_env:
            # A local checkpoint has no use for a hub token. Warn rather
            # than raise: a configured-but-unused credential is confusing,
            # but not itself invalid config.
            LOGGER.warning(
                "hf_token_env=%r is set but ignored because model_path is configured",
                cfg.hf_token_env,
            )
        return _verify(cfg.model_path)

    token = None
    if cfg.hf_token_env:
        token = os.environ.get(cfg.hf_token_env)
        if not token:
            raise ResolveError(
                f"hf_token_env names {cfg.hf_token_env!r} but that variable is unset or empty"
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

    # Never interpolate `token` here (or anywhere else in this module) --
    # hf_token_env exists specifically to keep the secret out of config and
    # logs alike.
    LOGGER.info("downloading checkpoint %s@%s", cfg.model_hub_id, cfg.model_revision)
    local = snapshot_download(
        repo_id=cfg.model_hub_id,
        revision=cfg.model_revision,
        cache_dir=_cache_dir(),
        token=token,
    )
    return _verify(local)
