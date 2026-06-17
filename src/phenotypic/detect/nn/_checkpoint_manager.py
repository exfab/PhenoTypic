"""Checkpoint download, caching, and device resolution for neural network detectors.

Provides :class:`Sam2CheckpointManager` and :class:`MicroSamCheckpointManager`
for offline-friendly checkpoint management (download/list/clear), plus
:func:`resolve_device` for GPU auto-detection.

Type aliases defined here (:data:`Sam2ModelSize`, :data:`MicroSamModelType`,
:data:`Device`, :data:`ResolvedDevice`) are the single source of truth and
should be imported by detector modules.

All ``torch`` imports are **lazy** (inside methods/functions) so the module
can be imported without PyTorch installed.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases — single source of truth, imported by detectors
# ---------------------------------------------------------------------------

Sam2ModelSize = Literal["tiny", "small", "base_plus", "large"]

MicroSamModelType = Literal[
    "vit_t",
    "vit_b",
    "vit_l",
    "vit_h",
    "vit_t_lm",
    "vit_b_lm",
    "vit_l_lm",
    "vit_b_em_organelles",
    "vit_l_em_organelles",
]

Device = Literal[
    "auto",
    "cpu",
    "cuda",
    "mps",
    "xpu",
    "hpu",
    "xla",
    "ipu",
    "hip",
    "ve",
    "fpga",
    "ort",
    "lazy",
    "vulkan",
    "meta",
    "mtia",
    "privateuseone",
]

ResolvedDevice = Literal[
    "cpu",
    "cuda",
    "mps",
    "xpu",
    "hpu",
    "xla",
    "ipu",
    "hip",
    "ve",
    "fpga",
    "ort",
    "lazy",
    "vulkan",
    "meta",
    "mtia",
    "privateuseone",
]


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def _check_hpu() -> bool:
    """Check Habana Gaudi (HPU) availability."""
    try:
        import habana_frameworks.torch  # noqa: F401
        import torch

        return torch.hpu.is_available()  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        return False


def _build_accelerator_checks() -> list[tuple[str, Callable[[], bool]]]:
    """Build the accelerator check list.

    Constructed lazily so ``torch`` is only imported when called.
    """
    import torch

    return [
        ("cuda", lambda: torch.cuda.is_available()),
        (
            "mps",
            lambda: hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available(),
        ),
        (
            "xpu",
            lambda: hasattr(torch, "xpu") and torch.xpu.is_available(),
        ),
        ("hpu", _check_hpu),
    ]


def resolve_device(
    device: Device = "auto",
    allow_cpu: bool = False,
) -> ResolvedDevice:
    """Resolve a device string to an available PyTorch device.

    Args:
        device: ``"auto"`` probes accelerators in priority order, or pass
            any PyTorch device string (``"cuda"``, ``"mps"``, ``"cpu"``, etc.).
        allow_cpu: When *True* and no accelerator is found in ``"auto"``
            mode, fall back to CPU with a warning instead of raising.

    Returns:
        Resolved device string suitable for ``torch.device()``.

    Raises:
        RuntimeError: If ``device="auto"``, no accelerator is found, and
            *allow_cpu* is *False*.
        RuntimeError: If an explicit accelerator is requested but
            unavailable (CUDA, MPS, or XPU).
    """
    import torch  # noqa: F811 — lazy import

    if device == "auto":
        for name, check in _build_accelerator_checks():
            if check():
                return name  # type: ignore[return-value]
        if allow_cpu:
            warnings.warn(
                "No GPU/accelerator detected — inference will be very slow "
                "on CPU.",
                UserWarning,
                stacklevel=2,
            )
            return "cpu"
        raise RuntimeError(
            "No accelerator available. GPU-based detectors require a GPU "
            "(CUDA, Apple MPS, Intel XPU, etc.). Set device='cpu' to "
            "force CPU (very slow)."
        )

    # Explicit device requested — validate known accelerators
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' requested but CUDA is not available."
        )
    if device == "mps":
        if not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError(
                "device='mps' requested but MPS is not available."
            )
    if device == "xpu":
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError(
                "device='xpu' requested but XPU is not available."
            )

    return device  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Sam2CheckpointManager
# ---------------------------------------------------------------------------


class Sam2CheckpointManager:
    """Download, cache, and manage SAM2 model checkpoints.

    Checkpoints are stored in the standard ``torch.hub`` cache directory
    (``~/.cache/torch/hub/checkpoints/`` by default, respects ``TORCH_HOME``
    and ``TORCH_HUB`` environment variables).

    All ``torch`` imports are deferred to individual methods so this class
    can be imported and inspected without PyTorch installed.
    """

    BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"

    MODELS: dict[str, dict[str, str]] = {
        "tiny": {
            "filename": "sam2.1_hiera_tiny.pt",
            "config": "sam2.1/sam2.1_hiera_t.yaml",
        },
        "small": {
            "filename": "sam2.1_hiera_small.pt",
            "config": "sam2.1/sam2.1_hiera_s.yaml",
        },
        "base_plus": {
            "filename": "sam2.1_hiera_base_plus.pt",
            "config": "sam2.1/sam2.1_hiera_b+.yaml",
        },
        "large": {
            "filename": "sam2.1_hiera_large.pt",
            "config": "sam2.1/sam2.1_hiera_l.yaml",
        },
    }

    # ------------------------------------------------------------------
    # Cache location
    # ------------------------------------------------------------------

    @staticmethod
    def cache_dir() -> Path:
        """Return the checkpoint cache directory.

        Returns:
            Path to ``<torch_hub_dir>/checkpoints``.
        """
        import torch.hub

        return Path(torch.hub.get_dir()) / "checkpoints"

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @classmethod
    def is_cached(cls, model_size: Sam2ModelSize) -> bool:
        """Check whether a checkpoint is already downloaded.

        Args:
            model_size: One of ``"tiny"``, ``"small"``, ``"base_plus"``,
                ``"large"``.

        Returns:
            *True* if the checkpoint file exists on disk.
        """
        info = cls.MODELS[model_size]
        return (cls.cache_dir() / info["filename"]).is_file()

    @classmethod
    def get_config(cls, model_size: Sam2ModelSize) -> str:
        """Return the SAM2 config YAML identifier for *model_size*.

        Args:
            model_size: One of ``"tiny"``, ``"small"``, ``"base_plus"``,
                ``"large"``.

        Returns:
            Config string (e.g. ``"sam2.1/sam2.1_hiera_t.yaml"``).
        """
        return cls.MODELS[model_size]["config"]

    # ------------------------------------------------------------------
    # Download / retrieve
    # ------------------------------------------------------------------

    @classmethod
    def download(
        cls,
        model_size: Sam2ModelSize,
        *,
        force: bool = False,
    ) -> Path:
        """Download a SAM2 checkpoint if not already cached.

        Args:
            model_size: One of ``"tiny"``, ``"small"``, ``"base_plus"``,
                ``"large"``.
            force: Re-download even if the file already exists.

        Returns:
            Path to the downloaded checkpoint file.
        """
        import torch.hub

        info = cls.MODELS[model_size]
        cache = cls.cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        dest = cache / info["filename"]

        if dest.is_file() and not force:
            logger.info("Checkpoint already cached: %s", dest)
            return dest

        url = cls.BASE_URL + info["filename"]
        logger.info("Downloading %s → %s", url, dest)
        torch.hub.download_url_to_file(url, str(dest))
        return dest

    @classmethod
    def get_checkpoint(cls, model_size: Sam2ModelSize) -> Path:
        """Return the checkpoint path, downloading if absent.

        Args:
            model_size: One of ``"tiny"``, ``"small"``, ``"base_plus"``,
                ``"large"``.

        Returns:
            Path to the checkpoint file.
        """
        return cls.download(model_size)

    # ------------------------------------------------------------------
    # Listing / cleanup
    # ------------------------------------------------------------------

    @classmethod
    def list_cached(cls) -> list[dict[str, Any]]:
        """Enumerate cached SAM2 checkpoints.

        Returns:
            List of dicts with keys ``"model_size"``, ``"filename"``,
            ``"path"``, ``"size_mb"``.
        """
        results: list[dict[str, Any]] = []
        for size, info in cls.MODELS.items():
            path = cls.cache_dir() / info["filename"]
            if path.is_file():
                results.append(
                    {
                        "model_size": size,
                        "filename": info["filename"],
                        "path": str(path),
                        "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
                    }
                )
        return results

    @classmethod
    def clear(
        cls,
        model_size: Sam2ModelSize | None = None,
    ) -> list[str]:
        """Delete cached checkpoint(s).

        Args:
            model_size: Specific size to delete, or *None* to delete all
                SAM2 checkpoints.

        Returns:
            List of deleted filenames.
        """
        deleted: list[str] = []
        sizes = [model_size] if model_size else list(cls.MODELS)
        for size in sizes:
            info = cls.MODELS[size]  # type: ignore[index]
            path = cls.cache_dir() / info["filename"]
            if path.is_file():
                path.unlink()
                deleted.append(info["filename"])
                logger.info("Deleted %s", path)
        return deleted


# ---------------------------------------------------------------------------
# MicroSamCheckpointManager
# ---------------------------------------------------------------------------


class MicroSamCheckpointManager:
    """Manage micro-sam model checkpoints.

    micro-sam handles its own caching via ``platformdirs`` (respects the
    ``MICROSAM_CACHEDIR`` environment variable). This wrapper provides a
    unified CLI interface for download/list/clear operations.
    """

    MODELS: dict[str, str] = {
        # Base SAM models
        "vit_t": "ViT-Tiny, base SAM",
        "vit_b": "ViT-Base, base SAM",
        "vit_l": "ViT-Large, base SAM",
        "vit_h": "ViT-Huge, base SAM",
        # Light microscopy (finetuned)
        "vit_t_lm": "ViT-Tiny, light microscopy",
        "vit_b_lm": "ViT-Base, light microscopy (default)",
        "vit_l_lm": "ViT-Large, light microscopy",
        # Electron microscopy (finetuned)
        "vit_b_em_organelles": "ViT-Base, EM organelles",
        "vit_l_em_organelles": "ViT-Large, EM organelles",
    }

    # ------------------------------------------------------------------
    # Cache location
    # ------------------------------------------------------------------

    @staticmethod
    def cache_dir() -> Path:
        """Return micro-sam's model cache directory.

        Returns:
            Path to the cache directory used by micro-sam.
        """
        try:
            from micro_sam.util import _get_default_model_folder

            return Path(_get_default_model_folder())
        except ImportError:
            # Provide a reasonable fallback path for display purposes
            import os

            env = os.environ.get("MICROSAM_CACHEDIR")
            if env:
                return Path(env)
            # platformdirs default
            try:
                from platformdirs import user_cache_dir

                return Path(user_cache_dir("micro_sam"))
            except ImportError:
                return Path.home() / ".cache" / "micro_sam"

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    @classmethod
    def download(cls, model_type: MicroSamModelType) -> None:
        """Download a micro-sam model checkpoint.

        Delegates to micro-sam's own download machinery which handles
        caching internally.

        Args:
            model_type: micro-sam model identifier (e.g. ``"vit_b_lm"``).

        Raises:
            ImportError: If micro-sam is not installed.
        """
        try:
            from micro_sam.util import get_sam_model
        except ImportError:
            raise ImportError(
                "micro-sam is not installed. "
                "Install with: pip install phenotypic[torch]"
            ) from None

        logger.info("Downloading micro-sam model: %s", model_type)
        get_sam_model(model_type=model_type)

    # ------------------------------------------------------------------
    # Listing / cleanup
    # ------------------------------------------------------------------

    @classmethod
    def list_cached(cls) -> list[dict[str, Any]]:
        """Enumerate cached micro-sam checkpoints.

        Returns:
            List of dicts with keys ``"model_type"``, ``"description"``,
            ``"path"``, ``"size_mb"``.
        """
        results: list[dict[str, Any]] = []
        cache = cls.cache_dir()
        if not cache.is_dir():
            return results

        for model_type, desc in cls.MODELS.items():
            # micro-sam stores models in subdirectories named after the type
            model_dir = cache / model_type
            if model_dir.is_dir():
                total_size = sum(
                    f.stat().st_size
                    for f in model_dir.rglob("*")
                    if f.is_file()
                )
                results.append(
                    {
                        "model_type": model_type,
                        "description": desc,
                        "path": str(model_dir),
                        "size_mb": round(total_size / (1024 * 1024), 1),
                    }
                )
            else:
                # Check for checkpoint files matching the model type name
                for ckpt in cache.glob(f"*{model_type}*"):
                    if ckpt.is_file():
                        results.append(
                            {
                                "model_type": model_type,
                                "description": desc,
                                "path": str(ckpt),
                                "size_mb": round(
                                    ckpt.stat().st_size / (1024 * 1024), 1
                                ),
                            }
                        )
                        break

        return results

    @classmethod
    def clear(
        cls,
        model_type: MicroSamModelType | None = None,
    ) -> list[str]:
        """Delete cached micro-sam checkpoint(s).

        Args:
            model_type: Specific model to delete, or *None* to delete all
                micro-sam checkpoints.

        Returns:
            List of deleted paths.
        """
        import shutil

        deleted: list[str] = []
        cache = cls.cache_dir()
        if not cache.is_dir():
            return deleted

        types = [model_type] if model_type else list(cls.MODELS)
        for mt in types:
            # Try directory-based storage
            model_dir = cache / mt  # type: ignore[operator]
            if model_dir.is_dir():
                shutil.rmtree(model_dir)
                deleted.append(str(model_dir))
                logger.info("Deleted %s", model_dir)
                continue

            # Try file-based storage
            for ckpt in cache.glob(f"*{mt}*"):  # type: ignore[arg-type]
                if ckpt.is_file():
                    ckpt.unlink()
                    deleted.append(str(ckpt))
                    logger.info("Deleted %s", ckpt)

        return deleted


# ---------------------------------------------------------------------------
# Gated foundation-model checkpoint managers (Spec 2a) — Hugging Face pulls
# ---------------------------------------------------------------------------
#
# SAM3 and the DINO backbones live on the Hugging Face Hub, not the SAM2
# ``torch.hub`` host. They pull a *snapshot* (multi-file repo) rather than a
# single ``.pt``, so they use ``huggingface_hub.snapshot_download`` instead of
# ``Sam2CheckpointManager``'s ``torch.hub.download_url_to_file``. The managers
# are deliberately **instance-style** (carry a ``repo_id`` per-instance for the
# size-parameterised DINOv2 case) — a different shape from the classmethod-only
# ``Sam2CheckpointManager`` because the underlying retrieval API differs.
#
# ``huggingface_hub`` is imported lazily inside the module-level
# :func:`snapshot_download` indirection so this module stays importable without
# it (mirrors the torch-lazy pattern). Tests patch the single name
# ``_checkpoint_manager.snapshot_download``.


def _hf_snapshot_download(**kwargs: Any) -> str:
    """Lazy-import wrapper around ``huggingface_hub.snapshot_download``.

    Args:
        **kwargs: Forwarded verbatim to ``huggingface_hub.snapshot_download``
            (``repo_id``, ``token``, ``revision``, ...).

    Returns:
        The local cache path of the downloaded snapshot.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
    """
    try:
        from huggingface_hub import snapshot_download as _dl
    except ImportError:
        raise ImportError(
            "Downloading foundation-model weights requires huggingface_hub. "
            "Install with: pip install phenotypic[foundation]"
        ) from None
    return _dl(**kwargs)


#: Module-level indirection so tests can patch one name. Re-assigned (not a
#: direct alias to the function object) so ``monkeypatch.setattr`` on this
#: attribute is the single override point.
snapshot_download = _hf_snapshot_download


class _GatedRepoError(Exception):
    """Local stand-in for ``huggingface_hub.errors.GatedRepoError``.

    Used by tests to simulate a 403/gated response without importing
    ``huggingface_hub``. :func:`_is_gated_or_auth_error` recognises both this
    class and the real ``huggingface_hub`` error hierarchy.
    """


def _is_gated_or_auth_error(exc: Exception) -> bool:
    """Return ``True`` if *exc* signals a gated/unauthorized Hugging Face pull.

    Matches the real ``huggingface_hub`` error hierarchy
    (``GatedRepoError`` / ``RepositoryNotFoundError`` / a ``401``/``403``
    ``HfHubHTTPError``) when it is importable, the local
    :class:`_GatedRepoError` test stand-in, and finally falls back to a string
    match on ``"401"`` / ``"403"`` / ``"gated"`` / ``"access"`` so a wrapped or
    re-raised error is still caught.

    Args:
        exc: The exception raised by :func:`snapshot_download`.

    Returns:
        Whether the error is an access/authentication failure (vs. an
        unrelated error like a disk-full ``OSError``, which should propagate).
    """
    if isinstance(exc, _GatedRepoError):
        return True
    try:
        from huggingface_hub.errors import (
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )

        if isinstance(exc, (GatedRepoError, RepositoryNotFoundError)):
            return True
        if isinstance(exc, HfHubHTTPError):
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403):
                return True
    except ImportError:
        pass
    text = str(exc).lower()
    return any(tok in text for tok in ("401", "403", "gated", "access"))


class Sam3CheckpointManager:
    """Download and cache Meta's gated SAM3 weights from the Hugging Face Hub.

    SAM3 weights (~3.45 GB) are **gated**: the user must accept the SAM License
    on the model page and authenticate locally (``hf auth login`` or
    ``HF_TOKEN``). PhenoTypic adds an *informational* acceptance gate on top of
    the binding Hugging Face gate (:func:`require_license_acceptance`) before any
    network call, and reworries a 401/403 into an actionable message.

    All ``huggingface_hub`` imports are deferred to
    :func:`snapshot_download` so this class can be imported and inspected
    without ``huggingface_hub`` installed.
    """

    repo_id = "facebook/sam3"
    license_key = "sam3"
    license_name = "SAM License"
    license_url = "https://huggingface.co/facebook/sam3"

    def download(self, *, interactive: bool = True) -> str:
        """Download the SAM3 snapshot, gated on license acceptance.

        Args:
            interactive: When *True*, fall back to a terminal y/N prompt if
                ``PHENOTYPIC_ACCEPT_MODEL_LICENSE`` does not already grant
                acceptance. SLURM/batch callers pass ``False``.

        Returns:
            The local cache path of the downloaded snapshot.

        Raises:
            RuntimeError: If the license has not been accepted, or if the pull
                fails because access was not granted / no token is present.
        """
        require_license_acceptance(
            self.license_key, self.license_name, self.license_url,
            interactive=interactive,
        )
        try:
            return snapshot_download(repo_id=self.repo_id)
        except Exception as exc:
            if _is_gated_or_auth_error(exc):
                raise RuntimeError(
                    f"Cannot download {self.repo_id}: access not granted or no "
                    f"token. Request access at {self.license_url}, then run "
                    f"`uv run hf auth login` (or export HF_TOKEN)."
                ) from exc
            raise


class Dinov2CheckpointManager:
    """Download and cache an **ungated** DINOv2 backbone from the Hub.

    DINOv2 is Apache-2.0 and ungated — no license-acceptance handshake and no
    token are required. The ``size`` maps to the Hugging Face id
    (``facebook/dinov2-{small|base|large}``).

    All ``huggingface_hub`` imports are deferred to
    :func:`snapshot_download`.
    """

    _SIZE_TO_REPO: dict[str, str] = {
        "small": "facebook/dinov2-small",
        "base": "facebook/dinov2-base",
        "large": "facebook/dinov2-large",
    }

    def __init__(self, *, size: str = "base") -> None:
        if size not in self._SIZE_TO_REPO:
            raise ValueError(
                f"Unknown DINOv2 size {size!r}; expected one of "
                f"{sorted(self._SIZE_TO_REPO)}."
            )
        self.size = size

    @property
    def repo_id(self) -> str:
        """The Hugging Face repo id for this manager's ``size``."""
        return self._SIZE_TO_REPO[self.size]

    def download(self) -> str:
        """Download the DINOv2 snapshot (no acceptance gate, no token).

        Returns:
            The local cache path of the downloaded snapshot.
        """
        return snapshot_download(repo_id=self.repo_id)


class Dinov3CheckpointManager:
    """Download and cache Meta's **gated** DINOv3 backbone from the Hub.

    DINOv3 is a hybrid of the two existing managers: it carries the
    ``Dinov2CheckpointManager``'s size-parameterised constructor
    (``__init__(self, *, size)`` mapping to the three
    ``dinov3-vit{s|b|l}16-pretrain-lvd1689m`` ids), but — unlike the ungated
    DINOv2 — it is **gated** under the DINOv3 License, so it runs SAM3's
    acceptance gate (:func:`require_license_acceptance`) before any network
    call and reworries a 401/403 into an actionable message.

    All ``huggingface_hub`` imports are deferred to :func:`snapshot_download`
    so this class can be imported and inspected without ``huggingface_hub``.
    """

    _SIZE_TO_REPO: dict[str, str] = {
        "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    }

    license_key = "dinov3"
    license_name = "DINOv3 License"
    license_url = (
        "https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m"
    )

    def __init__(self, *, size: str = "base") -> None:
        if size not in self._SIZE_TO_REPO:
            raise ValueError(
                f"Unknown DINOv3 size {size!r}; expected one of "
                f"{sorted(self._SIZE_TO_REPO)}."
            )
        self.size = size

    @property
    def repo_id(self) -> str:
        """The Hugging Face repo id for this manager's ``size``."""
        return self._SIZE_TO_REPO[self.size]

    def download(self, *, interactive: bool = True) -> str:
        """Download the DINOv3 snapshot, gated on license acceptance.

        Args:
            interactive: When *True*, fall back to a terminal y/N prompt if
                ``PHENOTYPIC_ACCEPT_MODEL_LICENSE`` does not already grant
                acceptance. SLURM/batch callers pass ``False``.

        Returns:
            The local cache path of the downloaded snapshot.

        Raises:
            RuntimeError: If the license has not been accepted, or if the pull
                fails because access was not granted / no token is present.
        """
        require_license_acceptance(
            self.license_key, self.license_name, self.license_url,
            interactive=interactive,
        )
        try:
            return snapshot_download(repo_id=self.repo_id)
        except Exception as exc:
            if _is_gated_or_auth_error(exc):
                raise RuntimeError(
                    f"Cannot download {self.repo_id}: access not granted or no "
                    f"token. Request access at {self.license_url}, then run "
                    f"`uv run hf auth login` (or export HF_TOKEN)."
                ) from exc
            raise


def require_license_acceptance(
    model: str, license_name: str, license_url: str, *, interactive: bool = True
) -> None:
    """Gate a gated-weights download on the user accepting the model's license.

    Acceptance is satisfied by ``PHENOTYPIC_ACCEPT_MODEL_LICENSE`` (a comma list
    of model names) for non-interactive / batch use, or by an interactive y/N
    prompt. Ungated components (SAM2, micro-sam) never call this; the hook exists
    for the gated foundation models added by later work (SAM3, DINOv3).

    Args:
        model: Model name to check against the accepted set (case-insensitive).
        license_name: Human-readable license name shown in the prompt/error.
        license_url: Where the user can read the license.
        interactive: When True, fall back to a terminal y/N prompt if the env
            var does not already grant acceptance.

    Raises:
        RuntimeError: If the license has not been accepted.
    """
    accepted = {
        m.strip().lower()
        for m in os.environ.get("PHENOTYPIC_ACCEPT_MODEL_LICENSE", "").split(",")
        if m.strip()
    }
    if model.lower() in accepted:
        return
    if interactive:
        print(f"\n{model} weights are under the {license_name}: {license_url}")
        resp = input(f"Accept the {license_name} to download {model}? [y/N] ")
        if resp.strip().lower() in ("y", "yes"):
            return
    raise RuntimeError(
        f"{model} weights require accepting the {license_name} license "
        f"({license_url}). Re-run after setting "
        f"PHENOTYPIC_ACCEPT_MODEL_LICENSE={model} "
        f"(and `hf auth login` for gated Hugging Face models)."
    )
