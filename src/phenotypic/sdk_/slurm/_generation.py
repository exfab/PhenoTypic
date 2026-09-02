"""Safe filesystem keys for lifecycle-generation-owned SLURM artifacts."""

from __future__ import annotations

from hashlib import sha256


def generation_script_key(generation: str) -> str:
    """Map any lifecycle generation to one collision-resistant path component."""
    return sha256(
        generation.encode("utf-8", errors="surrogatepass")
    ).hexdigest()
