"""CLI for managing neural network model checkpoints.

Invoked via ``python -m phenotypic.nn``. Provides ``download``, ``list``,
and ``clear`` subcommands for SAM2 and micro-sam checkpoints.
"""

from __future__ import annotations

import sys

import click


@click.group()
def nn_cli() -> None:
    """Manage PyTorch model checkpoints (SAM2, micro-sam)."""


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


@nn_cli.command()
@click.option(
    "--model-type",
    type=click.Choice(["sam2", "microsam"]),
    default="sam2",
    show_default=True,
    help="Model family to download.",
)
@click.option(
    "--model-size",
    type=click.Choice(["tiny", "small", "base_plus", "large"]),
    default="tiny",
    show_default=True,
    help="SAM2 model size (ignored for microsam).",
)
@click.option(
    "--model-name",
    default="vit_b_lm",
    show_default=True,
    help="micro-sam model name (ignored for sam2).",
)
@click.option(
    "--all",
    "download_all",
    is_flag=True,
    help="Download all models for the selected type.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-download even if cached.",
)
def download(
    model_type: str,
    model_size: str,
    model_name: str,
    download_all: bool,
    force: bool,
) -> None:
    """Download model checkpoints for offline use."""
    from rich.console import Console

    console = Console()

    if model_type == "sam2":
        from ._checkpoint_manager import Sam2CheckpointManager

        if download_all:
            sizes = list(Sam2CheckpointManager.MODELS)
            console.print(
                f"[cyan]Downloading all SAM2 checkpoints "
                f"({len(sizes)} models)...[/cyan]"
            )
            for size in sizes:
                console.print(f"  [dim]{size}[/dim]")
                try:
                    path = Sam2CheckpointManager.download(size, force=force)
                    console.print(f"    [green]Cached:[/green] {path}")
                except Exception as exc:
                    console.print(f"    [red]Failed:[/red] {exc}")
        else:
            console.print(
                f"[cyan]Downloading SAM2 checkpoint: {model_size}[/cyan]"
            )
            try:
                path = Sam2CheckpointManager.download(model_size, force=force)  # type: ignore[arg-type]
                console.print(f"[green]Cached:[/green] {path}")
            except Exception as exc:
                console.print(f"[red]Failed:[/red] {exc}")
                sys.exit(1)

    elif model_type == "microsam":
        from ._checkpoint_manager import MicroSamCheckpointManager

        if download_all:
            models = list(MicroSamCheckpointManager.MODELS)
            console.print(
                f"[cyan]Downloading all micro-sam checkpoints "
                f"({len(models)} models)...[/cyan]"
            )
            for mt in models:
                console.print(f"  [dim]{mt}[/dim]")
                try:
                    MicroSamCheckpointManager.download(mt)  # type: ignore[arg-type]
                    console.print("    [green]Done[/green]")
                except Exception as exc:
                    console.print(f"    [red]Failed:[/red] {exc}")
        else:
            console.print(
                f"[cyan]Downloading micro-sam checkpoint: {model_name}[/cyan]"
            )
            try:
                MicroSamCheckpointManager.download(model_name)  # type: ignore[arg-type]
                console.print("[green]Done[/green]")
            except Exception as exc:
                console.print(f"[red]Failed:[/red] {exc}")
                sys.exit(1)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@nn_cli.command("list")
def list_models() -> None:
    """List cached model checkpoints."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # --- SAM2 ---
    sam2_table = Table(title="SAM2 Checkpoints", show_lines=False)
    sam2_table.add_column("Size", style="cyan", no_wrap=True)
    sam2_table.add_column("Filename", style="white")
    sam2_table.add_column("Size (MB)", justify="right", style="green")
    sam2_table.add_column("Path", style="dim")

    try:
        from ._checkpoint_manager import Sam2CheckpointManager

        sam2_cached = Sam2CheckpointManager.list_cached()
        if sam2_cached:
            for entry in sam2_cached:
                sam2_table.add_row(
                    entry["model_size"],
                    entry["filename"],
                    str(entry["size_mb"]),
                    entry["path"],
                )
        else:
            sam2_table.add_row("[dim]No cached checkpoints[/dim]", "", "", "")
    except ImportError:
        sam2_table.add_row(
            "[yellow]torch not installed[/yellow]", "", "", ""
        )

    console.print(sam2_table)
    console.print()

    # --- micro-sam ---
    msam_table = Table(title="micro-sam Checkpoints", show_lines=False)
    msam_table.add_column("Model", style="cyan", no_wrap=True)
    msam_table.add_column("Description", style="white")
    msam_table.add_column("Size (MB)", justify="right", style="green")
    msam_table.add_column("Path", style="dim")

    try:
        from ._checkpoint_manager import MicroSamCheckpointManager

        msam_cached = MicroSamCheckpointManager.list_cached()
        if msam_cached:
            for entry in msam_cached:
                msam_table.add_row(
                    entry["model_type"],
                    entry["description"],
                    str(entry["size_mb"]),
                    entry["path"],
                )
        else:
            msam_table.add_row("[dim]No cached checkpoints[/dim]", "", "", "")
    except Exception:
        msam_table.add_row(
            "[dim]No cached checkpoints[/dim]", "", "", ""
        )

    console.print(msam_table)


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


@nn_cli.command()
@click.option(
    "--model-type",
    type=click.Choice(["sam2", "microsam", "all"]),
    default="all",
    show_default=True,
    help="Which model family to clear.",
)
@click.confirmation_option(prompt="Delete cached model checkpoints?")
def clear(model_type: str) -> None:
    """Delete cached model checkpoints."""
    from rich.console import Console

    console = Console()
    total_deleted: list[str] = []

    if model_type in ("sam2", "all"):
        try:
            from ._checkpoint_manager import Sam2CheckpointManager

            deleted = Sam2CheckpointManager.clear()
            total_deleted.extend(deleted)
            if deleted:
                console.print(
                    f"[green]Cleared {len(deleted)} SAM2 checkpoint(s)[/green]"
                )
            else:
                console.print("[dim]No SAM2 checkpoints to clear[/dim]")
        except ImportError:
            console.print(
                "[yellow]torch not installed — cannot locate SAM2 cache[/yellow]"
            )

    if model_type in ("microsam", "all"):
        try:
            from ._checkpoint_manager import MicroSamCheckpointManager

            deleted = MicroSamCheckpointManager.clear()
            total_deleted.extend(deleted)
            if deleted:
                console.print(
                    f"[green]Cleared {len(deleted)} micro-sam "
                    f"checkpoint(s)[/green]"
                )
            else:
                console.print("[dim]No micro-sam checkpoints to clear[/dim]")
        except Exception:
            console.print("[dim]No micro-sam checkpoints to clear[/dim]")

    if not total_deleted:
        console.print("[dim]Nothing to delete.[/dim]")
    else:
        console.print(
            f"\n[bold]Total deleted: {len(total_deleted)} item(s)[/bold]"
        )
