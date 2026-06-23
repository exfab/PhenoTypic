from __future__ import annotations

from phenotypic.tune.strategy import NoOpChannel, PruningChannel


def test_noop_channel_never_prunes():
    ch = NoOpChannel()
    ch.report(0.5, step=3)  # no-op, must not raise
    assert ch.should_prune() is False


def test_noop_satisfies_protocol():
    assert isinstance(NoOpChannel(), PruningChannel)
