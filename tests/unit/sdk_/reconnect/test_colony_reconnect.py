import numpy as np

from phenotypic.sdk_.reconnect import ReconnectConfig, identify_pseudo_fragments


def _cfg(**kw) -> ReconnectConfig:
    base = dict(
        beta=2.0, gamma=1.2, delta=1.0, coherence_window_radius=15, mad_window=7,
        gap_crossing_penalty=4.0, border_margin_px=50, frag_reach_px=10,
        tile_size=1200, tile_overlap=500, max_gap_length=30,
        path_dilation_radius=2, snr_margin=5, reconnection_tolerance=2.5,
    )
    base.update(kw)
    return ReconnectConfig(**base)


def test_reconnect_config_is_frozen():
    cfg = _cfg()
    assert cfg.beta == 2.0 and cfg.tile_size == 1200
    import dataclasses, pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.beta = 3.0  # type: ignore[misc]


def test_identify_pseudo_fragments_splits_central_and_fragments():
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[4:8, 4:8] = 1            # CC touching a center
    labels[20:24, 20:24] = 1        # CC NOT touching any center -> fragment
    center = np.zeros((30, 30), dtype=bool)
    center[6, 6] = True
    central_mask, fragment_labels = identify_pseudo_fragments(labels, center)
    assert central_mask[6, 6]
    assert not central_mask[22, 22]
    assert fragment_labels[22, 22] > 0
    assert fragment_labels[6, 6] == 0


def test_build_reconnect_cost_shapes_and_finiteness():
    from phenotypic.sdk_.reconnect import build_reconnect_cost
    rng = np.random.default_rng(0)
    H, W = 40, 40
    pc_sum = rng.random((H, W)).astype(np.float32) * 0.15
    M = rng.random((H, W)).astype(np.float32)
    m = rng.random((H, W)).astype(np.float32) * 0.5
    orientation = (rng.random((H, W)).astype(np.float32) - 0.5) * np.pi
    enhanced = rng.random((H, W)).astype(np.float32)
    colony_labels = np.zeros((H, W), dtype=np.int32)
    colony_labels[10:14, 10:30] = 1
    central = colony_labels > 0
    # border_margin_px small so the border ramp does not blanket this 40x40 image
    # (default 50 > the image's 19px max dist-to-edge would wash out the structure mask);
    # pixel [12,20] has dist-to-edge 12, so a margin of 2 leaves it in the interior.
    unmasked, masked = build_reconnect_cost(
        pc_sum, M, m, orientation, enhanced, colony_labels, central,
        _cfg(mad_window=7, border_margin_px=2),
    )
    assert unmasked.shape == masked.shape == (H, W)
    assert np.all(np.isfinite(unmasked))
    # inside the colony/central mask the masked surface is driven to near-zero traversal cost
    assert masked[12, 20] < unmasked[12, 20] or masked[12, 20] == 0.0


def test_reconnect_fragments_tiled_bridges_gap():
    from phenotypic.sdk_.reconnect import build_reconnect_cost, reconnect_fragments_tiled, identify_pseudo_fragments
    H, W = 60, 60
    # A colony bar + a nearby fragment separated by a 3px gap, both on a low-cost ridge.
    # The ridge is 12 rows thick (24:36) rather than a thin line: the path-quality
    # calibration derives its cost threshold from a single colony skeleton branch
    # (IQR == 0, zero tolerance band), so the traversed gap cost must match the
    # colony's own branch cost. A thick ridge keeps the local-MAD window (radius 5)
    # entirely inside the ridge for interior pixels, so the gap composite cost equals
    # the colony branch cost and the F1 median-cost filter admits the bridge. This is
    # a genuine reconnection: the fragment does not touch the inoculum center and is
    # only painted with the colony id after a valid low-cost bridge is found.
    colony = np.zeros((H, W), dtype=np.int32)
    colony[24:36, 8:28] = 1
    frag_src = np.zeros((H, W), dtype=bool)
    frag_src[24:36, 31:50] = True
    branch = (colony > 0) | frag_src
    # high pc_sum ridge along the bar + fragment + gap so Dijkstra can route cheaply
    pc = np.zeros((H, W), dtype=np.float32)
    pc[24:36, 8:50] = 0.6
    M = pc.copy(); m = pc * 0.2
    orient = np.zeros((H, W), dtype=np.float32)  # all-horizontal
    enhanced = pc.copy()
    central_mask, fragment_labels = identify_pseudo_fragments(
        np.where(branch, 1, 0).astype(np.int32), colony > 0,
    )
    # sanity: the fragment is genuinely disconnected from the colony center
    assert fragment_labels.max() > 0
    assert not central_mask[29, 40]
    cfg = _cfg(tile_size=60, tile_overlap=20, mad_window=5, frag_reach_px=8, max_gap_length=20, border_margin_px=2)
    unmasked, cost = build_reconnect_cost(pc, M, m, orient, enhanced, colony, central_mask, cfg)
    out = reconnect_fragments_tiled(
        colony, fragment_labels, cost, unmasked,
        pc.astype(np.float32), enhanced.astype(np.float32), cfg,
    )
    # the fragment pixels get painted with the colony id (1) after reconnection
    assert out[29, 40] == 1


def test_reconnect_fragments_tiled_noop_without_fragments():
    from phenotypic.sdk_.reconnect import reconnect_fragments_tiled
    colony = np.zeros((20, 20), dtype=np.int32)
    colony[5:9, 5:15] = 1
    empty_frags = np.zeros((20, 20), dtype=np.int32)
    cost = np.ones((20, 20), dtype=np.float64)
    out = reconnect_fragments_tiled(colony, empty_frags, cost, cost, np.zeros((20, 20), np.float32), np.zeros((20, 20), np.float32), _cfg())
    assert np.array_equal(out, colony)


def test_select_reconnect_fragments_pseudo_matches_identify():
    from phenotypic.sdk_.reconnect import select_reconnect_fragments, identify_pseudo_fragments
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[4:8, 4:8] = 1            # CC touching a center
    labels[20:24, 20:24] = 1        # CC not touching a center -> pseudo-fragment
    center = np.zeros((30, 30), dtype=bool); center[6, 6] = True
    colony_mask = labels > 0        # ignored by the pseudo path
    structure_mask = labels > 0
    c0, f0 = identify_pseudo_fragments(labels, center)
    c1, f1 = select_reconnect_fragments(labels, center, colony_mask, structure_mask, scope="pseudo")
    assert np.array_equal(c0, c1)
    assert np.array_equal(f0, f1)


def test_select_reconnect_fragments_branches_admits_disconnected():
    from phenotypic.sdk_.reconnect import select_reconnect_fragments
    center = np.zeros((30, 40), dtype=bool); center[10, 10] = True
    structure_mask = np.zeros((30, 40), dtype=bool); structure_mask[8:13, 8:20] = True  # body on center
    colony_labels = np.where(structure_mask, 1, 0).astype(np.int32)
    colony_mask = structure_mask.copy()
    colony_mask[8:13, 28:36] = True        # disconnected branch fragment, dropped by the overlap filter
    central, frags = select_reconnect_fragments(
        colony_labels, center, colony_mask, structure_mask, scope="branches")
    assert central[10, 10]                 # body is central
    assert frags[10, 30] > 0               # the disconnected fragment is a reconnect candidate
    # pseudo scope must NOT admit it
    _, frags_pseudo = select_reconnect_fragments(
        colony_labels, center, colony_mask, structure_mask, scope="pseudo")
    assert frags_pseudo[10, 30] == 0


def test_select_reconnect_fragments_min_size_drops_specks():
    from phenotypic.sdk_.reconnect import select_reconnect_fragments
    center = np.zeros((30, 40), dtype=bool); center[10, 10] = True
    structure_mask = np.zeros((30, 40), dtype=bool); structure_mask[8:13, 8:20] = True
    colony_labels = np.where(structure_mask, 1, 0).astype(np.int32)
    colony_mask = structure_mask.copy()
    colony_mask[8:13, 28:36] = True        # 40-px real fragment (kept)
    colony_mask[0, 0] = True               # 1-px speck (dropped)
    _, frags = select_reconnect_fragments(
        colony_labels, center, colony_mask, structure_mask, scope="branches", min_fragment_size=5)
    assert frags[10, 30] > 0
    assert frags[0, 0] == 0


def test_select_reconnect_fragments_rejects_bad_scope():
    import pytest
    from phenotypic.sdk_.reconnect import select_reconnect_fragments
    z = np.zeros((5, 5), dtype=np.int32); b = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError):
        select_reconnect_fragments(z, b, b, b, scope="nonsense")  # type: ignore[arg-type]
