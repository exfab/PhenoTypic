"""Behavioural contract for the Contrast* enhancers.

Covers the three new pointwise curves (:class:`ContrastGamma`,
:class:`ContrastLog`, :class:`ContrastSigmoid`) and the ``input_layer`` /
``keep_colors`` retrofit of :class:`ContrastStretching`.

Three families of assertion carry the design:

- **Field order** — both mixins append, so every op ends
  ``[...op params, norm, input_layer]``.
- **`norm` semantics** — ``gain`` is a post-curve multiplier on ``adjust_gamma``
  and ``adjust_log``, so ``norm="rescale"`` divides it back out; on
  ``adjust_sigmoid`` it lives inside the exponent and survives.
- **`input_layer` semantics** — running the curve on RGB and *then* projecting
  differs from projecting first, but only for a ``detect_mode`` that genuinely
  mixes channels. See :func:`test_rgb_and_detect_mat_agree_under_selection_modes`.
"""

import numpy as np
import pytest
from skimage.exposure import adjust_gamma

from phenotypic.data import load_synth_yeast_plate
from phenotypic.enhance import ContrastGamma

# ``detect_mode``s whose projection is a per-pixel *selection* (an order
# statistic or a single channel). Any monotonically increasing pointwise curve
# ``f`` commutes with them: ``min(f(r), f(g), f(b)) == f(min(r, g, b))``.
SELECTION_MODES = ("red", "green", "blue", "MinRGB", "HsvV")

# A ``detect_mode`` that forms a genuine mix of all three channels, so a
# non-linear curve does *not* commute with the projection.
MIXING_MODE = "LabA"


# --------------------------------------------------------------------------- #
# ContrastGamma
# --------------------------------------------------------------------------- #
def test_field_order_is_params_then_norm_then_input_layer():
    assert list(ContrastGamma.model_fields) == ["gamma", "gain", "norm", "input_layer"]


def test_identity_at_defaults():
    """gamma=1, gain=1 is the identity curve; clip is a no-op on in-range input."""
    image = load_synth_yeast_plate()
    before = image.detect_mat[:].copy()
    after = ContrastGamma().apply(image).detect_mat[:]
    np.testing.assert_allclose(after, before, atol=1e-6)


def test_gamma_gt_one_darkens_midtones():
    image = load_synth_yeast_plate()
    before = image.detect_mat[:].copy()
    after = ContrastGamma(gamma=2.0).apply(image).detect_mat[:]
    assert after.mean() < before.mean()


def test_matches_skimage_on_detect_mat():
    image = load_synth_yeast_plate()
    src = image.detect_mat[:].copy()
    expected = np.clip(adjust_gamma(src, gamma=2.0, gain=1.5), 0.0, 1.0)
    actual = ContrastGamma(gamma=2.0, gain=1.5).apply(image).detect_mat[:]
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_gain_is_meaningful_under_clip():
    """The whole reason `norm` defaults to clip rather than rescale."""
    image = load_synth_yeast_plate()
    a = ContrastGamma(gamma=2.0, gain=1.0, norm="clip").apply(image).detect_mat[:].copy()
    b = ContrastGamma(gamma=2.0, gain=1.9, norm="clip").apply(image).detect_mat[:]
    assert np.abs(a - b).max() > 1e-2


def test_gain_is_absorbed_under_rescale():
    """Documented consequence of norm='rescale': gain is a uniform post-curve scale.

    ``adjust_gamma`` computes ``(I ** gamma) * gain``. A full-range rescale maps
    ``min -> 0`` and ``max -> 1``, dividing the uniform factor back out exactly.
    The residual is float32 rounding, not a real difference.
    """
    image = load_synth_yeast_plate()
    a = (
        ContrastGamma(gamma=2.0, gain=1.0, norm="rescale")
        .apply(image)
        .detect_mat[:]
        .copy()
    )
    b = ContrastGamma(gamma=2.0, gain=1.9, norm="rescale").apply(image).detect_mat[:]
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_rgb_path_differs_from_detect_mat_path():
    """Non-linear curve then projection != projection then curve. The whole point.

    Uses a channel-*mixing* ``detect_mode``. Under a selection mode the two
    routes provably coincide -- see
    :func:`test_rgb_and_detect_mat_agree_under_selection_modes`.
    """
    dm_image = load_synth_yeast_plate()
    dm_image.set_detect_mode(MIXING_MODE)
    via_dm = (
        ContrastGamma(gamma=2.5, input_layer="detect_mat")
        .apply(dm_image)
        .detect_mat[:]
        .copy()
    )

    rgb_image = load_synth_yeast_plate()
    rgb_image.set_detect_mode(MIXING_MODE)
    via_rgb = (
        ContrastGamma(gamma=2.5, input_layer="rgb").apply(rgb_image).detect_mat[:]
    )
    assert np.abs(via_dm - via_rgb).max() > 1e-3


@pytest.mark.parametrize("mode", SELECTION_MODES)
def test_rgb_and_detect_mat_agree_under_selection_modes(mode):
    """A selection ``detect_mode`` commutes with any increasing pointwise curve.

    ``MinRGB``/``HsvV`` are per-pixel order statistics and ``red``/``green``/
    ``blue`` pick one channel. For a monotonically increasing ``f``,
    ``min(f(r), f(g), f(b)) == f(min(r, g, b))``, so ``input_layer="rgb"`` and
    ``input_layer="detect_mat"`` are the *same computation*.

    Pinned so a future reader does not mistake the equality for a wiring bug and
    "fix" it. The docstrings promise a difference only for mixing modes.
    """
    dm_image = load_synth_yeast_plate()
    dm_image.set_detect_mode(mode)
    via_dm = (
        ContrastGamma(gamma=2.5, input_layer="detect_mat")
        .apply(dm_image)
        .detect_mat[:]
        .copy()
    )

    rgb_image = load_synth_yeast_plate()
    rgb_image.set_detect_mode(mode)
    via_rgb = (
        ContrastGamma(gamma=2.5, input_layer="rgb").apply(rgb_image).detect_mat[:]
    )
    np.testing.assert_allclose(via_dm, via_rgb, atol=1e-6)


def test_rgb_path_writes_only_detect_mat():
    image = load_synth_yeast_plate()
    rgb_before = image.rgb[:].copy()
    gray_before = image.gray[:].copy()
    out = ContrastGamma(gamma=2.0, input_layer="rgb").apply(image)
    np.testing.assert_array_equal(out.rgb[:], rgb_before)
    np.testing.assert_array_equal(out.gray[:], gray_before)


def test_negative_input_is_rescaled_not_raised():
    """FocusEdgeLaplace can emit negatives; skimage would raise ValueError."""
    from phenotypic.enhance import FocusEdgeLaplace

    image = FocusEdgeLaplace(norm=None).apply(load_synth_yeast_plate())
    assert image.detect_mat[:].min() < 0
    out = ContrastGamma(gamma=2.0).apply(image)
    assert 0.0 <= out.detect_mat[:].min() and out.detect_mat[:].max() <= 1.0


def test_json_round_trip():
    from phenotypic.abc_ import ImageOperation

    op = ContrastGamma(gamma=2.0, gain=1.5, norm=None, input_layer="rgb")
    loaded = ImageOperation.from_json(op.to_json())
    assert loaded.gamma == 2.0 and loaded.norm is None and loaded.input_layer == "rgb"


def test_rgb_on_grayscale_image_raises():
    """A grayscale-only image has no ``rgb`` layer to read.

    ``ImageOperation.apply`` funnels every failure through ``RuntimeError``, so the
    root cause is asserted via ``__cause__`` rather than the surface type. The
    unwrapped ``_read_input_layer`` call pins the mixin's own contract, so this
    test stays honest if ``apply``'s wrapper ever changes.
    """
    from phenotypic import Image
    from phenotypic.sdk_.exceptions_ import NoArrayError

    gray_only = Image(np.full((32, 32), 0.5, dtype=np.float32))
    op = ContrastGamma(input_layer="rgb")

    with pytest.raises(NoArrayError):
        op._read_input_layer(gray_only)

    with pytest.raises(RuntimeError) as excinfo:
        op.apply(gray_only)
    assert isinstance(excinfo.value.__cause__.__cause__, NoArrayError)


# --------------------------------------------------------------------------- #
# ContrastLog
# --------------------------------------------------------------------------- #
def test_contrast_log_field_order():
    from phenotypic.enhance import ContrastLog

    assert list(ContrastLog.model_fields) == ["gain", "inv", "norm", "input_layer"]


def test_contrast_log_brightens_dark_regions():
    """log compresses highlights and expands shadows -> mean rises."""
    from phenotypic.enhance import ContrastLog

    image = load_synth_yeast_plate()
    before = image.detect_mat[:].copy()
    after = ContrastLog().apply(image).detect_mat[:]
    assert after.mean() > before.mean()


def test_contrast_log_inv_is_the_inverse_curve():
    from phenotypic.enhance import ContrastLog

    image = load_synth_yeast_plate()
    fwd = ContrastLog(inv=False).apply(load_synth_yeast_plate()).detect_mat[:]
    inv = ContrastLog(inv=True).apply(image).detect_mat[:]
    assert np.abs(fwd - inv).max() > 1e-2


def test_contrast_log_matches_skimage():
    from skimage.exposure import adjust_log

    from phenotypic.enhance import ContrastLog

    image = load_synth_yeast_plate()
    src = image.detect_mat[:].copy()
    expected = np.clip(adjust_log(src, gain=1.0, inv=False), 0.0, 1.0)
    actual = ContrastLog().apply(image).detect_mat[:]
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_contrast_log_gain_is_meaningful_under_clip():
    from phenotypic.enhance import ContrastLog

    image = load_synth_yeast_plate()
    a = ContrastLog(gain=1.0, norm="clip").apply(image).detect_mat[:].copy()
    b = ContrastLog(gain=1.9, norm="clip").apply(image).detect_mat[:]
    assert np.abs(a - b).max() > 1e-2


def test_contrast_log_gain_is_absorbed_under_rescale():
    """``adjust_log`` scales by ``gain`` after the curve, so rescale removes it."""
    from phenotypic.enhance import ContrastLog

    image = load_synth_yeast_plate()
    a = ContrastLog(gain=1.0, norm="rescale").apply(image).detect_mat[:].copy()
    b = ContrastLog(gain=1.9, norm="rescale").apply(image).detect_mat[:]
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_contrast_log_rgb_path_differs_from_detect_mat_path():
    from phenotypic.enhance import ContrastLog

    dm_image = load_synth_yeast_plate()
    dm_image.set_detect_mode(MIXING_MODE)
    via_dm = (
        ContrastLog(input_layer="detect_mat").apply(dm_image).detect_mat[:].copy()
    )

    rgb_image = load_synth_yeast_plate()
    rgb_image.set_detect_mode(MIXING_MODE)
    via_rgb = ContrastLog(input_layer="rgb").apply(rgb_image).detect_mat[:]
    assert np.abs(via_dm - via_rgb).max() > 1e-3


# --------------------------------------------------------------------------- #
# ContrastSigmoid
# --------------------------------------------------------------------------- #
def test_contrast_sigmoid_field_order():
    from phenotypic.enhance import ContrastSigmoid

    assert list(ContrastSigmoid.model_fields) == [
        "cutoff",
        "gain",
        "inv",
        "norm",
        "input_layer",
    ]


def test_contrast_sigmoid_defaults_match_skimage():
    from phenotypic.enhance import ContrastSigmoid

    assert ContrastSigmoid().cutoff == 0.5
    assert ContrastSigmoid().gain == 10.0


def test_contrast_sigmoid_gain_survives_rescale():
    """Contrast with ContrastGamma: sigmoid's gain is inside exp(), so it reshapes."""
    from phenotypic.enhance import ContrastSigmoid

    a = ContrastSigmoid(gain=5.0, norm="rescale").apply(load_synth_yeast_plate())
    b = ContrastSigmoid(gain=15.0, norm="rescale").apply(load_synth_yeast_plate())
    assert np.abs(a.detect_mat[:] - b.detect_mat[:]).max() > 1e-2


def test_contrast_sigmoid_increases_contrast_about_cutoff():
    from phenotypic.enhance import ContrastSigmoid

    image = load_synth_yeast_plate()
    before = image.detect_mat[:].copy()
    after = (
        ContrastSigmoid(cutoff=float(before.mean()), gain=10.0)
        .apply(image)
        .detect_mat[:]
    )
    assert after.std() > before.std()


def test_contrast_sigmoid_matches_skimage():
    from skimage.exposure import adjust_sigmoid

    from phenotypic.enhance import ContrastSigmoid

    image = load_synth_yeast_plate()
    src = image.detect_mat[:].copy()
    expected = np.clip(adjust_sigmoid(src, cutoff=0.4, gain=8.0, inv=False), 0.0, 1.0)
    actual = ContrastSigmoid(cutoff=0.4, gain=8.0).apply(image).detect_mat[:]
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_contrast_sigmoid_rgb_path_differs_from_detect_mat_path():
    from phenotypic.enhance import ContrastSigmoid

    dm_image = load_synth_yeast_plate()
    dm_image.set_detect_mode(MIXING_MODE)
    via_dm = (
        ContrastSigmoid(cutoff=0.4, gain=8.0, input_layer="detect_mat")
        .apply(dm_image)
        .detect_mat[:]
        .copy()
    )

    rgb_image = load_synth_yeast_plate()
    rgb_image.set_detect_mode(MIXING_MODE)
    via_rgb = (
        ContrastSigmoid(cutoff=0.4, gain=8.0, input_layer="rgb")
        .apply(rgb_image)
        .detect_mat[:]
    )
    assert np.abs(via_dm - via_rgb).max() > 1e-3


def test_contrast_sigmoid_inv_breaks_selection_mode_commutation():
    """``inv=True`` makes the sigmoid *decreasing*, so it stops commuting with min.

    ``min(f(r), f(g), f(b)) == f(max(r, g, b))`` for a decreasing ``f``, which is
    not ``f(min(r, g, b))``. The mirror image of
    :func:`test_rgb_and_detect_mat_agree_under_selection_modes`, and the reason
    that test's equality is a property of the *curve*, not of the plumbing.
    """
    from phenotypic.enhance import ContrastSigmoid

    dm_image = load_synth_yeast_plate()
    dm_image.set_detect_mode("MinRGB")
    via_dm = (
        ContrastSigmoid(inv=True, input_layer="detect_mat")
        .apply(dm_image)
        .detect_mat[:]
        .copy()
    )

    rgb_image = load_synth_yeast_plate()
    rgb_image.set_detect_mode("MinRGB")
    via_rgb = (
        ContrastSigmoid(inv=True, input_layer="rgb").apply(rgb_image).detect_mat[:]
    )
    assert np.abs(via_dm - via_rgb).max() > 1e-3


# --------------------------------------------------------------------------- #
# ContrastStretching retrofit
# --------------------------------------------------------------------------- #
def test_stretching_field_order():
    from phenotypic.enhance import ContrastStretching

    assert list(ContrastStretching.model_fields) == [
        "lower_percentile",
        "upper_percentile",
        "keep_colors",
        "input_layer",
    ]


def test_stretching_has_no_norm_field():
    """Percentile rescaling IS the algorithm; a norm field could only no-op or undo it."""
    from phenotypic.enhance import ContrastStretching

    assert "norm" not in ContrastStretching.model_fields


def _red_dominant_plate():
    """A 16x16 red-dominant RGB plate with a horizontal ramp in the red channel."""
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[..., 0] = np.linspace(120, 220, 16, dtype=np.uint8)[None, :]
    rgb[..., 1] = 40
    rgb[..., 2] = 20
    return rgb


def _stretched(rgb_u8, *, keep_colors):
    """Run ContrastStretching on a MinRGB image and return its ``detect_mat``."""
    from phenotypic import Image
    from phenotypic.enhance import ContrastStretching

    image = Image(rgb_u8)
    image.set_detect_mode("MinRGB")
    op = ContrastStretching(input_layer="rgb", keep_colors=keep_colors)
    return op.apply(image).detect_mat[:].copy()


def test_stretching_keep_colors_matches_joint_percentile_oracle():
    """``keep_colors=True`` takes ONE (p_lo, p_hi) over the flattened H*W*3 array.

    Asserted against an independently computed oracle, not merely "differs from the
    other branch" -- an inequality assertion is symmetric under a branch swap and
    so cannot detect one.
    """
    from skimage.exposure import rescale_intensity

    rgb_u8 = _red_dominant_plate()
    src = rgb_u8.astype(np.float32) / np.float32(255)

    p_lo, p_hi = np.percentile(src, (2, 98))
    expected = rescale_intensity(image=src, in_range=(p_lo, p_hi), out_range=(0, 1))
    expected = np.min(expected, axis=2)

    np.testing.assert_allclose(
        _stretched(rgb_u8, keep_colors=True), expected, atol=1e-6
    )


def test_stretching_keep_colors_false_matches_per_channel_oracle():
    """``keep_colors=False`` takes an independent (p_lo, p_hi) per channel."""
    from skimage.exposure import rescale_intensity

    rgb_u8 = _red_dominant_plate()
    src = rgb_u8.astype(np.float32) / np.float32(255)

    expected = np.empty_like(src)
    for channel in range(3):
        p_lo, p_hi = np.percentile(src[..., channel], (2, 98))
        expected[..., channel] = rescale_intensity(
            image=src[..., channel], in_range=(p_lo, p_hi), out_range=(0, 1)
        )
    expected = np.min(expected, axis=2)

    np.testing.assert_allclose(
        _stretched(rgb_u8, keep_colors=False), expected, atol=1e-6
    )


def test_stretching_keep_colors_preserves_channel_balance():
    """Joint percentiles keep a red-dominant plate red-dominant; per-channel does not.

    Under ``keep_colors=True`` the dim green/blue channels are rescaled by the
    *red* channel's percentiles, so they stay dim and the ``MinRGB`` projection
    stays near 0. Under ``keep_colors=False`` each channel is independently
    stretched to fill [0, 1] -- a white balance -- so the constant green and blue
    channels collapse and the hue is destroyed.
    """
    rgb_u8 = _red_dominant_plate()
    joint = _stretched(rgb_u8, keep_colors=True)
    split = _stretched(rgb_u8, keep_colors=False)

    assert not np.allclose(joint, split, atol=1e-4)
    # Joint: green (40/255) and blue (20/255) sit near the bottom of the red
    # channel's range, so min-projection stays dark across the whole plate.
    assert float(joint.max()) < 0.2, f"joint max {joint.max():.4f} lost channel balance"
    # Per-channel: green and blue are constant, so each stretches to a single
    # value; the red ramp no longer dominates the min.
    assert float(split.max()) > float(joint.max())


def test_stretching_keep_colors_ignored_for_2d_input():
    from phenotypic.enhance import ContrastStretching

    a = ContrastStretching(keep_colors=True).apply(load_synth_yeast_plate()).detect_mat[:]
    b = ContrastStretching(keep_colors=False).apply(load_synth_yeast_plate()).detect_mat[:]
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_stretching_output_always_unit_range():
    from phenotypic.enhance import ContrastStretching

    dm = ContrastStretching().apply(load_synth_yeast_plate()).detect_mat[:]
    assert abs(float(dm.min())) < 1e-6 and abs(float(dm.max()) - 1.0) < 1e-6


def test_stretching_rgb_path_differs_from_detect_mat_path():
    """Percentiles are a statistic *over the input*, so this op never commutes."""
    from phenotypic.enhance import ContrastStretching

    dm_image = load_synth_yeast_plate()
    dm_image.set_detect_mode("MinRGB")
    via_dm = (
        ContrastStretching(input_layer="detect_mat")
        .apply(dm_image)
        .detect_mat[:]
        .copy()
    )

    rgb_image = load_synth_yeast_plate()
    rgb_image.set_detect_mode("MinRGB")
    via_rgb = ContrastStretching(input_layer="rgb").apply(rgb_image).detect_mat[:]
    assert np.abs(via_dm - via_rgb).max() > 1e-3


def test_stretching_defaults_unchanged_by_retrofit():
    """Three call sites construct it with defaults; behaviour must be identical."""
    from skimage.exposure import rescale_intensity

    from phenotypic.enhance import ContrastStretching

    image = load_synth_yeast_plate()
    src = image.detect_mat[:].copy()
    p_lower, p_upper = np.percentile(src, (2, 98))
    expected = rescale_intensity(image=src, in_range=(p_lower, p_upper), out_range=(0, 1))
    actual = ContrastStretching().apply(image).detect_mat[:]
    np.testing.assert_allclose(actual, expected, atol=1e-6)
