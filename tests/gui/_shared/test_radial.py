"""Unit tests for :mod:`phenotypic.gui._shared._radial`.

These are pure component-tree tests — no browser, no Dash server.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

from dash.development.base_component import Component

from phenotypic.gui._design import category_color
from phenotypic.gui._shared._radial import (
    RADIAL_RESTORE_SENTINEL,
    _CORE_TOKENS,
    _DEFAULT_RADIUS,
    _WEDGE_SIZE,
    _wedge_positions,
    build_radial_body,
    build_radial_trigger,
    radial_popover_body_id,
    radial_store_id,
    radial_trigger_id,
    radial_wedge_id,
)
from phenotypic.schema import ErrorCategory


# ---------------------------------------------------------------------------
# Component-tree walking helpers (same pattern as test_tiles.py)
# ---------------------------------------------------------------------------


def _walk(component: object) -> Iterator[object]:
    """Yield ``component`` and every descendant depth-first."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk(child)
    else:
        yield from _walk(children)


def _buttons(root: object) -> list[object]:
    """Return all Button descendants."""
    import dash_bootstrap_components as dbc

    return [n for n in _walk(root) if isinstance(n, dbc.Button)]


# ---------------------------------------------------------------------------
# _wedge_positions
# ---------------------------------------------------------------------------


class TestWedgePositions:
    def test_returns_n_positions(self) -> None:
        positions = _wedge_positions(6, _DEFAULT_RADIUS)
        assert len(positions) == 6

    def test_positions_are_distinct(self) -> None:
        positions = _wedge_positions(6, _DEFAULT_RADIUS)
        assert len(set(positions)) == 6

    def test_all_positions_on_circle(self) -> None:
        """Each wedge centre should lie on the circle of the given radius."""
        radius = _DEFAULT_RADIUS
        positions = _wedge_positions(6, radius)
        button_half = _WEDGE_SIZE / 2
        center = radius + button_half
        for left, top in positions:
            cx = left + button_half
            cy = top + button_half
            dist = math.hypot(cx - center, cy - center)
            assert abs(dist - radius) < 0.5, f"dist={dist:.2f} vs radius={radius}"

    def test_top_wedge_is_near_top_center(self) -> None:
        """First wedge (index 0) should be at the top of the ring."""
        radius = _DEFAULT_RADIUS
        positions = _wedge_positions(4, radius)
        left, top = positions[0]
        button_half = _WEDGE_SIZE / 2
        center = radius + button_half
        # The top-centre button's left should be near the ring centre.
        assert abs(left - (center - button_half)) < 1.0
        # Its top should be near 0 (radius - radius = 0 offset on Y axis).
        assert abs(top) < 2.0

    def test_single_position(self) -> None:
        positions = _wedge_positions(1, _DEFAULT_RADIUS)
        assert len(positions) == 1

    def test_custom_radius(self) -> None:
        positions = _wedge_positions(8, radius=80)
        assert len(positions) == 8


# ---------------------------------------------------------------------------
# radial_wedge_id and surface collision
# ---------------------------------------------------------------------------


class TestIdFactories:
    def test_wedge_id_shape(self) -> None:
        id_ = radial_wedge_id("colony", "plateA.tif", 2, "debris")
        assert id_["type"] == "colony-cat-wedge"
        assert id_["image_file"] == "plateA.tif"
        assert id_["label"] == 2
        assert id_["category"] == "debris"

    def test_surface_differentiates_type(self) -> None:
        colony_id = radial_wedge_id("colony", "img.tif", 1, "debris")
        qc_id = radial_wedge_id("qc", "img.tif", 1, "debris")
        assert colony_id["type"] == "colony-cat-wedge"
        assert qc_id["type"] == "qc-cat-wedge"
        assert colony_id["type"] != qc_id["type"]

    def test_trigger_id_shape(self) -> None:
        id_ = radial_trigger_id("colony", "x.tif", 5)
        assert id_["type"] == "colony-radial-trigger"

    def test_popover_body_id_shape(self) -> None:
        id_ = radial_popover_body_id("qc", "x.tif", 5)
        assert id_["type"] == "qc-radial-popover-body"

    def test_store_id_shape(self) -> None:
        id_ = radial_store_id("colony", "x.tif", 5)
        assert id_["type"] == "colony-radial-store"

    def test_all_ids_are_distinct_types(self) -> None:
        """Trigger, popover-body, store, and wedge ids must not share a type."""
        trigger = radial_trigger_id("colony", "f.tif", 1)["type"]
        body = radial_popover_body_id("colony", "f.tif", 1)["type"]
        store = radial_store_id("colony", "f.tif", 1)["type"]
        wedge = radial_wedge_id("colony", "f.tif", 1, "debris")["type"]
        assert len({trigger, body, store, wedge}) == 4


# ---------------------------------------------------------------------------
# RADIAL_RESTORE_SENTINEL
# ---------------------------------------------------------------------------


def test_restore_sentinel_value() -> None:
    assert RADIAL_RESTORE_SENTINEL == "__restore__"


def test_restore_sentinel_differs_from_all_core_tokens() -> None:
    for token in ErrorCategory.labels():
        assert token != RADIAL_RESTORE_SENTINEL


# ---------------------------------------------------------------------------
# build_radial_body
# ---------------------------------------------------------------------------


class TestBuildRadialBody:
    """Tests for :func:`build_radial_body` component structure."""

    _CUSTOM: list[str] = []

    def _body(
        self,
        surface: str = "colony",
        image_file: str = "plateA.tif",
        label: int = 2,
        custom: list[str] | None = None,
        current_category: str | None = None,
    ) -> Component:
        return build_radial_body(
            surface,
            image_file,
            label,
            custom if custom is not None else self._CUSTOM,
            current_category,
        )

    def test_returns_body_wrap_with_ring_child(self) -> None:
        body = self._body()
        assert body.className == "radial-body-wrap"
        # The ring container is the first child of the body wrap.
        ring = body.children[0]
        assert ring.className == "radial-ring-container"

    def test_one_wedge_per_core_category_excluding_other(self) -> None:
        """Each non-other core token should have exactly one wedge button."""
        body = self._body()
        buttons = _buttons(body)
        wedge_categories = [
            btn.id.get("category")
            for btn in buttons
            if isinstance(btn.id, dict) and btn.id.get("type") == "colony-cat-wedge"
        ]
        for token in _CORE_TOKENS:
            assert token in wedge_categories, f"missing wedge for core token {token!r}"

    def test_other_wedge_present(self) -> None:
        body = self._body()
        buttons = _buttons(body)
        categories = [
            btn.id.get("category")
            for btn in buttons
            if isinstance(btn.id, dict) and btn.id.get("type") == "colony-cat-wedge"
        ]
        assert "other" in categories

    def test_custom_folder_wedge_present(self) -> None:
        body = self._body()
        buttons = _buttons(body)
        categories = [
            btn.id.get("category")
            for btn in buttons
            if isinstance(btn.id, dict) and btn.id.get("type") == "colony-cat-wedge"
        ]
        assert "__custom_folder__" in categories

    def test_center_restore_node_present(self) -> None:
        """The center restore/close node must carry RADIAL_RESTORE_SENTINEL."""
        body = self._body()
        buttons = _buttons(body)
        restore_buttons = [
            btn
            for btn in buttons
            if isinstance(btn.id, dict)
            and btn.id.get("category") == RADIAL_RESTORE_SENTINEL
        ]
        assert len(restore_buttons) == 1

    def test_wedge_ids_correct_surface_colony(self) -> None:
        body = self._body(surface="colony")
        buttons = _buttons(body)
        wedge_types = {
            btn.id["type"]
            for btn in buttons
            if isinstance(btn.id, dict) and "type" in btn.id
        }
        assert "colony-cat-wedge" in wedge_types
        assert "qc-cat-wedge" not in wedge_types

    def test_wedge_ids_correct_surface_qc(self) -> None:
        body = self._body(surface="qc")
        buttons = _buttons(body)
        wedge_types = {
            btn.id["type"]
            for btn in buttons
            if isinstance(btn.id, dict) and "type" in btn.id
        }
        assert "qc-cat-wedge" in wedge_types
        assert "colony-cat-wedge" not in wedge_types

    def test_surface_distinguishes_ids(self) -> None:
        """Colony and QC surfaces must produce non-overlapping wedge types."""
        colony_body = self._body(surface="colony")
        qc_body = self._body(surface="qc")
        colony_types = {
            btn.id["type"]
            for btn in _buttons(colony_body)
            if isinstance(btn.id, dict) and "type" in btn.id
        }
        qc_types = {
            btn.id["type"]
            for btn in _buttons(qc_body)
            if isinstance(btn.id, dict) and "type" in btn.id
        }
        assert colony_types.isdisjoint(qc_types)

    def test_wedge_image_file_and_label_correct(self) -> None:
        body = self._body(image_file="pA.tif", label=7)
        buttons = _buttons(body)
        for btn in buttons:
            if not isinstance(btn.id, dict) or "image_file" not in btn.id:
                continue
            assert btn.id["image_file"] == "pA.tif"
            assert btn.id["label"] == 7

    def test_debris_wedge_color_matches_category_color(self) -> None:
        body = self._body()
        buttons = _buttons(body)
        debris_btn = next(
            (
                btn
                for btn in buttons
                if isinstance(btn.id, dict) and btn.id.get("category") == "debris"
            ),
            None,
        )
        assert debris_btn is not None
        expected_color = category_color("debris")
        assert debris_btn.style.get("backgroundColor") == expected_color

    def test_at_most_7_primary_wedges(self) -> None:
        """Primary ring must not exceed 7 wedges (excl. the center restore node)."""
        body = self._body()
        buttons = _buttons(body)
        # The center restore node also uses the cat-wedge type (so callback routing
        # works), but its category is RADIAL_RESTORE_SENTINEL — exclude it from the
        # primary count.
        primary = [
            btn
            for btn in buttons
            if isinstance(btn.id, dict)
            and btn.id.get("type") == "colony-cat-wedge"
            and btn.id.get("category") != RADIAL_RESTORE_SENTINEL
        ]
        assert len(primary) <= 7

    def test_no_current_category_no_active_modifier(self) -> None:
        body = self._body(current_category=None)
        buttons = _buttons(body)
        active = [
            btn for btn in buttons if "radial-wedge--active" in (btn.className or "")
        ]
        assert len(active) == 0

    def test_active_modifier_on_current_category_wedge(self) -> None:
        body = self._body(current_category="debris")
        buttons = _buttons(body)
        active = [
            btn for btn in buttons if "radial-wedge--active" in (btn.className or "")
        ]
        assert len(active) == 1
        assert active[0].id.get("category") == "debris"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# build_radial_trigger
# ---------------------------------------------------------------------------


class TestBuildRadialTrigger:
    """Tests for :func:`build_radial_trigger`."""

    def _trigger(
        self,
        surface: str = "colony",
        image_file: str = "plateA.tif",
        label: int = 3,
        current_category: str | None = None,
        is_custom: bool = False,
    ) -> list[Component]:
        return build_radial_trigger(
            surface, image_file, label, current_category, is_custom
        )

    def test_returns_three_components(self) -> None:
        components = self._trigger()
        assert len(components) == 3

    def test_neutral_trigger_has_neutral_class(self) -> None:
        """No current category → neutral ▾ button (no badge class set)."""
        import dash_bootstrap_components as dbc

        components = self._trigger(current_category=None)
        trigger_btn = components[0]
        assert isinstance(trigger_btn, dbc.Button)
        assert "radial-badge--neutral" in trigger_btn.className

    def test_neutral_trigger_no_bg_color(self) -> None:
        components = self._trigger(current_category=None)
        trigger_btn = components[0]
        # The style dict should be empty (no backgroundColor override).
        assert not trigger_btn.style.get("backgroundColor")

    def test_badge_styled_with_category_color_when_set(self) -> None:
        components = self._trigger(current_category="debris")
        import dash_bootstrap_components as dbc

        trigger_btn = components[0]
        assert isinstance(trigger_btn, dbc.Button)
        expected = category_color("debris")
        assert trigger_btn.style.get("backgroundColor") == expected

    def test_badge_class_present_when_category_set(self) -> None:
        components = self._trigger(current_category="debris")
        trigger_btn = components[0]
        assert "radial-badge" in trigger_btn.className

    def test_is_custom_adds_custom_modifier(self) -> None:
        components = self._trigger(current_category="halo", is_custom=True)
        trigger_btn = components[0]
        assert "radial-badge--custom" in trigger_btn.className

    def test_core_category_no_custom_modifier(self) -> None:
        components = self._trigger(current_category="debris", is_custom=False)
        trigger_btn = components[0]
        assert "radial-badge--custom" not in trigger_btn.className

    def test_popover_is_second_component(self) -> None:
        import dash_bootstrap_components as dbc

        components = self._trigger()
        assert isinstance(components[1], dbc.Popover)

    def test_store_is_third_component(self) -> None:
        from dash import dcc

        components = self._trigger()
        assert isinstance(components[2], dcc.Store)

    def test_store_data_contains_surface_image_label(self) -> None:
        from dash import dcc

        components = self._trigger(surface="qc", image_file="x.tif", label=9)
        store = components[2]
        assert isinstance(store, dcc.Store)
        assert store.data["surface"] == "qc"
        assert store.data["image_file"] == "x.tif"
        assert store.data["label"] == 9

    def test_trigger_id_matches_factory(self) -> None:
        components = self._trigger(surface="colony", image_file="img.tif", label=5)
        trigger_btn = components[0]
        expected_id = radial_trigger_id("colony", "img.tif", 5)
        assert trigger_btn.id == expected_id

    def test_popover_body_id_matches_factory(self) -> None:
        import dash_bootstrap_components as dbc

        components = self._trigger(surface="colony", image_file="img.tif", label=5)
        popover = components[1]
        assert isinstance(popover, dbc.Popover)
        # The popover body is popover.children (a PopoverBody).
        body = popover.children
        expected_id = radial_popover_body_id("colony", "img.tif", 5)
        assert body.id == expected_id

    def test_colony_and_qc_trigger_ids_differ(self) -> None:
        colony = self._trigger(surface="colony", image_file="f.tif", label=1)
        qc = self._trigger(surface="qc", image_file="f.tif", label=1)
        assert colony[0].id != qc[0].id
        assert colony[0].id["type"] == "colony-radial-trigger"
        assert qc[0].id["type"] == "qc-radial-trigger"

    def test_different_labels_different_trigger_ids(self) -> None:
        a = self._trigger(surface="colony", image_file="f.tif", label=1)
        b = self._trigger(surface="colony", image_file="f.tif", label=2)
        assert a[0].id != b[0].id

    def test_custom_category_color_applied(self) -> None:
        """A custom category token still gets a color via category_color."""
        components = self._trigger(current_category="halo", is_custom=True)
        trigger_btn = components[0]
        expected = category_color("halo", custom_index=0)
        assert trigger_btn.style.get("backgroundColor") == expected


# ---------------------------------------------------------------------------
# Custom folder section (Task 7): chips + ＋ Add affordance
# ---------------------------------------------------------------------------


def _walk_components(root):
    """Yield root + every descendant (re-export of the module helper)."""
    yield from _walk(root)


class TestBuildRadialBodyCustomSection:
    """Tests for the expanded Custom folder section of ``build_radial_body``."""

    def _body(self, custom: list[str], current_category: str | None = None):
        return build_radial_body(
            "colony",
            "plateA.tif",
            2,
            custom,
            current_category,
        )

    def test_custom_chip_rendered_per_registered_token(self) -> None:
        """Each registered custom token renders a clickable wedge chip."""
        body = self._body(["halo", "ghost"])
        chip_categories = [
            btn.id.get("category")
            for btn in _buttons(body)
            if isinstance(btn.id, dict)
            and btn.id.get("type") == "colony-cat-wedge"
            and btn.id.get("category") in {"halo", "ghost"}
        ]
        assert "halo" in chip_categories
        assert "ghost" in chip_categories

    def test_custom_chip_carries_custom_discriminator(self) -> None:
        """Custom chips get the ``radial-badge--custom`` discriminator (decision D)."""
        body = self._body(["halo"])
        chip = next(
            btn
            for btn in _buttons(body)
            if isinstance(btn.id, dict) and btn.id.get("category") == "halo"
        )
        assert "radial-badge--custom" in (chip.className or "")

    def test_custom_chip_cycles_custom_palette_color(self) -> None:
        """Custom chips are colored by their registration index via category_color."""
        body = self._body(["halo", "ghost"])
        chips = {
            btn.id.get("category"): btn
            for btn in _buttons(body)
            if isinstance(btn.id, dict) and btn.id.get("category") in {"halo", "ghost"}
        }
        assert chips["halo"].style.get("backgroundColor") == category_color(
            "halo", custom_index=0
        )
        assert chips["ghost"].style.get("backgroundColor") == category_color(
            "ghost", custom_index=1
        )

    def test_add_custom_input_and_submit_present(self) -> None:
        """The ＋ Add affordance ships an input + a submit button."""
        from dash import dcc

        body = self._body([])
        ids = {
            n.id.get("type")
            for n in _walk_components(body)
            if isinstance(getattr(n, "id", None), dict)
        }
        assert "colony-radial-custom-input" in ids
        assert "colony-radial-custom-submit" in ids
        assert "colony-radial-custom-msg" in ids
        # The input is a dcc.Input.
        inputs = [n for n in _walk_components(body) if isinstance(n, dcc.Input)]
        assert len(inputs) == 1

    def test_active_custom_chip_gets_active_modifier(self) -> None:
        """The chip matching ``current_category`` is marked active."""
        body = self._body(["halo"], current_category="halo")
        chip = next(
            btn
            for btn in _buttons(body)
            if isinstance(btn.id, dict) and btn.id.get("category") == "halo"
        )
        assert "radial-wedge--active" in (chip.className or "")

    def test_custom_section_ids_carry_surface(self) -> None:
        """QC surface custom-add ids are distinct from colony's."""
        qc_body = build_radial_body("qc", "p.tif", 1, ["halo"], None)
        ids = {
            n.id.get("type")
            for n in _walk_components(qc_body)
            if isinstance(getattr(n, "id", None), dict)
        }
        assert "qc-radial-custom-input" in ids
        assert "colony-radial-custom-input" not in ids
