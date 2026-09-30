/*
 * Drag-to-resize for both pydata-sphinx-theme sidebars.
 *
 * pydata-sphinx-theme (0.16) gives the section navigation (left) a fixed 25%
 * column and the page contents (right) a fixed 17rem; it has no resize option.
 * This script inserts a thin separator on each sidebar's inner edge. Dragging it,
 * or pressing the arrow keys while it is focused, sets the width; a double-click
 * restores the theme default. Widths persist per browser in localStorage.
 *
 * Widths are applied from <head>, before the body renders, so a stored width
 * never flashes the default first. Handles and overrides apply only above the
 * theme's own breakpoints (960px left, 1200px right); below those the theme turns
 * the sidebars into drawers and this script leaves them alone. See
 * sidebar-resize.css.
 */
(function () {
  "use strict";

  var STORAGE_KEY = "phenotypic-docs:sidebar-widths";
  var KEY_STEP = 16;
  var KEY_STEP_LARGE = 64;
  var SIDES = {
    primary: {
      selector: ".bd-sidebar-primary",
      attribute: "data-ptx-primary-width",
      property: "--ptx-sidebar-primary-width",
      min: 192,
      max: 512,
      direction: 1, // the handle is on the right edge: dragging right widens
      label: "Resize section navigation",
    },
    secondary: {
      selector: ".bd-sidebar-secondary",
      attribute: "data-ptx-secondary-width",
      property: "--ptx-sidebar-secondary-width",
      min: 176,
      max: 480,
      direction: -1, // the handle is on the left edge: dragging left widens
      label: "Resize page contents",
    },
  };
  var root = document.documentElement;

  function readStoredWidths() {
    try {
      var parsed = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "{}");
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch (error) {
      return {};
    }
  }

  function writeStoredWidths() {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(widths));
    } catch (error) {
      // Storage blocked (private window, site data disabled): widths last for this page only.
    }
  }

  function clampWidth(side, width) {
    return Math.max(side.min, Math.min(side.max, Math.round(width)));
  }

  function applyWidth(name) {
    var side = SIDES[name];
    var width = widths[name];
    if (typeof width === "number" && isFinite(width)) {
      widths[name] = clampWidth(side, width);
      root.style.setProperty(side.property, widths[name] + "px");
      root.setAttribute(side.attribute, "");
    } else {
      delete widths[name];
      root.style.removeProperty(side.property);
      root.removeAttribute(side.attribute);
    }
  }

  var widths = readStoredWidths();
  Object.keys(SIDES).forEach(applyWidth);

  function createHandle(name, sidebar) {
    var side = SIDES[name];
    var handle = document.createElement("div");
    var startX = 0;
    var startWidth = 0;

    handle.className = "ptx-resize-handle ptx-resize-handle--" + name;
    handle.setAttribute("role", "separator");
    handle.setAttribute("aria-orientation", "vertical");
    handle.setAttribute("aria-label", side.label);
    handle.setAttribute("aria-valuemin", String(side.min));
    handle.setAttribute("aria-valuemax", String(side.max));
    handle.setAttribute("title", "Drag to resize. Double-click to reset.");
    handle.tabIndex = 0;
    if (sidebar.id) {
      handle.setAttribute("aria-controls", sidebar.id);
    }

    function renderedWidth() {
      return sidebar.getBoundingClientRect().width;
    }

    function syncValue() {
      handle.setAttribute("aria-valuenow", String(Math.round(renderedWidth())));
    }

    function setWidth(width) {
      widths[name] = clampWidth(side, width);
      applyWidth(name);
      syncValue();
    }

    handle.addEventListener("pointerdown", function (event) {
      if (event.button !== 0) {
        return;
      }
      startX = event.clientX;
      startWidth = renderedWidth();
      handle.setPointerCapture(event.pointerId);
      root.classList.add("ptx-resizing");
      event.preventDefault();
    });
    handle.addEventListener("pointermove", function (event) {
      if (handle.hasPointerCapture(event.pointerId)) {
        setWidth(startWidth + side.direction * (event.clientX - startX));
      }
    });
    function endDrag(event) {
      if (handle.hasPointerCapture(event.pointerId)) {
        handle.releasePointerCapture(event.pointerId);
        writeStoredWidths();
      }
      root.classList.remove("ptx-resizing");
    }
    handle.addEventListener("pointerup", endDrag);
    handle.addEventListener("pointercancel", endDrag);

    handle.addEventListener("dblclick", function () {
      delete widths[name];
      applyWidth(name);
      writeStoredWidths();
      syncValue();
    });

    handle.addEventListener("keydown", function (event) {
      var step = event.shiftKey ? KEY_STEP_LARGE : KEY_STEP;
      var delta;
      if (event.key === "ArrowRight") {
        delta = step;
      } else if (event.key === "ArrowLeft") {
        delta = -step;
      } else {
        return;
      }
      event.preventDefault();
      setWidth(renderedWidth() + side.direction * delta);
      writeStoredWidths();
    });

    syncValue();
    return handle;
  }

  function installHandles() {
    var primary = document.querySelector(SIDES.primary.selector);
    if (primary && primary.parentNode) {
      primary.parentNode.insertBefore(createHandle("primary", primary), primary.nextSibling);
    }
    var secondary = document.querySelector(SIDES.secondary.selector);
    if (secondary && secondary.parentNode) {
      secondary.parentNode.insertBefore(createHandle("secondary", secondary), secondary);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", installHandles);
  } else {
    installHandles();
  }
})();
