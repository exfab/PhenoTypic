"""``/sandbox/api/{root,children,classify}`` JSON blueprint.

The sidebar fetches its tree data from these routes, NOT from Dash callbacks.
The reasons (per ``GUI_SPEC_V1.md`` section 3):

    * Async ``fetch()`` from sidebar JS does not need to round-trip through
      Dash's serialization layer.
    * Independently testable with Flask's test client (no headless browser).
    * Cloud-deploy auth (deferred) attaches one ``@before_request`` hook here
      rather than per-callback.

Registered directly on ``shell_app.server`` (NOT under
``DispatcherMiddleware``) so it answers regardless of which Dash sub-app is
currently mounted.

Routes:
    ``GET /sandbox/api/root``
        Returns ``{"root": "<absolute>", "name": "<basename>", "badges": ...}``
        for the sandbox root.

    ``GET /sandbox/api/children?path=<rel>&hidden=0&symlinks=0``
        Returns ``{"children": [{"name", "type", "rel_path", "badges"}, ...],
        "truncated": <bool>}`` for direct children of ``path`` (default:
        root). Toggles map to the sidebar's "Hidden files" / "External
        symlinks" checkboxes. ``type`` is one of ``"dir"``, ``"file"``,
        ``"external_symlink"``. External symlinks are listed with empty
        ``badges`` — the route never reads content from outside the
        sandbox even when the user enables the toggle.

    ``GET /sandbox/api/classify?path=<rel>``
        Returns the :class:`Capabilities` JSON for one path.

Status codes
    ``400`` — ``path=`` query parameter escapes the sandbox. The JSON-API
    contract documents ``path=`` as a strict input, so a malformed input
    deserves a ``400``. (The ``/runs/`` blueprint, by contrast, returns
    ``404`` for the same condition because it does not want to disclose
    whether a target exists outside the sandbox.)

    ``403`` — sandbox child listing raised ``PermissionError``.

    ``404`` — sandbox child listing target does not exist or is not a
    directory.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from flask import Blueprint, abort, jsonify, request

from phenotypic.gui._config import (
    MOUNT_HOME,
    SANDBOX_API_PREFIX,
    SANDBOX_API_VIEWER_OUTPUT_ROOT,
    join_url_prefix,
)
from phenotypic.gui.shell._classifier import Capabilities, classify
from phenotypic.gui.shell._binding import BindingSupersededError

if TYPE_CHECKING:
    from flask import Flask

    from phenotypic.gui.shell._sandbox import SandboxRoot
    from phenotypic.gui.shell._binding_jobs import ResultsBindJobManager
    from phenotypic.gui.shell._session import ToolSession

logger = logging.getLogger(__name__)

__all__ = ["build_sandbox_api", "register_sandbox_api"]

# ``request.args.get("hidden")`` returns a string; we accept "1"/"true"/"yes"
# (case-insensitive) as truthy, anything else as falsy. Keeps the UI side
# simple — JS sends ``"1"`` or ``"0"``.
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Hard cap on the number of classified children per /sandbox/api/children
# request. Beyond this count the response carries ``"truncated": true`` and
# remaining entries are listed without ``classify`` badges so the sidebar
# still renders. Prevents pathological directories (10k+ entries) from
# wedging the sidebar request thread by thrashing the classifier's LRU.
_CHILDREN_CLASSIFY_CAP = 500

# All-False ``Capabilities`` used when we deliberately skip classification —
# e.g. for external symlinks (whose targets sit outside the sandbox), and for
# the tail of a truncated children listing.
_PLACEHOLDER_CAPS = Capabilities(
    is_image_dir=False,
    has_pipeline_json=False,
    is_cli_output=False,
    is_deliverables_bundle=False,
    has_dashboard=False,
    is_process_only_output=False,
    is_tune_output=False,
    image_count=None,
    bad_perms=False,
)


def _is_external_symlink(child: Path, sandbox: "SandboxRoot") -> bool:
    """Return ``True`` iff ``child`` is a symlink whose target leaves root.

    The classifier follows symlinks (``Path.stat`` / ``iterdir``) so calling
    ``classify(external_symlink)`` would read content from outside the
    sandbox — even when the user explicitly enabled "External symlinks" in
    the sidebar (the spec says external links render as a disabled node, not
    a fully-classified one). We detect external links here and return a
    placeholder capability instead.
    """
    if not child.is_symlink():
        return False
    try:
        return not sandbox.contains(child)
    except (OSError, RuntimeError):
        # Broken symlinks AND symlink cycles (``RuntimeError`` from
        # CPython's ``Path.resolve``). Treat both as external for safety.
        return True


def build_sandbox_api(
    sandbox: "SandboxRoot",
    *,
    viewer_session: "ToolSession[object] | None" = None,
    viewer_state: "dict[str, Any] | None" = None,
    extra_release_sessions: "tuple[ToolSession[object], ...] | None" = None,
    bind_output: "Callable[[Path | None], Any] | None" = None,
    binding_jobs: "ResultsBindJobManager | None" = None,
    browser_url_prefix: str = MOUNT_HOME,
    name: str = "phenotypic_sandbox_api",
    url_prefix: str = SANDBOX_API_PREFIX,
) -> Blueprint:
    """Build the ``/sandbox/api/*`` blueprint.

    Args:
        sandbox: Containment primitive. Every request resolves its ``path``
            argument through ``sandbox.resolve``; out-of-root requests get
            a 400 with a JSON error body.
        viewer_session: Optional :class:`ToolSession` whose ``touch()`` is
            called on every successful request (sidebar polling counts as
            user activity).
        extra_release_sessions: Additional sessions to ``release()``
            alongside ``viewer_session`` when a viewer hand-off succeeds.
            The analysis sub-app's session goes here so a single bind
            mutates ``viewer_state`` and rebuilds both tools in lock-step
            (per the locked "shared output_root" decision).
        bind_output: Optional production binder that constructs Results and
            Analysis candidates and atomically publishes both sessions. A
            ``None`` target means explicit refresh of the currently bound
            path. When omitted, the legacy release-and-lazy-rebuild adapter
            remains available for isolated route users.
        binding_jobs: Optional production asynchronous binding manager. When
            supplied, POST returns a polling job instead of holding the HTTP
            request through discovery and candidate construction.
        browser_url_prefix: Browser-visible reverse-proxy prefix prepended to
            returned polling and cancellation paths.
        name: Blueprint name. Defaults to ``"phenotypic_sandbox_api"``.
        url_prefix: Defaults to ``"/sandbox/api"``.

    Returns:
        Configured :class:`flask.Blueprint`.
    """
    bp = Blueprint(name, __name__, url_prefix=url_prefix)

    @bp.route("/root")
    def root_endpoint() -> Any:
        if viewer_session is not None:
            viewer_session.touch()
        caps = classify(sandbox.root)
        return jsonify(
            {
                "root": str(sandbox.root),
                "name": sandbox.root.name or str(sandbox.root),
                "badges": asdict(caps),
            }
        )

    @bp.route("/children")
    def children_endpoint() -> Any:
        rel = request.args.get("path", "")
        include_hidden = request.args.get("hidden", "0").lower() in _TRUTHY
        include_external = request.args.get(
            "symlinks", "0"
        ).lower() in _TRUTHY
        try:
            target = sandbox.resolve(rel) if rel else sandbox.root
        except ValueError:
            logger.warning(
                "rejected /sandbox/api/children traversal: %r", rel
            )
            return abort(400)

        try:
            # Materialise inside the try so the lazy ``iterdir`` failure
            # surfaces here, not later in ``sorted``. (``list_children`` is
            # a generator.)
            children = list(
                sandbox.list_children(
                    target,
                    include_hidden=include_hidden,
                    include_external_symlinks=include_external,
                )
            )
        except PermissionError:
            return abort(403)
        except FileNotFoundError:
            return abort(404)
        except NotADirectoryError:
            return abort(404)

        if viewer_session is not None:
            viewer_session.touch()

        sorted_children = sorted(
            children, key=lambda p: (not p.is_dir(), p.name.lower())
        )
        rows: list[dict[str, Any]] = []
        truncated = False
        for idx, child in enumerate(sorted_children):
            try:
                rel_path = str(child.relative_to(sandbox.root))
            except ValueError:
                # Should not happen — list_children only yields in-root
                # paths — but guard against future regressions.
                continue
            external = _is_external_symlink(child, sandbox)
            if idx >= _CHILDREN_CLASSIFY_CAP and not external:
                truncated = True
                row_type = "dir" if child.is_dir() else "file"
                badges = asdict(_PLACEHOLDER_CAPS)
            elif external:
                # Spec: external symlinks render as disabled nodes; never
                # classify the link target (which lives outside the sandbox).
                row_type = "external_symlink"
                badges = asdict(_PLACEHOLDER_CAPS)
            else:
                row_type = "dir" if child.is_dir() else "file"
                badges = asdict(classify(child))
            rows.append(
                {
                    "name": child.name,
                    "type": row_type,
                    "rel_path": rel_path,
                    "badges": badges,
                }
            )
        return jsonify({"children": rows, "truncated": truncated})

    @bp.route("/classify")
    def classify_endpoint() -> Any:
        rel = request.args.get("path", "")
        try:
            target = sandbox.resolve(rel) if rel else sandbox.root
        except ValueError:
            return abort(400)

        if viewer_session is not None:
            viewer_session.touch()
        return jsonify(asdict(classify(target)))

    @bp.route("/viewer/output-root", methods=["POST"])
    def viewer_output_root_endpoint() -> Any:
        """Bind or explicitly refresh the shared Results/Analysis snapshot.

        A ``{"path": "<sandbox-relative>"}`` payload selects an output.
        ``{"refresh": true}`` rediscovers the current selection. The hub's
        production binder constructs both candidate apps and publishes them
        together only after both pass their post-read fingerprint checks.
        The fallback adapter retained for isolated route users performs the
        former release-and-lazy-rebuild hand-off.

        Production returns 202 immediately with ``job_id``, ``poll_path``,
        ``cancel_path``, and normalized ``abs_path``. GET polls phase/progress
        to a terminal state; DELETE cooperatively cancels. Failures are
        represented on the terminal job and never replace either live session.
        """
        if viewer_state is None or viewer_session is None:
            return (
                jsonify({"status": "error", "error": "viewer hand-off disabled"}),
                501,
            )

        raw_payload = request.get_json(silent=True)
        if raw_payload is None:
            payload: dict[str, Any] = {}
        elif not isinstance(raw_payload, dict):
            return (
                jsonify(
                    {
                        "status": "error",
                        "error": "JSON body must be an object",
                    }
                ),
                400,
            )
        else:
            payload = raw_payload
        refresh = payload.get("refresh") is True
        rel = payload.get("path", "")
        if refresh and rel in ("", None):
            target: Path | None = None
        elif not isinstance(rel, str) or not rel:
            return (
                jsonify({"status": "error", "error": "missing 'path'"}),
                400,
            )
        else:
            try:
                target = sandbox.resolve(rel)
            except ValueError:
                return (
                    jsonify({"status": "error", "error": "path escapes sandbox"}),
                    400,
                )

        from phenotypic.gui.results_viewer._output_root import (
            OutputRoot,
            OutputSnapshotChangedError,
            sandbox_viewer_cache_root,
        )

        if binding_jobs is not None:
            selected = target
            if selected is None:
                selected = viewer_state.get("bound_path")
            if not isinstance(selected, Path):
                return (
                    jsonify(
                        {
                            "status": "error",
                            "error": "no output is currently bound",
                        }
                    ),
                    400,
                )
            try:
                submission = binding_jobs.submit(selected)
            except RuntimeError as exc:
                return (
                    jsonify({"status": "unavailable", "error": str(exc)}),
                    503,
                )
            payload = _binding_job_payload(
                submission.job,
                deduplicated=submission.deduplicated,
                browser_url_prefix=browser_url_prefix,
            )
            response = jsonify(payload)
            response.status_code = 202
            response.headers["Location"] = payload["poll_path"]
            return response

        try:
            if bind_output is not None:
                output_root = bind_output(target)
            else:
                if target is None:
                    raise ValueError("no output is currently bound")
                output_root = OutputRoot.discover(
                    target,
                    cache_root=sandbox_viewer_cache_root(sandbox.root),
                )
        except (OutputSnapshotChangedError, BindingSupersededError) as exc:
            logger.info("refused unstable viewer snapshot: %s", exc)
            return (
                jsonify(
                    {
                        "status": "stale",
                        "error": str(exc),
                    }
                ),
                409,
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.info(
                "rejected viewer hand-off for %s: %s", target, exc
            )
            return (
                jsonify({"status": "error", "error": str(exc)}),
                400,
            )
        except Exception as exc:  # noqa: BLE001 - preserve prior live sessions
            logger.exception(
                "viewer refresh construction failed for %s", target
            )
            return (
                jsonify(
                    {
                        "status": "unavailable",
                        "error": str(exc),
                    }
                ),
                500,
            )

        if bind_output is None:
            viewer_state["output_root"] = output_root
            viewer_session.release()
            viewer_session.touch()
            if extra_release_sessions:
                for sess in extra_release_sessions:
                    sess.release()
                    sess.touch()
        resolved_target = output_root.root
        snapshot = output_root.snapshot
        logger.info("viewer hand-off accepted: %s", resolved_target)
        return jsonify(
            {
                "status": "ok",
                "abs_path": str(resolved_target),
                "snapshot": {
                    "processing_fingerprint": snapshot.processing_fingerprint,
                    "consumed_state_fingerprint": (
                        snapshot.consumed_state_fingerprint
                    ),
                    "captured_at": snapshot.captured_at.isoformat(),
                    "active_run": snapshot.active_run,
                },
            }
        )

    @bp.route(
        "/viewer/output-root/jobs/<job_id>",
        methods=["GET", "DELETE"],
    )
    def viewer_output_root_job_endpoint(job_id: str) -> Any:
        """Poll or cooperatively cancel one asynchronous binding job."""
        if binding_jobs is None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "error": "asynchronous viewer hand-off disabled",
                    }
                ),
                501,
            )
        if request.method == "DELETE":
            snapshot = binding_jobs.cancel(job_id)
        else:
            snapshot = binding_jobs.get(job_id)
        if snapshot is None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "error": "binding job not found",
                    }
                ),
                404,
            )
        return jsonify(
            _binding_job_payload(
                snapshot,
                browser_url_prefix=browser_url_prefix,
            )
        )

    return bp


def register_sandbox_api(
    server: "Flask",
    sandbox: "SandboxRoot",
    *,
    viewer_session: "ToolSession[object] | None" = None,
    viewer_state: "dict[str, Any] | None" = None,
    extra_release_sessions: "tuple[ToolSession[object], ...] | None" = None,
    bind_output: "Callable[[Path | None], Any] | None" = None,
    binding_jobs: "ResultsBindJobManager | None" = None,
    browser_url_prefix: str = MOUNT_HOME,
) -> Blueprint:
    """Build and register the sandbox-API blueprint on ``server``."""
    bp = build_sandbox_api(
        sandbox,
        viewer_session=viewer_session,
        viewer_state=viewer_state,
        extra_release_sessions=extra_release_sessions,
        bind_output=bind_output,
        binding_jobs=binding_jobs,
        browser_url_prefix=browser_url_prefix,
    )
    server.register_blueprint(bp)
    logger.debug(
        "registered /sandbox/api blueprint on Flask app=%s sandbox=%s",
        server.name,
        sandbox.root,
    )
    return bp


def _binding_job_payload(
    snapshot: Any,
    *,
    deduplicated: bool | None = None,
    browser_url_prefix: str = MOUNT_HOME,
) -> dict[str, Any]:
    """Return a normalized polling payload with backward-compatible fields."""
    job = snapshot.as_dict()
    job_id = snapshot.job_id
    poll_path = (
        f"{join_url_prefix(browser_url_prefix, SANDBOX_API_VIEWER_OUTPUT_ROOT)}"
        f"/jobs/{job_id}"
    )
    payload: dict[str, Any] = {
        "status": snapshot.status,
        "job_id": job_id,
        "abs_path": str(snapshot.target),
        "job": job,
        "poll_path": poll_path,
        "cancel_path": poll_path,
    }
    if deduplicated is not None:
        payload["deduplicated"] = deduplicated
    if snapshot.result is not None:
        payload.update(dict(snapshot.result))
    return payload
