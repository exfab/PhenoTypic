/* Launch command live mirror (Task C1).
 *
 * Mirrors the pure Python ``render_launch_command`` (the unit-tested
 * source-of-truth in ``phenotypic.gui.tune._command``) into the live command
 * card as the Launch form changes. The Python function powers the initial
 * server-side render and the copy payload; this clientside mirror keeps the
 * card in sync without a server round-trip on every keystroke.
 *
 * Keep this EQUIVALENT to ``render_launch_command``: same subcommand
 * (``uv run phenotypic-tune run``), same positional spec + ``-i``/``-o``
 * flags, same optional-flag rules (``--n-trials`` only when a budget is given;
 * ``--storage-url`` only when non-empty; ``--screen``/``--slurm`` only when
 * toggled on). The CLI flag spellings are confirmed against
 * ``phenotypic.tune.__main__``'s ``run`` sub-parser.
 *
 * Tokens only: this file renders a shell string and references no colors /
 * fonts -- the command card owns its styling via the injected design tokens.
 */
window.dash_clientside = window.dash_clientside || {};

(function () {
    "use strict";

    /* POSIX shell-quote a single token, mirroring Python's ``shlex.quote``: a
     * token that is non-empty and contains only "safe" characters passes
     * through verbatim; anything else is single-quoted with embedded single
     * quotes escaped as ``'\''``. */
    function shQuote(token) {
        var s = String(token === null || token === undefined ? "" : token);
        if (s.length && /^[A-Za-z0-9_@%+=:,./-]+$/.test(s)) {
            return s;
        }
        return "'" + s.replace(/'/g, "'\\''") + "'";
    }

    /* Whether a single-item checklist's value list has the "on" sentinel. The
     * dbc.Checklist toggles return a list of selected option values; an empty
     * list (or null) means off. */
    function isOn(checklistValue) {
        return Array.isArray(checklistValue) && checklistValue.length > 0;
    }

    window.dash_clientside.tune_launch = {
        /* Render the ``uv run phenotypic-tune run …`` command from the form.
         *
         * Equivalent to ``render_launch_command``: see the module header for
         * the optional-flag rules. ``paths`` is the hidden paths store carrying
         * ``{spec, input, output}``; a missing store yields placeholders so the
         * card is never blank during the brief pre-store render. */
        renderCommand: function (strategy, nTrials, storageUrl, screen, slurm, paths) {
            var p = paths || {};
            var spec = p.spec || "<spec>";
            var input = p.input || "<images>";
            var output = p.output || "<output>";
            var resolvedStrategy = String(strategy || "tpe");
            var tokens = [
                "uv", "run", "phenotypic-tune", "run",
                spec,
                "-i", input,
                "-o", output,
                "--strategy", resolvedStrategy,
            ];
            // --n-trials only when a budget is given AND the strategy is not
            // grid (grid is exhaustive and ignores it — emitting it would be
            // misleading). Keep this equivalent to render_launch_command.
            if (
                nTrials !== null && nTrials !== undefined && nTrials !== "" &&
                resolvedStrategy !== "grid"
            ) {
                tokens.push("--n-trials", String(nTrials));
            }
            // --storage-url only when non-empty (local run resolves study.db).
            if (storageUrl) {
                tokens.push("--storage-url", String(storageUrl));
            }
            if (isOn(screen)) {
                tokens.push("--screen");
            }
            if (isOn(slurm)) {
                tokens.push("--slurm");
            }
            return tokens.map(shQuote).join(" ");
        },
    };
})();
