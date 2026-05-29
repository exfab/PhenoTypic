"""Measurement info container for the generic QualityCheck output columns."""

from textwrap import dedent

from ._measurement_info import MeasurementInfo


class QUALITY_CHECK(MeasurementInfo):
    """Generic QC output columns emitted by every QualityCheck subclass.

    The category convention used elsewhere in the codebase would produce
    column names ``QC_Flag``, ``QC_Metric``, ``QC_Status`` — but the
    actual emitted columns include each subclass's ``name`` between the
    category and the label (``QC_Count_Flag``, ``QC_SE_Flag``). This
    enum overrides :meth:`append_rst_to_doc` to substitute the subclass's
    ``name`` into the RST column header, so each concrete subclass's
    docstring documents its real emitted columns.
    """

    FLAG = (
        "Flag",
        "True when the metric crosses fail_threshold in the bad direction; "
        "eligible for curation.",
    )
    METRIC = (
        "Metric",
        "Headline metric in the check's own units; the bad direction is set "
        "by the check's _HIGHER_IS_BAD flag. Drives Status.",
    )
    STATUS = ("Status", "Categorical: pass | warn | fail.")

    @classmethod
    def category(cls) -> str:
        return "QC"

    @classmethod
    def append_rst_to_doc(  # type: ignore[override]
        cls,
        doc: str | object,
        *,
        check_name: str | None = None,
    ) -> str:
        """Append an RST table with optional per-check name substitution.

        Args:
            doc: Existing class docstring to splice into. If a string, it
                is treated as the docstring itself; otherwise its
                ``__doc__`` attribute is used.
            check_name: When provided, column names render as
                ``QC_<check_name>_<Label>`` (e.g. ``QC_Count_Flag``).
                When omitted (called on the base ``QualityCheck`` class
                or for general docs), column names render as
                ``QC_<name>_<Label>`` with ``<name>`` as a literal
                placeholder, signalling the per-subclass substitution.

        Returns:
            Docstring with an appended RST table of output columns.
        """
        slug = check_name if check_name is not None else "<name>"
        title = f"QC_{slug}"
        lines = [
            f".. list-table:: Category: **{title}**",
            "   :header-rows: 1",
            "",
            "   * - Name",
            "     - Description",
        ]
        for m in cls:
            label: str = m.label  # type: ignore[attr-defined]
            desc: str = m.desc  # type: ignore[attr-defined]
            lines += [
                f"   * - ``QC_{slug}_{label}``",
                f"     - {desc}",
            ]
        table = dedent("\n".join(lines))
        if isinstance(doc, str):
            base = doc
        else:
            base = doc.__doc__ or ""
        return base + "\n\n" + table
