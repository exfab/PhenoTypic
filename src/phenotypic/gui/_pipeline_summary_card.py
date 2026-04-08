"""Non-editable pipeline summary card for PhenoTypic GUI.

Displays a read-only summary of a loaded pipeline showing its operations
without edit controls.
"""

from __future__ import annotations


class PipelineSummaryCard:
    """Non-editable summary display for a loaded pipeline.

    Shows pipeline name and list of operations without edit controls.
    Used when:
    - Nesting depth exceeds MAX_NESTING_DEPTH
    - User loads a saved pipeline as a parameter value
    """

    def __init__(
        self,
        pipeline,  # ImagePipeline
        name: str = "Loaded Pipeline",
    ):
        """Initialize PipelineSummaryCard.

        Args:
            pipeline: ImagePipeline to display
            name: Display name for the pipeline
        """
        self._pipeline = pipeline
        self._name = name

    def panel(self):
        """Build non-editable summary card.

        Returns:
            Panel Card widget showing pipeline summary
        """
        import panel as pn

        # List operations as bullet points
        ops_list = []
        for op_name, op in self._pipeline._ops.items():
            op_type = op.__class__.__name__
            ops_list.append(f"- **{op_name}**: {op_type}")

        ops_md = "\n".join(ops_list) if ops_list else "*Empty pipeline*"

        return pn.Card(
            pn.pane.Markdown(f"**{self._name}**\n\n{ops_md}"),
            header="Pipeline (read-only)",
            collapsed=False,
            sizing_mode="stretch_width",
            styles={"background": "#f0f0f0", "border": "1px solid #ccc"},
        )
