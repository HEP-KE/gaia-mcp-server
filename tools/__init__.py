"""Science tool package. Only names in __all__ become MCP tools.

The Gaia colour-magnitude example lives in cmd_tools.py; add your own modules
here and extend __all__ to expose new tools — nothing else changes.
"""

from .cmd_tools import (
    ArtifactResult,
    apply_quality_filters,
    compute_absolute_magnitudes,
    fetch_gaia_sample,
    plot_cmd,
)

__all__ = [
    "fetch_gaia_sample",
    "apply_quality_filters",
    "compute_absolute_magnitudes",
    "plot_cmd",
]
