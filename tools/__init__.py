"""Science tool package. Only names in __all__ become MCP tools.

The Gaia colour-magnitude example lives in cmd_tools.py; add your own modules
here and extend __all__ to expose new tools — nothing else changes.
"""

from .cmd_tools import (
    ArtifactResult,
    apply_gaia_quality_filters,
    compare_distance_shells,
    compute_gaia_absolute_magnitudes,
    fetch_gaia_sample,
    plot_gaia_cmd,
    plot_hyades,
    plot_infrared_cmd,
    plot_kinematics_cmd,
    plot_gaia_luminosity_function,
    plot_gaia_sky_map,
    plot_variable_stars_cmd,
    plot_white_dwarfs,
)

__all__ = [
    "fetch_gaia_sample",
    "apply_gaia_quality_filters",
    "compute_gaia_absolute_magnitudes",
    "plot_gaia_cmd",
    "compare_distance_shells",
    "plot_kinematics_cmd",
    "plot_variable_stars_cmd",
    "plot_infrared_cmd",
    "plot_gaia_sky_map",
    "plot_hyades",
    "plot_white_dwarfs",
    "plot_gaia_luminosity_function",
]
