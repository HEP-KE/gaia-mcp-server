"""Publication-style figure helpers shared by every Gaia plotting tool.

Same conventions as the other HEP-Genesis MCP servers:
- NO system LaTeX (hosted VMs have none): math uses matplotlib's built-in
  mathtext with the STIX fonts, which ship with matplotlib.
- Nothing overflows the canvas: constrained layout, wrapped titles, short
  axis labels, a tight bounding box on save.
- A fixed-order, colorblind-validated categorical palette for highlighted
  populations; single-hue sequential maps for star densities.

Astronomy specifics: magnitude axes run bright-at-top; small samples (a
5-10 pc sphere has a few hundred stars) are drawn as points, because a
300x300 density histogram of 200 stars is a scatter of single-count pixels.
"""

import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# Okabe-Ito-derived, reordered so adjacent slots stay separable under
# deutan/protan colour-vision deficiency (validated: adjacent CVD dE >= 11).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#882255",
           "#B8860B", "#CC79A7", "#6B4C9A", "#E07B39"]
INK = "#222222"
MUTED = "#6b6b6b"
FIELD_GREY = "Greys"        # background field-star density under highlights
DENSITY_CMAP = "viridis"    # perceptually uniform, prints in greyscale

# Below this many stars a CMD is drawn as points rather than a density map.
SCATTER_BELOW = 5000

BP_RP = r"$G_{\rm BP} - G_{\rm RP}$"
ABS_G = r"$M_G$"


def rc_params(base_size: float = 12.0) -> dict:
    return {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base_size,
        "axes.labelsize": base_size + 1,
        "axes.titlesize": base_size,
        "legend.fontsize": base_size - 1.5,
        "xtick.labelsize": base_size - 0.5,
        "ytick.labelsize": base_size - 0.5,
        "axes.linewidth": 0.9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "lines.linewidth": 1.8,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "figure.dpi": 100,
        "axes.unicode_minus": True,
    }


@contextmanager
def style():
    """Agg backend + the house style for the duration of one figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(rc_params()):
        yield plt


def wrap(text: str, width: int = 58) -> str:
    """Wrap plain text without breaking inside $...$ math spans."""
    if len(text) <= width:
        return text
    tokens = re.findall(r"\$[^$]*\$|\S+", text)
    lines, current = [], ""
    for tok in tokens:
        candidate = f"{current} {tok}".strip()
        visible = len(re.sub(r"\\[a-zA-Z]+|[{}$^_\\]", "", candidate))
        if visible > width and current:
            lines.append(current)
            current = tok
        else:
            current = candidate
    lines.append(current)
    return "\n".join(lines)


def use_density(*sizes: int) -> bool:
    """One rendering mode per figure: density if any panel is large."""
    return max(sizes, default=0) >= SCATTER_BELOW


def shared_norm(panels, *, x_range, y_range, bins):
    """A LogNorm shared by several density panels, so one colorbar is true
    for all of them. panels: iterable of (x, y)."""
    from matplotlib.colors import LogNorm

    peak = 2.0
    for x, y in panels:
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.any():
            counts, _, _ = np.histogram2d(x[ok], y[ok], bins=bins,
                                          range=[x_range, y_range])
            peak = max(peak, float(counts.max()))
    return LogNorm(vmin=1, vmax=peak)


def density(ax, x, y, *, x_range, y_range, bins=300, cmap=DENSITY_CMAP,
            point_color=INK, invert_y=True, as_density=None, norm=None):
    """Star density in (x, y): log-scaled 2D histogram for large samples,
    points for small ones. as_density / norm let multi-panel figures force
    one mode and one colour scale. Returns the mappable for a colorbar, or
    None when points were drawn (or there is nothing to draw)."""
    from matplotlib.colors import LogNorm

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    mappable = None
    if as_density is None:
        as_density = use_density(len(x))
    if as_density and len(x):
        counts, xe, ye = np.histogram2d(x, y, bins=bins, range=[x_range, y_range])
        masked = np.ma.masked_less(counts.T, 1)
        if masked.count():
            mappable = ax.pcolormesh(
                xe, ye, masked, cmap=cmap, rasterized=True,
                norm=norm or LogNorm(vmin=1, vmax=max(masked.max(), 2)))
    elif len(x):
        size = 10 if len(x) < 500 else 3
        ax.scatter(x, y, s=size, color=point_color, alpha=0.75, linewidths=0,
                   rasterized=len(x) > 1000)
    ax.set_xlim(*x_range)
    ax.set_ylim(*(y_range[::-1] if invert_y else y_range))
    return mappable


def empty_note(ax, n: int) -> None:
    if n == 0:
        ax.text(0.5, 0.5, "no stars in this selection", transform=ax.transAxes,
                ha="center", va="center", color=MUTED)


def sample_radius_pc(parallax_mas) -> float | None:
    """Outer radius of a parallax-limited sample: 1000 / min(parallax),
    rounded to 2 significant figures (the cut, not the edge star: a 10 pc
    sample's most distant star sits at 9.97 pc)."""
    plx = np.asarray(parallax_mas, dtype=float)
    plx = plx[np.isfinite(plx) & (plx > 0)]
    if not len(plx):
        return None
    return float(f"{1000.0 / plx.min():.2g}")


def save(fig, path: Path, save_pdf: bool = False) -> list[str]:
    import matplotlib.pyplot as plt

    fig.savefig(path, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    files = [str(path)]
    if save_pdf:
        pdf = path.with_suffix(".pdf")
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.06)
        files.append(str(pdf))
    plt.close(fig)
    return files
