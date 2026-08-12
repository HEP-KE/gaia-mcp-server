"""MCP tool functions for the Gaia colour-magnitude diagram.

Plain Python functions — no MCP imports. The type hints, Field constraints,
and docstrings below become the MCP tool schema that agents see.

Data flows between tools as CSV file paths: fetch_gaia_sample writes the raw
sample, apply_quality_filters cleans it, compute_absolute_magnitudes turns it
into a CMD table, plot_cmd draws it. Only small numbers (row counts, file
paths) ever pass through the LLM context.

Target figure: Gaia Collaboration, Babusiaux et al. (2018), A&A 616, A10,
Fig. 5c — the Hertzsprung-Russell diagram of the 212,728 stars within 100 pc.
"""

from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, Field, validate_call

from . import gaia


class ArtifactResult(BaseModel):
    """Uniform result contract returned by every tool."""

    status: Literal["success"]
    files: list[str]
    message: str
    metadata: dict[str, Any]


def _outdir(output_dir: str) -> Path:
    path = Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


@validate_call
def fetch_gaia_sample(
    output_dir: Annotated[str, Field(min_length=1)],
    min_parallax_mas: Annotated[float, Field(ge=1.0, le=100.0)] = 10.0,
    min_parallax_snr: Annotated[float, Field(ge=0.0, le=100.0)] = 10.0,
    source: Literal["auto", "archive", "bundled"] = "auto",
) -> ArtifactResult:
    """Fetch the Gaia DR2 solar-neighbourhood sample and write it to a CSV.

    Use this tool first. The default cuts select the 100 pc sample of
    Babusiaux et al. (2018) Fig. 5c: parallax >= 10 mas (distance < 100 pc)
    and parallax_over_error > 10 (distance good to ~10%, which is what makes
    d = 1/parallax a safe estimator here). The query downloads ALL matching
    rows — no TOP truncation, which would NOT be a random subsample.

    Args:
        output_dir: Directory where the CSV is written.
        min_parallax_mas: Parallax floor in mas; 10 mas = a 100 pc sphere.
        min_parallax_snr: Minimum parallax/parallax_error. Raising it gives
            better distances but preferentially removes faint red stars —
            a biased, not just smaller, sample.
        source: "archive" queries the ESA Gaia archive live (needs network,
            ~1-2 min); "bundled" uses the snapshot shipped in data/;
            "auto" tries the archive and falls back to the snapshot.
    """
    used, note = source, ""
    if source == "archive":
        data = gaia.query_archive(min_parallax_mas, min_parallax_snr)
    elif source == "bundled":
        data = gaia.load_bundled(min_parallax_mas, min_parallax_snr)
    else:
        try:
            data = gaia.query_archive(min_parallax_mas, min_parallax_snr)
            used = "archive"
        except Exception as exc:  # archive down, offline, astroquery missing
            data = gaia.load_bundled(min_parallax_mas, min_parallax_snr)
            used = "bundled"
            note = f" (archive unavailable: {type(exc).__name__}; used bundled snapshot)"

    csv_path = _outdir(output_dir) / "gaia_sample.csv"
    gaia.write_sample_csv(data, csv_path)
    return ArtifactResult(
        status="success",
        files=[str(csv_path)],
        message=(
            f"Fetched {len(data):,} Gaia DR2 sources with parallax >= "
            f"{min_parallax_mas:g} mas and parallax SNR > {min_parallax_snr:g} "
            f"from the {used}{note}."
        ),
        metadata={
            "n_rows": len(data),
            "source": used,
            "adql": gaia.build_adql(min_parallax_mas, min_parallax_snr),
            "columns": list(gaia.COLUMNS),
        },
    )


@validate_call
def apply_quality_filters(
    input_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
    min_phot_g_snr: Annotated[float, Field(ge=0.0)] = 50.0,
    min_phot_bprp_snr: Annotated[float, Field(ge=0.0)] = 20.0,
    apply_excess_factor_cut: bool = True,
    apply_astrometry_cut: bool = True,
) -> ArtifactResult:
    """Apply the Babusiaux et al. (2018) quality filters to a Gaia sample CSV.

    Use this tool after fetch_gaia_sample. The defaults reproduce the
    published selection (their Sect. 2.1); with them, the 100 pc sample
    yields the paper's 212,728 stars. The returned metadata reports how many
    stars each filter removes on its own, plus a one-line justification for
    each — quote these when explaining your selection.

    Args:
        input_file: CSV written by fetch_gaia_sample.
        output_dir: Directory where the cleaned CSV is written.
        min_phot_g_snr: G-band flux SNR floor (paper: 50).
        min_phot_bprp_snr: BP and RP flux SNR floor (paper: 20).
        apply_excess_factor_cut: Remove blended/contaminated BP-RP photometry
            via the photometric excess factor (paper: on).
        apply_astrometry_cut: Remove poor astrometric solutions via
            visibility periods and the unit weight error, the DR2-era
            precursor of RUWE (paper: on).
    """
    data = gaia.load_sample_csv(input_file)
    n_input = len(data)

    masks = {"phot_g_snr": gaia._g_snr(data, min_phot_g_snr),
             "phot_bprp_snr": gaia._bprp_snr(data, min_phot_bprp_snr)}
    if apply_excess_factor_cut:
        masks["excess_factor"] = gaia._excess_factor(data)
    if apply_astrometry_cut:
        masks["astrometry"] = gaia._astrometry(data)

    combined = np.ones(n_input, dtype=bool)
    removed_alone = {}
    for name, mask in masks.items():
        mask = mask & ~np.isnan(data["bp_rp"])  # no colour -> cannot be plotted
        removed_alone[name] = int(n_input - mask.sum())
        combined &= mask
    clean = data[combined]

    csv_path = _outdir(output_dir) / "gaia_sample_clean.csv"
    gaia.write_sample_csv(clean, csv_path)
    return ArtifactResult(
        status="success",
        files=[str(csv_path)],
        message=(
            f"Quality filters kept {len(clean):,} of {n_input:,} stars "
            f"(published 100 pc count: {gaia.PUBLISHED_100PC_COUNT:,})."
        ),
        metadata={
            "n_input": n_input,
            "n_output": len(clean),
            "removed_by_each_filter_alone": removed_alone,
            "justifications": {k: gaia.JUSTIFICATIONS[k] for k in masks},
            "published_count_fig5c": gaia.PUBLISHED_100PC_COUNT,
        },
    )


@validate_call
def compute_absolute_magnitudes(
    input_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
) -> ArtifactResult:
    """Convert apparent G magnitudes to absolute using inverted parallaxes.

    Use this tool after apply_quality_filters. It computes
    M_G = G + 5 log10(parallax/mas) - 10 and writes a CMD table (columns:
    bp_rp, abs_g_mag). Inverting the parallax is only a safe distance
    estimator because this sample requires parallax SNR > 10: for noisy or
    negative parallaxes 1/parallax is biased or meaningless, and one should
    infer distances properly (e.g. Bailer-Jones et al. 2018). No extinction
    correction is applied — within 100 pc it is negligible.

    Args:
        input_file: CSV written by apply_quality_filters (or fetch_gaia_sample).
        output_dir: Directory where the CMD CSV is written.
    """
    data = gaia.load_sample_csv(input_file)
    n_input = len(data)
    valid = (data["parallax"] > 0) & ~np.isnan(data["bp_rp"])
    n_dropped = int(n_input - valid.sum())
    data = data[valid]

    abs_g = data["phot_g_mean_mag"] + 5 * np.log10(data["parallax"]) - 10

    csv_path = _outdir(output_dir) / "gaia_cmd.csv"
    np.savetxt(
        csv_path,
        np.column_stack([data["bp_rp"], abs_g]),
        delimiter=",",
        header="bp_rp,abs_g_mag",
        comments="",
        fmt="%.6f",
    )
    return ArtifactResult(
        status="success",
        files=[str(csv_path)],
        message=(
            f"Computed M_G for {len(data):,} stars"
            + (f" (dropped {n_dropped} with non-positive parallax or no colour)"
               if n_dropped else "")
            + "."
        ),
        metadata={
            "n_stars": len(data),
            "n_dropped": n_dropped,
            "formula": "M_G = phot_g_mean_mag + 5*log10(parallax_mas) - 10",
            "abs_g_range": [float(abs_g.min()), float(abs_g.max())] if len(data) else None,
        },
    )


@validate_call
def plot_cmd(
    input_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
    color_min: float = -1.0,
    color_max: float = 5.0,
    mag_bright: float = -5.0,
    mag_faint: float = 17.0,
    n_bins: Annotated[int, Field(ge=50, le=1000)] = 300,
) -> ArtifactResult:
    """Draw the density colour-magnitude diagram (observational HRD).

    Use this tool last, on the CSV written by compute_absolute_magnitudes.
    It draws a log-scaled 2D density of M_G vs BP-RP with the magnitude axis
    inverted (bright at the top), axes matched by default to Babusiaux et al.
    (2018) Fig. 5c. In the 100 pc diagram you should be able to identify the
    main sequence, the binary sequence just above it, the red clump near
    BP-RP = 1.2, M_G = 0.5, and the white dwarf sequence in the lower left.

    Args:
        input_file: CSV written by compute_absolute_magnitudes.
        output_dir: Directory where the PNG is written.
        color_min: Left edge of the BP-RP axis.
        color_max: Right edge of the BP-RP axis.
        mag_bright: Top of the M_G axis (bright end).
        mag_faint: Bottom of the M_G axis (faint end).
        n_bins: Histogram bins per axis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    cmd = np.genfromtxt(Path(input_file).expanduser().resolve(),
                        delimiter=",", names=True)
    color, abs_g = cmd["bp_rp"], cmd["abs_g_mag"]

    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    counts, _, _, image = ax.hist2d(
        color, abs_g,
        bins=n_bins,
        range=[[color_min, color_max], [mag_bright, mag_faint]],
        norm=LogNorm(), cmap="viridis", cmin=1,
    )
    ax.set_xlim(color_min, color_max)
    ax.set_ylim(mag_faint, mag_bright)  # bright stars at the top
    ax.set_xlabel(r"$G_{BP} - G_{RP}$")
    ax.set_ylabel(r"$M_G$")
    n_shown = int(np.isfinite(color).sum())
    ax.set_title(f"Gaia DR2 HRD, d < 100 pc — {n_shown:,} stars")
    fig.colorbar(image, ax=ax, label="stars per bin")

    plot_path = _outdir(output_dir) / "gaia_cmd_hrd.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return ArtifactResult(
        status="success",
        files=[str(plot_path)],
        message=(
            f"Plotted the density CMD of {n_shown:,} stars "
            f"(published Fig. 5c count: {gaia.PUBLISHED_100PC_COUNT:,})."
        ),
        metadata={
            "n_stars": n_shown,
            "published_count_fig5c": gaia.PUBLISHED_100PC_COUNT,
            "axes": {"bp_rp": [color_min, color_max],
                     "abs_g_mag": [mag_faint, mag_bright]},
            "reference": "Babusiaux et al. 2018, A&A 616, A10, Fig. 5c",
        },
    )


@validate_call
def compare_distance_shells(
    input_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
    distances_pc: Annotated[list[float], Field(min_length=2, max_length=4)] = [25.0, 50.0, 100.0],
) -> ArtifactResult:
    """Draw side-by-side HRDs for nested distance shells (full Fig. 5).

    Use this tool on the CSV written by apply_quality_filters (it needs the
    parallax column, so NOT the compute_absolute_magnitudes output). The
    default distances reproduce the three panels of Babusiaux et al. (2018)
    Fig. 5: stars within 25, 50, and 100 pc. Nearby shells contain far fewer
    stars but reach fainter absolute magnitudes — the sample is
    volume-limited in parallax yet magnitude-limited in G, so the faint end
    of the diagram is only complete close to the Sun.

    Args:
        input_file: CSV written by apply_quality_filters (or
            fetch_gaia_sample) — must still contain the parallax column.
        output_dir: Directory where the PNG is written.
        distances_pc: Shell radii in parsec, small to large. A star is in a
            shell when parallax >= 1000/distance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    data = gaia.load_sample_csv(input_file)
    if "parallax" not in (data.dtype.names or ()):
        raise ValueError(
            "input_file has no parallax column; pass the CSV from "
            "apply_quality_filters, not from compute_absolute_magnitudes."
        )
    distances = sorted(distances_pc)

    fig, axes = plt.subplots(1, len(distances),
                             figsize=(4.6 * len(distances), 6.2),
                             sharex=True, sharey=True)
    counts = {}
    for ax, d_pc in zip(np.atleast_1d(axes), distances):
        shell = data[(data["parallax"] >= 1000.0 / d_pc)
                     & ~np.isnan(data["bp_rp"])]
        abs_g = shell["phot_g_mean_mag"] + 5 * np.log10(shell["parallax"]) - 10
        ax.hist2d(shell["bp_rp"], abs_g, bins=250,
                  range=[[-1, 5], [-5, 17]], norm=LogNorm(),
                  cmap="viridis", cmin=1)
        ax.set_xlim(-1, 5)
        ax.set_ylim(17, -5)
        ax.set_xlabel(r"$G_{BP} - G_{RP}$")
        ax.set_title(f"d < {d_pc:g} pc — {len(shell):,} stars")
        counts[f"{d_pc:g}_pc"] = len(shell)
    np.atleast_1d(axes)[0].set_ylabel(r"$M_G$")
    fig.suptitle("Gaia DR2 HRD by distance shell", y=0.99)

    plot_path = _outdir(output_dir) / "gaia_cmd_shells.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return ArtifactResult(
        status="success",
        files=[str(plot_path)],
        message=(
            "Plotted HRDs for "
            + ", ".join(f"d < {d:g} pc ({counts[f'{d:g}_pc']:,} stars)"
                        for d in distances)
            + "."
        ),
        metadata={
            "star_counts": counts,
            "published_count_100pc": gaia.PUBLISHED_100PC_COUNT,
            "reference": "Babusiaux et al. 2018, A&A 616, A10, Fig. 5",
        },
    )
