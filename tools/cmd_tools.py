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


def _require_columns(data, names) -> None:
    missing = [n for n in names if n not in (data.dtype.names or ())]
    if missing:
        raise ValueError(
            f"input_file lacks the column(s) {missing}; pass the CSV written "
            "by fetch_gaia_sample or apply_quality_filters (the "
            "compute_absolute_magnitudes output keeps only bp_rp/abs_g_mag)."
        )


def _abs_g(data) -> "np.ndarray":
    return data["phot_g_mean_mag"] + 5 * np.log10(data["parallax"]) - 10


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


@validate_call
def plot_kinematics_cmd(
    input_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
    slow_max_km_s: Annotated[float, Field(gt=0)] = 40.0,
    mid_range_km_s: tuple[float, float] = (60.0, 150.0),
    halo_min_km_s: Annotated[float, Field(gt=0)] = 200.0,
) -> ArtifactResult:
    """Slice the HRD by tangential velocity (the paper's Fig. 7).

    Use this tool on the CSV from apply_quality_filters. The tangential
    velocity v_T = 4.74 * pm[mas/yr] / parallax[mas] km/s needs only Gaia
    astrometry, and slicing on it separates stellar populations by age and
    origin: slow stars are the young thin disc (upper main sequence
    present), fast stars are old (no upper main sequence, subdwarfs sitting
    blueward of the main sequence, halo white dwarfs). Writes TWO figures:
    the three velocity-sliced HRDs, and a map of mean v_T across the CMD.

    Args:
        input_file: CSV from fetch_gaia_sample or apply_quality_filters
            (needs pmra, pmdec, parallax).
        output_dir: Directory where the PNGs are written.
        slow_max_km_s: Upper v_T bound of the "thin disc" panel.
        mid_range_km_s: (low, high) v_T bounds of the middle panel.
        halo_min_km_s: Lower v_T bound of the "halo" panel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    data = gaia.load_sample_csv(input_file)
    _require_columns(data, ["pmra", "pmdec", "parallax", "phot_g_mean_mag", "bp_rp"])
    abs_g, color = _abs_g(data), data["bp_rp"]
    v_tan = 4.74047 * np.hypot(data["pmra"], data["pmdec"]) / data["parallax"]

    mid_lo, mid_hi = mid_range_km_s
    slices = [
        (f"$v_T$ < {slow_max_km_s:g} km/s — mostly thin disc", v_tan < slow_max_km_s),
        (f"{mid_lo:g} < $v_T$ < {mid_hi:g} km/s — older discs",
         (v_tan > mid_lo) & (v_tan < mid_hi)),
        (f"$v_T$ > {halo_min_km_s:g} km/s — halo", v_tan > halo_min_km_s),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharex=True, sharey=True)
    counts = {}
    for ax, (title, sel) in zip(axes, slices):
        ax.hist2d(color[sel], abs_g[sel], bins=200, range=[[-1, 5], [-5, 17]],
                  norm=LogNorm(), cmap="viridis", cmin=1)
        ax.set_ylim(17, -5)
        ax.set_xlabel(r"$G_{BP} - G_{RP}$")
        ax.set_title(f"{title}\n{int(sel.sum()):,} stars")
        counts[title.split("$")[0].strip() or title] = int(sel.sum())
    axes[0].set_ylabel(r"$M_G$")
    slices_path = _outdir(output_dir) / "gaia_cmd_velocity_slices.png"
    fig.savefig(slices_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    H_n, xe, ye = np.histogram2d(color, abs_g, bins=200, range=[[-1, 5], [-5, 17]])
    H_v, _, _ = np.histogram2d(color, abs_g, bins=200, range=[[-1, 5], [-5, 17]],
                               weights=np.nan_to_num(v_tan))
    mean_v = np.where(H_n >= 3, H_v / np.maximum(H_n, 1), np.nan)
    fig, ax = plt.subplots(figsize=(6.8, 7))
    im = ax.pcolormesh(xe, ye, mean_v.T, cmap="magma", vmin=10, vmax=100)
    ax.set_ylim(17, -5)
    ax.set_xlabel(r"$G_{BP} - G_{RP}$")
    ax.set_ylabel(r"$M_G$")
    ax.set_title("Mean tangential velocity across the HRD")
    fig.colorbar(im, ax=ax, label=r"mean $v_T$ [km/s] (bins with $\geq$ 3 stars)")
    map_path = _outdir(output_dir) / "gaia_cmd_mean_vtan.png"
    fig.savefig(map_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    slice_counts = [int(sel.sum()) for _, sel in slices]
    return ArtifactResult(
        status="success",
        files=[str(slices_path), str(map_path)],
        message=(
            f"Velocity-sliced HRDs: {slice_counts[0]:,} slow / "
            f"{slice_counts[1]:,} intermediate / {slice_counts[2]:,} halo "
            "stars, plus the mean-v_T map."
        ),
        metadata={
            "v_tan_slices_km_s": {"slow_max": slow_max_km_s,
                                  "mid": list(mid_range_km_s),
                                  "halo_min": halo_min_km_s},
            "slice_star_counts": slice_counts,
            "reference": "Babusiaux et al. 2018, A&A 616, A10, Fig. 7",
        },
    )


@validate_call
def plot_variable_stars_cmd(
    input_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
) -> ArtifactResult:
    """Highlight DR2's flagged variable stars on the HRD (the paper's Fig. 15).

    Use this tool on the CSV from apply_quality_filters. Within 100 pc the
    flagged variables are almost entirely flaring and spotted M dwarfs on
    the lower main sequence; the bright pulsators that fill this figure in
    the all-sky sample (Cepheids, RR Lyrae) have no representatives this
    close to the Sun.

    Args:
        input_file: CSV from fetch_gaia_sample or apply_quality_filters
            (needs the "variable" 0/1 column).
        output_dir: Directory where the PNG is written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    data = gaia.load_sample_csv(input_file)
    _require_columns(data, ["variable", "parallax", "phot_g_mean_mag", "bp_rp"])
    abs_g, color = _abs_g(data), data["bp_rp"]
    variable = data["variable"] > 0.5

    fig, ax = plt.subplots(figsize=(6.8, 7))
    ax.hist2d(color, abs_g, bins=300, range=[[-1, 5], [-5, 17]],
              norm=LogNorm(), cmap="Greys", cmin=1)
    ax.scatter(color[variable], abs_g[variable], s=8, color="C3",
               label=f"flagged VARIABLE — {int(variable.sum()):,} stars")
    ax.set_ylim(17, -5)
    ax.set_xlabel(r"$G_{BP} - G_{RP}$")
    ax.set_ylabel(r"$M_G$")
    ax.legend(loc="upper right")

    plot_path = _outdir(output_dir) / "gaia_cmd_variables.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ArtifactResult(
        status="success",
        files=[str(plot_path)],
        message=(
            f"Marked {int(variable.sum()):,} flagged variables on the HRD "
            f"of {len(data):,} stars."
        ),
        metadata={
            "n_variable": int(variable.sum()),
            "n_total": len(data),
            "reference": "Babusiaux et al. 2018, A&A 616, A10, Fig. 15",
        },
    )


@validate_call
def plot_infrared_cmd(
    input_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
) -> ArtifactResult:
    """Draw the infrared HRD from the 2MASS cross-match (the paper's Fig. 6).

    Use this tool on the CSV from apply_quality_filters. The sample carries
    2MASS J and Ks from a server-side cross-match (NaN where unmatched). In
    the infrared the main sequence is less sensitive to metallicity and the
    M dwarfs bunch up; white dwarfs are largely too faint for 2MASS and
    drop out — a photometric completeness lesson in one panel.

    Args:
        input_file: CSV from fetch_gaia_sample or apply_quality_filters
            (needs j_m, ks_m).
        output_dir: Directory where the PNG is written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    data = gaia.load_sample_csv(input_file)
    _require_columns(data, ["j_m", "ks_m", "parallax"])
    matched = np.isfinite(data["j_m"]) & np.isfinite(data["ks_m"])
    subset = data[matched]
    abs_ks = subset["ks_m"] + 5 * np.log10(subset["parallax"]) - 10
    j_ks = subset["j_m"] - subset["ks_m"]

    fig, ax = plt.subplots(figsize=(6.8, 7))
    h = ax.hist2d(j_ks, abs_ks, bins=250, range=[[-0.4, 1.4], [-6, 11]],
                  norm=LogNorm(), cmap="viridis", cmin=1)
    ax.set_ylim(11, -6)
    ax.set_xlabel(r"$J - K_s$")
    ax.set_ylabel(r"$M_{K_s}$")
    ax.set_title(f"2MASS HRD — {int(matched.sum()):,} of {len(data):,} stars matched")
    fig.colorbar(h[3], ax=ax, label="stars per bin")

    plot_path = _outdir(output_dir) / "gaia_cmd_infrared.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ArtifactResult(
        status="success",
        files=[str(plot_path)],
        message=(
            f"Infrared HRD of {int(matched.sum()):,} 2MASS-matched stars "
            f"(of {len(data):,})."
        ),
        metadata={
            "n_matched": int(matched.sum()),
            "n_total": len(data),
            "reference": "Babusiaux et al. 2018, A&A 616, A10, Fig. 6",
        },
    )


@validate_call
def plot_sky_map(
    input_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
) -> ArtifactResult:
    """Map the sample on the sky in galactic coordinates.

    Use this tool on the CSV from apply_quality_filters. Within 100 pc the
    sky should be nearly isotropic — and almost is: the overdensity at
    l = 180, b = -22 is the Hyades, the nearest open cluster (d = 47 pc),
    and the stark empty patches are regions Gaia's scanning law had visited
    too few times by DR2, emptied entirely by the visibility_periods_used
    cut. Quality filters imprint the survey's geometry on the sample.

    Args:
        input_file: CSV from fetch_gaia_sample or apply_quality_filters
            (needs l, b).
        output_dir: Directory where the PNG is written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    data = gaia.load_sample_csv(input_file)
    _require_columns(data, ["l", "b"])

    fig, ax = plt.subplots(figsize=(11, 5))
    h = ax.hist2d(data["l"], data["b"], bins=[360, 180],
                  range=[[0, 360], [-90, 90]], norm=LogNorm(), cmap="viridis")
    ax.set_xlabel("galactic longitude $l$ [deg]")
    ax.set_ylabel("galactic latitude $b$ [deg]")
    ax.set_title(f"{len(data):,} stars on the sky")
    ax.annotate("Hyades", (180, -22), xytext=(230, -55), color="white",
                arrowprops=dict(arrowstyle="->", color="white"))
    fig.colorbar(h[3], ax=ax, label="stars per deg$^2$ bin")

    plot_path = _outdir(output_dir) / "gaia_sky_map.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ArtifactResult(
        status="success",
        files=[str(plot_path)],
        message=f"Sky map of {len(data):,} stars in galactic coordinates.",
        metadata={
            "n_stars": len(data),
            "landmarks": {"Hyades": {"l_deg": 180, "b_deg": -22, "d_pc": 47}},
        },
    )


@validate_call
def plot_hyades(
    input_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
) -> ArtifactResult:
    """Extract the Hyades cluster from the field and draw its HRD.

    Use this tool on the CSV from apply_quality_filters. The Hyades is the
    nearest open cluster (d = 47 pc) and its members share one proper
    motion and one parallax — a box in (parallax, pmra, pmdec, l, b) pulls
    them cleanly out of the field with no colour information used at all.
    Because the cluster is a single age (~700 Myr) and single metallicity,
    its main sequence is razor thin compared to the field's spread; its
    unresolved binaries stand out above it (the paper studies 46 clusters
    this way, Sect. 4).

    Args:
        input_file: CSV from fetch_gaia_sample or apply_quality_filters
            (needs parallax, pmra, pmdec, l, b).
        output_dir: Directory where the PNG is written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    data = gaia.load_sample_csv(input_file)
    _require_columns(data, ["parallax", "pmra", "pmdec", "l", "b",
                            "phot_g_mean_mag", "bp_rp"])
    members = (
        (data["parallax"] > 19) & (data["parallax"] < 24)
        & (data["pmra"] > 80) & (data["pmra"] < 140)
        & (data["pmdec"] > -60) & (data["pmdec"] < -10)
        & (np.abs(data["l"] - 180) < 20) & (np.abs(data["b"] + 22) < 20)
    )
    cluster = data[members]
    abs_g_all, abs_g_cl = _abs_g(data), _abs_g(cluster)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.5))
    ax1.scatter(data["l"], data["b"], s=1, color="0.8")
    ax1.scatter(cluster["l"], cluster["b"], s=6, color="C3")
    ax1.set_xlim(220, 140)
    ax1.set_ylim(-45, 5)
    ax1.set_xlabel("galactic longitude $l$ [deg]")
    ax1.set_ylabel("galactic latitude $b$ [deg]")
    ax1.set_title(f"Hyades members on the sky — {int(members.sum()):,} stars")

    ax2.hist2d(data["bp_rp"], abs_g_all, bins=300, range=[[-1, 5], [-5, 17]],
               norm=LogNorm(), cmap="Greys", cmin=1)
    ax2.scatter(cluster["bp_rp"], abs_g_cl, s=8, color="C3",
                label="Hyades members")
    ax2.set_ylim(17, -5)
    ax2.set_xlabel(r"$G_{BP} - G_{RP}$")
    ax2.set_ylabel(r"$M_G$")
    ax2.set_title("One age, one metallicity: a razor-thin sequence")
    ax2.legend(loc="upper right")

    plot_path = _outdir(output_dir) / "gaia_hyades.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    mean_plx = float(np.mean(cluster["parallax"])) if len(cluster) else float("nan")
    return ArtifactResult(
        status="success",
        files=[str(plot_path)],
        message=(
            f"Selected {int(members.sum()):,} Hyades members by parallax and "
            f"proper motion (mean distance "
            f"{1000.0 / mean_plx:.1f} pc)." if len(cluster) else
            "No Hyades members found in this input (needs the 100 pc sample)."
        ),
        metadata={
            "n_members": int(members.sum()),
            "selection": {"parallax_mas": [19, 24], "pmra_mas_yr": [80, 140],
                          "pmdec_mas_yr": [-60, -10], "l_deg": [160, 200],
                          "b_deg": [-42, -2]},
            "mean_distance_pc": None if not len(cluster) else round(1000.0 / mean_plx, 1),
            "reference": "Babusiaux et al. 2018, A&A 616, A10, Sect. 4",
        },
    )


@validate_call
def plot_white_dwarfs(
    input_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
) -> ArtifactResult:
    """Zoom in on the white dwarf sequence (the paper's Fig. 13).

    Use this tool on the CSV from apply_quality_filters. White dwarfs are
    selected as everything well below the main sequence
    (M_G > 3.25 (BP-RP) + 9.63); within 100 pc that is a nearly complete,
    nearly extinction-free sample of degenerate remnants. At this precision
    the sequence splits into two parallel tracks — hydrogen- and
    helium-atmosphere white dwarfs (DA/DB), a bifurcation first seen
    clearly in exactly this DR2 sample.

    Args:
        input_file: CSV from fetch_gaia_sample or apply_quality_filters.
        output_dir: Directory where the PNG is written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = gaia.load_sample_csv(input_file)
    _require_columns(data, ["parallax", "phot_g_mean_mag", "bp_rp"])
    abs_g = _abs_g(data)
    wd = abs_g > 3.25 * data["bp_rp"] + 9.63
    dwarfs = data[wd]

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(dwarfs["bp_rp"], abs_g[wd], s=3, color="C0", alpha=0.5)
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylim(16.5, 8.5)
    ax.set_xlabel(r"$G_{BP} - G_{RP}$")
    ax.set_ylabel(r"$M_G$")
    ax.set_title(f"White dwarfs within 100 pc — {int(wd.sum()):,} stars")

    plot_path = _outdir(output_dir) / "gaia_white_dwarfs.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ArtifactResult(
        status="success",
        files=[str(plot_path)],
        message=(
            f"Zoomed on {int(wd.sum()):,} white dwarfs "
            "(selected by M_G > 3.25(BP-RP) + 9.63)."
        ),
        metadata={
            "n_white_dwarfs": int(wd.sum()),
            "selection": "abs_g_mag > 3.25 * bp_rp + 9.63",
            "reference": "Babusiaux et al. 2018, A&A 616, A10, Fig. 13",
        },
    )


@validate_call
def plot_luminosity_function(
    input_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
) -> ArtifactResult:
    """The stellar census: how many stars of each luminosity?

    Use this tool on the CSV from apply_quality_filters. It histograms M_G
    for the full sample and overlays the 25 pc sample scaled by the volume
    ratio (64x): where the scaled nearby counts exceed the full sample, the
    100 pc sample is incomplete (the faint end — the survey is
    magnitude-limited in G). The headline result: the most common stars are
    faint M dwarfs, and the Sun (M_G = 4.67) is brighter than the vast
    majority of its neighbours.

    Args:
        input_file: CSV from fetch_gaia_sample or apply_quality_filters.
        output_dir: Directory where the PNG is written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SUN_ABS_G = 4.67

    data = gaia.load_sample_csv(input_file)
    _require_columns(data, ["parallax", "phot_g_mean_mag"])
    abs_g = _abs_g(data)
    near = data["parallax"] >= 40.0  # d < 25 pc
    bins = np.arange(-4, 17.5, 0.5)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(abs_g, bins=bins, histtype="stepfilled", alpha=0.4, color="C0",
            label=f"d < 100 pc ({len(data):,} stars)")
    ax.hist(abs_g[near], bins=bins, histtype="step", color="C3", linewidth=1.8,
            weights=np.full(int(near.sum()), 64.0),
            label=f"d < 25 pc, scaled x64 ({int(near.sum()):,} stars)")
    ax.axvline(SUN_ABS_G, color="0.3", linestyle="--", linewidth=1.2)
    ax.text(SUN_ABS_G + 0.15, ax.get_ylim()[1] * 0.5, "Sun", rotation=90,
            color="0.3", va="center")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_G$")
    ax.set_ylabel("stars per 0.5 mag bin")
    ax.set_title("Luminosity function of the solar neighbourhood")
    ax.legend(loc="upper left", fontsize="small")

    plot_path = _outdir(output_dir) / "gaia_luminosity_function.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fainter = float(np.mean(abs_g > SUN_ABS_G))
    return ArtifactResult(
        status="success",
        files=[str(plot_path)],
        message=(
            f"Luminosity function of {len(data):,} stars: "
            f"{100 * fainter:.0f}% are fainter than the Sun; the faint end "
            "of the 100 pc sample is incomplete where the scaled 25 pc "
            "counts exceed it."
        ),
        metadata={
            "n_stars": len(data),
            "n_within_25pc": int(near.sum()),
            "fraction_fainter_than_sun": round(fainter, 3),
            "sun_abs_g_mag": SUN_ABS_G,
        },
    )
