"""MCP tool functions for the Gaia colour-magnitude diagram.

Plain Python functions — no MCP imports. The type hints, Field constraints,
and docstrings below become the MCP tool schema that agents see.

These are MEASUREMENTS from the Gaia DR2 catalogue (ESA archive), not
simulation or emulator output. Data flows between tools as CSV file paths:
fetch_gaia_sample writes the raw sample, apply_gaia_quality_filters cleans
it, compute_gaia_absolute_magnitudes turns it into a CMD table,
plot_gaia_cmd draws it. Only small numbers (row counts, file
paths) ever pass through the LLM context.

Target figure: Gaia Collaboration, Babusiaux et al. (2018), A&A 616, A10,
Fig. 5c — the Hertzsprung-Russell diagram of the 212,728 stars within 100 pc.
"""

from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, Field, validate_call

from . import gaia
from .plotting import (ABS_G, BP_RP, FIELD_GREY, MUTED, PALETTE, density,
                       empty_note, sample_radius_pc, save, shared_norm, style,
                       use_density, wrap)

SAVE_PDF = Field(description="Also write a vector PDF next to the PNG (for manuscripts).")


def _radius_text(radius_pc) -> str:
    return f"d < {radius_pc:g} pc" if radius_pc else "Gaia DR2 sample"


def _is_published_sample(radius_pc) -> bool:
    """The Babusiaux et al. Fig. 5c count only applies to the 100 pc sample."""
    return radius_pc is not None and 99.0 <= radius_pc <= 101.0


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


def _artifact(output_dir: str, base: str, suffix: str, **inputs) -> Path:
    """<base>_<hash of inputs><suffix>: calls with different samples or cuts
    in one output_dir get different files instead of overwriting each other;
    identical calls reuse the same name."""
    import hashlib
    blob = ",".join(f"{k}={inputs[k]}" for k in sorted(inputs))
    return _outdir(output_dir) / f"{base}_{hashlib.sha1(blob.encode()).hexdigest()[:6]}{suffix}"


def _require_columns(data, names) -> None:
    missing = [n for n in names if n not in (data.dtype.names or ())]
    if missing:
        raise ValueError(
            f"input_file lacks the column(s) {missing}; pass the CSV written "
            "by fetch_gaia_sample or apply_gaia_quality_filters (the "
            "compute_gaia_absolute_magnitudes output keeps only bp_rp, "
            "abs_g_mag and parallax)."
        )


def _abs_g(data) -> "np.ndarray":
    return data["phot_g_mean_mag"] + 5 * np.log10(data["parallax"]) - 10


@validate_call
def fetch_gaia_sample(
    output_dir: Annotated[str, Field(min_length=1)],
    min_parallax_mas: Annotated[float, Field(ge=1.0, le=1000.0, description="Parallax floor in mas = 1000 / (sample radius in pc): 10 -> 100 pc (default, the published sample), 100 -> 10 pc, 200 -> 5 pc. The nearest star (Proxima Cen) is 768 mas.")] = 10.0,
    min_parallax_snr: Annotated[float, Field(ge=0.0, le=100.0)] = 10.0,
    source: Literal["auto", "archive", "bundled"] = "auto",
) -> ArtifactResult:
    """Fetch a Gaia DR2 solar-neighbourhood star sample (measured astrometry
    and photometry from the ESA archive) and write it to a CSV.

    Use this tool first. The default cuts select the 100 pc sample of
    Babusiaux et al. (2018) Fig. 5c: parallax >= 10 mas (distance < 100 pc)
    and parallax_over_error > 10 (distance good to ~10%, which is what makes
    d = 1/parallax a safe estimator here). The query downloads ALL matching
    rows — no TOP truncation, which would NOT be a random subsample.

    Args:
        output_dir: Directory where the CSV is written.
        min_parallax_mas: Parallax floor in mas; 10 mas = a 100 pc sphere,
            100 mas = 10 pc, 200 mas = 5 pc. DR2 contains many spurious
            high-parallax sources, so small nearby samples shrink a lot under
            apply_gaia_quality_filters — that is expected.
        min_parallax_snr: Minimum parallax/parallax_error. Raising it gives
            better distances but preferentially removes faint red stars —
            a biased, not just smaller, sample.
        source: "archive" queries the ESA Gaia archive live (needs network,
            ~1-2 min); "bundled" uses the snapshot shipped in data/;
            "auto" tries the archive and falls back to the snapshot.

    When dispatch is set to an HPC site (set_dispatch tool), the archive
    query runs on a facility compute node and the CSV is fetched back here;
    source="bundled" always runs locally (the snapshot does not ship).
    """
    from mcp_server.dispatch import remote_site, run_kernel  # lazy: server-only

    site = remote_site()
    if site and source != "bundled":
        result = run_kernel(
            "gaia.fetch_sample_to_csv",
            {"min_parallax_mas": min_parallax_mas, "min_parallax_snr": min_parallax_snr},
            pip_deps=["astroquery"],
            duration=900,
        )
        fetched = [f for f in result.get("artifact_files", [])
                   if f.endswith("gaia_sample.csv")]
        if not fetched:
            raise RuntimeError(
                f"The {site} job succeeded ({result['result'].get('n_rows', '?')} rows) "
                "but the sample CSV did not come back as an artifact — check "
                "DISPATCH_ARTIFACT_DIR and Globus Connect Personal, then rerun."
            )
        import shutil

        csv_path = _artifact(output_dir, "gaia_sample", ".csv", plx=min_parallax_mas, snr=min_parallax_snr, source=source)
        shutil.move(fetched[0], csv_path)
        n_rows = int(result["result"]["n_rows"])
        return ArtifactResult(
            status="success",
            files=[str(csv_path)],
            message=(
                f"Fetched {n_rows:,} Gaia DR2 sources with parallax >= "
                f"{min_parallax_mas:g} mas and parallax SNR > {min_parallax_snr:g} "
                f"from the archive, computed on {result.get('host', site)}."
            ),
            metadata={
                "n_rows": n_rows,
                "source": "archive",
                "computed_on": result.get("host", site),
                "adql": gaia.build_adql(min_parallax_mas, min_parallax_snr),
                "columns": list(gaia.COLUMNS),
            },
        )

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

    csv_path = _artifact(output_dir, "gaia_sample", ".csv", plx=min_parallax_mas, snr=min_parallax_snr, source=source)
    gaia.write_sample_csv(data, csv_path)
    return ArtifactResult(
        status="success",
        files=[str(csv_path)],
        message=(
            f"Fetched {len(data):,} Gaia DR2 sources with parallax >= "
            f"{min_parallax_mas:g} mas and parallax SNR > {min_parallax_snr:g} "
            f"from the {'archive' if used == 'archive' else 'bundled snapshot'}{note}."
        ),
        metadata={
            "n_rows": len(data),
            "source": used,
            "adql": gaia.build_adql(min_parallax_mas, min_parallax_snr),
            "columns": list(gaia.COLUMNS),
        },
    )


@validate_call
def apply_gaia_quality_filters(
    input_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
    min_phot_g_snr: Annotated[float, Field(ge=0.0)] = 50.0,
    min_phot_bprp_snr: Annotated[float, Field(ge=0.0)] = 20.0,
    apply_excess_factor_cut: bool = True,
    apply_astrometry_cut: bool = True,
) -> ArtifactResult:
    """Apply the Babusiaux et al. (2018) Gaia DR2 quality cuts to a star sample CSV.

    Gaia-specific photometric/astrometric cuts (flux SNR, BP/RP excess
    factor, unit-weight error) — not a generic data filter.

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
    radius = sample_radius_pc(data["parallax"]) if "parallax" in data.dtype.names else None

    csv_path = _artifact(output_dir, "gaia_sample_clean", ".csv", f=input_file, g=min_phot_g_snr, bprp=min_phot_bprp_snr, excess=apply_excess_factor_cut, astrometry=apply_astrometry_cut)
    gaia.write_sample_csv(clean, csv_path)
    message = f"Quality filters kept {len(clean):,} of {n_input:,} stars"
    if _is_published_sample(radius):
        message += f" (published 100 pc count: {gaia.PUBLISHED_100PC_COUNT:,})."
    else:
        message += (f" ({_radius_text(radius)}; the published 212,728 count "
                    "applies only to the default 100 pc sample).")
    metadata = {
        "n_input": n_input,
        "n_output": len(clean),
        "sample_radius_pc": radius,
        "removed_by_each_filter_alone": removed_alone,
        "justifications": {k: gaia.JUSTIFICATIONS[k] for k in masks},
    }
    if _is_published_sample(radius):
        metadata["published_count_fig5c"] = gaia.PUBLISHED_100PC_COUNT
    return ArtifactResult(status="success", files=[str(csv_path)],
                          message=message, metadata=metadata)


@validate_call
def compute_gaia_absolute_magnitudes(
    input_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
) -> ArtifactResult:
    """Convert apparent G magnitudes to absolute using inverted parallaxes.

    Use this tool after apply_gaia_quality_filters. It computes
    M_G = G + 5 log10(parallax/mas) - 10 for Gaia DR2 stars and writes a
    CMD table (columns: bp_rp, abs_g_mag, parallax). Inverting the parallax is only a safe distance
    estimator because this sample requires parallax SNR > 10: for noisy or
    negative parallaxes 1/parallax is biased or meaningless, and one should
    infer distances properly (e.g. Bailer-Jones et al. 2018). No extinction
    correction is applied — within 100 pc it is negligible.

    Args:
        input_file: CSV written by apply_gaia_quality_filters (or fetch_gaia_sample).
        output_dir: Directory where the CMD CSV is written.
    """
    data = gaia.load_sample_csv(input_file)
    n_input = len(data)
    valid = (data["parallax"] > 0) & ~np.isnan(data["bp_rp"])
    n_dropped = int(n_input - valid.sum())
    data = data[valid]

    abs_g = data["phot_g_mean_mag"] + 5 * np.log10(data["parallax"]) - 10

    csv_path = _artifact(output_dir, "gaia_cmd", ".csv", f=input_file)
    np.savetxt(
        csv_path,
        np.column_stack([data["bp_rp"], abs_g, data["parallax"]]),
        delimiter=",",
        header="bp_rp,abs_g_mag,parallax",
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
def plot_gaia_cmd(
    input_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
    color_min: float = -1.0,
    color_max: float = 5.0,
    mag_bright: float = -5.0,
    mag_faint: float = 17.0,
    n_bins: Annotated[int, Field(ge=50, le=1000)] = 300,
    title: Annotated[str | None, Field(description="Optional title; default states the sample radius and star count.")] = None,
    save_pdf: Annotated[bool, SAVE_PDF] = False,
) -> ArtifactResult:
    """Draw the Gaia DR2 colour-magnitude diagram (observational HRD).

    Use this tool last, on the CSV written by compute_gaia_absolute_magnitudes.
    Large samples are drawn as a log-scaled 2D density of M_G vs BP-RP;
    small ones (< 5000 stars, e.g. a 10 pc sphere) as individual points.
    The magnitude axis is inverted (bright at the top), axes matched by
    default to Babusiaux et al. (2018) Fig. 5c. In the 100 pc diagram you
    should be able to identify the main sequence, the binary sequence just
    above it, the red clump near BP-RP = 1.2, M_G = 0.5, and the white
    dwarf sequence in the lower left.

    Args:
        input_file: CSV written by compute_gaia_absolute_magnitudes.
        output_dir: Directory where the PNG is written.
        color_min: Left edge of the BP-RP axis.
        color_max: Right edge of the BP-RP axis.
        mag_bright: Top of the M_G axis (bright end).
        mag_faint: Bottom of the M_G axis (faint end).
        n_bins: Histogram bins per axis (density mode).
    """
    cmd = np.genfromtxt(Path(input_file).expanduser().resolve(),
                        delimiter=",", names=True)
    color, abs_g = cmd["bp_rp"], cmd["abs_g_mag"]
    radius = (sample_radius_pc(cmd["parallax"])
              if "parallax" in (cmd.dtype.names or ()) else None)
    n_shown = int((np.isfinite(color) & np.isfinite(abs_g)).sum())

    with style() as plt:
        fig, ax = plt.subplots(figsize=(5.6, 6.6), layout="constrained")
        image = density(ax, color, abs_g, x_range=(color_min, color_max),
                        y_range=(mag_bright, mag_faint), bins=n_bins)
        empty_note(ax, n_shown)
        ax.set_xlabel(BP_RP)
        ax.set_ylabel(ABS_G)
        ax.set_title(wrap(title or f"Gaia DR2 HRD, {_radius_text(radius)}: "
                         f"{n_shown:,} stars", 48), loc="left")
        if image is not None:
            fig.colorbar(image, ax=ax, label="stars per bin", pad=0.02)
        files = save(fig, _artifact(output_dir, "gaia_cmd_hrd", ".png", f=input_file, c=(color_min, color_max), m=(mag_bright, mag_faint), n=n_bins, t=title), save_pdf)

    message = f"Plotted the CMD of {n_shown:,} stars ({_radius_text(radius)})"
    metadata = {
        "n_stars": n_shown,
        "sample_radius_pc": radius,
        "mode": "density" if image is not None else "points",
        "axes": {"bp_rp": [color_min, color_max],
                 "abs_g_mag": [mag_faint, mag_bright]},
        "reference": "Babusiaux et al. 2018, A&A 616, A10, Fig. 5c",
    }
    if _is_published_sample(radius) or radius is None:
        message += f"; published Fig. 5c 100 pc count: {gaia.PUBLISHED_100PC_COUNT:,}"
        metadata["published_count_fig5c"] = gaia.PUBLISHED_100PC_COUNT
    return ArtifactResult(status="success", files=files, message=message + ".",
                          metadata=metadata)


@validate_call
def compare_distance_shells(
    input_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
    distances_pc: Annotated[list[float], Field(min_length=2, max_length=4)] = [25.0, 50.0, 100.0],
    save_pdf: Annotated[bool, SAVE_PDF] = False,
) -> ArtifactResult:
    """Draw side-by-side Gaia DR2 HRDs for nested distance shells (full Fig. 5).

    Use this tool on the CSV written by apply_gaia_quality_filters (it needs
    the parallax column, so NOT the compute_gaia_absolute_magnitudes output). The
    default distances reproduce the three panels of Babusiaux et al. (2018)
    Fig. 5: stars within 25, 50, and 100 pc. Nearby shells contain far fewer
    stars but reach fainter absolute magnitudes — the sample is
    volume-limited in parallax yet magnitude-limited in G, so the faint end
    of the diagram is only complete close to the Sun. Shells larger than the
    input sample's own radius cannot add stars and are flagged.

    Args:
        input_file: CSV written by apply_gaia_quality_filters (or
            fetch_gaia_sample) — must still contain the parallax column.
        output_dir: Directory where the PNG is written.
        distances_pc: Shell radii in parsec, small to large. A star is in a
            shell when parallax >= 1000/distance.
    """
    data = gaia.load_sample_csv(input_file)
    if not {"parallax", "phot_g_mean_mag"} <= set(data.dtype.names or ()):
        raise ValueError(
            "input_file lacks the parallax/phot_g_mean_mag columns; pass the "
            "CSV from apply_gaia_quality_filters, not from "
            "compute_gaia_absolute_magnitudes."
        )
    distances = sorted(distances_pc)
    radius = sample_radius_pc(data["parallax"])

    counts = {}
    shells = [data[(data["parallax"] >= 1000.0 / d_pc) & ~np.isnan(data["bp_rp"])]
              for d_pc in distances]
    as_density = use_density(*(len(s) for s in shells))
    norm = (shared_norm([(s["bp_rp"], _abs_g(s)) for s in shells],
                        x_range=(-1, 5), y_range=(-5, 17), bins=250)
            if as_density else None)
    with style() as plt:
        fig, axes = plt.subplots(1, len(distances),
                                 figsize=(3.4 * len(distances) + 0.8, 5.4),
                                 sharex=True, sharey=True, layout="constrained")
        image = None
        for ax, d_pc, shell in zip(np.atleast_1d(axes), distances, shells):
            abs_g = _abs_g(shell)
            im = density(ax, shell["bp_rp"], abs_g, x_range=(-1, 5),
                         y_range=(-5, 17), bins=250, as_density=as_density,
                         norm=norm)
            image = im or image
            empty_note(ax, len(shell))
            ax.set_xlabel(BP_RP)
            ax.set_title(f"d < {d_pc:g} pc\n{len(shell):,} stars")
            counts[f"{d_pc:g}_pc"] = len(shell)
        np.atleast_1d(axes)[0].set_ylabel(ABS_G)
        if image is not None:
            fig.colorbar(image, ax=list(np.atleast_1d(axes)),
                         label="stars per bin", pad=0.02, aspect=30)
        files = save(fig, _artifact(output_dir, "gaia_cmd_shells", ".png", f=input_file, d=distances_pc), save_pdf)

    beyond = [d for d in distances if radius and d > radius * 1.001]
    message = ("Plotted HRDs for "
               + ", ".join(f"d < {d:g} pc ({counts[f'{d:g}_pc']:,} stars)"
                           for d in distances) + ".")
    if beyond:
        message += (f" NOTE: the input sample only reaches {radius:g} pc, so "
                    f"shell(s) {beyond} pc are incomplete — fetch with a "
                    "smaller min_parallax_mas to fill them.")
    return ArtifactResult(
        status="success", files=files, message=message,
        metadata={
            "star_counts": counts,
            "sample_radius_pc": radius,
            "shells_beyond_sample_pc": beyond,
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
    save_pdf: Annotated[bool, SAVE_PDF] = False,
) -> ArtifactResult:
    """Slice the Gaia DR2 HRD by tangential velocity (the paper's Fig. 7).

    Use this tool on the CSV from apply_gaia_quality_filters. The tangential
    velocity v_T = 4.74 * pm[mas/yr] / parallax[mas] km/s needs only Gaia
    astrometry, and slicing on it separates stellar populations by age and
    origin: slow stars are the young thin disc (upper main sequence
    present), fast stars are old (no upper main sequence, subdwarfs sitting
    blueward of the main sequence, halo white dwarfs). Writes TWO figures:
    the three velocity-sliced HRDs, and a map of mean v_T across the CMD.

    Args:
        input_file: CSV from fetch_gaia_sample or apply_gaia_quality_filters
            (needs pmra, pmdec, parallax).
        output_dir: Directory where the PNGs are written.
        slow_max_km_s: Upper v_T bound of the "thin disc" panel.
        mid_range_km_s: (low, high) v_T bounds of the middle panel.
        halo_min_km_s: Lower v_T bound of the "halo" panel.
    """
    data = gaia.load_sample_csv(input_file)
    _require_columns(data, ["pmra", "pmdec", "parallax", "phot_g_mean_mag", "bp_rp"])
    abs_g, color = _abs_g(data), data["bp_rp"]
    v_tan = 4.74047 * np.hypot(data["pmra"], data["pmdec"]) / data["parallax"]

    mid_lo, mid_hi = mid_range_km_s
    slices = [
        (rf"$v_T < {slow_max_km_s:g}$ km/s", "mostly thin disc",
         v_tan < slow_max_km_s),
        (rf"${mid_lo:g} < v_T < {mid_hi:g}$ km/s", "older discs",
         (v_tan > mid_lo) & (v_tan < mid_hi)),
        (rf"$v_T > {halo_min_km_s:g}$ km/s", "halo", v_tan > halo_min_km_s),
    ]
    as_density = use_density(*(int(sel.sum()) for *_, sel in slices))
    norm = (shared_norm([(color[sel], abs_g[sel]) for *_, sel in slices],
                        x_range=(-1, 5), y_range=(-5, 17), bins=200)
            if as_density else None)
    with style() as plt:
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.4), sharex=True,
                                 sharey=True, layout="constrained")
        image = None
        for ax, (cut, population, sel) in zip(axes, slices):
            im = density(ax, color[sel], abs_g[sel], x_range=(-1, 5),
                         y_range=(-5, 17), bins=200, as_density=as_density,
                         norm=norm)
            image = im or image
            empty_note(ax, int(sel.sum()))
            ax.set_xlabel(BP_RP)
            ax.set_title(f"{cut}: {population}\n{int(sel.sum()):,} stars")
        axes[0].set_ylabel(ABS_G)
        if image is not None:
            fig.colorbar(image, ax=list(axes), label="stars per bin",
                         pad=0.02, aspect=30)
        files = save(fig, _artifact(output_dir, "gaia_cmd_velocity_slices", ".png", f=input_file, v=(slow_max_km_s, mid_range_km_s, halo_min_km_s)), save_pdf)

        H_n, xe, ye = np.histogram2d(color, abs_g, bins=200, range=[[-1, 5], [-5, 17]])
        H_v, _, _ = np.histogram2d(color, abs_g, bins=200, range=[[-1, 5], [-5, 17]],
                                   weights=np.nan_to_num(v_tan))
        mean_v = np.where(H_n >= 3, H_v / np.maximum(H_n, 1), np.nan)
        fig, ax = plt.subplots(figsize=(5.8, 6.4), layout="constrained")
        im = ax.pcolormesh(xe, ye, np.ma.masked_invalid(mean_v.T), cmap="magma",
                           vmin=10, vmax=100, rasterized=True)
        ax.set_xlim(-1, 5)
        ax.set_ylim(17, -5)
        empty_note(ax, int(np.isfinite(mean_v).sum()))
        ax.set_xlabel(BP_RP)
        ax.set_ylabel(ABS_G)
        ax.set_title("Mean tangential velocity across the HRD", loc="left")
        fig.colorbar(im, ax=ax, pad=0.02,
                     label=r"mean $v_T$ [km s$^{-1}$] (bins with $\geq 3$ stars)")
        files += save(fig, _artifact(output_dir, "gaia_cmd_mean_vtan", ".png", f=input_file, v=(slow_max_km_s, mid_range_km_s, halo_min_km_s)), save_pdf)

    slice_counts = [int(sel.sum()) for *_, sel in slices]
    return ArtifactResult(
        status="success",
        files=files,
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


def _field_with_highlight(ax, data_color, data_mag, hl_color, hl_mag, label,
                          x_range=(-1, 5), y_range=(-5, 17)):
    """Grey field-star density with one highlighted population on top."""
    density(ax, data_color, data_mag, x_range=x_range, y_range=y_range,
            bins=300, cmap=FIELD_GREY, point_color="0.7")
    ax.scatter(hl_color, hl_mag, s=9, color=PALETTE[1], linewidths=0,
               label=label, zorder=3)


@validate_call
def plot_variable_stars_cmd(
    input_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
    save_pdf: Annotated[bool, SAVE_PDF] = False,
) -> ArtifactResult:
    """Highlight Gaia DR2's flagged variable stars on the HRD (the paper's Fig. 15).

    Use this tool on the CSV from apply_gaia_quality_filters. Within 100 pc the
    flagged variables are almost entirely flaring and spotted M dwarfs on
    the lower main sequence; the bright pulsators that fill this figure in
    the all-sky sample (Cepheids, RR Lyrae) have no representatives this
    close to the Sun.

    Args:
        input_file: CSV from fetch_gaia_sample or apply_gaia_quality_filters
            (needs the "variable" 0/1 column).
        output_dir: Directory where the PNG is written.
    """
    data = gaia.load_sample_csv(input_file)
    _require_columns(data, ["variable", "parallax", "phot_g_mean_mag", "bp_rp"])
    abs_g, color = _abs_g(data), data["bp_rp"]
    variable = data["variable"] > 0.5
    n_var = int(variable.sum())

    with style() as plt:
        fig, ax = plt.subplots(figsize=(5.6, 6.4), layout="constrained")
        _field_with_highlight(ax, color, abs_g, color[variable], abs_g[variable],
                              f"flagged VARIABLE ({n_var:,})")
        ax.set_xlabel(BP_RP)
        ax.set_ylabel(ABS_G)
        ax.set_title(wrap(f"Gaia DR2 variables among {len(data):,} stars", 48),
                     loc="left")
        ax.legend(loc="upper right", markerscale=1.8)
        files = save(fig, _artifact(output_dir, "gaia_cmd_variables", ".png", f=input_file), save_pdf)
    return ArtifactResult(
        status="success",
        files=files,
        message=f"Marked {n_var:,} flagged variables on the HRD of {len(data):,} stars.",
        metadata={
            "n_variable": n_var,
            "n_total": len(data),
            "reference": "Babusiaux et al. 2018, A&A 616, A10, Fig. 15",
        },
    )


@validate_call
def plot_infrared_cmd(
    input_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
    save_pdf: Annotated[bool, SAVE_PDF] = False,
) -> ArtifactResult:
    """Draw the infrared HRD of Gaia DR2 stars via the 2MASS cross-match (Fig. 6).

    Use this tool on the CSV from apply_gaia_quality_filters. The sample carries
    2MASS J and Ks from a server-side cross-match (NaN where unmatched). In
    the infrared the main sequence is less sensitive to metallicity and the
    M dwarfs bunch up; white dwarfs are largely too faint for 2MASS and
    drop out — a photometric completeness lesson in one panel.

    Args:
        input_file: CSV from fetch_gaia_sample or apply_gaia_quality_filters
            (needs j_m, ks_m).
        output_dir: Directory where the PNG is written.
    """
    data = gaia.load_sample_csv(input_file)
    _require_columns(data, ["j_m", "ks_m", "parallax"])
    matched = np.isfinite(data["j_m"]) & np.isfinite(data["ks_m"])
    subset = data[matched]
    abs_ks = subset["ks_m"] + 5 * np.log10(subset["parallax"]) - 10
    j_ks = subset["j_m"] - subset["ks_m"]

    with style() as plt:
        fig, ax = plt.subplots(figsize=(5.6, 6.4), layout="constrained")
        image = density(ax, j_ks, abs_ks, x_range=(-0.4, 1.4), y_range=(-6, 11),
                        bins=250)
        empty_note(ax, int(matched.sum()))
        ax.set_xlabel(r"$J - K_s$")
        ax.set_ylabel(r"$M_{K_s}$")
        ax.set_title(wrap(f"2MASS HRD: {int(matched.sum()):,} of {len(data):,} "
                          "Gaia stars matched", 48), loc="left")
        if image is not None:
            fig.colorbar(image, ax=ax, label="stars per bin", pad=0.02)
        files = save(fig, _artifact(output_dir, "gaia_cmd_infrared", ".png", f=input_file), save_pdf)
    return ArtifactResult(
        status="success",
        files=files,
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
def plot_gaia_sky_map(
    input_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
    save_pdf: Annotated[bool, SAVE_PDF] = False,
) -> ArtifactResult:
    """Map a Gaia DR2 star sample on the sky in galactic coordinates.

    Use this tool on the CSV from apply_gaia_quality_filters. Within 100 pc the
    sky should be nearly isotropic — and almost is: the overdensity at
    l = 180, b = -22 is the Hyades, the nearest open cluster (d = 47 pc),
    and the stark empty patches are regions Gaia's scanning law had visited
    too few times by DR2, emptied entirely by the visibility_periods_used
    cut. Quality filters imprint the survey's geometry on the sample.

    Args:
        input_file: CSV from fetch_gaia_sample or apply_gaia_quality_filters
            (needs l, b).
        output_dir: Directory where the PNG is written.
    """
    data = gaia.load_sample_csv(input_file)
    _require_columns(data, ["l", "b"])

    with style() as plt:
        fig, ax = plt.subplots(figsize=(9.5, 4.9), layout="constrained")
        image = density(ax, data["l"], data["b"], x_range=(0, 360),
                        y_range=(-90, 90), bins=[360, 180], invert_y=False)
        empty_note(ax, len(data))
        ax.set_xlim(360, 0)   # astronomical convention: l increases leftward
        ax.set_xlabel(r"Galactic longitude $l$ [deg]")
        ax.set_ylabel(r"Galactic latitude $b$ [deg]")
        ax.set_xticks(range(0, 361, 60))
        ax.set_yticks(range(-90, 91, 30))
        ax.set_title(f"Gaia DR2 sample on the sky: {len(data):,} stars", loc="left")
        ax.annotate("Hyades", (180, -22), xytext=(140, -64), fontsize=11,
                    color="black", ha="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                              alpha=0.9),
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="black"))
        if image is not None:
            # 1x1 deg (l, b) cells shrink toward the poles: per-bin, not per-deg^2
            fig.colorbar(image, ax=ax, label=r"stars per $1^\circ \times 1^\circ$ bin",
                         pad=0.015)
        files = save(fig, _artifact(output_dir, "gaia_sky_map", ".png", f=input_file), save_pdf)
    return ArtifactResult(
        status="success",
        files=files,
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
    save_pdf: Annotated[bool, SAVE_PDF] = False,
) -> ArtifactResult:
    """Extract the Hyades cluster from the Gaia DR2 field and draw its HRD.

    Use this tool on the CSV from apply_gaia_quality_filters. The Hyades is the
    nearest open cluster (d = 47 pc) and its members share one proper
    motion and one parallax — a box in (parallax, pmra, pmdec, l, b) pulls
    them cleanly out of the field with no colour information used at all.
    Because the cluster is a single age (~700 Myr) and single metallicity,
    its main sequence is razor thin compared to the field's spread; its
    unresolved binaries stand out above it (the paper studies 46 clusters
    this way, Sect. 4). Needs a sample reaching >= 53 pc (the default
    100 pc one).

    Args:
        input_file: CSV from fetch_gaia_sample or apply_gaia_quality_filters
            (needs parallax, pmra, pmdec, l, b).
        output_dir: Directory where the PNG is written.
    """
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
    n_mem = int(members.sum())

    with style() as plt:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.4),
                                       layout="constrained",
                                       gridspec_kw={"width_ratios": [1.15, 1]})
        near = (np.abs(data["l"] - 180) < 40) & (data["b"] > -45) & (data["b"] < 5)
        ax1.scatter(data["l"][near], data["b"][near], s=1.5, color="0.78",
                    linewidths=0, rasterized=True, label="field")
        ax1.scatter(cluster["l"], cluster["b"], s=9, color=PALETTE[1],
                    linewidths=0, label=f"Hyades members ({n_mem:,})")
        ax1.set_xlim(220, 140)
        ax1.set_ylim(-45, 5)
        ax1.set_xlabel(r"Galactic longitude $l$ [deg]")
        ax1.set_ylabel(r"Galactic latitude $b$ [deg]")
        ax1.set_title("Selected by parallax and proper motion only", loc="left")
        ax1.legend(loc="lower left", markerscale=3, frameon=True,
                   framealpha=0.92, edgecolor="none")

        _field_with_highlight(ax2, data["bp_rp"], _abs_g(data),
                              cluster["bp_rp"], _abs_g(cluster), "Hyades members")
        ax2.set_xlabel(BP_RP)
        ax2.set_ylabel(ABS_G)
        ax2.set_title("One age, one metallicity: a thin sequence", loc="left")
        ax2.legend(loc="upper right", markerscale=1.8)
        files = save(fig, _artifact(output_dir, "gaia_hyades", ".png", f=input_file), save_pdf)

    mean_plx = float(np.mean(cluster["parallax"])) if len(cluster) else float("nan")
    return ArtifactResult(
        status="success",
        files=files,
        message=(
            f"Selected {n_mem:,} Hyades members by parallax and "
            f"proper motion (mean distance "
            f"{1000.0 / mean_plx:.1f} pc)." if len(cluster) else
            "No Hyades members found in this input (needs a sample reaching "
            "~53 pc, e.g. the default 100 pc one)."
        ),
        metadata={
            "n_members": n_mem,
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
    save_pdf: Annotated[bool, SAVE_PDF] = False,
) -> ArtifactResult:
    """Zoom in on the Gaia DR2 white dwarf sequence (the paper's Fig. 13).

    Use this tool on the CSV from apply_gaia_quality_filters. White dwarfs are
    selected as everything well below the main sequence
    (M_G > 3.25 (BP-RP) + 9.63); within 100 pc that is a nearly complete,
    nearly extinction-free sample of degenerate remnants. At this precision
    the sequence splits into two parallel tracks — hydrogen- and
    helium-atmosphere white dwarfs (DA/DB), a bifurcation first seen
    clearly in exactly this DR2 sample.

    Args:
        input_file: CSV from fetch_gaia_sample or apply_gaia_quality_filters.
        output_dir: Directory where the PNG is written.
    """
    data = gaia.load_sample_csv(input_file)
    _require_columns(data, ["parallax", "phot_g_mean_mag", "bp_rp"])
    abs_g = _abs_g(data)
    wd = abs_g > 3.25 * data["bp_rp"] + 9.63
    dwarfs = data[wd]
    radius = sample_radius_pc(data["parallax"])

    with style() as plt:
        fig, ax = plt.subplots(figsize=(5.8, 5.6), layout="constrained")
        ax.scatter(dwarfs["bp_rp"], abs_g[wd], s=3 if wd.sum() > 2000 else 10,
                   color=PALETTE[0], alpha=0.5, linewidths=0,
                   rasterized=wd.sum() > 1000)
        ax.set_xlim(-0.7, 1.7)
        ax.set_ylim(16.5, 8.5)
        empty_note(ax, int(wd.sum()))
        ax.set_xlabel(BP_RP)
        ax.set_ylabel(ABS_G)
        ax.set_title(wrap(f"Gaia DR2 white dwarfs, {_radius_text(radius)}: "
                          f"{int(wd.sum()):,} stars", 48), loc="left")
        files = save(fig, _artifact(output_dir, "gaia_white_dwarfs", ".png", f=input_file), save_pdf)
    return ArtifactResult(
        status="success",
        files=files,
        message=(
            f"Zoomed on {int(wd.sum()):,} white dwarfs "
            "(selected by M_G > 3.25(BP-RP) + 9.63)."
        ),
        metadata={
            "n_white_dwarfs": int(wd.sum()),
            "sample_radius_pc": radius,
            "selection": "abs_g_mag > 3.25 * bp_rp + 9.63",
            "reference": "Babusiaux et al. 2018, A&A 616, A10, Fig. 13",
        },
    )


@validate_call
def plot_gaia_luminosity_function(
    input_file: Annotated[str, Field(min_length=1)],
    output_dir: Annotated[str, Field(min_length=1)],
    save_pdf: Annotated[bool, SAVE_PDF] = False,
) -> ArtifactResult:
    """Stellar luminosity function of a Gaia DR2 solar-neighbourhood sample.

    How many stars of each luminosity? Use this tool on the CSV from
    apply_gaia_quality_filters. It histograms M_G for the full sample (radius
    R = 1000 / min parallax) and overlays the inner R/4 sphere scaled by the
    volume ratio (64x): where the scaled inner counts exceed the full
    sample, the outer sample is incomplete (the faint end — the survey is
    magnitude-limited in G). For the default 100 pc sample that is the
    25 pc vs 100 pc comparison. The headline result: the most common stars
    are faint M dwarfs, and the Sun (M_G = 4.67) is brighter than the vast
    majority of its neighbours. (A stellar census, not a galaxy luminosity
    function.)

    Args:
        input_file: CSV from fetch_gaia_sample or apply_gaia_quality_filters.
        output_dir: Directory where the PNG is written.
    """
    SUN_ABS_G = 4.67

    data = gaia.load_sample_csv(input_file)
    _require_columns(data, ["parallax", "phot_g_mean_mag"])
    abs_g = _abs_g(data)
    radius = sample_radius_pc(data["parallax"]) or 100.0
    inner = radius / 4.0
    near = data["parallax"] >= 1000.0 / inner
    show_inner = int(near.sum()) >= 10   # fewer: the x64 overlay is pure noise
    bins = np.arange(-4, 17.5, 0.5)

    with style() as plt:
        fig, ax = plt.subplots(figsize=(6.8, 4.6), layout="constrained")
        ax.hist(abs_g, bins=bins, histtype="stepfilled", alpha=0.35,
                color=PALETTE[0], edgecolor=PALETTE[0], linewidth=1.2,
                label=f"d < {radius:g} pc ({len(data):,} stars)")
        if show_inner:
            ax.hist(abs_g[near], bins=bins, histtype="step", color=PALETTE[1],
                    linewidth=1.8, linestyle="--",
                    weights=np.full(int(near.sum()), 64.0),
                    label=rf"d < {inner:g} pc, $\times$64 ({int(near.sum()):,} stars)")
        ax.axvline(SUN_ABS_G, color=MUTED, linestyle=":", linewidth=1.3)
        ax.set_yscale("log")
        ax.text(SUN_ABS_G + 0.2, 0.96, "Sun", transform=ax.get_xaxis_transform(),
                color=MUTED, va="top")
        ax.set_xlabel(ABS_G)
        ax.set_ylabel("stars per 0.5 mag bin")
        ax.set_title("Luminosity function of the solar neighbourhood", loc="left")
        ax.legend(loc="upper left")
        files = save(fig, _artifact(output_dir, "gaia_luminosity_function", ".png", f=input_file),
                     save_pdf)

    fainter = float(np.mean(abs_g > SUN_ABS_G)) if len(data) else float("nan")
    completeness = (
        f"the faint end is incomplete where the scaled d < {inner:g} pc "
        "counts exceed it." if show_inner else
        f"too few stars within {inner:g} pc ({int(near.sum())}) for the "
        "completeness overlay — use a larger sample (e.g. the default 100 pc)."
    )
    return ArtifactResult(
        status="success",
        files=files,
        message=(
            f"Luminosity function of {len(data):,} stars (d < {radius:g} pc): "
            f"{100 * fainter:.0f}% are fainter than the Sun; {completeness}"
        ),
        metadata={
            "n_stars": len(data),
            "sample_radius_pc": radius,
            "inner_radius_pc": inner,
            "n_within_inner": int(near.sum()),
            "fraction_fainter_than_sun": round(fainter, 3),
            "sun_abs_g_mag": SUN_ABS_G,
        },
    )
