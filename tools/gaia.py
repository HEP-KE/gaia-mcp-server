"""Gaia DR2 data layer: the archive query, the bundled fallback, and the
Babusiaux et al. (2018) quality cuts.

Everything here is plain Python/NumPy — no MCP imports. The MCP tools in
cmd_tools.py call these functions.

Reference: Gaia Collaboration, Babusiaux et al., "Gaia Data Release 2:
Observational Hertzsprung-Russell diagrams", A&A 616, A10 (2018),
arXiv:1804.09378. The selection below is their Sect. 2.1 (Eqs. 1-3); the
100 pc sample of Fig. 5c (parallax >= 10 mas) contains 212,728 stars.
"""

import gzip
import io
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BUNDLED_FILE = DATA_DIR / "gaia_dr2_100pc.csv.gz"

# The published star count of Babusiaux et al. (2018) Fig. 5c, for comparison.
PUBLISHED_100PC_COUNT = 212_728

# Columns the pipeline needs. Names are exactly the Gaia archive column names
# (https://gea.esac.esa.int/archive/) so they can be looked up in the docs.
COLUMNS = [
    "source_id",
    "parallax",
    "parallax_over_error",
    "phot_g_mean_mag",
    "bp_rp",
    "phot_bp_rp_excess_factor",
    "phot_g_mean_flux_over_error",
    "phot_bp_mean_flux_over_error",
    "phot_rp_mean_flux_over_error",
    "visibility_periods_used",
    "astrometric_chi2_al",
    "astrometric_n_good_obs_al",
]

# The bundled file was fetched with exactly these cuts; a fallback query
# cannot go below them.
BUNDLED_MIN_PARALLAX_MAS = 10.0
BUNDLED_MIN_PARALLAX_SNR = 10.0


def build_adql(min_parallax_mas: float, min_parallax_snr: float) -> str:
    """The full-sample ADQL query (no TOP: a truncated result is NOT a random
    sample — the archive returns rows in an arbitrary, correlated order; use
    the random_index column if you ever need an unbiased subsample)."""
    return (
        f"SELECT {', '.join(COLUMNS)} "
        "FROM gaiadr2.gaia_source "
        f"WHERE parallax >= {min_parallax_mas} "
        f"AND parallax_over_error > {min_parallax_snr}"
    )


def query_archive(min_parallax_mas: float, min_parallax_snr: float) -> "np.ndarray":
    """Run the ADQL query against the ESA Gaia archive (asynchronous job —
    synchronous TAP queries time out on full-table scans). Returns a NumPy
    structured array with COLUMNS as field names."""
    from astroquery.gaia import Gaia  # imported here: bundled mode works without it

    job = Gaia.launch_job_async(build_adql(min_parallax_mas, min_parallax_snr))
    table = job.get_results()
    buffer = io.StringIO()
    table.write(buffer, format="ascii.csv")
    buffer.seek(0)
    return np.genfromtxt(buffer, delimiter=",", names=True)


def load_bundled(min_parallax_mas: float, min_parallax_snr: float) -> "np.ndarray":
    """Load the bundled snapshot and re-apply the requested cuts.

    The snapshot was fetched with parallax >= 10 mas and parallax_over_error
    > 10, so looser cuts than that cannot be honoured offline.
    """
    if min_parallax_mas < BUNDLED_MIN_PARALLAX_MAS or min_parallax_snr < BUNDLED_MIN_PARALLAX_SNR:
        raise ValueError(
            "The bundled snapshot only contains parallax >= "
            f"{BUNDLED_MIN_PARALLAX_MAS} mas and parallax_over_error > "
            f"{BUNDLED_MIN_PARALLAX_SNR}. Looser cuts need the live archive "
            '(source="archive").'
        )
    with gzip.open(BUNDLED_FILE, "rt", encoding="utf-8") as f:
        data = np.genfromtxt(f, delimiter=",", names=True)
    keep = (data["parallax"] >= min_parallax_mas) & (
        data["parallax_over_error"] > min_parallax_snr
    )
    return data[keep]


def load_sample_csv(path: str) -> "np.ndarray":
    """Read a CSV written by the tools back into a structured array."""
    return np.genfromtxt(Path(path).expanduser().resolve(), delimiter=",", names=True)


def write_sample_csv(data: "np.ndarray", path: Path) -> None:
    header = ",".join(data.dtype.names)
    fmt = ["%d" if name == "source_id" else "%.10g" for name in data.dtype.names]
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt=fmt)


# --------------------------------------------------------------------------
# Quality filters — Babusiaux et al. (2018), Sect. 2.1.
# Each entry: name -> (mask function, one-line justification).
# A True in the mask means KEEP the star.
# --------------------------------------------------------------------------

def _g_snr(data, threshold):
    return data["phot_g_mean_flux_over_error"] > threshold


def _bprp_snr(data, threshold):
    return (data["phot_bp_mean_flux_over_error"] > threshold) & (
        data["phot_rp_mean_flux_over_error"] > threshold
    )


def _excess_factor(data):
    color = data["bp_rp"]
    excess = data["phot_bp_rp_excess_factor"]
    return (excess > 1.0 + 0.015 * color**2) & (excess < 1.3 + 0.06 * color**2)


def _astrometry(data):
    with np.errstate(invalid="ignore", divide="ignore"):
        unit_weight_error = np.sqrt(
            data["astrometric_chi2_al"] / (data["astrometric_n_good_obs_al"] - 5)
        )
        limit = 1.2 * np.maximum(
            1.0, np.exp(-0.2 * (data["phot_g_mean_mag"] - 19.5))
        )
    return (data["visibility_periods_used"] > 8) & (unit_weight_error < limit)


JUSTIFICATIONS = {
    "phot_g_snr": "G-band flux SNR > 50: keeps G photometry good to ~2%.",
    "phot_bprp_snr": "BP and RP flux SNR > 20: keeps the colour good to ~5%.",
    "excess_factor": (
        "1.0 + 0.015(BP-RP)^2 < E < 1.3 + 0.06(BP-RP)^2: removes sources whose "
        "BP/RP flux is contaminated by neighbours or background (blends); "
        "without it a spurious plume crosses the diagram."
    ),
    "astrometry": (
        "visibility_periods_used > 8 and unit weight error "
        "sqrt(chi2/(N-5)) < 1.2 max(1, exp(-0.2(G-19.5))): removes poor "
        "astrometric solutions (the DR2-era precursor of the RUWE < 1.4 cut)."
    ),
}
