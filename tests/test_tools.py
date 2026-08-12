"""Tools tested as plain Python — no MCP, no network, no bundled data needed.

A small synthetic sample stands in for the Gaia archive: one clean star plus
one row that violates each quality filter, so every cut is exercised.
"""

import numpy as np
import pytest

from tools import (
    apply_quality_filters,
    compute_absolute_magnitudes,
    fetch_gaia_sample,
    plot_cmd,
)
from tools import gaia


def make_sample():
    """One good star, then one row failing exactly one filter each."""
    good = dict(
        source_id=1, parallax=50.0, parallax_over_error=100.0,
        phot_g_mean_mag=10.0, bp_rp=1.0, phot_bp_rp_excess_factor=1.1,
        phot_g_mean_flux_over_error=100.0, phot_bp_mean_flux_over_error=50.0,
        phot_rp_mean_flux_over_error=50.0, visibility_periods_used=12,
        astrometric_chi2_al=50.0, astrometric_n_good_obs_al=100,
    )
    rows = [good]
    rows.append({**good, "source_id": 2, "phot_g_mean_flux_over_error": 10.0})
    rows.append({**good, "source_id": 3, "phot_bp_mean_flux_over_error": 5.0})
    rows.append({**good, "source_id": 4, "phot_bp_rp_excess_factor": 2.0})
    rows.append({**good, "source_id": 5, "visibility_periods_used": 5})
    data = np.zeros(len(rows), dtype=[(name, "f8") for name in gaia.COLUMNS])
    for i, row in enumerate(rows):
        for name in gaia.COLUMNS:
            data[i][name] = row[name]
    return data


@pytest.fixture
def sample_csv(tmp_path):
    path = tmp_path / "gaia_sample.csv"
    gaia.write_sample_csv(make_sample(), path)
    return str(path)


def test_adql_has_no_top_truncation():
    adql = gaia.build_adql(10.0, 10.0)
    assert "TOP" not in adql.upper()
    assert "parallax >= 10" in adql
    for column in gaia.COLUMNS:
        assert column in adql


def test_each_filter_removes_its_bad_row(sample_csv, tmp_path):
    result = apply_quality_filters(sample_csv, str(tmp_path))
    assert result.status == "success"
    assert result.metadata["n_input"] == 5
    assert result.metadata["n_output"] == 1  # only the good star survives
    removed = result.metadata["removed_by_each_filter_alone"]
    assert all(count == 1 for count in removed.values()), removed
    assert set(result.metadata["justifications"]) == set(removed)


def test_filters_can_be_disabled(sample_csv, tmp_path):
    result = apply_quality_filters(
        sample_csv, str(tmp_path),
        min_phot_g_snr=0.0, min_phot_bprp_snr=0.0,
        apply_excess_factor_cut=False, apply_astrometry_cut=False,
    )
    assert result.metadata["n_output"] == 5


def test_absolute_magnitude_formula(sample_csv, tmp_path):
    result = compute_absolute_magnitudes(sample_csv, str(tmp_path))
    cmd = np.genfromtxt(result.files[0], delimiter=",", names=True)
    # good star: G=10, parallax=50 mas -> M_G = 10 + 5*log10(50) - 10
    expected = 10.0 + 5 * np.log10(50.0) - 10.0
    assert cmd["abs_g_mag"][0] == pytest.approx(expected, abs=1e-4)


def test_negative_parallax_is_dropped(tmp_path):
    data = make_sample()
    data["parallax"][1] = -2.0  # nonsense: cannot be inverted into a distance
    path = tmp_path / "sample.csv"
    gaia.write_sample_csv(data, path)
    result = compute_absolute_magnitudes(str(path), str(tmp_path))
    assert result.metadata["n_dropped"] == 1
    assert result.metadata["n_stars"] == 4


def test_plot_cmd_writes_png(sample_csv, tmp_path):
    cmd_result = compute_absolute_magnitudes(sample_csv, str(tmp_path))
    plot_result = plot_cmd(cmd_result.files[0], str(tmp_path))
    png = plot_result.files[0]
    assert png.endswith("gaia_cmd_hrd.png")
    assert (tmp_path / "gaia_cmd_hrd.png").stat().st_size > 10_000
    assert plot_result.metadata["published_count_fig5c"] == 212_728


def test_bundled_fallback_rejects_looser_cuts():
    with pytest.raises(ValueError, match="bundled snapshot"):
        gaia.load_bundled(min_parallax_mas=5.0, min_parallax_snr=10.0)


def test_fetch_bundled_roundtrip(tmp_path):
    if not gaia.BUNDLED_FILE.exists():
        pytest.skip("bundled snapshot not present")
    result = fetch_gaia_sample(str(tmp_path), source="bundled")
    assert result.metadata["source"] == "bundled"
    assert result.metadata["n_rows"] > 200_000
