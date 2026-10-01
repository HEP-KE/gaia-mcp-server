from pathlib import Path
"""Tools tested as plain Python — no MCP, no network, no bundled data needed.

A small synthetic sample stands in for the Gaia archive: one clean star plus
one row that violates each quality filter, so every cut is exercised.
"""

import numpy as np
import pytest

from tools import (
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
from tools import gaia


def _one(directory, base):
    """The single artifact a tool wrote for `base` (names carry an input hash)."""
    matches = sorted(Path(directory).glob(f"{base}_??????.png"))
    assert len(matches) == 1, matches
    return matches[0]


def make_sample():
    """One good star, then one row failing exactly one filter each."""
    good = dict(
        source_id=1, l=180.0, b=-22.0, parallax=50.0, parallax_over_error=100.0,
        phot_g_mean_mag=10.0, bp_rp=1.0, phot_bp_rp_excess_factor=1.1,
        phot_g_mean_flux_over_error=100.0, phot_bp_mean_flux_over_error=50.0,
        phot_rp_mean_flux_over_error=50.0, visibility_periods_used=12,
        astrometric_chi2_al=50.0, astrometric_n_good_obs_al=100,
        pmra=100.0, pmdec=50.0, radial_velocity=np.nan,
        variable=0.0, j_m=8.5, ks_m=8.0,
    )
    rows = [good]
    rows.append({**good, "source_id": 2, "phot_g_mean_flux_over_error": 10.0,
                 "variable": 1.0})
    rows.append({**good, "source_id": 3, "phot_bp_mean_flux_over_error": 5.0,
                 "j_m": np.nan, "ks_m": np.nan})
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
    assert "g.parallax >= 10" in adql
    assert "LEFT OUTER JOIN" in adql          # server-side 2MASS cross-match
    assert "phot_variable_flag" in adql       # becomes the "variable" column
    for column in gaia.COLUMNS:
        if column != "variable":
            assert column in adql


def test_each_filter_removes_its_bad_row(sample_csv, tmp_path):
    result = apply_gaia_quality_filters(sample_csv, str(tmp_path))
    assert result.status == "success"
    assert result.metadata["n_input"] == 5
    assert result.metadata["n_output"] == 1  # only the good star survives
    removed = result.metadata["removed_by_each_filter_alone"]
    assert all(count == 1 for count in removed.values()), removed
    assert set(result.metadata["justifications"]) == set(removed)


def test_filters_can_be_disabled(sample_csv, tmp_path):
    result = apply_gaia_quality_filters(
        sample_csv, str(tmp_path),
        min_phot_g_snr=0.0, min_phot_bprp_snr=0.0,
        apply_excess_factor_cut=False, apply_astrometry_cut=False,
    )
    assert result.metadata["n_output"] == 5


def test_absolute_magnitude_formula(sample_csv, tmp_path):
    result = compute_gaia_absolute_magnitudes(sample_csv, str(tmp_path))
    cmd = np.genfromtxt(result.files[0], delimiter=",", names=True)
    # good star: G=10, parallax=50 mas -> M_G = 10 + 5*log10(50) - 10
    expected = 10.0 + 5 * np.log10(50.0) - 10.0
    assert cmd["abs_g_mag"][0] == pytest.approx(expected, abs=1e-4)


def test_negative_parallax_is_dropped(tmp_path):
    data = make_sample()
    data["parallax"][1] = -2.0  # nonsense: cannot be inverted into a distance
    path = tmp_path / "sample.csv"
    gaia.write_sample_csv(data, path)
    result = compute_gaia_absolute_magnitudes(str(path), str(tmp_path))
    assert result.metadata["n_dropped"] == 1
    assert result.metadata["n_stars"] == 4


def test_plot_cmd_writes_png(sample_csv, tmp_path):
    cmd_result = compute_gaia_absolute_magnitudes(sample_csv, str(tmp_path))
    plot_result = plot_gaia_cmd(cmd_result.files[0], str(tmp_path))
    png = plot_result.files[0]
    assert Path(png).name.startswith("gaia_cmd_hrd_") and png.endswith(".png")
    assert _one(tmp_path, "gaia_cmd_hrd").stat().st_size > 10_000
    # fixture stars sit at 20 pc: not the published 100 pc sample, and few
    # enough to draw as points
    assert plot_result.metadata["sample_radius_pc"] == 20.0
    assert plot_result.metadata["mode"] == "points"
    assert "published_count_fig5c" not in plot_result.metadata
    assert "d < 20 pc" in plot_result.message


def test_distance_shells_counts_are_nested(sample_csv, tmp_path):
    # fixture stars all have parallax = 50 mas (d = 20 pc): inside every shell
    result = compare_distance_shells(sample_csv, str(tmp_path),
                                     distances_pc=[25.0, 100.0])
    counts = result.metadata["star_counts"]
    assert counts["25_pc"] == counts["100_pc"] == 5
    assert _one(tmp_path, "gaia_cmd_shells").stat().st_size > 10_000
    # the 20 pc sample cannot fill 25 or 100 pc shells: flagged, not silent
    assert result.metadata["shells_beyond_sample_pc"] == [25.0, 100.0]
    assert "incomplete" in result.message


def test_distance_shells_rejects_cmd_table(sample_csv, tmp_path):
    cmd_result = compute_gaia_absolute_magnitudes(sample_csv, str(tmp_path))
    with pytest.raises(ValueError, match="phot_g_mean_mag columns"):
        compare_distance_shells(cmd_result.files[0], str(tmp_path))


def test_kinematics_writes_both_figures(sample_csv, tmp_path):
    result = plot_kinematics_cmd(sample_csv, str(tmp_path))
    assert len(result.files) == 2
    assert _one(tmp_path, "gaia_cmd_velocity_slices").exists()
    assert _one(tmp_path, "gaia_cmd_mean_vtan").exists()
    # fixture stars: v_T = 4.74047 * hypot(100, 50) / 50 ~ 10.6 km/s -> all slow
    assert result.metadata["slice_star_counts"] == [5, 0, 0]


def test_variable_stars_are_counted(sample_csv, tmp_path):
    result = plot_variable_stars_cmd(sample_csv, str(tmp_path))
    assert result.metadata["n_variable"] == 1
    assert _one(tmp_path, "gaia_cmd_variables").exists()


def test_infrared_skips_unmatched(sample_csv, tmp_path):
    result = plot_infrared_cmd(sample_csv, str(tmp_path))
    assert result.metadata["n_matched"] == 4  # one fixture row has no 2MASS
    assert _one(tmp_path, "gaia_cmd_infrared").exists()


def test_sky_map_writes_png(sample_csv, tmp_path):
    result = plot_gaia_sky_map(sample_csv, str(tmp_path))
    assert _one(tmp_path, "gaia_sky_map").stat().st_size > 10_000
    assert result.metadata["n_stars"] == 5


def test_extended_tools_reject_cmd_table(sample_csv, tmp_path):
    cmd_result = compute_gaia_absolute_magnitudes(sample_csv, str(tmp_path))
    with pytest.raises(ValueError, match="lacks the column"):
        plot_kinematics_cmd(cmd_result.files[0], str(tmp_path))


def make_population_sample():
    """The 5 filter-test rows plus one Hyades-like member and one white dwarf."""
    base = make_sample()
    hyades = base[0:1].copy()
    hyades["source_id"] = 6
    hyades["parallax"] = 21.3          # 47 pc
    hyades["pmra"], hyades["pmdec"] = 105.0, -28.0
    wd = base[0:1].copy()
    wd["source_id"] = 7
    wd["phot_g_mean_mag"] = 13.5       # parallax 50 mas -> M_G = 15
    wd["bp_rp"] = 0.2                  # 15 > 3.25*0.2 + 9.63: a white dwarf
    return np.concatenate([base, hyades, wd])


@pytest.fixture
def population_csv(tmp_path):
    path = tmp_path / "population.csv"
    gaia.write_sample_csv(make_population_sample(), path)
    return str(path)


def test_hyades_selection(population_csv, tmp_path):
    result = plot_hyades(population_csv, str(tmp_path))
    assert result.metadata["n_members"] == 1
    assert result.metadata["mean_distance_pc"] == pytest.approx(47.0, abs=0.5)
    assert _one(tmp_path, "gaia_hyades").exists()


def test_white_dwarf_selection(population_csv, tmp_path):
    result = plot_white_dwarfs(population_csv, str(tmp_path))
    assert result.metadata["n_white_dwarfs"] == 1
    assert _one(tmp_path, "gaia_white_dwarfs").exists()


def test_luminosity_function(population_csv, tmp_path):
    result = plot_gaia_luminosity_function(population_csv, str(tmp_path))
    # sample radius from the farthest star (Hyades member, 47 pc); inner
    # completeness sphere is R/4 — too few stars there for the overlay
    assert result.metadata["sample_radius_pc"] == 47.0
    assert result.metadata["inner_radius_pc"] == pytest.approx(11.75)
    assert "too few stars" in result.message
    assert 0.0 <= result.metadata["fraction_fainter_than_sun"] <= 1.0
    assert _one(tmp_path, "gaia_luminosity_function").exists()


def test_bundled_fallback_rejects_looser_cuts():
    with pytest.raises(ValueError, match="bundled snapshot"):
        gaia.load_bundled(min_parallax_mas=5.0, min_parallax_snr=10.0)


def test_fetch_bundled_roundtrip(tmp_path):
    if not gaia.BUNDLED_FILE.exists():
        pytest.skip("bundled snapshot not present")
    result = fetch_gaia_sample(str(tmp_path), source="bundled")
    assert result.metadata["source"] == "bundled"
    assert result.metadata["n_rows"] > 200_000


# ---------- usage-driven fixes (Oct 2026 review) ----------

def test_fetch_accepts_nearby_samples(tmp_path):
    """An agent asked for a 5 pc sample (200 mas) and was rejected by an
    le=100 bound; the nearest star is 768 mas."""
    if not gaia.BUNDLED_FILE.exists():
        pytest.skip("bundled snapshot not present")
    result = fetch_gaia_sample(str(tmp_path), min_parallax_mas=200,
                               source="bundled")
    data = gaia.load_sample_csv(result.files[0])
    assert result.metadata["n_rows"] > 0 and np.all(data["parallax"] >= 200)
    assert "bundled snapshot" in result.message


def test_quality_message_is_sample_aware(sample_csv, tmp_path):
    result = apply_gaia_quality_filters(sample_csv, str(tmp_path))
    assert result.metadata["sample_radius_pc"] == 20.0
    assert "published_count_fig5c" not in result.metadata
    assert "applies only to the default 100 pc sample" in result.message


def test_cmd_table_carries_parallax(sample_csv, tmp_path):
    result = compute_gaia_absolute_magnitudes(sample_csv, str(tmp_path))
    cmd = np.genfromtxt(result.files[0], delimiter=",", names=True)
    assert set(cmd.dtype.names) == {"bp_rp", "abs_g_mag", "parallax"}


def test_empty_selection_does_not_crash(sample_csv, tmp_path):
    """All fixture stars are slow: the halo panel is empty (hist2d with a
    LogNorm used to be fragile there)."""
    result = plot_kinematics_cmd(sample_csv, str(tmp_path), halo_min_km_s=1000)
    assert result.metadata["slice_star_counts"][2] == 0


def test_save_pdf(sample_csv, tmp_path):
    result = plot_variable_stars_cmd(sample_csv, str(tmp_path), save_pdf=True)
    assert [f.rsplit(".", 1)[1] for f in result.files] == ["png", "pdf"]


def test_published_sample_reproduced(tmp_path):
    """The bundled 100 pc sample still reproduces Babusiaux+18 Fig. 5c."""
    if not gaia.BUNDLED_FILE.exists():
        pytest.skip("bundled snapshot not present")
    raw = fetch_gaia_sample(str(tmp_path), source="bundled")
    clean = apply_gaia_quality_filters(raw.files[0], str(tmp_path))
    assert clean.metadata["n_output"] == gaia.PUBLISHED_100PC_COUNT
    assert clean.metadata["published_count_fig5c"] == gaia.PUBLISHED_100PC_COUNT


def test_mcp_tool_names():
    import asyncio
    from mcp_server.server import create_server
    names = {t.name for t in asyncio.run(create_server().list_tools())}
    assert {"apply_gaia_quality_filters", "compute_gaia_absolute_magnitudes",
            "plot_gaia_cmd", "plot_gaia_sky_map",
            "plot_gaia_luminosity_function"} <= names
    # dispatch trio is a cross-server contract: names must not change
    assert {"set_dispatch", "get_dispatch", "auth_status"} <= names
    assert not names & {"apply_quality_filters", "plot_cmd", "plot_sky_map"}


def test_different_cuts_do_not_overwrite(sample_csv, tmp_path):
    strict = apply_gaia_quality_filters(sample_csv, str(tmp_path))
    loose = apply_gaia_quality_filters(sample_csv, str(tmp_path),
                                       apply_excess_factor_cut=False)
    assert strict.files[0] != loose.files[0]
    assert Path(strict.files[0]).exists() and Path(loose.files[0]).exists()
    again = apply_gaia_quality_filters(sample_csv, str(tmp_path))
    assert again.files[0] == strict.files[0]  # identical call -> same name
