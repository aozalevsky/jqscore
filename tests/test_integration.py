"""Integration / functional tests for the full Q-score pipeline.

Tests in TestComputeQscoresSynthetic use synthetic data (no external files).
Tests in TestEndToEnd390 use in-repo small test data from MapQ (100 atoms, sigma=0.6).
Tests in TestEndToEnd390Sigma04 use the same structure/map with sigma=0.4 (EMDB default).
"""

import tempfile

import numpy as np
import pytest

from jqscore.jqscore import (
    AtomRecord,
    compute_qscores,
    residue_qscores,
    chain_qscores,
    run,
    write_qscore_cif,
)
from jqscore.compare import compare_qscores

from .conftest import (
    SMALL_STRUCTURE,
    SMALL_MAP,
    SMALL_MAPQ_REF,
    SMALL_S04_REF,
    XZAK_STRUCTURE,
    XZAK_MAP,
    XZAK_MAPQ_REF,
)


# ── Synthetic Q-score computation ────────────────────────────────────────


class TestComputeQscoresSynthetic:
    """Test compute_qscores on a tiny synthetic map + atoms (no external data)."""

    @staticmethod
    def _make_gaussian_map(centre, sigma_map=1.5, shape=(20, 20, 20), voxel=1.0):
        """Create a 3D Gaussian blob map."""
        data = np.zeros(shape, dtype=np.float32)
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        voxel_size = np.array([voxel, voxel, voxel], dtype=np.float64)
        for iz in range(shape[0]):
            for iy in range(shape[1]):
                for ix in range(shape[2]):
                    dx = ix * voxel - centre[0]
                    dy = iy * voxel - centre[1]
                    dz = iz * voxel - centre[2]
                    r2 = dx * dx + dy * dy + dz * dz
                    data[iz, iy, ix] = np.exp(-r2 / (2.0 * sigma_map**2))
        return data, origin, voxel_size

    def test_single_atom_at_peak(self):
        """An atom at the centre of a Gaussian blob should get high Q-score."""
        centre = [10.0, 10.0, 10.0]
        data, origin, voxel_size = self._make_gaussian_map(centre)
        atoms = [
            AtomRecord("A", 1, "", "ALA", "CA", "C", *centre),
        ]
        result = compute_qscores(
            atoms, data, origin, voxel_size,
            sigma=0.4, mapq_compat=False,
        )
        q = result.atom_qscores[0]
        assert not np.isnan(q), "Q-score should not be NaN for atom at density peak"
        assert q > 0.5, f"Atom at density peak should have high Q-score, got {q:.4f}"

    def test_atom_outside_density_gets_low_q(self):
        """An atom far from any density should get low or zero Q-score."""
        centre = [10.0, 10.0, 10.0]
        data, origin, voxel_size = self._make_gaussian_map(centre)
        atoms = [
            AtomRecord("A", 1, "", "ALA", "CA", "C", 1.0, 1.0, 1.0),  # far from peak
        ]
        result = compute_qscores(
            atoms, data, origin, voxel_size,
            sigma=0.4, mapq_compat=False,
        )
        q = result.atom_qscores[0]
        assert q < 0.5, f"Atom far from density should have low Q-score, got {q:.4f}"

    def test_two_atoms_well_fit(self):
        """Two atoms each at a Gaussian peak should both get positive Q-scores."""
        shape = (30, 30, 30)
        data = np.zeros(shape, dtype=np.float32)
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        voxel_size = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        centres = [[10.0, 15.0, 15.0], [20.0, 15.0, 15.0]]
        for c in centres:
            for iz in range(shape[0]):
                for iy in range(shape[1]):
                    for ix in range(shape[2]):
                        r2 = (ix - c[0])**2 + (iy - c[1])**2 + (iz - c[2])**2
                        data[iz, iy, ix] += np.exp(-r2 / (2.0 * 1.5**2))

        atoms = [
            AtomRecord("A", 1, "", "ALA", "CA", "C", *centres[0]),
            AtomRecord("A", 2, "", "GLY", "CA", "C", *centres[1]),
        ]
        result = compute_qscores(
            atoms, data, origin, voxel_size,
            sigma=0.4, mapq_compat=False,
        )
        assert result.atom_qscores[0] > 0.3
        assert result.atom_qscores[1] > 0.3

    def test_hydrogen_gets_heavy_atom_qscore(self):
        """Hydrogen should inherit the nearest heavy atom's Q-score."""
        centre = [10.0, 10.0, 10.0]
        data, origin, voxel_size = self._make_gaussian_map(centre)
        atoms = [
            AtomRecord("A", 1, "", "ALA", "CA", "C", *centre),
            AtomRecord("A", 1, "", "ALA", "HA", "H", 10.5, 10.0, 10.0),  # 0.5 Å away
        ]
        result = compute_qscores(
            atoms, data, origin, voxel_size,
            sigma=0.4, mapq_compat=False,
        )
        assert not np.isnan(result.atom_qscores[1]), "H should get a Q-score from heavy atom"
        assert result.atom_qscores[0] == result.atom_qscores[1]

    def test_result_fields(self):
        """QScoreResult should have correct structure."""
        centre = [10.0, 10.0, 10.0]
        data, origin, voxel_size = self._make_gaussian_map(centre)
        atoms = [AtomRecord("A", 1, "", "ALA", "CA", "C", *centre)]
        result = compute_qscores(
            atoms, data, origin, voxel_size,
            sigma=0.4, mapq_compat=False,
        )
        assert len(result.atom_records) == 1
        assert len(result.atom_qscores) == 1
        assert result.sigma == 0.4
        assert isinstance(result.min_d, float)
        assert isinstance(result.max_d, float)

    def test_fast_mode_runs(self):
        """use_retry=False mode should complete without error."""
        centre = [10.0, 10.0, 10.0]
        data, origin, voxel_size = self._make_gaussian_map(centre)
        atoms = [AtomRecord("A", 1, "", "ALA", "CA", "C", *centre)]
        result = compute_qscores(
            atoms, data, origin, voxel_size,
            sigma=0.4, use_retry=False, mapq_compat=False,
        )
        assert not np.isnan(result.atom_qscores[0])

    def test_no_heavy_atoms_warns(self):
        """All-hydrogen input should warn and return NaN Q-scores."""
        centre = [10.0, 10.0, 10.0]
        data, origin, voxel_size = self._make_gaussian_map(centre)
        atoms = [AtomRecord("A", 1, "", "ALA", "HA", "H", *centre)]
        with pytest.warns(UserWarning, match="No non-hydrogen"):
            result = compute_qscores(
                atoms, data, origin, voxel_size,
                sigma=0.4, mapq_compat=False,
            )
        assert np.isnan(result.atom_qscores[0])

    def test_batch_size_consistency(self):
        """Results should be identical regardless of batch_size."""
        centre = [10.0, 10.0, 10.0]
        data, origin, voxel_size = self._make_gaussian_map(centre)
        atoms = [
            AtomRecord("A", i, "", "ALA", "CA", "C", 10.0 + i * 0.5, 10.0, 10.0)
            for i in range(5)
        ]
        r_single = compute_qscores(
            atoms, data, origin, voxel_size,
            sigma=0.4, batch_size=None, mapq_compat=False,
        )
        r_batched = compute_qscores(
            atoms, data, origin, voxel_size,
            sigma=0.4, batch_size=2, mapq_compat=False,
        )
        np.testing.assert_allclose(
            r_single.atom_qscores, r_batched.atom_qscores, atol=1e-5,
        )


# ── End-to-end on 390_7cec_A100 (in-repo, 100 atoms, sigma=0.6) ────────


class TestEndToEnd390:
    """Full pipeline on small MapQ test case, compared against MapQ reference.

    Data from the MapQ repository: 100 atoms of 7cec chain A, sigma=0.6.
    This is the case where MapQ's MaskMapResize double-interpolation causes
    known discrepancies; mapq_compat=True should bring them within tolerance.
    """

    def test_run_produces_valid_result(self):
        result = run(str(SMALL_STRUCTURE), str(SMALL_MAP), sigma=0.6, mapq_compat=True)
        assert len(result.atom_records) == 100
        assert len(result.atom_qscores) == 100
        n_valid = np.isfinite(result.atom_qscores).sum()
        assert n_valid > 50

    def test_qscores_match_mapq_reference(self):
        """Per-atom Q-scores with mapq_compat should be close to MapQ reference.

        Tolerances are wider than for 9XZK because:
        - sigma=0.6 (larger Gaussian → more sensitivity to map resampling)
        - MapQ reference has only 3 decimal places (±0.0005 rounding)
        - 100-atom subset: boundary atoms have truncated neighbour lists
        """
        result = run(str(SMALL_STRUCTURE), str(SMALL_MAP), sigma=0.6, mapq_compat=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = write_qscore_cif(result, str(SMALL_STRUCTURE), str(SMALL_MAP), output_dir=tmpdir)
            cmp = compare_qscores(str(SMALL_MAPQ_REF), out_path)

        assert abs(cmp.diff_mean) < 0.02, f"Mean diff too large: {cmp.diff_mean}"
        assert abs(cmp.diff_max) < 0.10, f"Max diff too large: {cmp.diff_max}"
        assert abs(cmp.diff_min) < 0.10, f"Min diff too large: {cmp.diff_min}"
        assert cmp.pearson_r > 0.98, f"Pearson r too low: {cmp.pearson_r}"

    def test_mapq_compat_vs_direct_both_reasonable(self):
        """Both mapq_compat and direct mode should produce reasonable Q-scores."""
        result_compat = run(str(SMALL_STRUCTURE), str(SMALL_MAP), sigma=0.6, mapq_compat=True)
        result_direct = run(str(SMALL_STRUCTURE), str(SMALL_MAP), sigma=0.6, mapq_compat=False)

        valid = np.isfinite(result_compat.atom_qscores) & np.isfinite(result_direct.atom_qscores)
        r = float(np.corrcoef(result_compat.atom_qscores[valid],
                               result_direct.atom_qscores[valid])[0, 1])
        # The two modes should be highly correlated
        assert r > 0.95, f"mapq_compat vs direct correlation too low: {r}"
        # Mean difference should be small
        diff = np.abs(result_compat.atom_qscores[valid] - result_direct.atom_qscores[valid])
        assert diff.mean() < 0.05, f"Mean abs diff too large: {diff.mean()}"

    def test_retry_vs_fast_mode(self):
        """Fast mode should produce correlated results with retry mode."""
        result_retry = run(str(SMALL_STRUCTURE), str(SMALL_MAP), sigma=0.6,
                           use_retry=True, mapq_compat=False)
        result_fast = run(str(SMALL_STRUCTURE), str(SMALL_MAP), sigma=0.6,
                          use_retry=False, mapq_compat=False)

        valid = np.isfinite(result_retry.atom_qscores) & np.isfinite(result_fast.atom_qscores)
        r = float(np.corrcoef(result_retry.atom_qscores[valid],
                               result_fast.atom_qscores[valid])[0, 1])
        assert r > 0.90, f"Retry vs fast mode correlation too low: {r}"

    def test_residue_and_chain_aggregation(self):
        result = run(str(SMALL_STRUCTURE), str(SMALL_MAP), sigma=0.6, mapq_compat=True)

        rq = residue_qscores(result)
        assert len(rq) >= 3, "Should have multiple residues"
        for key, d in rq.items():
            if not np.isnan(d["all"]):
                assert -1.0 <= d["all"] <= 1.0

        cq = chain_qscores(result)
        assert "A" in cq
        assert -1.0 <= cq["A"] <= 1.0

    def test_per_residue_qscores_match_mapq(self):
        """Check per-residue averages against known MapQ values from _All.txt."""
        result = run(str(SMALL_STRUCTURE), str(SMALL_MAP), sigma=0.6, mapq_compat=True)
        rq = residue_qscores(result)

        # Known MapQ per-residue Q-scores from 390_7cec_A100__Q__*_All.txt
        # (chain A, first few residues, sigma=0.6)
        mapq_residue_qs = {
            ("A", 1, "", "PHE"): 0.322624,
            ("A", 2, "", "ASN"): 0.531169,
            ("A", 3, "", "LEU"): 0.411691,
            ("A", 4, "", "ASP"): 0.509571,
            ("A", 5, "", "THR"): 0.566151,
        }

        for key, mapq_q in mapq_residue_qs.items():
            assert key in rq, f"Missing residue {key}"
            our_q = rq[key]["all"]
            assert abs(our_q - mapq_q) < 0.03, (
                f"Residue {key}: ours={our_q:.4f}, MapQ={mapq_q:.4f}, "
                f"diff={our_q - mapq_q:.4f}"
            )


# ── End-to-end on 390_7cec_A100 with sigma=0.4 (EMDB default) ────────


class TestEndToEnd390Sigma04:
    """Full pipeline with sigma=0.4 (the EMDB convention for most maps).

    Uses a jqscore golden reference for regression testing.
    Sigma=0.6 tests above validate against MapQ; since the algorithm is
    identical regardless of sigma, this ensures sigma=0.4 produces stable results.
    """

    def test_run_produces_valid_result(self):
        result = run(str(SMALL_STRUCTURE), str(SMALL_MAP), sigma=0.4, mapq_compat=True)
        assert len(result.atom_records) == 100
        assert len(result.atom_qscores) == 100
        n_valid = np.isfinite(result.atom_qscores).sum()
        assert n_valid > 50

    def test_qscores_match_golden_reference(self):
        """Per-atom Q-scores should match the golden reference exactly (regression)."""
        result = run(str(SMALL_STRUCTURE), str(SMALL_MAP), sigma=0.4, mapq_compat=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = write_qscore_cif(result, str(SMALL_STRUCTURE), str(SMALL_MAP), output_dir=tmpdir)
            cmp = compare_qscores(str(SMALL_S04_REF), out_path)

        assert abs(cmp.diff_mean) < 0.001, f"Mean diff too large: {cmp.diff_mean}"
        assert abs(cmp.diff_max) < 0.005, f"Max diff too large: {cmp.diff_max}"
        assert cmp.pearson_r > 0.9999, f"Pearson r too low: {cmp.pearson_r}"

    def test_sigma04_vs_sigma06_different(self):
        """Sigma=0.4 and sigma=0.6 should produce different but correlated Q-scores."""
        result_04 = run(str(SMALL_STRUCTURE), str(SMALL_MAP), sigma=0.4, mapq_compat=False)
        result_06 = run(str(SMALL_STRUCTURE), str(SMALL_MAP), sigma=0.6, mapq_compat=False)

        valid = np.isfinite(result_04.atom_qscores) & np.isfinite(result_06.atom_qscores)
        r = float(np.corrcoef(result_04.atom_qscores[valid],
                               result_06.atom_qscores[valid])[0, 1])
        assert r > 0.90, f"sigma=0.4 vs 0.6 correlation too low: {r}"
        # They should NOT be identical
        diff = np.abs(result_04.atom_qscores[valid] - result_06.atom_qscores[valid])
        assert diff.mean() > 0.01, "sigma=0.4 and 0.6 should produce different Q-scores"

    def test_mapq_compat_vs_direct_sigma04(self):
        """Both modes should produce correlated results at sigma=0.4."""
        result_compat = run(str(SMALL_STRUCTURE), str(SMALL_MAP), sigma=0.4, mapq_compat=True)
        result_direct = run(str(SMALL_STRUCTURE), str(SMALL_MAP), sigma=0.4, mapq_compat=False)

        valid = np.isfinite(result_compat.atom_qscores) & np.isfinite(result_direct.atom_qscores)
        r = float(np.corrcoef(result_compat.atom_qscores[valid],
                               result_direct.atom_qscores[valid])[0, 1])
        assert r > 0.95, f"mapq_compat vs direct correlation too low: {r}"

    def test_residue_qscores_sigma04(self):
        """Per-residue Q-scores should be reasonable at sigma=0.4."""
        result = run(str(SMALL_STRUCTURE), str(SMALL_MAP), sigma=0.4, mapq_compat=True)
        rq = residue_qscores(result)

        assert len(rq) >= 3, "Should have multiple residues"
        for key, d in rq.items():
            if not np.isnan(d["all"]):
                assert -1.0 <= d["all"] <= 1.0


# ── End-to-end on 9XZK subset with sigma=0.4 (MapQ reference) ────────


class TestEndToEnd9XZKSigma04:
    """Full pipeline on 9XZK chain C (15 residues) with sigma=0.4, vs MapQ reference.

    Data: 234 atoms from 9XZK chain C (residues 305-319), trimmed map (~365KB),
    MapQ reference generated with MapQ 1.9.12 on the full 9XZK structure.
    Boundary residues (305, 319) have wider tolerance due to truncated neighbour
    lists in the subset vs full structure.
    """

    def test_run_produces_valid_result(self):
        result = run(str(XZAK_STRUCTURE), str(XZAK_MAP), sigma=0.4, mapq_compat=True)
        assert len(result.atom_records) == 234
        assert len(result.atom_qscores) == 234
        n_valid = np.isfinite(result.atom_qscores).sum()
        assert n_valid > 100

    def test_qscores_match_mapq_reference(self):
        """Per-atom Q-scores with mapq_compat should be close to MapQ reference."""
        result = run(str(XZAK_STRUCTURE), str(XZAK_MAP), sigma=0.4, mapq_compat=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = write_qscore_cif(result, str(XZAK_STRUCTURE), str(XZAK_MAP), output_dir=tmpdir)
            cmp = compare_qscores(str(XZAK_MAPQ_REF), out_path)

        assert abs(cmp.diff_mean) < 0.01, f"Mean diff too large: {cmp.diff_mean}"
        assert abs(cmp.diff_max) < 0.10, f"Max diff too large: {cmp.diff_max}"
        assert abs(cmp.diff_min) < 0.05, f"Min diff too large: {cmp.diff_min}"
        assert cmp.pearson_r > 0.99, f"Pearson r too low: {cmp.pearson_r}"

    def test_mapq_compat_vs_direct(self):
        """Both modes should produce correlated results at sigma=0.4."""
        result_compat = run(str(XZAK_STRUCTURE), str(XZAK_MAP), sigma=0.4, mapq_compat=True)
        result_direct = run(str(XZAK_STRUCTURE), str(XZAK_MAP), sigma=0.4, mapq_compat=False)

        valid = np.isfinite(result_compat.atom_qscores) & np.isfinite(result_direct.atom_qscores)
        r = float(np.corrcoef(result_compat.atom_qscores[valid],
                               result_direct.atom_qscores[valid])[0, 1])
        assert r > 0.95, f"mapq_compat vs direct correlation too low: {r}"

    def test_residue_and_chain_aggregation(self):
        result = run(str(XZAK_STRUCTURE), str(XZAK_MAP), sigma=0.4, mapq_compat=True)

        rq = residue_qscores(result)
        assert len(rq) >= 10, "Should have multiple residues"
        for key, d in rq.items():
            if not np.isnan(d["all"]):
                assert -1.0 <= d["all"] <= 1.0

        cq = chain_qscores(result)
        assert "C" in cq
        assert -1.0 <= cq["C"] <= 1.0


# ── Retry mode against a float64 reference ──────────────────────────────


def _reference_qscores(atoms, data, origin, voxel_size, sigma=0.4, num_pts=8,
                       max_retries=50, to_rad=2.0, d_rad=0.1):
    """Straightforward float64 numpy Q-scores of the non-H atoms (mapq_compat=False)."""
    from scipy.spatial import cKDTree
    from jqscore.jqscore import _sphere_directions, min_max_d

    min_d, max_d = min_max_d(data)
    A, B = max_d - min_d, min_d
    nz, ny, nx = data.shape

    def density(pts):
        g = (pts - origin) / voxel_size
        i = np.floor(g).astype(int)
        f = g - i
        inside = ((i >= 0) & (i <= [nx - 2, ny - 2, nz - 2])).all(axis=1)
        i = np.where(inside[:, None], i, 0)
        out = np.zeros(len(pts))
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    w = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                         * (f[:, 2] if dz else 1 - f[:, 2]))
                    out += w * data[i[:, 2] + dz, i[:, 1] + dy, i[:, 0] + dx]
        return np.where(inside, out, 0.0)

    xyz = np.array([[a.x, a.y, a.z] for a in atoms if a.element != "H"],
                   dtype=np.float32).astype(np.float64)
    tree = cKDTree(xyz)
    radii = np.arange(d_rad, to_rad + 1e-6, d_rad, dtype=np.float32).astype(np.float64)
    q = []
    for x in xyz:
        nbrs = xyz[tree.query_ball_point(x, 2 * to_rad)]
        d = [density(x[None])[0]] * num_pts
        g = [A + B] * num_pts
        for r in radii:
            for k in range(max_retries):
                pts = x + _sphere_directions(num_pts + 2 * k) * r
                dist_sq = ((pts[:, None] - nbrs[None]) ** 2).sum(-1).min(axis=1)
                clear = dist_sq > (0.9 * r) ** 2
                if clear.sum() >= num_pts:
                    d += list(density(pts[clear]))
                    g += [A * np.exp(-0.5 * (r / sigma) ** 2) + B] * int(clear.sum())
                    break
        d, g = np.array(d), np.array(g)
        q.append(np.corrcoef(d, g)[0, 1] if d.std() > 0 and g.std() > 0 else 0.0)
    return np.array(q)


@pytest.fixture(scope="module")
def xzak_case():
    from jqscore.jqscore import read_cif, read_mrc
    atoms = read_cif(str(XZAK_STRUCTURE))
    data, origin, voxel_size = read_mrc(str(XZAK_MAP))
    return atoms, data, origin, voxel_size


class TestRetryModeReference:
    """Retry mode (attempts checked in rounds) gives the retry loop's Q-scores."""

    @staticmethod
    def _heavy(result):
        return np.array([q for a, q in zip(result.atom_records, result.atom_qscores)
                         if a.element != "H"])

    @pytest.mark.parametrize("max_retries", [50, 3, 1])
    def test_matches_float64_reference(self, xzak_case, max_retries):
        atoms, data, origin, voxel_size = xzak_case
        result = compute_qscores(atoms, data, origin, voxel_size, sigma=0.4,
                                 mapq_compat=False, max_retries=max_retries)
        expected = _reference_qscores(atoms, data.astype(np.float64), origin, voxel_size,
                                      max_retries=max_retries)
        np.testing.assert_allclose(self._heavy(result), expected, atol=1e-4)

    def test_translation_invariant(self, xzak_case):
        """Moving model and map by thousands of Å changes nothing (float32 precision)."""
        atoms, data, origin, voxel_size = xzak_case
        shift = np.array([3011.0, 2477.0, 3893.0])
        moved = [a._replace(x=a.x + shift[0], y=a.y + shift[1], z=a.z + shift[2]) for a in atoms]
        here = compute_qscores(atoms, data, origin, voxel_size, mapq_compat=False)
        there = compute_qscores(moved, data, origin + shift, voxel_size, mapq_compat=False)
        np.testing.assert_allclose(there.atom_qscores, here.atom_qscores, atol=1e-4)

    @pytest.mark.parametrize("batch_size", [32768, 64])
    def test_small_retry_tiles(self, xzak_case, monkeypatch, batch_size):
        """A batch with pending pairs spanning many small tiles gives the same Q-scores."""
        import jqscore.jqscore as jq
        atoms, data, origin, voxel_size = xzak_case
        whole = compute_qscores(atoms, data, origin, voxel_size, mapq_compat=False)
        monkeypatch.setattr(jq, "_RETRY_TILE_SIZES", (16, 8, 4))
        monkeypatch.setattr(jq, "_compiled_cache_retry", {})
        tiled = compute_qscores(atoms, data, origin, voxel_size, mapq_compat=False,
                                batch_size=batch_size)
        # Same points; kernels of other tile sizes may fuse differently, so
        # float32 sums may round differently
        np.testing.assert_allclose(tiled.atom_qscores, whole.atom_qscores, atol=1e-6)
