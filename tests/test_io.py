"""Tests for I/O functions: read_cif, read_mrc, read_qscores, compare_qscores, write_qscore_cif."""

import tempfile

import numpy as np
import pytest

from jqscore.jqscore import AtomRecord, QScoreResult, read_cif, read_mrc, write_qscore_cif
from jqscore.compare import read_qscores, compare_qscores

from .conftest import (
    SMALL_STRUCTURE,
    SMALL_MAP,
    SMALL_MAPQ_REF,
    ORIG_CIF,
    JAX_CIF,
    has_docs_cifs,
)


# ── read_cif (in-repo 390_7cec_A100) ────────────────────────────────────


class TestReadCif:
    def test_returns_list_of_atom_records(self):
        atoms = read_cif(str(SMALL_STRUCTURE))
        assert isinstance(atoms, list)
        assert len(atoms) > 0
        assert isinstance(atoms[0], AtomRecord)

    def test_atom_count(self):
        """390_7cec_A100 should have exactly 100 atoms."""
        atoms = read_cif(str(SMALL_STRUCTURE))
        assert len(atoms) == 100

    def test_atom_fields_populated(self):
        atoms = read_cif(str(SMALL_STRUCTURE))
        a = atoms[0]
        assert a.chain_id != ""
        assert a.res_name != ""
        assert a.atom_name != ""
        assert a.element != ""
        assert isinstance(a.x, float)
        assert isinstance(a.y, float)
        assert isinstance(a.z, float)

    def test_has_heavy_atoms(self):
        atoms = read_cif(str(SMALL_STRUCTURE))
        elements = {a.element for a in atoms}
        assert "C" in elements
        assert "N" in elements

    def test_single_chain(self):
        atoms = read_cif(str(SMALL_STRUCTURE))
        chains = {a.chain_id for a in atoms}
        assert len(chains) == 1
        assert "A" in chains


# ── read_mrc (in-repo small map) ────────────────────────────────────────


class TestReadMrc:
    def test_returns_tuple(self):
        data, origin, voxel_size = read_mrc(str(SMALL_MAP))
        assert isinstance(data, np.ndarray)
        assert data.ndim == 3
        assert data.dtype == np.float32
        assert origin.shape == (3,)
        assert voxel_size.shape == (3,)

    def test_positive_voxel_size(self):
        _, _, voxel_size = read_mrc(str(SMALL_MAP))
        assert (voxel_size > 0).all()

    def test_data_not_all_zero(self):
        data, _, _ = read_mrc(str(SMALL_MAP))
        assert data.max() > 0


# ── read_qscores (MapQ reference output, in-repo) ───────────────────────


class TestReadQscores:
    def test_read_mapq_reference(self):
        struct, qscores = read_qscores(str(SMALL_MAPQ_REF))
        assert len(qscores) == 100
        n_valid = np.isfinite(qscores).sum()
        assert n_valid > 50

    def test_qscore_values_in_range(self):
        _, qscores = read_qscores(str(SMALL_MAPQ_REF))
        valid = qscores[np.isfinite(qscores)]
        assert valid.min() >= -1.0
        assert valid.max() <= 1.0


# ── compare_qscores (docs/ CIFs: orig vs jax 9XZK) ─────────────────────


@has_docs_cifs
class TestCompareQscores:
    def test_compare_orig_vs_jax(self):
        result = compare_qscores(str(ORIG_CIF), str(JAX_CIF))
        assert result.n_valid > 100
        # Per-atom max diff should be < 0.02 (float32 precision)
        assert abs(result.diff_max) < 0.02
        assert abs(result.diff_min) < 0.02
        # Mean diff should be near zero
        assert abs(result.diff_mean) < 0.001
        # Pearson r should be very close to 1
        assert result.pearson_r > 0.9999

    def test_self_comparison_is_perfect(self):
        result = compare_qscores(str(ORIG_CIF), str(ORIG_CIF))
        assert result.diff_max == pytest.approx(0.0)
        assert result.diff_min == pytest.approx(0.0)
        assert result.pearson_r == pytest.approx(1.0)


# ── write_qscore_cif roundtrip ──────────────────────────────────────────


class TestWriteQscoreCif:
    def test_roundtrip(self):
        """Write a CIF with synthetic Q-scores and read them back."""
        atoms = read_cif(str(SMALL_STRUCTURE))
        n = len(atoms)
        fake_q = np.random.default_rng(42).uniform(0.0, 1.0, size=n)
        # Leave some as NaN (like H atoms without assignment)
        fake_q[::10] = np.nan
        result = QScoreResult(atoms, fake_q, 0.0, 1.0, 0.4)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = write_qscore_cif(result, str(SMALL_STRUCTURE), "test.map", output_dir=tmpdir)
            assert out_path.endswith(".cif")

            struct, qscores_back = read_qscores(out_path)
            assert len(qscores_back) == n

            # Check non-NaN values match within write precision (6 decimal places)
            valid = np.isfinite(fake_q)
            np.testing.assert_allclose(
                qscores_back[valid], fake_q[valid], atol=1e-5,
            )
            # NaN positions should still be NaN
            assert np.isnan(qscores_back[~valid]).all()
