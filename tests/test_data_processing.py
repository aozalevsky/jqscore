"""Unit tests for data processing helpers: neighbour lists, H assignment, aggregation."""

import numpy as np
import pytest

from jqscore.jqscore import (
    AtomRecord,
    QScoreResult,
    _build_neighbour_lists,
    _assign_hydrogen_qscores,
    _resample_mapq_compat,
    residue_qscores,
    chain_qscores,
)


# ── _build_neighbour_lists ───────────────────────────────────────────────


class TestBuildNeighbourLists:
    def test_single_atom(self):
        coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        nbr_c, nbr_v = _build_neighbour_lists(coords, radius=5.0)
        assert nbr_c.shape[0] == 1
        assert nbr_v.shape[0] == 1
        # The atom is its own neighbour
        assert nbr_v[0, 0] == True

    def test_two_close_atoms(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        nbr_c, nbr_v = _build_neighbour_lists(coords, radius=2.0)
        # Each atom should see the other as a neighbour
        assert nbr_c.shape == (2, 2, 3)
        assert nbr_v.shape == (2, 2)
        assert nbr_v.all()

    def test_two_distant_atoms(self):
        coords = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float32)
        nbr_c, nbr_v = _build_neighbour_lists(coords, radius=2.0)
        # Each atom should only see itself
        assert nbr_v[0].sum() == 1
        assert nbr_v[1].sum() == 1

    def test_padding_shape(self):
        """All atoms padded to the same K (max neighbour count)."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],  # isolated
        ], dtype=np.float32)
        nbr_c, nbr_v = _build_neighbour_lists(coords, radius=1.5)
        N, K, _ = nbr_c.shape
        assert N == 4
        # The middle atom (idx=1) sees all three close atoms → K ≥ 3
        assert K >= 3
        # Isolated atom should have padding
        assert nbr_v[3].sum() == 1

    def test_dtypes(self):
        coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        nbr_c, nbr_v = _build_neighbour_lists(coords, radius=1.0)
        assert nbr_c.dtype == np.float32
        assert nbr_v.dtype == np.bool_


# ── _assign_hydrogen_qscores ────────────────────────────────────────────


def _make_atom(chain="A", res_seq=1, res_name="ALA", atom_name="CA", element="C",
               x=0.0, y=0.0, z=0.0):
    return AtomRecord(chain, res_seq, "", res_name, atom_name, element, x, y, z)


class TestAssignHydrogenQscores:
    def test_h_gets_nearest_heavy_qscore(self):
        atoms = [
            _make_atom(atom_name="CA", element="C", x=0.0, y=0.0, z=0.0),
            _make_atom(atom_name="N", element="N", x=1.5, y=0.0, z=0.0),
            _make_atom(atom_name="HA", element="H", x=0.5, y=0.0, z=0.0),  # closer to CA
        ]
        coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float32)
        non_h_mask = np.array([True, True, False])
        non_h_indices = np.array([0, 1])
        q_scores = np.array([0.8, 0.6, np.nan])

        _assign_hydrogen_qscores(atoms, coords, non_h_mask, non_h_indices, q_scores)
        assert q_scores[2] == pytest.approx(0.8)  # gets CA's Q-score

    def test_h_too_far_stays_nan(self):
        atoms = [
            _make_atom(atom_name="CA", element="C", x=0.0, y=0.0, z=0.0),
            _make_atom(atom_name="HA", element="H", x=5.0, y=0.0, z=0.0),  # too far
        ]
        coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float32)
        non_h_mask = np.array([True, False])
        non_h_indices = np.array([0])
        q_scores = np.array([0.8, np.nan])

        _assign_hydrogen_qscores(atoms, coords, non_h_mask, non_h_indices, q_scores)
        assert np.isnan(q_scores[1])

    def test_h_uses_same_residue_only(self):
        atoms = [
            _make_atom(res_seq=1, atom_name="CA", element="C", x=0.0, y=0.0, z=0.0),
            _make_atom(res_seq=2, atom_name="CA", element="C", x=0.5, y=0.0, z=0.0),
            _make_atom(res_seq=1, atom_name="HA", element="H", x=0.3, y=0.0, z=0.0),
        ]
        coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float32)
        non_h_mask = np.array([True, True, False])
        non_h_indices = np.array([0, 1])
        q_scores = np.array([0.8, 0.3, np.nan])

        _assign_hydrogen_qscores(atoms, coords, non_h_mask, non_h_indices, q_scores)
        # H at res_seq=1 should get res_seq=1 CA's score (0.8), not res_seq=2 (0.3)
        assert q_scores[2] == pytest.approx(0.8)

    def test_no_hydrogens(self):
        atoms = [
            _make_atom(atom_name="CA", element="C"),
            _make_atom(atom_name="N", element="N", x=1.0),
        ]
        coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float32)
        non_h_mask = np.array([True, True])
        non_h_indices = np.array([0, 1])
        q_scores = np.array([0.8, 0.6])

        _assign_hydrogen_qscores(atoms, coords, non_h_mask, non_h_indices, q_scores)
        np.testing.assert_array_equal(q_scores, [0.8, 0.6])


# ── residue_qscores ─────────────────────────────────────────────────────


class TestResidueQscores:
    def _make_result(self, atoms, qscores):
        return QScoreResult(atoms, np.array(qscores), 0.0, 1.0, 0.4)

    def test_basic_aggregation(self):
        atoms = [
            _make_atom(res_seq=1, atom_name="N", element="N"),
            _make_atom(res_seq=1, atom_name="CA", element="C"),
            _make_atom(res_seq=1, atom_name="CB", element="C"),
        ]
        result = self._make_result(atoms, [0.8, 0.6, 0.4])
        rq = residue_qscores(result)
        key = ("A", 1, "", "ALA")
        assert key in rq
        # N and CA are backbone, CB is sidechain
        assert rq[key]["backbone"] == pytest.approx(0.7)  # (0.8+0.6)/2
        assert rq[key]["sidechain"] == pytest.approx(0.4)
        assert rq[key]["all"] == pytest.approx(0.6)  # (0.8+0.6+0.4)/3

    def test_nan_atoms_excluded(self):
        atoms = [
            _make_atom(res_seq=1, atom_name="CA", element="C"),
            _make_atom(res_seq=1, atom_name="CB", element="C"),
        ]
        result = self._make_result(atoms, [0.8, np.nan])
        rq = residue_qscores(result)
        key = ("A", 1, "", "ALA")
        assert rq[key]["all"] == pytest.approx(0.8)

    def test_multiple_residues(self):
        atoms = [
            _make_atom(res_seq=1, atom_name="CA"),
            _make_atom(res_seq=2, atom_name="CA", res_name="GLY"),
        ]
        result = self._make_result(atoms, [0.8, 0.4])
        rq = residue_qscores(result)
        assert len(rq) == 2


# ── chain_qscores ───────────────────────────────────────────────────────


class TestChainQscores:
    def test_basic(self):
        atoms = [
            _make_atom(chain="A", atom_name="CA"),
            _make_atom(chain="A", atom_name="CB"),
            _make_atom(chain="B", atom_name="CA"),
        ]
        result = QScoreResult(atoms, np.array([0.8, 0.6, 0.4]), 0.0, 1.0, 0.4)
        cq = chain_qscores(result)
        assert cq["A"] == pytest.approx(0.7)
        assert cq["B"] == pytest.approx(0.4)

    def test_nan_excluded(self):
        atoms = [
            _make_atom(chain="A", atom_name="CA"),
            _make_atom(chain="A", atom_name="HA", element="H"),
        ]
        result = QScoreResult(atoms, np.array([0.8, np.nan]), 0.0, 1.0, 0.4)
        cq = chain_qscores(result)
        assert cq["A"] == pytest.approx(0.8)


# ── _resample_mapq_compat ───────────────────────────────────────────────


class TestResampleMapqCompat:
    def test_zero_offset_is_identity(self):
        """When origin is already grid-aligned, resampling should be near-identity."""
        data = np.random.default_rng(0).standard_normal((10, 10, 10)).astype(np.float32)
        # Origin at exact grid multiple → offset = 0
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        voxel_size = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        resampled, new_origin = _resample_mapq_compat(data, origin, voxel_size)
        np.testing.assert_allclose(new_origin, origin, atol=1e-6)
        # Interior should be nearly identical (boundaries may differ due to cval=0)
        np.testing.assert_allclose(
            resampled[2:-2, 2:-2, 2:-2],
            data[2:-2, 2:-2, 2:-2],
            atol=1e-5,
        )

    def test_output_shape_preserved(self):
        data = np.ones((8, 8, 8), dtype=np.float32)
        origin = np.array([0.3, 0.7, 0.1], dtype=np.float64)
        voxel_size = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        resampled, new_origin = _resample_mapq_compat(data, origin, voxel_size)
        assert resampled.shape == data.shape
        assert resampled.dtype == np.float32
