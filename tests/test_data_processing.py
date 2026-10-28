"""Unit tests for data processing helpers: neighbour lookup, H assignment, aggregation."""

import numpy as np
import pytest

from jqscore.jqscore import (
    AtomRecord,
    QScoreResult,
    _FAR,
    _GpuCellList,
    _NeighbourIndex,
    _neighbour_groups,
    _assign_hydrogen_qscores,
    _residue_codes,
    _resample_mapq_compat,
    residue_qscores,
    chain_qscores,
)


# ── _NeighbourIndex ──────────────────────────────────────────────────────


def _neighbours(coords, radius, k=None):
    """Neighbour indices of every atom and the mask of real (non-padding) slots."""
    coords = np.asarray(coords, dtype=np.float32)
    index = _NeighbourIndex(coords, radius)
    if k is None:
        k = int(index.counts(coords).max())
    idx = index.query(coords, k)
    return idx, idx < len(coords)


class TestNeighbourIndex:
    def test_single_atom(self):
        idx, valid = _neighbours([[0.0, 0.0, 0.0]], radius=5.0)
        assert idx.shape[0] == 1
        # The atom is its own neighbour
        assert valid[0, 0] and idx[0, 0] == 0

    def test_two_close_atoms(self):
        idx, valid = _neighbours([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], radius=2.0)
        # Each atom should see the other as a neighbour
        assert idx.shape == (2, 2)
        assert valid.all()
        assert sorted(idx[0]) == [0, 1]

    def test_two_distant_atoms(self):
        idx, valid = _neighbours([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], radius=2.0)
        # Each atom should only see itself
        assert valid[0].sum() == 1
        assert valid[1].sum() == 1

    def test_padding_shape(self):
        """All atoms padded to the same K (max neighbour count)."""
        coords = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],  # isolated
        ]
        idx, valid = _neighbours(coords, radius=1.5)
        N, K = idx.shape
        assert N == 4
        # The middle atom (idx=1) sees all three close atoms → K ≥ 3
        assert K >= 3
        # Isolated atom should have padding, pointing one past the last atom
        assert valid[3].sum() == 1
        assert (idx[3][~valid[3]] == 4).all()

    def test_extra_slots_are_padding(self):
        idx, valid = _neighbours([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], radius=2.0, k=8)
        assert idx.shape == (2, 8)
        assert valid.sum(axis=1).tolist() == [2, 2]

    def test_counts(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float32)
        assert _NeighbourIndex(coords, 2.0).counts(coords).tolist() == [2, 2, 1]


class TestNeighbourGroups:
    def test_one_group_for_uniform_counts(self):
        counts = np.array([10, 12, 13, 11] * 1000)
        groups = _neighbour_groups(counts)
        assert len(groups) == 1
        members, k = groups[0]
        assert k == 16 and len(members) == len(counts)

    def test_outliers_get_own_group(self):
        counts = np.array([12] * 100000 + [200])
        (regular, k_regular), (many, k_many) = _neighbour_groups(counts)
        assert k_regular == 16 and k_many == 200
        assert many.tolist() == [100000]
        assert len(regular) == 100000

    def test_atom_at_radius_included(self):
        """The cutoff is inclusive (as scipy's query_ball_point)."""
        _, valid = _neighbours([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], radius=4.0)
        assert valid.all()

    def test_dtype(self):
        idx, _ = _neighbours([[0.0, 0.0, 0.0]], radius=1.0)
        assert idx.dtype == np.int32


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

        _assign_hydrogen_qscores(_residue_codes(atoms), coords, non_h_mask, non_h_indices, q_scores)
        assert q_scores[2] == pytest.approx(0.8)  # gets CA's Q-score

    def test_thiol_h_gets_sulfur_qscore(self):
        """S-H (1.34 Å) is longer than C/N/O-H bonds."""
        atoms = [
            _make_atom(res_name="CYS", atom_name="CB", element="C", x=0.0, y=0.0, z=0.0),
            _make_atom(res_name="CYS", atom_name="SG", element="S", x=1.81, y=0.0, z=0.0),
            _make_atom(res_name="CYS", atom_name="HG", element="H", x=2.25, y=1.27, z=0.0),
        ]
        coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float32)
        non_h_mask = np.array([True, True, False])
        non_h_indices = np.array([0, 1])
        q_scores = np.array([0.8, 0.6, np.nan])

        _assign_hydrogen_qscores(_residue_codes(atoms), coords, non_h_mask, non_h_indices, q_scores)
        assert q_scores[2] == pytest.approx(0.6)

    def test_h_too_far_stays_nan(self):
        atoms = [
            _make_atom(atom_name="CA", element="C", x=0.0, y=0.0, z=0.0),
            _make_atom(atom_name="HA", element="H", x=5.0, y=0.0, z=0.0),  # too far
        ]
        coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float32)
        non_h_mask = np.array([True, False])
        non_h_indices = np.array([0])
        q_scores = np.array([0.8, np.nan])

        _assign_hydrogen_qscores(_residue_codes(atoms), coords, non_h_mask, non_h_indices, q_scores)
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

        _assign_hydrogen_qscores(_residue_codes(atoms), coords, non_h_mask, non_h_indices, q_scores)
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

        _assign_hydrogen_qscores(_residue_codes(atoms), coords, non_h_mask, non_h_indices, q_scores)
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

    @pytest.mark.parametrize("origin", [
        [0.3, 0.7, 0.1],       # offsets of both signs
        [-0.45, 0.2, -0.05],
        [0.0, 0.5, 0.0],       # 0.5 rounds to even: offset 0 or ±0.5
    ])
    def test_matches_scipy_shift(self, origin):
        """Same result as scipy.ndimage.shift(order=1, mode='constant'), edges included."""
        from scipy.ndimage import shift

        data = np.random.default_rng(1).standard_normal((7, 9, 11)).astype(np.float32)
        origin = np.array(origin, dtype=np.float64)
        voxel_size = np.array([1.0, 1.25, 0.8], dtype=np.float64)

        resampled, new_origin = _resample_mapq_compat(data, origin, voxel_size)

        offset = (new_origin - origin.astype(np.float32).astype(np.float64)) / voxel_size.astype(np.float32)
        expected = shift(data, -offset[::-1], order=1, prefilter=False, mode="constant", cval=0.0)
        np.testing.assert_allclose(np.asarray(resampled), expected, atol=1e-5)


# ── _GpuCellList ────────────────────────────────────────────────────────


class TestGpuCellList:
    @pytest.mark.parametrize("wide_cluster", [False, True])
    def test_matches_kdtree_neighbour_sets(self, wide_cluster):
        import jax.numpy as jnp

        rng = np.random.default_rng(42)
        coords = rng.uniform(0.0, 20.0, size=(200, 3))
        if wide_cluster:
            # Place 40 atoms within a single 3-cell column (32 < max_3col <= 64)
            cluster = rng.uniform(5.0, 5.5, size=(40, 3))
            coords = np.vstack([coords, cluster])
        n = len(coords)
        radius = 4.0

        cl = _GpuCellList.build(coords, radius)
        assert cl is not None
        assert cl.wide is wide_cluster

        ref = _NeighbourIndex( cl.sorted_coords, radius)
        ref_counts = ref.counts(cl.sorted_coords)
        k = max(cl.k_init, int(ref_counts.max()))
        k = max(8, -(-k // 8) * 8)
        ref_idx = ref.query(cl.sorted_coords, k)

        padded = np.vstack([cl.sorted_coords, np.full((1, 3), _FAR)])
        chi = padded.astype(np.float32)
        clo = (padded - chi).astype(np.float32)
        max_nbr, gpu_idx = cl.nbr_fn(
            jnp.arange(n, dtype=jnp.int32),
            jnp.asarray(cl.sorted_cell_id),
            cl.j_cell_edge,
            cl.j_xy_offsets,
            jnp.asarray(chi),
            jnp.asarray(clo),
            cl.j_rad_sq,
            k,
        )
        assert int(max_nbr) == int(ref_counts.max())
        gpu_idx_np = np.asarray(gpu_idx)
        for i in range(n):
            s_ref = set(ref_idx[i][ref_idx[i] < n].tolist())
            s_gpu = set(gpu_idx_np[i][gpu_idx_np[i] < n].tolist())
            assert s_gpu == s_ref

    def test_fallback_when_column_exceeds_64(self):
        rng = np.random.default_rng(7)
        # 70 atoms in a single cell (> 64 limit) → build returns None
        coords = rng.uniform(1.0, 1.1, size=(70, 3))
        assert _GpuCellList.build(coords, 4.0) is None
