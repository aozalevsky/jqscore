"""Unit tests for core helper functions in jqscore.jqscore."""

import numpy as np
import pytest

import jax.numpy as jnp

from jqscore.jqscore import (
    _sphere_directions,
    _trilinear,
    _masked_pearson,
    min_max_d,
)


# ── _sphere_directions (Saff-Kuijlaars spiral) ──────────────────────────


class TestSphereDirections:
    def test_n_zero(self):
        d = _sphere_directions(0)
        assert d.shape == (0, 3)

    def test_n_one(self):
        d = _sphere_directions(1)
        assert d.shape == (1, 3)
        np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-12)

    def test_correct_count(self):
        for n in [2, 8, 16, 100]:
            d = _sphere_directions(n)
            assert d.shape == (n, 3), f"Expected {n} directions, got {d.shape[0]}"

    def test_unit_vectors(self):
        d = _sphere_directions(50)
        norms = np.linalg.norm(d, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_roughly_uniform_coverage(self):
        """Mean direction should be near zero for a roughly uniform distribution."""
        d = _sphere_directions(500)
        mean_dir = d.mean(axis=0)
        assert np.linalg.norm(mean_dir) < 0.1, f"Mean direction too far from origin: {mean_dir}"

    def test_caching(self):
        """Same n should return the exact same array object (caching)."""
        d1 = _sphere_directions(20)
        d2 = _sphere_directions(20)
        assert d1 is d2

    def test_dtype_float64(self):
        d = _sphere_directions(10)
        assert d.dtype == np.float64

    def test_poles_present(self):
        """First and last point should be near the poles for n > 1."""
        d = _sphere_directions(100)
        # First point should be near south pole (z ~ -1)
        assert d[0, 2] < -0.9
        # Last point should be near north pole (z ~ +1)
        assert d[-1, 2] > 0.9


# ── min_max_d ────────────────────────────────────────────────────────────


class TestMinMaxD:
    def test_uniform_data(self):
        data = np.ones((5, 5, 5), dtype=np.float32) * 3.0
        min_d, max_d = min_max_d(data)
        # std=0 → max_d = min(mean+0, max) = 3, min_d = max(mean-0, min) = 3
        assert min_d == pytest.approx(3.0)
        assert max_d == pytest.approx(3.0)

    def test_known_values(self):
        # Simple 1D-like data to verify the formula
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32).reshape(1, 1, 5)
        mean = 3.0
        std = np.std([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_max_d = min(mean + 10 * std, 5.0)
        expected_min_d = max(mean - 1 * std, 1.0)
        min_d, max_d = min_max_d(data)
        assert min_d == pytest.approx(expected_min_d, rel=1e-5)
        assert max_d == pytest.approx(expected_max_d, rel=1e-5)

    def test_min_d_capped_at_data_min(self):
        # If mean - std < min_val, min_d should be min_val
        data = np.array([10.0, 10.0, 10.0, 10.1], dtype=np.float32).reshape(1, 1, 4)
        min_d, _ = min_max_d(data)
        assert min_d >= 10.0

    def test_max_d_capped_at_data_max(self):
        # If mean + 10*std > max_val, max_d should be max_val
        data = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32).reshape(1, 1, 4)
        _, max_d = min_max_d(data)
        assert max_d <= 1.0


# ── _trilinear ───────────────────────────────────────────────────────────


class TestTrilinear:
    def test_at_grid_points(self, ramp_map):
        """Values at exact grid points should match the data."""
        data, origin, voxel_size = ramp_map
        jdata = jnp.array(data)
        jorigin = jnp.array(origin, dtype=jnp.float32)
        jvs = jnp.array(voxel_size, dtype=jnp.float32)

        pts = jnp.array([[2.0, 3.0, 1.0]], dtype=jnp.float32)  # x=2, y=3, z=1
        expected = 2.0 + 10.0 * 3.0 + 100.0 * 1.0  # = 132.0
        result = _trilinear(jdata, jorigin, jvs, pts)
        np.testing.assert_allclose(float(result[0]), expected, atol=1e-4)

    def test_midpoint_interpolation(self, ramp_map):
        """Midpoint between two grid points should average their values."""
        data, origin, voxel_size = ramp_map
        jdata = jnp.array(data)
        jorigin = jnp.array(origin, dtype=jnp.float32)
        jvs = jnp.array(voxel_size, dtype=jnp.float32)

        # Midpoint between (1,0,0) and (2,0,0) in x
        pts = jnp.array([[1.5, 0.0, 0.0]], dtype=jnp.float32)
        v1 = 1.0  # data[0,0,1]
        v2 = 2.0  # data[0,0,2]
        expected = (v1 + v2) / 2.0
        result = _trilinear(jdata, jorigin, jvs, pts)
        np.testing.assert_allclose(float(result[0]), expected, atol=1e-4)

    def test_outside_map_returns_zero(self, ramp_map):
        data, origin, voxel_size = ramp_map
        jdata = jnp.array(data)
        jorigin = jnp.array(origin, dtype=jnp.float32)
        jvs = jnp.array(voxel_size, dtype=jnp.float32)

        pts = jnp.array([[-1.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=jnp.float32)
        result = _trilinear(jdata, jorigin, jvs, pts)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_with_nonzero_origin(self):
        """Trilinear should account for origin offset."""
        data = np.ones((3, 3, 3), dtype=np.float32) * 5.0
        origin = jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32)
        voxel_size = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)

        # Point at origin + (1,1,1) should be inside
        pts = jnp.array([[11.0, 21.0, 31.0]], dtype=jnp.float32)
        result = _trilinear(jnp.array(data), origin, voxel_size, pts)
        np.testing.assert_allclose(float(result[0]), 5.0, atol=1e-4)

        # Point at physical (0,0,0) should be outside
        pts_outside = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        result_out = _trilinear(jnp.array(data), origin, voxel_size, pts_outside)
        np.testing.assert_allclose(float(result_out[0]), 0.0, atol=1e-6)

    def test_with_nonunit_voxel_size(self):
        """Trilinear should handle non-unit voxel spacing."""
        data = np.zeros((3, 3, 3), dtype=np.float32)
        data[0, 0, 0] = 1.0
        data[0, 0, 1] = 3.0
        origin = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
        voxel_size = jnp.array([2.0, 2.0, 2.0], dtype=jnp.float32)

        # At x=1.0 physical → grid x=0.5 → midpoint between data[0,0,0]=1 and data[0,0,1]=3
        pts = jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32)
        result = _trilinear(jnp.array(data), origin, voxel_size, pts)
        np.testing.assert_allclose(float(result[0]), 2.0, atol=1e-4)

    def test_batch_of_points(self, ramp_map):
        data, origin, voxel_size = ramp_map
        jdata = jnp.array(data)
        jorigin = jnp.array(origin, dtype=jnp.float32)
        jvs = jnp.array(voxel_size, dtype=jnp.float32)

        pts = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=jnp.float32)
        expected = np.array([0.0, 1.0, 10.0, 100.0])
        result = _trilinear(jdata, jorigin, jvs, pts)
        np.testing.assert_allclose(result, expected, atol=1e-4)


# ── _masked_pearson ──────────────────────────────────────────────────────


class TestMaskedPearson:
    def test_perfect_positive_correlation(self):
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
        g = jnp.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=jnp.float32)
        mask = jnp.ones(5, dtype=jnp.bool_)
        r = float(_masked_pearson(d, g, mask))
        assert r == pytest.approx(1.0, abs=1e-5)

    def test_perfect_negative_correlation(self):
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
        g = jnp.array([10.0, 8.0, 6.0, 4.0, 2.0], dtype=jnp.float32)
        mask = jnp.ones(5, dtype=jnp.bool_)
        r = float(_masked_pearson(d, g, mask))
        assert r == pytest.approx(-1.0, abs=1e-5)

    def test_zero_correlation(self):
        d = jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float32)
        g = jnp.array([1.0, 1.0, -1.0, -1.0], dtype=jnp.float32)
        mask = jnp.ones(4, dtype=jnp.bool_)
        r = float(_masked_pearson(d, g, mask))
        assert r == pytest.approx(0.0, abs=1e-5)

    def test_masking_excludes_elements(self):
        d = jnp.array([1.0, 2.0, 3.0, 999.0], dtype=jnp.float32)
        g = jnp.array([2.0, 4.0, 6.0, -999.0], dtype=jnp.float32)
        mask = jnp.array([True, True, True, False])
        r = float(_masked_pearson(d, g, mask))
        assert r == pytest.approx(1.0, abs=1e-5)

    def test_all_masked_returns_zero(self):
        d = jnp.array([1.0, 2.0], dtype=jnp.float32)
        g = jnp.array([3.0, 4.0], dtype=jnp.float32)
        mask = jnp.zeros(2, dtype=jnp.bool_)
        r = float(_masked_pearson(d, g, mask))
        assert r == pytest.approx(0.0, abs=1e-5)

    def test_constant_input_returns_zero(self):
        """If d is constant, correlation is undefined → returns 0."""
        d = jnp.array([5.0, 5.0, 5.0], dtype=jnp.float32)
        g = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        mask = jnp.ones(3, dtype=jnp.bool_)
        r = float(_masked_pearson(d, g, mask))
        assert r == pytest.approx(0.0, abs=1e-5)

    def test_matches_numpy_corrcoef(self):
        rng = np.random.default_rng(42)
        d_np = rng.standard_normal(50).astype(np.float32)
        g_np = rng.standard_normal(50).astype(np.float32)
        mask = jnp.ones(50, dtype=jnp.bool_)

        expected = float(np.corrcoef(d_np, g_np)[0, 1])
        result = float(_masked_pearson(jnp.array(d_np), jnp.array(g_np), mask))
        assert result == pytest.approx(expected, abs=1e-4)
