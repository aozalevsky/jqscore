"""Shared fixtures and markers for jqscore tests."""

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
TEST_DATA = TESTS_DIR / "data"
DOCS_VS_MAPQ = PROJECT_ROOT / "docs" / "jqscore_vs_mapq"

# In-repo test data (small, from MapQ repo: 100 atoms of 7cec chain A, sigma=0.6)
SMALL_STRUCTURE = TEST_DATA / "390_7cec_A100.cif"
SMALL_MAP = TEST_DATA / "390_emd_30342_A_z4.mrc"
SMALL_MAPQ_REF = TEST_DATA / "390_7cec_A100__Q__390_emd_30342_A_z4.cif"
SMALL_S04_REF = TEST_DATA / "390_7cec_A100__Q__390_emd_30342_A_z4_s04.cif"

# 9XZK subset: 15 residues of chain C, trimmed map, MapQ reference with sigma=0.4
XZAK_STRUCTURE = TEST_DATA / "9XZK_C15.cif"
XZAK_MAP = TEST_DATA / "9XZK_C15_map.mrc"
XZAK_MAPQ_REF = TEST_DATA / "9XZK_C15__Q__mapq_s04.cif"

# Docs comparison CIFs (in-repo, ~800K each)
ORIG_CIF = DOCS_VS_MAPQ / "9XZK.cif__Q__emd_72359.map.cif"
JAX_CIF = DOCS_VS_MAPQ / "jax_gpu_9XZK.cif__Q__EMD-72359.map.cif"


# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------

has_docs_cifs = pytest.mark.skipif(
    not (ORIG_CIF.exists() and JAX_CIF.exists()),
    reason="Reference CIF files not found in docs/jqscore_vs_mapq/",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_map():
    """A tiny 10x10x10 synthetic density map with a Gaussian blob at centre."""
    shape = (10, 10, 10)
    data = np.zeros(shape, dtype=np.float32)
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    voxel_size = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    # Place a Gaussian blob at (5,5,5)
    for iz in range(shape[0]):
        for iy in range(shape[1]):
            for ix in range(shape[2]):
                r2 = (ix - 5.0) ** 2 + (iy - 5.0) ** 2 + (iz - 5.0) ** 2
                data[iz, iy, ix] = np.exp(-r2 / (2.0 * 1.5**2))

    return data, origin, voxel_size


@pytest.fixture
def ramp_map():
    """A 5x5x5 map where data[z,y,x] = x + 10*y + 100*z (for trilinear checks)."""
    data = np.zeros((5, 5, 5), dtype=np.float32)
    for iz in range(5):
        for iy in range(5):
            for ix in range(5):
                data[iz, iy, ix] = float(ix + 10 * iy + 100 * iz)
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    voxel_size = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    return data, origin, voxel_size
