"""jqscore — JAX reimplementation of Q-score.

Q-score measures how well each atom in an atomic model fits into a cryo-EM
density map by comparing the observed radial density profile around each atom
to a theoretical Gaussian profile, returning the Pearson correlation.

Reference: Pintilie et al., Nature Methods 2020.
Reference implementation: https://github.com/gregdp/mapq

Algorithm
---------
For each non-H atom:
  1. Sample the map at the atom centre (r = 0), replicated num_pts times.
  2. For r = d_rad, 2*d_rad, …, to_rad:
     - Generate sphere points via the Saff-Kuijlaars spiral.
     - Discard points whose closest non-H atom is within 0.9 * r.
     - Retry with more points until ≥ num_pts valid points are found.
     - Sample map density at the valid points.
     - Compute expected Gaussian value: G(r) = A·exp(-(r/σ)²/2) + B
  3. Q-score = Pearson correlation over all assembled (d_vals, g_vals).

Architecture
------------
1. CPU  (numpy / scipy): parse structure, map stats, build per-atom neighbour
   lists via cKDTree.  Per-atom neighbour lists are padded to a fixed shape K
   so all downstream arrays have static shapes — a prerequisite for JAX JIT.

2. GPU/CPU (JAX, JIT-compiled): sphere sampling, neighbour exclusion,
   trilinear interpolation, and masked Pearson correlation — all batched over
   atoms via ``jax.vmap`` and over shells via ``jax.lax.scan``.

Two computation modes (``use_retry`` parameter of :func:`compute_qscores`):

* ``use_retry=True`` (default): Replicates the original retry loop with
  ``jax.lax.while_loop``.  Q-scores differ from the pure-numpy reference only
  by float32 rounding (~1e-4); ~5× faster than numpy on CPU.

* ``use_retry=False`` (fast): Fixed ``n_gen`` sphere points per shell; all
  valid ones contribute.  ~7× faster than numpy at the cost of ~0.03 mean
  Q-score difference.

Default sigma: 0.4 Å (EMDB convention). Use ``sigma=0.6`` for older datasets.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from math import sqrt
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from scipy.ndimage import shift as _ndimage_shift
from scipy.spatial import cKDTree

try:
    import mrcfile
    _HAVE_MRCFILE = True
except ImportError:
    _HAVE_MRCFILE = False

try:
    import biotite.structure.io.pdb as _biotite_pdb
    import biotite.structure.io.pdbx as _biotite_pdbx
    _HAVE_BIOTITE = True
except ImportError:
    _HAVE_BIOTITE = False

try:
    import jax
    import jax.lax as lax
    import jax.numpy as jnp
    from jax import jit, vmap

    _HAVE_JAX = True
except ImportError:
    _HAVE_JAX = False


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

class AtomRecord(NamedTuple):
    """Minimal per-atom data needed for Q-score computation."""
    chain_id: str
    res_seq: int
    ins_code: str
    res_name: str
    atom_name: str
    element: str        # single-letter element symbol, upper-cased
    x: float
    y: float
    z: float
    alt_loc: str = ""   # alternate location indicator


class QScoreResult(NamedTuple):
    """Q-score results for all atoms in a structure."""
    atom_records:  List[AtomRecord]
    atom_qscores:  np.ndarray      # shape (N); non-H from correlation, H from nearest heavy atom
    min_d:         float
    max_d:         float
    sigma:         float


# ---------------------------------------------------------------------------
# Map I/O
# ---------------------------------------------------------------------------

def read_mrc(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read an MRC / CCP4 density map.

    Returns
    -------
    data : ndarray, shape (nz, ny, nx), float32
        Map data in ZYX (C-order) layout.
    origin : ndarray, shape (3,), float64
        Physical origin (x, y, z) in Ångströms.
    voxel_size : ndarray, shape (3,), float64
        Voxel spacing (dx, dy, dz) in Ångströms.

    Raises
    ------
    ImportError
        If the ``mrcfile`` package is not installed.
    """
    if not _HAVE_MRCFILE:
        raise ImportError("mrcfile is required to read MRC maps: pip install mrcfile")

    with mrcfile.open(path, mode="r", permissive=True) as mrc:
        raw = mrc.data.copy()

        vs = mrc.voxel_size
        voxel_size = np.array([float(vs.x), float(vs.y), float(vs.z)], dtype=np.float64)

        orig = mrc.header.origin
        origin = np.array([float(orig.x), float(orig.y), float(orig.z)], dtype=np.float64)

        # Fall back to nxstart / nystart / nzstart for old CCP4 files
        if np.all(origin == 0.0) and voxel_size[0] > 0:
            origin = np.array([
                int(mrc.header.nxstart) * voxel_size[0],
                int(mrc.header.nystart) * voxel_size[1],
                int(mrc.header.nzstart) * voxel_size[2],
            ], dtype=np.float64)

        mapc = int(mrc.header.mapc)
        mapr = int(mrc.header.mapr)
        maps = int(mrc.header.maps)

    data = raw.astype(np.float32)

    # mrcfile gives data.shape = (ns, nr, nc); reorder to ZYX.
    axis_to_cart = {0: maps, 1: mapr, 2: mapc}
    cart_to_axis = {v: k for k, v in axis_to_cart.items()}
    transpose_order = (cart_to_axis[3], cart_to_axis[2], cart_to_axis[1])
    if transpose_order != (0, 1, 2):
        data = np.ascontiguousarray(np.transpose(data, transpose_order))

    return data, origin, voxel_size


# ---------------------------------------------------------------------------
# Structure I/O
# ---------------------------------------------------------------------------

def _structure_to_atom_records(structure) -> List[AtomRecord]:
    """Convert a biotite AtomArray (or single model from AtomArrayStack) to AtomRecords."""
    if not _HAVE_BIOTITE:
        raise ImportError("biotite is required for structure I/O: pip install biotite")
    # Handle AtomArrayStack: take first model
    if hasattr(structure, "stack_depth") and structure.stack_depth() > 0:
        structure = structure[0]
    n = structure.array_length()
    categories = structure.get_annotation_categories()
    coords = structure.coord
    chain_id = structure.chain_id
    res_id = structure.res_id
    res_name = structure.res_name
    atom_name = structure.atom_name
    element = structure.element if "element" in categories else None
    ins_code = structure.ins_code if "ins_code" in categories else None
    alt_loc_id = structure.alt_loc_id if "alt_loc_id" in categories else None

    records: List[AtomRecord] = []
    for i in range(n):
        el = str(element[i]).strip().upper() if element is not None else ""
        if not el and atom_name is not None:
            an = str(atom_name[i]).strip()
            el = an.lstrip("0123456789").lstrip()[0].upper() if an else ""
        x, y, z = float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])
        ins = str(ins_code[i]).strip().replace("?", "").replace(".", "") if ins_code is not None else ""
        alt = str(alt_loc_id[i]).strip().replace("?", "").replace(".", "") if alt_loc_id is not None else ""
        records.append(AtomRecord(
            chain_id=str(chain_id[i]).strip(),
            res_seq=int(res_id[i]),
            ins_code=ins,
            res_name=str(res_name[i]).strip(),
            atom_name=str(atom_name[i]).strip(),
            element=el,
            x=x, y=y, z=z,
            alt_loc=alt,
        ))
    return records


def read_pdb(path: str) -> List[AtomRecord]:
    """Parse a PDB file and return ATOM / HETATM records (first model only)."""
    if not _HAVE_BIOTITE:
        raise ImportError("biotite is required for structure I/O: pip install biotite")
    pdb_file = _biotite_pdb.PDBFile.read(path)
    structure = pdb_file.get_structure(model=1, altloc="all")
    return _structure_to_atom_records(structure)


def read_cif(path: str) -> List[AtomRecord]:
    """Parse an mmCIF file and return atom_site records (first model only)."""
    if not _HAVE_BIOTITE:
        raise ImportError("biotite is required for structure I/O: pip install biotite")
    cif_file = _biotite_pdbx.CIFFile.read(path)
    structure = _biotite_pdbx.get_structure(cif_file, model=1, altloc="all")
    return _structure_to_atom_records(structure)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def min_max_d(data: np.ndarray) -> Tuple[float, float]:
    """Compute reference Gaussian offset B (minD) and max D (maxD).

        maxD = min(mean + 10 * std, max_val)
        minD = max(mean -  1 * std, min_val)

    Note: the Pearson correlation is invariant to affine rescaling of the
    reference Gaussian, so these values do not affect Q-scores themselves.
    """
    flat = data.ravel().astype(np.float64)
    mean = float(np.mean(flat))
    std  = float(np.std(flat))
    max_val = float(np.max(flat))
    min_val = float(np.min(flat))
    max_d = min(mean + 10.0 * std, max_val)
    min_d = max(mean -  1.0 * std, min_val)
    return min_d, max_d


_sphere_dir_cache: Dict[int, np.ndarray] = {}


def _sphere_directions(n: int) -> np.ndarray:
    """Return cached (n, 3) unit-sphere directions via Saff-Kuijlaars spiral."""
    if n in _sphere_dir_cache:
        return _sphere_dir_cache[n]
    if n <= 0:
        d: np.ndarray = np.empty((0, 3), dtype=np.float64)
        _sphere_dir_cache[n] = d
        return d
    if n == 1:
        d = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        _sphere_dir_cache[n] = d
        return d

    k = np.arange(1, n + 1, dtype=np.float64)
    h = np.clip(-1.0 + 2.0 * (k - 1) / float(n - 1), -1.0, 1.0)
    phi = np.arccos(h)
    denom = np.sqrt(np.maximum(0.0, n * (1.0 - h * h)))
    pos   = denom > 0.0
    incr  = np.zeros(n, dtype=np.float64)
    incr[pos] = 3.6 / denom[pos]
    incr[0] = 0.0
    theta = np.cumsum(incr) % (2.0 * np.pi)
    theta[-1] = 0.0

    sp = np.sin(phi)
    d = np.empty((n, 3), dtype=np.float64)
    d[:, 0] = sp * np.cos(theta)
    d[:, 1] = sp * np.sin(theta)
    d[:, 2] = np.cos(phi)
    _sphere_dir_cache[n] = d
    return d


# Maximum X-H distance (Å) to assign H the heavy atom's Q-score (typical bonds ~1.0–1.1)
_MAX_H_HEAVY_DIST = 1.2


def _assign_hydrogen_qscores(
    atoms: List[AtomRecord],
    all_coords: np.ndarray,
    non_h_mask: np.ndarray,
    non_h_indices: np.ndarray,
    q_scores: np.ndarray,
) -> None:
    """Set each hydrogen's Q-score to the Q-score of its nearest heavy atom in the same residue (in-place)."""
    # residue key -> list of (global index, coords) for non-H atoms in that residue
    residue_heavy: Dict[Tuple, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    for idx in non_h_indices:
        at = atoms[idx]
        key = (at.chain_id, at.res_seq, at.ins_code)
        residue_heavy[key].append((idx, all_coords[idx]))

    h_indices = np.where(~non_h_mask)[0]
    for i in h_indices:
        at = atoms[i]
        key = (at.chain_id, at.res_seq, at.ins_code)
        heavies = residue_heavy.get(key)
        if not heavies:
            continue
        pt = all_coords[i]
        best_idx = heavies[0][0]
        best_d2 = float("inf")
        for j, c in heavies:
            d2 = float(np.sum((pt - c) ** 2))
            if d2 < best_d2:
                best_d2 = d2
                best_idx = j
        if best_d2 <= _MAX_H_HEAVY_DIST ** 2:
            q_scores[i] = q_scores[best_idx]


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def residue_qscores(result: QScoreResult) -> Dict[Tuple, Dict[str, float]]:
    """Aggregate per-atom Q-scores to residue level (includes H via heavy-atom assignment).

    Returns
    -------
    dict
        Key: (chain_id, res_seq, ins_code, res_name)
        Value: dict with keys "all", "backbone", "sidechain"
    """
    BACKBONE = {"N", "CA", "C", "O", "OXT"}
    buckets: Dict[Tuple, Dict[str, List[float]]] = defaultdict(
        lambda: {"bb": [], "sc": []}
    )
    for at, q in zip(result.atom_records, result.atom_qscores):
        if np.isnan(q):
            continue
        key = (at.chain_id, at.res_seq, at.ins_code, at.res_name)
        if at.atom_name in BACKBONE:
            buckets[key]["bb"].append(q)
        else:
            buckets[key]["sc"].append(q)

    out: Dict[Tuple, Dict[str, float]] = {}
    for key, d in buckets.items():
        bb = d["bb"]
        sc = d["sc"]
        all_q = bb + sc
        out[key] = {
            "all":       float(np.mean(all_q)) if all_q else float("nan"),
            "backbone":  float(np.mean(bb))    if bb    else float("nan"),
            "sidechain": float(np.mean(sc))    if sc    else float("nan"),
        }
    return out


def chain_qscores(result: QScoreResult) -> Dict[str, float]:
    """Average Q-score per chain (all atoms with a Q-score, including H from heavy-atom assignment)."""
    chain_vals: Dict[str, List[float]] = defaultdict(list)
    for at, q in zip(result.atom_records, result.atom_qscores):
        if not np.isnan(q):
            chain_vals[at.chain_id].append(q)
    return {
        ch: float(np.mean(vals)) for ch, vals in chain_vals.items()
    }


# ---------------------------------------------------------------------------
# JAX helpers
# ---------------------------------------------------------------------------

def _require_jax() -> None:
    if not _HAVE_JAX:
        raise ImportError(
            "JAX is required.  Install with:\n"
            "  CPU only : pip install jax\n"
            "  CUDA GPU : pip install jax[cuda12]"
        )


def backend() -> str:
    """Return the active JAX backend string: ``'gpu'``, ``'tpu'``, or ``'cpu'``."""
    _require_jax()
    return jax.default_backend()


# ---------------------------------------------------------------------------
# JAX primitives  (float32 — GPU-friendly)
# ---------------------------------------------------------------------------

def _trilinear(data, origin, voxel_size, pts):
    """Trilinear map interpolation.

    Parameters
    ----------
    data       : (nz, ny, nx) float32 JAX array — density map in ZYX order.
    origin     : (3,) float32 — physical origin (x, y, z) in Ångströms.
    voxel_size : (3,) float32 — voxel spacing (dx, dy, dz).
    pts        : (N, 3) float32 — query points (x, y, z).

    Returns
    -------
    (N,) float32 — density values; 0 for points outside the map.
    """
    nz, ny, nx = data.shape

    gx = (pts[:, 0] - origin[0]) / voxel_size[0]
    gy = (pts[:, 1] - origin[1]) / voxel_size[1]
    gz = (pts[:, 2] - origin[2]) / voxel_size[2]

    ix = jnp.floor(gx).astype(jnp.int32)
    iy = jnp.floor(gy).astype(jnp.int32)
    iz = jnp.floor(gz).astype(jnp.int32)

    fx = gx - ix
    fy = gy - iy
    fz = gz - iz

    inside = (
        (ix >= 0) & (ix <= nx - 2) &
        (iy >= 0) & (iy <= ny - 2) &
        (iz >= 0) & (iz <= nz - 2)
    )

    cx = jnp.clip(ix, 0, nx - 2)
    cy = jnp.clip(iy, 0, ny - 2)
    cz = jnp.clip(iz, 0, nz - 2)

    v = (
        data[cz,     cy,     cx    ] * (1 - fx) * (1 - fy) * (1 - fz)
      + data[cz,     cy,     cx + 1] *      fx  * (1 - fy) * (1 - fz)
      + data[cz,     cy + 1, cx    ] * (1 - fx) *      fy  * (1 - fz)
      + data[cz,     cy + 1, cx + 1] *      fx  *      fy  * (1 - fz)
      + data[cz + 1, cy,     cx    ] * (1 - fx) * (1 - fy) *      fz
      + data[cz + 1, cy,     cx + 1] *      fx  * (1 - fy) *      fz
      + data[cz + 1, cy + 1, cx    ] * (1 - fx) *      fy  *      fz
      + data[cz + 1, cy + 1, cx + 1] *      fx  *      fy  *      fz
    )
    return jnp.where(inside, v, 0.0)


def _masked_pearson(d, g, mask):
    """Pearson correlation with a boolean validity mask.

    Parameters
    ----------
    d, g : (M,) float32 — density samples and Gaussian reference values.
    mask : (M,) bool    — True for valid (included) samples.

    Returns
    -------
    float32 scalar — Pearson r; 0 if denominator is zero.
    """
    n    = jnp.sum(mask.astype(jnp.float32))
    safe = jnp.maximum(n, jnp.float32(1.0))

    d_m  = jnp.where(mask, d, jnp.float32(0.0))
    g_m  = jnp.where(mask, g, jnp.float32(0.0))

    mu_d = jnp.sum(d_m) / safe
    mu_g = jnp.sum(g_m) / safe

    cd   = jnp.where(mask, d - mu_d, jnp.float32(0.0))
    cg   = jnp.where(mask, g - mu_g, jnp.float32(0.0))

    num      = jnp.dot(cd, cg)
    denom_sq = jnp.dot(cd, cd) * jnp.dot(cg, cg)

    return jnp.where(
        denom_sq > jnp.float32(0.0),
        num / jnp.sqrt(jnp.maximum(denom_sq, jnp.float32(1e-30))),
        jnp.float32(0.0),
    )


# ---------------------------------------------------------------------------
# Compiled kernel factory — fast (fixed n_gen) mode
# ---------------------------------------------------------------------------

def _make_compute_fn(n_gen: int, num_pts: int):
    """Build a JIT+vmap compiled function (fixed sphere-point budget).

    Signature
    ---------
    fn(atom_coords, nbr_coords, nbr_valid,
       map_data, origin, voxel_size,
       sphere_dirs, shell_radii, gv_shells, gv_r0)
    → q_scores : (N,) float32
    """
    def _qscore_one(atom_coord, nbr_coords, nbr_valid,
                    data, origin, voxel_size,
                    sphere_dirs, shell_radii, gv_shells, gv_r0):

        val_r0  = _trilinear(data, origin, voxel_size, atom_coord[None, :])[0]
        d_r0    = jnp.full((n_gen,), val_r0, dtype=jnp.float32)
        g_r0    = jnp.full((n_gen,), gv_r0, dtype=jnp.float32)
        mask_r0 = jnp.ones((n_gen,), dtype=jnp.bool_)

        def _shell(_, xs):
            r, gv = xs

            sph = atom_coord + sphere_dirs * r

            diff    = sph[:, None, :] - nbr_coords[None, :, :]
            dist_sq = jnp.sum(diff * diff, axis=-1)
            dist_sq = jnp.where(nbr_valid[None, :], dist_sq, jnp.inf)

            min_dsq = jnp.min(dist_sq, axis=1)
            out_r   = r * jnp.float32(0.9)
            sph_ok  = min_dsq > out_r * out_r

            n_ok   = jnp.sum(sph_ok.astype(jnp.int32))
            sph_ok = jnp.where(n_ok >= num_pts, sph_ok, jnp.zeros_like(sph_ok))

            d_vals = _trilinear(data, origin, voxel_size, sph)
            g_vals = jnp.full((n_gen,), gv, dtype=jnp.float32)

            return None, (d_vals, g_vals, sph_ok)

        _, (all_d, all_g, all_ok) = lax.scan(
            _shell, None, (shell_radii, gv_shells)
        )

        d_flat    = jnp.concatenate([d_r0,    all_d.ravel()])
        g_flat    = jnp.concatenate([g_r0,    all_g.ravel()])
        mask_flat = jnp.concatenate([mask_r0, all_ok.ravel()])

        return _masked_pearson(d_flat, g_flat, mask_flat)

    _batched = vmap(
        _qscore_one,
        in_axes=(0, 0, 0, None, None, None, None, None, None, None),
    )
    return jit(_batched)


_compiled_cache: dict = {}


def _get_compute_fn(n_gen: int, num_pts: int):
    key = (n_gen, num_pts)
    if key not in _compiled_cache:
        _compiled_cache[key] = _make_compute_fn(n_gen, num_pts)
    return _compiled_cache[key]


# ---------------------------------------------------------------------------
# Compiled kernel factory — retry mode
# ---------------------------------------------------------------------------

def _make_compute_fn_retry(num_pts: int, max_retries: int = 50):
    """Build a JIT+vmap function that replicates the numpy retry strategy.

    Iterates Saff-Kuijlaars(num_pts), Saff-Kuijlaars(num_pts+2), …
    until ≥ num_pts valid sphere points are found, using
    ``jax.lax.while_loop``.

    Signature
    ---------
    fn(atom_coords, nbr_coords, nbr_valid,
       map_data, origin, voxel_size,
       shell_radii, gv_shells, gv_r0)
    → q_scores : (N,) float32
    """
    max_n_try = num_pts + 2 * (max_retries - 1)

    all_dirs_np = np.zeros((max_retries, max_n_try, 3), dtype=np.float32)
    for k in range(max_retries):
        n_k = num_pts + 2 * k
        all_dirs_np[k, :n_k] = _sphere_directions(n_k).astype(np.float32)

    all_dirs_jax    = jnp.array(all_dirs_np)
    attempt_offsets = jnp.arange(max_n_try, dtype=jnp.int32)

    def _qscore_one(atom_coord, nbr_coords, nbr_valid,
                    data, origin, voxel_size,
                    shell_radii, gv_shells, gv_r0):

        val_r0  = _trilinear(data, origin, voxel_size, atom_coord[None, :])[0]
        d_r0    = jnp.full((num_pts,), val_r0, dtype=jnp.float32)
        g_r0    = jnp.full((num_pts,), gv_r0, dtype=jnp.float32)
        mask_r0 = jnp.ones((num_pts,), dtype=jnp.bool_)

        def _shell(_, xs):
            r, gv = xs

            init = (
                jnp.int32(0),
                jnp.bool_(False),
                jnp.zeros(max_n_try, dtype=jnp.float32),
                jnp.zeros(max_n_try, dtype=jnp.bool_),
            )

            def _cond(state):
                attempt, found, _, _ = state
                return (~found) & (attempt < jnp.int32(max_retries))

            def _body(state):
                attempt, found, d_acc, ok_acc = state

                n_try  = jnp.int32(num_pts) + attempt * jnp.int32(2)
                dirs_k = all_dirs_jax[attempt]
                sph    = atom_coord + dirs_k * r

                diff   = sph[:, None, :] - nbr_coords[None, :, :]
                dsq    = jnp.sum(diff * diff, axis=-1)
                dsq    = jnp.where(nbr_valid[None, :], dsq, jnp.inf)
                min_dq = jnp.min(dsq, axis=1)

                out_r  = r * jnp.float32(0.9)
                sph_ok = (min_dq > out_r * out_r) & (attempt_offsets < n_try)

                n_ok   = jnp.sum(sph_ok.astype(jnp.int32))

                d_vals = _trilinear(data, origin, voxel_size, sph)

                use    = (~found) & (n_ok >= jnp.int32(num_pts))
                new_d  = jnp.where(use, d_vals, d_acc)
                new_ok = jnp.where(use, sph_ok, ok_acc)

                return (attempt + jnp.int32(1), found | use, new_d, new_ok)

            _, _, d_vals, ok_mask = lax.while_loop(_cond, _body, init)
            g_vals = jnp.full((max_n_try,), gv)

            return None, (d_vals, g_vals, ok_mask)

        _, (all_d, all_g, all_ok) = lax.scan(
            _shell, None, (shell_radii, gv_shells)
        )

        d_flat    = jnp.concatenate([d_r0,    all_d.ravel()])
        g_flat    = jnp.concatenate([g_r0,    all_g.ravel()])
        mask_flat = jnp.concatenate([mask_r0, all_ok.ravel()])

        return _masked_pearson(d_flat, g_flat, mask_flat)

    _batched = vmap(
        _qscore_one,
        in_axes=(0, 0, 0, None, None, None, None, None, None),
    )
    return jit(_batched)


_compiled_cache_retry: dict = {}


def _get_compute_fn_retry(num_pts: int, max_retries: int):
    key = (num_pts, max_retries)
    if key not in _compiled_cache_retry:
        _compiled_cache_retry[key] = _make_compute_fn_retry(num_pts, max_retries)
    return _compiled_cache_retry[key]


# ---------------------------------------------------------------------------
# CPU-side neighbour pre-bucketing
# ---------------------------------------------------------------------------

def _build_neighbour_lists(
    coords: np.ndarray,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build padded per-atom neighbour arrays using a CPU cKDTree.

    Returns
    -------
    nbr_coords : (N, K, 3) float32 — neighbour positions, zero-padded.
    nbr_valid  : (N, K) bool       — True for real (non-padding) slots.
    """
    tree  = cKDTree(coords)
    lists = tree.query_ball_point(coords, radius)
    K     = max((len(nb) for nb in lists), default=1)
    N     = len(coords)

    nbr_c = np.zeros((N, K, 3), dtype=np.float32)
    nbr_v = np.zeros((N, K),    dtype=np.bool_)

    for i, nbrs in enumerate(lists):
        k = len(nbrs)
        if k:
            nbr_c[i, :k] = coords[nbrs]
            nbr_v[i, :k] = True

    return nbr_c, nbr_v


# ---------------------------------------------------------------------------
# MapQ compatibility: replicate MaskMapResize origin rounding
# ---------------------------------------------------------------------------

def _resample_mapq_compat(
    data: np.ndarray,
    origin: np.ndarray,
    voxel_size: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample the map to replicate MapQ's MaskMapResize trimmed-map effect.

    MapQ creates a per-subprocess trimmed map whose origin is rounded to
    ``round(nO/step)*step`` using float32 arithmetic (Chimera reads MRC
    headers in float32).  This origin is not in general aligned with the
    original voxel grid, introducing a fixed fractional voxel offset that
    acts as a slight blur.  The subprocess then interpolates *this* map,
    giving double interpolation in total.

    This function reproduces that effect for the whole map so that jqscore
    queries the same blurred densities that MapQ does.

    The fractional offset depends only on ``O_orig / step mod 1`` — it is
    identical across all MapQ subprocesses and all atoms.

    Parameters
    ----------
    data : (nz, ny, nx) float32
    origin, voxel_size : (3,) float64 — map coordinate system (x, y, z)

    Returns
    -------
    resampled : (nz, ny, nx) float32 — blurred map
    new_origin : (3,) float64 — rounded origin to pass to JAX
    """
    # Chimera reads MRC origin/step in float32 — replicate that precision
    O  = origin.astype(np.float32).astype(np.float64)
    vs = voxel_size.astype(np.float32).astype(np.float64)

    # MaskMapResize: ox = round(nO[0]/step)*step  (same fractional offset for all procs)
    new_origin = np.round(O / vs) * vs  # (3,) float64

    # Fractional shift in original voxel space (x, y, z)
    offset_xyz = (new_origin - O) / vs

    # scipy.ndimage.shift: output[iz,iy,ix] = input[iz-sz, iy-sy, ix-sx]
    # We want:             output[iz,iy,ix] = input[iz+dz, iy+dy, ix+dx]
    # so shift = [-dz, -dy, -dx]  (note data layout is z,y,x)
    shift_zyx = np.array([-offset_xyz[2], -offset_xyz[1], -offset_xyz[0]], dtype=np.float64)

    resampled = _ndimage_shift(
        data, shift_zyx, order=1, prefilter=False, mode="constant", cval=0.0,
    )

    return resampled.astype(np.float32), new_origin


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_qscores(
    atoms: List[AtomRecord],
    data: np.ndarray,
    origin: np.ndarray,
    voxel_size: np.ndarray,
    sigma: float = 0.4,
    num_pts: int = 8,
    to_rad: Optional[float] = None,
    d_rad: Optional[float] = None,
    use_retry: bool = True,
    max_retries: int = 50,
    n_gen: Optional[int] = None,
    nbr_radius: Optional[float] = None,
    mapq_compat: bool = True,
    batch_size: Optional[int] = 2048,
    verbose: bool = False,
) -> QScoreResult:
    """Compute per-atom Q-scores using JAX (GPU by default, CPU fallback).

    Parameters
    ----------
    atoms : list of AtomRecord
        All atoms from the parsed structure file.
    data : ndarray, shape (nz, ny, nx)
        Density map in ZYX order.
    origin, voxel_size : ndarray, shape (3,)
        Map coordinate system.
    sigma : float
        Gaussian width in Ångströms.  Default 0.4 Å (EMDB convention).
        Use 0.6 for older datasets, or larger values for SSE scoring.
    num_pts : int
        Target sample points per shell.
    to_rad, d_rad : float, optional
        Radial range and step.  Defaults:
          sigma < 1.0 → to_rad=2.0, d_rad=0.1
          sigma ≥ 1.0 → to_rad=6.0, d_rad=0.5
    use_retry : bool, default True
        When True (default), replicates the original retry loop via
        ``jax.lax.while_loop``.  Q-scores differ from a pure-numpy reference
        only by float32 rounding (~1e-4); ~5× faster than numpy on CPU.

        When False, uses a fixed ``n_gen`` sphere-point budget per shell.
        ~7× faster at the cost of ~0.03 mean Q-score difference.
    max_retries : int, default 50
        Maximum retry attempts per shell (``use_retry=True`` only).
    n_gen : int, optional
        Fixed sphere-point budget per shell for ``use_retry=False``.
        Default: ``max(num_pts * 16, 128)``.
    nbr_radius : float, optional
        Radius (Å) for the CPU neighbour search.
        Default: ``to_rad * 2.0``.
    mapq_compat : bool, default True
        Resample the map to match MapQ's ``MaskMapResize`` behavior before
        computing Q-scores.  MapQ rounds the trimmed-map origin to the nearest
        multiple of the voxel size (using float32 arithmetic), introducing a
        fixed fractional voxel offset that slightly blurs the map.  Enabling
        this option replicates that effect so that Q-scores match MapQ's output
        to within float32 rounding (~1e-4).

        Set to ``False`` for the physically more accurate computation (single
        interpolation directly from the original map).  See
        ``docs/jqscore_vs_mapq/mapq_trimmed_map_discrepancy.md`` for details.
    batch_size : int or None, default 2048
        Number of atoms to process per JAX kernel call.  The kernel's peak GPU
        memory (including XLA compilation overhead) grows with this value.
        Reduce for tight GPUs; set to ``None`` to process all atoms in a single
        call (may OOM for large structures).
    verbose : bool
        Print progress information.

    Returns
    -------
    QScoreResult
    """
    _require_jax()

    if to_rad is None:
        to_rad = 2.0 if sigma < 1.0 else 6.0
    if d_rad is None:
        d_rad = 0.1 if sigma < 1.0 else 0.5
    if nbr_radius is None:
        nbr_radius = to_rad * 2.0

    min_d, max_d = min_max_d(data)
    A     = float(max_d - min_d)
    B     = float(min_d)
    gv_r0 = float(A + B)

    if mapq_compat:
        if verbose:
            print("[mapq_compat] Resampling map to replicate MaskMapResize origin rounding …")
        data, origin = _resample_mapq_compat(data, origin, voxel_size)

    non_h_mask    = np.array([a.element != "H" for a in atoms], dtype=bool)
    non_h_indices = np.where(non_h_mask)[0]

    if len(non_h_indices) == 0:
        warnings.warn("No non-hydrogen atoms found.")
        return QScoreResult(atoms, np.full(len(atoms), np.nan), min_d, max_d, sigma)

    all_coords   = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float32)
    non_h_coords = all_coords[non_h_mask]
    N            = len(non_h_indices)

    shell_radii_np = np.arange(d_rad, to_rad + 1e-6, d_rad, dtype=np.float32)
    gv_shells_np   = A * np.exp(-0.5 * (shell_radii_np / sigma) ** 2) + B

    if verbose:
        mode_str = "retry" if use_retry else f"fast (n_gen={n_gen or max(num_pts*16,128)})"
        print(
            f"[JAX / {jax.default_backend().upper()} / {mode_str}]  "
            f"Building neighbour lists (r = {nbr_radius:.1f} Å) …"
        )
    nbr_coords_np, nbr_valid_np = _build_neighbour_lists(non_h_coords, nbr_radius)

    jdata   = jnp.array(data.astype(np.float32))
    jorigin = jnp.array(origin.astype(np.float32))
    jvs     = jnp.array(voxel_size.astype(np.float32))
    jradii  = jnp.array(shell_radii_np)
    jgv     = jnp.array(gv_shells_np.astype(np.float32))
    jgv_r0  = jnp.float32(gv_r0)

    K = nbr_coords_np.shape[1]

    if verbose:
        n_shells = len(shell_radii_np)
        print(
            f"Computing Q-scores:  {N} atoms  ×  {n_shells} shells  "
            f"(max {K} neighbours/atom)"
        )

    # None = legacy single-call (may OOM for large structures)
    effective_batch = N if batch_size is None else max(1, batch_size)

    if use_retry:
        compute_fn = _get_compute_fn_retry(num_pts, max_retries)
        jdirs = None
    else:
        if n_gen is None:
            n_gen = max(num_pts * 16, 128)
        jdirs = jnp.array(_sphere_directions(n_gen).astype(np.float32))
        compute_fn = _get_compute_fn(n_gen, num_pts)

    def _call_compute(bcoords_j, bc_j, bv_j):
        if use_retry:
            return compute_fn(bcoords_j, bc_j, bv_j, jdata, jorigin, jvs, jradii, jgv, jgv_r0)
        else:
            return compute_fn(bcoords_j, bc_j, bv_j, jdata, jorigin, jvs, jdirs, jradii, jgv, jgv_r0)

    if effective_batch >= N:
        if verbose and effective_batch < 2 * N:
            print(f"  No batching needed ({N} atoms fit in one call)")
        q_non_h = np.array(_call_compute(
            jnp.array(non_h_coords),
            jnp.array(nbr_coords_np),
            jnp.array(nbr_valid_np),
        ))
    else:
        if verbose:
            n_batches = (N + effective_batch - 1) // effective_batch
            print(f"  Batching: {effective_batch} atoms/batch ({n_batches} batches)")
        q_non_h = np.empty(N, dtype=np.float32)
        for start in range(0, N, effective_batch):
            actual   = min(start + effective_batch, N)
            bc_np    = nbr_coords_np[start:actual]
            bv_np    = nbr_valid_np[start:actual]
            coord_np = non_h_coords[start:actual]
            if actual < start + effective_batch:
                # Pad last batch to fixed shape to avoid JIT recompilation
                pad      = start + effective_batch - actual
                bc_np    = np.concatenate([bc_np,    np.zeros((pad, K, 3), dtype=np.float32)])
                bv_np    = np.concatenate([bv_np,    np.zeros((pad, K),    dtype=np.bool_)])
                coord_np = np.concatenate([coord_np, np.zeros((pad, 3),    dtype=np.float32)])
            batch_q = np.array(_call_compute(
                jnp.array(coord_np),
                jnp.array(bc_np),
                jnp.array(bv_np),
            ))
            q_non_h[start:actual] = batch_q[:actual - start]

    q_scores = np.full(len(atoms), np.nan, dtype=np.float64)
    q_scores[non_h_indices] = q_non_h.astype(np.float64)

    # Assign H atoms the Q-score of their nearest heavy atom in the same residue (reference behavior)
    _assign_hydrogen_qscores(atoms, all_coords, non_h_mask, non_h_indices, q_scores)

    if verbose:
        for k, global_idx in enumerate(non_h_indices):
            at = atoms[global_idx]
            print(f"  {at.res_seq}.{at.chain_id}.{at.atom_name}: Q = {q_non_h[k]:.6f}")

    return QScoreResult(atoms, q_scores, min_d, max_d, sigma)


def run(
    structure_path: str,
    map_path: str,
    sigma: float = 0.4,
    num_pts: int = 8,
    to_rad: Optional[float] = None,
    d_rad: Optional[float] = None,
    use_retry: bool = True,
    max_retries: int = 50,
    n_gen: Optional[int] = None,
    nbr_radius: Optional[float] = None,
    mapq_compat: bool = True,
    batch_size: Optional[int] = 2048,
    verbose: bool = False,
) -> QScoreResult:
    """Compute Q-scores from file paths using the JAX backend.

    Parameters
    ----------
    structure_path : str
        Path to a PDB (``.pdb``) or mmCIF (``.cif`` / ``.mmcif``) file.
    map_path : str
        Path to an MRC / CCP4 density map.
    sigma, num_pts, to_rad, d_rad, use_retry, max_retries, n_gen, nbr_radius,
    mapq_compat, verbose
        Forwarded to :func:`compute_qscores`.

    Returns
    -------
    QScoreResult
    """
    data, origin, voxel_size = read_mrc(map_path)

    path_lower = structure_path.lower()
    if path_lower.endswith(".cif") or path_lower.endswith(".mmcif"):
        atoms = read_cif(structure_path)
    else:
        atoms = read_pdb(structure_path)

    return compute_qscores(
        atoms, data, origin, voxel_size,
        sigma=sigma, num_pts=num_pts, to_rad=to_rad, d_rad=d_rad,
        use_retry=use_retry, max_retries=max_retries,
        n_gen=n_gen, nbr_radius=nbr_radius,
        mapq_compat=mapq_compat, batch_size=batch_size, verbose=verbose,
    )


# ---------------------------------------------------------------------------
# mmCIF output
# ---------------------------------------------------------------------------

def write_qscore_cif(
    result: QScoreResult,
    input_cif_path: str,
    map_path: str,
    output_dir: Optional[str] = None,
) -> str:
    """Write a mapq-compatible mmCIF file with an ``_atom_site.Q-score`` column.

    Reads *input_cif_path* with biotite, adds Q-scores from *result* as an
    ``_atom_site.Q-score`` annotation, and writes via biotite's CIF writer to::

        {output_dir}/{cif_basename}__Q__{map_basename}.cif

    Parameters
    ----------
    result:
        Output of :func:`run` or :func:`compute_qscores`.  The ordering of
        ``result.atom_records`` must match the atom rows in *input_cif_path*
        (i.e. the file must not have been modified between reading and this
        call).
    input_cif_path:
        Source mmCIF structure file.
    map_path:
        Density map path — only its basename is used for naming the output.
    output_dir:
        Directory to write into.  Defaults to the directory of
        *input_cif_path*.

    Returns
    -------
    str
        Absolute path of the written file.
    """
    import os

    if not _HAVE_BIOTITE:
        raise ImportError("biotite is required for CIF output: pip install biotite")

    cif_basename = os.path.basename(input_cif_path)
    map_basename = os.path.basename(map_path)
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_cif_path))
    out_path = os.path.join(output_dir, f"{cif_basename}__Q__{map_basename}.cif")

    cif_file = _biotite_pdbx.CIFFile.read(input_cif_path)
    structure = _biotite_pdbx.get_structure(
        cif_file, model=1, altloc="all", extra_fields=["b_factor", "occupancy"]
    )

    if structure.array_length() != len(result.atom_qscores):
        raise ValueError(
            f"Atom count mismatch: CIF has {structure.array_length()} atoms, "
            f"result has {len(result.atom_qscores)}. "
            "Input CIF must not be modified between run() and write_qscore_cif()."
        )

    qs = result.atom_qscores
    qscore_col = np.array(
        ["?" if np.isnan(x) else f"{x:.6f}" for x in qs],
        dtype="U16",
    )
    structure.set_annotation("Q-score", qscore_col)

    _biotite_pdbx.set_structure(cif_file, structure, extra_fields=["Q-score"])
    cif_file.write(out_path)

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="jqscore: JAX-accelerated Q-score (atom-map correlation metric)."
    )
    parser.add_argument("structure", help="PDB or mmCIF structure file")
    parser.add_argument("map",       help="MRC / CCP4 density map")
    parser.add_argument("--sigma",       type=float, default=0.4,
                        help="Gaussian sigma (Å, default 0.4; use 0.6 for older datasets)")
    parser.add_argument("--num-pts",     type=int,   default=8,
                        help="Sample points per shell (default 8)")
    parser.add_argument("--to-rad",      type=float, default=None,
                        help="Max radial distance (Å)")
    parser.add_argument("--d-rad",       type=float, default=None,
                        help="Radial step size (Å)")
    parser.add_argument("--no-retry",    action="store_true",
                        help="Use fast fixed-budget mode instead of retry loop")
    parser.add_argument("--max-retries", type=int,   default=50,
                        help="Max retry attempts per shell (default 50)")
    parser.add_argument("--n-gen",       type=int,   default=None,
                        help="Sphere-point budget per shell for --no-retry mode")
    parser.add_argument("--no-mapq-compat", action="store_true",
                        help="Disable MapQ trimmed-map resampling (more accurate, "
                             "but Q-scores may differ from MapQ by up to ~0.18 "
                             "for atoms in unusual density environments)")
    parser.add_argument("--batch-size",  type=int, default=2048,
                        help="Atoms per JAX kernel call (default 2048). "
                             "Reduce for tight GPUs.")
    parser.add_argument("--verbose",     action="store_true",
                        help="Print per-atom Q-scores during computation")
    parser.add_argument("--output",      default=None,
                        help="Write tab-separated per-residue results to this file "
                             "(default: print summary to stdout)")
    args = parser.parse_args()

    result = run(
        args.structure, args.map,
        sigma=args.sigma, num_pts=args.num_pts,
        to_rad=args.to_rad, d_rad=args.d_rad,
        use_retry=not args.no_retry,
        max_retries=args.max_retries,
        n_gen=args.n_gen,
        mapq_compat=not args.no_mapq_compat,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    res_qs = residue_qscores(result)
    ch_qs  = chain_qscores(result)

    if args.output:
        with open(args.output, "w") as fout:
            fout.write("chain\tres_seq\tres_name\tQ_backbone\tQ_sidechain\tQ_residue\n")
            for (ch, rs, ic, rn), d in sorted(res_qs.items()):
                fout.write(
                    f"{ch}\t{rs}{ic}\t{rn}\t"
                    f"{d['backbone']:.6f}\t{d['sidechain']:.6f}\t{d['all']:.6f}\n"
                )
        print(f"Wrote residue Q-scores to {args.output}", file=sys.stderr)
    else:
        print(f"sigma={result.sigma:.2f}  minD={result.min_d:.6f}  maxD={result.max_d:.6f}")
        print()
        for ch, q in sorted(ch_qs.items()):
            n = sum(1 for a, v in zip(result.atom_records, result.atom_qscores)
                    if a.chain_id == ch and not np.isnan(v))
            print(f"Chain {ch}: Q = {q:.6f}  ({n} non-H atoms)")
        print()
        print("chain\tres_seq\tres_name\tQ_backbone\tQ_sidechain\tQ_residue")
        for (ch, rs, ic, rn), d in sorted(res_qs.items())[:20]:
            bb = f"{d['backbone']:.6f}" if not np.isnan(d['backbone']) else "   -    "
            sc = f"{d['sidechain']:.6f}" if not np.isnan(d['sidechain']) else "   -    "
            print(f"{ch}\t{rs}{ic}\t{rn}\t{bb}\t{sc}\t{d['all']:.6f}")


if __name__ == "__main__":
    _cli()
