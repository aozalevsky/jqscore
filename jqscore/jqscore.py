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
1. CPU (numpy / scipy): parse structure; per batch of atoms, look up
   neighbours in a cKDTree (all cores) as an index array padded to K slots.

2. GPU/CPU (JAX, JIT-compiled): map statistics, MapQ-compatible resampling,
   sphere sampling, neighbour exclusion, trilinear interpolation.  The map
   stays on the device; the neighbour search for the next batch runs on the
   CPU while the device works on the current one.

Two computation modes (``use_retry`` parameter of :func:`compute_qscores`):

* ``use_retry=True`` (default): Replicates the original retry loop.  The
  attempts are checked in rounds (see :func:`_retry_rounds`); only the
  (atom, shell) pairs still unresolved after a round go on to the next, and
  the map is sampled once, for the attempt that succeeded.  Each shell
  contributes the count, sum and sum of squares of its densities, and the
  correlation is assembled from these in float64.

* ``use_retry=False`` (fast): Fixed ``n_gen`` sphere points per shell; all
  valid ones contribute.  ~0.03 mean Q-score difference from retry mode.

Default sigma: 0.4 Å (EMDB convention). Use ``sigma=0.6`` for older datasets.
"""

from __future__ import annotations

import heapq
import itertools
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

try:
    import gemmi
    _HAVE_GEMMI = True
except ImportError:
    _HAVE_GEMMI = False

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
        If the ``gemmi`` package is not installed.
    """
    if not _HAVE_GEMMI:
        raise ImportError("gemmi is required to read MRC maps: pip install gemmi")

    m = gemmi.read_ccp4_map(path)

    # Axis mapping (header words 17-19): MAPC, MAPR, MAPS — the Cartesian
    # axis (1=x, 2=y, 3=z) of the file's columns, rows and sections
    mapc = m.header_i32(17)
    mapr = m.header_i32(18)
    maps = m.header_i32(19)

    # Box size and start (words 1-3, 5-7) are in file axis order
    n_xyz = np.zeros(3, dtype=np.float64)
    start_xyz = np.zeros(3, dtype=np.float64)
    for word, cart in enumerate((mapc, mapr, maps)):
        n_xyz[cart - 1] = m.header_i32(1 + word)
        start_xyz[cart - 1] = m.header_i32(5 + word)

    # Voxel size = cell length / sampling MX, MY, MZ (words 8-10, along x, y,
    # z).  The box (NC, NR, NS) may cover only part of the cell.
    cell = np.array([m.header_float(11), m.header_float(12), m.header_float(13)],
                    dtype=np.float64)
    sampling = np.array([m.header_i32(8), m.header_i32(9), m.header_i32(10)],
                        dtype=np.float64)
    sampling = np.where(sampling > 0, sampling, n_xyz)
    voxel_size = cell / sampling

    # MRC2000 origin (header words 50-52)
    origin = np.array(
        [m.header_float(50), m.header_float(51), m.header_float(52)],
        dtype=np.float64,
    )

    # Fall back to nxstart / nystart / nzstart (words 5-7) for old CCP4 files
    if np.all(origin == 0.0) and voxel_size[0] > 0:
        origin = start_xyz * voxel_size

    # gemmi grid shape = (nu, nv, nw) = (NC, NR, NS) in Fortran order; its
    # transpose (NS, NR, NC) is a C-ordered view of the same memory, which
    # we reorder from file order to ZYX.  No copy for the usual x, y, z order.
    data = np.asarray(m.grid).transpose(2, 1, 0)

    axis_to_cart = {0: maps, 1: mapr, 2: mapc}
    cart_to_axis = {v: k for k, v in axis_to_cart.items()}
    transpose_order = (cart_to_axis[3], cart_to_axis[2], cart_to_axis[1])
    data = np.ascontiguousarray(np.transpose(data, transpose_order), dtype=np.float32)

    return data, origin, voxel_size


# ---------------------------------------------------------------------------
# Structure I/O
# ---------------------------------------------------------------------------

def _gemmi_model_to_atom_records(model) -> List[AtomRecord]:
    """Convert atoms from the first model of a gemmi.Structure to AtomRecords.

    All alternate conformations are included (matching biotite ``altloc="all"``).
    """
    records: List[AtomRecord] = []
    append = records.append
    for chain in model:
        chain_id = chain.name
        for res in chain:
            seqid = res.seqid
            res_seq = seqid.num
            ins = seqid.icode.strip() if seqid.icode not in (' ', '\x00') else ""
            res_name = res.name
            for atom in res:
                el = atom.element.name.strip().upper()
                if not el:
                    an = atom.name.strip()
                    el = an.lstrip("0123456789").lstrip()[0].upper() if an else ""
                alt = atom.altloc if atom.altloc not in ('\x00', ' ', '') else ""
                pos = atom.pos
                append(AtomRecord(chain_id, res_seq, ins, res_name, atom.name, el,
                                  pos.x, pos.y, pos.z, alt))
    return records


def read_pdb(path: str) -> List[AtomRecord]:
    """Parse a PDB file and return ATOM / HETATM records (first model only)."""
    if not _HAVE_GEMMI:
        raise ImportError("gemmi is required for structure I/O: pip install gemmi")
    st = gemmi.read_structure(path)
    return _gemmi_model_to_atom_records(st[0])


# Categories gemmi turns into helices, sheets and connections, none of which
# Q-scores need.  For an entry of thousands of chains they take most of the
# time to build the structure (13 of 13.2 s for 400k atoms of 36ZA).
_UNUSED_CIF_CATEGORIES = ("_struct_conf.", "_struct_sheet_range.", "_struct_sheet_order.",
                          "_pdbx_struct_sheet_hbond.", "_struct_conn.")


def read_cif(path: str) -> List[AtomRecord]:
    """Parse an mmCIF file and return atom_site records (first model only)."""
    if not _HAVE_GEMMI:
        raise ImportError("gemmi is required for structure I/O: pip install gemmi")
    # What gemmi.read_structure does for mmCIF, without the unused categories
    doc = gemmi.cif.read(path)
    block = doc[0]
    for name in _UNUSED_CIF_CATEGORIES:
        table = block.find_mmcif_category(name)
        if len(table):
            table.erase()
    st = gemmi.make_structure_from_block(block)
    del table, block, doc       # the parsed text: several GB for the largest entries
    st.merge_chain_parts()
    return _gemmi_model_to_atom_records(st[0])


_UNKNOWN_SEQ = -(2 ** 62)      # residue number of a non-numeric seq_id, as one value


class AtomSiteColumns:
    """The ``_atom_site`` columns of a gemmi mmCIF block, each read once, when first asked for.

    ``codes(*tags)``: for the first of *tags* present, its distinct values
    (unquoted, "?" and "." kept as they are) and each row's index into them;
    None if none is present.  ``floats(tag)``: the column as float64, or None.

    :func:`block_qscores` takes any object with these two methods, so a
    caller that reads the same columns can share them.
    """

    def __init__(self, block):
        self.block = block
        self._codes: dict = {}
        self._floats: dict = {}

    def _find(self, tags):
        for tag in tags:
            values = self.block.find_values("_atom_site." + tag)
            if values:
                return tag, values
        return None, None

    def codes(self, *tags):
        tag, values = self._find(tags)
        if tag is None:
            return None
        if tag not in self._codes:
            code, raw = _factorize(values, len(values))
            text = [v if v in ("?", ".") else gemmi.cif.as_string(v) for v in raw]
            distinct = list(dict.fromkeys(text))        # quoted and bare spellings as one
            if len(distinct) < len(text):
                index = {t: i for i, t in enumerate(distinct)}
                code = np.array([index[t] for t in text], dtype=np.int64)[code]
            self._codes[tag] = (distinct, code)
        return self._codes[tag]

    def floats(self, tag):
        if tag not in self._floats:
            values = self.block.find_values("_atom_site." + tag)
            self._floats[tag] = (np.fromiter(values, dtype=np.float64, count=len(values))
                                 if values else None)
        return self._floats[tag]


def _atom_site_arrays(block, columns=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The first model's atoms of a gemmi mmCIF block, as :func:`read_cif` reads them, as arrays.

    Returns ``(rows, coords, heavy, residue)``: each atom's row in the
    ``_atom_site`` table, its position, whether it is not hydrogen, and its
    residue code (see :func:`_residue_codes`).  The order is gemmi's: a
    model's atoms chain by chain, chains in order of first appearance, file
    order within a chain.  No structure or per-atom objects are built.
    *columns*: see :class:`AtomSiteColumns`.
    """
    if columns is None:
        columns = AtomSiteColumns(block)
    x = columns.floats("Cartn_x")
    if x is None:
        raise ValueError("No _atom_site table")
    n_total = len(x)
    rows = np.arange(n_total)
    model = columns.codes("pdbx_PDB_model_num")
    if model is not None and len(model[0]) > 1:     # gemmi's first model is that of the first row
        rows = rows[model[1] == model[1][0]]
    if len(rows) == 0:
        return rows, np.empty((0, 3)), np.empty(0, dtype=bool), np.empty(0, dtype=np.int64)

    # Chains in order of first appearance, the file's order within each
    chain_names, chain = columns.codes("auth_asym_id", "label_asym_id")
    if len(rows) < n_total:
        chain = np.unique(chain[rows], return_inverse=True)[1]
        first = np.full(chain.max() + 1, len(rows))
        np.minimum.at(first, chain, np.arange(len(rows)))
        rank = np.argsort(np.argsort(first, kind="stable"), kind="stable")
        chain = rank[chain]
    if not np.all(chain[1:] >= chain[:-1]):
        order = np.argsort(chain, kind="stable")
        rows, chain = rows[order], chain[order]

    idx = slice(None) if len(rows) == n_total and np.all(rows == np.arange(n_total)) else rows
    coords = np.column_stack([columns.floats("Cartn_" + c)[idx] for c in "xyz"])

    symbols = columns.codes("type_symbol")
    if symbols is not None:
        distinct, code = symbols
        heavy = np.array([gemmi.Element(v).name.upper() != "H" for v in distinct], dtype=bool)[code[idx]]
    else:   # (read_cif fails here) the element from the name, as for a blank type_symbol
        distinct, code = columns.codes("label_atom_id", "auth_atom_id")
        heavy = np.array([not v.lstrip("0123456789").upper().startswith("H") for v in distinct],
                         dtype=bool)[code[idx]]

    # Residue: chain, number, insertion code (as AtomRecord's chain_id, res_seq, ins_code)
    distinct, code = columns.codes("auth_seq_id", "label_seq_id")
    numbers = np.array([int(v) if v.lstrip("+-").isdigit() else _UNKNOWN_SEQ for v in distinct],
                       dtype=np.int64)
    _, num_code = np.unique(numbers, return_inverse=True)
    number = num_code[code[idx]]
    residue = chain * (len(num_code) + 1) + number
    ins_values = columns.codes("pdbx_PDB_ins_code")
    if ins_values is not None:
        distinct, code = ins_values
        ins_uniq, ins_code = np.unique([v if v not in ("?", ".", " ", "") else "" for v in distinct],
                                       return_inverse=True)
        if len(ins_uniq) > 1:
            residue = residue * len(ins_uniq) + ins_code[code[idx]]
    return rows, coords, heavy, residue


def block_qscores(block, data: np.ndarray, origin: np.ndarray, voxel_size: np.ndarray,
                  columns=None, **options) -> np.ndarray:
    """Q-score of each ``_atom_site`` row of an mmCIF block read with ``gemmi.cif.read``.

    The same Q-scores as ``compute_qscores(read_cif(path), …)`` for the
    file, without building a structure: for a caller that already has the
    file parsed, and to write the scores into the same document with
    :func:`add_qscore_column`.  Rows of models after the first are NaN.
    *columns*: the block's columns, if the caller has read them already
    (see :class:`AtomSiteColumns`).  *options*: as for :func:`compute_qscores`.
    """
    rows, coords, heavy, residue = _atom_site_arrays(block, columns)
    q_scores, _, _ = qscores_from_arrays(coords, heavy, residue, data, origin, voxel_size, **options)
    by_row = np.full(len(block.find_values("_atom_site.Cartn_x")), np.nan)
    by_row[rows] = q_scores
    return by_row


def add_qscore_column(block, q_scores: np.ndarray) -> None:
    """Add an ``_atom_site.Q-score`` column to a gemmi mmCIF block, as MapQ writes it.

    One value per ``_atom_site`` row, with six decimals; "?" for NaN.
    """
    loop = block.find_loop("_atom_site.Cartn_x").get_loop()
    if loop.length() != len(q_scores):
        raise ValueError(f"{len(q_scores)} Q-scores for {loop.length()} atom_site rows")
    loop.add_columns(["_atom_site.Q-score"], "?")
    column = block.find_values("_atom_site.Q-score")
    for row, q in enumerate(np.asarray(q_scores, dtype=np.float64).tolist()):
        if q == q:      # not NaN
            column[row] = f"{q:.6f}"


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def min_max_d(data: np.ndarray) -> Tuple[float, float]:
    """Compute reference Gaussian offset B (minD) and max D (maxD).

        maxD = min(mean + 10 * std, max_val)
        minD = max(mean -  1 * std, min_val)

    Note: the Pearson correlation is invariant to affine rescaling of the
    reference Gaussian, so these values do not affect Q-scores themselves.

    *data* may be a numpy or a JAX array; the statistics are computed on the
    JAX default device.
    """
    _require_jax()
    mean, std, min_val, max_val = (float(v) for v in jax.device_get(_map_stats(jnp.asarray(data))))
    max_d = min(mean + 10.0 * std, max_val)
    min_d = max(mean -  1.0 * std, min_val)
    return min_d, max_d


def _map_stats(data):
    """(mean, std, min, max) of a map as float32 scalars, on the JAX device."""
    def total(x):
        # One axis at a time: float32 partial sums stay accurate on every
        # backend, including a sequential CPU reduction over 1e9 voxels
        for _ in range(x.ndim):
            x = jnp.sum(x, axis=-1)
        return x

    n = data.size
    mean = total(data) / n
    std = jnp.sqrt(total(jnp.square(data - mean)) / n)
    return mean, std, jnp.min(data), jnp.max(data)


if _HAVE_JAX:
    _map_stats = jit(_map_stats)


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


# Maximum X-H distance (Å) to assign H the heavy atom's Q-score (C/N/O-H bonds
# ~1.0–1.1, S-H 1.34)
_MAX_H_HEAVY_DIST = 1.4


def _assign_hydrogen_qscores(
    residue: np.ndarray,
    all_coords: np.ndarray,
    non_h_mask: np.ndarray,
    non_h_indices: np.ndarray,
    q_scores: np.ndarray,
) -> None:
    """Set each hydrogen's Q-score to the Q-score of its nearest heavy atom in the same residue (in-place).

    *residue*: residue code of each atom (see :func:`_residue_codes`).
    """
    h_indices = np.flatnonzero(~non_h_mask)
    n_heavy = len(non_h_indices)
    if len(h_indices) == 0 or n_heavy == 0:
        return

    # The nearest heavy atoms within the bond cutoff, closest first; the
    # first of them in the H's residue is the residue's nearest heavy atom
    k = min(8, n_heavy)
    tree = cKDTree(all_coords[non_h_indices])
    _, nearest = tree.query(all_coords[h_indices], k=k, workers=-1,
                            distance_upper_bound=np.nextafter(_MAX_H_HEAVY_DIST, np.inf))
    nearest = nearest.reshape(len(h_indices), k)
    heavy = non_h_indices[np.minimum(nearest, n_heavy - 1)]
    same = (nearest < n_heavy) & (residue[heavy] == residue[h_indices, None])
    has = same.any(axis=1)
    pick = heavy[has, same[has].argmax(axis=1)]
    q_scores[h_indices[has]] = q_scores[pick]


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _factorize(keys, n: int) -> Tuple[np.ndarray, list]:
    """Integer codes for *n* hashable keys, and the distinct keys in order of first appearance."""
    table: dict = defaultdict(itertools.count().__next__)
    codes = np.fromiter(map(table.__getitem__, keys), dtype=np.int64, count=n)
    return codes, list(table)


def _residue_codes(atoms: List[AtomRecord]) -> np.ndarray:
    """Residue code of each atom: equal for atoms of one residue (chain, number, insertion code)."""
    return _factorize(((a.chain_id, a.res_seq, a.ins_code) for a in atoms), len(atoms))[0]


def _group_means(codes: np.ndarray, values: np.ndarray, n_groups: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-group mean (NaN for empty groups) and count of *values*."""
    counts = np.bincount(codes, minlength=n_groups)
    sums = np.bincount(codes, weights=values, minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / counts, counts


_BACKBONE = frozenset({"N", "CA", "C", "O", "OXT"})


def residue_qscores(result: QScoreResult) -> Dict[Tuple, Dict[str, float]]:
    """Aggregate per-atom Q-scores to residue level (includes H via heavy-atom assignment).

    Returns
    -------
    dict
        Key: (chain_id, res_seq, ins_code, res_name)
        Value: dict with keys "all", "backbone", "sidechain"
    """
    atoms = result.atom_records
    q = np.asarray(result.atom_qscores, dtype=np.float64)
    residue, keys = _factorize(((a.chain_id, a.res_seq, a.ins_code, a.res_name) for a in atoms),
                               len(atoms))
    backbone = np.fromiter((a.atom_name in _BACKBONE for a in atoms), dtype=bool, count=len(atoms))
    ok = ~np.isnan(q)
    sel_bb, sel_sc = ok & backbone, ok & ~backbone

    all_q, n_q = _group_means(residue[ok], q[ok], len(keys))
    bb_q, _ = _group_means(residue[sel_bb], q[sel_bb], len(keys))
    sc_q, _ = _group_means(residue[sel_sc], q[sel_sc], len(keys))
    return {
        key: {"all": float(all_q[i]), "backbone": float(bb_q[i]), "sidechain": float(sc_q[i])}
        for i, key in enumerate(keys) if n_q[i]
    }


def chain_qscores(result: QScoreResult) -> Dict[str, float]:
    """Average Q-score per chain (all atoms with a Q-score, including H from heavy-atom assignment)."""
    atoms = result.atom_records
    q = np.asarray(result.atom_qscores, dtype=np.float64)
    chain, names = _factorize((a.chain_id for a in atoms), len(atoms))
    ok = ~np.isnan(q)
    mean_q, n_q = _group_means(chain[ok], q[ok], len(names))
    return {ch: float(mean_q[i]) for i, ch in enumerate(names) if n_q[i]}


# ---------------------------------------------------------------------------
# JAX helpers
# ---------------------------------------------------------------------------

def _require_jax() -> None:
    if not _HAVE_JAX:
        raise ImportError(
            "JAX is required.  Install with:\n"
            "  CPU only : pip install jax\n"
            "  CUDA GPU : pip install \"jqscore[gpu]\"  (or jax[cuda12] for NVIDIA drivers < 580)"
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
    return _trilinear_grid(data, (pts - origin) / voxel_size)


def _trilinear_grid(data, grid_pts):
    """Trilinear map interpolation at (N, 3) points in voxel units (x, y, z) from the origin."""
    nz, ny, nx = data.shape

    gx = grid_pts[:, 0]
    gy = grid_pts[:, 1]
    gz = grid_pts[:, 2]

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

def _closest_dsq(dirs, r, nbr_rel):
    """Squared distance from each of (P, 3) points ``dirs * r`` to the closest of (K, 3) neighbours.

    Both are relative to the atom, so float32 keeps ~1e-7 Å even for models
    with coordinates of thousands of Å, where absolute float32 coordinates
    carry ~1e-4 Å of rounding — enough to flip points at the 0.9 r cutoff.

    A running minimum over the neighbours, in a loop: device memory and
    compile time stay independent of K and of the batch shape (a jnp.min
    reduction over K, or a fully unrolled minimum, takes minutes to compile
    or gigabytes of temporaries for some shapes).  Multiplying ``dirs * r``
    inside the loop body avoids materializing a ``(B, P, 3)`` offset tensor
    in global device memory across loop iterations.
    """
    def _closer(j, min_dsq):
        nb = nbr_rel[j]
        return jnp.minimum(min_dsq, jnp.square(dirs[:, 0] * r - nb[0])
                                  + jnp.square(dirs[:, 1] * r - nb[1])
                                  + jnp.square(dirs[:, 2] * r - nb[2]))
    return lax.fori_loop(0, nbr_rel.shape[0], _closer,
                         jnp.full(dirs.shape[0], jnp.inf, dtype=jnp.float32),
                         unroll=_NBR_UNROLL)


# Neighbours per iteration of the closest-neighbour loop (K is a multiple of 8)
_NBR_UNROLL = 8


def _make_compute_fn(n_gen: int, num_pts: int):
    """Build a JIT+vmap compiled function (fixed sphere-point budget).

    Signature
    ---------
    fn(atom_grid, nbr_rel, map_data, voxel_size,
       sphere_dirs, shell_radii, gv_shells, gv_r0)
    → q_scores : (N,) float32

    ``atom_grid``: (N, 3) atom positions in voxel units from the map origin;
    ``nbr_rel``: (N, K, 3) neighbour positions relative to the atom, in Å
    (padding slots far away).
    """
    def _qscore_one(atom_grid, nbr_rel, data, voxel_size,
                    sphere_dirs, shell_radii, gv_shells, gv_r0):

        val_r0  = _trilinear_grid(data, atom_grid[None, :])[0]
        d_r0    = jnp.full((n_gen,), val_r0, dtype=jnp.float32)
        g_r0    = jnp.full((n_gen,), gv_r0, dtype=jnp.float32)
        mask_r0 = jnp.ones((n_gen,), dtype=jnp.bool_)

        def _shell(_, xs):
            r, gv = xs

            out_r  = r * jnp.float32(0.9)
            sph_ok = _closest_dsq(sphere_dirs, r, nbr_rel) > out_r * out_r

            n_ok   = jnp.sum(sph_ok.astype(jnp.int32))
            sph_ok = jnp.where(n_ok >= num_pts, sph_ok, jnp.zeros_like(sph_ok))

            d_vals = _trilinear_grid(data, atom_grid + (sphere_dirs * r) / voxel_size)
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
        in_axes=(0, 0, None, None, None, None, None, None),
    )
    return jit(_batched)


_compiled_cache: dict = {}


def _get_compute_fn(n_gen: int, num_pts: int):
    key = (n_gen, num_pts)
    if key not in _compiled_cache:
        _compiled_cache[key] = _make_compute_fn(n_gen, num_pts)
    return _compiled_cache[key]


# ---------------------------------------------------------------------------
# Compiled kernels — retry mode
# ---------------------------------------------------------------------------
#
# Attempt k of the retry loop puts num_pts + 2k points on the shell and
# succeeds when at least num_pts of them are clear of other atoms.  Most
# (atom, shell) pairs succeed within a few attempts; a few need all of them.
# A while_loop per pair under vmap would run every pair of a batch for as
# many attempts as its slowest pair, so the attempts are checked in rounds
# instead: all pairs take the first round, and only the pairs still
# unresolved take the next one.

# Attempt index at which each round but the last ends; the last round runs
# to max_retries
_RETRY_ROUND_ENDS = (4, 12, 24)
# Tile sizes for later rounds: active pending pairs are processed in fixed-size
# tiles up to n_pending so unused capacity slots are skipped and per-tile
# working sets stay small
_RETRY_TILE_SIZES = (16384, 8192, 2048)


class _RetryRound(NamedTuple):
    dirs:   np.ndarray   # (P, 3) float32 — sphere directions of the round's attempts, concatenated
    starts: np.ndarray   # (A,) int32 — row of each attempt's first direction in dirs
    sizes:  np.ndarray   # (A,) int32 — number of points of each attempt


def _retry_rounds(num_pts: int, max_retries: int) -> List[_RetryRound]:
    """Split attempts 0 … max_retries-1 into the rounds of :data:`_RETRY_ROUND_ENDS`."""
    ends = [e for e in _RETRY_ROUND_ENDS if e < max_retries] + [max_retries]
    rounds = []
    first = 0
    for end in ends:
        sizes = num_pts + 2 * np.arange(first, end)
        rounds.append(_RetryRound(
            dirs=np.concatenate([_sphere_directions(int(n)) for n in sizes]).astype(np.float32),
            starts=np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int32),
            sizes=sizes.astype(np.int32),
        ))
        first = end
    return rounds


def _make_shell_fn(rnd: _RetryRound, num_pts: int):
    """Build the per-(atom, shell) function for one round of attempts.

    Signature
    ---------
    fn(atom_grid (3,), nbr_rel (K, 3), centre_val, r, map_data, voxel_size)
    → (found, n, s1, s2)

    *atom_grid*: the atom's position in voxel units from the map origin;
    *nbr_rel*: neighbour positions relative to the atom, in Å.
    *found*: whether an attempt of this round has ≥ num_pts clear points.
    For the first such attempt, *n* is its number of clear points and *s1*,
    *s2* are the sum and sum of squares of (density − centre_val) over them;
    all three are 0 when nothing was found.
    """
    dirs    = jnp.asarray(rnd.dirs)
    starts  = jnp.asarray(rnd.starts)
    sizes   = jnp.asarray(rnd.sizes)
    n_dirs  = len(rnd.dirs)
    slots   = jnp.asarray(np.arange(int(rnd.sizes.max()), dtype=np.int32))

    # member[p, a] = 1 if direction p belongs to attempt a: clear points per
    # attempt as one small matmul (exact for 0/1 values; a cumsum over P
    # takes seconds to compile for the larger rounds)
    member = np.zeros((n_dirs, len(rnd.sizes)), dtype=np.float32)
    for a, (start, size) in enumerate(zip(rnd.starts, rnd.sizes)):
        member[start:start + size, a] = 1.0
    member = jnp.asarray(member)

    def _shell(atom_grid, nbr_rel, centre_val, r, data, voxel_size):
        out_r = r * jnp.float32(0.9)
        clear = _closest_dsq(dirs, r, nbr_rel) > out_r * out_r          # (P,)

        # Clear points per attempt, and the first attempt with enough
        n_clear = jnp.dot(clear.astype(jnp.float32), member, precision=lax.Precision.HIGHEST)
        enough = n_clear >= num_pts                                     # (A,)
        found = jnp.any(enough)
        k = jnp.argmax(enough)

        rows = jnp.minimum(starts[k] + slots, n_dirs - 1)
        use = found & (slots < sizes[k]) & clear[rows]
        dev = _trilinear_grid(data, atom_grid + (dirs[rows] * r) / voxel_size) - centre_val
        dev = jnp.where(use, dev, jnp.float32(0.0))
        return found, jnp.sum(use, dtype=jnp.int32), jnp.sum(dev), jnp.sum(dev * dev)

    return _shell


def _relative_coords(coords_hi, coords_lo, atom_idx, nbr_idx):
    """Neighbour positions relative to their atom, (…, K, 3) float32.

    Coordinates come as float32 pairs, ``hi = float32(x)`` and
    ``lo = float32(x - hi)``: ``hi_nbr - hi_atom`` is exact for nearby atoms,
    so the difference keeps the float64 coordinates' precision.
    """
    return ((coords_hi[nbr_idx] - coords_hi[atom_idx][..., None, :])
            + (coords_lo[nbr_idx] - coords_lo[atom_idx][..., None, :]))


def _make_retry_fns(num_pts: int, max_retries: int):
    """Build the functions for retry mode (JAX-traceable, not yet compiled).

    Returns ``(first_round, later_rounds)``.

    ``first_round(atom_idx (B,), nbr_idx (B, K), atom_grid (B, 3),
    coords_hi, coords_lo, shell_radii (S,), map_data, voxel_size)``
        → ``(nbr_rel (B, K, 3), centre_vals (B,), found, n, s1, s2)``, each
        of the last four (B, S) — every (atom, shell) pair of the batch.

    ``later_rounds[i](found, n, s1, s2, atom_grid, nbr_rel, centre_vals,
    shell_radii, map_data, voxel_size)``
        → ``(found, n, s1, s2)``: round i+2 for the pairs still unresolved,
        gathered on the device in fixed-size tiles up to ``n_pending``.

    ``atom_idx`` and ``nbr_idx`` index rows of ``coords_hi`` / ``coords_lo``
    (see :func:`_relative_coords`); padding slots point at a row far from
    every atom.  ``atom_grid``: the atoms' positions in voxel units from the
    map origin.
    """
    shell_fns = [_make_shell_fn(rnd, num_pts) for rnd in _retry_rounds(num_pts, max_retries)]

    per_atom = vmap(shell_fns[0], in_axes=(None, None, None, 0, None, None))
    per_batch = vmap(per_atom, in_axes=(0, 0, 0, None, None, None))

    def first_round(atom_idx, nbr_idx, atom_grid, coords_hi, coords_lo,
                    shell_radii, data, voxel_size):
        nbr_rel = _relative_coords(coords_hi, coords_lo, atom_idx, nbr_idx)
        centre_vals = _trilinear_grid(data, atom_grid)
        return (nbr_rel, centre_vals) + per_batch(atom_grid, nbr_rel, centre_vals,
                                                  shell_radii, data, voxel_size)

    def _later(shell_fn, tile_target: int):
        per_pair = vmap(shell_fn, in_axes=(0, 0, 0, 0, None, None))

        def later_round(found, n, s1, s2, atom_grid, nbr_rel, centre_vals,
                        shell_radii, data, voxel_size):
            shape = found.shape
            n_pairs, n_shells = found.size, shape[1]
            found, n, s1, s2 = found.ravel(), n.ravel(), s1.ravel(), s2.ravel()
            pending = ~found
            n_pending = jnp.sum(pending, dtype=jnp.int32)
            tile_size = min(n_pairs, tile_target)
            n_max_tiles = -(-n_pairs // tile_size)
            (pairs,) = jnp.nonzero(pending, size=n_max_tiles * tile_size, fill_value=n_pairs)
            pairs_2d = pairs.reshape(n_max_tiles, tile_size)
            n_tiles = (n_pending + (tile_size - 1)) // tile_size

            def body(t, state):
                f_arr, n_arr, s1_arr, s2_arr = state
                p_tile = pairs_2d[t]
                atom_i, shell_i = jnp.divmod(jnp.minimum(p_tile, n_pairs - 1), n_shells)
                f, nn, a, b = per_pair(atom_grid[atom_i], nbr_rel[atom_i], centre_vals[atom_i],
                                       shell_radii[shell_i], data, voxel_size)
                hit = jnp.where(f & (p_tile < n_pairs), p_tile, n_pairs)    # n_pairs: dropped
                f_arr = f_arr.at[hit].set(True, mode="drop")
                n_arr = n_arr.at[hit].set(nn, mode="drop")
                s1_arr = s1_arr.at[hit].set(a, mode="drop")
                s2_arr = s2_arr.at[hit].set(b, mode="drop")
                return f_arr, n_arr, s1_arr, s2_arr

            found, n, s1, s2 = lax.fori_loop(0, n_tiles, body, (found, n, s1, s2))
            return found.reshape(shape), n.reshape(shape), s1.reshape(shape), s2.reshape(shape)

        return later_round

    laters = [_later(fn, _RETRY_TILE_SIZES[i] if i < len(_RETRY_TILE_SIZES) else 2048)
              for i, fn in enumerate(shell_fns[1:])]
    return first_round, laters


if _HAVE_JAX:
    _relative_coords_jit = jit(_relative_coords)

_compiled_cache_retry: dict = {}


def _get_retry_fns(num_pts: int, max_retries: int):
    key = (num_pts, max_retries)
    if key not in _compiled_cache_retry:
        first_round, later_rounds = _make_retry_fns(num_pts, max_retries)
        _compiled_cache_retry[key] = (jit(first_round), [jit(f) for f in later_rounds])
    return _compiled_cache_retry[key]


class _RetryRunner:
    """Retry mode for the batches of one run: compiled rounds for each batch shape."""

    def __init__(self, num_pts: int, max_retries: int, shapes, coords_hi, coords_lo,
                 shell_radii, data, voxel_size, cell_list: Optional[_GpuCellList] = None):
        self.first_round, self.later_rounds = _get_retry_fns(num_pts, max_retries)
        self.args = (coords_hi, coords_lo, shell_radii, data, voxel_size)
        self.cell_list = cell_list
        self.n_shells = shell_radii.shape[0]
        self.compiled = {}
        r_list = (["nbr"] if cell_list is not None else []) + [None] + list(range(len(self.later_rounds)))
        self._compile([(shape, r) for shape in shapes for r in r_list])

    def _compile(self, jobs) -> None:
        """Compile the first round (r None) and later rounds r for batch shapes, in parallel.

        XLA compiles outside the GIL; the compilations take seconds, longer
        than the computation itself for a structure of a few thousand atoms.
        """
        if not jobs:
            return
        coords_hi, coords_lo, shell_radii, data, voxel_size = self.args

        def compile_one(job):
            (size, k), r = job
            sds = jax.ShapeDtypeStruct
            grid = sds((size, 3), jnp.float32)
            if r == "nbr":
                cl = self.cell_list
                exe = cl.nbr_fn.lower(
                    sds((size,), jnp.int32), sds((size,), jnp.int32),
                    cl.j_cell_edge, cl.j_xy_offsets,
                    coords_hi, coords_lo, cl.j_rad_sq, k).compile()
            elif r is None:
                exe = self.first_round.lower(sds((size,), jnp.int32), sds((size, k), jnp.int32),
                                             grid, *self.args).compile()
            else:
                pairs = (size, self.n_shells)
                exe = self.later_rounds[r].lower(
                    sds(pairs, jnp.bool_), sds(pairs, jnp.int32), sds(pairs, jnp.float32),
                    sds(pairs, jnp.float32), grid, sds((size, k, 3), jnp.float32),
                    sds((size,), jnp.float32), shell_radii, data, voxel_size).compile()
            return job, exe

        with ThreadPoolExecutor(max_workers=min(len(jobs), 8)) as pool:
            for job, exe in pool.map(compile_one, jobs):
                self.compiled[job] = exe

    def launch(self, shape, atom_idx, nbr_or_cell, atom_grid):
        """Dispatch all rounds of a batch (asynchronous: no host round trip between them)."""
        coords_hi, coords_lo, shell_radii, data, voxel_size = self.args
        if self.cell_list is not None:
            cl = self.cell_list
            nbr_idx = self.compiled[shape, "nbr"](
                atom_idx, nbr_or_cell, cl.j_cell_edge, cl.j_xy_offsets,
                coords_hi, coords_lo, cl.j_rad_sq)
        else:
            nbr_idx = nbr_or_cell
        nbr_rel, centre_vals, found, n, s1, s2 = self.compiled[shape, None](
            atom_idx, nbr_idx, atom_grid, *self.args)
        for r in range(len(self.later_rounds)):
            found, n, s1, s2 = self.compiled[shape, r](
                found, n, s1, s2, atom_grid, nbr_rel, centre_vals, shell_radii, data, voxel_size)
        return n, s1, s2

    @staticmethod
    def sums(launched) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Host (n, s1, s2) of a launched batch."""
        n_host, s1_host, s2_host = jax.device_get(launched)
        return (np.asarray(n_host, dtype=np.float64), np.asarray(s1_host, dtype=np.float64),
                np.asarray(s2_host, dtype=np.float64))


def _pearson_from_shell_sums(n: np.ndarray, s1: np.ndarray, s2: np.ndarray,
                             g_shells: np.ndarray, g_r0: float, num_pts: int) -> np.ndarray:
    """Pearson r per atom over the centre sample (num_pts copies) and the shells' samples.

    *n*, *s1*, *s2*: (B, S) count, sum and sum of squares of the shells'
    densities relative to the atom's centre density, so the centre samples
    are all 0.  The Gaussian is taken relative to its centre value likewise.
    """
    h = np.asarray(g_shells, dtype=np.float64) - g_r0              # (S,)
    m = np.column_stack([np.ones_like(h), h, h * h])               # (S, 3): 2D BLAS dgemm
    n_m = np.asarray(n, dtype=np.float64) @ m                      # (B, 3)
    s1_m = np.asarray(s1, dtype=np.float64) @ m[:, :2]             # (B, 2)
    s2_m = (np.asarray(s2, dtype=np.float64) @ m[:, :1]).ravel()   # (B,)
    total = num_pts + n_m[:, 0]
    sum_d, s1_h = s1_m[:, 0], s1_m[:, 1]
    sum_h, sum_hh = n_m[:, 1], n_m[:, 2]
    cov = s1_h - sum_d * sum_h / total
    var_d = s2_m - sum_d * sum_d / total
    var_h = sum_hh - sum_h * sum_h / total
    denom_sq = var_d * var_h
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(denom_sq > 0.0, cov / np.sqrt(np.maximum(denom_sq, 1e-300)), 0.0)
    return np.clip(q, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Neighbour lookup (GPU 3D cell-list and CPU cKDTree)
# ---------------------------------------------------------------------------

_gpu_nbr_fn_cache: dict = {}


def _get_gpu_nbr_fn(col_cap: int):
    """Return JIT-compiled GPU 3D cell-list neighbour count and compaction kernels.

    Gathers up to *col_cap* candidates from each of the 9 contiguous 3-cell
    Z-columns around an atom's cell, tests distances, and either counts hits
    or compacts up to *k_slots* neighbour indices with ``jnp.nonzero``.
    """
    if col_cap not in _gpu_nbr_fn_cache:
        slots = jnp.arange(col_cap, dtype=jnp.int32)[None, :]

        def _hits(a_idx, cid, cell_edge, xy_offsets, coords_hi, coords_lo, radius_sq):
            n_pad = jnp.int32(coords_hi.shape[0] - 1)
            bases = cid + xy_offsets
            starts = cell_edge[bases - 1][:, None]
            ends = cell_edge[bases + 2][:, None]
            cand = jnp.where(starts + slots < ends, starts + slots, n_pad).ravel()
            rel = ((coords_hi[cand] - coords_hi[a_idx])
                   + (coords_lo[cand] - coords_lo[a_idx]))
            return cand, jnp.sum(jnp.square(rel), axis=1) <= radius_sq, n_pad

        def count_nbrs(atom_idx, atom_cell, cell_edge, xy_offsets,
                       coords_hi, coords_lo, radius_sq):
            def count_one(a_idx, cid):
                _, hit, _ = _hits(a_idx, cid, cell_edge, xy_offsets, coords_hi, coords_lo, radius_sq)
                return jnp.sum(hit, dtype=jnp.int32)
            return vmap(count_one)(atom_idx, atom_cell)

        def find_nbrs(atom_idx, atom_cell, cell_edge, xy_offsets,
                      coords_hi, coords_lo, radius_sq, k_slots):
            def find_one(a_idx, cid):
                cand, hit, n_pad = _hits(a_idx, cid, cell_edge, xy_offsets, coords_hi, coords_lo, radius_sq)
                (idx,) = jnp.nonzero(hit, size=k_slots, fill_value=cand.size)
                return jnp.where(idx < cand.size, cand[jnp.minimum(idx, cand.size - 1)], n_pad)
            return vmap(find_one)(atom_idx, atom_cell)

        _gpu_nbr_fn_cache[col_cap] = (jit(count_nbrs), jit(find_nbrs, static_argnums=(7,)))
    return _gpu_nbr_fn_cache[col_cap]


class _GpuCellList:
    """3D uniform grid cell-list for pure-JAX GPU neighbour search."""

    def __init__(self, order: np.ndarray, sorted_coords: np.ndarray,
                 sorted_cell_id: np.ndarray, cell_edge: np.ndarray,
                 xy_offsets: np.ndarray, empty_cell: int,
                 radius: float, col_cap: int):
        self.order = order
        self.sorted_coords = sorted_coords
        self.sorted_cell_id = sorted_cell_id
        self.empty_cell = np.int32(empty_cell)
        self.col_cap = col_cap
        self.count_fn, self.nbr_fn = _get_gpu_nbr_fn(col_cap)
        self.j_cell_edge = jnp.asarray(cell_edge)
        self.j_xy_offsets = jnp.asarray(xy_offsets)
        self.j_rad_sq = jnp.float32(radius * radius)

    def counts(self, jcoords_hi, jcoords_lo, chunk_size: int = 65536) -> np.ndarray:
        """Exact neighbour count for each atom in ``sorted_coords``, computed on the device."""
        n = len(self.sorted_cell_id)
        step = min(chunk_size, n)
        out = np.empty(n, dtype=np.int32)
        for i in range(0, n, step):
            m = min(step, n - i)
            idx = np.arange(i, i + m, dtype=np.int32)
            cid = self.sorted_cell_id[i:i + m]
            if m < step:
                pad = step - m
                idx = np.concatenate([idx, np.full(pad, n, dtype=np.int32)])
                cid = np.concatenate([cid, np.full(pad, self.empty_cell, dtype=np.int32)])
            res = self.count_fn(jnp.asarray(idx), jnp.asarray(cid), self.j_cell_edge,
                                self.j_xy_offsets, jcoords_hi, jcoords_lo, self.j_rad_sq)
            out[i:i + m] = np.asarray(res)[:m]
        return out

    @classmethod
    def build(cls, coords: np.ndarray, radius: float) -> Optional[_GpuCellList]:
        n = len(coords)
        if n == 0:
            return None
        r_bound = float(np.nextafter(radius, np.inf))
        inv_cell = np.float32(1.0 / (r_bound * 1.0001))
        coords_f32 = coords.astype(np.float32)

        cx = coords_f32[:, 0]
        ix = ((cx - cx.min()) * inv_cell + np.float32(1.0)).astype(np.int32)
        nx = int(ix.max()) + 2

        cy = coords_f32[:, 1]
        iy = ((cy - cy.min()) * inv_cell + np.float32(1.0)).astype(np.int32)
        ny = int(iy.max()) + 2

        cz = coords_f32[:, 2]
        iz = ((cz - cz.min()) * inv_cell + np.float32(1.0)).astype(np.int32)
        nz = int(iz.max()) + 4
        del coords_f32, cx, cy, cz

        n_cells = nx * ny * nz
        if n_cells > 200_000_000:
            return None
        cell_id = (ix * ny + iy) * nz + iz
        del ix, iy, iz
        order = np.argsort(cell_id, kind="stable").astype(np.int32)
        sorted_cell_id = cell_id[order]
        del cell_id

        col_cap = 8
        while col_cap < n and bool(np.any(sorted_cell_id[col_cap:] - sorted_cell_id[:-col_cap] <= 2)):
            col_cap += 8
            if col_cap > 64:
                return None

        counts_cell = np.bincount(sorted_cell_id, minlength=n_cells)
        cell_edge = np.empty(n_cells + 1, dtype=np.int32)
        cell_edge[0] = 0
        np.cumsum(counts_cell, dtype=np.int32, out=cell_edge[1:])
        del counts_cell

        xy_offsets = np.array([(dx * ny + dy) * nz for dx in (-1, 0, 1) for dy in (-1, 0, 1)],
                              dtype=np.int32)
        empty_cell = (1 * ny + 1) * nz + (nz - 2)
        sorted_coords = coords[order]
        return cls(order, sorted_coords, sorted_cell_id, cell_edge, xy_offsets,
                   empty_cell, radius, col_cap)

class _NeighbourIndex:
    """Neighbours within *radius* of query points, among a fixed set of atoms.

    ``query`` returns an index array with K slots per point; padding slots
    hold ``len(coords)``.  Uses all CPU cores.
    """

    def __init__(self, coords: np.ndarray, radius: float):
        self.n = len(coords)
        self.radius = radius
        self.tree = cKDTree(coords)

    def counts(self, pts: np.ndarray) -> np.ndarray:
        return self.tree.query_ball_point(pts, self.radius, return_length=True, workers=-1)

    def query(self, pts: np.ndarray, k: int) -> np.ndarray:
        """Indices of the neighbours of each point; *k* must be ≥ their counts."""
        # query() excludes its bound, query_ball_point() includes it
        _, idx = self.tree.query(pts, k=k, workers=-1,
                                 distance_upper_bound=np.nextafter(self.radius, np.inf))
        return idx.reshape(len(pts), k).astype(np.int32)


def _neighbour_groups(counts: np.ndarray) -> List[Tuple[np.ndarray, int]]:
    """Split atoms by neighbour count into (atom indices, K slots) groups.

    One K for all atoms keeps one set of shapes to compile.  The cost of the
    closest-neighbour loop grows with K, so if a few atoms have far more
    neighbours than the rest (overlapping copies, clashes) they form a group
    of their own.  K is a multiple of 8.
    """
    def slots(c):
        return max(8, -(-int(c) // 8) * 8)

    k_all = slots(counts.max())
    k_typical = slots(np.percentile(counts, 99.9))
    # Below ~8M atom×slot checks (~5 ms on GPU), a second K-group costs more to compile than it saves
    if k_all <= 2 * k_typical and len(counts) * (k_all - k_typical) < 8_000_000:
        return [(np.arange(len(counts)), k_all)]
    many = counts > k_typical
    return [(np.flatnonzero(~many), k_typical), (np.flatnonzero(many), k_all)]


# A coordinate far from every atom, for padding neighbour slots: its squared
# distance to any sphere point stays finite in float32 and exceeds any cutoff
_FAR = 1.0e9


# ---------------------------------------------------------------------------
# MapQ compatibility: replicate MaskMapResize origin rounding
# ---------------------------------------------------------------------------

def _resample_mapq_compat(data, origin: np.ndarray, voxel_size: np.ndarray):
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
    data : (nz, ny, nx) float32, numpy or JAX array
    origin, voxel_size : (3,) float64 — map coordinate system (x, y, z)

    Returns
    -------
    resampled : (nz, ny, nx) float32 JAX array on the default device — blurred map
    new_origin : (3,) float64 — rounded origin to pass to JAX
    """
    # Chimera reads MRC origin/step in float32 — replicate that precision
    O  = origin.astype(np.float32).astype(np.float64)
    vs = voxel_size.astype(np.float32).astype(np.float64)

    # MaskMapResize: ox = round(nO[0]/step)*step  (same fractional offset for all procs)
    new_origin = np.round(O / vs) * vs  # (3,) float64

    # Fractional shift in original voxel space (x, y, z): the resampled
    # voxel i lies at original voxel coordinate i + offset
    offset_xyz = (new_origin - O) / vs
    offset_zyx = tuple(float(o) for o in offset_xyz[::-1])

    return _shift_linear(jnp.asarray(data, dtype=jnp.float32), offset_zyx), new_origin


def _shift_linear(data, offsets):
    """``out[i] = data[i + offset]`` per axis, by linear interpolation.

    Matches ``scipy.ndimage.shift(data, -offset, order=1, prefilter=False,
    mode="constant", cval=0)``: a voxel whose source coordinate falls
    outside ``[0, n-1]`` on any axis is 0.  One axis at a time (trilinear
    interpolation is separable).
    """
    for axis, off in enumerate(offsets):
        if off == 0.0:
            continue
        n = data.shape[axis]
        lo = lax.slice_in_dim(data, 0, n - 1, axis=axis)
        hi = lax.slice_in_dim(data, 1, n, axis=axis)
        edge = jnp.zeros_like(lax.slice_in_dim(data, 0, 1, axis=axis))
        if off > 0:     # between i and i+1; the last voxel's source is past the end
            data = jnp.concatenate([(1.0 - off) * lo + off * hi, edge], axis=axis)
        else:           # between i-1 and i; the first voxel's source is before the start
            data = jnp.concatenate([edge, -off * lo + (1.0 + off) * hi], axis=axis)
    return data


if _HAVE_JAX:
    _shift_linear = jit(_shift_linear, static_argnums=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def qscores_from_arrays(
    coords: np.ndarray,
    heavy: np.ndarray,
    residue: np.ndarray,
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
    batch_size: Optional[int] = 32768,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, float]:
    """Per-atom Q-scores of atoms given as arrays.

    The core of :func:`compute_qscores`, without per-atom Python objects;
    the map and the options are as there.

    Parameters
    ----------
    coords : ndarray, shape (N, 3)
        Atom positions in Ångströms.
    heavy : ndarray of bool, shape (N,)
        False for hydrogens.  These get the Q-score of the nearest heavy
        atom of their residue, or NaN if there is none within 1.4 Å.
    residue : ndarray of int, shape (N,)
        Residue code of each atom, equal for the atoms of one residue.

    Returns
    -------
    (q_scores, min_d, max_d)
        Q-score of each atom (float64), and the map's reference minimum and
        maximum (see :func:`min_max_d`).
    """
    _require_jax()
    if max_retries < 1:
        raise ValueError(f"max_retries must be ≥ 1, got {max_retries}")

    t_start = time.perf_counter()

    def _log(msg: str) -> None:
        if verbose:
            print(f"[{time.perf_counter() - t_start:8.1f} s] {msg}", flush=True)

    if to_rad is None:
        to_rad = 2.0 if sigma < 1.0 else 6.0
    if d_rad is None:
        d_rad = 0.1 if sigma < 1.0 else 0.5
    if nbr_radius is None:
        nbr_radius = to_rad * 2.0

    # The map goes to the device once; statistics and resampling are
    # dispatched asynchronously so host coordinate/cell-list setup overlaps
    jdata_in = jnp.asarray(data, dtype=jnp.float32)
    stats_dev = _map_stats(jdata_in)
    if mapq_compat:
        _log("[mapq_compat] Resampling map to replicate MaskMapResize origin rounding …")
        jdata, origin = _resample_mapq_compat(jdata_in, origin, voxel_size)
    else:
        jdata = jdata_in

    non_h_mask    = np.asarray(heavy, dtype=bool)
    non_h_indices = np.flatnonzero(non_h_mask)

    if len(non_h_indices) == 0:
        mean, std, min_val, max_val = (float(v) for v in jax.device_get(stats_dev))
        max_d = min(mean + 10.0 * std, max_val)
        min_d = max(mean -  1.0 * std, min_val)
        if mapq_compat and jdata_in is not data:
            jdata_in.delete()
        if jdata is not data:
            jdata.delete()
        warnings.warn("No non-hydrogen atoms found.")
        return np.full(len(non_h_mask), np.nan), min_d, max_d

    # Geometry stays in float64 on the host; the device gets each atom's
    # position in voxel units and its neighbours relative to it (float32
    # keeps those precise, unlike absolute coordinates of thousands of Å)
    all_coords   = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    non_h_coords = all_coords[non_h_mask]
    N            = len(non_h_indices)

    shell_radii_np = np.arange(d_rad, to_rad + 1e-6, d_rad, dtype=np.float32)

    mode_str = "retry" if use_retry else f"fast (n_gen={n_gen or max(num_pts*16,128)})"
    _log(f"[JAX / {jax.default_backend().upper()} / {mode_str}]  "
         f"Counting neighbours (r = {nbr_radius:.1f} Å) …")
    cell_list = (_GpuCellList.build(non_h_coords, nbr_radius)
                 if jax.default_backend() == "gpu" and N >= 8192 else None)
    if cell_list is not None:
        neighbours = None
        work_coords = cell_list.sorted_coords
    else:
        neighbours = _NeighbourIndex(non_h_coords, nbr_radius)
        work_coords = non_h_coords

    mean, std, min_val, max_val = (float(v) for v in jax.device_get(stats_dev))
    max_d = min(mean + 10.0 * std, max_val)
    min_d = max(mean -  1.0 * std, min_val)
    if mapq_compat and jdata_in is not data:
        jdata_in.delete()

    # Coordinates as float32 pairs (see _relative_coords); neighbour slots
    # index their rows, padding slots the last row, far from every atom
    padded = np.vstack([work_coords, np.full((1, 3), _FAR)])
    coords_hi = padded.astype(np.float32)
    jcoords_hi = jnp.asarray(coords_hi)
    jcoords_lo = jnp.asarray((padded - coords_hi).astype(np.float32))
    del padded, coords_hi

    if cell_list is not None:
        groups = _neighbour_groups(cell_list.counts(jcoords_hi, jcoords_lo))
    else:
        groups = _neighbour_groups(neighbours.counts(non_h_coords))

    A     = float(max_d - min_d)
    B     = float(min_d)
    gv_r0 = float(A + B)
    gv_shells_np = A * np.exp(-0.5 * (shell_radii_np.astype(np.float64) / sigma) ** 2) + B

    origin_np = np.asarray(origin, dtype=np.float64)
    vs_np     = np.asarray(voxel_size, dtype=np.float64)
    jvs       = jnp.asarray(vs_np.astype(np.float32))
    jradii    = jnp.asarray(shell_radii_np)

    # Batches of atom indices, (members, padded size, K).  All batches of a
    # group have one shape (the last is padded): one compilation per group.
    # None = a single batch per group (may OOM for large structures).
    batches = []
    for members, k in groups:
        size = len(members) if batch_size is None else max(1, min(batch_size, len(members)))
        batches += [(members[i:i + size], size, k) for i in range(0, len(members), size)]
    _log(f"Computing Q-scores:  {N} atoms  ×  {len(shell_radii_np)} shells  "
         f"({len(batches)} batch{'es' if len(batches) > 1 else ''} of ≤ {batches[0][1]} atoms, "
         f"{' / '.join(str(k) for _, k in groups)} neighbour slots)")

    if use_retry:
        retry = _RetryRunner(num_pts, max_retries, {(size, k) for _, size, k in batches},
                             jcoords_hi, jcoords_lo, jradii, jdata, jvs, cell_list=cell_list)
        _log("Compiled")
    else:
        if n_gen is None:
            n_gen = max(num_pts * 16, 128)
        compute_fn = _get_compute_fn(n_gen, num_pts)
        jdirs  = jnp.asarray(_sphere_directions(n_gen).astype(np.float32))
        jgv    = jnp.asarray(gv_shells_np.astype(np.float32))
        jgv_r0 = jnp.float32(gv_r0)

    def _prepare(members: np.ndarray, size: int, k: int):
        """Host side of a batch: neighbour lookup, padding to the group's shape."""
        pos = work_coords[members]
        atom_idx = members.astype(np.int32)
        if cell_list is not None:
            nbr_or_cell = cell_list.sorted_cell_id[members]
            pad_nbr = cell_list.empty_cell
        else:
            nbr_or_cell = neighbours.query(pos, k)
            pad_nbr = N
        if len(pos) < size:     # pad the last batch: no recompilation
            pad = size - len(pos)
            pos = np.concatenate([pos, np.zeros((pad, 3))])
            pad_shape = (pad,) + nbr_or_cell.shape[1:]
            nbr_or_cell = np.concatenate([nbr_or_cell, np.full(pad_shape, pad_nbr, np.int32)])
            atom_idx = np.concatenate([atom_idx, np.full(pad, N, np.int32)])
        return (size, k), (atom_idx, nbr_or_cell, ((pos - origin_np) / vs_np).astype(np.float32))

    def _launch(prepared):
        """Dispatch a prepared batch to the device (asynchronous)."""
        shape, arrays = prepared
        batch = tuple(jnp.asarray(a) for a in arrays)
        if use_retry:
            return retry.launch(shape, *batch)
        if cell_list is not None:
            nbr_idx = cell_list.nbr_fn(
                batch[0], batch[1], cell_list.j_cell_edge, cell_list.j_xy_offsets,
                jcoords_hi, jcoords_lo, cell_list.j_rad_sq, shape[1])
        else:
            nbr_idx = batch[1]
        nbr_rel = _relative_coords_jit(jcoords_hi, jcoords_lo, batch[0], nbr_idx)
        return compute_fn(batch[2], nbr_rel, jdata, jvs, jdirs, jradii, jgv, jgv_r0)

    def _finish(launched) -> np.ndarray:
        if use_retry:
            return _pearson_from_shell_sums(*retry.sums(launched), gv_shells_np, gv_r0, num_pts)
        return np.asarray(launched, dtype=np.float64)

    # Pipeline: the next batch is prepared in a thread (cKDTree and numpy
    # release the GIL) and dispatched before the previous one is finished,
    # so the device always has work queued
    q_non_h = np.empty(N, dtype=np.float64)
    done = 0
    with ThreadPoolExecutor(max_workers=1) as prep_pool:
        preparing = prep_pool.submit(_prepare, *batches[0])
        previous = None
        for i in range(len(batches) + 1):
            current = _launch(preparing.result()) if i < len(batches) else None
            if i + 1 < len(batches):
                preparing = prep_pool.submit(_prepare, *batches[i + 1])
            if previous is not None:
                members = batches[i - 1][0]
                q_non_h[members] = _finish(previous)[:len(members)]
                done += len(members)
                if len(batches) > 1 and i % max(1, len(batches) // 20) == 0:
                    _log(f"  {done}/{N} atoms")
            previous = current

    q_scores = np.full(len(non_h_mask), np.nan, dtype=np.float64)
    if cell_list is not None:
        q_scores[non_h_indices[cell_list.order]] = q_non_h
        cell_list.j_cell_edge.delete()
    else:
        q_scores[non_h_indices] = q_non_h

    if jdata is not data:
        jdata.delete()
    jcoords_hi.delete()
    jcoords_lo.delete()

    # Assign H atoms the Q-score of their nearest heavy atom in the same residue (reference behavior)
    _assign_hydrogen_qscores(np.asarray(residue), all_coords, non_h_mask, non_h_indices, q_scores)
    _log("Done")
    return q_scores, min_d, max_d



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
    batch_size: Optional[int] = 32768,
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
        When True (default), replicates the original retry loop: up to
        ``max_retries`` attempts with num_pts, num_pts+2, … sphere points,
        until one has ≥ num_pts points clear of other atoms.

        When False, uses a fixed ``n_gen`` sphere-point budget per shell,
        at the cost of ~0.03 mean Q-score difference.
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
    batch_size : int or None, default 32768
        Number of atoms per batch.  Device memory per batch grows with this
        value (the map itself is on the device throughout).  Reduce for tight
        GPUs; ``None`` processes all atoms in a single batch.
    verbose : bool
        Print progress information.

    Returns
    -------
    QScoreResult
    """
    heavy = np.fromiter((a.element != "H" for a in atoms), dtype=bool, count=len(atoms))
    coords = np.fromiter((c for a in atoms for c in (a.x, a.y, a.z)),
                         dtype=np.float64, count=3 * len(atoms)).reshape(-1, 3)
    q_scores, min_d, max_d = qscores_from_arrays(
        coords, heavy, _residue_codes(atoms), data, origin, voxel_size, sigma=sigma,
        num_pts=num_pts, to_rad=to_rad, d_rad=d_rad, use_retry=use_retry,
        max_retries=max_retries, n_gen=n_gen, nbr_radius=nbr_radius, mapq_compat=mapq_compat,
        batch_size=batch_size, verbose=verbose)

    if verbose:
        for at, q in zip(atoms, q_scores.tolist()):
            if at.element != "H":
                print(f"  {at.res_seq}.{at.chain_id}.{at.atom_name}: Q = {q:.6f}")

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
    batch_size: Optional[int] = 32768,
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

    Reads *input_cif_path* with gemmi, adds Q-scores from *result* as an
    ``_atom_site.Q-score`` column, and writes the modified CIF to::

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

    if not _HAVE_GEMMI:
        raise ImportError("gemmi is required for CIF output: pip install gemmi")

    cif_basename = os.path.basename(input_cif_path)
    map_basename = os.path.basename(map_path)
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_cif_path))
    out_path = os.path.join(output_dir, f"{cif_basename}__Q__{map_basename}.cif")

    doc = gemmi.cif.read(input_cif_path)
    block = doc[0]

    # Count atoms in the CIF atom_site loop
    id_col = block.find_loop("_atom_site.id")
    n_atoms = len(id_col) if id_col else 0

    if n_atoms != len(result.atom_qscores):
        raise ValueError(
            f"Atom count mismatch: CIF has {n_atoms} atoms, "
            f"result has {len(result.atom_qscores)}. "
            "Input CIF must not be modified between run() and write_qscore_cif()."
        )

    # Map result.atom_records (gemmi order) to the rows of the CIF loop.  The
    # orders can differ: gemmi.read_structure() groups ligand residues with
    # their parent chain, while many CIF files list all polymer chains first,
    # then all ligands.  Atoms are matched by position, parsed from the same
    # text by gemmi and numpy (both round correctly, so the floats are
    # equal); atoms at one position are taken in order of appearance.  This
    # needs none of the auth_*/label_* identifier columns, which differ
    # between writers (python-ihm omits auth_atom_id).
    cif_xyz = np.column_stack([
        np.fromiter(block.find_values(f"_atom_site.Cartn_{c}"), dtype=np.float64, count=n_atoms)
        for c in "xyz"
    ])
    atoms = result.atom_records
    res_xyz = np.fromiter((c for a in atoms for c in (a.x, a.y, a.z)),
                          dtype=np.float64, count=3 * len(atoms)).reshape(-1, 3)
    if np.array_equal(cif_xyz, res_xyz):
        row_of_atom = np.arange(n_atoms)
    else:
        cif_order = np.lexsort(cif_xyz.T[::-1])
        res_order = np.lexsort(res_xyz.T[::-1])
        if not np.array_equal(cif_xyz[cif_order], res_xyz[res_order]):
            raise ValueError(
                f"Atom positions in {input_cif_path} differ from those of the result. "
                "Input CIF must not be modified between run() and write_qscore_cif()."
            )
        row_of_atom = np.empty(n_atoms, dtype=np.int64)
        row_of_atom[res_order] = cif_order
    q_by_row = np.empty(n_atoms, dtype=np.float64)
    q_by_row[row_of_atom] = result.atom_qscores
    add_qscore_column(block, q_by_row)
    doc.write_file(out_path)

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
    parser.add_argument("--batch-size",  type=int, default=32768,
                        help="Atoms per batch (default 32768). "
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
        atoms = result.atom_records
        chain, names = _factorize((a.chain_id for a in atoms), len(atoms))
        heavy = np.fromiter((a.element != "H" for a in atoms), dtype=bool, count=len(atoms))
        scored = heavy & ~np.isnan(result.atom_qscores)
        n_heavy = dict(zip(names, np.bincount(chain[scored], minlength=len(names))))

        print(f"sigma={result.sigma:.2f}  minD={result.min_d:.6f}  maxD={result.max_d:.6f}")
        print()
        for ch, q in sorted(ch_qs.items()):
            print(f"Chain {ch}: Q = {q:.6f}  ({n_heavy[ch]} non-H atoms)")
        print()
        print("chain\tres_seq\tres_name\tQ_backbone\tQ_sidechain\tQ_residue")
        for (ch, rs, ic, rn), d in heapq.nsmallest(20, res_qs.items()):
            bb = f"{d['backbone']:.6f}" if not np.isnan(d['backbone']) else "   -    "
            sc = f"{d['sidechain']:.6f}" if not np.isnan(d['sidechain']) else "   -    "
            print(f"{ch}\t{rs}{ic}\t{rn}\t{bb}\t{sc}\t{d['all']:.6f}")


if __name__ == "__main__":
    _cli()
