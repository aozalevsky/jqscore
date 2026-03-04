# MapQ vs jqscore: Trimmed-Map Double-Interpolation Discrepancy

## Summary

For most atoms the per-atom Q-score difference between jqscore and MapQ is
<0.01 (float32 noise). However, a small fraction of atoms — typically those
sitting in unusually flat or rising density profiles — can differ by up to ~0.18.
The root cause is **MapQ's `MaskMapResize` function**, which resamples the
density map before the subprocess queries it, introducing a fixed fractional
voxel offset and thus a systematic blurring of the map.

`jqscore` now replicates this behavior via the `mapq_compat=True` option
(default), reducing the maximum per-atom error to <0.001 for all tested cases.

---

## Root Cause: `MaskMapResize` in MapQ

MapQ splits its atoms across `numProc` subprocesses.  Before spawning each
subprocess it calls:

```python
nmap = MaskMapResize(P.atoms, bound=8.0, dmap, P.dmapPath)
```

`MaskMapResize` (in `mapq/qscores.py`) does the following:

1. Converts atom world-coordinates to voxel-indices (float32 arithmetic,
   because Chimera stores origin and step as float32 in the MRC header).
2. Computes the bounding box of the process's atoms (+8 Å padding) in voxel
   space, giving a float `li,lj,lk`.
3. Computes a preliminary new origin:
   ```python
   nO = (O[0] + float(li)*step[0],
         O[1] + float(lj)*step[1],
         O[2] + float(lk)*step[2])
   ```
4. **Rounds** the origin to the nearest multiple of the step:
   ```python
   ox = round(nO[0]/step[0]) * step[0]
   oy = round(nO[1]/step[1]) * step[1]
   oz = round(nO[2]/step[2]) * step[2]
   ```
5. Creates a new map array (`nmat`, float32) and fills it by interpolating
   the original map at the new grid:
   ```python
   npoints = grid_indices((nn1,nn2,nn3), numpy.single)   # float32 ijk
   transform_vertices(npoints, ndata.ijk_to_xyz_transform) # → float32 xyz
   dvals = dmap.interpolated_values(npoints, dmap.openState.xform)
   nmat  = dvals.reshape((nn3, nn2, nn1))
   ```
6. Saves the trimmed map and spawns the subprocess, which loads **only the
   trimmed map**.

### The fractional offset

The rounded origin `ox` is **not in general aligned** with the original voxel
grid.  The original grid has voxels at `O_orig + k·step` for integer `k`, while
`ox = round(nO/step)·step` places the trimmed grid at integer multiples of
`step` from zero, not from `O_orig`.

The mismatch is:

```
frac_offset = (ox - O_orig) / step   mod 1
            = (1 - O_orig/step mod 1) mod 1
```

This depends **only** on `O_orig/step mod 1` — it is the same for every
process and every atom within a given map.

For the test map `390_emd_30342_A_z4.mrc`:

| dimension | O_orig (Å) | step (Å) | frac\_offset |
|-----------|-----------|---------|------------|
| x         | 145.825   | 1.113   | ≈ 0.980    |
| y         | 112.825   | 1.113   | ≈ 0.630    |
| z         | 120.517   | 1.113   | ≈ 0.719    |

Because step 5 in `MaskMapResize` interpolates the original at these
non-integer positions, the trimmed map is a **blurred** version of the
original.  The subprocess then performs a second interpolation when it
queries the trimmed map for each sphere point — giving **double interpolation**
total.

---

## Quantitative impact

Atoms with unusual radial density profiles (e.g. density that rises rather
than falls with radius) are most sensitive.  Verification by replicating
`MaskMapResize` in Python (scipy `map_coordinates` at float32 coordinates)
and re-running `QscorePt3`:

| Atom         | MapQ ref | jqscore (original) | jqscore (mapq\_compat) | Δ (compat vs ref) |
|--------------|----------|--------------------|------------------------|-------------------|
| LEU402/C     | 0.291    | 0.111              | 0.291                  | −0.000            |
| PHE177/CZ    | 0.203    | 0.365              | 0.203                  | +0.000            |
| GLU281/CG    | 0.048    | 0.209              | 0.048                  | −0.000            |
| ARG157/CZ    | 0.039    | 0.192              | 0.039                  | −0.000            |
| ARG555/C     | 0.414    | 0.271              | 0.414                  | −0.000            |
| ALA231/CA    | 0.459    | 0.600              | 0.459                  | +0.000            |

All six previously-discrepant atoms match MapQ within ±0.0004 after enabling
`mapq_compat`.  (The residual ±0.0004 is from float32 rounding in the JAX
computation, consistent with all other atoms.)

---

## Implementation in jqscore

`jqscore` replicates the effect via `_resample_mapq_compat(data, origin,
voxel_size)`, which:

1. Casts `origin` and `voxel_size` to float32 (matching Chimera's MRC reader).
2. Computes `new_origin = round(origin_f32 / step_f32) * step_f32`.
3. Uses `scipy.ndimage.shift` to resample the map with the fractional offset
   `(new_origin − origin_f32) / step_f32`, producing an array of the same
   shape.
4. Returns `(resampled_data, new_origin)`.

`compute_qscores()` and `run()` both accept `mapq_compat=True` (default).
When enabled, the resampled data and new origin are passed to JAX instead of
the raw map.

The extra cost is one CPU pass over the map data with `scipy.ndimage.shift`
(O(Nvoxels)), typically a few seconds even for large maps.

---

## When to disable `mapq_compat`

Set `mapq_compat=False` when:

* You want the **physically more accurate** answer (single interpolation from
  the original map, no extra blurring).
* You are comparing against a MapQ run where `MaskMapResize` was bypassed
  (the commented-out `copyfile` branch in `CalcQpn`).
* Benchmarking runtime without the one-time resampling overhead.

---

## Files involved

| File | Purpose |
|------|---------|
| `jqscore/jqscore.py` | `_resample_mapq_compat()`, `mapq_compat` param |
| `docs/jqscore_vs_mapq/mapq_trimmed_map_discrepancy.md` | This document |
| `/tmp/mapq_replica.py` | Pure-Python 3 replica used to isolate the issue |
| `/tmp/chi_pt3_test.py` | Chimera script used to confirm Chimera interpolation matches scipy |
