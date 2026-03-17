# jqscore vs MapQ Benchmark Script Specification

## Overview

A standalone benchmark script (`jqscore/benchmark.py`) that downloads real
PDB+EMDB data, runs both MapQ (reference) and jqscore (on CPU and GPU),
compares Q-scores per atom, and produces publication-quality plots.

## CLI Interface

```
python -m jqscore.benchmark [OPTIONS] [PDB_IDS...]
```

### Positional Arguments

| Argument  | Description                        | Default   |
|-----------|------------------------------------|-----------|
| `PDB_IDS` | One or more PDB IDs to benchmark   | `[4AQW, 9XZK, 9UX2, 9QHX, 8RQB, 3J7O, 8WQL]` |

### Options

| Option             | Description                                      | Default                          |
|--------------------|--------------------------------------------------|----------------------------------|
| `-o, --output-dir` | Root output directory                            | `jqscore_mapq_benchmark`        |
| `--cache-dir`      | Directory for downloaded files                   | `<output-dir>/cache`             |
| `--no-cache`       | Ignore all cached files, re-download and re-run  | `false`                          |

### Environment Variables

| Variable   | Description                                           | Default |
|------------|-------------------------------------------------------|---------|
| `MAPQ_CMD` | Shell command template to invoke MapQ. Must contain `{pdb_id}`, `{emdb_id}`, and `{resolution}` placeholders. | `singularity exec --pid $MAPQ_SIF python /opt/chimera/share/mapq/mapq_cmd.py /opt/chimera/ {pdb_id}.cif {emdb_id}.map sigma=0.4 res={resolution} np=48` |

## Data Acquisition

1. For each PDB ID, query the **RCSB REST API** (`https://data.rcsb.org/rest/v1/core/entry/{PDB_ID}`) to obtain:
   - Corresponding EMDB ID (from `pdbx_database_related` where `db_name == "EMDB"`)
   - Map resolution (from `em_3d_reconstruction[0].resolution`, with **EMDB API fallback** via `https://www.ebi.ac.uk/emdb/api/entry/{EMDB_ID}`)
2. Download the **mmCIF** file from RCSB: `https://files.rcsb.org/download/{PDB_ID}.cif`
3. Download the **EMDB map** (gzipped): `https://files.wwpdb.org/pub/emdb/structures/{EMDB_ID}/map/emd_{numeric}.map.gz`
4. **Decompress** the `.map.gz` to `.map` in the cache directory. Both MapQ and jqscore use the same uncompressed `.map` file for fair I/O comparison.
5. All downloads go to `cache_dir`. Existing files are reused unless `--no-cache` is set.

## Execution

### MapQ (reference)

- Create a temporary working directory
- Copy the CIF and symlink the uncompressed `.map` file into it
- Run the `MAPQ_CMD` template with substituted `{pdb_id}`, `{emdb_id}`, `{resolution}`
- Copy the output CIF (`{pdb_id}.cif__Q__{emdb_id}.map.cif`) to `output_dir`
- Record wall-clock time
- MapQ output is cached in `output_dir` (skipped if file exists, unless `--no-cache`)

### jqscore (CPU and GPU)

jqscore is run **twice** in separate subprocesses to get independent timings for both backends:

1. **CPU run**: subprocess using the current Python (`sys.executable`) with `JAX_PLATFORMS=cpu`
   - Output CIF: `jax_cpu_{PDB_ID}.cif__Q__{map_basename}.cif`
2. **GPU run**: subprocess using the current Python without `JAX_PLATFORMS` (JAX auto-selects best backend)
   - Output CIF: `jax_gpu_{PDB_ID}.cif__Q__{map_basename}.cif`
   - If GPU is not available, the subprocess fails gracefully and is skipped

Subprocesses are used because JAX caches the backend on first use within a process, so `JAX_PLATFORMS` must be set before import.

For Q-score comparison against MapQ, the GPU output is used if available, otherwise CPU.

Both outputs are cached in `output_dir` (skipped if file exists, unless `--no-cache`).

### Comparison

- Use `jqscore.compare.compare_qscores()` to get per-atom Q-score arrays and statistics
- Count non-hydrogen atoms from atom labels (names not starting with "H")

## Plots

All plots saved as PNG (150 dpi) in `output_dir`.

### 1. Per-structure scatter (`{PDB_ID}_scatter.png`)

- X-axis: MapQ Q-score per non-H atom
- Y-axis: jqscore Q-score per non-H atom
- Red dashed y=x reference line
- Title: `{PDB_ID} / {EMDB_ID} (res={resolution} A, N_non-H={count})`
- Text box showing Pearson r
- Axis range: [-0.2, 1.05], aspect ratio 1:1
- Small markers (s=2), alpha=0.3, rasterized for file size

### 2. Summary (`summary.png`)

- Only generated when >1 structure is benchmarked
- Two-panel figure
- **Top panel**: scatter of structure-level mean Q-scores — jqscore (Y) vs MapQ (X), with y=x reference line and PDB ID labels. Equal aspect ratio.
- **Bottom panel**: runtime vs structure size (number of non-H atoms, sorted ascending). Three series: MapQ (np=N), jqscore (CPU), jqscore (GPU). Zero-time (cached) entries filtered out. GPU series omitted if not available. Points labeled with PDB ID.
- Grid enabled on both panels

## Output Structure

```
<output_dir>/
  cache/
    {PDB_ID}.cif                                  # downloaded mmCIF
    {EMDB_ID}.map.gz                              # downloaded EMDB map (compressed)
    {EMDB_ID}.map                                 # decompressed EMDB map
  {PDB_ID}.cif__Q__{emdb_id}.map.cif              # MapQ output
  jax_cpu_{PDB_ID}.cif__Q__{emdb_id}.map.cif      # jqscore CPU output
  jax_gpu_{PDB_ID}.cif__Q__{emdb_id}.map.cif      # jqscore GPU output
  {PDB_ID}_scatter.png                             # per-structure scatter
  summary.png                                      # summary: Q-score agreement + runtimes (if >1)
```

## Summary Table

Printed to stdout at the end:

```
   PDB         EMDB   Res  N_nonH         r  mean_diff   max_diff   t_mapq    t_cpu    t_gpu
----------------------------------------------------------------------------------------------------
  9XZK    EMD-72359  3.91    3985  1.000000  -0.000001   0.003806    17.7s    24.8s    19.5s
```

## Markdown Summary (`README.md`)

Written to `output_dir/README.md`. Includes:

- Date, structure count, sigma value
- **Hardware table**: CPU model, core count, GPU model, MapQ `np` value (parsed from `MAPQ_CMD`)
- **Accuracy table**: min/mean/max of Pearson r and max |diff| across structures
- **Per-structure results table**: PDB, EMDB, resolution, non-H atoms, r, mean diff, max |diff|, and timings (if available)
- **Plots section**: embedded image links to per-structure scatter plots and summary plot

Hardware info is detected automatically:
- CPU: parsed from `/proc/cpuinfo`
- GPU: queried via `nvidia-smi`
- MapQ np: extracted from the `MAPQ_CMD` template via regex

## Dependencies

- **Required**: numpy, scipy, jax, gemmi, matplotlib (add `matplotlib` to project optional deps if needed)
- **External**: singularity (for MapQ via container) -- or user-supplied `MAPQ_CMD`

## Error Handling

- If a single entry fails (download, MapQ, jqscore), print the error and continue with remaining entries
- Exit with code 1 only if ALL entries fail
- If GPU is not available, GPU run is skipped gracefully (not an error)

## Notes

- sigma=0.4 is hardcoded (EMDB convention for modern maps)
- MapQ's np=48 is embedded in the default command template; adjust via `MAPQ_CMD`
- Cached results include both downloads and computed Q-score CIFs
- `--no-cache` forces re-download AND re-computation (full clean run)
- Both MapQ and jqscore read the same uncompressed `.map` file for fair comparison
- MapQ command template uses `.map` (not `.map.gz`) because Chimera cannot reliably read gzipped maps
