# jqscore

[![Tests](https://github.com/aozalevsky/jqscore/actions/workflows/tests.yml/badge.svg)](https://github.com/aozalevsky/jqscore/actions/workflows/tests.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

JAX reimplementation of [Q-score](https://github.com/gregdp/mapq) — a per-atom cryo-EM map-to-model fit metric.

Q-score measures how well each atom in an atomic model fits into a cryo-EM
density map by comparing the observed radial density profile around each atom
to a theoretical Gaussian profile (Pearson correlation).

> Pintilie et al., *Nature Methods* 2020. Reference implementation: [gregdp/mapq](https://github.com/gregdp/mapq)

## Installation

```bash
pip install git+https://github.com/aozalevsky/jqscore
```

Soon will be available for installation from PyPI:

```bash
pip install jqscore          # CPU (default JAX backend)
pip install jqscore[gpu]     # CUDA GPU
```

## Python API

```python
import jqscore

# From file paths (PDB/mmCIF + MRC/CCP4)
result = jqscore.run("structure.cif", "map.mrc")

# Access results
print(result.atom_qscores)          # per-atom Q-scores (NaN for H)
print(jqscore.chain_qscores(result))    # per-chain averages
print(jqscore.residue_qscores(result))  # per-residue breakdown

# Lower-level: supply parsed atoms and map arrays
atoms = jqscore.read_cif("structure.cif")
data, origin, voxel_size = jqscore.read_mrc("map.mrc")
result = jqscore.compute_qscores(atoms, data, origin, voxel_size, sigma=0.4)
```

### Computation modes

| Mode | Accuracy vs numpy | Speed |
|------|-------------------|-------|
| `use_retry=True` (default) | ~1e-4 (float32 rounding) | ~5× faster |
| `use_retry=False` | ~0.03 mean diff | ~7× faster |

```python
# Fast mode
result = jqscore.run("structure.cif", "map.mrc", use_retry=False)

# Check JAX backend
print(jqscore.backend())  # 'cpu', 'gpu', or 'tpu'
```

## CLI

```
jqscore structure.cif map.mrc [options]

options:
  --sigma FLOAT       Gaussian sigma in Å (default 0.4; use 0.6 for older datasets)
  --num-pts INT       Sample points per shell (default 8)
  --to-rad FLOAT      Max radial distance in Å
  --d-rad FLOAT       Radial shell spacing in Å
  --no-retry          Use fast fixed-budget mode
  --max-retries INT   Max retry attempts per shell (default 50)
  --n-gen INT         Sphere-point budget for --no-retry mode
  --verbose           Print per-atom Q-scores
  --output FILE       Write per-residue TSV to file (default: stdout summary)
```

## MapQ-compatibility mode

The `jqscore-cmd` entry point matches the argument style of the original MapQ Chimera plugin’s `mapq_cmd.py`, so tools like the [VA validation package](https://pypi.org/project/va/) can use jqscore as a drop-in replacement without changes.

**How to run:**

```bash
jqscore-cmd <chimera_root> map=<mapfile> cif=<cif1> [cif=<cif2> ...] [np=<n>] [res=<resolution>] [sigma=<sigma>]
```

- **Required:** `map=<path>` — path to the MRC/CCP4 density map.  
- **Required:** `cif=<path>` — path to an mmCIF structure; repeat for multiple models.  
- **Optional:** first positional argument is the Chimera root path (accepted but unused; jqscore does not require Chimera).

All parameters other than the map path, structure path(s), and **sigma** are accepted for compatibility but **ignored** (e.g. `np=`, `res=`, Chimera root). The optional `sigma=<float>` sets the Gaussian width in Å (default 0.4). Output CIFs with a Q-score column are written to the current working directory as `{cif_basename}__Q__{map_basename}.cif`, matching the naming expected by VA.

## Benchmarking against MapQ

The included benchmark script downloads real PDB+EMDB structures, runs both the original [MapQ](https://github.com/gregdp/mapq) (via Singularity) and jqscore, and produces per-atom scatter plots, a summary comparison, and a markdown report.

```bash
# Run with default structures (7 entries):
python -m jqscore.benchmark

# Run on specific PDB IDs:
python -m jqscore.benchmark 9XZK 9UX2

# Custom output directory:
python -m jqscore.benchmark -o my_benchmark
```

The MapQ reference is invoked via the `MAPQ_CMD` environment variable (expects a Singularity container by default). A full benchmark run with all 7 default structures takes approximately 30 minutes.

Results are saved to `jqscore_mapq_benchmark/` (plots, output CIFs, and `SUMMARY.md`). See [`spec/benchmark.md`](spec/benchmark.md) for the full specification.

## License

MIT
