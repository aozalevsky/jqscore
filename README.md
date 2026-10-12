# jqscore

[![Tests](https://github.com/aozalevsky/jqscore/actions/workflows/tests.yml/badge.svg)](https://github.com/aozalevsky/jqscore/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/jqscore)](https://pypi.org/project/jqscore/)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

JAX reimplementation of [Q-score](https://github.com/gregdp/mapq) — a per-atom cryo-EM map-to-model fit metric.

Q-score measures how well each atom in an atomic model fits into a cryo-EM
density map by comparing the observed radial density profile around each atom
to a theoretical Gaussian profile (Pearson correlation).

> Pintilie et al., *Nature Methods* 2020. Reference implementation: [gregdp/mapq](https://github.com/gregdp/mapq)

## Installation

```bash
pip install jqscore            # CPU
pip install "jqscore[gpu]"     # NVIDIA GPU: CUDA 13 build of JAX, driver ≥ 580
```

With an older driver (≥ 525), use the CUDA 12 build of JAX:
`pip install jqscore "jax[cuda12]"`. The development version:
`pip install git+https://github.com/aozalevsky/jqscore`.

## Python API

```python
import jqscore

# From file paths (PDB/mmCIF + MRC/CCP4)
result = jqscore.run("structure.cif", "map.mrc")

# Access results
print(result.atom_qscores)          # per-atom Q-scores (H: nearest heavy atom's in the residue)
print(jqscore.chain_qscores(result))    # per-chain averages
print(jqscore.residue_qscores(result))  # per-residue breakdown

# Lower-level: supply parsed atoms and map arrays
atoms = jqscore.read_cif("structure.cif")
data, origin, voxel_size = jqscore.read_mrc("map.mrc")
result = jqscore.compute_qscores(atoms, data, origin, voxel_size, sigma=0.4)
```

### Computation modes

| Mode | What it does |
|------|--------------|
| `use_retry=True` (default) | MapQ's retry loop: up to 50 attempts with 8, 10, 12, … sphere points per shell, until ≥ 8 are clear of other atoms. Matches a float64 reference to ~1e-5, except for about 1 atom in 100,000 with a sample point within ~1e-7 Å of the exclusion cutoff, which float32 cannot decide. |
| `use_retry=False` | A fixed budget of `n_gen` points per shell; all clear points count. ~0.03 mean Q-score difference from retry mode. |

```python
# Fast mode
result = jqscore.run("structure.cif", "map.mrc", use_retry=False)

# Check JAX backend
print(jqscore.backend())  # 'cpu', 'gpu', or 'tpu'
```

Atoms are processed in batches (`batch_size`, default 32768); the map stays on
the GPU throughout. Lower `batch_size` if the GPU runs out of memory.

On an RTX A4000, `jqscore 36ZA.cif emd_76130.map` (11.2 M atoms, 2888
chains, 800³ map) takes 2 min, half of it reading the 1.2 GB mmCIF, and
`jqscore-cmd` with the annotated CIF written 2.5 min.

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
  --no-mapq-compat    Interpolate the map directly instead of replicating
                      MapQ's trimmed-map resampling
  --batch-size INT    Atoms per batch (default 32768; lower for small GPUs)
  --verbose           Print progress and per-atom Q-scores
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

## Container

`container/jqscore.def` builds an Apptainer/Singularity image with the CUDA
build of JAX (the host needs only an NVIDIA driver, ≥ 580 for CUDA 13).
From the repository root:

```bash
apptainer build --fakeroot jqscore.sif container/jqscore.def
# driver older than 580: --build-arg JAX="jax[cuda12]==0.9.2"

apptainer run --nv jqscore.sif structure.cif map.mrc
apptainer exec --nv jqscore.sif jqscore-cmd /opt/chimera map=map.mrc cif=structure.cif
```

The build ends by running the test suite on the CPU. `apptainer run-help
jqscore.sif` lists the commands.

## Benchmarking against MapQ

The included benchmark script (needs `pip install "jqscore[benchmark]"` for matplotlib) downloads real PDB+EMDB structures, runs both the original [MapQ](https://github.com/gregdp/mapq) (via Singularity) and jqscore, and produces per-atom scatter plots, a summary comparison, and a markdown report.

```bash
# Run with default structures (7 entries):
python -m jqscore.benchmark

# Run on specific PDB IDs:
python -m jqscore.benchmark 9XZK 9UX2

# Custom output directory:
python -m jqscore.benchmark -o my_benchmark

# After changing jqscore: re-run it only, keeping MapQ results and times
python -m jqscore.benchmark --rerun-jqscore
```

The MapQ reference is invoked via the `MAPQ_CMD` environment variable (expects a Singularity container by default). A full benchmark run with all 7 default structures takes approximately 30 minutes.

Results are saved to `jqscore_mapq_benchmark/` (plots, output CIFs, and `README.md`); the latest run is in [`docs/jqscore_vs_mapq/`](https://github.com/aozalevsky/jqscore/blob/main/docs/jqscore_vs_mapq/README.md). Of the 750,165 non-H atoms of its 7 entries, 15 differ from MapQ by more than 1e-3: 5 in 3J7O where the density around the atom is flat (MapQ gives ~0.1, jqscore 0), and 10 with a sample point within ~1e-7 Å of the exclusion cutoff, a tie that MapQ also breaks differently from a float64 reference in 3 of the 10. See [`spec/benchmark.md`](https://github.com/aozalevsky/jqscore/blob/main/spec/benchmark.md) for the full specification.

## License

MIT
