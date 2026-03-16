#!/usr/bin/env python3
"""Benchmark jqscore against MapQ reference using real PDB+EMDB data.

Downloads structures and maps from RCSB/EMDB, runs both MapQ (via
singularity/chimera) and jqscore, then produces comparison plots.

Usage
-----
    python benchmark.py [OPTIONS] [PDB_IDS...]

Examples
--------
    python benchmark.py                          # default: 9XZK
    python benchmark.py 9XZK 3J7O 7A6A
    python benchmark.py --output-dir my_bench --no-cache 9XZK
    MAPQ_CMD="..." python benchmark.py 9XZK
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jqscore import run, write_qscore_cif
from jqscore.compare import compare_qscores, read_qscores

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PDB_IDS = ["9XZK", "9UX2", "9QHX"]
DEFAULT_OUTPUT_DIR = "jqscore_mapq_benchmark"
DEFAULT_GPU_PYTHON = sys.executable
DEFAULT_MAPQ_CMD = (
    "singularity exec --pid $MAPQ_SIF"
    " python /opt/chimera/share/mapq/mapq_cmd.py /opt/chimera/"
    " {pdb_id}.cif {emdb_id}.map sigma=0.4 res={resolution} np=48"
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EntryInfo:
    pdb_id: str
    emdb_id: str = ""
    resolution: float = 0.0
    cif_path: str = ""
    map_path: str = ""


@dataclass
class BenchmarkResult:
    entry: EntryInfo
    jqscore_cif: str = ""
    mapq_cif: str = ""
    jqscore_cpu_time: float = 0.0
    jqscore_gpu_time: float = 0.0
    mapq_time: float = 0.0
    n_non_h: int = 0
    pearson_r: float = float("nan")
    diff_mean: float = float("nan")
    diff_max: float = float("nan")
    diff_min: float = float("nan")
    q_jqscore: Optional[np.ndarray] = None
    q_mapq: Optional[np.ndarray] = None
    gpu_available: bool = False


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: str) -> None:
    """Download a URL to a local file with a progress indicator."""
    print(f"  Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def fetch_entry_info(pdb_id: str) -> EntryInfo:
    """Fetch EMDB ID and resolution for a PDB entry from RCSB + EMDB APIs."""
    pdb_id = pdb_id.upper()

    # Step 1: Get EMDB ID from RCSB
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read())

    emdb_id = ""
    em_list = data.get("pdbx_database_related", [])
    for rel in em_list:
        if rel.get("db_name", "").upper() == "EMDB":
            emdb_id = rel["db_id"]
            break

    if not emdb_id:
        raise ValueError(f"No EMDB entry found for PDB {pdb_id}")

    # Step 2: Get resolution — try RCSB first, fall back to EMDB API
    resolution = 0.0
    em_recon = data.get("em_3d_reconstruction", [])
    if em_recon:
        resolution = float(em_recon[0].get("resolution", 0.0))
    if resolution == 0.0:
        refine = data.get("refine", [])
        if refine:
            resolution = float(refine[0].get("ls_d_res_high", 0.0))

    if resolution == 0.0:
        # Fall back to EMDB API
        emdb_url = f"https://www.ebi.ac.uk/emdb/api/entry/{emdb_id}"
        with urllib.request.urlopen(emdb_url) as resp:
            emdb_data = json.loads(resp.read())
        try:
            sd = emdb_data["structure_determination_list"]["structure_determination"][0]
            ip = sd["image_processing"][0]
            resolution = float(ip["final_reconstruction"]["resolution"]["valueOf_"])
        except (KeyError, IndexError, TypeError):
            print(f"  WARNING: Could not determine resolution for {pdb_id}/{emdb_id}")

    return EntryInfo(pdb_id=pdb_id, emdb_id=emdb_id, resolution=resolution)


def download_cif(pdb_id: str, cache_dir: str) -> str:
    """Download mmCIF file from RCSB, return local path."""
    pdb_id = pdb_id.upper()
    dest = os.path.join(cache_dir, f"{pdb_id}.cif")
    if os.path.exists(dest):
        print(f"  Cached: {dest}")
        return dest
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    _download(url, dest)
    return dest


def download_map(emdb_id: str, cache_dir: str) -> str:
    """Download EMDB map (.map.gz), return local path."""
    emdb_id = emdb_id.upper()
    # EMDB ID format: EMD-XXXX -> numeric part
    numeric = emdb_id.replace("EMD-", "")
    dest = os.path.join(cache_dir, f"{emdb_id}.map.gz")
    if os.path.exists(dest):
        print(f"  Cached: {dest}")
        return dest
    url = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/{emdb_id}/map/emd_{numeric}.map.gz"
    _download(url, dest)
    return dest


def prepare_entry(pdb_id: str, cache_dir: str, no_cache: bool) -> EntryInfo:
    """Fetch metadata and download files for a PDB entry.

    Downloads .gz from EMDB, then decompresses to .map in cache.
    ``entry.map_path`` points to the uncompressed .map file so that
    both MapQ and jqscore read the same file (fair I/O comparison).
    """
    pdb_id = pdb_id.upper()
    print(f"\n{'='*60}")
    print(f"Preparing {pdb_id}")
    print(f"{'='*60}")

    # Metadata (always fetch fresh — it's tiny)
    info = fetch_entry_info(pdb_id)
    print(f"  EMDB: {info.emdb_id}, Resolution: {info.resolution:.2f} A")

    if no_cache:
        # Remove cached files so they are re-downloaded
        for f in [f"{pdb_id}.cif", f"{info.emdb_id}.map.gz", f"{info.emdb_id}.map"]:
            p = os.path.join(cache_dir, f)
            if os.path.exists(p):
                os.remove(p)

    info.cif_path = download_cif(pdb_id, cache_dir)
    gz_path = download_map(info.emdb_id, cache_dir)

    # Decompress into cache so both MapQ and jqscore use uncompressed .map
    map_path = os.path.join(cache_dir, f"{info.emdb_id}.map")
    if not os.path.exists(map_path):
        print(f"  Decompressing map -> {map_path}")
        _decompress_gz(gz_path, map_path)
    info.map_path = map_path
    return info


# ---------------------------------------------------------------------------
# Run MapQ
# ---------------------------------------------------------------------------

def _decompress_gz(gz_path: str, out_path: str) -> None:
    """Decompress a .gz file to out_path."""
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def run_mapq(
    entry: EntryInfo,
    output_dir: str,
    mapq_cmd_template: str,
    no_cache: bool,
) -> tuple[str, float]:
    """Run MapQ and return (output_cif_path, elapsed_seconds)."""
    pdb_id = entry.pdb_id
    emdb_numeric = entry.emdb_id.replace("EMD-", "")
    map_name = f"emd_{emdb_numeric}.map"

    # Expected output filename (MapQ writes to cwd)
    expected_out = os.path.join(
        output_dir, f"{pdb_id}.cif__Q__{map_name}.cif"
    )

    if os.path.exists(expected_out) and not no_cache:
        print(f"  MapQ cached: {expected_out}")
        return expected_out, 0.0

    # Build command
    cmd_str = mapq_cmd_template.format(
        pdb_id=pdb_id,
        emdb_id=f"emd_{emdb_numeric}",
        resolution=f"{entry.resolution:.2f}",
    )

    # MapQ expects files in cwd; copy CIF and symlink the uncompressed map.
    work_dir = tempfile.mkdtemp(prefix=f"mapq_{pdb_id}_")
    try:
        cif_dst = os.path.join(work_dir, f"{pdb_id}.cif")
        map_dst = os.path.join(work_dir, map_name)
        shutil.copy2(entry.cif_path, cif_dst)
        os.symlink(os.path.abspath(entry.map_path), map_dst)

        print(f"  Running MapQ: {cmd_str}")
        t0 = time.time()
        result = subprocess.run(
            cmd_str, shell=True, cwd=work_dir,
            capture_output=True, text=True,
        )
        elapsed = time.time() - t0
        print(f"  MapQ finished in {elapsed:.1f}s (exit code {result.returncode})")

        if result.returncode != 0:
            print(f"  MapQ stderr: {result.stderr[:500]}")
            raise RuntimeError(f"MapQ failed for {pdb_id}")

        # Find output file
        mapq_out = os.path.join(work_dir, f"{pdb_id}.cif__Q__{map_name}.cif")
        if not os.path.exists(mapq_out):
            raise FileNotFoundError(
                f"MapQ output not found in {work_dir}. "
                f"Files: {os.listdir(work_dir)}"
            )

        shutil.copy2(mapq_out, expected_out)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return expected_out, elapsed


# ---------------------------------------------------------------------------
# Run jqscore
# ---------------------------------------------------------------------------

def _run_jqscore_subprocess(
    entry: EntryInfo,
    output_dir: str,
    backend_name: str,
    python_exe: str | None = None,
) -> tuple[str, float]:
    """Run jqscore in a subprocess with a specific JAX backend.

    We use a subprocess so that JAX_PLATFORMS takes effect (JAX caches
    the backend on first use within a process).
    Returns (output_cif_path, elapsed_seconds).
    """
    python_exe = python_exe or sys.executable
    map_basename = os.path.basename(entry.map_path)
    prefix = f"jax_{backend_name}_"
    expected_out = os.path.join(
        output_dir, f"{prefix}{entry.pdb_id}.cif__Q__{map_basename}.cif"
    )

    # Inline script that runs jqscore and prints the elapsed time
    script = (
        "import time, os, sys; "
        "from jqscore import run, write_qscore_cif; "
        f"t0 = time.time(); "
        f"r = run({entry.cif_path!r}, {entry.map_path!r}, sigma=0.4); "
        "elapsed = time.time() - t0; "
        f"p = write_qscore_cif(r, {entry.cif_path!r}, {entry.map_path!r}, output_dir={output_dir!r}); "
        f"os.rename(p, {expected_out!r}) if p != {expected_out!r} else None; "
        "print(f'ELAPSED={elapsed:.4f}')"
    )

    env = os.environ.copy()
    if backend_name == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
    else:
        # Let JAX pick the best available (gpu > tpu > cpu)
        env.pop("JAX_PLATFORMS", None)

    print(f"  Running jqscore ({backend_name}) on {entry.pdb_id} [{python_exe}]...")
    result = subprocess.run(
        [python_exe, "-c", script],
        env=env, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  jqscore ({backend_name}) stderr:\n{result.stderr[:1000]}")
        raise RuntimeError(f"jqscore ({backend_name}) failed for {entry.pdb_id}")

    # Parse elapsed time from stdout
    elapsed = 0.0
    for line in result.stdout.strip().splitlines():
        if line.startswith("ELAPSED="):
            elapsed = float(line.split("=", 1)[1])
    print(f"  jqscore ({backend_name}) finished in {elapsed:.1f}s")

    return expected_out, elapsed


def run_jqscore(
    entry: EntryInfo,
    output_dir: str,
    no_cache: bool,
    gpu_python: str | None = None,
) -> tuple[str, float, float, bool]:
    """Run jqscore on CPU and GPU, return (cif_path, cpu_time, gpu_time, gpu_available)."""
    map_basename = os.path.basename(entry.map_path)

    # --- CPU run ---
    cpu_out = os.path.join(
        output_dir, f"jax_cpu_{entry.pdb_id}.cif__Q__{map_basename}.cif"
    )
    if os.path.exists(cpu_out) and not no_cache:
        print(f"  jqscore (cpu) cached: {cpu_out}")
        cpu_time = 0.0
    else:
        cpu_out, cpu_time = _run_jqscore_subprocess(entry, output_dir, "cpu")

    # --- GPU run ---
    gpu_out = os.path.join(
        output_dir, f"jax_gpu_{entry.pdb_id}.cif__Q__{map_basename}.cif"
    )
    gpu_time = 0.0
    gpu_available = False
    if os.path.exists(gpu_out) and not no_cache:
        print(f"  jqscore (gpu) cached: {gpu_out}")
        gpu_available = True
    else:
        try:
            gpu_out, gpu_time = _run_jqscore_subprocess(
                entry, output_dir, "gpu", python_exe=gpu_python,
            )
            gpu_available = True
        except RuntimeError:
            print("  jqscore (gpu) not available, skipping")

    # Use GPU output for comparison if available, else CPU
    jqscore_cif = gpu_out if gpu_available else cpu_out
    return jqscore_cif, cpu_time, gpu_time, gpu_available


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def count_non_hydrogen(cif_path: str) -> int:
    """Count non-hydrogen atoms from a Q-score CIF using atom labels."""
    atom_info, qscores = read_qscores(cif_path)
    n_non_h = sum(
        1 for name in atom_info.atom_name
        if not str(name).strip().startswith("H")
    )
    return n_non_h


def plot_structure_scatter(result: BenchmarkResult, output_dir: str) -> str:
    """Per-structure scatter: jqscore vs mapq per non-H atom."""
    q_j = result.q_jqscore
    q_m = result.q_mapq
    valid = np.isfinite(q_j) & np.isfinite(q_m)

    # Filter to non-H (read from jqscore CIF)
    atom_info, _ = read_qscores(result.jqscore_cif)
    non_h = np.array([
        not str(name).strip().startswith("H")
        for name in atom_info.atom_name
    ])
    mask = valid & non_h

    q_j_plot = q_j[mask]
    q_m_plot = q_m[mask]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(q_m_plot, q_j_plot, s=2, alpha=0.3, rasterized=True)
    ax.plot([0, 1], [0, 1], "r--", lw=0.8, label="y = x")

    r = np.corrcoef(q_j_plot, q_m_plot)[0, 1] if len(q_j_plot) > 1 else float("nan")
    entry = result.entry
    title = (
        f"{entry.pdb_id} / {entry.emdb_id} "
        f"(res={entry.resolution:.2f} A, "
        f"N_non-H={result.n_non_h})"
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("MapQ Q-score")
    ax.set_ylabel("jqscore Q-score")
    ax.text(
        0.05, 0.92, f"r = {r:.6f}",
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.set_xlim(-0.2, 1.05)
    ax.set_ylim(-0.2, 1.05)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)

    path = os.path.join(output_dir, f"{entry.pdb_id}_scatter.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def plot_summary_scatter(results: list[BenchmarkResult], output_dir: str) -> str:
    """Structure-level comparison: mean Q-scores and Pearson r sorted by atom count."""
    # Sort by number of non-H atoms
    results_sorted = sorted(results, key=lambda r: r.n_non_h)

    pdb_ids = []
    n_atoms = []
    mean_j = []
    mean_m = []
    pearson_rs = []

    for res in results_sorted:
        q_j = res.q_jqscore
        q_m = res.q_mapq
        valid = np.isfinite(q_j) & np.isfinite(q_m)

        atom_info, _ = read_qscores(res.jqscore_cif)
        non_h = np.array([
            not str(name).strip().startswith("H")
            for name in atom_info.atom_name
        ])
        mask = valid & non_h

        if mask.sum() == 0:
            continue

        pdb_ids.append(res.entry.pdb_id)
        n_atoms.append(res.n_non_h)
        mean_j.append(float(np.mean(q_j[mask])))
        mean_m.append(float(np.mean(q_m[mask])))
        pearson_rs.append(res.pearson_r)

    mean_j = np.array(mean_j)
    mean_m = np.array(mean_m)
    n_atoms = np.array(n_atoms)
    pearson_rs = np.array(pearson_rs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Top panel: mean Q-scores
    ax1.plot(n_atoms, mean_m, "o-", ms=8, label="MapQ", color="#4878CF")
    ax1.plot(n_atoms, mean_j, "s-", ms=8, label="jqscore", color="#6ACC65")
    for i, pid in enumerate(pdb_ids):
        ax1.annotate(
            pid, (n_atoms[i], max(mean_j[i], mean_m[i])),
            textcoords="offset points", xytext=(0, 10),
            fontsize=8, ha="center", color="0.3",
        )
    ax1.set_ylabel("Mean Q-score (non-H)")
    ax1.set_title("Structure-level comparison (sorted by atom count)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Pearson r
    ax2.plot(n_atoms, pearson_rs, "D-", ms=8, color="#D65F5F")
    for i, pid in enumerate(pdb_ids):
        ax2.annotate(
            pid, (n_atoms[i], pearson_rs[i]),
            textcoords="offset points", xytext=(0, 10),
            fontsize=8, ha="center", color="0.3",
        )
    ax2.set_xlabel("Number of non-H atoms")
    ax2.set_ylabel("Pearson r (jqscore vs MapQ)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "summary_scatter.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def plot_runtimes(results: list[BenchmarkResult], output_dir: str) -> str:
    """Line+scatter plot: runtime vs number of non-H atoms."""
    # Sort by atom count for a meaningful line
    results_sorted = sorted(results, key=lambda r: r.n_non_h)
    n_atoms = [r.n_non_h for r in results_sorted]
    t_mapq = [r.mapq_time for r in results_sorted]
    t_cpu = [r.jqscore_cpu_time for r in results_sorted]
    t_gpu = [r.jqscore_gpu_time for r in results_sorted]
    labels = [r.entry.pdb_id for r in results_sorted]
    has_gpu = any(r.gpu_available for r in results_sorted)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Filter out zero-time (cached) entries per series
    def _plot(x, y, **kw):
        xf = [xi for xi, yi in zip(x, y) if yi > 0]
        yf = [yi for yi in y if yi > 0]
        if xf:
            ax.plot(xf, yf, **kw)

    _plot(n_atoms, t_mapq, marker="o", ms=8, label="MapQ (np=48)", color="#4878CF")
    _plot(n_atoms, t_cpu, marker="s", ms=8, label="jqscore (CPU)", color="#D65F5F")
    if has_gpu:
        _plot(n_atoms, t_gpu, marker="D", ms=8, label="jqscore (GPU)", color="#6ACC65")

    # Label each point with PDB ID
    for i, lbl in enumerate(labels):
        for y_vals, va_off in [(t_mapq, -12), (t_cpu, 8), (t_gpu if has_gpu else [], 8)]:
            if i < len(y_vals) and y_vals[i] > 0:
                ax.annotate(
                    lbl, (n_atoms[i], y_vals[i]),
                    textcoords="offset points", xytext=(5, va_off),
                    fontsize=7, color="0.3",
                )

    ax.set_xlabel("Number of non-H atoms")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Runtime vs structure size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "runtimes.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def benchmark_entry(
    pdb_id: str,
    output_dir: str,
    cache_dir: str,
    mapq_cmd_template: str,
    no_cache: bool,
    gpu_python: str | None = None,
) -> BenchmarkResult:
    """Run full benchmark for one PDB entry."""
    entry = prepare_entry(pdb_id, cache_dir, no_cache)

    # Run MapQ
    mapq_cif, mapq_time = run_mapq(entry, output_dir, mapq_cmd_template, no_cache)

    # Run jqscore (CPU + GPU)
    jqscore_cif, cpu_time, gpu_time, gpu_available = run_jqscore(
        entry, output_dir, no_cache, gpu_python=gpu_python,
    )

    # Compare
    cmp = compare_qscores(jqscore_cif, mapq_cif)
    n_non_h = count_non_hydrogen(jqscore_cif)

    result = BenchmarkResult(
        entry=entry,
        jqscore_cif=jqscore_cif,
        mapq_cif=mapq_cif,
        jqscore_cpu_time=cpu_time,
        jqscore_gpu_time=gpu_time,
        mapq_time=mapq_time,
        n_non_h=n_non_h,
        pearson_r=cmp.pearson_r,
        diff_mean=cmp.diff_mean,
        diff_max=cmp.diff_max,
        diff_min=cmp.diff_min,
        q_jqscore=cmp.q1,
        q_mapq=cmp.q2,
        gpu_available=gpu_available,
    )

    print(f"\n  Results for {pdb_id}:")
    print(f"    Pearson r:  {cmp.pearson_r:.6f}")
    print(f"    Diff mean:  {cmp.diff_mean:.6f}")
    print(f"    Diff range: [{cmp.diff_min:.6f}, {cmp.diff_max:.6f}]")
    print(f"    Non-H atoms: {n_non_h}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark jqscore against MapQ using real PDB+EMDB data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "pdb_ids", nargs="*", default=DEFAULT_PDB_IDS,
        help=f"PDB IDs to benchmark (default: {DEFAULT_PDB_IDS})",
    )
    parser.add_argument(
        "-o", "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Cache directory for downloads (default: <output-dir>/cache)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", default=False,
        help="Ignore all cached files and re-download/re-run everything",
    )
    parser.add_argument(
        "--gpu-python", default=DEFAULT_GPU_PYTHON,
        help=(
            "Python interpreter with CUDA-enabled JAX for GPU runs "
            f"(default: {DEFAULT_GPU_PYTHON})"
        ),
    )

    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    cache_dir = args.cache_dir or os.path.join(output_dir, "cache")
    mapq_cmd_template = os.environ.get("MAPQ_CMD", DEFAULT_MAPQ_CMD)
    gpu_python = args.gpu_python

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    pdb_ids = [pid.upper() for pid in args.pdb_ids]

    print(f"Output dir:  {output_dir}")
    print(f"Cache dir:   {cache_dir}")
    print(f"PDB IDs:     {pdb_ids}")
    print(f"MapQ cmd:    {mapq_cmd_template}")
    print(f"GPU Python:  {gpu_python}")
    print(f"No-cache:    {args.no_cache}")

    # Run benchmarks
    results: list[BenchmarkResult] = []
    for pdb_id in pdb_ids:
        try:
            result = benchmark_entry(
                pdb_id, output_dir, cache_dir, mapq_cmd_template, args.no_cache,
                gpu_python=gpu_python,
            )
            results.append(result)
        except Exception as exc:
            print(f"\n  ERROR benchmarking {pdb_id}: {exc}")
            continue

    if not results:
        print("\nNo successful benchmarks. Exiting.")
        sys.exit(1)

    # Generate plots
    print(f"\n{'='*60}")
    print("Generating plots")
    print(f"{'='*60}")

    # Per-structure scatter plots
    for result in results:
        plot_structure_scatter(result, output_dir)

    # Summary plots (only meaningful with >1 structure)
    if len(results) > 1:
        plot_summary_scatter(results, output_dir)

    # Runtimes (only if we have timing data)
    has_times = any(
        r.mapq_time > 0 or r.jqscore_cpu_time > 0 or r.jqscore_gpu_time > 0
        for r in results
    )
    if has_times:
        plot_runtimes(results, output_dir)

    # Print summary table
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'PDB':>6} {'EMDB':>12} {'Res':>5} {'N_nonH':>7} "
          f"{'r':>9} {'mean_diff':>10} {'max_diff':>10} "
          f"{'t_mapq':>8} {'t_cpu':>8} {'t_gpu':>8}")
    print("-" * 100)
    for r in results:
        gpu_str = f"{r.jqscore_gpu_time:>7.1f}s" if r.gpu_available else "    n/a"
        print(
            f"{r.entry.pdb_id:>6} {r.entry.emdb_id:>12} "
            f"{r.entry.resolution:>5.2f} {r.n_non_h:>7} "
            f"{r.pearson_r:>9.6f} {r.diff_mean:>10.6f} {r.diff_max:>10.6f} "
            f"{r.mapq_time:>7.1f}s {r.jqscore_cpu_time:>7.1f}s {gpu_str}"
        )

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
