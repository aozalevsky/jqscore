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

DEFAULT_PDB_IDS = ["4AQW", "9XZK", "9UX2", "9QHX", "8RQB", "3J7O", "8WQL"]
DEFAULT_OUTPUT_DIR = "jqscore_mapq_benchmark"
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
    jqscore_cpu_compute_time: float = 0.0
    jqscore_gpu_compute_time: float = 0.0
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
    emdb_fallback = ""
    em_list = data.get("pdbx_database_related", [])
    for rel in em_list:
        if rel.get("db_name", "").upper() == "EMDB":
            ct = rel.get("content_type", "").lower()
            if ct == "associated em volume":
                emdb_id = rel["db_id"]
                break
            elif not emdb_fallback:
                emdb_fallback = rel["db_id"]
    emdb_id = emdb_id or emdb_fallback

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
    url = f"https://files.wwpdb.org/pub/emdb/structures/{emdb_id}/map/emd_{numeric}.map.gz"
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
) -> tuple[str, float, float]:
    """Run jqscore in a subprocess with a specific JAX backend.

    We use a subprocess so that JAX_PLATFORMS takes effect (JAX caches
    the backend on first use within a process).
    Returns (output_cif_path, total_seconds, compute_seconds).
    """
    map_basename = os.path.basename(entry.map_path)
    prefix = f"jax_{backend_name}_"
    expected_out = os.path.join(
        output_dir, f"{prefix}{entry.pdb_id}.cif__Q__{map_basename}.cif"
    )

    # Inline script: warmup JIT, then time full run and compute separately
    script = (
        "import time, os, sys, numpy as np; "
        "from jqscore.jqscore import compute_qscores, AtomRecord, read_cif, read_mrc; "
        # Warmup: 2 dummy atoms + tiny 4x4x4 map -> triggers JIT compilation
        "dummy_atoms = ["
        "AtomRecord('A',1,'','ALA','CA','C',0.,0.,0.),"
        "AtomRecord('A',1,'','ALA','N','N',1.,0.,0.)"
        "]; "
        "compute_qscores(dummy_atoms, np.zeros((4,4,4),dtype=np.float32), "
        "np.zeros(3), np.ones(3), sigma=0.4); "
        # Full timed run (I/O + compute)
        "t_full = time.time(); "
        f"atoms = read_cif({entry.cif_path!r}); "
        f"data, origin, voxel_size = read_mrc({entry.map_path!r}); "
        "t_compute = time.time(); "
        "r = compute_qscores(atoms, data, origin, voxel_size, sigma=0.4); "
        "compute_elapsed = time.time() - t_compute; "
        "full_elapsed = time.time() - t_full; "
        f"from jqscore import write_qscore_cif; "
        f"p = write_qscore_cif(r, {entry.cif_path!r}, {entry.map_path!r}, output_dir={output_dir!r}); "
        f"os.rename(p, {expected_out!r}) if p != {expected_out!r} else None; "
        "print(f'FULL={full_elapsed:.4f}'); "
        "print(f'COMPUTE={compute_elapsed:.4f}')"
    )

    env = os.environ.copy()
    if backend_name == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
    else:
        # Let JAX pick the best available (gpu > tpu > cpu)
        env.pop("JAX_PLATFORMS", None)

    print(f"  Running jqscore ({backend_name}) on {entry.pdb_id}...")
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  jqscore ({backend_name}) stderr:\n{result.stderr[:1000]}")
        raise RuntimeError(f"jqscore ({backend_name}) failed for {entry.pdb_id}")

    # Parse times from stdout
    full_elapsed = 0.0
    compute_elapsed = 0.0
    for line in result.stdout.strip().splitlines():
        if line.startswith("FULL="):
            full_elapsed = float(line.split("=", 1)[1])
        elif line.startswith("COMPUTE="):
            compute_elapsed = float(line.split("=", 1)[1])
    print(f"  jqscore ({backend_name}) finished: {full_elapsed:.1f}s total, {compute_elapsed:.1f}s compute")

    return expected_out, full_elapsed, compute_elapsed


@dataclass
class _JqscoreTimes:
    cpu_total: float = 0.0
    cpu_compute: float = 0.0
    gpu_total: float = 0.0
    gpu_compute: float = 0.0
    gpu_available: bool = False


def run_jqscore(
    entry: EntryInfo,
    output_dir: str,
    no_cache: bool,
) -> tuple[str, _JqscoreTimes]:
    """Run jqscore on CPU and GPU, return (cif_path, times)."""
    map_basename = os.path.basename(entry.map_path)
    times = _JqscoreTimes()

    # --- CPU run ---
    cpu_out = os.path.join(
        output_dir, f"jax_cpu_{entry.pdb_id}.cif__Q__{map_basename}.cif"
    )
    if os.path.exists(cpu_out) and not no_cache:
        print(f"  jqscore (cpu) cached: {cpu_out}")
    else:
        cpu_out, times.cpu_total, times.cpu_compute = _run_jqscore_subprocess(
            entry, output_dir, "cpu",
        )

    # --- GPU run ---
    gpu_out = os.path.join(
        output_dir, f"jax_gpu_{entry.pdb_id}.cif__Q__{map_basename}.cif"
    )
    if os.path.exists(gpu_out) and not no_cache:
        print(f"  jqscore (gpu) cached: {gpu_out}")
        times.gpu_available = True
    else:
        try:
            gpu_out, times.gpu_total, times.gpu_compute = _run_jqscore_subprocess(
                entry, output_dir, "gpu",
            )
            times.gpu_available = True
        except RuntimeError:
            print("  jqscore (gpu) not available, skipping")

    # Use GPU output for comparison if available, else CPU
    jqscore_cif = gpu_out if times.gpu_available else cpu_out
    return jqscore_cif, times


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
    max_diff = float(np.max(np.abs(q_j_plot - q_m_plot))) if len(q_j_plot) > 0 else float("nan")
    entry = result.entry
    title = (
        f"{entry.pdb_id} / {entry.emdb_id} "
        f"(res={entry.resolution:.2f} A, "
        f"N_non-H={result.n_non_h})\n"
        f"r={r:.6f}, max|diff|={max_diff:.6f}"
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("MapQ Q-score")
    ax.set_ylabel("jqscore Q-score")
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


def plot_summary(results: list[BenchmarkResult], output_dir: str) -> str:
    """Two-panel summary: (top) mean Q-score scatter jqscore vs MapQ,
    (bottom) runtimes vs structure size."""
    results_sorted = sorted(results, key=lambda r: r.n_non_h)

    pdb_ids = []
    n_atoms = []
    mean_j = []
    mean_m = []

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

    mean_j = np.array(mean_j)
    mean_m = np.array(mean_m)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # --- Left panel: scatter of mean Q-scores (jqscore vs MapQ) ---
    lo = min(float(np.min(mean_j)), float(np.min(mean_m))) - 0.02
    hi = max(float(np.max(mean_j)), float(np.max(mean_m))) + 0.02
    ax1.plot([lo, hi], [lo, hi], "r--", lw=0.8, label="y = x")
    ax1.scatter(mean_m, mean_j, s=60, zorder=3)
    for i, pid in enumerate(pdb_ids):
        ax1.annotate(
            pid, (mean_m[i], mean_j[i]),
            textcoords="offset points", xytext=(6, 6),
            fontsize=8, color="0.3",
        )
    ax1.set_xlabel("MapQ mean Q-score (non-H)")
    ax1.set_ylabel("jqscore mean Q-score (non-H)")
    ax1.set_title("Structure-level Q-score agreement")
    ax1.set_aspect("equal")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Shared data for runtime panels ---
    has_gpu = any(r.gpu_available for r in results_sorted)

    def _plot_series(ax, x, y, **kw):
        xf = [xi for xi, yi in zip(x, y) if yi > 0]
        yf = [yi for yi in y if yi > 0]
        if xf:
            ax.plot(xf, yf, **kw)

    def _label_points(ax, x, labels, series_list):
        """One label per structure, by its slowest point."""
        for i, lbl in enumerate(labels):
            y = max(y_vals[i] for y_vals in series_list)
            if y > 0:
                ax.annotate(
                    lbl, (x[i], y),
                    textcoords="offset points", xytext=(5, 8),
                    fontsize=7, color="0.3",
                )

    na = [r.n_non_h for r in results_sorted]
    labels = [r.entry.pdb_id for r in results_sorted]

    # --- Middle panel: compute-only runtimes ---
    t_mapq = [r.mapq_time for r in results_sorted]
    t_cpu_c = [r.jqscore_cpu_compute_time for r in results_sorted]
    t_gpu_c = [r.jqscore_gpu_compute_time for r in results_sorted]

    _plot_series(ax2, na, t_mapq, marker="o", ms=8, label="MapQ (np=48)", color="#4878CF")
    _plot_series(ax2, na, t_cpu_c, marker="s", ms=8, label="jqscore (CPU)", color="#D65F5F")
    if has_gpu:
        _plot_series(ax2, na, t_gpu_c, marker="D", ms=8, label="jqscore (GPU)", color="#6ACC65")

    series_c = [t_mapq, t_cpu_c]
    if has_gpu:
        series_c.append(t_gpu_c)
    _label_points(ax2, na, labels, series_c)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of non-H atoms")
    ax2.set_ylabel("Runtime (seconds)")
    ax2.set_title("Compute time")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # --- Right panel: full runtimes (I/O + compute) ---
    t_cpu_f = [r.jqscore_cpu_time for r in results_sorted]
    t_gpu_f = [r.jqscore_gpu_time for r in results_sorted]

    _plot_series(ax3, na, t_mapq, marker="o", ms=8, label="MapQ (np=48)", color="#4878CF")
    _plot_series(ax3, na, t_cpu_f, marker="s", ms=8, label="jqscore (CPU)", color="#D65F5F")
    if has_gpu:
        _plot_series(ax3, na, t_gpu_f, marker="D", ms=8, label="jqscore (GPU)", color="#6ACC65")

    series_f = [t_mapq, t_cpu_f]
    if has_gpu:
        series_f.append(t_gpu_f)
    _label_points(ax3, na, labels, series_f)

    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Number of non-H atoms")
    ax3.set_ylabel("Runtime (seconds)")
    ax3.set_title("Total time (I/O + compute)")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "summary.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _get_hardware_info() -> dict[str, str]:
    """Detect CPU model, core count, and GPU model."""
    info: dict[str, str] = {}

    # CPU info from /proc/cpuinfo
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
        # Socket, core, and thread counts
        phys_ids: set[str] = set()
        core_ids: set[tuple[str, str]] = set()
        n_threads = 0
        phys_id = core_id = ""
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("physical id"):
                    phys_id = line.split(":", 1)[1].strip()
                    phys_ids.add(phys_id)
                elif line.startswith("core id"):
                    core_id = line.split(":", 1)[1].strip()
                    core_ids.add((phys_id, core_id))
                elif line.startswith("processor"):
                    n_threads += 1
        n_sockets = len(phys_ids) if phys_ids else 1
        n_cores = len(core_ids) if core_ids else (os.cpu_count() or 0)
        n_threads = n_threads or (os.cpu_count() or 0)
        info["cpu_sockets"] = str(n_sockets)
        info["cpu_cores"] = str(n_cores)
        info["cpu_threads"] = str(n_threads)
    except OSError:
        info["cpu_model"] = "unknown"
        info["cpu_cores"] = str(os.cpu_count() or "?")

    # GPU info from nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            gpus = [g.strip() for g in result.stdout.strip().splitlines() if g.strip()]
            info["gpu_model"] = gpus[0] if gpus else "not available"
        else:
            info["gpu_model"] = "not available"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["gpu_model"] = "not available"

    return info


def _parse_mapq_np(mapq_cmd_template: str) -> str:
    """Extract np=N value from MapQ command template."""
    import re
    m = re.search(r"np=(\d+)", mapq_cmd_template)
    return m.group(1) if m else "?"


def write_summary_md(
    results: list[BenchmarkResult],
    output_dir: str,
    mapq_cmd_template: str,
) -> str:
    """Write a markdown executive summary to output_dir/README.md."""
    from datetime import date

    hw = _get_hardware_info()
    mapq_np = _parse_mapq_np(mapq_cmd_template)

    lines: list[str] = []
    w = lines.append

    w("# jqscore vs MapQ Benchmark Summary")
    w("")
    w(f"**Date:** {date.today().isoformat()}  ")
    w(f"**Structures:** {len(results)}  ")
    w(f"**sigma:** 0.4  ")
    w("")
    w("## Hardware")
    w("")
    w(f"| Component | Details |")
    w(f"|-----------|---------|")
    w(f"| CPU | {hw.get('cpu_model', 'unknown')} |")
    w(f"| CPU sockets / cores / threads | {hw.get('cpu_sockets', '?')} / {hw.get('cpu_cores', '?')} / {hw.get('cpu_threads', '?')} |")
    w(f"| GPU | {hw.get('gpu_model', 'not available')} |")
    w(f"| MapQ np | {mapq_np} |")
    w("")

    # Aggregate stats
    rs = [r.pearson_r for r in results]
    max_diffs = [max(abs(r.diff_min), abs(r.diff_max)) for r in results]
    w("## Accuracy")
    w("")
    w(f"| Metric | Min | Mean | Max |")
    w(f"|--------|-----|------|-----|")
    w(f"| Pearson r | {min(rs):.6f} | {np.mean(rs):.6f} | {max(rs):.6f} |")
    w(f"| Max \\|diff\\| | {min(max_diffs):.6f} | {np.mean(max_diffs):.6f} | {max(max_diffs):.6f} |")
    w("")

    # Per-structure table
    w("## Per-structure results")
    w("")
    has_gpu = any(r.gpu_available for r in results)
    has_times = any(
        r.mapq_time > 0 or r.jqscore_cpu_time > 0 or r.jqscore_gpu_time > 0
        for r in results
    )
    hdr = "| PDB | EMDB | Res (A) | Non-H atoms | r | mean diff | max \\|diff\\| |"
    sep = "|-----|------|---------|-------------|---|-----------|-------------|"
    if has_times:
        hdr += " MapQ (s) | CPU compute (s) | CPU total (s) |"
        sep += "----------|-----------------|---------------|"
        if has_gpu:
            hdr += " GPU compute (s) | GPU total (s) |"
            sep += "-----------------|---------------|"
    w(hdr)
    w(sep)
    for r in results:
        md = max(abs(r.diff_min), abs(r.diff_max))
        row = (
            f"| {r.entry.pdb_id} | {r.entry.emdb_id} "
            f"| {r.entry.resolution:.2f} | {r.n_non_h} "
            f"| {r.pearson_r:.6f} | {r.diff_mean:.6f} | {md:.6f} |"
        )
        if has_times:
            row += (
                f" {r.mapq_time:.1f} "
                f"| {r.jqscore_cpu_compute_time:.1f} "
                f"| {r.jqscore_cpu_time:.1f} |"
            )
            if has_gpu:
                if r.gpu_available:
                    row += (
                        f" {r.jqscore_gpu_compute_time:.1f} "
                        f"| {r.jqscore_gpu_time:.1f} |"
                    )
                else:
                    row += " n/a | n/a |"
        w(row)
    w("")

    # Plots
    w("## Plots")
    w("")
    for r in results:
        w(f"### {r.entry.pdb_id}")
        w(f"![{r.entry.pdb_id} scatter]({r.entry.pdb_id}_scatter.png)")
        w("")
    if len(results) > 1:
        w("### Summary")
        w("![Summary](summary.png)")
        w("")

    path = os.path.join(output_dir, "README.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {path}")
    return path


def _stats_path(pdb_id: str, output_dir: str) -> str:
    return os.path.join(output_dir, f"{pdb_id}_stats.json")


def _save_stats(result: BenchmarkResult, output_dir: str) -> None:
    """Save benchmark stats to a JSON file."""
    data = {
        "pdb_id": result.entry.pdb_id,
        "emdb_id": result.entry.emdb_id,
        "resolution": result.entry.resolution,
        "cif_path": result.entry.cif_path,
        "map_path": result.entry.map_path,
        "jqscore_cif": result.jqscore_cif,
        "mapq_cif": result.mapq_cif,
        "jqscore_cpu_time": result.jqscore_cpu_time,
        "jqscore_gpu_time": result.jqscore_gpu_time,
        "jqscore_cpu_compute_time": result.jqscore_cpu_compute_time,
        "jqscore_gpu_compute_time": result.jqscore_gpu_compute_time,
        "mapq_time": result.mapq_time,
        "n_non_h": result.n_non_h,
        "pearson_r": result.pearson_r,
        "diff_mean": result.diff_mean,
        "diff_max": result.diff_max,
        "diff_min": result.diff_min,
        "gpu_available": result.gpu_available,
    }
    path = _stats_path(result.entry.pdb_id, output_dir)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved stats: {path}")


def _load_stats(pdb_id: str, output_dir: str) -> BenchmarkResult | None:
    """Load cached stats from JSON. Returns None if missing or invalid."""
    path = _stats_path(pdb_id, output_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        entry = EntryInfo(
            pdb_id=data["pdb_id"],
            emdb_id=data["emdb_id"],
            resolution=data["resolution"],
            cif_path=data["cif_path"],
            map_path=data["map_path"],
        )
        # Re-read per-atom arrays from the output CIFs (cheap)
        cmp = compare_qscores(data["jqscore_cif"], data["mapq_cif"])
        return BenchmarkResult(
            entry=entry,
            jqscore_cif=data["jqscore_cif"],
            mapq_cif=data["mapq_cif"],
            jqscore_cpu_time=data["jqscore_cpu_time"],
            jqscore_gpu_time=data["jqscore_gpu_time"],
            jqscore_cpu_compute_time=data.get("jqscore_cpu_compute_time", 0.0),
            jqscore_gpu_compute_time=data.get("jqscore_gpu_compute_time", 0.0),
            mapq_time=data["mapq_time"],
            n_non_h=data["n_non_h"],
            pearson_r=data["pearson_r"],
            diff_mean=data["diff_mean"],
            diff_max=data["diff_max"],
            diff_min=data["diff_min"],
            q_jqscore=cmp.q1,
            q_mapq=cmp.q2,
            gpu_available=data["gpu_available"],
        )
    except (KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"  Warning: could not load cached stats for {pdb_id}: {exc}")
        return None


def benchmark_entry(
    pdb_id: str,
    output_dir: str,
    cache_dir: str,
    mapq_cmd_template: str,
    no_cache: bool,
    rerun_jqscore: bool = False,
) -> BenchmarkResult:
    """Run full benchmark for one PDB entry."""
    pdb_id = pdb_id.upper()

    # Check for cached stats
    cached = None if no_cache else _load_stats(pdb_id, output_dir)
    if cached is not None and not rerun_jqscore:
        print(f"\n  Using cached stats for {pdb_id}")
        print(f"    Pearson r:  {cached.pearson_r:.6f}")
        print(f"    Non-H atoms: {cached.n_non_h}")
        return cached

    entry = prepare_entry(pdb_id, cache_dir, no_cache)

    # Run MapQ
    mapq_cif, mapq_time = run_mapq(entry, output_dir, mapq_cmd_template, no_cache)
    if mapq_time == 0.0 and cached is not None:
        # MapQ output was cached: keep the time recorded when it ran
        mapq_time = cached.mapq_time

    # Run jqscore (CPU + GPU)
    jqscore_cif, jqs_times = run_jqscore(entry, output_dir, no_cache or rerun_jqscore)

    # Compare
    cmp = compare_qscores(jqscore_cif, mapq_cif)
    n_non_h = count_non_hydrogen(jqscore_cif)

    result = BenchmarkResult(
        entry=entry,
        jqscore_cif=jqscore_cif,
        mapq_cif=mapq_cif,
        jqscore_cpu_time=jqs_times.cpu_total,
        jqscore_gpu_time=jqs_times.gpu_total,
        jqscore_cpu_compute_time=jqs_times.cpu_compute,
        jqscore_gpu_compute_time=jqs_times.gpu_compute,
        mapq_time=mapq_time,
        n_non_h=n_non_h,
        pearson_r=cmp.pearson_r,
        diff_mean=cmp.diff_mean,
        diff_max=cmp.diff_max,
        diff_min=cmp.diff_min,
        q_jqscore=cmp.q1,
        q_mapq=cmp.q2,
        gpu_available=jqs_times.gpu_available,
    )

    # Save stats for future runs
    _save_stats(result, output_dir)

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
        "--rerun-jqscore", action="store_true", default=False,
        help="Re-run jqscore but keep cached downloads and MapQ results",
    )

    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    cache_dir = args.cache_dir or os.path.join(output_dir, "cache")
    mapq_cmd_template = os.environ.get("MAPQ_CMD", DEFAULT_MAPQ_CMD)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    pdb_ids = [pid.upper() for pid in args.pdb_ids]

    print(f"Output dir:  {output_dir}")
    print(f"Cache dir:   {cache_dir}")
    print(f"PDB IDs:     {pdb_ids}")
    print(f"MapQ cmd:    {mapq_cmd_template}")
    print(f"No-cache:    {args.no_cache}")

    # Run benchmarks
    results: list[BenchmarkResult] = []
    for pdb_id in pdb_ids:
        try:
            result = benchmark_entry(
                pdb_id, output_dir, cache_dir, mapq_cmd_template, args.no_cache,
                args.rerun_jqscore,
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

    # Summary plot (only meaningful with >1 structure)
    if len(results) > 1:
        plot_summary(results, output_dir)

    # Print summary table
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'PDB':>6} {'EMDB':>12} {'Res':>5} {'N_nonH':>7} "
          f"{'r':>9} {'mean_diff':>10} {'max_diff':>10} "
          f"{'t_mapq':>8} {'cpu_comp':>9} {'cpu_tot':>9} "
          f"{'gpu_comp':>9} {'gpu_tot':>9}")
    print("-" * 120)
    for r in results:
        if r.gpu_available:
            gpu_c = f"{r.jqscore_gpu_compute_time:>8.1f}s"
            gpu_t = f"{r.jqscore_gpu_time:>8.1f}s"
        else:
            gpu_c = gpu_t = "      n/a"
        print(
            f"{r.entry.pdb_id:>6} {r.entry.emdb_id:>12} "
            f"{r.entry.resolution:>5.2f} {r.n_non_h:>7} "
            f"{r.pearson_r:>9.6f} {r.diff_mean:>10.6f} {r.diff_max:>10.6f} "
            f"{r.mapq_time:>7.1f}s "
            f"{r.jqscore_cpu_compute_time:>8.1f}s {r.jqscore_cpu_time:>8.1f}s "
            f"{gpu_c} {gpu_t}"
        )

    # Write markdown summary
    write_summary_md(results, output_dir, mapq_cmd_template)

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
