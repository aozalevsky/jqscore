"""Q-score comparison utilities for jqscore output CIF files."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import biotite.structure.io.pdbx as pdbx


@dataclass
class CompareResult:
    """Result of comparing Q-scores from two CIF files."""

    n_valid: int
    n_total: int
    diff_min: float
    diff_max: float
    diff_mean: float
    pearson_r: float
    # Per-atom arrays (NaN where either file has no Q-score)
    q1: np.ndarray
    q2: np.ndarray
    diff: np.ndarray
    atom_labels: list[str]


def read_qscores(cif_path: str) -> tuple:
    """Read structure and Q-score column from a jqscore output CIF.

    Parameters
    ----------
    cif_path : str
        Path to a CIF file with ``_atom_site.Q-score``.

    Returns
    -------
    structure : biotite AtomArray
    qscores : ndarray of float64 (NaN where missing)
    """
    cif = pdbx.CIFFile.read(cif_path)
    structure = pdbx.get_structure(cif, model=1, altloc="all", extra_fields=["Q-score"])
    if "Q-score" not in structure.get_annotation_categories():
        raise ValueError(f"No _atom_site.Q-score in {cif_path}")
    raw = structure.get_annotation("Q-score")
    out = np.full(len(raw), np.nan, dtype=np.float64)
    for i, s in enumerate(raw):
        s = str(s).strip()
        if s and s not in ("?", "."):
            try:
                out[i] = float(s)
            except ValueError:
                pass
    return structure, out


def _atom_label(structure, i: int) -> str:
    return (
        f"{structure.chain_id[i]}/{structure.res_id[i]}"
        f"/{structure.res_name[i]}/{structure.atom_name[i]}"
    )


def compare_qscores(path1: str, path2: str) -> CompareResult:
    """Compare Q-scores from two jqscore output CIF files.

    Parameters
    ----------
    path1, path2 : str
        Paths to CIF files with ``_atom_site.Q-score``.

    Returns
    -------
    CompareResult
        Statistics and per-atom arrays.  ``diff = q1 - q2``.
    """
    struct1, q1 = read_qscores(path1)
    struct2, q2 = read_qscores(path2)
    if len(q1) != len(q2):
        raise ValueError(
            f"Length mismatch: {path1} has {len(q1)}, {path2} has {len(q2)}"
        )

    diff = q1 - q2
    valid = np.isfinite(q1) & np.isfinite(q2)
    n_valid = int(np.sum(valid))

    diff_valid = diff[valid]
    d_min = float(np.min(diff_valid)) if n_valid else float("nan")
    d_mean = float(np.mean(diff_valid)) if n_valid else float("nan")
    d_max = float(np.max(diff_valid)) if n_valid else float("nan")
    r = float(np.corrcoef(q1[valid], q2[valid])[0, 1]) if n_valid > 1 else float("nan")

    labels = [_atom_label(struct1, i) for i in range(len(q1))]

    return CompareResult(
        n_valid=n_valid,
        n_total=len(q1),
        diff_min=d_min,
        diff_max=d_max,
        diff_mean=d_mean,
        pearson_r=r,
        q1=q1,
        q2=q2,
        diff=diff,
        atom_labels=labels,
    )


def _print_result(result: CompareResult, path1: str, path2: str, n_extremes: int = 3) -> None:
    print(f"Valid pairs: {result.n_valid} / {result.n_total}")
    print(
        f"Diff (file1 - file2):  min = {result.diff_min:.6f}"
        f"  mean = {result.diff_mean:.6f}  max = {result.diff_max:.6f}"
    )
    print(f"Pearson r: {result.pearson_r:.6f}")

    def extremes(indices, label):
        print(f"\n{label}:")
        for i in indices[:n_extremes]:
            print(
                f"  {result.atom_labels[i]}"
                f"  Q1={result.q1[i]:.6f}  Q2={result.q2[i]:.6f}"
                f"  diff={result.diff[i]:.6f}"
            )

    i_min = np.where(result.diff == result.diff_min)[0]
    i_max = np.where(result.diff == result.diff_max)[0]
    extremes(i_min, "Min diff (file1 - file2)")
    extremes(i_max, "Max diff (file1 - file2)")


def main() -> None:
    """CLI entry point: compare_qscores file1.cif file2.cif"""
    if len(sys.argv) != 3:
        print(main.__doc__, file=sys.stderr)
        print(
            "\nUsage:\n"
            "  compare_qscores file1.cif file2.cif\n"
            "  python -m jqscore.compare file1.cif file2.cif",
            file=sys.stderr,
        )
        sys.exit(1)
    path1, path2 = sys.argv[1], sys.argv[2]
    result = compare_qscores(path1, path2)
    _print_result(result, path1, path2)


if __name__ == "__main__":
    main()
