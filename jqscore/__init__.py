"""jqscore — JAX-accelerated Q-score (cryo-EM atom-map fit metric)."""

from .jqscore import (
    AtomRecord,
    QScoreResult,
    read_mrc,
    read_pdb,
    read_cif,
    compute_qscores,
    qscores_from_arrays,
    block_qscores,
    add_qscore_column,
    AtomSiteColumns,
    run,
    write_qscore_cif,
    residue_qscores,
    chain_qscores,
    backend,
    _cli,  # re-exported for the `jqscore` console_scripts entry point
)
from .compare import CompareResult, read_qscores, compare_qscores

__all__ = [
    "AtomRecord",
    "QScoreResult",
    "read_mrc",
    "read_pdb",
    "read_cif",
    "compute_qscores",
    "qscores_from_arrays",
    "block_qscores",
    "add_qscore_column",
    "AtomSiteColumns",
    "run",
    "write_qscore_cif",
    "residue_qscores",
    "chain_qscores",
    "backend",
    "CompareResult",
    "read_qscores",
    "compare_qscores",
]
