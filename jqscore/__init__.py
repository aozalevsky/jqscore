"""jqscore — JAX-accelerated Q-score (cryo-EM atom-map fit metric)."""

from .jqscore import (
    AtomRecord,
    QScoreResult,
    read_mrc,
    read_pdb,
    read_cif,
    compute_qscores,
    run,
    write_qscore_cif,
    residue_qscores,
    chain_qscores,
    backend,
    _cli,  # re-exported for the `jqscore` console_scripts entry point
)

__all__ = [
    "AtomRecord",
    "QScoreResult",
    "read_mrc",
    "read_pdb",
    "read_cif",
    "compute_qscores",
    "run",
    "write_qscore_cif",
    "residue_qscores",
    "chain_qscores",
    "backend",
]
