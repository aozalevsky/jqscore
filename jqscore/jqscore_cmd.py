"""mapq_cmd.py-compatible CLI entry point for jqscore.

This script accepts the same argument syntax as the original mapq_cmd.py that
ships with UCSF Chimera, so that tools like the VA validation package can use
jqscore as a drop-in replacement without modification.

Usage
-----
    jqscore-cmd <chimera_root> map=<mapfile> cif=<cif1> [cif=<cif2> ...] \\
                np=<num_cores> res=<resolution> sigma=<sigma>

Arguments
---------
chimera_root
    Path to the Chimera installation root.  Accepted for compatibility but
    ignored — jqscore does not require Chimera.
map=<path>
    Path to the MRC / CCP4 density map.
cif=<path>
    Path to an mmCIF structure file.  May be repeated for multiple models.
np=<int>
    Number of parallel cores.  Accepted for compatibility; jqscore currently
    runs single-threaded on the CPU (JAX handles its own parallelism).
res=<float>
    Map resolution in Å.  Accepted for compatibility but not used by the
    Q-score algorithm.
sigma=<float>
    Gaussian sigma in Å (default 0.4).

Output
------
For each model ``foo.cif`` and map ``bar.mrc`` the script writes::

    {cwd}/foo.cif__Q__bar.mrc.cif

where *cwd* is the working directory at invocation time — matching the
filename convention expected by VA (``validationanalysis.py``).
"""

from __future__ import annotations

import os
import sys

from jqscore.jqscore import run, write_qscore_cif

def _mapq_cli() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    # First positional arg is chimera_root — accepted but ignored.
    # Remaining args are key=value pairs.
    cif_files: list[str] = []
    params: dict[str, str] = {}

    for arg in args[1:]:
        if "=" in arg:
            key, _, val = arg.partition("=")
            key_lc = key.strip().lower()
            if key_lc == "cif":
                cif_files.append(val.strip())
            else:
                params[key_lc] = val.strip()
        # bare positional args after chimera_root are silently ignored

    map_file = params.get("map")
    if not map_file:
        print("Error: map= argument is required", file=sys.stderr)
        sys.exit(1)

    if not cif_files:
        print("Error: at least one cif= argument is required", file=sys.stderr)
        sys.exit(1)

    sigma = float(params.get("sigma", 0.4))
    # np= and res= are accepted for compatibility; jqscore ignores them.

    output_dir = os.getcwd()  # VA expects output files in the working directory

    for cif_path in cif_files:
        print(f"Q-score: processing {cif_path}", flush=True)
        try:
            result = run(cif_path, map_file, sigma=sigma)
            out_path = write_qscore_cif(result, cif_path, map_file, output_dir=output_dir)
            print(f"Q-score: wrote {out_path}", flush=True)
        except Exception as exc:
            # VA watches for the word "error" in stdout/stderr
            print(f"Q-score error processing {cif_path}: {exc}", file=sys.stderr, flush=True)
            sys.exit(1)


if __name__ == "__main__":
    _mapq_cli()
