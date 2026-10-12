"""Q-scores from a parsed mmCIF block: the same atoms, scores and output file as the AtomRecord path."""

import filecmp

import gemmi
import numpy as np
import pytest

from jqscore.jqscore import (
    _atom_site_arrays,
    _residue_codes,
    add_qscore_column,
    block_qscores,
    compute_qscores,
    read_cif,
    read_mrc,
    write_qscore_cif,
)

from .conftest import SMALL_STRUCTURE, TEST_DATA, XZAK_MAP, XZAK_STRUCTURE

TAGS = ("group_PDB id type_symbol label_atom_id label_alt_id label_comp_id label_asym_id "
        "label_entity_id label_seq_id pdbx_PDB_ins_code Cartn_x Cartn_y Cartn_z occupancy "
        "B_iso_or_equiv auth_seq_id auth_asym_id pdbx_PDB_model_num").split()

# Chain A, chain B, a ligand of chain A after both, a residue with an insertion
# code, an atom with two conformations, and the whole again as a second model
ROWS = [
    "ATOM 1 N N . GLY A 1 1 ? 0.000 0.000 0.000 1.00 10.0 1 A 1",
    "ATOM 2 C CA . GLY A 1 1 ? 1.450 0.000 0.000 1.00 10.0 1 A 1",
    "ATOM 3 H HA2 . GLY A 1 1 ? 1.800 0.900 0.000 1.00 10.0 1 A 1",
    "ATOM 4 N N . GLY A 1 2 A 2.500 1.000 0.000 1.00 10.0 1 A 1",
    "ATOM 5 N N . SER B 1 1 ? 10.000 0.000 0.000 1.00 10.0 1 B 1",
    "ATOM 6 C CA A SER B 1 1 ? 11.450 0.000 0.000 0.60 10.0 1 B 1",
    "ATOM 7 C CA B SER B 1 1 ? 11.460 0.100 0.000 0.40 10.0 1 B 1",
    "HETATM 8 ZN ZN . ZN C 2 . ? 3.000 1.000 0.000 1.00 10.0 101 A 1",
]


def _write(path, rows, tags=TAGS):
    path.write_text("data_test\nloop_\n" + "".join(f"_atom_site.{t}\n" for t in tags)
                    + "\n".join(rows) + "\n")
    return str(path)


def _same_partition(a, b):
    """Whether two code arrays group the same elements together."""
    pairs = np.unique(np.stack([a, b]), axis=1).shape[1]
    return pairs == len(np.unique(a)) == len(np.unique(b))


def _check_against_read_cif(path):
    rows, coords, heavy, residue = _atom_site_arrays(gemmi.cif.read(path)[0])
    atoms = read_cif(path)
    np.testing.assert_array_equal(coords, [(a.x, a.y, a.z) for a in atoms])
    np.testing.assert_array_equal(heavy, [a.element != "H" for a in atoms])
    assert _same_partition(residue, _residue_codes(atoms))
    return rows


@pytest.mark.parametrize("path", sorted(TEST_DATA.glob("*.cif")), ids=lambda p: p.name)
def test_atoms_as_read_cif_reads_them(path):
    _check_against_read_cif(str(path))


def test_ligand_after_the_chains(tmp_path):
    rows = _check_against_read_cif(_write(tmp_path / "t.cif", ROWS))
    assert rows.tolist() == [0, 1, 2, 3, 7, 4, 5, 6]


def test_later_models_left_out(tmp_path):
    second = [r.replace(" 1\n", " 2\n").rsplit(" ", 1)[0] + " 2" for r in ROWS]
    rows = _check_against_read_cif(_write(tmp_path / "t.cif", ROWS + second))
    assert rows.max() < len(ROWS)


def test_without_type_symbol(tmp_path):
    """read_cif (gemmi) reads no model without type_symbol; hydrogen then goes by name."""
    tags = [t for t in TAGS if t != "type_symbol"]
    rows = [" ".join(f for i, f in enumerate(r.split()) if i != 2) for r in ROWS]
    _, _, heavy, _ = _atom_site_arrays(gemmi.cif.read(_write(tmp_path / "t.cif", rows, tags))[0])
    assert heavy.tolist() == [True, True, False, True, True, True, True, True]


@pytest.fixture(scope="module")
def xzak_map():
    return read_mrc(str(XZAK_MAP))


@pytest.mark.parametrize("structure", [XZAK_STRUCTURE, SMALL_STRUCTURE], ids=lambda p: p.name)
def test_same_qscores_as_compute_qscores(structure, xzak_map):
    rows = _atom_site_arrays(gemmi.cif.read(str(structure))[0])[0]
    by_row = block_qscores(gemmi.cif.read(str(structure))[0], *xzak_map)
    expected = compute_qscores(read_cif(str(structure)), *xzak_map).atom_qscores
    np.testing.assert_array_equal(by_row[rows], expected)


def test_same_file_as_write_qscore_cif(tmp_path, xzak_map):
    path = str(XZAK_STRUCTURE)
    reference = write_qscore_cif(compute_qscores(read_cif(path), *xzak_map), path,
                                 str(XZAK_MAP), output_dir=str(tmp_path))
    doc = gemmi.cif.read(path)
    add_qscore_column(doc[0], block_qscores(doc[0], *xzak_map))
    doc.write_file(str(tmp_path / "from_block.cif"))
    assert filecmp.cmp(reference, tmp_path / "from_block.cif", shallow=False)


def test_qscore_column_length_checked():
    doc = gemmi.cif.read(str(XZAK_STRUCTURE))
    with pytest.raises(ValueError):
        add_qscore_column(doc[0], np.zeros(3))
