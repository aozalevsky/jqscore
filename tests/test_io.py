"""Tests for I/O functions: read_cif, read_mrc, read_qscores, compare_qscores, write_qscore_cif."""

import os
import tempfile

import gemmi
import numpy as np
import pytest

from jqscore.jqscore import AtomRecord, QScoreResult, read_cif, read_mrc, write_qscore_cif
from jqscore.compare import read_qscores, compare_qscores

from .conftest import (
    SMALL_STRUCTURE,
    SMALL_MAP,
    SMALL_MAPQ_REF,
    ORIG_CIF,
    JAX_CIF,
    has_docs_cifs,
)


# ── read_cif (in-repo 390_7cec_A100) ────────────────────────────────────


class TestReadCif:
    def test_returns_list_of_atom_records(self):
        atoms = read_cif(str(SMALL_STRUCTURE))
        assert isinstance(atoms, list)
        assert len(atoms) > 0
        assert isinstance(atoms[0], AtomRecord)

    def test_atom_count(self):
        """390_7cec_A100 should have exactly 100 atoms."""
        atoms = read_cif(str(SMALL_STRUCTURE))
        assert len(atoms) == 100

    def test_atom_fields_populated(self):
        atoms = read_cif(str(SMALL_STRUCTURE))
        a = atoms[0]
        assert a.chain_id != ""
        assert a.res_name != ""
        assert a.atom_name != ""
        assert a.element != ""
        assert isinstance(a.x, float)
        assert isinstance(a.y, float)
        assert isinstance(a.z, float)

    def test_has_heavy_atoms(self):
        atoms = read_cif(str(SMALL_STRUCTURE))
        elements = {a.element for a in atoms}
        assert "C" in elements
        assert "N" in elements

    def test_single_chain(self):
        atoms = read_cif(str(SMALL_STRUCTURE))
        chains = {a.chain_id for a in atoms}
        assert len(chains) == 1
        assert "A" in chains


# ── read_mrc (in-repo small map) ────────────────────────────────────────


class TestReadMrc:
    def test_returns_tuple(self):
        data, origin, voxel_size = read_mrc(str(SMALL_MAP))
        assert isinstance(data, np.ndarray)
        assert data.ndim == 3
        assert data.dtype == np.float32
        assert origin.shape == (3,)
        assert voxel_size.shape == (3,)

    def test_positive_voxel_size(self):
        _, _, voxel_size = read_mrc(str(SMALL_MAP))
        assert (voxel_size > 0).all()

    def test_data_not_all_zero(self):
        data, _, _ = read_mrc(str(SMALL_MAP))
        assert data.max() > 0

    def test_box_smaller_than_cell(self, tmp_path):
        """Voxel size is cell / MX,MY,MZ, not cell / box size; origin from NCSTART…"""
        nc, nr, ns = 20, 30, 40
        arr = np.arange(nc * nr * ns, dtype=np.float32).reshape(nc, nr, ns)
        m = gemmi.Ccp4Map()
        m.grid = gemmi.FloatGrid(arr, gemmi.UnitCell(100, 100, 100, 90, 90, 90))
        m.update_ccp4_header()
        for word, value in ((8, 100), (9, 100), (10, 100), (5, 7), (6, 8), (7, 9)):
            m.set_header_i32(word, value)
        path = str(tmp_path / "box.map")
        m.write_ccp4_map(path)

        data, origin, voxel_size = read_mrc(path)
        np.testing.assert_allclose(voxel_size, [1.0, 1.0, 1.0])
        np.testing.assert_allclose(origin, [7.0, 8.0, 9.0])
        assert data.shape == (ns, nr, nc)
        assert data[3, 2, 1] == arr[1, 2, 3]


# ── read_qscores (MapQ reference output, in-repo) ───────────────────────


class TestReadQscores:
    def test_read_mapq_reference(self):
        struct, qscores = read_qscores(str(SMALL_MAPQ_REF))
        assert len(qscores) == 100
        n_valid = np.isfinite(qscores).sum()
        assert n_valid > 50

    def test_qscore_values_in_range(self):
        _, qscores = read_qscores(str(SMALL_MAPQ_REF))
        valid = qscores[np.isfinite(qscores)]
        assert valid.min() >= -1.0
        assert valid.max() <= 1.0


# ── compare_qscores (docs/ CIFs: orig vs jax 9XZK) ─────────────────────


@has_docs_cifs
class TestCompareQscores:
    def test_compare_orig_vs_jax(self):
        result = compare_qscores(str(ORIG_CIF), str(JAX_CIF))
        assert result.n_valid > 100
        # Every atom of 9XZK within 1e-3 of MapQ (max ~3e-5)
        assert abs(result.diff_max) < 1e-3
        assert abs(result.diff_min) < 1e-3
        # Mean diff should be near zero
        assert abs(result.diff_mean) < 1e-5
        # Pearson r should be very close to 1
        assert result.pearson_r > 0.999999

    def test_self_comparison_is_perfect(self):
        result = compare_qscores(str(ORIG_CIF), str(ORIG_CIF))
        assert result.diff_max == pytest.approx(0.0)
        assert result.diff_min == pytest.approx(0.0)
        assert result.pearson_r == pytest.approx(1.0)


# ── write_qscore_cif roundtrip ──────────────────────────────────────────


class TestWriteQscoreCif:
    def test_roundtrip(self):
        """Write a CIF with synthetic Q-scores and read them back."""
        atoms = read_cif(str(SMALL_STRUCTURE))
        n = len(atoms)
        fake_q = np.random.default_rng(42).uniform(0.0, 1.0, size=n)
        # Leave some as NaN (like H atoms without assignment)
        fake_q[::10] = np.nan
        result = QScoreResult(atoms, fake_q, 0.0, 1.0, 0.4)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = write_qscore_cif(result, str(SMALL_STRUCTURE), "test.map", output_dir=tmpdir)
            assert out_path.endswith(".cif")

            struct, qscores_back = read_qscores(out_path)
            assert len(qscores_back) == n

            # Check non-NaN values match within write precision (6 decimal places)
            valid = np.isfinite(fake_q)
            np.testing.assert_allclose(
                qscores_back[valid], fake_q[valid], atol=1e-5,
            )
            # NaN positions should still be NaN
            assert np.isnan(qscores_back[~valid]).all()

    def test_missing_auth_atom_id_cif(self):
        """CIFs lacking _atom_site.auth_atom_id must still round-trip.

        python-ihm's ihm.dumper (used for integrative-model CIFs in the EMDB
        VA pipeline) emits the auth_asym_id/seq_id/comp_id columns but omits
        auth_atom_id. write_qscore_cif must fall back to label_atom_id rather
        than IndexError on an empty list.
        """
        # Build a fixture from SMALL_STRUCTURE with auth_atom_id removed,
        # mirroring the column layout python-ihm emits.
        doc = gemmi.cif.read(str(SMALL_STRUCTURE))
        block = doc[0]
        for item in block:
            if item.loop and any("_atom_site" in t for t in item.loop.tags):
                item.loop.remove_column("_atom_site.auth_atom_id")
                break

        with tempfile.TemporaryDirectory() as tmpdir:
            no_auth_cif = os.path.join(tmpdir, "no_auth_atom_id.cif")
            doc.write_file(no_auth_cif)

            # Sanity: auth_atom_id stripped, label_atom_id retained.
            check = gemmi.cif.read(no_auth_cif)[0]
            assert not list(check.find_loop("_atom_site.auth_atom_id"))
            assert list(check.find_loop("_atom_site.label_atom_id"))

            atoms = read_cif(no_auth_cif)
            n = len(atoms)
            fake_q = np.random.default_rng(0).uniform(0.0, 1.0, size=n)
            result = QScoreResult(atoms, fake_q, 0.0, 1.0, 0.4)

            out_path = write_qscore_cif(result, no_auth_cif, "test.map", output_dir=tmpdir)

            _, qscores_back = read_qscores(out_path)
            assert len(qscores_back) == n
            np.testing.assert_allclose(qscores_back, fake_q, atol=1e-5)

    def test_trailing_ligand_rows(self, tmp_path):
        """Q-scores go to the right rows when gemmi's atom order differs from the file's.

        gemmi.read_structure() puts a ligand listed after all chains into its
        parent chain; the CIF keeps it last.
        """
        tags = ("group_PDB id type_symbol label_atom_id label_alt_id label_comp_id "
                "label_asym_id label_entity_id label_seq_id pdbx_PDB_ins_code Cartn_x "
                "Cartn_y Cartn_z occupancy B_iso_or_equiv auth_seq_id auth_asym_id "
                "pdbx_PDB_model_num").split()
        rows = [
            "ATOM 1 N N . GLY A 1 1 ? 0.000 0.000 0.000 1.00 10.0 1 A 1",
            "ATOM 2 C CA . GLY A 1 1 ? 1.450 0.000 0.000 1.00 10.0 1 A 1",
            "ATOM 3 N N . GLY B 1 1 ? 10.000 0.000 0.000 1.00 10.0 1 B 1",
            "ATOM 4 C CA . GLY B 1 1 ? 11.450 0.000 0.000 1.00 10.0 1 B 1",
            "HETATM 5 ZN ZN . ZN C 2 . ? 3.000 1.000 0.000 1.00 10.0 101 A 1",
        ]
        file_x = [float(r.split()[10]) for r in rows]
        text = "data_test\nloop_\n" + "".join(f"_atom_site.{t}\n" for t in tags)
        text += "\n".join(rows) + "\n"
        path = tmp_path / "ligand_last.cif"
        path.write_text(text)

        atoms = read_cif(str(path))
        assert [a.x for a in atoms] != file_x, "gemmi kept the file order"
        # Q-score = the atom's x coordinate, so each row can be checked
        result = QScoreResult(atoms, np.array([a.x for a in atoms]), 0.0, 1.0, 0.4)
        out = write_qscore_cif(result, str(path), "test.map", output_dir=str(tmp_path))

        block = gemmi.cif.read(out)[0]
        written = [float(v) for v in block.find_values("_atom_site.Q-score")]
        np.testing.assert_allclose(written, file_x)
