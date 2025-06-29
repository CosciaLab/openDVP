from pathlib import Path

import pytest

from opendvp.io import DIANN_to_adata

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data" / "io"

DIANN_181 = TEST_DATA_DIR / "DIANN_v1.8.1_pg_matrix.tsv"
DIANN_19 = TEST_DATA_DIR / "DIANN_v1.9_pg_matrix.tsv"
DIANN_2 = TEST_DATA_DIR / "DIANN_v2_pg_matrix.tsv"


def test_DIANN_to_adata_181():

    adata = DIANN_to_adata(
        DIANN_path=str(DIANN_181),
        n_of_protein_metadata_cols=5
    )
    assert adata.shape == (18,5159)
    assert list(adata.var.columns) == [
        'Protein.Group', 'Protein.Ids', 
        'Protein.Names', 'Genes',
        'First.Protein.Description']

def test_DIANN_to_adata_19():

    adata = DIANN_to_adata(
        DIANN_path=str(DIANN_19),
        n_of_protein_metadata_cols=4
    )
    assert adata.shape == (18,5233)
    assert list(adata.var.columns) == [
        'Protein.Group', 'Protein.Names', 
        'Genes', 'First.Protein.Description']


def test_DIANN_to_adata_2():

    adata = DIANN_to_adata(
        DIANN_path=str(DIANN_2),
        n_of_protein_metadata_cols=4
    )
    assert adata.shape == (15,7815)
    assert list(adata.var.columns) == [
        'Protein.Group', 'Protein.Names', 
        'Genes', 'First.Protein.Description']