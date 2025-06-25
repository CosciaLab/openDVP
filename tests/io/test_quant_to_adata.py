from pathlib import Path

import pandas as pd
import pytest
from anndata import AnnData

from opendvp.io import quant_to_adata

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data" / "io"

QUANT = TEST_DATA_DIR / "quant.csv"

@pytest.fixture
def sample_csv(tmp_path):
    data = {
        'CellID': [0, 1, 2],
        'Y_centroid': [10.0, 20.0, 30.0],
        'X_centroid': [15.0, 25.0, 35.0],
        'Area': [100, 150, 120],
        'MajorAxisLength': [12, 14, 13],
        'MinorAxisLength': [8, 9, 10],
        'Eccentricity': [0.5, 0.6, 0.7],
        'Orientation': [0, 45, 90],
        'Extent': [0.8, 0.85, 0.9],
        'Solidity': [0.95, 0.96, 0.97],
        'mean_CD3': [1.0, 2.0, 3.0],
        'mean_CD8': [0.5, 0.8, 1.1]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "quant.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


def test_valid_input(sample_csv):
    adata = quant_to_adata(sample_csv)
    assert isinstance(adata, AnnData)
    assert adata.n_obs == 3
    assert adata.n_vars == 2
    assert 'mean_CD3' in adata.var_names


def test_index_into_1_based(sample_csv):
    adata = quant_to_adata(sample_csv, index_into_1_based='CellID')
    # Check that CellID values were incremented by 1
    assert (adata.obs['CellID'] == [1, 2, 3]).all()


def test_skip_indexing(sample_csv):
    adata = quant_to_adata(sample_csv, index_into_1_based=None)
    # Check that CellID values remain unchanged
    assert (adata.obs['CellID'] == [0, 1, 2]).all()


def test_missing_metadata_columns(tmp_path):
    df = pd.DataFrame({'CellID': [1], 'mean_CD3': [1.0]})
    path = tmp_path / "missing_meta.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Not all metadata columns are present in the csv file"):
        quant_to_adata(str(path))


def test_invalid_extension():
    with pytest.raises(ValueError, match="should be a csv file"):
        quant_to_adata("data.txt")


def test_custom_meta_columns(sample_csv):
    # Use a reduced set of metadata columns
    adata = quant_to_adata(sample_csv, meta_columns=['CellID', 'Y_centroid', 'X_centroid'])
    assert set(['CellID', 'Y_centroid', 'X_centroid']).issubset(adata.obs.columns)
    assert 'mean_CD3' in adata.var.index


def test_exemplar001_mcmicro():
    adata = quant_to_adata(path=str(QUANT))
    assert isinstance(adata, AnnData)
    assert adata.shape == (9711,12)
    assert adata.var.shape == (12,0)
    assert adata.obs.shape == (9711,10)