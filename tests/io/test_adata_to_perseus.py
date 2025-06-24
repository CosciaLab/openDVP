import pandas as pd
import numpy as np
import anndata as ad
import pytest

from opendvp.io import adata_to_perseus

def test_adata_to_perseus_creates_files(tmp_path):
    # Create a minimal AnnData object
    X = np.array([[1, 2], [3, 4]])
    obs = pd.DataFrame({
        "sample_id": ["S1", "S2"],
        "condition": ["A", "B"]
    }, index=["cell1", "cell2"])
    var = pd.DataFrame(index=["gene1", "gene2"])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    # Run the function
    adata_to_perseus(adata, path_to_dir=str(tmp_path), suffix="test", obs_key="sample_id")

    # Get list of output files
    output_files = list(tmp_path.iterdir())
    assert len(output_files) == 2

    # Check that the files are named correctly and contain expected content
    data_file = [f for f in output_files if "data_test" in f.name][0]
    metadata_file = [f for f in output_files if "metadata_test" in f.name][0]

    # Check file contents
    data_df = pd.read_csv(data_file, sep="\t", index_col=0)
    metadata_df = pd.read_csv(metadata_file, sep="\t", index_col=0)

    assert list(data_df.columns) == ["gene1", "gene2"]
    assert data_df.shape == (2, 2)
    assert "condition" in metadata_df.columns
    assert list(metadata_df.index) == ["S1", "S2"]

