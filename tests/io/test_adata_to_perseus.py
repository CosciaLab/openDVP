import os
import tempfile
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from opendvp.io import adata_to_perseus


def test_adata_to_perseus() -> None:
    # Create a small AnnData object
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    obs = pd.DataFrame({'cell_id': ['cell1', 'cell2'], 'group': ['A', 'B']})
    var = pd.DataFrame(index=['gene1', 'gene2'])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    fixed_timestamp = "20250101_1200"
    
    # Use a temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch datetime to produce a fixed timestamp
        with patch("your_module.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = fixed_timestamp

            # Run the function
            adata_to_perseus(adata, path_to_dir=tmpdir, suffix="test", obs_key="cell_id")

        # Expected file paths
        data_file = os.path.join(tmpdir, f"{fixed_timestamp}_data_test.txt")
        metadata_file = os.path.join(tmpdir, f"{fixed_timestamp}_metadata_test.txt")

        # Check files exist
        assert os.path.exists(data_file)
        assert os.path.exists(metadata_file)

        # Validate contents of data file
        df_data = pd.read_csv(data_file, sep="\t", index_col=0)
        np.testing.assert_array_equal(df_data.index, ['cell1', 'cell2'])
        np.testing.assert_array_equal(df_data.columns, ['gene1', 'gene2'])
        np.testing.assert_array_almost_equal(df_data.values, X)

        # Validate contents of metadata file
        df_meta = pd.read_csv(metadata_file, sep="\t", index_col=0)
        np.testing.assert_array_equal(df_meta.index, ['cell1', 'cell2'])
        assert 'group' in df_meta.columns
        assert list(df_meta['group']) == ['A', 'B']
