from opendvp.io import export_adata
import numpy as np
import pandas as pd
import anndata as ad

def test_export_adata_callable(tmp_path) -> None:
    # Create a minimal AnnData object
    X = np.array([[1, 2], [3, 4]])
    obs = pd.DataFrame({
        "sample_id": ["S1", "S2"],
        "condition": ["A", "B"]
    }, index=["cell1", "cell2"])
    var = pd.DataFrame(index=["gene1", "gene2"])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    export_adata(
        adata=adata,
        path_to_dir=tmp_path,
        checkpoint_name="test")
    
    # Get list of output files
    output_files = list(tmp_path.iterdir())
    assert len(output_files) == 2
