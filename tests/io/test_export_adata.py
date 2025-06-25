import anndata as ad
import numpy as np
import pandas as pd

from opendvp.io import export_adata


def test_export_adata_just_h5ad(tmp_path) -> None:
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
    assert len(output_files) == 1

def test_export_adata_h5ad_csvs(tmp_path) -> None:
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
        checkpoint_name="test",
        export_as_cvs=True)
    
    # Get list of output files
    created_folders = list(tmp_path.iterdir())
    assert len(created_folders) == 1
    test_folder = tmp_path / "test"
    assert test_folder.is_dir()
    created_files = list(test_folder.iterdir())
    assert len(created_files) == 3  # h5ad, data.txt, metadata.txt