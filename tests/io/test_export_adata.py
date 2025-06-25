import pytest
import anndata as ad
import numpy as np
import pandas as pd

from opendvp.io import export_adata


@pytest.fixture
def minimal_adata():
    X = np.array([[1, 2], [3, 4]])
    obs = pd.DataFrame({
        "sample_id": ["S1", "S2"],
        "condition": ["A", "B"]
    }, index=["cell1", "cell2"])
    var = pd.DataFrame(index=["gene1", "gene2"])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_export_adata_just_h5ad(tmp_path, minimal_adata) -> None:
    export_adata(
        adata=minimal_adata,
        path_to_dir=tmp_path,
        checkpoint_name="test")
    # Get list of output files
    output_files = list(tmp_path.iterdir())
    assert len(output_files) == 1

def test_export_adata_h5ad_csvs(tmp_path, minimal_adata) -> None:
    export_adata(
        adata=minimal_adata,
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

def test_export_adata_perseus(tmp_path, minimal_adata) -> None:
    export_adata(
        adata=minimal_adata,
        path_to_dir=tmp_path,
        checkpoint_name="test",
        perseus=True)
    # Check for perseus subfolder and files
    test_folder = tmp_path / "test"
    perseus_folder = test_folder / "perseus"
    assert perseus_folder.is_dir()
    perseus_files = list(perseus_folder.iterdir())
    # Should be two files: one data, one metadata
    assert len(perseus_files) == 2
    data_file = [f for f in perseus_files if "data" in f.name][0]
    metadata_file = [f for f in perseus_files if "metadata" in f.name][0]
    # Check file contents (basic structure)
    with open(data_file) as f:
        header = f.readline()
        assert header.startswith("Name")
    with open(metadata_file) as f:
        header = f.readline()
        assert header.startswith("Name")

def test_export_adata_h5ad_csvs_and_perseus(tmp_path, minimal_adata) -> None:
    export_adata(
        adata=minimal_adata,
        path_to_dir=tmp_path,
        checkpoint_name="test",
        export_as_cvs=True,
        perseus=True)
    # Check for main folder and files
    test_folder = tmp_path / "test"
    assert test_folder.is_dir()

    created_files = list(test_folder.iterdir())
    # Should be h5ad, data.txt, metadata.txt, perseus/

    assert any(f.name.endswith(".h5ad") for f in created_files)
    txt_files = [f for f in created_files if f.is_file() and f.name in ("data.txt", "metadata.txt")]
    other_txt_files = [f for f in created_files if f.is_file() and f.name.endswith(".txt")]
    assert txt_files or other_txt_files
    perseus_folder = test_folder / "perseus"
    assert perseus_folder.is_dir()
    perseus_files = list(perseus_folder.iterdir())
    assert len(perseus_files) == 2
    data_file = [f for f in perseus_files if "data" in f.name][0]
    metadata_file = [f for f in perseus_files if "metadata" in f.name][0]
    with open(data_file) as f:
        header = f.readline()
        assert header.startswith("Name")
    with open(metadata_file) as f:
        header = f.readline()
        assert header.startswith("Name")