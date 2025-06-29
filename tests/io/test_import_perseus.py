from pathlib import Path

from opendvp.io import import_perseus

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data" / "io"

PERSEUS = TEST_DATA_DIR / "Perseus_v1.6.15.0.txt"

def test_import_perseus() -> None:

    adata = import_perseus(
        path_to_perseus_txt=str(PERSEUS),
        n_var_metadata_rows=5
    )
    
    test_shape = (11, 3526)
    assert adata.shape == test_shape
    test_list = ['Column Name','Heart_Condition','Ischemia_region','Sample_type','Replicate']
    assert adata.obs.columns.tolist() == test_list
