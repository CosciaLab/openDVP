from opendvp.io import import_perseus

def test_import_perseus() -> None:

    adata = import_perseus(
        path_to_perseus_txt="../test_data/io/Perseus_v1.6.15.0.txt",
        n_var_metadata_rows=5
    )
    
    test_shape = (11, 3535)
    assert adata.shape == test_shape
    test_list = ['Column Name','Heart_Condition','MI_Region','replicate_number','Group_Type']
    assert adata.obs.columns.tolist() == test_list

