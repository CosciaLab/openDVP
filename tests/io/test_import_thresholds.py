import numpy as np
import pandas as pd
import pytest

from opendvp.io import import_thresholds


@pytest.fixture
def gates_csv(tmp_path):
    # Create a test CSV with mock gate data
    data = pd.DataFrame({
        'marker_id': ['CD3', 'CD4', 'CD8', 'CD3', 'CD4'],
        'gate_value': [1.0, 0.0, 2.5, 3.0, 0.0],
        'sample_id': ['sample1', 'sample1', 'sample2', 'sample1', 'sample2']
    })
    file_path = tmp_path / "gates.csv"
    data.to_csv(file_path, index=False)
    return str(file_path)

def test_invalid_extension():
    with pytest.raises(ValueError, match="The file should be a csv file"):
        import_thresholds("invalid_file.txt")

def test_missing_gate_value_column(tmp_path):
    df = pd.DataFrame({'marker_id': ['CD3'], 'sample_id': ['sample1']})
    file_path = tmp_path / "bad.csv"
    df.to_csv(file_path, index=False)
    with pytest.raises(ValueError, match="gate_value is not present"):
        import_thresholds(str(file_path))

def test_missing_sample_id_column(tmp_path):
    df = pd.DataFrame({'marker_id': ['CD3'], 'gate_value': [1.0]})
    file_path = tmp_path / "bad.csv"
    df.to_csv(file_path, index=False)
    with pytest.raises(ValueError, match="sample_id is not present"):
        import_thresholds(str(file_path), sample_id='sample1')

def test_filter_zero_gates(gates_csv):
    df = import_thresholds(gates_csv, log1p=False)
    assert (df['gate_value'] == 0.0).sum() == 0
    assert 'marker_id' in df.columns

def test_log1p_transformation(gates_csv):
    df = import_thresholds(gates_csv)
    assert 'markers' in df.columns
    assert df.shape[0] == 3  # Only 3 rows with gate_value != 0.0
    assert df.iloc[0, 1] == pytest.approx(np.log1p(1.0), rel=1e-3)

def test_filter_by_sample(gates_csv):
    df = import_thresholds(gates_csv, sample_id='sample1')
    assert 'sample1' in df.columns
    assert df.shape[0] == 2  # Two valid gates for sample1
