import re
from opendvp.utils import get_datetime, switch_adat_var_index
import pandas as pd
import numpy as np
import pytest

def test_get_datetime_format():
    dt = get_datetime()
    # Should match YYYYMMDD_HHMM
    assert re.match(r"^\d{8}_\d{4}$", dt)

def test_switch_adat_var_index():
    df = pd.DataFrame({
        'gene': ['A', 'B', 'C'],
        'protein': ['X', 'Y', 'Z'],
        'value': [1, 2, 3]
    }).set_index('gene')
    class DummyAdata:
        def __init__(self, var):
            self.var = var
        def copy(self):
            return DummyAdata(self.var.copy())
    adata = DummyAdata(df)
    adata2 = switch_adat_var_index(adata, 'protein')
    assert adata2.var.index.name == 'protein'
    assert list(adata2.var.index) == ['X', 'Y', 'Z']
