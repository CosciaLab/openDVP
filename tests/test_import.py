import pytest

def test_can_import_package():
    import opendvp
    version = opendvp.__version__
    assert version is not None