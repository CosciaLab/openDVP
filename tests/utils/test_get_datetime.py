import re
import time

from opendvp.utils import get_datetime


def test_format_is_yyyymmdd_hhmm():
    assert re.fullmatch(r"\d{8}_\d{4}", get_datetime())


def test_matches_the_current_local_time():
    # export_adata stamps filenames with this, so it has to be local wall-clock time
    assert get_datetime() == time.strftime("%Y%m%d_%H%M")


def test_sorts_chronologically():
    """Tutorial 3 relies on lexicographic order matching chronological order."""
    stamps = ["20250709_1322", "20250709_0900", "20261231_2359", "20250101_0000"]
    assert sorted(stamps) == ["20250101_0000", "20250709_0900", "20250709_1322", "20261231_2359"]
