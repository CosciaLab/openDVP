"""Tests for `opendvp.datasets.tutorial_data`.

The real archive is 133 MB, so these build a miniature one with the same internal layout and
serve it over loopback. That keeps the download, the MD5 check, the untar and the key mapping
all under test without touching the network.
"""

import hashlib
import importlib
import tarfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from opendvp.datasets import tutorial_data
from opendvp.datasets.tutorial_data import _MEMBERS, _index_members

# `opendvp.datasets.tutorial_data` is both the module and the function it exports, and the
# package `__init__` rebinds the name to the function. Go through importlib to patch the module.
tutorial_data_module = importlib.import_module("opendvp.datasets.tutorial_data")


def _build_archive(directory: Path, members: list[str]) -> Path:
    """Write a data.tar.gz containing `members` as small text files."""
    staging = directory / "staging"
    for member in members:
        target = staging / member
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(member)

    archive = directory / "data.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(staging / "data", arcname="data")
    return archive


@pytest.fixture
def served_archive(tmp_path):
    """Serve a miniature tutorial archive over loopback, and point the module at it."""
    served = tmp_path / "served"
    served.mkdir()
    archive = _build_archive(served, list(_MEMBERS.values()))
    md5 = hashlib.md5(archive.read_bytes()).hexdigest()

    handler = partial(SimpleHTTPRequestHandler, directory=str(served))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/", f"md5:{md5}"
    server.shutdown()
    server.server_close()


@pytest.fixture
def local_dataset(monkeypatch, served_archive, tmp_path):
    """`tutorial_data` wired to the local archive, caching into a temporary directory."""
    base_url, archive_hash = served_archive
    monkeypatch.setattr(tutorial_data_module, "_BASE_URL", base_url)
    monkeypatch.setattr(tutorial_data_module, "_ARCHIVE_HASH", archive_hash)
    return tmp_path / "cache"


def test_returns_every_documented_key(local_dataset):
    paths = tutorial_data(path=local_dataset)
    assert set(paths) == set(_MEMBERS) | {"root"}


def test_returned_paths_exist(local_dataset):
    paths = tutorial_data(path=local_dataset)
    for key, value in paths.items():
        assert value.exists(), f"{key} -> {value}"


def test_keys_point_at_the_right_files(local_dataset):
    paths = tutorial_data(path=local_dataset)
    for key, member in _MEMBERS.items():
        assert paths[key].as_posix().endswith(member)


def test_root_contains_the_other_paths(local_dataset):
    paths = tutorial_data(path=local_dataset)
    assert paths["root"].is_dir()
    for key, value in paths.items():
        if key != "root":
            assert paths["root"] in value.parents


def test_caches_instead_of_downloading_twice(local_dataset):
    first = tutorial_data(path=local_dataset)
    archive = local_dataset / "data.tar.gz"
    stamp = archive.stat().st_mtime_ns

    second = tutorial_data(path=local_dataset)
    assert first == second
    assert archive.stat().st_mtime_ns == stamp, "archive was re-downloaded on the second call"


def test_honours_an_explicit_path(local_dataset):
    paths = tutorial_data(path=local_dataset)
    assert local_dataset in paths["root"].parents


def test_corrupt_download_is_rejected(monkeypatch, served_archive, tmp_path):
    base_url, _ = served_archive
    monkeypatch.setattr(tutorial_data_module, "_BASE_URL", base_url)
    monkeypatch.setattr(tutorial_data_module, "_ARCHIVE_HASH", "md5:" + "0" * 32)
    with pytest.raises(ValueError, match="does not match"):
        tutorial_data(path=tmp_path / "cache")


def test_missing_member_names_the_key():
    extracted = [f"/somewhere/{member}" for member in _MEMBERS.values()]
    extracted.remove("/somewhere/" + _MEMBERS["gates"])
    with pytest.raises(FileNotFoundError, match="gates"):
        _index_members(extracted)


def test_index_members_is_indifferent_to_the_extraction_directory():
    for prefix in ("/a/b/data.tar.gz.untar/", "/tmp/x/"):
        paths = _index_members([prefix + member for member in _MEMBERS.values()])
        assert paths["image"].as_posix().endswith(_MEMBERS["image"])
