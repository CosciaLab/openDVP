from pathlib import Path

import pooch

from opendvp.utils import logger

#: Zenodo record backing the tutorials. Versioned DOI, so the contents cannot change under us.
DOI = "10.5281/zenodo.15830141"
_BASE_URL = "https://zenodo.org/records/15830141/files/"
_ARCHIVE = "data.tar.gz"
_ARCHIVE_HASH = "md5:a767c99a32ab0138e7d4ad3577e170c2"

#: Keys returned by :func:`tutorial_data`, mapped to their path inside the archive.
_MEMBERS = {
    "image": "data/image/mIF.ome.tif",
    "segmentation": "data/segmentation/segmentation_mask.tif",
    "quantification": "data/quantification/quant.csv",
    "artefact_annotations": "data/manual_artefact_annotations/artefacts.geojson",
    "gates": "data/phenotyping/gates.csv",
    "celltype_matrix": "data/phenotyping/celltype_matrix.csv",
    "proteomics": "data/proteomics/DIANN_pg_matrix.csv",
    "proteomics_metadata": "data/proteomics/DIANN_metadata.csv",
    "collection_shapes": "data/proteomics/collection_shapes.geojson",
}


def tutorial_data(path: str | Path | None = None) -> dict[str, Path]:
    """Download, verify and cache the openDVP tutorial dataset.

    Fetches a single 133 MB archive from Zenodo, checks it against a known MD5, extracts it, and
    returns the paths of the files the tutorials use. The download happens once: later calls
    verify the cache and return immediately.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Directory to cache the dataset in. Defaults to the per-user cache directory
        (``~/Library/Caches/opendvp`` on macOS, ``~/.cache/opendvp`` on Linux), which can also be
        overridden by setting the ``OPENDVP_DATA_DIR`` environment variable.

    Returns
    -------
    dict[str, pathlib.Path]
        One entry per tutorial input, plus ``"root"`` for the extracted directory itself:

        ``image``
            Multiplexed immunofluorescence image, OME-TIFF.
        ``segmentation``
            Single-cell segmentation mask, TIFF.
        ``quantification``
            Per-cell marker intensities, CSV.
        ``artefact_annotations``
            Manually drawn artefact regions, GeoJSON.
        ``gates``
            Per-marker gating thresholds, CSV.
        ``celltype_matrix``
            Marker-to-cell-type definitions for phenotyping, CSV.
        ``proteomics``
            DIA-NN protein group matrix, tab-separated.
        ``proteomics_metadata``
            Sample metadata for the DIA-NN matrix, semicolon-separated.
        ``collection_shapes``
            Laser-microdissection collection shapes, GeoJSON.

    Raises
    ------
    FileNotFoundError
        If the archive extracts without a file the tutorials expect.

    Notes
    -----
    The dataset is archived on Zenodo under DOI `10.5281/zenodo.15830141
    <https://doi.org/10.5281/zenodo.15830141>`_. That DOI points at one specific version, so the
    contents cannot change and the MD5 in this module will keep matching.

    Examples
    --------
    >>> import opendvp as dvp
    >>> paths = dvp.datasets.tutorial_data()  # doctest: +SKIP
    >>> adata = dvp.io.quant_to_adata(paths["quantification"])  # doctest: +SKIP
    """
    cache = pooch.os_cache("opendvp") if path is None else Path(path).expanduser()
    fetcher = pooch.create(
        path=cache,
        base_url=_BASE_URL,
        registry={_ARCHIVE: _ARCHIVE_HASH},
        env="OPENDVP_DATA_DIR",
    )

    logger.info(f"Fetching the openDVP tutorial dataset into {fetcher.abspath}")
    extracted = fetcher.fetch(_ARCHIVE, processor=pooch.Untar(), progressbar=True)

    paths = _index_members(extracted)
    logger.success(f"Tutorial dataset ready at {paths['root']}")
    return paths


def _index_members(extracted: list[str]) -> dict[str, Path]:
    """Map the extracted file list onto the documented keys.

    Matching on path suffix rather than joining onto a hardcoded extraction directory keeps this
    working whichever way pooch names the untar target.
    """
    by_path = [Path(p) for p in extracted]
    paths: dict[str, Path] = {}
    for key, member in _MEMBERS.items():
        matches = [p for p in by_path if p.as_posix().endswith(member)]
        if not matches:
            raise FileNotFoundError(
                f"'{member}' is missing from the tutorial archive, so the key '{key}' cannot be "
                f"returned. Delete the cache and retry; if it persists, the Zenodo record "
                f"({DOI}) no longer matches this version of openDVP."
            )
        paths[key] = matches[0]

    # every member lives two levels below the archive root, e.g. <root>/image/mIF.ome.tif
    paths["root"] = paths["image"].parent.parent
    return paths
