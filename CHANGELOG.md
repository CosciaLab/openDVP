# Changelog

All notable changes to this project will be documented in this file.

Add your entry under `## [Unreleased]` as part of the PR that makes the change — that is much
easier than reconstructing it at release time. When cutting a release, rename that heading to
the version and date. See [`.github/RELEASING.md`](.github/RELEASING.md).

---

## [0.8.0] - 2026-09-16

### Added

- Issue templates for bug reports and feature requests
- `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` (adapted from scverse)
- `.github/RELEASING.md` — maintainer guide to versioning, tags and releases
- `opendvp[napari]` extra, for the interactive viewer stack
- Eleven dependencies that openDVP imports directly but never declared: `anndata`, `pandas`,
  `matplotlib`, `seaborn`, `geopandas`, `shapely`, `libpysal`, `networkx`, `tqdm`, `dask` and
  `dask-image`. They previously arrived only transitively, so any upstream change could have
  broken installs.
- Tests covering the `opendvp.plotting` deprecation shim
- **`opendvp.datasets.tutorial_data()`** — downloads the tutorial dataset from Zenodo, verifies it
  against a known MD5, extracts it, caches it per user, and returns a dict of paths. Honours the
  `OPENDVP_DATA_DIR` environment variable.
- `scripts/check_tutorials.py`, which executes the three tutorials in order and fails on any
  erroring cell. Nothing else ran them: the docs build renders their committed outputs without
  executing them. It is now a step in `.github/RELEASING.md`.
- Tests for the ten public functions that had none: `rankplot`, `abundance_histograms`,
  `pca_loadings`, `stacked_barplot`, `scimap_phenotype`, `scimap_spatial_cluster`,
  `scimap_spatial_lda`, and the three `utils` exports. Every public function is now covered
  (342 tests, up from 227 before this release).
- `[tool.pytest.ini_options]`, which pins the test paths and turns a `FutureWarning` raised from
  openDVP's own code into a test failure, so dependency deprecations get fixed while they are
  still warnings

### Changed

- **Relicensed from GPL-3.0 to MIT**, to reduce friction for other packages that want to
  build on openDVP and to match the permissive licensing common across the scverse ecosystem
- Declared the licence as an SPDX expression (`license = "MIT"`), so package metadata now
  reports `License-Expression: MIT` instead of embedding the full licence text
- **`opendvp.plotting` is now `opendvp.pl`**, matching scverse conventions. The old name keeps
  working and emits a `DeprecationWarning`; it will be removed in 1.0.
- **openDVP now requires Python 3.12 or newer** (was 3.11). Python 3.11 was dropped so that
  `spatialdata` could be bumped to 0.8, which is 3.12+ only.
- **`napari-spatialdata`, `spatialdata-plot` and `pyqt6` are no longer core dependencies.**
  openDVP never imports them, and `pyqt6` is GPL-3.0-only, so it does not belong in a core MIT
  install. Install `opendvp[napari]` for the interactive viewer.
- Raised the `spatialdata` floor from 0.4 to 0.8, which also moves `anndata` to 0.13,
  `zarr` to 3.x, and `xarray` and `dask` to their 2026 releases
- Dropped the upper version caps on `scanpy`, `pingouin`, `esda`, `gensim`, `perseuspy`,
  `scipy` and `loguru`, all of which were blocking updates
- Moved `ipykernel` from the runtime dependencies to the `docs` dependency group
- **The API reference renders properly for the first time.** 109 of 110 docstring section
  headers were silently skipped by Sphinx, because they carried a trailing colon and because
  `conf.py` was configured for Google-style rather than numpydoc docstrings.
- `adata.uns["anova_posthoc"]` column names are now underscored (`p_tukey` rather than
  `p-tukey`), consistently across pingouin versions
- CI: adjusted triggers for the test, docs and publish workflows; the docs build now fails on
  warnings and reruns when `src/` changes
- Updated the README screenshot
- All three tutorials load their inputs through `datasets.tutorial_data()` rather than hardcoded
  `../data/...` paths, and write their outputs under `../outputs/`, which they now create
- `quant_to_adata`, `import_thresholds`, `segmask_to_qupath` and `export_adata` accept
  `pathlib.Path` as well as `str`. The first three previously raised on a `Path`.

### Fixed

- `stats_anova` raised `ValueError: Length of values does not match length of index` under
  pingouin 0.6, which renamed its result columns (`p-unc` → `p_unc`). Both 0.5 and 0.6 now work.
- `stats_anova` could append an F value without its matching p value when reading pingouin's
  output failed part-way through, corrupting the results for every later feature
- `segmask_to_qupath` raised an `ImportError` directing users to `pip install
  opendvp[spatialdata]`, an extra that has never existed
- Untracked local files such as `.DS_Store` could be picked up into the built wheel and sdist
- The CI badge in `README.md` and `docs/index.md` pointed at `workflows/testing.yml`; the file is
  `test.yml`, so the badge had never rendered a status
- `README.md` showed `conda create` under a "you can install openDVP via pip" heading, and never
  activated the environment it created
- Running the test suite opened real plot windows, because `pl` functions call `plt.show()` and
  nothing forced a non-interactive matplotlib backend
- `abundance_histograms` titled each panel using `adata.obs.raw_file_id[i]`, which indexes by
  label rather than position. With a non-default `obs` index every panel was mislabelled.
- `rankplot` used `matplotlib.cm.get_cmap`, which is deprecated and scheduled for removal
- `scimap_phenotype` used `fillna(method="ffill")`, which pandas will remove, and relied on
  `replace` downcasting an all-NaN column, which pandas has deprecated
- `scimap_rescale` wrote gate values into a DataFrame slice rather than a copy
- `stats_bootstrap` passed a numpy callable to `.agg`, which pandas is about to stop translating
  to its own implementation
- `impute_gaussian` emitted a bare numpy `RuntimeWarning` for a protein with no measured values
  at all, and then imputed nothing for it without saying so; it now logs a warning
- `scimap_spatial_lda` failed with `UnboundLocalError` several frames deep when given a `method`
  other than `knn` or `radius`; it now raises a clear `ValueError`
- Tutorial 1 called `skimage.io.imshow`, which scikit-image removes in 0.27 — and openDVP places
  no upper bound on scikit-image, so the tutorial would have broken. It now uses matplotlib,
  which is what scikit-image's own deprecation message recommends.
- Tutorial 1 downloaded the 133 MB dataset with raw `requests` and then never extracted it, so
  every later cell failed on a fresh machine
- Tutorial 3 called `sdata.pl.render_images()` without importing `spatialdata_plot`, which is what
  registers the `.pl` accessor, so every plotting cell raised `AttributeError`
- Tutorial 3 failed on any second run, because `sdata.write()` refuses to overwrite an existing
  zarr store
- Tutorial 3 read a checkpoint by its exact filename
  (`20250709_1322_5_DAP_adata.h5ad`). That name is stamped with the minute tutorial 2 was run, so
  it could never exist for anyone else; it now picks up the most recent checkpoint

### Removed
- `xarray` and `pyogrio` from the declared dependencies — neither is imported, and both arrive
  via `spatialdata` and `geopandas` anyway
- Dead `importlib_metadata` import fallback, unreachable on any supported Python

---

## [0.7.3] - 2025-10-18

Releases from `0.1.1` to `0.7.3` predate this changelog being maintained.
See the [releases page](https://github.com/CosciaLab/openDVP/releases) and the
[commit history](https://github.com/CosciaLab/openDVP/commits/main/) for what changed.

---

## [0.2.6] 2025-06-26

### Changed
- Updated docs, removed html folder

### Fixed
- `pixi install` was failing because of `pyproj` from PyPi was failing

---

## [0.2.0] - 2025-06-14

### Added
Pyproteomics functionalities
Tools, plotting, io

---

## [0.1.1] - 2025-05-18

- Initial release 🚀
