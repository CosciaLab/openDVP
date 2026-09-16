# Datasets

```{eval-rst}
.. module:: opendvp.datasets
.. currentmodule:: opendvp

.. autosummary::
    :toctree: generated

    datasets.tutorial_data
```

## Where the data lives

`tutorial_data()` downloads a single archive from Zenodo
([10.5281/zenodo.15830141](https://doi.org/10.5281/zenodo.15830141)), checks it against a known
MD5, extracts it, and returns a dict of paths. The download happens once and is then cached
per-user, so the tutorials are re-runnable offline.

The cache location defaults to your platform's user cache directory — `~/Library/Caches/opendvp`
on macOS, `~/.cache/opendvp` on Linux. Override it either per-call or for the whole environment:

```python
import opendvp as dvp

paths = dvp.datasets.tutorial_data(path="/scratch/opendvp-data")
```

```bash
export OPENDVP_DATA_DIR=/scratch/opendvp-data
```

The archive is 133 MB compressed and roughly 1 GB extracted, almost all of it the
multiplexed image.
