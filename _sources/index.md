# openDVP - community empowered Deep Visual Proteomics

[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://coscialab.github.io/openDVP/)
[![CI](https://github.com/CosciaLab/openDVP/actions/workflows/testing.yml/badge.svg)](https://github.com/CosciaLab/openDVP/actions/workflows/testing.yml)
[![Python versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey.svg)](https://github.com/CosciaLab/openDVP/actions/workflows/testing.yml)
[![PyPI version](https://img.shields.io/pypi/v/openDVP.svg)](https://pypi.org/project/openDVP/)
[![License](https://img.shields.io/github/license/CosciaLab/openDVP.svg)](https://github.com/CosciaLab/opendvp/blob/main/LICENSE)

`opendvp` is a python package offering a comprehensive set of tools for deep visual proteomics. It supports quality control and image analysis of multiplex immunofluorescence data. openDVP facilitates the integration of imaging datasets with proteomic datasets with [Spatialdata](https://github.com/scverse/spatialdata). Lastly, it contains a powerful toolkit for label-free downstream proteomic analysis.

It is a package that leverages the [scverse](https://www.scverse.org/) ecosystem, designed for easy interoperability with `anndata`, `scanpy`, `decoupler`, `scimap`, and other related packages.

## Getting started

openDVP is a framework that has both experimental and software aspects.

Please check our [API documentation](api/index.md) for detailed functionalities.

## Installation

You need at least Python 3.10 installed.

### First time trying python?

<details>
<summary> Click here for extra instructions</summary>

1. IF you need software to run jupyter notebooks, I suggest you install [Visual Studio Code](https://code.visualstudio.com/download).
2. Install `uv`, a python environment manager, following instructions at [installing uv](https://docs.astral.sh/uv/getting-started/installation/). 
3. Create a local folder you would like to use for your project, and open that folder it in `VSCode`
4. Open the terminal and run:

```python
uv init
```

your project folder will be created, then run:

```python
uv add opendvp
```

</details>

### There are three alternatives to install openDVP:

1. Install the latest stable release from [PyPI](https://pypi.org/project/openDVP/) with minimal dependencies:

```bash
pip install openDVP
```

2. Install the latest development version from github:

```bash
pip install git+https://github.com/CosciaLab/openDVP.git@main
```

## Tutorials

- [Tutorial 1: Image analysis](Tutorials/T1_ImageAnalysis)
- [Tutorial 2: Downstream proteomics analysis](Tutorials/T2_DownstreamProteomics)
- [Tutorial 3: Integration of imaging with proteomics](Tutorials/T3_ProteomicsIntegration)


## Contact

For questions about openDVP and the DVP workflow you are very welcome to post a message in the [discussion board](https://github.com/CosciaLab/openDVP/discussions). For issues with the software, please post issues on [Github Issues](https://github.com/CosciaLab/openDVP/issues).

## Citation

Not yet available.

```{toctree}
:maxdepth: 2
:hidden:

api/index
Tutorials/index
Workflows/index
```
