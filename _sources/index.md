# openDVP - community empowered Deep Visual Proteomics


`opendvp` is a python package containing different tools enabling users to perform deep visual proteomics. To perform quality control and image analysis of multiplex immunofluorescence. Also to integrate imaging datasets with proteomic datasets with [Spatialdata](https://github.com/scverse/spatialdata). Lastly, it contains a powerful toolkit for label-free downstream proteomic analysis.

It is a package that leverages the [scverse]() ecosystem, designed for easy interoperability with `anndata`, `scanpy`, `decoupler`, `scimap`, and other related packages.

## Getting started

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
**OR** to install spatialdata capabilities:
```python
uv add 'opendvp[spatialdata]'
```

</details>


### There are three alternatives to install openDVP:

1. Install the latest stable release from [PyPI](https://pypi.org/project/openDVP/) with minimal dependencies:
```bash
pip install openDVP
```
2. Install the latest stable release from [PyPI](https://pypi.org/project/openDVP/) with spatialdata capabilities:
```bash
pip install 'openDVP[spatialdata]'
```
3. Install the latest development version from github:
```
pip install git+https://github.com/CosciaLab/openDVP.git@main
```


## Tutorials

### Tutorial 1: Image analysis

### Tutorial 2: Integration of imaging with proteomics

### Tutorial 3: Downstream proteomics analysis


## Contact

For questions about openDVP and the DVP workflow you are very welcome to post a message in the [discussion board](https://github.com/CosciaLab/openDVP/discussions). For issues with the software, please post issues on [Github Issues](https://github.com/CosciaLab/openDVP/issues).

## Citation

Not yet available.


```{toctree}
:maxdepth: 2
:hidden:

api/index
Tutorials/index
references
```