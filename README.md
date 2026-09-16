# OpenDVP

[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://coscialab.github.io/openDVP/)
[![CI](https://github.com/CosciaLab/openDVP/actions/workflows/testing.yml/badge.svg)](https://github.com/CosciaLab/openDVP/actions/workflows/testing.yml)
[![Python versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey.svg)](https://github.com/CosciaLab/openDVP/actions/workflows/testing.yml)
[![PyPI version](https://img.shields.io/pypi/v/openDVP.svg)](https://pypi.org/project/openDVP/)
[![License](https://img.shields.io/github/license/CosciaLab/openDVP.svg)](https://github.com/CosciaLab/opendvp/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/CosciaLab/openDVP/graph/badge.svg?token=IWGKMCHAA1)](https://codecov.io/gh/CosciaLab/openDVP)

<img width="1239" height="696" alt="Screenshot 2026-01-16 at 09 09 26" src="https://github.com/user-attachments/assets/c06fa45e-71fb-49e1-8afc-fddd755e0943" />

## Overview

**OpenDVP** is an open-source framework designed to support Deep Visual Proteomics (DVP) across multiple modalities using community-supported tools. OpenDVP empowers researchers to perform Deep Visual Proteomics using open-source software. It integrates with community data standards such as [AnnData](https://anndata.readthedocs.io/en/latest/) and [SpatialData](https://spatialdata.scverse.org/) to ensure interoperability with popular analysis tools like [Scanpy](https://github.com/scverse/scanpy), [Squidpy](https://github.com/scverse/squidpy), and [Scimap](https://github.com/labsyspharm/scimap).

## Getting started

Please refer to the [**documentation**](https://coscialab.github.io/openDVP/), particularly the [API documentation](https://coscialab.github.io/openDVP/api/index.html).

## Installation

You will need Python 3.11 or 3.12 installed on your system.
If you are new to creating Python environments, we suggest you use [uv](https://docs.astral.sh/uv/) or [pixi](https://pixi.sh/latest/).

You can install openDVP via pip:

```bash
conda create --name opendvp -y python=3.12
```

```bash
pip install opendvp
```

The interactive napari viewer is an optional extra, because it pulls in `PyQt6`, which is
GPL-3.0-only:

```bash
pip install "opendvp[napari]"
```

To install the latest version:

```bash
pip install git+https://github.com/CosciaLab/openDVP.git@main
```

## Tutorials

To understand what are the applications of openDVP, please check our
[**Tutorials**](https://coscialab.github.io/openDVP/Tutorials/index.html).  
Briefly, they introduce users to **(1) Image analysis**, **(2) downstream proteomic analysis**, and **(3) Integration of imaging with proteomic data**.  Please download our [**Demo Dataset**](https://zenodo.org/records/15830141) to best follow the tutorials :)  

## Community & Discussions

We are excited to hear from you and together we can improve spatial protemics.
We welcome questions, feedback, and community contributions!  
Join the conversation in the [GitHub Discussions](https://github.com/CosciaLab/opendvp/discussions) tab.

## Citation

Please cite the [BioArxiv](https://www.biorxiv.org/content/10.1101/2025.07.13.662099v1):

>Nimo, J., Fritzsche, S., Valdes, D. S., Trinh, M., Pentimalli, T., Schallenberg, S., Klauschen, F., Herse, F., Florian, S., Rajewsky, N., & Coscia, F. (2025). OpenDVP: An experimental and computational framework for community-empowered deep visual proteomics [Preprint]. bioRxiv. https://doi.org/10.1101/2025.07.13.662099

## Motivation

[Deep Visual Proteomics (DVP)](https://www.nature.com/articles/s41587-022-01302-5) combines high-dimensional imaging, spatial analysis, and machine learning to extract complex biological insights from tissue samples. However, many current DVP tools are locked into proprietary formats, restricted software ecosystems, or closed-source pipelines that limit reproducibility, accessibility, and community collaboration.

- Work transparently across modalities and analysis environments
- Contribute improvements back to a growing ecosystem
- Avoid vendor lock-in for critical workflows

## Qupath-to-LMD

Qupath to lmd is a tool we use to make it as easy as possible to go from QuPath annotations to LMD contours
Check our [Qupath-to-LMD Webapp](https://qupath-to-lmd-mdcberlin.streamlit.app/), or watch our Youtube tutorial:

[![Tutorial](https://img.youtube.com/vi/jimBIqGUaXg/0.jpg)](https://www.youtube.com/watch?v=jimBIqGUaXg&t=2s)
