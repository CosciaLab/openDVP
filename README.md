# OpenDVP

[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://coscialab.github.io/openDVP/)

![Screenshot 2025-06-17 at 11 43 44](https://github.com/user-attachments/assets/c8ac779d-a7bb-401a-b12d-93599f75528a)

**OpenDVP** is an open-source framework designed to support Deep Visual Profiling (DVP) across multiple modalities using community-supported tools.

---

## Overview

OpenDVP empowers researchers to perform Deep Visual Proteomics using open-source software. It integrates with community data standards such as [AnnData](https://anndata.readthedocs.io/en/latest/) and [SpatialData](https://spatialdata.scverse.org/) to ensure interoperability with popular analysis tools like [Scanpy](https://github.com/scverse/scanpy), [Squidpy](https://github.com/scverse/squidpy), and [Scimap](https://github.com/labsyspharm/scimap).

This repository outlines four major use cases for OpenDVP:

1. **Image Processing and Analysis**
2. **Matrix Processing and Analysis**
3. **Quality Control with QuPath and Napari**
4. **Exporting to LMD (Laser Microdissection)**

## Installation

You can install openDVP via pip:
```bash
pip install opendvp
```
If you want to install with spatialdata capacity please run:
```bash
pip install 'opendvp[spatialdata]'
```

## Motivation

Deep Visual Profiling (DVP) combines high-dimensional imaging, spatial analysis, and machine learning to extract complex biological insights from tissue samples. However, many current DVP tools are locked into proprietary formats, restricted software ecosystems, or closed-source pipelines that limit reproducibility, accessibility, and community collaboration.

- Work transparently across modalities and analysis environments
- Contribute improvements back to a growing ecosystem
- Avoid vendor lock-in for critical workflows

## Community & Discussions

We are excited to hear from you and together we can improve spatial protemics.
We welcome questions, feedback, and community contributions!  
Join the conversation in the [GitHub Discussions](https://github.com/CosciaLab/opendvp/discussions) tab.


## Citation

Please cite the corresponding bioarxiv for now, Coming Soon!

## Demo data
A comprehensive tutorial of openDVP features is on the works, to test it, feel free to download our demo data.

https://zenodo.org/records/15397560
