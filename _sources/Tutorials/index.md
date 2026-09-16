# Tutorials

This section provides hands-on tutorials to guide you through the core workflows of `openDVP`. Each tutorial uses our demo dataset and covers a key aspect of a Deep Visual Proteomics analysis.

## The demo dataset

Every tutorial starts by calling {func}`opendvp.datasets.tutorial_data`, which downloads the demo
dataset from Zenodo, verifies it, extracts it and caches it per user. The first call fetches
133 MB; after that it is instant and works offline. Run the tutorials in order — tutorial 3 picks
up a checkpoint that tutorial 2 writes.

## Tutorial 1: Image Analysis

Learn the fundamentals of processing and analyzing high-plex immunofluorescence images. This tutorial covers essential steps like quality control, cell segmentation, and feature extraction to prepare your imaging data for integration.

## Tutorial 2: Downstream Proteomics Analysis

Dive into the analysis of your integrated proteomics data. This tutorial walks you through common downstream tasks such as data normalization, identifying cell populations, and performing differential abundance analysis to uncover biological insights.

## Tutorial 3: Integration of Imaging with Proteomics

Discover how to link your imaging-derived cellular data with quantitative proteomics measurements. This tutorial leverages `SpatialData` to create a unified data object, enabling powerful spatial-omics analysis.

## Tutorial links

- [Tutorial 1: Image analysis](T1_ImageAnalysis)
- [Tutorial 2: Downstream proteomics analysis](T2_DownstreamProteomics)
- [Tutorial 3: Integration of imaging with proteomics](T3_ProteomicsIntegration)

```{toctree}
:maxdepth: 2
:hidden:

T1_ImageAnalysis
T2_DownstreamProteomics
T3_ProteomicsIntegration
```
