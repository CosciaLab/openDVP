# OpenDVP

![Graphical_Abstract](https://github.com/user-attachments/assets/bc2ade23-1622-42cf-a5c5-bb80e7be5b1f)


## Introduction

openDVP is a framework that aims to empower users to perform Deep Visual Proteomics without propietary software.

This repository encompasses various many steps and our recommended approach to analysis, but also with resources for users to explore and decide what is best for their question at hand.

This repository also includes a variety of python functions that aim to facilitate the conversion of images, segmentation masks, and other dataformats between python data structures such as pandas DataFrames, scverse's AnnData and SpatialData, and other software more focused with user friendliness like QuPath.

## Contents

### Expectations

- Users must be aquainted with the DVP workflow and methodology

### Inputs

#### Essential

- Images independent of modality (HE, IHC, mIF) in BioFormats compatible format.
- Shapes in a geojson file with the QuPath format
- LCMS proteomic data, so far DIANN outputs are the only acceptable format.

#### Optional

- Segmentation mask of images
- Quantification matrix of cells from images (cells x features)
- Metadata of LCMS proteomic data

### Jupyternotebooks

Here we have the jupyter notebooks to guide users through the steps to create a spatialdata object.

- Parsing geojson, especially important to merge many experiments together
- Creating a spatialdata object
- Reading and vizualizing the spatialdata object for quality control
- Exporting analysis of cells to QuPath compatible visualization (recommended for large images)
- Filtering of imaging artefacts and outlier cells
- Filtering and labelling of tissue areas by manual annotations
- Phenotyping cells
- Performing spatial analysis and cellular neighborhoods

# TODO

[] standardize functions to have docstrings
[] multi OS pixi project
[] integrate proteomic analysis functions
[] brainstorm how to run conflicting environments (scimap, cellcharter (they dont like spatialdata))
[] establish codecov (kinda annoying since pixi is so new)
[] create some tests, use pytest I suppose
