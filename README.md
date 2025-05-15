# OpenDVP

![Graphical_Abstract](https://github.com/user-attachments/assets/bc2ade23-1622-42cf-a5c5-bb80e7be5b1f)

## Introduction

openDVP is a framework that empowers users to perform Deep Visual propietary with open sourced software.

This repository summarizes our recommendations with 4 distinct areas of use for openDVP.
1. Image processing and Analysis
2. Matrix processing and analysis
3. Quality control with QuPath and Napari
4. Export to LMD

openDVP uses scverse's data formats, AnnData and SpatialData, in order to facilitate further changes and or use of other analytical packages such as scanpy, squidpy, or scimap.

## Citation

Please cite the corresponding bioarxiv for now:
<REF> 

## Installation

To install run:   
``` pip install opendvp ```

## Demo
### Download demo data

https://zenodo.org/records/15397560

### How to run demo data


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
