# Workflows

## Overview

openDVP is a framework to empower users to perform spatial proteomics as easily as possible.  

We suggest two main workflows:

 - flashDVP (optimized for speed)
 - DVP (optimized for complexity)

## flashDVP

Quick and efficient, flashDVP skips through many friction points of DVP and let's you focus on your biology.

You require :

- Images in which you can recognize tissue of interest
- Laser Microdissection device, or someone willing to collaborate that has one.
- LCMS setup, or someone willing to collaborate that has one ;) .
 
Workflow is:

1. Acquire images of tissue of interest
2. Create manual annotations in QuPath
3. Transform annotations into Laser Microdissection coordenates
4. Collect tissue of interest
5. Prepare samples and acquire proteomes via LCMS
6. Perform downstream proteomic data analysis


## DVP

Ready to explore the proteomics of more complex tissues? or you are planning a large-scale project that needs automation? openDVP can help you.  

Requirements and workflow vary so much between projects that it is not worth generalizing. But there are four main components:

1. Image acquisition
2. Image processing
3. Image analysis
4. Sampling strategy
5. Sample collection
6. Sample prepatation and LCMS
7. Downstream proteomic analysis

openDVP highly recommends utilizing open-source image processing pipelines: MCMICRO and SOPA are wonderful examples, each with their tradeoffs.

Image analysis can vary, but openDVP can help you filter common artefacts such as cells by morphological or intensity features. Filter dropped out cells by calculating the ratio of marker intensity between cycles. It also enables a easy back and forth between user-friendly annotation software like QuPath to easily integrate collaborators insights into the analysis. We use scimap for phenotyping, but we suggest you compare between the released approaches, and use what fits your problem best, that is the beauty of open source.

We will release more details soon :)