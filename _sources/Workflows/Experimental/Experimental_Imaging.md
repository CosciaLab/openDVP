# Imaging

You can image with whatever technology suits your project best. In general most people use:

## Imaging modalities

- H&E: Quick, simple, cheap, and pathology friendly
- IHC: Targeted with antibody, does not require fluoresence
- IF: Targeted with antibodies, a single round can have up to 5 stains.
- mIF: IF with manual handling between cycles

### H&E

- Fast, cheap, and standard for pathologists world-wide
- Many deep learning models exist to predict features
- Fast to stain tissue, fast to image tissue

Hematoxylin and Eosin (H&E) staining remains the gold standard for histological visualization and is widely used to guide laser microdissection (LMD) of specific regions within tissue sections. The staining provides high-contrast morphological information that enables the identification of cellular and extracellular structures with precision, making it particularly valuable for isolating defined histological compartments or specific cell populations.

There is plenty of information about how to interpet H&E, pathologists have plenty of experience in annotating this kind of imaging. We are also seeing an increase in the number of available Pathology Foundational Models that can predict a variety of features from a simple H&E stain.

Depending on the experimental goal, H&E staining enables:

- Single-cell microdissection: Identification of individual cells based on nuclear morphology, cytoplasmic boundaries, or specific tissue localization.
- Microdissection of cellular clusters: Selection of small groups of phenotypically similar cells or histological niches, such as tumor margins or immune infiltrates.
- Acellular region collection: Targeting non-cellular compartments like extracellular matrix, necrotic zones, or stromal regions, which are visually distinct due to the eosinophilic staining pattern.

### Immunohistochemistry

- Antibody-based + enzymatic dying
- Brightfield imaging, therefore fast and low-storage cost.

Immunohistochemistry (IHC) enhances laser microdissection (LMD) by providing molecular specificity in addition to morphological context. Through the use of antibodies targeting specific proteins, IHC allows precise visualization and isolation of cells or tissue regions defined by molecular phenotype—such as immune subsets, tumor subpopulations, or stromal compartments. This is particularly advantageous when morphological cues alone are insufficient to distinguish the cells of interest. Chromogenic detection methods, typically using DAB (brown) or AEC (red) substrates, produce permanent, high-contrast labeling compatible with standard light microscopy and microdissection workflows.

### Immunofluorescence

- Antibody-based
- Panel design is important to ensure that signal spillover is minimal

Immunofluorescence (IF) enables multiplexed molecular visualization in tissue sections, allowing up to five distinct markers to be simultaneously detected using spectrally separated fluorophores. This approach combines molecular specificity with spatial context, making it particularly valuable for laser microdissection (LMD) of defined cell types, microenvironments, or rare subpopulations that cannot be reliably distinguished by morphology alone. Fluorescent labeling can target cell identity markers, signaling proteins, or extracellular components, offering rich contextual information while maintaining subcellular resolution.

Note: Here you should consider creating calibration points by etching the membrane with the LMD.

### Multiplex Immunofluorescence

- Antibody-based
- Increased panel design complexity and cost
- Increased lab work complexity OR expensive apparatus
- Enables the phenotyping of many cell types, particularly immune cells.

Multiplex immunofluorescence (mIF) enables the simultaneous visualization of numerous protein markers within a single tissue section, providing a detailed molecular map of cellular phenotypes and spatial relationships. Unlike conventional IF, mIF relies on iterative staining and imaging cycles or advanced multiplexing chemistries (e.g., tyramide signal amplification, DNA barcoding, or spectral unmixing) to expand marker capacity far beyond the limits of standard fluorophore sets. This makes it a powerful tool for defining complex tissue microenvironments and guiding laser microdissection (LMD) toward highly specific cellular niches or interaction zones identified by combinatorial marker expression patterns.

Note: Here you should consider creating calibration points by etching the membrane with the LMD.

## Microscopes and critical settings

Microscopes vary plenty

### Magnification

- 10X good enough for FlashDVP ROI manual annotation (not single cells).
- 20X minimum for image analysis driven DVP or single cell manual annotations.
- 40X and above: Allows for more granular morphological and more detailed cell-based analysis.

### Binning

- 1x1 binning, highest resolution, largest file sizes
- 2x2 binning: half the resolution, 1/4 the file size, higher signal-to-noise ration

### File formats

If you plan to just perform manual annotations, your bottlenect is getting your image into QuPath.  
QuPath can take in a large variety of formats, please consult the [Bioformats Compatibility Table](https://docs.openmicroscopy.org/bio-formats/5.8.2/supported-formats.html).

If you plan to perform image-analysis, ensure that your software solution can digest the file format. Perhaps consult your friendly image analyst for clarification.

### Protocols are coming (work in progress)