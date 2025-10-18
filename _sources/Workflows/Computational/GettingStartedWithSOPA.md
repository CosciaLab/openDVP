# Getting started with SOPA

## SOPA

SOPA is a wonderful pipeline, similar to MCMICRO but has some fundamental differences.

Similarly, for a overview of SOPA read the [paper](https://www.nature.com/articles/s41467-024-48981-z), for a more technical overview visit the [website](https://gustaveroussy.github.io/sopa/).

### General:

- SOPA runs on [Snakemake](https://snakemake.github.io/), nextflow version is in [development](https://nf-co.re/sopa/usage).
- SOPA uses the Spatialdata object natively, some experience with SpatialData is preferred.

### Advantages:

- SOPA can run on Windows.
- SOPA is easier to customize than MCMICRO. For adding a function quickly to your pipeline use this (Python required).
- SOPA can be used in four flavours: jupyter notebooks with python API, snakemake, nextflow, and CLI.
- SOPA is compatible with various modalities:  Xenium, Visium HD, MERSCOPE, CosMx, PhenoCycler, MACSima, Molecular Cartography.

### Disadvantages:

- More modality compatibility leads to complexity of setup.
- SOPA cannot perform **Illumination correction**, **Stitching and Registration**, or **Background Subtraction**.(Jose is considering helping out SOPA devs with these).
- Output is spatialdata object, not super friendly to get data out of.
- Segmentation output is `.geojson` not `.tif` (you can convert easily with spatialdata function).

## Setup SOPA locally in your computer (for testing subsets)

- Familiarize yourself with the [Snakemake framework](https://snakemake.github.io/).
- Follow SOPA [Getting Started](https://gustaveroussy.github.io/sopa/getting_started/).
- Follow SOPA's [snakemake tutorial](https://gustaveroussy.github.io/sopa/tutorials/snakemake/).

### Step 1: Ensure you have conda/mamba installed

- Download micromamba from [Micromamba Releases](https://github.com/mamba-org/micromamba-releases/releases). You might have to click `show all 27 assets` to see the version you need.
- Check that you can create environments and download packages

### Step 2: Create sopa environment

```console
conda create --name sopa python=3.12
```

```console
conda activate sopa
```

Install sopa (with cellpose extension) and snakemake

```console
pip install 'sopa[cellpose]' snakemake
```

for some reason cellpose 4 breaks sopa, we have to install cellpose 3

```console
pip install 'cellpose <4'
```

### Step 3: Download sopa defaults

SOPA have many processes, and many processes are modality specific. You can concatenate them as you want (not simple, not super complex either). To simplify things for now, We will use SOPA-provided defaults for dealing with ome.tif

In the terminal, go to directory you know and this will download the entire github repository.

```bash
git clone https://github.com/gustaveroussy/sopa.git
```

### Step 4: Edit default parameters

Open the following file inside the downloaded `sopa` directory:
`sopa/workflow/configs/misc/ome_tif.yaml`

This is the parameter file that sets the parameters for all the processes.  
For a deeper look into what each of these does check [parameter_guide](https://github.com/gustaveroussy/sopa/blob/main/workflow/config/example_commented.yaml)

We will change the cellpose parameters to the following to make it work with our demo image.

```yaml
# these are the settings you should have (ensure you save the file after you are done).
read:
  technology: ome_tif

patchify:
  patch_width_pixel: 400
  patch_overlap_pixel: 40

segmentation:
  cellpose:
    model_type: "nuclei"
    diameter: 25
    channels: [ "DAPI_bg" ]
    flow_threshold: 2
    cellprob_threshold: -6
    min_area: 25
    gaussian_sigma: 1

aggregate:
  aggregate_channels: true
  min_intensity_ratio: 0.1
  expand_radius_ratio: 0.1

explorer:
  ram_threshold_gb: 4
  pixel_size: 1
```

### Step 5: Run sopa with defaults

There are three paths you need to pass to the snakemake command:

1. path to your image
2. path to the edited config file
3. path to the workflow profile (provided by sopa download)

For the demo image copy:  
`RD_Coscia/Jose/__TestDatasets/TD_01_verysmall_mIF.ome.tif`  
and place it in here:  
`sopa/data/TD_01_verysmall_mIF.ome.tif`

SOPA command template

```bash
snakemake \
    --config data_path=$PATH_TO_IMAGE \
    --configfile=$PATH_TO_YAML \
    --workflow-profile ./workflow/profile/local \
    --cores 4
```

I suggest you run the command from the sopa directory, and it will look something like this:

```bash
snakemake \
    --config data_path=./data/TD_01_verysmall_mIF.ome.tif \
    --configfile=./config/misc/ome_tif.yaml \
    --workflow-profile ./workflow/profile/local \
    --cores 6
```

Note, for Windows you must replace:

- < \\ > with a caret < ^ > for the traditional Command Prompt (cmd.exe) and a backtick <`> for PowerShell.
- you can also remove them and create a single line command in an editor, and then copy paste.


### Step 6: Check post-run

Before:

```console
.
├── command.txt
├── config
│   └── ome_tif.yaml
└── data
    └── TD_01_verysmall_mIF.ome.tif
```

After:

```console
.
├── command.txt
├── config
│   └── ome_tif.yaml
└── data
    ├── TD_01_verysmall_mIF.ome.explorer
    ├── TD_01_verysmall_mIF.ome.tif
    └── TD_01_verysmall_mIF.ome.zarr
```

The `.explorer` file can be opened with the Xenium explorer software (free).
The `.zarr` file can be opened with `spatialdata`.

## Run SOPA on HPC

- Create a conda environment with `snakemake`, `snakemake-slurm`, and `snakemake-executor-plugin-slurm`.
- Use profile that will use the executor plugin check [Snakemake with Slurm](https://hpc-docs.cubi.bihealth.org/slurm/snakemake/)

<details>
<summary> Example bash script for single snakemake run </summary>

```bash
#!/bin/bash

#SBATCH --partition=medium
#SBATCH --job-name=smk_main_${job_prefix}
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=250M
#SBATCH --output=slurm_logs/%x-%j.log

FILE_PATH=$1  # The file path is passed as an argument to the script
filename=$(basename "$FILE_PATH")
job_prefix="${filename:0:4}"

export SBATCH_DEFAULTS="--output=slurm_logs/%x-%j.log"

date

source /data/cephfs-1/home/users/jnimoca_m/work/miniconda/etc/profile.d/conda.sh

echo "Activate sopa env"
conda activate sopa

echo "Running snakemake with file $FILE_PATH"
srun snakemake \
--config data_path="$FILE_PATH" \
--use-conda -j 100 --profile=cubi-v1

date
```
</details>

<details>
<summary> Example array script to run many images </summary>

```bash
files=(
    "992_backsub.ome.tif"
    "993_backsub.ome.tif"
    "994_backsub.ome.tif"
    "997_backsub.ome.tif"
)

for file in "${files[@]}"; do
    echo "Submitting job for file $file"
    sbatch snakemake_run.sh "data/input/${file}"
done
```

</details>

<details>
<summary> Example custom snakemake file; pipeline with every step and parameters </summary>

```python
configfile: "ome_tif.yaml"

# version 2.1.0
# date: 09.09.2024
# comments:
# v2.0.0: increased time of quantification rule to 6 hours and changed partition to medium
# v2.1.0: added pyrimidize

from utils import WorkflowPaths, Args

paths = WorkflowPaths(config)
args = Args(paths, config)

localrules: all

rule all:
    input:
        f"data/quantification/{paths.sample_name}_quantification.csv",
        f"data/processed_images/{paths.sample_name}_8bit_pyrimidized.ome.tif"
    shell:
        """
        echo 🎉 Successfully run sopa
        echo → SpatialData output directory: {paths.sdata_path}
        """

rule downscale_image:
    input:
        paths.data_path,
    output:
        path_downscaled_image = f"data/processed_images/{paths.sample_name}_8bit.tif"
    conda:
        "sopa"
    resources:
        mem_mb=256_000,
        partition="short",
        runtime="3h",
    threads: 4
    shell:
        """
        mkdir -p ./data/processed_images

        python scripts/downscale_image_to8bit.py \
        --input {paths.data_path} \
        --output {output.path_downscaled_image} \
        """

rule pyrimidize:
    input:
        processed_imaged = f"data/processed_images/{paths.sample_name}_8bit.tif"
    output:
        pyrimidized_image = f"data/processed_images/{paths.sample_name}_8bit_pyrimidized.ome.tif"
    conda:
        "sopa"
    resources:
        mem_mb=128_000,
        partition="medium",
        runtime="6h",
    threads: 4
    params:
        tilesize = config["pyramid"]["tile-size"],
    shell:
        """
        python scripts/pyramidize.py \
        --input {input.processed_imaged} \
        --output {output.pyrimidized_image} \
        --tile-size {params.tilesize}
        """

rule to_spatialdata:
    input:
        paths.data_path if config["read"]["technology"] != "uniform" else [],
    output:
        paths.sdata_zgroup if paths.data_path else [],
    conda:
        "sopa"
    resources:
        mem_mb=128_000,
        partition="short",
        runtime="3h",
    threads: 2
    params:
        args_reader = str(args['read'])
    shell:
        """
        sopa read {paths.data_path} --sdata-path {paths.sdata_path} {params.args_reader}
        """

checkpoint patchify_cellpose:
    input:
        paths.sdata_zgroup,
    output:
        patches_file = paths.smk_patches_file_image,
        patches = touch(paths.smk_patches),
    params:
        args_patchify = str(args["patchify"].where(contains="pixel")),
    conda:
        "sopa"
    resources:
        mem_mb=32_000,
        partition="short",
        runtime="1h",
    threads: 4
    shell:
        """
        sopa patchify image {paths.sdata_path} {params.args_patchify}
        """

rule patch_segmentation_cellpose:
    input:
        paths.smk_patches_file_image,
        paths.smk_patches,
    output:
        paths.smk_cellpose_temp_dir / "{index}.parquet",
    conda:
        "sopa"
    resources:
        mem_mb=32_000,
        partition="short",
        runtime="1h",
    threads: 4
    params:
        args_cellpose = str(args["segmentation"]["cellpose"]),
    shell:
        """
        sopa segmentation cellpose {paths.sdata_path} --patch-dir {paths.smk_cellpose_temp_dir} --patch-index {wildcards.index} {params.args_cellpose}
        """


def get_input_resolve(name, dirs=False):
    def _(wilcards):
        with getattr(checkpoints, f"patchify_{name}").get(**wilcards).output.patches_file.open() as f:
            return paths.cells_paths(f.read(), name, dirs=dirs)
    return _

rule resolve_cellpose:
    input:
        get_input_resolve("cellpose"),
    output:
        touch(paths.smk_cellpose_boundaries),
    conda:
        "sopa"
    resources:
        mem_mb=32_000,
        partition="short",
        runtime="1h",
    threads: 4
    shell:
        """
        sopa resolve cellpose {paths.sdata_path} --patch-dir {paths.smk_cellpose_temp_dir}
        """

rule rasterize:
    input:
        paths.sdata_zgroup,
        paths.smk_cellpose_boundaries
    output:
        mask_tif=f"data/masks/{paths.sample_name}_mask.tif"
    conda:
        "sopa"
    resources:
        mem_mb=256_000,
        partition="short",
        runtime="1h",
    threads: 4
    shell:
        """
        mkdir -p ./data/masks

        python scripts/rasterize.py \
        --input {paths.sdata_path} \
        --output {output.mask_tif}
        """

rule expand_markers:
    input:
        mask = f"data/masks/{paths.sample_name}_mask.tif"
    output:
        exp_mask = f"data/masks/{paths.sample_name}_mask_expanded.tif"
    resources:
        mem_mb=256_000,
        partition="short",
        runtime="1h",
    threads: 4
    params:
        pixels = config["expand"]["pixels"]
    shell:
        """
        python scripts/expand_mask_singlemask.py \
        --input {input.mask} \
        --output {output.exp_mask} \
        --pixels {params.pixels} \
        """

rule quantify:
    input:
        image = paths.data_path,
        mask = f"data/masks/{paths.sample_name}_mask_expanded.tif",
        markers = "data/input/markers.csv"
    output:
        quantification = f"data/quantification/{paths.sample_name}_quantification.csv"
    conda:
        "sopa"
    resources:
        mem_mb=256_000,
        partition="medium",
        runtime="6h",
    threads: 4
    params:
        math = config["quantify"]["math"],
        quantile = config["quantify"]["quantile"],
    shell:
        """
        mkdir -p ./data/quantification

        python scripts/quant.py \
        --image {input.image} \
        --label {input.mask} \
        --markers {input.markers} \
        --output {output.quantification} \
        --math {params.math} \
        --quantile {params.quantile}
        """

```

</details>

Thank you for your interest!
