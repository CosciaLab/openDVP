# Setup imaging processing pipelines for the MDC

## Setting Expectations

If you want to run an image processing and analyis pipeline, it will take some time to setup. Depending on your familiarity to things like HPC, Nextflow, Snakemake, Python; the amount of effort will vary.

### General tips

- Before running your full data, try things out with very small data(<100MB). Where the feedback loop is less than 2 minutes. This will allow you to play with settings, parameters, and lower the cost of failure. You will fail, and you will learn.
- Asking LLMs for help is great, just be careful, some of these pipelines and packages are not very popular and the tendency to hallucinate is greater.
- Ask humans for help, but help with log files, tracebacks, and context. Just like LLMs the more context the better we understand your problem.

## MCMICRO

MCMICRO is a wonderful pipeline developed by a group of image analysts. It runs on [Nextflow](https://www.nextflow.io/) which allows it to be run on any computational environment. It is powerful, modular, and scalable. Once setup it can be ran in parallel with as many files as your compute allows.

For a conceptual overview read the paper [MCMICRO paper](https://www.nature.com/articles/s41592-021-01308-y), for a more technical overview I suggest you first read the documentation of the [MCMICRO website ](https://mcmicro.org/overview/) and define what steps are important for your workflow.

## How to setup: simplified

### Windows users

- This will be a challenge because you will need [WSL subsystem](https://learn.microsoft.com/en-us/windows/wsl/install). This is not simple, consider running in your cluster, or friendly bioinformatician with MacOS or Linux.

### MacOS and Linux

- Follow instructions in [MCMICRO Installation](https://mcmicro.org/tutorial/installation.html). 
- You will need to install **Java** and **Docker** on your machine.

### High Performance Cluster (HPC)

- You need an environment with **Java**, follow HPC admin suggestions (conda works for me).
- Install [Singularity/Apptainer](https://docs.sylabs.io/guides/3.0/user-guide/quick_start.html), a Docker analogue for the HPC.
- Install [Nextflow](https://www.nextflow.io/), the pipeline manager.
- Familiarize yourself how to run scripts in the HPC.  
- For example, in SLURM I would use `$ sbatch myscript.sh`
        <details>
        <summary> Bash Script Example</summary>

        ```bash
        #!/bin/bash
        #SBATCH --job-name=P30E01_Tonsil    # Job name
        #SBATCH --output=P30E01.%j.out      # Output file
        #SBATCH --error=P30E01.%j.err       # Error file
        #SBATCH --time=24:00:00             # Time limit hrs:min:sec
        #SBATCH --mem=10G                   # Memory per node
        #SBATCH --cpus-per-task=2           # Number of CPU cores per task
        #SBATCH --partition=medium          # Partition/queue name (check with your HPC)

        eval "$(conda shell.bash hook)"     # This exposes conda hook to node
        conda activate mcmicro              # This activates environment in node

        PATH_TO_DATA="/data/cephfs-1/home/users/jnimoca_m/work/P30_SF_HNSCC/P30E01_Tonsil"

        nextflow run labsyspharm/mcmicro --in $PATH_TO_DATA -profile singularity --params $PATH_TO_DATA/P30E01_params.yml

        echo "--end--"

        ```

        </details>

### Checklist

- Ensure access to HPC
- Ensure you can move files into HPC environment (consider image sizes)
- Ensure enough storage space is available for your images and processing steps
- Ensure HPC can run **java**
- Ensure HPC can run **apptainer/singularity** images
- Ensure HPC can run **nextflow**
- Run **MCMICRO** [demo data](https://mcmicro.org/datasets/), I suggest Exemplar001. You can download it with nextflow command.

### Optimize MCMICRO for your images

MCMICRO has many steps, each step has many parameters that can be optimized to your images.  
All of these parameters should be passed to MCMICRO by means of a `params.yml` file. This is the file that you pass on to your `nextflow run` command.
<details>
<summary> Example params.yml file </summary>

```yml
workflow:
start-at: illumination
stop-at: registration

options:
    ashlar: --flip-y --align-channel 1 -m 50 --filter-sigma 1
```
</details>

Please check the [MCMICRO website](https://mcmicro.org/parameters/) for all potential parameters.  
I suggest for modules such as segmentation to run your segmentation locally with a subset of your images.

### Considerations, ideas, and tips

- You do not have to run everything all the time, sometimes I like to run MCMICRO just for (1) Illumination correction and (2) Stitching and registration and the move on to another software. Make use of the workflow options like `stop-at`.  
- Inside the [nextflow.config](https://github.com/labsyspharm/mcmicro/blob/master/nextflow.config) file, MCMICRO developers link the default computational requirements for each module. They have set `profiles`. For example here you can find the [WSI defaults](https://github.com/labsyspharm/mcmicro/blob/master/config/nf/wsi.config) profile optimized for Whole slide imaging, and here the `profile` for  [TMA defaults](https://github.com/labsyspharm/mcmicro/blob/master/config/nf/tma.config). If you feel you need your own computational requirements, I suggest you fork the MCMICRO github repository, modify it, and then run mcmicro like this `nextflow run josenimo/mcmicro`. Nextflow is smart and will use that github repository instead.
- For MDC peeps, feel free to use [josenimo/mcmicro](https://github.com/josenimo/mcmicro/) with `-profile singularity` in your `nextflow run` command. 
- Nextflow has a nice `-resume` parameter to automatically check what already ran, and go on from there.

## SOPA

SOPA is a wonderful pipeline, similar to MCMICRO but has some fundamental differences.

Similarly, for a overview of SOPA read the [paper](https://www.nature.com/articles/s41467-024-48981-z), for a more technical overview visit the [website](https://gustaveroussy.github.io/sopa/).

- SOPA runs on [Snakemake](https://snakemake.github.io/), nextflow version is in [development](https://nf-co.re/sopa/usage).
- SOPA is easier to customize than MCMICRO. For adding a function quickly to your pipeline use this (Python required).
- SOPA can be used in four flavours: jupyter notebooks with python API, snakemake, nextflow, and CLI.
- SOPA uses the Spatialdata object natively, experience with SpatialData is recommended.
- SOPA is compatible with various modalities:  Xenium, Visium HD, MERSCOPE, CosMx, PhenoCycler, MACSima, Molecular Cartography.

### Run SOPA locally

- Familiarize yourself with the [Snakemake framework](https://snakemake.github.io/).
- Follow SOPA [Getting Started](https://gustaveroussy.github.io/sopa/getting_started/) to install necessary packages.
- Follow SOPA's [snakemake tutorial](https://gustaveroussy.github.io/sopa/tutorials/snakemake/)

### Run SOPA on HPC

- Create environment with `snakemake`, `snakemake-slurm`, and `snakemake-executor-plugin-slurm`.
- Use profile that will use the executor plugin
- For MDC peeps check [Snakemake with Slurm](https://hpc-docs.cubi.bihealth.org/slurm/snakemake/)

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
<summary> Example Snakemake file; pipeline with every step and parameters </summary>

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