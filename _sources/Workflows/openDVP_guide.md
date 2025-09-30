# Setup imaging processing pipelines for the MDC

### Setting Expectations

If you want to run an image processing and analyis pipeline, it will take some time to setup.  
Depending on your familiarity to things like HPC, Nextflow, Snakemake, Python; the amount of effort will vary.

### General tips

- Before running your full data, try things out with very small data(<100MB). Where the feedback loop is less than 2 minutes. This will allow you to play with settings, parameters, and lower the cost of failure. You will fail, and you will learn.
- Asking LLMs for help is great, just be careful, some of these pipelines and packages are not very popular and there are tendencies to hallucinate.
- Ask humans for help, but help with log files, tracebacks, and context. Just like LLMs the more context the better we understand your problem.

## MCMICRO

MCMICRO is a wonderful pipeline developed by a group of image analysts. It runs on [Nextflow](https://www.nextflow.io/) which allows it to be run on any computational environment. It is powerful, modular, and scalable. Once setup it can be ran in parallel with as many files as your compute allows.

For a conceptual overview read the paper [MCMICRO paper](https://www.nature.com/articles/s41592-021-01308-y), for a more technical overview I suggest you first read the documentation of the [MCMICRO website ](https://mcmicro.org/overview/) and define what steps are important for your workflow.

## How to setup in the BIH HPC

### Step 1: Ensure you can access the cluster

Please take the time to read and go through the documentation at [Getting Access](https://hpc-docs.cubi.bihealth.org/admin/getting-access/).

<details>
<summary> Check access is working, you see this message: </summary>

```bash

❯ ssh jnimoca_m@hpc-login-1.cubi.bihealth.org
Welcome to the BIH HPC 4 Research Cluster!

 You are on a login node.
 You should not do much here besides starting "screen"/"tmux"
 and using Slurm commands such as "sbatch"/"srun" for connecting to nodes.

Helpdesk: hpc-helpdesk@bih-charite.de
Mailing List: bih-cluster@charite.de
Documentation: https://hpc-docs.cubi.bihealth.org
OnDemand Portal: https://hpc-portal.cubi.bihealth.org
HPC Access Portal: https://hpc-access.cubi.bihealth.org
Last login: Thu Aug  7 12:54:25 2025 from 141.80.221.57
-bash: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)
-bash: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)
[jnimoca_m@hpc-login-1 ~]$

```

</details>

### Step 2: Ensure you can move files to cluster

Follow [BIH HPC instructions](https://hpc-docs.cubi.bihealth.org/how-to/service/file-exchange/) for using software to transfer files.
Technically you should be able to connect from virtual machines. Sometimes there are issues with IP adresses, please contact IT for help.

<details>
<summary> Check if it is working </summary>
You are able to drag and drop files, and you see them in the command line.
</details>


### Step 3: Ensure there is enough space in your directory

Familiarize yourself with how [storage works in the BIH HPC](https://hpc-docs.cubi.bihealth.org/storage/storage-locations/).

In the terminal, connect to the HPC. Go to your home directory and check what is there with `ls`

```bash
# Your home directory should look something like this:
[jnimoca_m@hpc-login-1 ~]$ ls
bin  ondemand  scratch  work
```

To know how much data there is please check [BIH HPC dashboard](https://hpc-access.cubi.bihealth.org/)  
Consider that for each 10gb of raw data you will need approx. 100gb of storage space.  
If you need more storage space, bring it up in meeting or Mattermost. We might have to ask for extra storage.

### Step 4: Create environment in the HPC to run nextflow pipelines

```python
# these are the problems

# (1) Java
[jnimoca_m@hpc-login-1 ~]$ java -version
-bash: java: command not found
#(2) Nextflow
[jnimoca_m@hpc-login-1 ~]$ nextflow
-bash: /data/cephfs-1/home/users/jnimoca_m/bin/nextflow: No such file or directory
```

We will use **miniforge**(Conda analogue) to install these, please read the [Software Installation with Conda](https://hpc-docs.cubi.bihealth.org/best-practice/software-installation-with-conda/).

```python
# ensure you are running an interactive session in the HPC
#this commands creates and interactive session with more computational resources
hpc-login-1:~$ srun --mem=5G --pty bash -i
hpc-cpu-123:~$ wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
hpc-cpu-123:~$ bash Miniforge3-Linux-x86_64.sh -b -f -p $HOME/work/miniforge
hpc-cpu-123:~$ eval "$(/$HOME/work/miniforge/bin/conda shell.bash hook)"
hpc-cpu-123:~$ conda init
hpc-cpu-123:~$ conda config --set auto_activate_base false

# these commands to ensure conda channel is not used
hpc-cpu-123:~$ conda config --add channels bioconda
hpc-cpu-123:~$ conda config --add channels conda-forge
hpc-cpu-123:~$ conda config --set channel_priority strict
```

Conda hopefully is properly installed.

<details>
<summary>Check conda is working:</summary>

```bash

[jnimoca_m@hpc-cpu-61 ~]$ conda
usage: conda [-h] [-v] [--no-plugins] [-V] COMMAND ...

conda is a tool for managing and deploying applications, environments and packages.

options:
  -h, --help          Show this help message and exit.
  -v, --verbose       Can be used multiple times. Once for detailed output, twice for INFO logging, thrice for DEBUG logging, four times for
                      TRACE logging.
  --no-plugins        Disable all plugins that are not built into conda.
  -V, --version       Show the conda version number and exit.

commands:
  The following built-in and plugins subcommands are available.

  COMMAND
    activate          Activate a conda environment.
    clean             Remove unused packages and caches.
    compare           Compare packages between conda environments.
    config            Modify configuration values in .condarc.
    content-trust     Signing and verification tools for Conda
    create            Create a new conda environment from a list of specified packages.
    deactivate        Deactivate the current active conda environment.
    doctor            Display a health report for your environment.
    export            Export a given environment
    info              Display information about current conda install.
    init              Initialize conda for shell interaction.
    install           Install a list of packages into a specified conda environment.
    list              List installed packages in a conda environment.
    notices           Retrieve latest channel notifications.
    package           Create low-level conda packages. (EXPERIMENTAL)
    remove (uninstall)
                      Remove a list of packages from a specified conda environment.
    rename            Rename an existing environment.
    repoquery         Advanced search for repodata.
    run               Run an executable in a conda environment.
    search            Search for packages and display associated information using the MatchSpec format.
    update (upgrade)  Update conda packages to the latest compatible version.

```

</details>

Now we will install the necessary packages

```python
# Create a conda environment called 'mcmicro'
hpc-cpu-123:~$ conda create -n mcmicro -y

# this activates the environment
hpc-cpu-123:~$ conda activate mcmicro

# this install java and singularity
hpc-cpu-123:~$ conda install openjdk singularity -y
```

Now we proceed to install **Nextflow**

```python
# Download and run nextflow executable
hpc-cpu-123:~$ curl -s https://get.nextflow.io | bash

# Trust and make the file executable
hpc-cpu-123:~$ chmod +x nextflow

# move the file somewhere your environment can find it
hpc-cpu-123:~$ mkdir -p $HOME/.local/bin/
hpc-cpu-123:~$ mv nextflow $HOME/.local/bin/
```

<details>
<summary>Check nextflow is working part 1</summary>

```bash
(mcmicro) [jnimoca_m@hpc-cpu-213 ~]$ nextflow info
  Version: 25.04.7 build 5955
  Created: 08-09-2025 13:29 UTC (15:29 CEST)
  System: Linux 5.14.0-570.21.1.el9_6.x86_64
  Runtime: Groovy 4.0.26 on OpenJDK 64-Bit Server VM 22.0.1-internal-adhoc.conda.src
  Encoding: UTF-8 (UTF-8)
# this means nextflow is found in your PATH and can be run
```

</details>


<details>
<summary>Check nextflow is working part 2:</summary>

```bash
(mcmicro) [jnimoca_m@hpc-cpu-213 ~]$ nextflow run nextflow-io/hello

 N E X T F L O W   ~  version 25.04.7

NOTE: Your local project version looks outdated - a different revision is available in the remote repository [2ce0b0e294]
Launching `https://github.com/nextflow-io/hello` [naughty_kimura] DSL2 - revision: afff16a9b4 [master]

executor >  local (4)
[31/222763] sayHello (4) [100%] 4 of 4 ✔
Bonjour world!

Hello world!

Ciao world!

Hola world!

# this means you have connection to download pipelines and run them

```

</details>

### Step 5: Run Nextflow with test data

From each code block, please run the commands one at a time.

```bash
# create directory for test data
(mcmicro) [jnimoca_m@hpc-cpu-213 ~]$ mkdir -p ~/work/test1/
# download test data from internet to test directory
(mcmicro) [jnimoca_m@hpc-cpu-213 ~]$ nextflow run labsyspharm/mcmicro/exemplar.nf --name exemplar-001 --path ~/work/test1/

 N E X T F L O W   ~  version 25.04.7

NOTE: Your local project version looks outdated - a different revision is available in the remote repository [b0175102db]
Launching `https://github.com/labsyspharm/mcmicro` [awesome_cori] DSL2 - revision: 9122980d88 [master]

executor >  local (7)
[db/88249f] getImages (2)       [100%] 3 of 3 ✔
[6a/e497fd] getIllumination (2) [100%] 3 of 3 ✔
[41/ef43e7] getMarkers          [100%] 1 of 1 ✔
[-        ] getParams           -
Completed at: 12-Sep-2025 15:20:05
Duration    : 6m 33s
CPU hours   : 0.2
Succeeded   : 7
```

```bash
# Run with -profile singularity, otherwise it defaults to Docker and it will fail.
(mcmicro) [jnimoca_m@hpc-cpu-213 ~]$ nextflow run labsyspharm/mcmicro --in ~/work/test1/exemplar-001 -profile singularity

 N E X T F L O W   ~  version 25.04.7

NOTE: Your local project version looks outdated - a different revision is available in the remote repository [b0175102db]
Launching `https://github.com/labsyspharm/mcmicro` [scruffy_aryabhata] DSL2 - revision: 9122980d88 [master]

executor >  local (3)
executor >  local (4)
[-        ] staging:phenoimager2mc          -
[-        ] illumination                    -
[b3/8c40c6] registration:ashlar (1)         [100%] 1 of 1 ✔
[-        ] background:backsub              -
[-        ] dearray:coreograph              -
[-        ] segmentation:roadie:runTask     -
[fa/ee9f6f] segmentation:worker (unmicst-1) [100%] 1 of 1 ✔
[e6/07e148] segmentation:s3seg (1)          [100%] 1 of 1 ✔
[5c/8570c6] quantification:mcquant (1)      [100%] 1 of 1 ✔
[-        ] downstream:worker               -
[-        ] viz:autominerva                 -
Completed at: 12-Sep-2025 15:43:15
Duration    : 18m 3s
CPU hours   : 0.3
Succeeded   : 4
```

```python
# Let's check the outputs
(mcmicro) [jnimoca_m@hpc-cpu-213 ~]$ tree ~/work/test1/exemplar-001
.
└── exemplar-001
    ├── illumination
    │   ├── exemplar-001-cycle-06-dfp.tif
    │   ├── exemplar-001-cycle-06-ffp.tif
    │   ├── exemplar-001-cycle-07-dfp.tif
    │   ├── exemplar-001-cycle-07-ffp.tif
    │   ├── exemplar-001-cycle-08-dfp.tif
    │   └── exemplar-001-cycle-08-ffp.tif
    ├── markers.csv
    ├── probability-maps
    │   └── unmicst
    │       └── exemplar-001-pmap.tif
    ├── qc
    │   ├── metadata.yml
    │   ├── params.yml
    │   ├── provenance
    │   │   ├── ashlar.log
    │   │   ├── ashlar.sh
    │   │   ├── mcquant-1.log
    │   │   ├── mcquant-1.sh
    │   │   ├── s3seg-1.log
    │   │   ├── s3seg-1.sh
    │   │   ├── unmicst-1.log
    │   │   └── unmicst-1.sh
    │   ├── s3seg
    │   │   └── unmicst-exemplar-001
    │   │       ├── cellOutlines.ome.tif
    │   │       └── nucleiOutlines.ome.tif
    │   └── unmicst
    │       └── exemplar-001_Preview_1.tif
    ├── quantification
    │   └── exemplar-001--unmicst_cell.csv
    ├── raw
    │   ├── exemplar-001-cycle-06.ome.tiff
    │   ├── exemplar-001-cycle-07.ome.tiff
    │   └── exemplar-001-cycle-08.ome.tiff
    ├── registration
    │   └── exemplar-001.ome.tif
    └── segmentation
        └── unmicst-exemplar-001
            ├── cell.ome.tif
            └── nuclei.ome.tif
```

</details>

<details>
<summary> Reasoning question: Why did it take so long to run that small (400mb) dataset?</summary>

Because you ran that entire analysis in that single node, with the default resources.  
When you run your dataset nextflow will dispatch jobs based on the requirements of each process.

</details>

### Step 6: How to run MCMICRO with HPC job

Let's rerun the same dataset with a script

```bash
# let's create the script file

# go to demo data
(mcmicro) [jnimoca_m@hpc-cpu-213 ~]$ cd work/test1/exemplar-001/
# create script file
(mcmicro) [jnimoca_m@hpc-cpu-213 exemplar-001]$ touch script.sh
# edit script 
(mcmicro) [jnimoca_m@hpc-cpu-213 exemplar-001]$ nano script.sh 
# then you type everything you need and press (CTRL+X) to leave
# consider creating this file in a text editor (VSCode) and copy paste into it
```

Here is the demo script:

```bash
#!/bin/bash
#SBATCH --job-name=test_job         # Job name
#SBATCH --time=4:00:00              # Time limit hrs:min:sec
#SBATCH --mem=10G                   # Memory for orchestrating node
#SBATCH --cpus-per-task=2           # Number of CPU cores for orchestrating node

eval "$(conda shell.bash hook)"     # This exposes conda hook to node
conda activate mcmicro              # This activates environment in node
PATH_TO_DATA="/data/cephfs-1/home/users/jnimoca_m/work/test1/exemplar-001"
nextflow run labsyspharm/mcmicro --in $PATH_TO_DATA -profile singularity
```

Then run your script:

```bash
#the location of the script doesnt matter as long as the path in it directs to your data
(mcmicro) [jnimoca_m@hpc-cpu-213 exemplar-001]$ sbatch demo_script.sh 
sbatch: routed your job to partition short
Submitted batch job 18622303
```

### Checklist

- Ensure access to HPC
- Ensure you can move files into HPC environment (consider image sizes)
- Ensure enough storage space is available for your images and processing steps
- Ensure HPC can run **java**
- Ensure HPC can run **apptainer/singularity** images
- Ensure HPC can run **nextflow**
- Run **MCMICRO** [demo data](https://mcmicro.org/datasets/) directly on interactive session
- Run **MCMICRO** [demo data](https://mcmicro.org/datasets/) with a HPC script

SUCCESS, you managed to install everything, now let's dig into the biology!

## How to prepare your images for MCMICRO

Please familiarize yourself with [MCMICRO's Input/Output expectactions](https://mcmicro.org/io.html)

### Project organization

Minimum requiremets are the following:

```bash
myproject/
├── markers.csv
├── params.yml
└── raw/
```
### markers.csv

- cannot be renamed
- column names must be: `cycle_number`, `channel_number`, and `marker_name`
- `marker_name` values must all be unique
- Separator **must** be a *comma* (,) -- Be careful with excel and german defaults, use libreOffice if needed.

markers.csv schema (without background subtraction):

| cycle_number | channel_number | marker_name |
|--------------|----------------|-------------|
| 1            | 1              | CDH1        |
| 1            | 2              | IDO         |
| 1            | 3              | panCK       |
| 1            | 4              | CD20        |
| 1            | 5              | DAPI_1      |
| 2            | 6              | aSMA        |
| 2            | 7              | CD56        |
| 2            | 8              | FOXP3       |
| 2            | 9              | GranzymeB   |


markers.csv (with [background subtraction](https://github.com/SchapiroLabor/Background_subtraction)):

| cycle_number | channel_number | marker_name | background | exposure | remove |
|--------------|----------------|-------------|------------|----------|--------|
| 1            | 1              | 750_bg      |            | 200      | TRUE   |
| 1            | 2              | 647_bg      |            | 50       | TRUE   |
| 1            | 3              | 555_bg      |            | 200      | TRUE   |
| 1            | 4              | 488_bg      |            | 100      | TRUE   |
| 1            | 5              | DAPI_bg     |            | 3.5      |        |
| 2            | 6              | CDH1        | 750_bg     | 50       |        |
| 2            | 7              | PDL1        | 647_bg     | 12       |        |
| 2            | 8              | panCK       | 555_bg     | 15       |        |
| 2            | 9              | CD20        | 488_bg     | 15       |        |

### params.yml

Familiarize yourself with the [official documentation](https://mcmicro.org/parameters/)

- holds all parameters, for the pipeline and for each module
- must be a `.yml` file
- must be passed with `nextflow run ... --params params.yml

Example params.yml for just **stitching and registration** of a `.czi` file:

```yml
workflow:
start-at: illumination
stop-at: registration

options:
    ashlar: --flip-y --align-channel 1 -m 50 --filter-sigma 1
```

Example params.yml including **background subtraction**

```yml
workflow:
    start-at: illumination
    stop-at: background
    background: true
    background-method: backsub

options:
    ashlar: --flip-y --align-channel 4 -m 50 --filter-sigma 1
```

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