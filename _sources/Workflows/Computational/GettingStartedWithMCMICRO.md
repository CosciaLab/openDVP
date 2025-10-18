# Getting started with MCMICRO

## MCMICRO

MCMICRO is a wonderful pipeline developed by a group of image analysts. It runs on [Nextflow](https://www.nextflow.io/) which allows it to be run on any computational environment. It is powerful, modular, and scalable. Once setup it can be ran in parallel with as many files as your compute allows.

For a conceptual overview read the paper [MCMICRO paper](https://www.nature.com/articles/s41592-021-01308-y), for a more technical overview I suggest you first read the documentation of the [MCMICRO website ](https://mcmicro.org/overview/) and define what steps are important for your workflow.

### Setting Expectations

If you want to run an image processing and analyis pipeline, it will take some time to setup.  
Depending on your familiarity to things like HPC, Nextflow, Snakemake, Python; the amount of effort will vary.
This guide is for users that are not experienced with MCMICRO or HPC, so it goes step-by-step. In the end going through the documentation of the various softwares is the only way to explore all the possibilities.

### General tips

- Before running your full data, try things out with very small data(<100MB). Where the feedback loop is less than 2 minutes. This will allow you to play with settings, parameters, and lower the cost of failure. You will fail, and you will learn.
- Asking LLMs for help is great, just be careful, some of these pipelines and packages are not very popular and there are tendencies to hallucinate.
- Ask humans for help, but help with log files, tracebacks, and context. Just like LLMs the more context the better we understand your problem.

## How to setup in MacOS/Linux environment

## How to setup in the BIH HPC

### Step 1: Ensure you can access the cluster

Please take the time to read and go through the documentation at [Getting Access](https://hpc-docs.cubi.bihealth.org/admin/getting-access/).

<details>
<summary> Check access is working, you see this message: </summary>

```console
ssh <USERNAME>@hpc-login-1.cubi.bihealth.org
```

Output:

```bash
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
```

</details>

### Step 2: Ensure you can move files to cluster

Follow [BIH HPC instructions](https://hpc-docs.cubi.bihealth.org/how-to/service/file-exchange/) for using software to transfer files.
Technically you should be able to connect from virtual machines. Sometimes there are issues with IP adresses, please contact IT for help.

Download FileZilla, WinSCP, or CyberDuck for easy file transfer.  

- MacOS: https://filezilla-project.org/  
- Windows: https://winscp.net/eng/download.php  
- Both: https://cyberduck.io/  

#### Important Settings

- Use SFTP

#### MDC Users login nodes

- username_m@hpc-login-1.cubi.bihealth.org
- username_m@hpc-login-2.cubi.bihealth.org

<details>
<summary> Check if it is working </summary>
You are able to drag and drop files, and you see them in the command line.
</details>


### Step 3: Ensure there is enough space in your directory

Familiarize yourself with how [storage works in the BIH HPC](https://hpc-docs.cubi.bihealth.org/storage/storage-locations/).

In the terminal, connect to the HPC. Go to your home directory and check what is there with `ls`

```console
ls
```

```bash
# Your home directory should look something like this:
bin  ondemand  scratch  work
```

To know how much data there is please check [BIH HPC dashboard](https://hpc-access.cubi.bihealth.org/)  
Consider that for each 10gb of raw data you will need approx. 100gb of storage space.  
If you need more storage space, bring it up in meeting or Mattermost. We might have to ask for extra storage.

### Step 4: Create environment in the HPC to run nextflow pipelines

We will use **miniforge**(Conda analogue) to install these, please read the [Software Installation with Conda](https://hpc-docs.cubi.bihealth.org/best-practice/software-installation-with-conda/).

#### Step 4.1: Activate interactive session

```console
srun --mem=5G --pty bash -i
```

#### Step 4.2: Download miniforge

```console
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
```

#### Step 4.3: Install miniforge

```console
bash Miniforge3-Linux-x86_64.sh -b -f -p $HOME/work/miniforge
```

#### Step 4.4: Make conda command accesible

```console
eval "$(/$HOME/work/miniforge/bin/conda shell.bash hook)"
```

#### Step 4.5: Initialize conda

```console
conda init
```

#### Step 4.6: Change conda configuration

```console
conda config --set auto_activate_base false
```

```console
conda config --add channels bioconda conda-forge
```

```console
conda config --set channel_priority strict
```

Conda hopefully is properly installed.

<details>
<summary>Check conda is working:</summary>

```console
conda
```

Should output the following:

```bash
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

### Step 4.7: Install the necessary packages 

Create a conda environment called 'mcmicro'

```console
conda create -n mcmicro -y
```

This activates the environment

```console
conda activate mcmicro
```

This install java and singularity

```console
conda install openjdk singularity -y
```

#### Step 4.8: Install **Nextflow**

Download and run nextflow executable

```console
curl -s https://get.nextflow.io | bash
```

Trust and make the file executable

```console
chmod +x nextflow
```

Move the file somewhere your environment can find it

```console
mkdir -p ~/.local/bin/
```

Move it to the local bin

```console
mv nextflow ~/.local/bin/
```

#### Step 4.9: Check Nextflow is working

```console
nextflow info
```

Output:

```bash
Version: 25.04.7 build 5955
Created: 08-09-2025 13:29 UTC (15:29 CEST)
System: Linux 5.14.0-570.21.1.el9_6.x86_64
Runtime: Groovy 4.0.26 on OpenJDK 64-Bit Server VM 22.0.1-internal-adhoc.conda.src
Encoding: UTF-8 (UTF-8)
# this means nextflow is found in your PATH and can be run
```

#### Step 4.10: Tell Nextflow where to place files

To prevent nextflow from placing files where they do not go, we must add variables (that nextflow will look for) and the paths. For example, we tell nextflow that NXF_SINGULARITY_CACHEDIR (where to store singularity images) is `/data/cephfs-1/work/groups/coscia/Singularity_Cache`, therefore it will place those files there.

First open your `.bashrc` file wiht:

```console
nano ~/.bashrc
```

This will open a terminal text editor, add the following 5 lines of code.
Windows users you will not be able to copy paste, please be careful with typos.

```console
export SINGULARITY_CACHEDIR=/data/cephfs-1/work/groups/coscia/Singularity_Cache
export NXF_SINGULARITY_CACHEDIR=/data/cephfs-1/work/groups/coscia/Singularity_Cache
export NXF_WORK=/data/cephfs-1/home/users/$USERNAME/scratch/
export NXF_TEMP=/data/cephfs-1/home/users/$USERNAME/scratch/
export NXF_HOME=/data/cephfs-1/home/users/$USERNAME/work/.nextflow
```

Restart the `.bashrc` file to trigger those changes

```console
source ~/.bashrc
```

### Step 6: Run Nextflow

In case something went wrong before, remember you must have, and do any of these if needed:

- interactive session activated (no login nodes) : `srun --mem=5G --pty bash -i`
- mcmicro conda environment activated : `conda activate mcmicro`
- have the `.bashrc` variables set and activated

#### Step 6.1: Demo pipeline

Run nextflow hello (demo) pipeline

```console
nextflow run nextflow-io/hello
```

This is how it should look like

```console
 N E X T F L O W   ~  version 25.04.7

NOTE: Your local project version looks outdated - a different revision is available in the remote repository [2ce0b0e294]
Launching `https://github.com/nextflow-io/hello` [naughty_kimura] DSL2 - revision: afff16a9b4 [master]

executor >  local (4)
[31/222763] sayHello (4) [100%] 4 of 4 ✔
Bonjour world!

Hello world!

Ciao world!

Hola world!
```

</details>

#### Step 6.2: Download demo data (Exemplar-001)

Create a folder for demo data:

```console
mkdir -p ~/work/test1/
```

Download data using nextflow pipeline:

```bash
nextflow run labsyspharm/mcmicro/exemplar.nf --name exemplar-001 --path ~/work/test1/
```

Output should look like this:

```console
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

#### Step 6.3: Run MCMICRO with demo data

```console
nextflow run labsyspharm/mcmicro --in ~/work/test1/exemplar-001 -profile singularity
```

Output should look like this:

```console
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

We can check the outputs of the pipeline:

```console
tree ~/work/test1/exemplar-001
```

```python
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

## Running MCMICRO with a script for distributed computing

MCMICRO works best when we distribute all the tasks to the HPC.

You will need:

- raw image files
- param.yml file
- script.sh file

## How to prepare your images for MCMICRO

Please familiarize yourself with [MCMICRO's Input/Output expectactions](https://mcmicro.org/io.html)

### Project organization

Minimum requirements are the following:

```bash
myproject/
├── markers.csv
├── params.yml # actually not needed, but strongly recommended
└── raw/
```

Important: if a run of mcmicro creates new directories, and you would like not to start from the first step, use the `--resume` option in your `nextflow run` command. Nextflow will check what has already been done, and carry on from the last step.

### markers.csv

- **cannot** be renamed.
- column names must be: `cycle_number`, `channel_number`, and `marker_name`
- optional column names are: `background`, `exposure`, and `remove` (for backsub).
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

### script.sh

- The `.sh` suffix denotes a script file.
- The script will tell the HPC how to process your images.
- I suggest you create a script in your preferred IDE, like VSCode, and then import it to the HPC when it is time to run your script.  

Demo script:

```bash
#!/bin/bash
#SBATCH --job-name=test_job         # Job name
#SBATCH --time=16:00:00             # Time limit hrs:min:sec
#SBATCH --mem=10G                   # Memory for orchestrating node
#SBATCH --cpus-per-task=2           # Number of CPU cores for orchestrating node

eval "$(conda shell.bash hook)"     # This exposes conda hook to node
conda activate mcmicro              # This activates environment in node
PATH_TO_DATA="/data/cephfs-1/home/users/<USERNAME>/work/test1/exemplar-001"
PATH_TO_PARAMS="/data/cephfs-1/home/users/<USERNAME>/work/test1/exemplar001_params.yml"

nextflow run labsyspharm/mcmicro --in $PATH_TO_DATA --params $PATH_TO_PARAMS -profile singularity
```

Then run your script from the terminal

```console
sbatch demo_script.sh 
```

Output will look something like this:

```console
sbatch: routed your job to partition short
Submitted batch job 18622303
```

Conceptually:  
This script will start an orchestrating job, which will manage all the other jobs that have to be created and managed.
Every daughter job will be managed from it. If the orchestrating job runs out of time, everything collapses.
There are also ways one can run a batch script that can run one script per dataset. In that way processing multiple datasets from a single script. This is excellent for reproducibility.

Looking for details?
This part can get very complex, and highly depends on your needs. So I refrain from overexplaining. Look at your HPC documentation on how to run scripts, and what are their recommended workflows. Asking your friendly bioinformatician is a great first point of help.

### Considerations, ideas, and tips

- You do not have to run everything all the time, sometimes I like to run MCMICRO just for (1) Illumination correction and (2) Stitching and registration and the move on to another software. Make use of the workflow options like `stop-at`.  
- Inside the [nextflow.config](https://github.com/labsyspharm/mcmicro/blob/master/nextflow.config) file, MCMICRO developers link the default computational requirements for each module. They have set `profiles`. For example here you can find the [WSI defaults](https://github.com/labsyspharm/mcmicro/blob/master/config/nf/wsi.config) profile optimized for Whole slide imaging, and here the `profile` for  [TMA defaults](https://github.com/labsyspharm/mcmicro/blob/master/config/nf/tma.config). If you feel you need your own computational requirements, I suggest you fork the MCMICRO github repository, modify it, and then run mcmicro like this `nextflow run josenimo/mcmicro`. Nextflow is smart and will use that github repository instead.
- For MDC peeps, feel free to use [josenimo/mcmicro](https://github.com/josenimo/mcmicro/) with `-profile singularity` in your `nextflow run` command.
- Nextflow has a nice `-resume` parameter to automatically check what already ran, and go on from there.
  
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
