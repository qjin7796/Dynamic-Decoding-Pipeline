#!/bin/bash -l
#SBATCH --cluster=wice
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=18                          # for 1 gpu, 18 cores have to be requested
#SBATCH --time=02:00:00                      # walltime
#SBATCH --job-name=test_keops_doc            # name of this job
#SBATCH --account=lp_action_representation   # VSC project account to charge
#SBATCH --mail-type=ALL                      # mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=qiuhan.jin@kuleuven.be   # where to send mail
#SBATCH --output=logs/test_keops_doc.out     # output file name
#SBATCH --error=logs/test_keops_doc.err      # error file name

echo "### Running $SLURM_JOB_NAME ###"

# clean out modules loaded in interactive and inherited by default
# module --force purge
# load modules
# module use /apps/leuven/icelake/2021a/modules/all
# module load 

# echo of launched commands
set -x
# set or unset options and positional parameters for bash, see https://pubs.opengroup.org/onlinepubs/9699919799.2018edition/utilities/V3_chap02.html#set
cd $SLURM_SUBMIT_DIR           # usually not needed because Slurm starts job from where it's submitted
# cd /data/leuven/347/vsc34741   # ./code/chamfer_distance/script..py, ./code/keops/

# Set $SINGULARITY_ALLOWED_DIR, launch a shell inside the container on a computing node, --nv for NVIDIA GPU
# echo "SINGULARITY_ALLOWED_DIR:"
# echo $SINGULARITY_ALLOWED_DIR
# SINGULARITY_ALLOWED_DIR=/data/leuven/347/vsc34741/containers/singularity_homes/keops-full
singularity exec \
--bind /data/leuven/347/vsc34741/code \
--bind /data/leuven/347/vsc34741/code/keops \
--bind /data/leuven/347/vsc34741/code/chamfer_distance \
--nv \
/data/leuven/347/vsc34741/containers/singularity_homes/keops-full/keops-full.sif \ 
python3 script_feature_comparison_gpu.py


###############
# KeOps manual:
# singularity exec \
# -H $WORK/containers/singularity_homes/keops-full/:/home \
# --bind ~/keops-doc.sh:/home/keops-doc.sh \
# --bind $WORK/code:/home/code \
# --bind $WORK/code/keops:/opt/keops \
# --bind $WORK/data/scikit_learn_data:/home/scikit_learn_data \
# --nv \
# $SINGULARITY_ALLOWED_DIR/keops-full.sif \
# /home/keops-doc.sh 
# keops-doc.sh is an example program to render a webpage to be executed in the container environment

# VSC manual:
# singularity exec grace.sif gracebat -data data.dat \
#                                     -batch plot.bat

# To run a python script in the built container environment, run:
# singularity exec ... python3 -m  # see python cmd: https://stackoverflow.com/questions/7610001/what-is-the-purpose-of-the-m-switch & https://docs.python.org/3/using/cmdline.html

# FATAL:   While checking container encryption: could not open image /data/leuven/347/vsc34741/containers/singularity_homes/keops-full/keops-full.sif: failed to retrieve path for /data/leuven/347/vsc34741/containers/singularity_homes/keops-full/keops-full.sif: lstat /vsc-hard-mounts/leuven-data/347/vsc34741/containers/singularity_homes/keops-full/keops-full.sif: no such file or directory
# + python3 -m /data/leuven/347/vsc34741/code/chamfer_distance/script_feature_comparison_gpu.py
# /data/leuven/347/vsc34741/miniconda3/bin/python3: Error while finding module specification for '/data/leuven/347/vsc34741/code/chamfer_distance/script_feature_comparison_gpu.py' (ModuleNotFoundError: No module named '/data/leuven/347/vsc34741/code/chamfer_distance/script_feature_comparison_gpu'). Try using '/data/leuven/347/vsc34741/code/chamfer_distance/script_feature_comparison_gpu' instead of '/data/leuven/347/vsc34741/code/chamfer_distance/script_feature_comparison_gpu.py' as the module name.
# ~                                                                                       