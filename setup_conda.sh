#!/bin/sh
if [ -z ${conda+x} ];
then echo "ERROR!!!!! remove all conda entries from Path and conda init from .rc files";
else 
set -e
module load anaconda3/5.0
source /opt/anaconda3/5.0/etc/profile.d/conda.sh   
export DUST=/nfs/dust/cms/user/$USER/
cd $DUST
mkdir Anaconda Anaconda/envs  Anaconda/pkgs
cd Anaconda
conda config --add envs_dirs $DUST/Anaconda/envs
conda config --add pkgs_dirs $DUST/Anaconda/pkgs
conda create -n susy1lep -y
conda activate susy1lep
conda install conda -y
conda install -c conda-forge mamba -y
mamba install -c conda-forge coffea -y
mamba update -n base -c defaults conda -y
mamba install cudatoolkit=10.2 -c pytorch -y
mamba install -c conda-forge pytorch -y
mamba install -c conda-forge law -y
$DUST/Anaconda/envs/susy1lep/bin/pip install order -y
mamba install -c conda-forge scikit-learn -y
mamba install -c conda-forge black -y
 fi