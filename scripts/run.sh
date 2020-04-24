#!/bin/bash

#SBATCH --account=rpp-bengioy
#SBATCH --time=10:0:0
#SBATCH --mem=12g
#SBATCH --gres=gpu:1
#SBATCH --time=10:0:0
#SBATCH --ntasks-per-node=8

hostname
export LANG=C.UTF-8
source $HOME/.bashrc

python3 $@

# sbatch run.sh 
