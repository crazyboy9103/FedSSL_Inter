#!/bin/bash

#SBATCH --job-name=van_byol
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=0-12:00:00
#SBATCH --mem=32000MB
#SBATCH --partition=3090,titan
#SBATCH --cpus-per-task=32
#SBATCH --output=/home/kwangyeongill/FedSSL_recent/scripts/slurm/van_byol_%j.out

eval "$(conda shell.bash hook)"
conda activate FedSSL

cd /home/kwangyeongill/FedSSL_recent/Exps/ && python vanilla_byol.py