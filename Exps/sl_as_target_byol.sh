#!/bin/bash

#SBATCH --job-name=sl_at_byol
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=0-12:00:00
#SBATCH --mem=32000MB
#SBATCH --partition=3090,titan
#SBATCH --cpus-per-task=32
#SBATCH --output=/home/kwangyeongill/FedSSL_recent/scripts/slurm/sl_at_byol_%j.out

eval "$(conda shell.bash hook)"
conda activate FedSSL

cd /home/kwangyeongill/FedSSL_recent/Exps/ && python sl_as_target_byol.py