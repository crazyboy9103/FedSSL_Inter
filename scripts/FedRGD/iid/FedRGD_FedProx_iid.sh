#!/bin/bash

#SBATCH --job-name=fr_pro_i
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=0-12:00:00
#SBATCH --mem=64000MB
#SBATCH --partition=3090,titan
#SBATCH --cpus-per-task=64
#SBATCH --output=/home/kwangyeongill/FedSSL_recent/scripts/slurm/fr_pro_i_%j.out

eval "$(conda shell.bash hook)"
conda activate FedSSL

exp="FedRGD"
dist="iid"
iid="True"
gn="False"
agg="FedProx"
wandb_tag="$exp"_"$agg"_"$dist"
cd /home/kwangyeongill/FedSSL_recent/ && python main.py --parallel True --gn $gn --exp $exp --iid $iid --agg $agg --wandb_tag $wandb_tag
