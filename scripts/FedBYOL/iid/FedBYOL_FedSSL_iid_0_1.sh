#!/bin/bash

#SBATCH --job-name=fb_fsl_i
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=0-12:00:00
#SBATCH --mem=64000MB
#SBATCH --partition=3090,titan
#SBATCH --cpus-per-task=64
#SBATCH --output=/home/kwangyeongill/FedSSL_recent/scripts/slurm/fb_fsl_i_%j_0_1.out

eval "$(conda shell.bash hook)"
conda activate FedSSL

exp="FedBYOL"
dist="iid"
iid="True"
gn="False"
agg="FedSSL"
mse_ratio="0.1"
wandb_tag="$exp"_"$agg"_"$dist"_"$mse_ratio"
cd /home/kwangyeongill/FedSSL_recent/ && python main.py --parallel True --mse_ratio $mse_ratio --gn $gn --exp $exp --iid $iid --agg $agg --wandb_tag $wandb_tag
