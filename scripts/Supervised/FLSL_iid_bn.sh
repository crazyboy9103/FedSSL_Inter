#!/bin/bash

#SBATCH --job-name=avg_i_0.1
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH --mem=64000MB
#SBATCH --partition=3090,titan
#SBATCH --cpus-per-task=64
#SBATCH --output=/home/kwangyeongill/FedSSL_recent/scripts/slurm/avg_iid_0.1_%j.out

eval "$(conda shell.bash hook)"
conda activate FedSSL

exp="FLSL"
dist="iid"
iid="True"
gn="False"
agg="FedAvg"
wandb_tag="$exp"_"$agg"_"$dist"
cd /home/kwangyeongill/FedSSL_recent/ && python main.py \
                                        --parallel True \
                                        --gn $gn \
                                        --exp $exp \
                                        --iid $iid \
                                        --agg $agg \
                                        --wandb_tag $wandb_tag