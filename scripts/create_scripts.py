import os
exps = ["clr", "sia", "pl", "fm", "by", "fem", "fr", "fb"]
full_exps = ["SimCLR", "SimSiam", "PseudoLabel", "FixMatch", "BYOL", "FedMatch", "FedRGD", "FedBYOL"]
aggs = ["avg", "pro", "fsl"]
full_aggs = ["FedAvg", "FedProx", "FedSSL"]
iids = ["i", "n"]
bool_iids = ["True", "False"]
full_iids = ["iid", "noniid"]

for exp, full_exp in zip(exps, full_exps):
  for iid, bool_iid, full_iid in zip(iids, bool_iids, full_iids):
    for agg, full_agg in zip(aggs, full_aggs):
      os.makedirs(f'./{full_exp}/{full_iid}/', exist_ok=True)
      with open(f"./{full_exp}/{full_iid}/{full_exp}_{full_agg}_{full_iid}.sh", "w") as f:
        a=f'''#!/bin/bash

#SBATCH --job-name={exp}_{agg}_{iid}
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=0-12:00:00
#SBATCH --mem=64000MB
#SBATCH --partition=3090,titan
#SBATCH --cpus-per-task=64
#SBATCH --output=/home/kwangyeongill/FedSSL_recent/scripts/slurm/{exp}_{agg}_{iid}_%j.out

eval "$(conda shell.bash hook)"
conda activate FedSSL

exp="{full_exp}"
dist="{full_iid}"
iid="{bool_iid}"
gn="False"
agg="{full_agg}"
wandb_tag="$exp"_"$agg"_"$dist"
cd /home/kwangyeongill/FedSSL_recent/ && python main.py --parallel True --gn $gn --exp $exp --iid $iid --agg $agg --wandb_tag $wandb_tag
'''

        f.writelines(a)