eval "$(conda shell.bash hook)"
conda activate FedSSL
exp=$1
dist=$2
iid=$3
gn=$4
agg=$5
wandb_tag="$exp"_"$agg"_"$dist"
cd /home/shared/FedSSL_Inter/ && python main.py --parallel True --gn $gn --exp $exp --iid $iid --agg $agg --wandb_tag $wandb_tag
