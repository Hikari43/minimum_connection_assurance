#!/bin/bash

set -e

config=$1
exp_count=$2
pruner=$3
exp=$4
compression_list=("${@:5}") 


echo config           = $config
echo exp_count        = $exp_count
echo pruner           = $pruner
echo exp              = $exp
echo compression_list = [${compression_list[@]}]

if [ $pruner = "sf" ]; then
  pepoch=100
else
  pepoch=1
fi

for ((i=0; i<$exp_count; i++)); do
  for compression in "${compression_list[@]}"; do  
    python main.py \
    --config $config \
    --run-number $i \
    --seed $(($i * 1000)) \
    --compression $compression \
    --mask-file Results/data/singleshot/$exp/$pruner-$compression-$pepoch/run_$i/model_effective_comp.pt \
    || exit 1
    wait
  done
done



