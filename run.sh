#!/bin/bash

# Script to reproduce results

envs=(
	"hopper-medium-v2"
	)

for ((i=0;i<5;i+=1))
do
    python main.py \
    --env hopper-medium-v2 \
    --seed $i \
    --alpha 2.5 \
    --vae_weight 1.0 \
    --device cuda:3 \
    --mask 0.4 \
    --discount 0.995
done

