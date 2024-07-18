#!/bin/bash

SCRIPT="/home/ajc340/rlhf-abc/experiments/scripts/rlhf_tldr.py" 

for i in {1..5}; do
    for method in rlhf abc uniform; do
        echo "Run $i for method $method"
        python $SCRIPT --method $method --max_epochs 200 --beta 0.8 --l_rate 1.41e-5 --min_generation 8 --max_generation 48  --project_name tldr_seeded --batch_size 4 --seed $i
    done
done