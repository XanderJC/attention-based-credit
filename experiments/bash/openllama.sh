#!/bin/bash

SCRIPT="/home/ajc340/rlhf-abc/experiments/scripts/rlhf_openllama.py" 

for i in {1..5}; do
    for method in rlhf abc uniform; do
        echo "Run $i for method $method"
        python $SCRIPT --method $method --max_epochs 200 --beta 0.8 --l_rate 1.41e-5 --min_generation 8 --max_generation 256 --repetition_penalty 1.0 --project_name openllama_seeded --batch_size 16 --seed $i
    done
done