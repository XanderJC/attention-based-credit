#!/bin/bash

SCRIPT="/home/ajc340/rlhf-abc/experiments/scripts/rlhf_imdb.py" 

for i in {1..10}; do
    # Run each method 10 times
    for method in rlhf abc uniform abcde abcde2; do
        echo "Run $i for method $method"
        python $SCRIPT --method $method --beta 0.8 --min_generation 140 --max_generation 160 --project_name generation_length_seeded --batch_size 16 --max_epochs 150 --seed $i
        python $SCRIPT --method $method --beta 0.8 --min_generation 90 --max_generation 110 --project_name generation_length_seeded --batch_size 16 --max_epochs 150 --seed $i
        python $SCRIPT --method $method --beta 0.8 --min_generation 40 --max_generation 60 --project_name generation_length_seeded --batch_size 16 --max_epochs 150 --seed $i
        python $SCRIPT --method $method --beta 0.8 --min_generation 20 --max_generation 30 --project_name generation_length_seeded --batch_size 16 --max_epochs 150 --seed $i
    done
done