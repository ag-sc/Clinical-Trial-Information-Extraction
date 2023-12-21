#!/bin/bash
cd "$(dirname "$0")"
cd ..
source venv/bin/activate
export PYTHONPATH=./src/
configdate=$(date '+%Y-%m-%d-%H:%M:%S')
modelname=$(echo "$1" | sed -E 's/.*?\/(.*)/\1/g')
echo 'allenai/led-base-16384' 'google/flan-t5-base'
python src/generative_approach/create_task_config_file.py --host local --disease-prefix dm2 --min-slot-freq 10 --filename config_gen_dm2_"$modelname"_"$configdate".json --batch-size 1 --epochs 50 --model $1 #'allenai/led-base-16384' 'google/flan-t5-base'
python src/generative_approach/training.py config_gen_dm2_"$modelname"_"$configdate".json 
