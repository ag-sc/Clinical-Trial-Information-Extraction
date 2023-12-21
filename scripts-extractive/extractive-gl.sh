#!/bin/bash
cd "$(dirname "$0")"
cd ..
source venv/bin/activate
export PYTHONPATH=./src/
configdate=$(date '+%Y-%m-%d-%H:%M:%S')
modelname=$(echo "$1" | sed -E 's/.*?\/(.*)/\1/g')
echo 'allenai/led-base-16384' 'google/flan-t5-base' 'allenai/longformer-base-4096'
python -u src/extractive_approach/create_task_config_file.py --host local --disease-prefix gl --min-slot-freq 10 --filename config_extr_gl_"$modelname"_"$configdate".json --batch-size 1 --epochs 50 --model $1
python -u src/extractive_approach/training.py config_extr_gl_"$modelname"_"$configdate".json
