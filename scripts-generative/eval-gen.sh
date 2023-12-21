cd "$(dirname "$0")"
cd ..
source venv/bin/activate
export PYTHONPATH=./src/
export projectbase=/path/to/clinical-trial-ie/
if [ -n "$1" ]; then
    export configdate=$1
else
    export configdate=$(date '+%Y-%m-%d')
fi
ls -d1 "$projectbase"/results_"$configdate"/gen/*/ | xargs -n 1 -P 16 nice python src/full_eval.py --gen