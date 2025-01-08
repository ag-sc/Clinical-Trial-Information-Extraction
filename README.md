# Artifact "Comparing Generative and Extractive Approaches to Information Extraction from Abstracts Describing Randomized Clinical Trials"

DOI: [http://doi.org/10.1186/s13326-024-00305-2](http://doi.org/10.1186/s13326-024-00305-2)

Zenodo: [https://doi.org/10.5281/zenodo.10419785](https://doi.org/10.5281/zenodo.10419785)

## Setup

We suggest using Python 3.10.12 or newer to run this artifact, older versions have not been tested. The following instructions should work for most mayor Linux distributions.

Start by setting up a virtual environment and installing the required packages. For this, run the following at the top level of a cloned version of this repository:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

All scripts in this artifact automatically activate the virtual environment assuming it is named and located as shown above and set the `PYTHONPATH` accordingly. If you want to execute a single Python file manually, you have to activate the virtual environment and add the `src/` subdirectory as `PYTHONPATH`:

```bash
source venv/bin/activate
export PYTHONPATH=./src/
python some/python/file.py
```

## Artifact Structure

* data/ - annotated datasets for type 2 diabetes and glaucoma RCT abstracts used in the paper
* scripts-extractive/ - scripts to execute training and evaluation of the extractive approach
    * all_runs.txt - list of commands necessary to start full hyperparameter search training run
    * extractive-dm2.sh - given a model name, executes an extractive hyperparameter optimization training run with 30 trials for type 2 diabetes dataset
    * extractive-gl.sh - given a model name, executes an extractive hyperparameter optimization training run with 30 trials for glaucoma dataset
    * extractive-best.sh - given path to a `best_params.pkl` file generated by `src/eval_summary.py`, executes 10 training runs with the best parameters found during hyperparameter optimization
    * eval-extr.sh - executes evaluation for extractive part of directory of trained models, CHANGE PATH IN FILE TO ACTUAL LOCATION OF RESULTS!
* scripts-generative/ - scripts to execute training and evaluation of the generative approach
    * same as extractive, but for the generative approach
* src/ - source code of both approaches used in the paper
    * extractive_approach - source code of the extractive approach (training file is `training.py`)
    * generative_approach - source code of the generative approach (training file is `training.py`)
    * template_lib - source code of general classes and functions to load and use the dataset
    * full_eval.py - runs evaluation for whole given training results directory
    * eval_summary.py - generates summary of evaluated training results of hyperparameter search
    * eval_summary_best.py - generates summary of evaluated training results of the 10 training runs executed separately with the best hyperparameters
    * main.py - can be used to play around with loaded datasets, contains code to list and count slot fillers of "Journal"
* requirements.txt - Python requirements of this project
* sort_results.sh - expecting training to have been executed in top directory of project, sorts models etc. into folders grouped by approach, disease and model, CHANGE PATH IN FILE TO ACTUAL LOCATION OF RESULTS!

## Replication Steps

1. Go to the top directory of this project

2. Execute all hyperparameter optimization trainings, i.e.:
```bash
scripts-extractive/extractive-dm2.sh 'allenai/longformer-base-4096' | tee train-longformer-dm2.txt
scripts-extractive/extractive-dm2.sh "allenai/led-base-16384" | tee train-led-dm2.txt
scripts-extractive/extractive-dm2.sh "google/flan-t5-base" | tee train-t5-dm2.txt
scripts-extractive/extractive-gl.sh 'allenai/longformer-base-4096' | tee train-longformer-gl.txt
scripts-extractive/extractive-gl.sh "allenai/led-base-16384" | tee train-led-gl.txt
scripts-extractive/extractive-gl.sh "google/flan-t5-base" | tee train-t5-gl.txt
scripts-generative/generative-dm2.sh 'allenai/led-base-16384' | tee train-led-dm2-gen.txt
scripts-generative/generative-dm2.sh 'google/flan-t5-base' | tee train-t5-dm2-gen.txt
scripts-generative/generative-gl.sh 'allenai/led-base-16384' | tee train-led-gl-gen.txt
scripts-generative/generative-gl.sh 'google/flan-t5-base' | tee train-t5-gl-gen.txt
```

3. Sort results into folders: `sort_results.sh` (change paths in file!)

4. Run evaluation for extractive and generative models (change paths in file!):
```bash
scripts-extractive/eval-extr.sh
scripts-generative/eval-gen.sh
```

5. Generate evaluation summary for hyperparameter optimization (first activate virtual environment and set `PYTHONPATH` as shown above!). To generate case study data, append `--casestudy`. Some tables are only printed and generated if you run the command a second time because they are first only saved to pickle files.
```bash
python src/eval_summary.py --results /path/to/results/folder/ 
```

6. Run training again for best found parameters:
```bash
scripts-extractive/extractive-best.sh /path/to/best_params.pkl
scripts-generative/generative-best.sh /path/to/best_params.pkl
```

7. Sort new results into folders: `sort_results.sh` and make sure the files are sorted into a different directory such that you can differentiate the hyperparameter optimization from the training with best parameters. Do not forget to also copy the `config_*.json` files from the original results to the directories of the new results (e.g. using `cp --parents`) as they are necessary for running the evaluation.

8. Run evaluation for new extractive and generative models (change paths in file accordingly!):
```bash
scripts-extractive/eval-extr.sh
scripts-generative/eval-gen.sh
```

9. Generate evaluation summary with mean and standard deviations for training with best parameters (first activate virtual environment and set `PYTHONPATH` as shown above!). Some tables are only printed and generated if you run the command a second time because they are first only saved to pickle files.
```bash
python src/eval_summary_best.py --results /path/to/results/folder/ 
```

## Citation

Please consider citing our work if you find the provided resources useful:

```bibtex
@article{witte-schmidt-cimiano-2024-ct-ie,
  author       = {Christian Witte and
                  David M. Schmidt and
                  Philipp Cimiano},
  title        = {Comparing generative and extractive approaches to information extraction
                  from abstracts describing randomized clinical trials},
  journal      = {J. Biomed. Semant.},
  volume       = {15},
  number       = {1},
  pages        = {3},
  year         = {2024},
  url          = {https://doi.org/10.1186/s13326-024-00305-2},
  doi          = {10.1186/S13326-024-00305-2},
  timestamp    = {Sun, 04 Aug 2024 19:50:36 +0200},
}
```
