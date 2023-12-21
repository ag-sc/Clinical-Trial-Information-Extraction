import csv
import os
import pickle
from argparse import ArgumentParser
from datetime import datetime
import generative_approach.evaluation as gen
import extractive_approach.evaluation as extr

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--extr', required=False, default=None)
    argparser.add_argument('--gen', required=False, default=None)
    argparser.add_argument('--extr-out',
                           required=False,
                           default=None)
    argparser.add_argument('--gen-out',
                           required=False,
                           default=None)
    arguments = argparser.parse_args()
    assert arguments.extr is not None or arguments.gen is not None
    extr_out = arguments.extr_out
    gen_out = arguments.gen_out
    if extr_out is None:
        extr_out = arguments.extr
    if gen_out is None:
        gen_out = arguments.gen

    extr_path = arguments.extr
    gen_path = arguments.gen

    if extr_path is not None:
        extr_out_path = os.path.join(extr_out, datetime.today().strftime('%Y-%m-%d') + "-extr-stats.csv")
        extr_pickle_path = os.path.join(extr_out, datetime.today().strftime('%Y-%m-%d') + "-extr-stats.pkl")

        extr_stats = []
        extr_eval: csv.DictWriter

        with open(extr_out_path, "w", newline='') as extr_eval_file:
            fieldnames = ["model",
                          "validation-precision", "validation-recall", "validation-f1",
                          "test-precision", "test-recall", "test-f1"]

            extr_eval = csv.DictWriter(extr_eval_file, fieldnames=fieldnames)
            extr_eval.writeheader()

            extr_models = sorted([os.path.join(extr_path, name)
                                  for name in os.listdir(extr_path)
                                  if os.path.isfile(os.path.join(extr_path, name)) and name.endswith(".pt")])

            extr_configs = [os.path.join(extr_path, name)
                            for name in os.listdir(extr_path)
                            if os.path.isfile(os.path.join(extr_path, name))
                            and name.startswith("config_") and name.endswith(".json")]

            extr_sims = sorted([os.path.join(extr_path, name)
                                for name in os.listdir(extr_path)
                                if os.path.isfile(os.path.join(extr_path, name))
                                and name.startswith("extr_simstats_") and name.endswith(".pkl")])

            assert len(extr_configs) == 1
            extr_config = extr_configs[0]

            for model, sim in zip(extr_models, extr_sims):
                try:
                    print(model)
                    res_valid = extr.cli_eval(task_config=extr_config, model_path=model, sim_path=sim,
                                              use_validation=True)
                    stats_valid = res_valid.compute_micro_stats()

                    res_test = extr.cli_eval(task_config=extr_config, model_path=model, sim_path=sim,
                                             use_validation=False)
                    stats_test = res_test.compute_micro_stats()

                    print({
                        "model": model,
                        "validation-precision": stats_valid.precision(),
                        "validation-recall": stats_valid.recall(),
                        "validation-f1": stats_valid.f1(),
                        "test-precision": stats_test.precision(),
                        "test-recall": stats_test.recall(),
                        "test-f1": stats_test.f1()
                    }, flush=True)

                    extr_eval.writerow({
                        "model": model,
                        "validation-precision": stats_valid.precision(),
                        "validation-recall": stats_valid.recall(),
                        "validation-f1": stats_valid.f1(),
                        "test-precision": stats_test.precision(),
                        "test-recall": stats_test.recall(),
                        "test-f1": stats_test.f1()
                    })

                    extr_stats.append({
                        "model": model,
                        "res_valid": res_valid,
                        "res_test": res_test,
                    })
                except Exception as e:
                    print(e)
                    print("Failed to evaluate extr model: " + model)
                    continue

        with open(extr_pickle_path, "wb") as extr_pkl_file:
            pickle.dump(extr_stats, extr_pkl_file)

    if gen_path is not None:
        gen_out_path = os.path.join(gen_out, datetime.today().strftime('%Y-%m-%d') + "-gen-stats.csv")
        gen_pickle_path = os.path.join(gen_out, datetime.today().strftime('%Y-%m-%d') + "-gen-stats.pkl")

        gen_stats = []
        extr_eval: csv.DictWriter

        with open(gen_out_path, "w", newline='') as gen_eval_file:
            fieldnames = ["model",
                          "validation-precision", "validation-recall", "validation-f1",
                          "test-precision", "test-recall", "test-f1"]

            gen_eval = csv.DictWriter(gen_eval_file, fieldnames=fieldnames)
            gen_eval.writeheader()

            gen_models = sorted([os.path.join(gen_path, name)
                                 for name in os.listdir(gen_path)
                                 if os.path.isdir(os.path.join(gen_path, name))])

            gen_configs = [os.path.join(gen_path, name)
                           for name in os.listdir(gen_path)
                           if os.path.isfile(os.path.join(gen_path, name))
                           and name.startswith("config_") and name.endswith(".json")]

            gen_tokenizers = sorted([os.path.join(gen_path, name)
                                     for name in os.listdir(gen_path)
                                     if os.path.isfile(os.path.join(gen_path, name))
                                     and name.startswith("tokenizer_gen_") and name.endswith(".json")])

            # TODO: Tokenizers!

            assert len(gen_configs) == 1
            gen_config = gen_configs[0]

            for model, tokenizer_path in zip(gen_models, gen_tokenizers):
                try:
                    print(model)
                    res_valid = gen.cli_eval(task_config=gen_config, model_path=model, tokenizer_path=tokenizer_path, use_validation=True)
                    stats_valid = res_valid.compute_micro_stats()

                    res_test = gen.cli_eval(task_config=gen_config, model_path=model, tokenizer_path=tokenizer_path, use_validation=False)
                    stats_test = res_test.compute_micro_stats()

                    print({
                        "model": model,
                        "validation-precision": stats_valid.precision(),
                        "validation-recall": stats_valid.recall(),
                        "validation-f1": stats_valid.f1(),
                        "test-precision": stats_test.precision(),
                        "test-recall": stats_test.recall(),
                        "test-f1": stats_test.f1()
                    }, flush=True)

                    gen_eval.writerow({
                        "model": model,
                        "validation-precision": stats_valid.precision(),
                        "validation-recall": stats_valid.recall(),
                        "validation-f1": stats_valid.f1(),
                        "test-precision": stats_test.precision(),
                        "test-recall": stats_test.recall(),
                        "test-f1": stats_test.f1()
                    })

                    gen_stats.append({
                        "model": model,
                        "res_valid": res_valid,
                        "res_test": res_test,
                    })
                except Exception as e:
                    print(e)
                    print("Failed to evaluate model: " + model)
                    continue

        with open(gen_pickle_path, "wb") as gen_pkl_file:
            pickle.dump(gen_stats, gen_pkl_file)
