import csv
import json
import os
import pickle
import re
from argparse import ArgumentParser
from collections import defaultdict
from pprint import pprint
from typing import Optional

import pandas as pd
import torch

import extractive_approach.data_handling as extr_dh
import generative_approach.data_handling as gen_dh
import extractive_approach.evaluation as extr_eval
import generative_approach.evaluation as gen_eval
from extractive_approach.modules import IntraTemplateCompModule, ITCLongformer, ITCLED, ITCFlanT5
from extractive_approach.utils import ITCTokenizer
from generative_approach.models.flan_t5 import FlanT5
from generative_approach.template_coding import LinearTemplateEncoder
from generative_approach.utils import import_task_config, get_model_class, TemplateGenerationTokenizer
from template_lib import evaluation_utils
from template_lib.data_classes.Template import Template
from template_lib.data_classes.TemplateCollection import TemplateCollection
from template_lib.data_handling import import_train_test_valid_data_from_task_config
from template_lib.max_cardinalities_estimator import gen_max_slot_cardinalities


def single_result_extr(task_config: str, model_path: str, sim_path: Optional[str] = None):
    torch.cuda.empty_cache()
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # load task config
    with open(task_config) as fp:
        task_config_dict = json.load(fp)

    # import trained pytorch model
    model: IntraTemplateCompModule
    itc_tokenizer: ITCTokenizer
    if 'allenai/longformer' in task_config_dict['model_name']:
        model = ITCLongformer(
            num_entity_classes=len(task_config_dict['slot_indices']),
            model_name=task_config_dict['model_name']
        )
    elif 'allenai/led' in task_config_dict['model_name']:
        model = ITCLED(
            num_entity_classes=len(task_config_dict['slot_indices']),
            model_name=task_config_dict['model_name']
        )
    elif 'google/t5' in task_config_dict['model_name'] or 'google/flan-t5' in task_config_dict['model_name']:
        model = ITCFlanT5(num_entity_classes=len(task_config_dict['slot_indices']),
                          model_name=task_config_dict['model_name'])
    else:
        raise RuntimeError("Unknown model name!")

    model.load_state_dict(torch.load(model_path, map_location=device))

    itc_tokenizer = model.tokenizer

    # import dataset
    train_set, test_set, validation_data = import_train_test_valid_data_from_task_config(task_config_dict,
                                                                                         itc_tokenizer)
    labels = set()
    for doc, templcl in test_set:
        labels.update([e.get_label() for e in templcl.get_all_entities()])
    print(labels)
    # exit(0)

    datat_elements = extr_dh.extract_data_elements_from_dataset(
        dataset=test_set,
        tokenizer=itc_tokenizer,
        task_config_dict=task_config_dict
    )

    max_slots_cardinalities = gen_max_slot_cardinalities(train_set)

    same_inst_vals, diff_inst_vals = None, None

    if sim_path is not None:
        with open(sim_path, "rb") as fp:
            same_inst_vals, diff_inst_vals = pickle.load(fp)

    pmid = itc_tokenizer.tokenize("27740719")
    studydata = [de for de in datat_elements if pmid in sum([[e.get_tokens() for e in pub.get_slot_fillers("hasPMID")] for pub in de.template_collection.grouped_by_type["Publication"]], [])]
    assert len(studydata) == 1

    # evaluation
    result, _ = extr_eval.single_run(
        data_element=studydata[0],
        itc_module=model,
        task_config_dict=task_config_dict,
        max_slots_cardinalities=max_slots_cardinalities,
        device=device,
        same_inst_vals=same_inst_vals,
        diff_inst_vals=diff_inst_vals,
        instance_counter=0
    )

    return studydata[0].template_collection, result, itc_tokenizer


def single_result_gen(task_config: str, model_path: str, tokenizer_path: Optional[str] = None):
    torch.cuda.empty_cache()
    task_config_dict = import_task_config(task_config)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    # device = "cpu" #torch.device("cpu")

    # load pretrained led model and create instantiate tokenizers #########################
    model_name = task_config_dict['model_name']
    model_class = get_model_class(model_name)
    model = model_class(model_path=model_path, model_name=model_name, device=device_str)

    # led_model = LEDForConditionalGeneration.from_pretrained(model_name).to(device)
    # led_tokenizer = LEDTokenizer.from_pretrained(model_name)
    # led_config = LEDConfig.from_pretrained(model_name)

    # create template generation tokenizer ################################################
    temp_generation_tokenizer: TemplateGenerationTokenizer
    temp_generation_tokenizer = TemplateGenerationTokenizer(
        tokenizer=model.tokenizer,
        json_filename=tokenizer_path,
        template_names=task_config_dict['slots_ordering_dict'].keys(),
        slot_names=task_config_dict['used_slots'],
        start_of_sentence_token=model.tokenizer.bos_token,
        end_of_sentence_token=model.tokenizer.eos_token,
        filler_sep_token='[FILLER_SEP]',
        control_tokens_start_id=model.tokenizer.vocab_size
    )
    # temp_generation_tokenizer.to_json(sys.argv[2])

    # add embeddings for control tokens
    model.model.resize_token_embeddings(
        model.tokenizer.vocab_size + len(temp_generation_tokenizer.control_token_ids)
        - (1 if model.tokenizer.bos_token is not None else 0)
        - (1 if model.tokenizer.eos_token is not None else 0)
        - (1 if model.tokenizer.pad_token is not None else 0))
    # print(led_model.led.shared.num_embeddings)

    # load training data ###############################################
    training_dataset, test_dataset, validation_dataset = import_train_test_valid_data_from_task_config(task_config_dict,
                                                                                                       temp_generation_tokenizer)

    max_slots_cardinalities = gen_max_slot_cardinalities(training_dataset)

    pmid = temp_generation_tokenizer.tokenize("27740719")
    studydata = [(doc, tc) for doc, tc in test_dataset if pmid in sum([[e.get_tokens() for e in pub.get_slot_fillers("hasPMID")] for pub in tc.grouped_by_type["Publication"]], [])]
    assert len(studydata) == 1


    #doc = [doc for doc, tc in test_dataset][0] #PMID = 27740719 temp_generation_tokenizer.tokenize(["27740719"])
    #tc = [tc for doc, tc in test_dataset][0]
    doc = studydata[0][0]
    tc = studydata[0][1]
    predicted_template_collection = gen_eval.single_run(
        document=doc,
        model=model,
        temp_generation_tokenizer=temp_generation_tokenizer,
        task_config_dict=task_config_dict,
        max_slots_cardinalities=max_slots_cardinalities,
        device=device
    )

    return tc, predicted_template_collection, temp_generation_tokenizer


def escape(s: str):
    return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%").replace(">", "\\textgreater{}").replace("<",
                                                                                                                 "\\textless{}")


def case_study_creator(gold_collection: TemplateCollection,
                       extr_collection: TemplateCollection,
                       gen_collection: TemplateCollection,
                       extr_tokenizer: ITCTokenizer,
                       gen_tokenizer: TemplateGenerationTokenizer,
                       label: str = None,
                       caption: str = None):
    case_study = []  # pd.DataFrame(columns=["Slot name", "Gold Standard", "Extractive Prediction", "Generative Prediction"])
    gold_grouped = gold_collection.grouped_by_type
    extr_grouped = extr_collection.grouped_by_type
    gen_grouped = gen_collection.grouped_by_type
    template_types = set(gold_grouped.keys()).union(set(extr_grouped.keys())).union(set(gen_grouped.keys()))
    assert "Publication" in template_types
    template_types.remove("Publication")
    template_types = ["Publication"] + list(template_types)

    for template_type in template_types:
        slot_names = list(sorted(set().union(
            *[[sn for sn in t.get_filled_slot_names() if len(t.get_slot_fillers(sn)) > 0] for t in
              gold_grouped[template_type]]
        ).union(
            *[[sn for sn in t.get_filled_slot_names() if len(t.get_slot_fillers(sn)) > 0] for t in
              extr_grouped[template_type]]
        ).union(
            *[[sn for sn in t.get_filled_slot_names() if len(t.get_slot_fillers(sn)) > 0] for t in
              gen_grouped[template_type]]
        )))
        gold_templates = gold_grouped[template_type]
        extr_templates = extr_grouped[template_type]
        gen_templates = gen_grouped[template_type]

        # TODO: tokens back to proper string
        gold_rows = [
            escape(' | '.join([
                gen_tokenizer.tokenizer.decode(
                    gen_tokenizer.convert_tokens_to_ids(sf.get_tokens())
                )
                for sf in t.get_slot_fillers(slot_name) if not isinstance(sf, Template)
            ]).replace("<unk>", "<"))  # for some reason T5 does not tokenize < correctly...
            for t in gold_templates
            for slot_name in slot_names
        ]
        extr_rows = [
            escape(' | '.join([
                extr_tokenizer.tokenizer.decode(
                    extr_tokenizer.convert_tokens_to_ids(sf.get_tokens())
                )
                for sf in t.get_slot_fillers(slot_name) if not isinstance(sf, Template)
            ]).replace("<unk>", "<"))
            for t in extr_templates
            for slot_name in slot_names
        ]
        gen_rows = [
            escape(' | '.join([
                gen_tokenizer.tokenizer.decode(
                    gen_tokenizer.convert_tokens_to_ids(sf.get_tokens())
                )
                for sf in t.get_slot_fillers(slot_name) if not isinstance(sf, Template)
            ]).replace("<unk>", "<"))
            for t in gen_templates
            for slot_name in slot_names
        ]

        max_rows = max(len(gold_rows), len(extr_rows), len(gen_rows))
        gold_rows += [""] * (max_rows - len(gold_rows))
        gold_rows = [""] + gold_rows
        extr_rows += [""] * (max_rows - len(extr_rows))
        extr_rows = [""] + extr_rows
        gen_rows += [""] * (max_rows - len(gen_rows))
        gen_rows = [""] + gen_rows

        labels = [
                     sn.replace("has", "") if i != 0 else "\\midrule " + sn.replace("has", "")
                     for i, sn in enumerate(slot_names)
                 ] * (max_rows // len(slot_names))
        labels = [("\\midrule " if len(case_study) > 0 else "") + "\\textbf{Template " + template_type + "}"] + labels
        # labels[1] = "\\midrule " + labels[1]

        data = {
            "\\textbf{Slot name}": labels,
            "\\textbf{Gold Standard}": gold_rows,
            "\\textbf{Extractive Prediction}": extr_rows,
            "\\textbf{Generative Prediction}": gen_rows
        }
        case_study.append(pd.DataFrame(data))

        pass
    pd_case_study = pd.concat(case_study, ignore_index=True)
    latex = pd_case_study.to_latex(index=False,
                                   multicolumn_format="c",
                                   longtable=True,
                                   column_format='r | p{4cm} | p{4cm} | p{4cm}',
                                   label=label,
                                   caption=caption)
    latex = re.sub(r"\\textbf{Template (.*?)}\s+&\s+&\s+&\s+\\\\", r"\\multicolumn{4}{c}{\\textbf{Template \1}} \\\\", latex)
    print("### Case Study")
    print(latex)
    return latex


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--results', required=True)
    argparser.add_argument('--casestudy', action='store_true')
    #argparser.add_argument('--nosloteval', action='store_false')
    #argparser.add_argument('--noinstancecount', action='store_false')
    arguments = argparser.parse_args()

    results_path = arguments.results
    run_case_study = arguments.casestudy
    run_best_eval = True #arguments.nosloteval
    #run_instance_count = arguments.noinstancecount

    names = {
        "gen": ["t5", "led"],
        "extr": ["t5", "led", "longformer"]
    }

    name_res = {
        "gen": re.compile(r"^(?P<path>.*)/gen_model_(?P<disease>.*)_(?P<model>.*)_(?P<date>.*)_(?P<time>.*)_(?P<batchsize>.*)_(?P<learningrate>.*)_(?P<lambda>.*)$"),
        "extr": re.compile(r"^(?P<path>.*)/extr_model_(?P<disease>.*)_(?P<model>.*)_(?P<date>.*)_(?P<time>.*)_(?P<batchsize>.*)_(?P<learningrate>.*)_(?P<lambda>.*).pt$")
    }

    diseases = ["dm2", "gl"]

    case_study_path = os.path.join(results_path, "case_study_results.pkl")
    besteval_path = os.path.join(results_path, "best_res_sloteval.pkl")
    bestparams_path = os.path.join(results_path, "best_params.pkl")
    #instancecount_path = os.path.join(results_path, "best_res_instancecount.pkl")

    case_study_results = {k: defaultdict(list) for k in diseases}

    if run_case_study and os.path.isfile(case_study_path):
        with open(case_study_path, "rb") as res_pkl_file:
            case_study_results = pickle.load(res_pkl_file)
            best_results = {k: dict() for k in diseases}
            for k in diseases:
                for approach in case_study_results[k]:
                    case_study_results[k][approach].sort(key=lambda x: x[0], reverse=True)
                    best_results[k][approach] = case_study_results[k][approach][0]

            for k in diseases:
                # print("Case study " + k)
                if len(best_results[k].values()) == 0:
                    continue

                assert set([t.get_id() for t in best_results[k]["extr"][1]]) == set([t.get_id() for t in best_results[k]["gen"][1]])
                case_study_creator(gold_collection=best_results[k]["extr"][1],
                                   extr_collection=best_results[k]["extr"][2],
                                   gen_collection=best_results[k]["gen"][2],
                                   extr_tokenizer=best_results[k]["extr"][7],
                                   gen_tokenizer=best_results[k]["gen"][7],
                                   label=f"tab:case_study_{k}",
                                   caption=f"Case study for disease {k}. Multiple entries for same slot in same template instance separated by |.")
        #exit(0)
        run_case_study = False

    best_res_evals = {
        x: {
            y: {z: None for z in names[y]}
            for y in names.keys()
        }
        for x in diseases
    }

    if os.path.isfile(besteval_path):
        with open(besteval_path, "rb") as sres_pkl_file:
            best_res_evals = pickle.load(sres_pkl_file)
            total_best = []
            avg_slot_f1_best = []
            count_best = []

            #total_best_labels = []
            for disease in diseases:
                for approach in best_res_evals[disease]:
                    modelres = [(v.get_dataframe_f1(col_name=f"{disease} {approach} $F_1$"), v) for v in best_res_evals[disease][approach].values()]
                    modelres.sort(key=lambda x: x[0].iloc[-1].tolist()[1],
                                  reverse=True)  # [best_res_sloteval[k][approach][model] for model in best_res_sloteval[k][approach]]
                    total_best.append(modelres[0][0])
                    #total_best_labels.append((disease, approach))

                    print("### Count eval " + disease + " " + approach)
                    print(modelres[0][1].get_count_dataframe().round(2).to_latex(index=False, multicolumn_format="c", longtable=True, na_rep="-", float_format="%.2f"))

                    count_best.append(modelres[0][1].get_diff_count_dataframe(col_name=f"{disease} {approach} Mean Abs Diff"))

                    tns, f1s = evaluation_utils.compute_mean_f1_over_templates(modelres[0][1])

                    avg_slot_f1 = pd.DataFrame.from_dict({"Template Name": tns, f"{disease} {approach} $F_1$": f1s})
                    avg_slot_f1_best.append(avg_slot_f1)

                    #print(evaluation_utils.compute_mean_f1_over_templates(modelres[0][1]))

            joined = total_best[0]
            for df in total_best[1:]:
                joined = pd.merge(joined, df, on="Slot Name", how="outer")
            joined["Mean $F_1$"] = joined.mean(numeric_only=True, axis=1)
            joined.sort_values(by="Mean $F_1$", inplace=True, ascending=False)
            print("### Joined slot eval")
            print(joined.round(2).to_latex(index=False, multicolumn_format="c", longtable=True, na_rep="-", float_format="%.2f"))

            joined2 = avg_slot_f1_best[0]
            for df in avg_slot_f1_best[1:]:
                joined2 = pd.merge(joined2, df, on="Template Name", how="outer")

            print("### Joined avg slot f1 eval")
            print(joined2.round(2).to_latex(index=False, multicolumn_format="c", longtable=True, na_rep="-", float_format="%.2f"))

            joined3 = count_best[0]
            for df in count_best[1:]:
                joined3 = pd.merge(joined3, df, on="Template Name", how="outer")

            print("### Joined count eval")
            print(joined3.round(2).to_latex(index=False, multicolumn_format="c", longtable=True, na_rep="-", float_format="%.2f"))

            run_best_eval = False

    best_hyperparams = []


    with pd.ExcelWriter(os.path.join(results_path, 'summary.xlsx'), engine='xlsxwriter') as writer:
        workbook = writer.book

        for approach in names.keys():
            summarydf = pd.DataFrame(columns=pd.MultiIndex.from_arrays([sum([[d, d] for d in diseases], []),
                                                                        ["model", "$F_1$"] * len(diseases)]))
            # ["model", "f1"]*len(diseases))

            for model in names[approach]:
                row = []
                for disease in diseases:
                    print(f"{approach}-{disease}-{model}")
                    csv_files = [os.path.join(results_path, approach, f"{disease}-{model}", name)
                                 for name in os.listdir(os.path.join(results_path, approach, f"{disease}-{model}"))
                                 if os.path.isfile(os.path.join(results_path, approach, f"{disease}-{model}", name))
                                 and name.endswith(".csv")]

                    config_files = [os.path.join(results_path, approach, f"{disease}-{model}", name)
                                    for name in os.listdir(os.path.join(results_path, approach, f"{disease}-{model}"))
                                    if os.path.isfile(os.path.join(results_path, approach, f"{disease}-{model}", name))
                                    and name.startswith("config_") and name.endswith(".json")]

                    stats_files = [os.path.join(results_path, approach, f"{disease}-{model}", name)
                                   for name in os.listdir(os.path.join(results_path, approach, f"{disease}-{model}"))
                                   if os.path.isfile(os.path.join(results_path, approach, f"{disease}-{model}", name))
                                   and name.endswith("-stats.pkl")]

                    # print(csv_files)
                    assert len(csv_files) >= 1
                    assert len(config_files) == 1
                    assert len(stats_files) >= 1
                    csv_files.sort(reverse=True)# newest files to top
                    stats_files.sort(reverse=True)# newest files to top
                    csv_file_path = csv_files[0]
                    config_file_path = config_files[0]
                    stats_file_path = stats_files[0]

                    with open(stats_file_path, "rb") as fp:
                        stats_dicts = pickle.load(fp)

                    worksheet = workbook.add_worksheet(f"{approach}-{disease}-{model}")
                    writer.sheets[f"{approach}-{disease}-{model}"] = worksheet

                    df = pd.read_csv(csv_file_path)
                    df.sort_values(by=["validation-f1"], inplace=True, ascending=False)
                    df.reset_index(drop=True, inplace=True)

                    model_path = df.iloc[0]["model"]
                    if run_case_study:
                        if approach == "gen":
                            tokenizer_path = model_path.replace("gen_model", "tokenizer_gen_model") + ".json"
                            assert os.path.isfile(tokenizer_path)
                            try:
                                res_gold, res_pred, tokenizer = single_result_gen(config_file_path, model_path, tokenizer_path)
                                res_gold = res_gold.cpu()
                                res_pred = res_pred.cpu()
                                case_study_results[disease][approach].append((df.iloc[0]["test-f1"],
                                                                              res_gold,
                                                                              res_pred,
                                                                              model,
                                                                              config_file_path,
                                                                              model_path,
                                                                              tokenizer_path,
                                                                              tokenizer))
                            except AssertionError:
                                pass
                            pass
                        elif approach == "extr":
                            sim_path = model_path.replace("extr_model", "extr_simstats").replace(".pt", ".pkl")
                            assert os.path.isfile(sim_path)
                            try:
                                res_gold, res_pred, tokenizer = single_result_extr(config_file_path, model_path, sim_path)
                                res_gold = res_gold.cpu()
                                res_pred = res_pred.cpu()
                                case_study_results[disease][approach].append((df.iloc[0]["test-f1"],
                                                                              res_gold,
                                                                              res_pred,
                                                                              model,
                                                                              config_file_path,
                                                                              model_path,
                                                                              sim_path,
                                                                              tokenizer))
                            except AssertionError:
                                pass
                            pass

                    match = re.search(name_res[approach], df.iloc[0]["model"])

                    # "gen": re.compile(r"^(?P<path>.*)/gen_model_(?P<disease>.*)_(?P<model>.*)_(?P<date>.*)_(?P<time>.*)_(?P<batchsize>.*)_(?P<learningrate>.*)_(?P<lambda>.*)$"),
                    # "extr": re.compile(r"^(?P<path>.*)/extr_model_(?P<disease>.*)_(?P<model>.*)_(?P<date>.*)_(?P<time>.*)_(?P<batchsize>.*)_(?P<learningrate>.*)_(?P<lambda>.*).pt$")

                    best_hyperparams.append({
                        "modelpath": df.iloc[0]["model"],
                        "configpath": config_file_path,
                        "disease": disease,
                        "approach": approach,
                        "model": model,
                        "batchsize": match.group("batchsize"),
                        "learningrate": match.group("learningrate"),
                        "lambda": match.group("lambda"),
                        "date": match.group("date"),
                        "time": match.group("time"),
                    })

                    # row.append(df.iloc[0]["model"])
                    row.append(match.group("model"))
                    row.append(df.iloc[0]["test-f1"])
                    df.to_excel(writer, sheet_name=f"{approach}-{disease}-{model}", startrow=0, startcol=0)
                    print(f"### Eval results {approach}-{disease}-{model}")
                    print(df.round(2).to_latex(index=False, multicolumn_format="c", float_format="%.2f"))  # , column_format="c"

                    best_stats = [i for i in stats_dicts if i["model"] == df.iloc[0]["model"]][0]

                    #if run_instance_count:
                    #    best_res_instancecount[disease][approach][model] = best_stats["res_test"].get_dataframe_f1(col_name=f"{disease} {approach} $F_1$")

                    #if run_slot_eval:
                    #best_stats["res_test"].print_out()
                    #print(best_stats["res_test"].get_dataframe().to_latex(index=False, multicolumn_format="c", longtable=True))
                    best_res_evals[disease][approach][model] = best_stats["res_test"]#.get_dataframe_f1(col_name=f"{disease} {approach} $F_1$")

                    for column in df:
                        column_length = max(df[column].astype(str).map(len).max(), len(column))
                        col_idx = df.columns.get_loc(column)
                        writer.sheets[f"{approach}-{disease}-{model}"].set_column(col_idx + 1, col_idx + 1,
                                                                                  column_length)

                # with open(csv_file_path) as csv_file:
                #     csv_dict = csv.DictReader(csv_file, delimiter=',')
                #     for row in csv_dict:
                #         print(row)

                summarydf.loc[len(summarydf)] = row
                print("### Eval summary")
                print(summarydf.round(2).to_latex(index=False, multicolumn_format="c", float_format="%.2f"))  # , column_format="c"
            sworksheet = workbook.add_worksheet(f"summary-{approach}")
            writer.sheets[f"summary-{approach}"] = sworksheet
            summarydf.to_excel(writer, sheet_name=f"summary-{approach}", startrow=0, startcol=0)

            for column in summarydf:
                column_length = max(summarydf[column].astype(str).map(len).max(), len(column))
                col_idx = summarydf.columns.get_loc(column)
                writer.sheets[f"summary-{approach}"].set_column(col_idx + 1, col_idx + 1, column_length)

    if run_case_study:
        print("### Case Study results raw")
        print(case_study_results)
        with open(case_study_path, "wb") as res_pkl_file:
            pickle.dump(case_study_results, res_pkl_file)

    if run_best_eval:
        print("### Best results raw")
        print(best_res_evals)
        with open(besteval_path, "wb") as sres_pkl_file:
            pickle.dump(best_res_evals, sres_pkl_file)

    print("### Best params raw")
    for hp in best_hyperparams:
        pprint(hp)
    #print(best_hyperparams)
    with open(bestparams_path, "wb") as pres_pkl_file:
        pickle.dump(best_hyperparams, pres_pkl_file)
