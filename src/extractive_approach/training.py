import os
import pickle
import statistics
import random
import json
from collections import defaultdict
from datetime import datetime

import numpy as np
import optuna
import torch
from optuna import Trial
from optuna.storages import JournalStorage, JournalFileStorage

from data_handling import *
from extractive_approach.evaluation import evaluate_slot_filling
from extractive_approach.utils import divide_chunks
from generative_approach.max_cardinalities_estimator import gen_max_slot_cardinalities
from modules import *
from template_lib.data_handling import import_train_test_valid_data_from_task_config

task_config_dict: Any
train_data_elements: List[DataElement]
validation_data_elements: List[DataElement]
max_slots_cardinalities: Dict[str, int]


def calc_similarity_stats(pred_sims: List[torch.Tensor], target_sims: List[torch.Tensor]):
    for ps, ts in zip(pred_sims, target_sims):
        assert ps.shape == ts.shape
        same_inst_vals = []
        diff_inst_vals = []

        for pred, target in zip(ps.reshape(-1), ts.reshape(-1)):
            # print(pred, target)
            if target.item() > 0.9:
                same_inst_vals.append(pred.item())
            else:
                diff_inst_vals.append(pred.item())

        return same_inst_vals, diff_inst_vals


def train(
        data_elements: List[DataElement],
        valid_data_elements: List[DataElement],
        itc_module: IntraTemplateCompModule,
        num_epochs: int = 30,
        batch_size: int = 1,
        ld: float = 0.99,
        learning_rate: float = 1e-5,
        device=torch.device("cpu"),
        trial: Trial = None,
):
    itc_module.to(device)

    # class_counts = defaultdict(int)

    class_weights = []

    zero_to_one_ratios = []

    for j, chunk in enumerate(divide_chunks(data_elements, batch_size)):
        num_entities = len(
            set(sum([list(data_element.template_collection.get_all_entities()) for data_element in chunk], [])))
        num_zeros = num_entities * num_entities
        num_ones = 0
        for de in chunk:
            de: DataElement

            class_counts = defaultdict(int)
            vals, counts = np.unique(de.entity_start_positions, return_counts=True)
            for v, c in zip(vals, counts):
                class_counts[v] += c

            class_weights.append([
                class_counts[i] if i in class_counts.keys() else 0.0
                for i in range(itc_module.num_entity_classes)
            ])

            for template in de.template_collection:
                template: Template
                num_ones += len(template.get_assigned_entities()) * len(template.get_assigned_entities())

        num_zeros -= num_ones
        zero_to_one_ratios.append(num_zeros / num_ones)

    zero_to_one_ratio = torch.tensor(statistics.mean(zero_to_one_ratios)).to(device)

    basic_zero_to_one_ratio = torch.tensor(2.0).to(device)

    # weight_sum = sum(class_counts.values())
    # class_weights = torch.tensor([
    #     weight_sum / class_counts[i] if i in class_counts.keys() else 0.0
    #     for i in range(itc_module.num_entity_classes)
    # ], dtype=torch.float).to(device)

    class_weight = torch.tensor(class_weights).mean(dim=0)
    class_weight = class_weight.sum() / class_weight
    class_weight = class_weight.to(device)

    basic_class_weight = torch.tensor([0.5] + [1.0] * (itc_module.num_entity_classes - 1)).to(device)

    # loss functions
    loss_fc_entity_positions = torch.nn.CrossEntropyLoss(
        weight=basic_class_weight)  # torch.nn.BCEWithLogitsLoss()#torch.nn.CrossEntropyLoss()
    loss_fc_entity_compatibilities = torch.nn.BCEWithLogitsLoss(pos_weight=basic_zero_to_one_ratio)
    optimizer = torch.optim.AdamW(itc_module.parameters(), lr=learning_rate)

    lambda1 = lambda epoch: ld ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    last_loss = None
    same_inst_vals, diff_inst_vals = [], []

    # training iterations
    for epoch in range(num_epochs):
        random.shuffle(data_elements)
        losses = []

        for j, chunk in enumerate(divide_chunks(data_elements, batch_size)):
            torch.cuda.empty_cache()
            max_len = max([len(data_element.input_ids) for data_element in chunk])
            input_ids = torch.stack(
                [torch.tensor(data_element.padded_input_ids(max_len)) for data_element in chunk]
            ).to(device)
            attention_mask = torch.stack(
                [torch.tensor(data_element.padded_attention_mask(max_len)) for data_element in chunk]
            ).to(device)
            # initialize to local attention
            # attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
            # initialize to global attention to be deactivated for all tokens
            # global_attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
            # call encoder
            encoder_output = itc_module.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # global_attention_mask=global_attention_mask,
            )

            # Todo: token representation
            last_hidden_state = encoder_output.last_hidden_state.to(device)

            # entity start/end positions
            pred_start_positions = itc_module.layer_entity_start_positions(last_hidden_state)
            pred_end_positions = itc_module.layer_entity_end_positions(last_hidden_state)

            new_shape = list(pred_start_positions.shape)[1:]
            new_shape[0] *= pred_start_positions.shape[0]

            pred_start_positions = pred_start_positions.reshape(new_shape)
            pred_end_positions = pred_end_positions.reshape(new_shape)

            target_start_pos = torch.stack(
                [torch.tensor(data_element.padded_entity_start_positions(max_len)) for data_element in chunk]
            )
            target_end_pos = torch.stack(
                [torch.tensor(data_element.padded_entity_end_positions(max_len)) for data_element in chunk]
            )
            new_shape2 = list(target_start_pos.shape)[1:]
            new_shape2[0] *= target_start_pos.shape[0]
            target_start_pos = target_start_pos.reshape(new_shape2).to(device)
            target_end_pos = target_end_pos.reshape(new_shape2).to(device)

            # prediction = torch.vstack([pred_start_positions, pred_end_positions]).to(device)
            # target = torch.vstack([torch.tensor(data_element.class_tensor_start_positions(itc_module.num_entity_classes)),
            #                       torch.tensor(data_element.class_tensor_end_positions(itc_module.num_entity_classes))]).to(device)
            #
            # loss_entity_positions = loss_fc_entity_positions(prediction, target)

            # loss_entity_positions = (loss_fc_entity_positions(pred_start_positions, torch.tensor(data_element.class_tensor_start_positions(itc_module.num_entity_classes)).to(device))
            #                          + loss_fc_entity_positions(pred_end_positions, torch.tensor(data_element.class_tensor_end_positions(itc_module.num_entity_classes)).to(device)))

            loss_entity_positions = (loss_fc_entity_positions(pred_start_positions, target_start_pos)
                                     + loss_fc_entity_positions(pred_end_positions, target_end_pos))

            # loss_entity_positions = (loss_fc_entity_positions(torch.argmax(pred_start_positions, dim=2), torch.tensor(data_element.entity_start_positions).to(device))
            #                         + loss_fc_entity_positions(torch.argmax(pred_end_positions, dim=2), torch.tensor(data_element.entity_end_positions).to(device)))

            all_entities = [list(data_element.template_collection.get_all_entities()) for data_element in chunk]

            # compute entity representations
            itc_module.compute_entity_representations(
                chunk=all_entities,
                sentence_boundaries=[data_element.sentence_boundaries for data_element in chunk],
                last_hidden_state=last_hidden_state
            )

            # pred_entity_compatibilities = torch.zeros((len(all_entities), len(all_entities)), dtype=torch.float)

            chunk_ent_pairs = [
                [(ent1, ent2) for ent1 in batch for ent2 in batch]
                for batch in all_entities
            ]

            single_pred_entity_comps = itc_module.compute_entity_compatibilities(chunk_ent_pairs)

            pred_entity_comp_list = [
                single_pred_entity_comps[batch_idx].reshape(
                    (len(all_entities[batch_idx]), len(all_entities[batch_idx])))
                for batch_idx in range(len(all_entities))
            ]

            target_entity_comp_list = []

            for batch_idx, data_element in enumerate(chunk):
                curr_target_entity_comps = torch.zeros((len(all_entities[batch_idx]), len(all_entities[batch_idx])),
                                                       dtype=torch.float)

                entity_idx: Dict[Entity, int] = dict()

                for idx, entity in enumerate(all_entities[batch_idx]):
                    entity_idx[entity] = idx
                    curr_target_entity_comps[idx][idx] = 1.0

                for template in data_element.template_collection:
                    template: Template
                    tset = template.get_assigned_entities()
                    for ent1, ent2 in itertools.combinations(tset, 2):
                        curr_target_entity_comps[entity_idx[ent1]][entity_idx[ent2]] = 1.0

                target_entity_comp_list.append(curr_target_entity_comps)

            same_inst_vals, diff_inst_vals = calc_similarity_stats(pred_sims=pred_entity_comp_list,
                                                                   target_sims=target_entity_comp_list)

            # target_entity_compatibilities = torch.stack(target_entity_comp_list)
            loss_entity_compatibilities = sum([
                loss_fc_entity_compatibilities(pred_entity_comp_list[batch_idx].to(device),
                                               target_entity_comp_list[batch_idx].to(device))
                for batch_idx in range(0, len(chunk))
            ])

            total_loss = (loss_entity_positions + loss_entity_compatibilities) / batch_size
            last_loss = total_loss.item()
            print(epoch, j,
                  'loss: ' + str(total_loss.item()),
                  "positions: " + str((loss_entity_positions / batch_size).item()),
                  "compatibilities: " + str((loss_entity_compatibilities / batch_size).item()),
                  flush=True)
            losses.append(total_loss.item())
            total_loss.backward()
            # if j % 50 == 0:
            #    print("Step")
            optimizer.step()
            optimizer.zero_grad()

        print("LR:", optimizer.param_groups[0]["lr"])
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()
            itc_module.eval()
            slot_filling_evaluation = evaluate_slot_filling(
                data_elements=valid_data_elements,
                itc_module=itc_module,
                task_config_dict=task_config_dict,
                max_slots_cardinalities=max_slots_cardinalities,
                device=device,
                same_inst_vals=same_inst_vals,
                diff_inst_vals=diff_inst_vals
            )
            itc_module.train()

            slot_filling_evaluation.print_out()

            if trial is not None:
                # trial.report(statistics.mean(losses), i)
                trial.report(slot_filling_evaluation.compute_micro_stats().f1(), epoch + 1)

        print(f"Epoch {epoch} avg loss: {statistics.mean(losses)}")
        # optimizer.step()

    torch.cuda.empty_cache()
    itc_module.eval()
    slot_filling_evaluation = evaluate_slot_filling(
        data_elements=valid_data_elements,
        itc_module=itc_module,
        task_config_dict=task_config_dict,
        max_slots_cardinalities=max_slots_cardinalities,
        device=device,
        same_inst_vals=same_inst_vals,
        diff_inst_vals=diff_inst_vals
    )
    itc_module.train()

    print("Final evaluation on validation set:")
    slot_filling_evaluation.print_out()

    return slot_filling_evaluation.compute_micro_stats().f1(), same_inst_vals, diff_inst_vals


def objective(trial: Trial,
              device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
              batch_size=None,
              learning_rate=None,
              ld=None):
    device = torch.device(device_str)
    if batch_size is None:
        batch_size = task_config_dict['batch_size']  # trial.suggest_int('batch_size', 1, 10)
    if learning_rate is None:
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    if ld is None:
        ld = trial.suggest_float('ld', 0.9, 1.0, log=True)

    print(batch_size, learning_rate, ld, flush=True)

    # instantiate neural network
    itc_module: IntraTemplateCompModule
    itc_tokenizer: ITCTokenizer
    if 'allenai/longformer' in task_config_dict['model_name']:
        itc_module = ITCLongformer(
            num_entity_classes=len(task_config_dict['slot_indices']),
            model_name=task_config_dict['model_name']
        )
    elif 'allenai/led' in task_config_dict['model_name']:
        itc_module = ITCLED(
            num_entity_classes=len(task_config_dict['slot_indices']),
            model_name=task_config_dict['model_name']
        )
    elif 'google/t5' in task_config_dict['model_name'] or 'google/flan-t5' in task_config_dict['model_name']:
        itc_module = ITCFlanT5(num_entity_classes=len(task_config_dict['slot_indices']),
                               model_name=task_config_dict['model_name'])
    else:
        raise RuntimeError("Unknown model name!")

    # training
    f1, same_inst_vals, diff_inst_vals = train(
        data_elements=train_data_elements,
        valid_data_elements=validation_data_elements,
        itc_module=itc_module,
        num_epochs=task_config_dict['epochs'],
        device=device,
        batch_size=batch_size,
        ld=ld,
        learning_rate=learning_rate,
        trial=trial,
    )

    out_name = f"{task_config_dict['disease_prefix']}_{task_config_dict['model_name'][task_config_dict['model_name'].rfind('/') + 1:]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{batch_size}_{learning_rate}_{ld}"

    torch.save(itc_module.state_dict(),
               f"extr_model_{out_name}.pt")

    with open(f"extr_simstats_{out_name}.pkl", 'wb') as pklfile:
        pickle.dump((same_inst_vals, diff_inst_vals), pklfile)

    return f1


def cli_train(task_config_dict,
              # hyperparam_search: bool = False,
              device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
              n_trials: int = 30,
              batch_size=None,
              learning_rate=None,
              ld=None):
    # instantiate neural network
    itc_module: IntraTemplateCompModule
    itc_tokenizer: ITCTokenizer
    if 'allenai/longformer' in task_config_dict['model_name']:
        itc_module = ITCLongformer(
            num_entity_classes=len(task_config_dict['slot_indices']),
            model_name=task_config_dict['model_name']
        )
    elif 'allenai/led' in task_config_dict['model_name']:
        itc_module = ITCLED(
            num_entity_classes=len(task_config_dict['slot_indices']),
            model_name=task_config_dict['model_name']
        )
    elif 'google/t5' in task_config_dict['model_name'] or 'google/flan-t5' in task_config_dict['model_name']:
        itc_module = ITCFlanT5(num_entity_classes=len(task_config_dict['slot_indices']),
                               model_name=task_config_dict['model_name'])
    else:
        raise RuntimeError("Unknown model name!")

    itc_tokenizer = itc_module.tokenizer

    # import data
    train_data, test_data, validation_data = import_train_test_valid_data_from_task_config(task_config_dict,
                                                                                           tokenizer=itc_tokenizer)
    slot_indices = task_config_dict['slot_indices']
    slot_indices_reverse = {slot_indices[slot_name]: slot_name for slot_name in slot_indices}
    global train_data_elements, validation_data_elements, max_slots_cardinalities
    train_data_elements = extract_data_elements_from_dataset(
        dataset=train_data,
        tokenizer=itc_tokenizer,
        task_config_dict=task_config_dict
    )

    validation_data_elements = extract_data_elements_from_dataset(
        dataset=validation_data,
        tokenizer=itc_tokenizer,
        task_config_dict=task_config_dict
    )

    max_slots_cardinalities = gen_max_slot_cardinalities(train_data)

    # if hyperparam_search:
    storage = JournalStorage(JournalFileStorage(f"extr_optuna_{datetime.now().strftime('%Y-%m-%d')}.log"))
    study = optuna.create_study(direction='maximize',
                                study_name=f"Extractive {task_config_dict['disease_prefix']} {task_config_dict['model_name'][task_config_dict['model_name'].rfind('/') + 1:]} {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                                storage=storage, load_if_exists=True)
    study.optimize(lambda trial: objective(trial=trial,
                                           device_str=device_str,
                                           batch_size=batch_size,
                                           learning_rate=learning_rate,
                                           ld=ld),
                   n_trials=n_trials,
                   n_jobs=1)
    print(study.best_params)
    # else:
    #     batch_size = task_config_dict['batch_size']  # trial.suggest_int('batch_size', 1, 10)
    #     learning_rate = 0.0001
    #     ld = 0.99
    #
    #     print(batch_size, learning_rate, flush=True)
    #
    #     # training
    #     loss, same_inst_vals, diff_inst_vals = train(
    #         data_elements=train_data_elements,
    #         valid_data_elements=validation_data_elements,
    #         itc_module=itc_module,
    #         num_epochs=task_config_dict['epochs'],
    #         device=device,
    #         batch_size=batch_size,
    #         ld=ld,
    #         learning_rate=learning_rate,
    #         trial=None,
    #     )
    #
    #     out_name = f"{task_config_dict['disease_prefix']}_{task_config_dict['model_name'][task_config_dict['model_name'].rfind('/') + 1:]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{batch_size}_{learning_rate}_{ld}"
    #
    #     torch.save(itc_module.state_dict(),
    #                f"extr_model_{out_name}.pt")
    #
    #     with open(f"extr_simstats_{out_name}.pkl", 'wb') as pklfile:
    #         pickle.dump((same_inst_vals, diff_inst_vals), pklfile)

    # torch.save(itc_module.state_dict(),
    #           f"extr_model_{task_config_dict['disease_prefix']}_{task_config_dict['model_name'][task_config_dict['model_name'].rfind('/') + 1:]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{batch_size}_{learning_rate}_{ld}.pt")


if __name__ == '__main__':
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    if sys.argv[1].lower().endswith(".pkl"):  # len(sys.argv) > 2:
        if os.path.isfile(sys.argv[1]):
            with open(sys.argv[1], "rb") as bp_file:
                best_params = pickle.load(bp_file)
                print("Running with fixed best parameters")
                for bp in best_params:
                    if bp["approach"] != "extr":
                        continue
                    print(bp)
                    with open(bp["configpath"]) as fp:
                        task_config_dict = json.load(fp)
                        # task_config_dict = import_task_config(bp["configpath"])
                        cli_train(task_config_dict,
                                  device_str,
                                  n_trials=10,
                                  batch_size=int(bp["batchsize"]),
                                  learning_rate=float(bp["learningrate"]),
                                  ld=float(bp["lambda"]))
    elif sys.argv[1].lower().endswith(".json"):
        with open(sys.argv[1]) as fp:
            task_config_dict = json.load(fp)
            # task_config_dict = import_task_config(sys.argv[1])
            print("Running hyperparameter search")
            cli_train(task_config_dict, device_str, n_trials=30)

    # process command line arguments
    # task_config_filename = sys.argv[1]
    # # load task config
    # with open(task_config_filename) as fp:
    #     task_config_dict = json.load(fp)
    #
    # device_str = "cuda" if torch.cuda.is_available() else "cpu"
