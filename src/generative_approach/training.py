import pickle
import random
import sys
from datetime import datetime
from typing import Any

import optuna
from optuna import Trial
from optuna.storages import JournalStorage, JournalFileStorage
from torch.optim import AdamW

from data_handling import *
from generative_approach.evaluation import evaluate_slot_filling
from generative_approach.models.ed_model import EDModel
from template_lib.data_handling import import_train_test_data_from_task_config, \
    import_train_test_valid_data_from_task_config
from template_lib.max_cardinalities_estimator import gen_max_slot_cardinalities

task_config_dict: Any


def train(
        data_elements: List[DataElement],
        valid_dataset: SantoDataset,
        model: EDModel,
        temp_generation_tokenizer: TemplateGenerationTokenizer,
        max_slots_cardinalities: Dict[str, int],
        num_epochs: int = 30,
        batch_size: int = 1,
        ld: float = 0.99,
        learning_rate: float = 1e-5,
        device=torch.device("cpu"),
        out_name: str = None,
        trial: Trial = None,
):
    # training loop #################################################################
    model.model.to(device)

    if out_name is None:
        out_name = f"{task_config_dict['disease_prefix']}_{task_config_dict['model_name'][task_config_dict['model_name'].rfind('/') + 1:]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{batch_size}_{learning_rate}_{ld}",

    pad_token_id = model.tokenizer.pad_token_id
    optimizer = AdamW(model.model.parameters(), lr=learning_rate)  # TODO prev 1e-5
    lambda1 = lambda epoch: ld ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # data_elements = data_elements[:3]

    loss: Any = None

    for epoch in range(num_epochs):
        random.shuffle(data_elements)
        epoch_losses = list()

        for batch_offset in range(0, len(data_elements), batch_size):
            torch.cuda.empty_cache()
            batch = data_elements[batch_offset:batch_offset + batch_size]

            sos_offset = 1 if temp_generation_tokenizer.start_of_sentence_token is not None else 0
            eos_offset = -1 if temp_generation_tokenizer.end_of_sentence_token is not None else None

            input_ids = pt_stack_lists([data_element.input_token_ids for data_element in batch], pad_token_id).to(
                device)
            attention_mask = pt_stack_lists([[1] * len(data_element.input_token_ids) for data_element in batch], 0).to(
                device)

            decoder_input_ids = pt_stack_lists([data_element.output_token_ids[:eos_offset] for data_element in batch],
                                               pad_token_id).to(device)
            decoder_attention_mask = pt_stack_lists(
                [[1] * (len(data_element.output_token_ids) - (0 if eos_offset is None else 1)) for data_element in
                 batch], 0
            ).to(device)

            labels = pt_stack_lists(
                [data_element.output_token_ids[sos_offset:(None if isinstance(model, Longformer) else eos_offset)] for
                 data_element in batch], -100).to(device)

            global_attention_mask = torch.zeros_like(input_ids).to(device)
            global_attention_mask[:, 0] = 1

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,  # .to(device),
                # decoder_input_ids=decoder_input_ids,  # .to(device),
                # decoder_attention_mask=decoder_attention_mask,  # .to(device),
                global_attention_mask=global_attention_mask,  # .to(device),
                labels=labels,  # .to(device),
                return_dict=True
            )

            loss = outputs.loss
            print('loss: ' + str(loss.item()), flush=True)
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('***** epoch loss: ', sum(epoch_losses) / len(epoch_losses))
        print("LR:", optimizer.param_groups[0]["lr"])
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()
            model.model.eval()
            slot_filling_evaluation = evaluate_slot_filling(
                dataset=valid_dataset,
                model=model,
                temp_generation_tokenizer=temp_generation_tokenizer,
                task_config_dict=task_config_dict,
                max_slots_cardinalities=max_slots_cardinalities,
                device=device
            )
            model.model.train()

            slot_filling_evaluation.print_out()

            if trial is not None:
                # trial.report(statistics.mean(losses), i)
                trial.report(slot_filling_evaluation.compute_micro_stats().f1(), epoch + 1)

        # save model every 10th epoch
        # if (epoch + 1) % 10 == 0:
        #     # if (epoch + 1) % 10 == 0 or max(range(1)) == epoch:
        #     dirname = f"epoch_{epoch+1}_model_{task_config_dict['disease_prefix']}_{task_config_dict['model_name'][task_config_dict['model_name'].rfind('/')+1:]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{batch_size}_{learning_rate}_{ld}"
        #     save_model(model.model, dirname)

    dirname = f"gen_model_{out_name}"
    save_model(model.model, dirname)

    torch.cuda.empty_cache()
    model.model.eval()
    slot_filling_evaluation = evaluate_slot_filling(
        dataset=valid_dataset,
        model=model,
        temp_generation_tokenizer=temp_generation_tokenizer,
        task_config_dict=task_config_dict,
        max_slots_cardinalities=max_slots_cardinalities,
        device=device
    )
    model.model.train()

    slot_filling_evaluation.print_out()

    return slot_filling_evaluation.compute_micro_stats().f1()


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

    # load pretrained led model and create instantiate tokenizers #########################
    model_name = task_config_dict['model_name']
    model_class = get_model_class(model_name)
    model = model_class(model_name=model_name, device=device_str)

    # led_model = LEDForConditionalGeneration.from_pretrained(model_name).to(device)
    # led_tokenizer = LEDTokenizer.from_pretrained(model_name)
    # led_config = LEDConfig.from_pretrained(model_name)

    # create template generation tokenizer ################################################
    temp_generation_tokenizer = TemplateGenerationTokenizer(
        tokenizer=model.tokenizer,
        template_names=task_config_dict['slots_ordering_dict'].keys(),
        slot_names=task_config_dict['used_slots'],
        start_of_sentence_token=model.tokenizer.bos_token,
        end_of_sentence_token=model.tokenizer.eos_token,
        filler_sep_token='[FILLER_SEP]',
        control_tokens_start_id=model.tokenizer.vocab_size
    )
    out_name = f"{task_config_dict['disease_prefix']}_{task_config_dict['model_name'][task_config_dict['model_name'].rfind('/') + 1:]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{batch_size}_{learning_rate}_{ld}"
    temp_generation_tokenizer.to_json(f"tokenizer_gen_model_{out_name}.json")

    # add embeddings for control tokens
    model.model.resize_token_embeddings(
        model.tokenizer.vocab_size + len(temp_generation_tokenizer.control_token_ids)
        - (1 if model.tokenizer.bos_token is not None else 0)
        - (1 if model.tokenizer.eos_token is not None else 0)
        - (1 if model.tokenizer.pad_token is not None else 0))
    # print(led_model.led.shared.num_embeddings)

    start_token = temp_generation_tokenizer.start_of_sentence_token
    if temp_generation_tokenizer.start_of_sentence_token is None:
        if temp_generation_tokenizer.pad_token is not None and isinstance(model, FlanT5):
            start_token = temp_generation_tokenizer.pad_token

    template_encoder = LinearTemplateEncoder(
        top_level_templates=task_config_dict['top_level_templates'],
        slots_ordering=task_config_dict['slots_ordering_dict'],
        templates_ordering=sorted(task_config_dict['slots_ordering_dict'].keys()),
        used_slots=task_config_dict['used_slots'],
        slots_containing_templates=task_config_dict['slots_containing_templates'],
        filler_sep_token=temp_generation_tokenizer.filler_sep_token,
        start_of_document_token=start_token,
        end_of_document_token=temp_generation_tokenizer.end_of_sentence_token
    )

    # load training data ###############################################
    training_dataset, test_dataset, validation_dataset = import_train_test_valid_data_from_task_config(task_config_dict,
                                                                                                       temp_generation_tokenizer)

    training_data_elements = extract_data_elements_from_dataset(
        dataset=training_dataset,
        tokenizer=temp_generation_tokenizer,
        template_encoder=template_encoder,
        max_input_seq_len=model.max_encoder_position_embeddings,
        max_output_seq_len=model.max_decoder_position_embeddings,
        encode_flat=False,
        per_sentence=False
    )

    # validation_data_elements = extract_data_elements_from_dataset(
    #     dataset=validation_dataset,
    #     tokenizer=temp_generation_tokenizer,
    #     template_encoder=template_encoder,
    #     max_input_seq_len=model.max_encoder_position_embeddings,
    #     max_output_seq_len=model.max_decoder_position_embeddings,
    #     encode_flat=False,
    #     per_sentence=False
    # )
    print('number of training data elements: ', len(training_data_elements))

    max_slots_cardinalities = gen_max_slot_cardinalities(training_dataset)

    f1 = train(
        data_elements=training_data_elements,
        valid_dataset=validation_dataset,
        model=model,
        temp_generation_tokenizer=temp_generation_tokenizer,
        max_slots_cardinalities=max_slots_cardinalities,
        num_epochs=task_config_dict['num_epochs'],
        batch_size=task_config_dict['batch_size'],
        ld=ld,
        learning_rate=learning_rate,
        device=device,
        out_name=out_name,
        trial=trial
    )

    return f1


def cli_train(task_config_dict,
              #hyperparam_search: bool = False,
              device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
              n_trials: int = 30,
              batch_size=None,
              learning_rate=None,
              ld=None):
    #device = torch.device(device_str)

    # load pretrained led model and create instantiate tokenizers #########################
    model_name = task_config_dict['model_name']
    model_class = get_model_class(model_name)
    model = model_class(model_name=model_name, device=device_str)

    # led_model = LEDForConditionalGeneration.from_pretrained(model_name).to(device)
    # led_tokenizer = LEDTokenizer.from_pretrained(model_name)
    # led_config = LEDConfig.from_pretrained(model_name)

    # create template generation tokenizer ################################################
    temp_generation_tokenizer = TemplateGenerationTokenizer(
        tokenizer=model.tokenizer,
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

    start_token = temp_generation_tokenizer.start_of_sentence_token
    if temp_generation_tokenizer.start_of_sentence_token is None:
        if temp_generation_tokenizer.pad_token is not None and isinstance(model, FlanT5):
            start_token = temp_generation_tokenizer.pad_token

    template_encoder = LinearTemplateEncoder(
        top_level_templates=task_config_dict['top_level_templates'],
        slots_ordering=task_config_dict['slots_ordering_dict'],
        templates_ordering=sorted(task_config_dict['slots_ordering_dict'].keys()),
        used_slots=task_config_dict['used_slots'],
        slots_containing_templates=task_config_dict['slots_containing_templates'],
        filler_sep_token=temp_generation_tokenizer.filler_sep_token,
        start_of_document_token=start_token,
        end_of_document_token=temp_generation_tokenizer.end_of_sentence_token
    )

    # load training data ###############################################
    training_dataset, test_dataset, validation_dataset = import_train_test_valid_data_from_task_config(task_config_dict,
                                                                                                       temp_generation_tokenizer)

    training_data_elements = extract_data_elements_from_dataset(
        dataset=training_dataset,
        tokenizer=temp_generation_tokenizer,
        template_encoder=template_encoder,
        max_input_seq_len=model.max_encoder_position_embeddings,
        max_output_seq_len=model.max_decoder_position_embeddings,
        encode_flat=False,
        per_sentence=False
    )

    # validation_data_elements = extract_data_elements_from_dataset(
    #     dataset=validation_dataset,
    #     tokenizer=temp_generation_tokenizer,
    #     template_encoder=template_encoder,
    #     max_input_seq_len=model.max_encoder_position_embeddings,
    #     max_output_seq_len=model.max_decoder_position_embeddings,
    #     encode_flat=False,
    #     per_sentence=False
    # )
    print('number of training data elements: ', len(training_data_elements))

    max_slots_cardinalities = gen_max_slot_cardinalities(training_dataset)

    #if hyperparam_search:
    storage = JournalStorage(JournalFileStorage(f"gen_optuna_{datetime.now().strftime('%Y-%m-%d')}.log"))
    study = optuna.create_study(direction='maximize',
                                study_name=f"Generative {task_config_dict['disease_prefix']} {task_config_dict['model_name'][task_config_dict['model_name'].rfind('/') + 1:]} {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
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
    #     learning_rate = 0.0001
    #     ld = 0.99
    #
    #     temp_generation_tokenizer.to_json(
    #         f"tokenizer_gen_model_{task_config_dict['disease_prefix']}_{task_config_dict['model_name'][task_config_dict['model_name'].rfind('/') + 1:]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{learning_rate}_{ld}.json"
    #     )
    #
    #     train(
    #         data_elements=training_data_elements,
    #         valid_dataset=validation_dataset,
    #         model=model,
    #         temp_generation_tokenizer=temp_generation_tokenizer,
    #         max_slots_cardinalities=max_slots_cardinalities,
    #         num_epochs=task_config_dict['num_epochs'],
    #         batch_size=task_config_dict['batch_size'],
    #         ld=ld,
    #         learning_rate=learning_rate,
    #         device=device
    #     )


if __name__ == "__main__":
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    if sys.argv[1].lower().endswith(".pkl"):#len(sys.argv) > 2:
        if os.path.isfile(sys.argv[1]):
            with open(sys.argv[1], "rb") as bp_file:
                best_params = pickle.load(bp_file)
                print("Running with fixed best parameters")
                for bp in best_params:
                    if bp["approach"] != "gen":
                        continue
                    print(bp)
                    task_config_dict = import_task_config(bp["configpath"])
                    cli_train(task_config_dict,
                              device_str,
                              n_trials=10,
                              batch_size=int(bp["batchsize"]),
                              learning_rate=float(bp["learningrate"]),
                              ld=float(bp["lambda"]))
    elif sys.argv[1].lower().endswith(".json"):
        task_config_dict = import_task_config(sys.argv[1])
        print("Running hyperparameter search")
        cli_train(task_config_dict, device_str, n_trials=30)

    # device = "cpu" #torch.device("cpu")
