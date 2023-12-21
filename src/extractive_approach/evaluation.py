import json
import pickle
from argparse import ArgumentParser

from extractive_approach.clustering_approaches import *
from generative_approach.max_cardinalities_estimator import gen_max_slot_cardinalities
from template_lib.SlotFillingEvaluation import SlotFillingEvaluation
from template_lib.TemplateAlignment import TemplateAlignment
from extractive_approach.data_handling import *
from extractive_approach.entity_extraction import *
from extractive_approach.modules import IntraTemplateCompModule, ITCLongformer, ITCLED, ITCFlanT5
from extractive_approach.entity_clustering import EntityCompatibilityCollection
import torch

from template_lib.data_handling import import_train_test_valid_data_from_task_config


def select_entities_by_labels(
        entities: Iterable[Entity],
        labels: Iterable[str]
) -> List[Entity]:
    repl_labels = [l for l in labels]
    return list(filter(
        lambda e: e.get_label() in repl_labels,
        entities
    ))


def evaluate_slot_filling(
        data_elements: Iterable[DataElement],
        itc_module: IntraTemplateCompModule,
        task_config_dict: Dict,
        max_slots_cardinalities: Optional[Dict[str, int]],
        device=torch.device("cpu"),
        same_inst_vals: Optional[List] = None,
        diff_inst_vals: Optional[List] = None
):
    used_slots = set(task_config_dict['used_slots']) - set(task_config_dict['slots_containing_templates'].keys())

    if max_slots_cardinalities is None:
        max_slots_cardinalities = dict()
    itc_module.to(device)
    itc_module.eval()
    slot_filling_evaluation = SlotFillingEvaluation()

    instance_counter = 0

    for data_element in data_elements:
        torch.cuda.empty_cache()

        predicted_template_collection, instance_counter = single_run(data_element, itc_module, task_config_dict,
                                                                     max_slots_cardinalities, device, same_inst_vals,
                                                                     diff_inst_vals, instance_counter)

        # update evaluation
        template_alignment = TemplateAlignment(
            gt_temp_collection=data_element.template_collection,
            predicted_temp_collection=predicted_template_collection,
            used_slots=used_slots
        )
        template_alignment.update_evaluation(slot_filling_evaluation)
        slot_filling_evaluation.update_instance_counts(
            gt_template_collection=data_element.template_collection,
            predicted_template_collection=predicted_template_collection
        )

    return slot_filling_evaluation


def single_run(data_element: DataElement,
               itc_module: IntraTemplateCompModule,
               task_config_dict: Dict,
               max_slots_cardinalities: Optional[Dict[str, int]],
               device=torch.device("cpu"),
               same_inst_vals: Optional[List] = None,
               diff_inst_vals: Optional[List] = None,
               instance_counter: int = 0):
    itc_module.to(device)
    with torch.no_grad():
        slot_indices_reverse_dict = {slot_index: slot_name for slot_name, slot_index in
                                     task_config_dict['slot_indices'].items()}

        predicted_template_collection = TemplateCollection()
        # call encoder
        input_ids = torch.tensor([data_element.input_ids]).to(device)
        attention_mask = torch.tensor([data_element.attention_mask]).to(device)
        encoder_output = itc_module.encoder(input_ids=input_ids,
                                            attention_mask=attention_mask)
        last_hidden_state = encoder_output.last_hidden_state.to(device)
        # predict entities
        pred_start_positions = itc_module.layer_entity_start_positions(last_hidden_state)
        pred_end_positions = itc_module.layer_entity_end_positions(last_hidden_state)
        target_start_pos = torch.stack(
            [torch.tensor(data_element.padded_entity_start_positions(len(data_element.input_ids))) for data_element in
             [data_element]])  # .to(device)
        target_end_pos = torch.stack(
            [torch.tensor(data_element.padded_entity_end_positions(len(data_element.input_ids))) for data_element in
             [data_element]])  # .to(device)
        new_shape2 = list(target_start_pos.shape)[1:]
        new_shape2[0] *= target_start_pos.shape[0]
        target_start_pos = target_start_pos.reshape(new_shape2).to(device)
        target_end_pos = target_end_pos.reshape(new_shape2).to(device)
        new_shape = list(pred_start_positions.shape)[1:]
        new_shape[0] *= pred_start_positions.shape[0]
        pred_start_positions2 = pred_start_positions.reshape(new_shape)
        pred_end_positions2 = pred_end_positions.reshape(new_shape)
        loss_fc_entity_positions = torch.nn.CrossEntropyLoss()
        loss_entity_positions = (loss_fc_entity_positions(pred_start_positions2, target_start_pos)
                                 + loss_fc_entity_positions(pred_end_positions2, target_end_pos))
        print("Loss:", loss_entity_positions.item(), flush=True)
        print(pred_start_positions2.argmax(dim=1), flush=True)
        print(target_start_pos, flush=True)
        print(pred_end_positions2.argmax(dim=1), flush=True)
        print(target_end_pos, flush=True)
        pred_start_positions = pred_start_positions.argmax(dim=2).tolist()[0]
        pred_end_positions = pred_end_positions.argmax(dim=2).tolist()[0]
        predicted_entities = extract_entities_from_slot_indices_sequences(
            start_indices_sequence=pred_start_positions,
            end_indices_sequence=pred_end_positions,
            sentence_boundaries=data_element.sentence_boundaries,
            slot_indices_reverse_dict=slot_indices_reverse_dict,
            neutral_label_index=task_config_dict['slot_indices']['none'],
            document=data_element.document
        )
        print("Predicted\n" + "\n".join(
            [str((e.get_label(), e)) for e in sorted(predicted_entities, key=lambda x: x.get_label())]))
        print("##############################################")
        print("GroundTruth\n" + "\n".join(
            [str((e.get_label(), e)) for e in data_element.template_collection.get_all_entities()]))
        print("##############################################")
        # for ent in predicted_entities:
        #     ent.set_label("has" + ent.get_label())
        # compute entity representations
        itc_module.compute_entity_representations(
            chunk=[predicted_entities],
            sentence_boundaries=[data_element.sentence_boundaries],
            last_hidden_state=last_hidden_state
        )
        # assign entities to template instances
        for template_type in task_config_dict['template_slots'].keys():
            if template_type == 'EvidenceQuality':
                continue

            # only consider entities which can be assigned to current template type
            selected_entities = select_entities_by_labels(
                entities=predicted_entities,
                labels=task_config_dict['template_slots'][template_type]
            )

            if len(selected_entities) == 0:
                continue

            # single instance case
            if template_type in [
                'Publication',
                'Population',
                'ClinicalTrial'
            ]:
                # create new template instance and assign entities
                instance_counter += 1
                template = Template(
                    _type=template_type,
                    _id=template_type + str(instance_counter)
                )
                for entity in selected_entities:
                    template.add_slot_filler(
                        slot_name=entity.get_label(),
                        slot_value=entity
                    )
                predicted_template_collection.add_template(template)
            else:  # multi instance case
                # compute entity pair compatibilities
                entity_pairs = list(itertools.combinations_with_replacement(selected_entities, 2))
                entity_pair_compatibilities = itc_module.compute_entity_compatibilities(
                    chunk_entity_pairs=[entity_pairs])

                entity_pair_compatibilities = [pc.sigmoid() for pc in entity_pair_compatibilities]

                compatibility_collection = EntityCompatibilityCollection()
                for i, entity_pair in enumerate(entity_pairs):
                    compatibility_collection.add_entity_compatibility(
                        entity_pair=entity_pair,
                        compatibility=entity_pair_compatibilities[0][i]
                    )

                print(selected_entities, flush=True)

                clustering_approach: ClusteringApproach
                if same_inst_vals is None or diff_inst_vals is None:
                    clustering_approach = DummyClustering(
                        entities=selected_entities,
                        entity_compatibilities=compatibility_collection,
                        template_type=template_type,
                        max_slots_cardinalities=max_slots_cardinalities
                    )
                else:
                    avg_same = statistics.mean([torch.sigmoid(torch.tensor(v)).item() for v in same_inst_vals])
                    avg_diff = statistics.mean([torch.sigmoid(torch.tensor(v)).item() for v in diff_inst_vals])
                    threshold = (avg_same + avg_diff) / 2  # TODO: weight by stddev?

                    clustering_approach = AgglomerativeClustering(
                        entities=selected_entities,
                        entity_compatibilities=compatibility_collection,
                        template_type=template_type,
                        distance_threshold=threshold,
                        max_slots_cardinalities=max_slots_cardinalities
                    )

                # apply clustering approach here
                # predicted_templates = ...
                predicted_templates = clustering_approach.clusters().templates
                for template in predicted_templates:
                    predicted_template_collection.add_template(template)
    return predicted_template_collection, instance_counter


def cli_eval(task_config: str, model_path: str, sim_path: Optional[str] = None, use_validation: bool = False):
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

    datat_elements = extract_data_elements_from_dataset(
        dataset=test_set if not use_validation else validation_data,
        # dataset=validation_data,
        # dataset=list(train_set)[:1],
        # dataset=train_set,
        tokenizer=itc_tokenizer,
        task_config_dict=task_config_dict
    )

    max_slots_cardinalities = gen_max_slot_cardinalities(train_set)

    same_inst_vals, diff_inst_vals = None, None

    if sim_path is not None:
        with open(sim_path, "rb") as fp:
            same_inst_vals, diff_inst_vals = pickle.load(fp)

    # evaluation
    slot_filling_evaluation = evaluate_slot_filling(
        data_elements=datat_elements,
        itc_module=model,
        task_config_dict=task_config_dict,
        max_slots_cardinalities=max_slots_cardinalities,
        device=device,
        same_inst_vals=same_inst_vals,
        diff_inst_vals=diff_inst_vals
    )
    slot_filling_evaluation.print_out()

    return slot_filling_evaluation


if __name__ == '__main__':
    # process required command line arguments
    argparser = ArgumentParser()
    argparser.add_argument('--task_config', required=True)
    argparser.add_argument('--model', required=True)
    argparser.add_argument('--sims', required=False, default=None)
    arguments = argparser.parse_args()

    cli_eval(task_config=arguments.task_config,
             model_path=arguments.model,
             sim_path=arguments.sims,
             use_validation=True)
    cli_eval(task_config=arguments.task_config,
             model_path=arguments.model,
             sim_path=arguments.sims,
             use_validation=False)
