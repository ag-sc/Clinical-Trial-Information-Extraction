import sys
import os
import json
import collections
from argparse import ArgumentParser
from template_lib import ctro, consts
from template_lib.santo.SantoDataset import SantoDataset, import_trial_ids
from utils import IdentityTokenizer

if __name__ == "__main__":
    # process arguments ###################################################
    argparser = ArgumentParser()
    argparser.add_argument('--host', required=True)
    argparser.add_argument('--disease-prefix', required=True)
    argparser.add_argument('--min-slot-freq', required=True)
    argparser.add_argument('--filename', required=True)
    argparser.add_argument('--model', required=False, default='allenai/longformer-base-4096')
    #argparser.add_argument('--model-dim', required=False, default=None)
    argparser.add_argument('--batch-size', required=False, default=None)
    argparser.add_argument('--epochs', required=False, default=None)
    arguments = argparser.parse_args()

    host = arguments.host
    disease_prefix = arguments.disease_prefix
    min_slot_freq = int(arguments.min_slot_freq)
    task_config_filename = arguments.filename
    model_name = arguments.model
    batch_size = 1
    if arguments.batch_size is not None:
        batch_size = int(arguments.batch_size)
    epochs = 30
    if arguments.epochs is not None:
        epochs = int(arguments.epochs)

    # model_dim = None
    # if arguments.model_dim is not None:
    #     model_dim = int(arguments.model_dim)
    # elif model_name == 'allenai/longformer-base-4096':
    #     model_dim = 768
    # else:
    #     raise RuntimeError("Unknown model dims!")

    print('host:', host)
    print('disease prefix:', disease_prefix)
    print('min slot freq:', min_slot_freq)
    print('task config filename:', task_config_filename)
    print('model name:', model_name)
    #print('model dim:', model_dim)
    print('batch size:', batch_size)
    print('epochs:', epochs)

    ontology = ctro.CtroOntology(consts.template_lib_path)


    task_config_dict = {
        'disease_prefix': disease_prefix,

        # dataset paths
        'dataset_path': None,

        # set below from Ctro ontology
        'template_slots': None,

        # set below
        'used_slots': None,

        # set below
        'slot_indices': None,

        # list of slot names which contain template instances as slot fillers
        'slots_containing_templates': {
            'describes': 'ClinicalTrial',
            'hasArm': 'Arm',
            'hasPopulation': 'Population',
            'hasDiffBetweenGroups': 'DiffBetweenGroups',
            'hasOutcome1': 'Outcome',
            'hasOutcome2': 'Outcome',
            'hasOutcome': 'Outcome',
            'hasAdverseEffect': 'Outcome',
            'hasIntervention': 'Intervention',
            'hasMedication': 'Medication',
            'hasEndpoint': 'Endpoint',
        },
        'model_name': model_name,
        #'model_dim': model_dim,
        'batch_size': batch_size,
        'epochs': epochs,
    }

    # dataset path ######################################
    if host == 'local':
        task_config_dict['dataset_path'] = consts.data_path
    elif host == 'rational':
        task_config_dict['dataset_path'] = consts.data_path
    else:
        raise Exception('unknown host: ' + host)


    # set id filenames ####################################
    if disease_prefix == 'dm2':
        task_config_dict['rel_train_ids_filename'] = 'dataset_splits/dm2_train_ids.txt'
        task_config_dict['rel_validation_ids_filename'] = 'dataset_splits/dm2_validation_ids.txt'
        task_config_dict['rel_test_ids_filename'] = 'dataset_splits/dm2_test_ids.txt'
    elif disease_prefix == 'gl':
        task_config_dict['rel_train_ids_filename'] = 'dataset_splits/glaucoma_train_ids.txt'
        task_config_dict['rel_validation_ids_filename'] = 'dataset_splits/glaucoma_validation_ids.txt'
        task_config_dict['rel_test_ids_filename'] = 'dataset_splits/glaucoma_test_ids.txt'
    else:
        raise Exception('unknown disease prefix: ' + disease_prefix)


    # template slots ##############################################
    slots_ordering_dict = dict()
    for template_type in ontology.used_group_names:
        slots_ordering_dict[template_type] = sorted(ontology.group_slots[template_type])
    task_config_dict['template_slots'] = slots_ordering_dict


    # estimate used slots ###################################################
    dataset_path = os.path.join(os.path.dirname(sys.modules["template_lib"].__file__), "..", "..", "data")
    rel_train_ids_filename = task_config_dict['rel_train_ids_filename']

    train_ids_filename = os.path.join(dataset_path, rel_train_ids_filename)
    train_ids = import_trial_ids(train_ids_filename)

    tokenizer = IdentityTokenizer()
    dataset = SantoDataset(dataset_path, train_ids, disease_prefix, 'admin', tokenizer)

    # count how often slots are used
    slot_names = list()
    for doc, temp_collection in dataset:
        for template in temp_collection:
            for slot_name, slot_fillers in template:
                slot_names.append(slot_name)

    slot_freqs = collections.Counter(slot_names)
    #used_slots = [slot_name.replace('has', '') for slot_name, freq in slot_freqs.items() if freq >= min_slot_freq]
    used_slots = [slot_name for slot_name, freq in slot_freqs.items() if freq >= min_slot_freq]
    task_config_dict['used_slots'] = used_slots

    # create indices for textual slots ######################################
    textual_slots = ['none'] # add special None slot
    for slot_name in used_slots:
        if slot_name not in task_config_dict['slots_containing_templates']:
            textual_slots.append(slot_name)


    # assign indices to used slots
    slot_indices = {slot_name: index for index, slot_name in enumerate(textual_slots)}
    print(slot_indices)
    task_config_dict['slot_indices'] = slot_indices


    # save task config dict as json file #################################
    fp = open(task_config_filename, 'w')
    json.dump(task_config_dict, fp)
    fp.close()
