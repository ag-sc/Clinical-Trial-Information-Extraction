from typing import List, Dict, Iterable, Tuple, Sequence, Any
from template_lib.data_classes.TemplateCollection import *
from template_lib.data_classes.Document import *
from template_lib.data_classes.Sentence import *
from template_lib.santo.SantoDataset import SantoDataset
import torch
from extractive_approach.utils import ITCTokenizer


def encode_sentences_longformer(
        sentences: Sequence[Sentence],
        tokenizer: ITCTokenizer,
) -> Tuple[List[str], Dict[int, Tuple[int, int]]]:
    tokens = list()
    sentence_boundaries = dict()

    for sentence in sentences:
        if tokenizer.start_of_sentence_token is not None:
            tokens.append(tokenizer.start_of_sentence_token)
        sentence_tokens = sentence.get_tokens()

        # estimate sentence boundaries, both offsets inclusive
        sentence_start_offset = len(tokens)
        sentence_end_offset = sentence_start_offset + len(sentence_tokens) - 1
        sentence_boundaries[sentence.get_index()] = (sentence_start_offset, sentence_end_offset)

        tokens.extend(sentence.get_tokens())
        if tokenizer.end_of_sentence_token is not None:
            tokens.append(tokenizer.end_of_sentence_token)

    return tokens, sentence_boundaries


def encode_sentences_bert(
        sentences: Sequence[Sentence],
        tokenizer: ITCTokenizer,
        max_chunk_size: int = 512
) -> Tuple[List[List[int]], Dict[int, Tuple[int, int]]]:
    pass


def encode_sentences_bert_paragraph_split(
        sentences: Sequence[Sentence],
        tokenizer: ITCTokenizer,
) -> Tuple[List[List[int]], Dict[int, Tuple[int, int]]]:
    pass


def encode_sentences(
        sentences: Sequence[Sentence],
        tokenizer: ITCTokenizer,
        encoder_type: str = 'longformer'
) -> Tuple[List[str], Dict[int, Tuple[int, int]]]:
    if encoder_type == 'longformer':
        return encode_sentences_longformer(sentences, tokenizer)
    elif encoder_type == 'bert':
        # todo: add max_chunk_size from tokenizer
        return encode_sentences_bert(sentences, tokenizer)
    elif encoder_type == 'bert_paragraph_split':
        # todo add arguments for split boundaries
        return encode_sentences_bert_paragraph_split(sentences, tokenizer)
    else:
        raise ValueError(f'invalid encoder type: {encoder_type}')


def encode_entities(
        entities: Iterable[Entity],
        label_indices: Dict[str, int],
        sentence_boundaries: Dict,
        num_tokens: int,
        neutral_index: int
) -> Tuple[List[int], List[int]]:
    encoded_start_positions = [neutral_index] * num_tokens
    encoded_end_positions = [neutral_index] * num_tokens

    for entity in entities:
        if entity.get_label() not in label_indices:
            continue

        sentence_start_offset, _ = sentence_boundaries[entity.get_sentence_index()]

        global_entity_start_pos = sentence_start_offset + entity.get_start_pos()
        global_entity_end_pos = sentence_start_offset + entity.get_end_pos()
        label_index = label_indices[entity.get_label()]

        encoded_start_positions[global_entity_start_pos] = label_index
        encoded_end_positions[global_entity_end_pos] = label_index

    return encoded_start_positions, encoded_end_positions


def entity_pair_compatibilities_(entities: Iterable[Entity]):
    entities = list(entities)

    # check if there is any entity pair
    if len(entities) < 2:
        return list()

    entity_pair_compatibilities = list()

    # check for each entity pair if the corresponding entities belong to the same template
    for entity_pair in itertools.combinations(entities, 2):
        if len(entity_pair[0].get_referencing_template_ids() & entity_pair[1].get_referencing_template_ids()) > 0:
            compatibility_label = 1.0
        else:
            compatibility_label = 0.0

        entity_pair_compatibilities.append((entity_pair, compatibility_label))

    return entity_pair_compatibilities


def encode_entity_pair_compatibilities(
        template_collection: TemplateCollection,
        template_types: Iterable[str]
):
    entity_pair_compatibilities = list()
    for template_type in template_types:
        templates = template_collection.get_templates_by_type(template_type)
        entities = [entity for temp in templates for entity in temp.get_assigned_entities()]
        entity_pair_compatibilities.extend(entity_pair_compatibilities_(entities))

    return entity_pair_compatibilities



class DataElement:
    def __init__(
            self,
            document: Document,
            template_collection: TemplateCollection,
            tokenizer: ITCTokenizer,
            task_config_dict,
    ):
        self.document = document
        self.template_collection = template_collection

        # encode input text #########################################
        input_tokens, self.sentence_boundaries = encode_sentences(
            sentences=document.get_sentences(),
            tokenizer=tokenizer,
            encoder_type='longformer'
        )

        self.tokenizer = tokenizer
        self.task_config_dict = task_config_dict

        self.input_tokens = input_tokens
        self.input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        self.attention_mask = [1] * len(input_tokens)

        # encode start/end positions of slot_fillers
        self.entity_start_positions, self.entity_end_positions = encode_entities(
            entities=template_collection.get_all_entities(),
            label_indices=task_config_dict['slot_indices'],
            sentence_boundaries=self.sentence_boundaries,
            num_tokens=len(input_tokens),
            neutral_index=task_config_dict['slot_indices']['none']
        )

        # entity compatibility labels
        self.entity_pair_compatibility_labels = encode_entity_pair_compatibilities(
            template_collection=template_collection,
            template_types=set(task_config_dict['template_slots'].keys()) - {'Publication', 'Population', 'ClinicalTrial', 'EvidenceQuality'}
        )

    @staticmethod
    def pad_list(l: List, total_len: int, pad_elem: Any):
        orig_len = len(l)
        assert total_len >= orig_len
        if total_len == orig_len:
            return l
        else:
            return list(l) + [pad_elem] * (total_len - orig_len)
    def padded_input_ids(self, total_len: int):
        return self.pad_list(self.input_ids, total_len, self.tokenizer.pad_token_id)

    def padded_attention_mask(self, total_len: int):
        return self.pad_list(self.attention_mask, total_len, 0)

    def padded_entity_start_positions(self, total_len: int):
        return self.pad_list(self.entity_start_positions, total_len, self.task_config_dict['slot_indices']['none'])
    def padded_entity_end_positions(self, total_len: int):
        return self.pad_list(self.entity_end_positions, total_len, self.task_config_dict['slot_indices']['none'])

    def class_tensor_start_positions(self, num_classes):
        res = torch.zeros(size=(1, len(self.entity_start_positions), num_classes), dtype=torch.float)

        for i in range(len(self.entity_start_positions)):
            res[0, i, self.entity_start_positions[i]] = 1.0

        return res


    def class_tensor_end_positions(self, num_classes):
        res = torch.zeros(size=(1, len(self.entity_end_positions), num_classes), dtype=torch.float)

        for i in range(len(self.entity_end_positions)):
            res[0, i, self.entity_end_positions[i]] = 1.0

        return res



def extract_data_elements_from_dataset(
        dataset: SantoDataset,
        tokenizer: ITCTokenizer,
        task_config_dict: Dict
):
    data_elements_list = list()
    for document, template_collection in dataset:
        data_elements_list.append(DataElement(
            document=document,
            template_collection=template_collection,
            tokenizer=tokenizer,
            task_config_dict=task_config_dict
        ))
    return data_elements_list
