from typing import List, Dict, Iterable, Sequence, Union, Tuple
from collections import namedtuple
from template_lib.data_classes.Entity import Entity
from template_lib.data_classes.Sentence import Sentence
from template_lib.data_classes.Document import Document
from extractive_approach.utils import ITCTokenizer

LabelledOffset = namedtuple('LabelledOffset', ['offset', 'label'])



def extract_labelled_offsets_from_label_index_seq(
        label_indices_sequence: Sequence[int],
        sentence_boundaries: Dict[int, Tuple[int, int]],
        label_indices_reverse_dict: Dict[int, str],
        neutral_label_index: int
) -> Dict[int, List[LabelledOffset]]:
    labelled_offsets_per_sentence = dict()

    for sentence_index in sentence_boundaries:
        start_offset, end_offset = sentence_boundaries[sentence_index]
        label_index_seq_sentence = label_indices_sequence[start_offset:end_offset + 1]

        labelled_offsets_per_sentence[sentence_index] = list()
        for offset, label_index in enumerate(label_index_seq_sentence):
            if label_index != neutral_label_index:
                labelled_offsets_per_sentence[sentence_index].append(LabelledOffset(
                    offset=offset,
                    label=label_indices_reverse_dict[label_index]
                ))

    return labelled_offsets_per_sentence


def join_start_end_labelled_offsets(
        labelled_offsets_start: Iterable[LabelledOffset],
        labelled_offsets_end: Iterable[LabelledOffset],
) -> List[Tuple[LabelledOffset, LabelledOffset]]:
    joined_labelled_offsets = list()

    for labelled_offset_start in labelled_offsets_start:
        filtered_end_offsets = list(filter(
            lambda x: x.label == labelled_offset_start.label and x.offset >= labelled_offset_start.offset,
            labelled_offsets_end
        ))

        if len(filtered_end_offsets) == 0:
            continue
        labelled_offset_end = sorted(filtered_end_offsets, key=lambda x: x.offset)[0]
        joined_labelled_offsets.append((labelled_offset_start, labelled_offset_end))

    return joined_labelled_offsets


def extract_entities_from_slot_indices_sequences(
        start_indices_sequence: List[int],
        end_indices_sequence: List[int],
        sentence_boundaries: Dict,
        slot_indices_reverse_dict: Dict[int, str],
        neutral_label_index: int,
        document: Document
) -> List[Entity]:
    assert len(start_indices_sequence) == len(end_indices_sequence)
    entities = list()

    labelled_offsets_start = extract_labelled_offsets_from_label_index_seq(
        label_indices_sequence=start_indices_sequence,
        sentence_boundaries=sentence_boundaries,
        label_indices_reverse_dict=slot_indices_reverse_dict,
        neutral_label_index=neutral_label_index
    )
    labelled_offsets_end = extract_labelled_offsets_from_label_index_seq(
        label_indices_sequence=end_indices_sequence,
        sentence_boundaries=sentence_boundaries,
        label_indices_reverse_dict=slot_indices_reverse_dict,
        neutral_label_index=neutral_label_index
    )

    # extract entities sentence-wise
    for sentence_index in sentence_boundaries:
        labelled_offsets_pairs = join_start_end_labelled_offsets(
            labelled_offsets_start=labelled_offsets_start[sentence_index],
            labelled_offsets_end=labelled_offsets_end[sentence_index]
        )

        # create entities from labelled offsets pairs
        for lo_start, lo_end in labelled_offsets_pairs:
            assert lo_start.label == lo_end.label
            assert lo_start.offset <= lo_end.offset

            entity = Entity()
            entity.set_label(lo_start.label)
            entity.set_start_pos(lo_start.offset)
            entity.set_end_pos(lo_end.offset)
            entity.set_sentence_index(sentence_index)
            entities.append(entity)

            # set entity tokens
            sentence_tokens = document.get_sentence_by_index(sentence_index).get_tokens()
            entity.set_tokens(
                sentence_tokens[entity.get_start_pos():entity.get_end_pos()+1]
            )


    return entities



if __name__ == '__main__':
    sentence = Sentence(tokens=['0', '1', '2', '3', '4', '5', '6', '7', '8'], index=0)

    slot_indices_reverse = {1: 'slot1', 2: 'slot2'}
    start_slot_indices_seq = [0, 0, 1, 2, 0]
    end_slot_indices_seq = [1, 2, 0, 0, 0]
