import sys
import itertools
from typing import Iterable, Tuple, List, Dict
import torch
from torch import nn
from transformers import LongformerModel, LongformerConfig, LongformerForSequenceClassification, PreTrainedModel, \
    PreTrainedTokenizer, LongformerTokenizer, T5Config, T5Tokenizer, T5Model, T5EncoderModel, LEDConfig, LEDTokenizer, \
    LEDModel

from extractive_approach.utils import ITCTokenizer
from generative_approach.models.longformer import Longformer
from template_lib.data_classes.Entity import Entity


class IntraTemplateCompModule(torch.nn.Module):
    def __init__(self, num_entity_classes, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = ITCTokenizer(tokenizer)
        self.encoder = model

        *_, last_layer = self.encoder.parameters()
        self.model_dim = len(last_layer)
        self.num_entity_classes = num_entity_classes

        # layers for entity prediction
        self.layer_entity_start_positions = nn.Sequential(
            nn.Linear(self.model_dim, num_entity_classes),
            # nn.Softmax(dim=2)
        )
        self.layer_entity_end_positions = nn.Sequential(
            nn.Linear(self.model_dim, num_entity_classes),
            # nn.Softmax(dim=2)
        )

        # layer for entity representation
        self.layer_entity_representation = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.ReLU()
        )

        # layer for entity pair compatibility
        self.layer_entity_pair_compatibility = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.ReLU(),
            nn.Linear(self.model_dim, 1),
        )

        # layers for template linking
        self.layer_source_template = nn.Linear(self.model_dim, self.model_dim)
        self.layer_target_template = nn.Linear(self.model_dim, self.model_dim)

    def compute_entity_representations(
            self,
            chunk: List[List[Entity]],
            sentence_boundaries: List[Dict[int, Tuple[int, int]]],
            last_hidden_state: torch.Tensor
    ):
        # compute global entity start/end offsets
        global_start_offsets, global_end_offsets = list(), list()
        for batch_idx, entities in enumerate(chunk):
            curr_global_start_offsets, curr_global_end_offsets = list(), list()
            for entity in entities:
                sentence_offset = sentence_boundaries[batch_idx][entity.get_sentence_index()][0]
                curr_global_start_offsets.append(sentence_offset + entity.get_start_pos())
                curr_global_end_offsets.append(sentence_offset + entity.get_end_pos())
            global_start_offsets.append(curr_global_start_offsets)
            global_end_offsets.append(curr_global_end_offsets)

        for batch_idx in range(len(chunk)):
            # gather hidden states of global start/end positions
            start_hidden_state = torch.index_select(
                input=last_hidden_state[batch_idx].unsqueeze(0),
                dim=1,
                index=torch.tensor(global_start_offsets[batch_idx], dtype=torch.int32).to(last_hidden_state.device)
            )

            end_hidden_state = torch.index_select(
                input=last_hidden_state[batch_idx].unsqueeze(0),
                dim=1,
                index=torch.tensor(global_end_offsets[batch_idx], dtype=torch.int32).to(last_hidden_state.device)
            )

            # res = torch.vstack([
            #     last_hidden_state[batch_idx].unsqueeze(0)[:, global_start_offsets[batch_idx][ent_idx]:global_end_offsets[batch_idx][ent_idx]+1].sum(dim=1)
            #     for ent_idx in range(len(global_start_offsets[batch_idx]))
            # ]).unsqueeze(0) if len(global_start_offsets[batch_idx]) > 0 else torch.zeros((last_hidden_state.shape[0], 0, last_hidden_state.shape[2]))
            # res = res.to(last_hidden_state.device)

            # compute entity representations
            entity_representations = self.layer_entity_representation(start_hidden_state + end_hidden_state)  # res
            for i in range(len(global_start_offsets[batch_idx])):
                chunk[batch_idx][i].vector_representation = entity_representations[0][i]

    def compute_entity_compatibilities(self, chunk_entity_pairs: List[List[Tuple[Entity, Entity]]]):
        res = []
        for entity_pairs in chunk_entity_pairs:
            entity_vectors1 = [entity_pair[0].vector_representation for entity_pair in entity_pairs]
            entity_vectors2 = [entity_pair[1].vector_representation for entity_pair in entity_pairs]
            entity_pair_representations = torch.stack(entity_vectors1) + torch.stack(entity_vectors2)
            res.append(self.layer_entity_pair_compatibility(entity_pair_representations))
        return res


class ITCLongformer(IntraTemplateCompModule):
    def __init__(self, num_entity_classes, model_name='allenai/longformer-base-4096'):
        config = LongformerConfig.from_pretrained(model_name)
        tokenizer = LongformerTokenizer.from_pretrained(model_name)
        # encoder
        model = LongformerModel.from_pretrained(model_name, config=config)
        model.encoder.gradient_checkpointing = True
        super().__init__(num_entity_classes=num_entity_classes, model=model, tokenizer=tokenizer)

class ITCLED(IntraTemplateCompModule):
    def __init__(self, num_entity_classes, model_name="allenai/led-base-16384"):
        config = LEDConfig.from_pretrained(model_name)
        tokenizer = LEDTokenizer.from_pretrained(model_name)
        # encoder
        LEDModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        model = LEDModel.from_pretrained(model_name, config=config)
        model.encoder.gradient_checkpointing = True
        super().__init__(num_entity_classes=num_entity_classes, model=model.encoder, tokenizer=tokenizer)

class ITCFlanT5(IntraTemplateCompModule):
    def __init__(self, num_entity_classes, model_name="google/flan-t5-small"):
        config = T5Config.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        # encoder
        model = T5EncoderModel.from_pretrained(model_name, config=config)
        model.encoder.gradient_checkpointing = True
        super().__init__(num_entity_classes=num_entity_classes, model=model, tokenizer=tokenizer)
