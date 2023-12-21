from typing import Sequence
import os
from transformers import LongformerTokenizer, PreTrainedTokenizer
from template_lib.santo.SantoDataset import SantoDataset, import_trial_ids


class IdentityTokenizer:
    def tokenize(self, pretokens: Sequence[str]):
        return pretokens


class ITCTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.start_of_sentence_token = self.tokenizer.bos_token#'<s>'
        self.start_of_sentence_token_id = self.tokenizer.bos_token_id
        self.end_of_sentence_token = self.tokenizer.eos_token#'</s>'
        self.end_of_sentence_token_id = self.tokenizer.eos_token_id
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id

    def tokenize(self, pretokens: Sequence[str]):
        is_split = (not isinstance(pretokens, str))
        token_ids = self.tokenizer.tokenize(text=pretokens, is_split_into_words=is_split)#['input_ids']#[1:-1]
        # if self.start_of_sentence_token_id is not None and token_ids[0] == self.start_of_sentence_token_id:
        #     token_ids = token_ids[1:]
        # if self.end_of_sentence_token_id is not None and token_ids[-1] == self.end_of_sentence_token_id:
        #     token_ids = token_ids[:-1]

        return token_ids#self.tokenizer.convert_ids_to_tokens(token_ids)

    def convert_tokens_to_ids(self, tokens: Sequence[str]):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, token_ids: Sequence[int]):
        return self.tokenizer.convert_ids_to_tokens(token_ids)


# def import_train_test_data_from_task_config(task_config_dict, tokenizer):
#     dataset_path = task_config_dict['dataset_path']
#     rel_train_ids_filename = task_config_dict['rel_train_ids_filename']
#     rel_test_ids_filename = task_config_dict['rel_test_ids_filename']
#
#     train_ids_filename = os.path.join(dataset_path, rel_train_ids_filename)
#     test_ids_filename = os.path.join(dataset_path, rel_test_ids_filename)
#
#     train_ids = import_trial_ids(train_ids_filename)
#     test_ids = import_trial_ids(test_ids_filename)
#
#     train_dataset = SantoDataset(dataset_path, train_ids, task_config_dict['disease_prefix'], 'admin', tokenizer)
#     test_dataset = SantoDataset(dataset_path, test_ids, task_config_dict['disease_prefix'], 'admin', tokenizer)
#
#     return train_dataset, test_dataset


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]