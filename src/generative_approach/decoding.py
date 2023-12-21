import math
import sys
from collections import deque

import numpy as np

from generative_approach.data_handling import *
from generative_approach.grammar import Grammar
from template_lib.data_handling import import_train_test_data_from_task_config

DEBUG = False


def create_token_mask(logits: torch.Tensor, token_ids: Iterable[int]):
    mask = np.ones(logits.shape)
    #mask = torch.ones_like(logits)

    #for token_id in token_ids:
    #    mask[token_id] = 0
    mask[list(token_ids)] = 0

    mask = mask * (-1e9)

    return torch.from_numpy(mask).to(logits.device)


class SlotCardinalityTracker:
    def __init__(self, template_names):
        self._template_names = template_names
        self._cardinality_dicts_stack = list()

    def reset(self):
        self._cardinality_dicts_stack = list()

    def get_stack_size(self):
        return len(self._cardinality_dicts_stack)

    def add_token(self, token):
        if is_start_token(token):
            decoded_token = decode_start_token(token)

            if decoded_token in self._template_names:
                self._cardinality_dicts_stack.append(dict())
            else:
                top_dict = self._cardinality_dicts_stack[-1]
                cardinality = top_dict.setdefault(decoded_token, 0)
                top_dict[decoded_token] = cardinality + 1
        elif is_end_token(token):
            decoded_token = decode_end_token(token)

            if decoded_token in self._template_names:
                self._cardinality_dicts_stack.pop(-1)

    def get_saturated_slots(
            self,
            max_slots_cardinalities: Dict[str, int],
            functional_only: bool = True
    ) -> List[str]:
        if len(self._cardinality_dicts_stack) == 0:
            return list()

        cardinality_dict = self._cardinality_dicts_stack[-1]
        saturated_slots = list()

        for slot_name in cardinality_dict:
            if slot_name not in max_slots_cardinalities:
                continue
            if functional_only and max_slots_cardinalities[slot_name] > 1:
                continue

            if cardinality_dict[slot_name] >= max_slots_cardinalities[slot_name]:
                saturated_slots.append(slot_name)

        return saturated_slots

    def print_stack(self):
        print('-------')
        for i, cardinality_dict in enumerate(self._cardinality_dicts_stack):
            for key, value in cardinality_dict.items():
                print('\t' * (i) + f'{key}: {value}')


class DecodingState:
    def __init__(self, grammar: Grammar, template_names: List[str]):
        self.grammar = grammar
        self.slot_cardinality_tracker = SlotCardinalityTracker(template_names)
        self.grammar_rules_stack = list()
        self.token_ids = list()
        self.logits = list()
        self.logits_sum = 0.0
        self.free_text_tokens = list()
        self.past_key_values = None

    def print_rules_stack(self):
        if len(self.grammar_rules_stack) == 0:
            print('empty')
        else:
            for i, symbols_deque in enumerate(self.grammar_rules_stack):
                print('\t' * (i + 1) + str(list(symbols_deque)))

    def add_token_prediction(
            self,
            token_id: int,
            token: str,
            logit: float,
            past_key_values=None
    ):
        self.token_ids.append(token_id)
        self.logits.append(logit)
        self.logits_sum += logit
        self.past_key_values = past_key_values

        self.slot_cardinality_tracker.add_token(token)

        current_symbols_deque = self.grammar_rules_stack[-1]
        current_symbol = current_symbols_deque[0]

        # update grammar rules stack
        if current_symbol == self.grammar.pointer_symbol:
            self.free_text_tokens.append(token)
            pointer_termination_symbol = current_symbols_deque[1]

            if token == pointer_termination_symbol:
                # remove pointer symbol
                self.remove_grammar_symbol_from_stack_top()

                # remove pointer termination symbol
                self.remove_grammar_symbol_from_stack_top()

                # clear free text prediction list
                self.free_text_tokens.clear()
        elif current_symbol in self.grammar.get_nonterminal_symbols():
            next_grammar_rule = self.grammar.get_rule_option_by_start_symbol(current_symbol, token)

            # remove current nonterminal
            self.remove_grammar_symbol_from_stack_top()

            self.grammar_rules_stack.append(deque(next_grammar_rule.get_symbols()))

            # remove first symbol of new grammar rule
            self.remove_grammar_symbol_from_stack_top()
        else:
            self.remove_grammar_symbol_from_stack_top()

    def remove_empty_deques_from_stack(self):
        while len(self.grammar_rules_stack) > 0:
            top_deque = self.grammar_rules_stack[-1]

            if len(top_deque) == 0:
                del self.grammar_rules_stack[-1]
                continue
            else:
                return

    def get_current_grammar_symbol(self):
        if len(self.grammar_rules_stack) == 0:
            raise IndexError('No grammar symbols left')

        return self.grammar_rules_stack[-1][0]

    def remove_grammar_symbol_from_stack_top(self):
        self.remove_empty_deques_from_stack()

        if len(self.grammar_rules_stack) == 0:
            raise IndexError('No grammar symbols left')

        self.grammar_rules_stack[-1].popleft()
        self.remove_empty_deques_from_stack()

    def get_next_possible_tokens(
            self,
            subseq_manager: DocumentSubsequenceManager = None,
            max_slots_cardinalities: Dict[str, int] = None,
            constrain_only_functional_slots: bool = False,
            used_slots: List[str] = None,
    ) -> Set[str]:
        # nothing to decode case
        if len(self.grammar_rules_stack) == 0:
            return set()

        current_symbols_deque = self.grammar_rules_stack[-1]
        current_symbol = current_symbols_deque[0]

        if current_symbol == self.grammar.pointer_symbol:
            # there has to be another symbol right next to pointer symbol
            assert len(current_symbols_deque) > 1
            next_possible_tokens = {current_symbols_deque[1]}

            # free text tokens
            next_possible_tokens.update(subseq_manager.get_next_possible_tokens(prefix=self.free_text_tokens))
        elif current_symbol in self.grammar.get_nonterminal_symbols():
            next_possible_tokens = set(self.grammar.get_rule_start_symbols(current_symbol))
        else:
            next_possible_tokens = {current_symbols_deque[0]}

        # incorporate max slots cardinalities
        if max_slots_cardinalities is not None:
            saturated_slots = self.slot_cardinality_tracker.get_saturated_slots(
                max_slots_cardinalities=max_slots_cardinalities,
                functional_only=constrain_only_functional_slots
            )
            next_possible_tokens -= {create_start_token(slot_name) for slot_name in saturated_slots}

        if used_slots is not None and current_symbol != self.grammar.pointer_symbol:
            valid_names = {create_start_token(slot_name) for slot_name in used_slots}.union({create_end_token(slot_name) for slot_name in used_slots})
            # if len(next_possible_tokens - valid_names) > 0:
            #     print("!!!", next_possible_tokens - valid_names)
            next_possible_tokens.intersection_update(valid_names)

        return next_possible_tokens


class DecodingStateSuccessor:
    def __init__(
            self,
            decoding_state: DecodingState,
            token_id: int = None,
            token: str = None,
            token_logit: float = None,
            start_pos_logit: float = None,
            end_pos_logit: float = None,
            past_key_values=None
    ):
        self.decoding_state = decoding_state
        self.token_id = token_id
        self.token = token
        self.token_logit = token_logit
        self.start_pos_logit = start_pos_logit
        self.end_pos_logit = end_pos_logit
        self.past_key_values = past_key_values

    def compute_token_logits_score(self) -> float:
        unnormalized_score = self.decoding_state.logits_sum
        normalization_factor = len(self.decoding_state.logits)

        # if decoding has not finished, incorporate most recent prediction
        if self.token is not None:
            unnormalized_score += self.token_logit
            normalization_factor += 1

        return unnormalized_score / normalization_factor

    def compute_pointer_network_logits_score(self) -> float:
        pass

    def compute_score(self, include_pointer_network=False) -> float:
        score = self.compute_token_logits_score()
        if include_pointer_network:
            score += self.compute_pointer_network_logits_score()
        return score

    def next_decoiding_state(self) -> DecodingState:
        if self.token is not None:
            self.decoding_state.add_token_prediction(
                token_id=self.token_id,
                token=self.token,
                logit=self.token_logit,
                past_key_values=self.past_key_values
            )
        return self.decoding_state


class EncoderDecoderPrediction:
    def __init__(self):
        self.token_logits: torch.Tensor = None
        self.encoder_last_hidden_state: torch.Tensor = None
        self.decoder_last_hidden_state: torch.Tensor = None
        self.start_pos_logits: torch.Tensor = None
        self.end_pos_logits: torch.Tensor = None
        self.past_key_values = None


class EDPredictionProxy:
    def __init__(self, led_model):
        self._model = led_model

    def predict(
            self,
            input_ids: List[int],
            output_ids: List[int],
            encoder_last_hidden_state: torch.Tensor = None,
            past_key_values=None
    ) -> EncoderDecoderPrediction:
        torch.cuda.empty_cache()
        assert len(input_ids) > 0 and len(output_ids) > 0
        device = self._model.device

        # prepare inputs
        input_ids_tensor = torch.tensor([input_ids])
        if past_key_values is not None:
            output_ids = [output_ids[-1]]

        pos = len(output_ids) - 1
        global_attention_mask = torch.zeros_like(input_ids_tensor)
        global_attention_mask[:, 0] = 1

        if encoder_last_hidden_state is None:
            input_ids = input_ids_tensor.to(device)
            encoder_outputs = None
        else:
            input_ids = None
            encoder_outputs = (encoder_last_hidden_state.to(device),)

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                decoder_input_ids=torch.tensor([output_ids]).to(device),
                global_attention_mask=global_attention_mask.to(device),
                encoder_outputs=encoder_outputs,
                output_hidden_states=True,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True,
            )

        encoder_decoder_prediction = EncoderDecoderPrediction()
        encoder_decoder_prediction.token_logits = outputs.logits[0, pos, :]
        encoder_decoder_prediction.encoder_last_hidden_state = outputs.encoder_last_hidden_state
        encoder_decoder_prediction.decoder_last_hidden_state = outputs.decoder_hidden_states[-1]
        encoder_decoder_prediction.past_key_values = outputs.past_key_values

        return encoder_decoder_prediction


class Decoder:
    def __init__(
            self,
            grammar: Grammar,
            tokenizer: TemplateGenerationTokenizer,
            prediction_proxy,
            template_names: List[str]
    ):
        self._grammar = grammar
        self._tokenizer = tokenizer
        self._prediction_proxy = prediction_proxy
        self._template_names = template_names

        self._input_ids: List[int] = None
        self._encoder_last_hidden_state: torch.Tensor = None

        self._start_of_sentence_token_id = None
        self._end_of_sentence_token_id = None
        self._pad_token_id = None
        if tokenizer.start_of_sentence_token is not None:
            self._start_of_sentence_token_id = tokenizer.convert_tokens_to_ids(
                tokens=[tokenizer.start_of_sentence_token]
            )[0]
        if tokenizer.end_of_sentence_token is not None:
            self._end_of_sentence_token_id = tokenizer.convert_tokens_to_ids(
                tokens=[tokenizer.end_of_sentence_token]
            )[0]
        if tokenizer.pad_token is not None:
            self._pad_token_id = tokenizer.convert_tokens_to_ids(
                tokens=[tokenizer.pad_token]
            )[0]

    def generate_decoding_state_successors(
            self,
            decoding_state: DecodingState,
            num_successors: int,
            use_pointer_network=False
    ) -> List[DecodingStateSuccessor]:
        assert self._input_ids is not None or self._encoder_last_hidden_state is not None
        if num_successors < 1:
            raise ValueError(f'num_successors has to be positive integer: {num_successors}')

        # check if decoding has finished
        if len(decoding_state.grammar_rules_stack) == 0:
            return [DecodingStateSuccessor(decoding_state=decoding_state)]

        # predict distribution of next token
        encoder_decoder_prediction = self._prediction_proxy.predict(
            input_ids=self._input_ids,
            output_ids=decoding_state.token_ids,
            encoder_last_hidden_state=self._encoder_last_hidden_state,
            past_key_values=decoding_state.past_key_values
        )

        # cache last hidden state of encoder
        self._encoder_last_hidden_state = encoder_decoder_prediction.encoder_last_hidden_state

        if use_pointer_network:
            pass
        else:
            # TODO apply next possible token mask
            token_logits: torch.Tensor = encoder_decoder_prediction.token_logits
            token_probs = torch.nn.Softmax(token_logits)
            topk_result = torch.topk(token_probs, num_successors)

            # generate k most likely successors
            successors = list()
            for index, value in zip(topk_result.indices.tolist(), topk_result.values.tolist()):
                logit = math.log(value)
                [token] = self._tokenizer.tokenize([index])

                successors.append(DecodingStateSuccessor(
                    decoding_state=decoding_state,
                    token_id=index,
                    token=token,
                    token_logit=logit
                ))

    def decode_document_greedy(
            self,
            document: Document,
            start_symbol: str,
            max_len=1020,
            max_slots_cardinalities: Dict[str, int] = None,
            constrain_only_functional_slots=False,
            used_slots: List[str] = None
    ):
        if used_slots is None:
            used_slots = []
        # create list of token ids for each sentence in document
        sentences_token_ids = [self._tokenizer.convert_tokens_to_ids(sentence.get_tokens())
                               for sentence in document.get_sentences()]
        sos = [self._start_of_sentence_token_id] if self._start_of_sentence_token_id is not None else []
        eos = [self._end_of_sentence_token_id] if self._end_of_sentence_token_id is not None else []
        input_ids = [sos + sentence_token_ids + eos
                     for sentence_token_ids in sentences_token_ids]

        input_ids = list(chain(*input_ids))

        # save last encode hidden state for better decoding speed
        encoder_last_hidden_state = None

        # create subseq manager
        subseq_manager = DocumentSubsequenceManager([sentence.get_tokens() for sentence in document.get_sentences()])

        # initial decoding sate
        decoding_state = DecodingState(grammar=self._grammar, template_names=self._template_names)
        if self._start_of_sentence_token_id is not None:
            decoding_state.token_ids.append(self._start_of_sentence_token_id)
        elif isinstance(self._prediction_proxy._model, FlanT5):
            decoding_state.token_ids.append(self._pad_token_id)
        decoding_state.grammar_rules_stack.append(deque([start_symbol]))

        # predict tokens
        while len(decoding_state.token_ids) < max_len and len(decoding_state.grammar_rules_stack) > 0:
            if DEBUG:
                print('--- grammar rules stack ---')
                decoding_state.print_rules_stack()
                print()

            # predict next token
            encoder_decoder_prediction = self._prediction_proxy.predict(
                input_ids=input_ids,
                output_ids=decoding_state.token_ids,
                encoder_last_hidden_state=encoder_last_hidden_state,
                past_key_values=decoding_state.past_key_values
            )
            encoder_last_hidden_state = encoder_decoder_prediction.encoder_last_hidden_state

            if DEBUG:
                print('--- next possible tokens ---')
            next_possible_tokens = decoding_state.get_next_possible_tokens(
                subseq_manager=subseq_manager,
                max_slots_cardinalities=max_slots_cardinalities,
                constrain_only_functional_slots=constrain_only_functional_slots,
                used_slots=used_slots
            )
            if 2 in self._tokenizer.convert_tokens_to_ids(list(next_possible_tokens)):
                pass
            next_possible_token_ids = self._tokenizer.convert_tokens_to_ids(list(next_possible_tokens))
            if DEBUG:
                print(next_possible_tokens)
                print()

            # mask invalid tokens
            mask = create_token_mask(
                logits=encoder_decoder_prediction.token_logits,
                token_ids=next_possible_token_ids
            )
            logits = encoder_decoder_prediction.token_logits + mask

            # estimate predicted token by argmax
            token_id = logits.argmax(-1).item()
            [token] = self._tokenizer.convert_ids_to_tokens([token_id])

            # add token prediction
            if DEBUG:
                print(logits.max(-1).values.item())
            decoding_state.add_token_prediction(
                token_id=token_id,
                token=token,
                logit=logits.max(-1).values.item(),
                past_key_values=encoder_decoder_prediction.past_key_values
            )
            if DEBUG:
                print('-------------------------')

        print("Final:", document.get_id(), decoding_state.token_ids + eos)
        print("Final2:", document.get_id(), self._tokenizer.convert_ids_to_tokens(decoding_state.token_ids + eos))
        # convert predicted token ids to tokens and return
        return self._tokenizer.convert_ids_to_tokens(decoding_state.token_ids + eos)

    def decode_document(
            self,
            document: Document,
            start_symbol: str,
            beam_size: int,
            max_len=1020,
            max_slots_cardinalities: Dict[str, int] = None,
            constrain_only_functional_slots=False
    ):
        # clear caching values of prev document
        self._encoder_last_hidden_state = None

        # set constraints of decoding process
        self._max_slots_cardinalities = max_slots_cardinalities
        self._constrain_only_functional_slots = constrain_only_functional_slots


# arg1: task config file name
# arg2: special tokens json file name
# arg3: directory of trained model
# arg4: grammar file
# arg5: json filename for decoded strings
# arg6: max cardinalities json file
if __name__ == '__main__':
    task_config_dict = import_task_config(sys.argv[1])
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    # device = "cpu" #torch.device("cpu")

    model_name = task_config_dict['model_name']
    model_class = get_model_class(model_name)
    model = model_class(model_path=sys.argv[3], model_name=model_name, device=device_str)

    temp_generation_tokenizer = TemplateGenerationTokenizer(
        tokenizer=model.tokenizer,
        json_filename=sys.argv[2]
    )
    identity_tokenizer = IdentityTokenizer()

    grammar = Grammar(sys.argv[4])
    if DEBUG:
        grammar.print_out()

    decoder = Decoder(
        grammar=grammar,
        tokenizer=temp_generation_tokenizer,
        prediction_proxy=EDPredictionProxy(model),
        template_names=list(task_config_dict['slots_ordering_dict'].keys())
    )

    training_set, test_dataset = import_train_test_data_from_task_config(
        task_config_dict,
        temp_generation_tokenizer  # temp_generation_tokenizer ################### modify
    )

    dataset_to_use = test_dataset

    max_slots_cardinalities = None
    if os.path.isfile(sys.argv[6]):
        with open(sys.argv[6]) as fp:
            max_slots_cardinalities = json.load(fp)

    # decode documents in test set
    result_dict = dict()
    for document, _ in dataset_to_use:#TODO!
        if DEBUG:
            print(f'decoding document {document.get_id()}')
        output_str = decoder.decode_document_greedy(
            document=document,
            start_symbol='#PUBLICATION_HEAD',
            max_len=1020,
            max_slots_cardinalities=max_slots_cardinalities  # None
        )
        result_dict[document.get_id()] = output_str
        if DEBUG:
            print('======================================')

    # save output strings to file
    fp = open(sys.argv[5], 'w')
    json.dump(result_dict, fp)
    fp.close()
