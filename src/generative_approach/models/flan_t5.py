from typing import Optional

from torch import Tensor
from transformers import LEDForConditionalGeneration, LEDTokenizer, LEDConfig, AutoModelForSeq2SeqLM, AutoTokenizer, \
    AutoConfig

from generative_approach.models.ed_model import EDModel


class FlanT5(EDModel):
    def __init__(self, model_path: str = None, model_name: str = "google/flan-t5-small", device: Optional[str] = None):
        super().__init__(device=device)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name if model_path is None else model_path, use_cache=False).to(self.device)
        self.model.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.max_encoder_position_embeddings = 100000  # Theoretically unlimited??
        self.max_decoder_position_embeddings = 100000

    def __call__(self, *args, **kwargs):
        kwargs.pop("global_attention_mask")
        nkwargs = dict()
        for k, v in kwargs.items():
            if isinstance(v, Tensor):
                nkwargs[k] = v.to(self.device)
            else:
                nkwargs[k] = v
        return self.model(*args, **nkwargs)

    # @property
    # def tokenizer(self):
    #     pass
    #
    # @property
    # def model(self):
    #     pass
    #
    # @property
    # def config(self):
    #     pass