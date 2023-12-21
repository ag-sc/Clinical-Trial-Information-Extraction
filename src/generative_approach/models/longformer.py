from typing import Optional

from torch import Tensor
from transformers import LEDForConditionalGeneration, LEDTokenizer, LEDConfig

from generative_approach.models.ed_model import EDModel


class Longformer(EDModel):
    def __init__(self, model_path: str = None, model_name: str = "allenai/led-base-16384", device: Optional[str] = None):
        super().__init__(device=device)
        self.model = LEDForConditionalGeneration.from_pretrained(
            model_name if model_path is None else model_path).to(self.device)
        self.tokenizer = LEDTokenizer.from_pretrained(model_name)
        self.config = LEDConfig.from_pretrained(model_name)
        self.max_encoder_position_embeddings = self.config.max_encoder_position_embeddings
        self.max_decoder_position_embeddings = self.config.max_decoder_position_embeddings

    def __call__(self, *args, **kwargs):
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