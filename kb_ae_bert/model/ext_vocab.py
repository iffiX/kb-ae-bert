from typing import Dict, Any
from urllib.parse import urlparse
from transformers import AutoModelForQuestionAnswering
from ..utils.settings import model_cache_dir, http_proxy
import torch as t
import torch.nn as nn


class ExtendVocabForQA(nn.Module):
    def __init__(
        self,
        base_type: str = "bert-base-uncased",
        extend_config: Dict[str, Any] = None,
        extend_mode: str = "ratio_mix",
        **base_configs,
    ):
        """
        Args:
            base_type: Base type of model used to initialize the AutoModel.
            extend_config: Configs for `extend_mode`.
            extend_mode:
                "ratio_mix" for mixing model default tokens and new embedding, by
                a fixed ratio alpha, requires: "alpha": float in config.
                "ratio_mix_learnable" for mixing model default tokens and new
                embedding, by a learnable ratio per dim, requires: None.
                "replace" for replacing each sub token embedding with the new embedding.
            **base_configs: Additional configs passed to AutoModel.
        """
        self.base = AutoModelForQuestionAnswering.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            proxies={"http": urlparse(http_proxy).path},
            return_dict=True,
            **base_configs,
        )
        if extend_mode == "ratio_mix_learnable":
            self.mix_weight = nn.Parameter(t.rand(self.base.config.hidden_size))
        elif extend_mode in ("ratio_mix", "replace"):
            pass
        else:
            raise ValueError(f"Unknown extend_mode {extend_mode}")
        self.extend_mode = extend_mode
        self.extend_config = extend_config
        super().__init__()

    def forward(
        self,
        token_embeds,
        extend_tokens,
        extend_embeds,
        start_positions=None,
        end_positions=None,
    ):
        """
        Args:
            token_embeds: Token embeddings, Float Tensor of shape
                (batch_size, sequence_length, hidden_size), set `return_tensors`
                in your tokenizer to get this.
            extend_tokens: Extended tokens, 0 is not extended, 1 is extended,
                LongTensor of shape (batch_size, sequence_length)
            extend_embeds: Extended embedding, FloatTensor of shape
                (batch_size, sequence_length, hidden_size)
            start_positions: A value in 0, 1 indicating whether current position is
                a start position of the answer, LongTensor of shape (batch_size,)
            end_positions: A value in 0, 1 indicating whether current position is
                an end position of the answer, LongTensor of shape (batch_size,)

        Returns:
            loss: CrossEntropyLoss of predicted start/ends and given start/ends. None
                of FloatTensor.
            start_logits FloatTensor of shape (batch_size, sequence_length),
                Span-start scores (before SoftMax).
            end_logits FloatTensor of shape (batch_size, sequence_length),
                Span-end scores (before SoftMax).
        """
        if self.extend_mode == "ratio_mix":
            alpha = self.extend_config["alpha"]
            token_embeds = (
                token_embeds * alpha
                + extend_tokens.unsqueeze(-1) * (1 - alpha) * extend_embeds
            )
        elif self.extend_mode == "ratio_mix_learnable":
            weight = self.mix_weight.view(1, 1, -1)
            token_embeds = (
                token_embeds * weight
                - +extend_tokens.unsqueeze(-1) * (1 - weight) * extend_embeds
            )
        elif self.extend_mode == "replace":
            token_embeds = t.where(
                extend_tokens.unsqueeze(-1) == 1, extend_embeds, token_embeds
            )
        out = self.base(
            input_embeds=token_embeds,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        return (
            None if start_positions is None or end_positions is None else out.loss,
            out.start_logits,
            out.end_logits,
        )
