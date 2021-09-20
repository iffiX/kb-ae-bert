from typing import Dict, Any
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
)
from ..utils.settings import model_cache_dir, proxies
import torch as t
import torch.nn as nn


class ExtendVocab(nn.Module):
    def __init__(
        self,
        base_model,
        base_type: str = "bert-base-uncased",
        extend_mode: str = "ratio_mix",
        extend_config: Dict[str, Any] = None,
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
        super().__init__()
        self.base = base_model.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            return_dict=True,
            **base_configs,
        )
        if extend_mode == "ratio_mix_learnable":
            self.mix_weight = nn.Parameter(t.rand(self.base.config.hidden_size))
        elif extend_mode in ("ratio_mix", "replace"):
            pass
        else:
            raise ValueError(f"Unknown extend_mode {extend_mode}")
        self.base_type = base_type
        self.model_name = base_type.split("-")[0].lower()
        self.extend_mode = extend_mode
        self.extend_config = extend_config or {}


class ExtendVocabForQA(ExtendVocab):
    def __init__(
        self,
        base_type: str = "bert-base-uncased",
        extend_mode: str = "ratio_mix",
        extend_config: Dict[str, Any] = None,
        **base_configs,
    ):
        # DOC INHERITED
        super().__init__(
            AutoModelForQuestionAnswering,
            base_type=base_type,
            extend_mode=extend_mode,
            extend_config=extend_config,
            **base_configs,
        )

    def forward(
        self,
        token_ids,
        extend_tokens,
        extend_embeds,
        attention_mask=None,
        token_type_ids=None,
        start_positions=None,
        end_positions=None,
    ):
        """
        Args:
            token_ids: Token ids, Long Tensor of shape
                (batch_size, sequence_length,), set `return_tensors`
                in your tokenizer to get this.
            extend_tokens: Extended tokens, 0 is not extended, 1 is extended,
                LongTensor of shape (batch_size, sequence_length)
            extend_embeds: Extended embedding, FloatTensor of shape
                (batch_size, sequence_length, hidden_size)
            attention_mask: Attention mask, 0 or 1, FloatTensor of shape
                (batch_size, sequence_length)
            token_type_ids: Type id of tokens (segment embedding), 0 or 1,
                LongTensor of shape (batch_size, sequence_length)
            start_positions: A value in [0, sequence_length) indicating which
                index is the start position of the answer, LongTensor of shape
                (batch_size,)
            end_positions: A value in [0, sequence_length) indicating which
                index is the end position of the answer, LongTensor of shape
                (batch_size,)

        Returns:
            loss: CrossEntropyLoss of predicted start/ends and given start/ends. None
                is start_positions or end_positions is not given, otherwise FloatTensor.
            start_logits FloatTensor of shape (batch_size, sequence_length),
                Span-start scores (before SoftMax).
            end_logits FloatTensor of shape (batch_size, sequence_length),
                Span-end scores (before SoftMax).
        """
        # get token embeddings
        token_embeds = getattr(self.base, self.model_name).embeddings.word_embeddings(
            token_ids
        )
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
                + extend_tokens.unsqueeze(-1) * (1 - weight) * extend_embeds
            )
        elif self.extend_mode == "replace":
            token_embeds = t.where(
                extend_tokens.unsqueeze(-1) == 1, extend_embeds, token_embeds
            )
        out = self.base(
            inputs_embeds=token_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        return (
            None if start_positions is None or end_positions is None else out.loss,
            out.start_logits,
            out.end_logits,
        )


class ExtendVocabForSequenceClassification(ExtendVocab):
    def __init__(
        self,
        base_type: str = "bert-base-uncased",
        extend_mode: str = "ratio_mix",
        extend_config: Dict[str, Any] = None,
        **base_configs,
    ):
        # DOC INHERITED
        super().__init__(
            AutoModelForSequenceClassification,
            base_type=base_type,
            extend_mode=extend_mode,
            extend_config=extend_config,
            **base_configs,
        )

    def forward(
        self,
        token_ids,
        extend_tokens,
        extend_embeds,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        """
        Args:
            token_ids: Token ids, Long Tensor of shape
                (batch_size, sequence_length,), set `return_tensors`
                in your tokenizer to get this.
            extend_tokens: Extended tokens, 0 is not extended, 1 is extended,
                LongTensor of shape (batch_size, sequence_length)
            extend_embeds: Extended embedding, FloatTensor of shape
                (batch_size, sequence_length, hidden_size)
            attention_mask: Attention mask, 0 or 1, FloatTensor of shape
                (batch_size, sequence_length)
            token_type_ids: Type id of tokens (segment embedding), 0 or 1,
                LongTensor of shape (batch_size, sequence_length)
            labels: torch.LongTensor of shape (batch_size,), used for computing
                the sequence classification/regression loss. Indices should be in
                [0, ..., config.num_labels - 1]. If config.num_labels == 1 a regression
                loss is computed (Mean-Square loss), If config.num_labels > 1 a
                classification loss is computed (Cross-Entropy).

        Returns:
            loss: CrossEntropyLoss of predicted start/ends and given start/ends. None
                if labels is not given, otherwise FloatTensor.
            logits: Classification (or regression if config.num_labels==1) scores
                (before SoftMax).
        """
        # get token embeddings
        token_embeds = (
            getattr(self.base, self.model_name)
            .embeddings.word_embeddings(token_ids)
            .detach()
        )
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
                + extend_tokens.unsqueeze(-1) * (1 - weight) * extend_embeds
            )
        elif self.extend_mode == "replace":
            token_embeds = t.where(
                extend_tokens.unsqueeze(-1) == 1, extend_embeds, token_embeds
            )
        out = self.base(
            inputs_embeds=token_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        return (
            None if labels is None else out.loss,
            out.logits,
        )
