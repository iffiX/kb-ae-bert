from typing import Tuple
from urllib.parse import urlparse
from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification
from ..utils.settings import model_cache_dir, http_proxy
import torch as t
import torch.nn as nn
import numpy as np


class KBMaskedLMEncoder(nn.Module):
    def __init__(
        self,
        relation_size: int,
        base_type: str = "bert-base-uncased",
        relation_mode: str = "concatenation",
        mlp_hidden_size: Tuple[int] = (),
        **base_configs,
    ):
        """
        For entity and its fixed depth relation graph encoding.

        Args:
            relation_size: Number of relations, related to the output size of MLP.
            base_type: Base type of model used to initialize the AutoModel.
            relation_mode:
                "concatenation_mlp" for using [entity1, entity2] as MLP input, directly
                predict score for each relation.
                "subtraction" for using entity2 - entity1 and compare it to internal
                direction embedding.
            mlp_hidden_size: Size of hidden layers, default is not using hidden layers
            **base_configs: Additional configs passed to AutoModel.
        """
        self.base = AutoModelForMaskedLM.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            proxies={"http": urlparse(http_proxy).path},
            output_hidden_states=True,
            return_dict=True,
            **base_configs,
        )

        # relation head on [cls]
        mlp = []
        if relation_mode == "concatenation_mlp":
            input_size = self.base.config.hidden_size * 2
            for size in list(mlp_hidden_size) + [relation_size]:
                mlp.append(nn.Linear(input_size, size))
                input_size = size
            self.mlp = nn.Sequential(*mlp)
        elif relation_mode == "subtraction":
            H = np.random.randn(relation_size, self.base.config.hidden_size)
            u, s, vh = np.linalg.svd(H, full_matrices=False)
            self.relation_embedding = t.tensor(np.matmul(u, vh))
        else:
            raise ValueError(f"Unknown relation_mode {relation_mode}")

        self.relation_mode = relation_mode
        super().__init__()

    @property
    def hidden_size(self):
        return self.base.config.hidden_size

    def compute_relation(self, tokens1, tokens2):
        """
        Args:
            tokens1: Token ids, LongTensor of shape (batch_size, sequence_length)
            tokens2: Token ids, LongTensor of shape (batch_size, sequence_length)

        Returns:
            Relation between the two masked context vocabulary, which are logits
            before softmax, FloatTensor of shape (batch_size, relation_size).
        """
        cls1 = self.__call__(tokens1)[0]
        cls2 = self.__call__(tokens2)[0]
        if self.relation_mode == "concatenation_mlp":
            return self.mlp(t.cat((cls1, cls2), dim=1))
        elif self.relation_mode == "subtraction":
            diff = cls2 - cls1
            # batched dot product
            dot = t.einsum("ij,kj->ik", diff, self.relation_embedding)
            # use absolute value of dot product to compare similarity
            sim = t.abs(dot)
            return sim

    def forward(self, tokens, labels=None):
        """
        Args:
            tokens: Token ids, LongTensor of shape (batch_size, sequence_length)
            labels: Output token labels, value is in [0, vocab_size), LongTensor
                of shape (batch_size, sequence_length)
            Proposal labels: Output proposal labels, 1 indicates that the input token
                is the target entity (masked in input). LongTensor
                of shape (batch_size, sequence_length)
        Returns:
            cls embedding: Float tensor of shape (batch_size, hidden_size)
            proposal: LongTensor of shape (batch_size, sequence_length)
            loss: CrossEntropyLoss of predicted labels and given labels, None if
                `labels` is not set, otherwise a float tensor of shape (1,).
            proposal loss: CrossEntropyLoss of predicted proposals and given proposals,
                None if `proposal_labels` is not set, otherwise a float tensor
                of shape (1,).
            logits: Prediction scores of the language modeling head
                (scores for each vocabulary token before SoftMax),
                FloatTensor of shape (batch_size, sequence_length, vocab_size),

        """
        out = self.base(input_ids=tokens, labels=labels)

        return (
            out.hidden_states[-1][:, 0, :],
            None if labels is None else out.loss,
            out.logits,
        )


class KBEntityDetector(nn.Module):
    def __init__(
        self, base_type: str = "bert-base-uncased", **base_configs,
    ):
        """
        For entity detection.

        Args:
            base_type: Base type of model used to initialize the AutoModel.
            **base_configs: Additional configs passed to AutoModel.
        """
        self.base = AutoModelForTokenClassification.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            proxies={"http": urlparse(http_proxy).path},
            return_dict=True,
            num_labels=2,
            **base_configs,
        )
        super().__init__()

    def forward(self, tokens, proposal_labels=None):
        """
        Args:
            tokens: Token ids, LongTensor of shape (batch_size, sequence_length)
            proposal_labels: Output proposal labels, 1 indicates that the input token
                is the target entity (masked in input). LongTensor
                of shape (batch_size, sequence_length)
        Returns:
            proposal: LongTensor of shape (batch_size, sequence_length)
            loss: CrossEntropyLoss of predicted proposals and given proposals,
                None if `proposal_labels` is not set, otherwise a float tensor
                of shape (1,).
            logits: Prediction scores of the language modeling head
                (scores for each vocabulary token before SoftMax),
                FloatTensor of shape (batch_size, sequence_length, 2),
        """
        out = self.base(input_ids=tokens, token_type_ids=proposal_labels)

        return (
            t.argmax(out.logits, dim=2),
            None if proposal_labels is None else out.loss,
            out.logits,
        )
