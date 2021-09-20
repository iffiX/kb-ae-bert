from typing import List
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
)
from ..utils.settings import model_cache_dir, proxies
from ..utils.token import get_context_of_masked
import random
import torch as t
import torch.nn as nn
import numpy as np


class KBMaskedLMEncoder(nn.Module):
    def __init__(
        self,
        relation_size: int,
        base_type: str = "bert-base-uncased",
        relation_mode: str = "concatenation",
        mlp_hidden_size: List[int] = None,
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
        super().__init__()
        self.base = AutoModelForMaskedLM.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            output_hidden_states=True,
            return_dict=True,
            **base_configs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_type, cache_dir=model_cache_dir, proxies=proxies,
        )
        self._pad_id = tokenizer.pad_token_id
        self._mask_id = tokenizer.mask_token_id
        self._cls_id = tokenizer.cls_token_id
        self._sep_id = tokenizer.sep_token_id
        self._input_sequence_length = (
            tokenizer.max_model_input_sizes.get(base_type, None) or 512
        )

        # relation head on [cls]
        mlp = []
        mlp_hidden_size = mlp_hidden_size or []
        if relation_mode == "concatenation_mlp":
            input_size = self.base.config.hidden_size * 2
            for size in list(mlp_hidden_size) + [2 + relation_size]:
                mlp.append(nn.Linear(input_size, size))
                input_size = size
            self.mlp = nn.Sequential(*mlp)
        elif relation_mode == "subtraction":
            H = np.random.randn(2 + relation_size, self.base.config.hidden_size)
            u, s, vh = np.linalg.svd(H, full_matrices=False)
            self.relation_embedding = t.nn.Parameter(
                t.tensor(np.matmul(u, vh), dtype=t.float32), requires_grad=False
            )
        else:
            raise ValueError(f"Unknown relation_mode {relation_mode}")

        self.relation_mode = relation_mode

    @property
    def hidden_size(self):
        return self.base.config.hidden_size

    def compute_relation(
        self, tokens1, tokens2, attention_mask=None, token_type_ids=None
    ):
        """
        Compute the relation between two entities. The input tokens should be like:

            tokens1/2 = [CLS] [Masked context] [SEP] [Masked Relation tuples] [SEP]

        Args:
            tokens1: Token ids, LongTensor of shape (batch_size, sequence_length)
            tokens2: Token ids, LongTensor of shape (batch_size, sequence_length)
            attention_mask: Attention mask, FloatTensor of shape
                (batch_size, sequence_length).
            token_type_ids: Token type ids, LongTensor of shape
                (batch_size, sequence_length).

        Returns:
            Relation between the two masked context vocabulary, which are logits
            before softmax, FloatTensor of shape (batch_size, 2 + relation_size).
            First two columns are direction of the relation.
        """
        cls1 = self.__call__(
            tokens1, attention_mask=attention_mask, token_type_ids=token_type_ids
        )[0]
        cls2 = self.__call__(
            tokens2, attention_mask=attention_mask, token_type_ids=token_type_ids
        )[0]
        if self.relation_mode == "concatenation_mlp":
            return self.mlp(t.cat((cls1, cls2), dim=1))
        elif self.relation_mode == "subtraction":
            diff = cls2 - cls1
            # batched dot product
            dot = t.einsum("ij,kj->ik", diff, self.relation_embedding)
            # use absolute value of dot product to compare similarity
            sim = t.abs(dot)
            return sim

    def compute_sentence_embeds(
        self, sentence_tokens, context_length: int, with_gradient_num: int = 1
    ):
        """
        Compute the embedding for words in a batch of sentences. The embedding is
        meant to be used by the second BERT model.

        The input tokens should be like:

            sentence_tokens = [token ids for each token in the input of BERT-2]

        Note that the sentence tokens may include [CLS] [SEP], etc. You can use the
        `extend_tokens` argument in the second BERT with extended vocabulary to remove
        them.

        For each token:

            masked context = [left context tokens] [mask] [right context tokens]

        If the left or right context is out of the input sentence, they are padded with
        [PAD], the length of the context is equal to `context_length`.

        Then the tokens fed to this KB encoder will be like:

            tokens = [CLS] [Masked context] [SEP] [MASK padded section] [SEP]

        The relation tuples section is padded with mask since we do not know relations,
        we wish to predict them.

        Args:
            sentence_tokens: Token ids, LongTensor of shape
                (batch_size, sequence_length).
            context_length: Length of the context provided to this model.
            with_gradient_num: Since it is not possible to perform backward operation
                on all input words (batch_size = sequence_length, exceeds memory),
                select this number of token inputs per sample in batch to perform
                forward with gradient. The total with gradient token number is equal to
                batch_size * with_gradient_num

        Returns:
            cls embedding: Float tensor of shape
                (batch_size, sequence_length, hidden_size).
        """
        batch_size = sentence_tokens.shape[0]
        sequence_length = sentence_tokens.shape[1]
        device = sentence_tokens.device

        # generate masked context
        result = []
        for sentence in sentence_tokens:
            # find first sep
            end = sentence.cpu().tolist().index(self._sep_id)
            length = end - 1
            mask_position = t.arange(length, dtype=t.long, device=device)
            _, masked_context = get_context_of_masked(
                sentence_tokens=sentence[1:end].unsqueeze(0).repeat(length, 1),
                mask_position=mask_position,
                context_length=context_length,
                mask_id=self._mask_id,
                pad_id=self._pad_id,
            )

            # generate input
            cls_ = t.full([length, 1], self._cls_id, dtype=t.long, device=device,)
            sep = t.full([length, 1], self._sep_id, dtype=t.long, device=device,)
            mask_pad = t.full(
                [length, self._input_sequence_length - 3 - context_length,],
                self._sep_id,
                dtype=t.long,
                device=device,
            )
            input_tokens = t.cat((cls_, masked_context, sep, mask_pad, sep), dim=1)
            with_gradient_indexes = random.sample(
                range(input_tokens.shape[0]), with_gradient_num * batch_size
            )
            embedding_list = []
            for i in range(input_tokens.shape[0]):
                if i in with_gradient_indexes:
                    embedding = self.__call__(input_tokens[i].unsqueeze(0))[0]
                else:
                    with t.no_grad():
                        embedding = self.__call__(input_tokens[i].unsqueeze(0))[0]
                embedding_list.append(embedding)
            pad = t.zeros_like(embedding_list[0])
            embedding_list = (
                [pad]
                + embedding_list
                + [pad] * (sequence_length - len(embedding_list) - 1)
            )
            result.append(t.cat(embedding_list, dim=0).view(1, sequence_length, -1))
        return t.cat(result, dim=0).view(batch_size, sequence_length, -1)

    def forward(self, tokens, attention_mask=None, token_type_ids=None, labels=None):
        """
        The input tokens should be like:

            tokens = [CLS] [Masked context] [SEP] [Masked Relation tuples] [SEP]

        Note:
            The Masked relation tuples forms a DAG graph, the graph is extracted
            from the dataset using fixed-depth BFS, tuples are added until they exceed
            the max-length of masked relation tuples section.

        Note:
            If masked token(s) in masked context is not a complete entity, then the
            for example if you only mask the "##anti" token in the "anti-authoritarian"
            entity, the relation tuples should be like:

            (anti token-part-of anti-authoritarian) (anti-authoritarian instanceof ...)

        Args:
            tokens: Token ids, LongTensor of shape (batch_size, sequence_length).
            attention_mask: Attention mask, FloatTensor of shape
                (batch_size, sequence_length).
            token_type_ids: Token type ids, LongTensor of shape
                (batch_size, sequence_length).
            labels: Output token labels, value is in [0, vocab_size), LongTensor
                of shape (batch_size, sequence_length).
        Returns:
            cls embedding: Float tensor of shape (batch_size, hidden_size).
            loss: CrossEntropyLoss of predicted labels and given labels, None if
                `labels` is not set, otherwise a float tensor of shape (1,).
            logits: Prediction scores of the language modeling head
                (scores for each vocabulary token before SoftMax),
                FloatTensor of shape (batch_size, sequence_length, vocab_size).

        """
        out = self.base(
            input_ids=tokens,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

        return (
            out.hidden_states[-1][:, 0, :],
            None if labels is None else out.loss,
            out.logits,
        )
