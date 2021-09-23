from typing import List
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
)
from ..utils.settings import model_cache_dir, proxies
from ..utils.token import get_context_of_masked
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
        self,
        sentences_tokens,
        context_length: int,
        sequence_length: int,
        process_batch_size: int,
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
            sentences_tokens: Token ids, LongTensor of shape
                (batch_size, sequence_length).
            context_length: Length of the context provided to this model.
            sequence_length: Length of the sequence provided to this model.
            process_batch_size: Batch size for processing meaningful tokens from
                all sentences.

        Returns:
            cls embedding: Float tensor of shape
                (batch_size, sequence_length, hidden_size).
        """
        device = sentences_tokens.device

        # generate masked context
        kb_input = []
        mask_positions = [[], []]
        embeddings = []
        helper_tokens = {self._pad_id, self._mask_id, self._cls_id, self._sep_id}
        for idx, sentence in enumerate(sentences_tokens):
            mask_position = []
            for i in range(len(sentence)):
                if int(sentence[i]) not in helper_tokens:
                    mask_position.append(i)
                else:
                    # clear out mask, cls and sep
                    sentence[i] = self._pad_id

            mask_length = len(mask_position)
            _, masked_context = get_context_of_masked(
                sentence_tokens=sentence.unsqueeze(0).repeat(mask_length, 1),
                mask_position=t.tensor(mask_position, dtype=t.long, device=device),
                context_length=context_length,
                mask_id=self._mask_id,
                pad_id=self._pad_id,
            )

            # generate input
            cls = t.full([mask_length, 1], self._cls_id, dtype=t.long, device=device,)
            sep = t.full([mask_length, 1], self._sep_id, dtype=t.long, device=device,)
            # The relation region is filled with [MASK]
            relation = t.full(
                [mask_length, sequence_length - 3 - context_length],
                self._mask_id,
                dtype=t.long,
                device=device,
            )
            # appended tensor is of shape [mask_length, sequence_length]
            kb_input.append(t.cat((cls, masked_context, sep, relation, sep), dim=1))
            mask_positions[0] += [idx] * len(mask_position)
            mask_positions[1] += mask_position

        kb_input = t.cat(kb_input, dim=0)
        for inp in t.split(kb_input, process_batch_size, dim=0):
            embeddings.append(self.__call__(inp)[0])
        embeddings = t.cat(embeddings, dim=0)

        result = t.zeros(
            [sentences_tokens.shape[0], sentences_tokens.shape[1], embeddings.shape[1]],
            dtype=embeddings.dtype,
            device=device,
        )
        result[mask_positions[0], mask_positions[1]] = embeddings
        return result

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
