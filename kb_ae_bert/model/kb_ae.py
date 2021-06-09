from typing import Tuple
from urllib.parse import urlparse
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from ..utils.settings import model_cache_dir, proxies
import torch as t
import torch.nn as nn
import numpy as np


def get_context_of_masked(
    sentence_tokens: t.Tensor,
    mask_position: t.Tensor,
    context_length: int,
    pad_id: int,
    mask_id: int,
):
    """
    Args:
        sentence_tokens: Token ids, LongTensor of shape (batch_size, sequence_length).

        mask_position: Mask position, LongTensor of shape (batch_size,).
        context_length: Length of the context provided to this model.
        pad_id: Id of the `[PAD]` token.
        mask_id: Id of the `[MASK]` token.

    Returns:
        For each mask position:
        masked context = [left context tokens] [mask] [right context tokens]

        masked context is token id tensor, LongTensor of shape
            (batch_size, context_length).

    """
    assert sentence_tokens.shape[0] == mask_position.shape[0], "Batch size unequal."
    batch_size = sentence_tokens.shape[0]
    sequence_length = sentence_tokens.shape[1]
    left_context_length = int((context_length - 1) / 2)
    right_context_length = context_length - 1 - left_context_length

    # pad both sides first
    padded_sentence_tokens = t.full(
        [batch_size, left_context_length + sequence_length + right_context_length],
        pad_id,
        dtype=t.long,
        device=sentence_tokens.device,
    )
    padded_sentence_tokens[
        :, left_context_length : left_context_length + sequence_length
    ] = sentence_tokens

    # create index tensor and gather
    offset = t.arange(
        left_context_length,
        context_length + left_context_length,
        dtype=t.long,
        device=sentence_tokens.device,
    ).unsqueeze(0)
    
    index = mask_position.unsqueeze(1).repeat(1, context_length) + offset

    print(pad_id, mask_id)
    print(batch_size, context_length)

    print(batch_size, left_context_length + sequence_length + right_context_length)
    print(offset.shape) 
    print(index.shape)
    print(padded_sentence_tokens.shape)

    masked_context = t.gather(padded_sentence_tokens, dim=-1, index=index)
    
    masked_context[:, left_context_length] = mask_id

    return masked_context


class KBMaskedLMEncoder(nn.Module):
    def __init__(
        self,
        relation_size: int,
        base_type: str = "bert-base-uncased",
        relation_mode: str = "concatenation_mlp",
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
        super().__init__()
        self.base = AutoModelForMaskedLM.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            # proxies={"http": urlparse(http_proxy).path},
            output_hidden_states=True,
            return_dict=True,
            **base_configs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_type, 
            # cache_dir=model_cache_dir, 
            # proxies={"http": proxies},
        )
        self._pad_id = tokenizer.pad_token_id
        self._mask_id = tokenizer.mask_token_id
        self._cls_id = tokenizer.cls_token_id
        self._sep_id = tokenizer.sep_token_id
        self._input_sequence_length = (
            tokenizer.max_model_input_sizes.get(base_type) or 512
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

    @property
    def hidden_size(self):
        return self.base.config.hidden_size

    def compute_relation(self, tokens1, tokens2):
        """
        Compute the relation between two entities. The input tokens should be like:

            tokens1/2 = [CLS] [Masked context] [SEP] [Masked Relation tuples] [SEP]

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

    def compute_sentence_embeds(self, sentence_tokens, context_length: int):
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

        Returns:
            cls embedding: Float tensor of shape
                (batch_size, sequence_length, hidden_size).
        """
        batch_size = sentence_tokens.shape[0]
        sequence_length = sentence_tokens.shape[1]
        device = sentence_tokens.device
        # generate masked context
        # [batch_size * sequence_length, sequence_length]
        
        """
        print(sentence_tokens.unsqueeze(1).shape)
        This has 3 dimensions, repeat was only addressing 2 dimensions
        """
        sentence_tokens = (
            sentence_tokens.unsqueeze(1)
            .repeat(1, sequence_length, 1)
            .flatten(start_dim=0, end_dim=1)
        )
        mask_position = t.arange(sequence_length, dtype=t.long, device=device).repeat(
            batch_size
        )
        masked_context = get_context_of_masked(
            sentence_tokens=sentence_tokens,
            mask_position=mask_position,
            context_length=context_length,
            mask_id=self._mask_id,
            pad_id=self._pad_id,
        )

        # generate input
        cls_ = t.full(
            [batch_size * sequence_length, 1],
            self._cls_id,
            dtype=t.long,
            device=device,
        )
        sep = t.full(
            [batch_size * sequence_length, 1],
            self._sep_id,
            dtype=t.long,
            device=device,
        )
        mask_pad = t.full(
            [
                batch_size * sequence_length,
                self._input_sequence_length - 3 - context_length,
            ],
            self._sep_id,
            dtype=t.long,
            device=device,
        )
        input_tokens = t.cat((cls_, masked_context, sep, mask_pad, sep), dim=1)
        cls = self.__call__(input_tokens)[0]
        return cls.view(batch_size, sequence_length, -1)

    def forward(self, tokens, labels=None):
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
        super().__init__()
        self.base = AutoModelForTokenClassification.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            proxies={"http": proxies},
            return_dict=True,
            num_labels=2,
            **base_configs,
        )

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
