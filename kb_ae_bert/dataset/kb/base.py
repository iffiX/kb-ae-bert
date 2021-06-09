import torch as t
from transformers import BatchEncoding
from ..base import StaticMapDataset


class KnowledgeBaseDataset:
    """
    Base class for general text datasets with words cross-linked with knowledge bases.
    Text is the context where one or more entities occur, if text section is not
    available, it could be generated from text comments / attached text files in a
    knowledge base, but that would limit the versatility of the context.

    There are 2 training objectives based on MLM:

    For entity encoding training with MLM, a sampled batch would contain the following
    fields:
        1. input_ids:
            context with part/all of one type of entity masked,
            fixed-depth relation graph tuples with same part/all of entity masked.
        2. attention_mask
        3. token_type_ids
        4. labels: input ids without mask.

    The validation function is the cross entropy loss between predicted labels and
    given labels.

    For relation encoding training with MLM, a sampled batch would contain the following
    fields:
        1. input_ids_1:
            context with part/all of first entity masked,
            empty relation sequence padded with [MASK].
        2. input_ids_2:
            context with part/all of second entity masked,
            empty relation sequence padded with [MASK].
        3. attention_mask: apply to both input ids
        4. token_type_ids: apply to both input ids.
        5. relation: a LongTensor of shape (batch_size,), with each element being
            the relation index between entity 1 and entity 2
        5. direction: a LongTensor of shape (batch_size,), with each element being
            the direction of relation, 0 for 1 -> 2 and 1 for 1 <- 2.
    """

    def validate_relation_encode(
        self, batch: BatchEncoding, relation_logits: t.Tensor,
    ):
        """
        Args:
            batch: Sampled batch.
            relation_logits: FloatTensor of shape (batch_size, 2 + relation_size),
                first two columns are direction, remaining columns are relation scores.
                All values are inputs before SoftMax.

        Returns:
            A dictionary of various metrics that will be logged by the validating_step.
        """
        raise NotImplementedError

    @property
    def train_entity_encode_dataset(self):
        raise NotImplementedError

    @property
    def validate_entity_encode_dataset(self):
        # Note: this validation is done by common MLM training objective,
        # just input "labels" when using the MaskedLM models.
        # therefore there is no dedicate validate function
        raise NotImplementedError

    @property
    def train_relation_encode_dataset(self):
        raise NotImplementedError

    @property
    def validate_relation_encode_dataset(self):
        raise NotImplementedError
