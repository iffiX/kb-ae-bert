import torch as t
from transformers import BatchEncoding
from ..base import StaticMapDataset


class QADataset:
    """
    Base class for general context-question paired QA datasets, which requires
    predictions of start & end positions in context for answers.

    Generally a sampled batch would contain the following fields:
        1. input_ids (no masked tokens)
        2. attention_mask
        3. token_type_ids
        4. start_positions
        5. end_positions

    Please refer to the transformers library (especially the QA model) for their
    meanings.
    """

    def validate(
        self, batch: BatchEncoding, start_logits: t.Tensor, end_logits: t.Tensor
    ):
        """
        Args:
            batch: Sampled batch.
            start_logits: FloatTensor of shape (batch_size, sequence_length),
                Span-start scores (before SoftMax).
            end_logits: FloatTensor of shape (batch_size, sequence_length),
                Span-end scores (before SoftMax).

        Returns:
            A dictionary of various metrics that will be logged by the validating_step.
        """
        raise NotImplementedError

    @property
    def train_dataset(self):
        raise NotImplementedError

    @property
    def validate_dataset(self):
        raise NotImplementedError
