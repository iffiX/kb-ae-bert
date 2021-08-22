import torch as t


# def get_context_of_masked(
#     sentence_tokens: t.Tensor,
#     mask_position: t.Tensor,
#     context_length: int,
#     pad_id: int,
#     mask_id: int,
#     **__,
# ):
#     """
#     Note:
#         This implementation centers te masked token in the context.
#
#     Eg:
#         sentence_tokens = [[1,2,3,4,5], [6,7,8,9,10]]
#         mask_position = [1, 2] (So element 2 and 8 will be masked)
#         context_length = 3
#         pad_id = 0
#         mask_id = -1
#
#         Result:
#             [[1, -1, 3], [7, -1, 9]]
#
#     Args:
#         sentence_tokens: Token ids, LongTensor of shape (batch_size, sequence_length).
#         mask_position: Mask position, in range [0, sequence_length), LongTensor
#             of shape (batch_size,).
#         context_length: Length of the context provided to this model. Must be smaller
#             than sequence_length.
#         pad_id: Id of the `[PAD]` token.
#         mask_id: Id of the `[MASK]` token.
#
#     Returns:
#         A tuple of original context and masked context
#
#         For each mask position:
#         masked context = [left context tokens] [mask] [right context tokens]
#
#         Original context is token id tensor, LongTensor of shape
#             (batch_size, context_length).
#
#         Masked context is token id tensor, LongTensor of shape
#             (batch_size, context_length).
#     """
#     assert sentence_tokens.shape[0] == mask_position.shape[0], "Batch size unequal."
#     batch_size = sentence_tokens.shape[0]
#     sequence_length = sentence_tokens.shape[1]
#     left_context_length = int((context_length - 1) / 2)
#     right_context_length = context_length - 1 - left_context_length
#
#     # pad both sides first
#     padded_sentence_tokens = t.full(
#         [batch_size, left_context_length + sequence_length + right_context_length],
#         pad_id,
#         dtype=t.long,
#         device=sentence_tokens.device,
#     )
#     padded_sentence_tokens[
#         :, left_context_length : left_context_length + sequence_length
#     ] = sentence_tokens
#
#     # create index tensor and gather
#     offset = t.arange(
#         0, context_length, dtype=t.long, device=sentence_tokens.device,
#     ).unsqueeze(0)
#     index = mask_position.unsqueeze(-1).repeat(1, context_length) + offset
#     original_context = t.gather(padded_sentence_tokens, dim=-1, index=index)
#     masked_context = original_context.clone()
#     masked_context[:, left_context_length] = mask_id
#
#     return masked_context


def get_context_of_masked(
    sentence_tokens: t.Tensor,
    mask_position: t.Tensor,
    context_length: int,
    pad_id: int,
    mask_id: int,
    generator: t.Generator = None,
    **__,
):
    """
    Note:
        This implementation randomly samples the context position, so masked token
        could be anywhere in the context window.

    Eg:
        sentence_tokens = [[1,2,3,4,5], [6,7,8,9,10]]
        mask_position = [1, 2] (So element 2 and 8 will be masked)
        context_length = 3
        pad_id = 0
        mask_id = -1

        Some possible masked results:
            [[1, -1, 3], [6, 7, -1]]
            [[0, 1, -1], [7, -1, 9]]
            [[-1, 3, 4], [-1, 9, 10]]

    Args:
        sentence_tokens: Token ids, LongTensor of shape (batch_size, sequence_length).
        mask_position: Mask position, in range [0, sequence_length), LongTensor
            of shape (batch_size,).
        context_length: Length of the context provided to this model. Must be smaller
            than sequence_length.
        pad_id: Id of the `[PAD]` token.
        mask_id: Id of the `[MASK]` token.
        generator: A pseudorandom number generator for sampling.

    Returns:
        A tuple of original context and masked context

        For each mask position:
        masked context = [left context tokens] [mask] [right context tokens]

        Original context is token id tensor, LongTensor of shape
            (batch_size, context_length).

        Masked context is token id tensor, LongTensor of shape
            (batch_size, context_length).
    """
    assert sentence_tokens.shape[0] == mask_position.shape[0], "Batch size unequal."
    batch_size = sentence_tokens.shape[0]
    sequence_length = sentence_tokens.shape[1]

    # pad both sides first
    padded_sentence_tokens = t.full(
        [batch_size, (context_length - 1) + sequence_length + (context_length - 1)],
        pad_id,
        dtype=t.long,
        device=sentence_tokens.device,
    )
    padded_sentence_tokens[
        :, (context_length - 1) : (context_length - 1) + sequence_length
    ] = sentence_tokens

    # create index tensor and gather
    offset = t.arange(
        0, context_length, dtype=t.long, device=sentence_tokens.device,
    ).unsqueeze(0)

    right_context_length = t.randint(
        low=0, high=context_length, size=(batch_size,), generator=generator
    )
    left_context_length = context_length - 1 - right_context_length
    index = (mask_position + right_context_length).unsqueeze(-1).repeat(
        1, context_length
    ) + offset
    original_context = t.gather(padded_sentence_tokens, dim=-1, index=index)
    masked_context = original_context.clone()
    masked_context[list(range(batch_size)), left_context_length] = mask_id

    return original_context, masked_context
