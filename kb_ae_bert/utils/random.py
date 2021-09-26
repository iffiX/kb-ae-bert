import logging
import random
from typing import Optional

import numpy as np
import torch

from pytorch_lightning.utilities import _TORCH_GREATER_EQUAL_1_7
from pytorch_lightning.utilities.distributed import rank_zero_only


def worker_init_function(
    worker_id: int, rank: Optional = None
) -> None:  # pragma: no cover
    """
    The worker_init_fn that Lightning automatically adds to your dataloader if you previously set
    set the seed with ``seed_everything(seed, workers=True)``.
    See also the PyTorch documentation on
    `randomness in DataLoaders <https://pytorch.org/docs/stable/notes/randomness.html#dataloader>`_.
    """
    # implementation notes: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    global_rank = rank if rank is not None else rank_zero_only.rank
    process_seed = torch.initial_seed()
    # back out the base seed so we can use all the bits
    base_seed = process_seed - worker_id
    logging.info(
        f"Initializing random number generators of process {global_rank} "
        f"worker {worker_id} with base seed {base_seed}"
    )
    ss = np.random.SeedSequence([base_seed, worker_id, global_rank])
    # use 128 bits (4 x 32-bit words)
    np.random.seed(ss.generate_state(4))
    # Spawn distinct SeedSequences for the PyTorch PRNG and the stdlib random module
    torch_ss, stdlib_ss = ss.spawn(2)
    # PyTorch 1.7 and above takes a 64-bit seed
    dtype = np.uint64 if _TORCH_GREATER_EQUAL_1_7 else np.uint32
    torch.manual_seed(torch_ss.generate_state(1, dtype=dtype)[0])
    # use 128 bits expressed as an integer
    stdlib_seed = (
        stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]
    ).sum()
    random.seed(stdlib_seed)
