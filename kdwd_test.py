import logging
import torch as t
import time
import pickle

logging.root.setLevel(logging.INFO)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from kb_ae_bert.dataset.kb.kdwd import KDWDDataset
from kb_ae_bert.dataset.base import collate_function_dict_to_batch_encoding


# kdwd = KDWDDataset(
#     relation_size=10,
#     local_root_path="/nfs/vdisk/AI/datasets",
#     mongo_docker_name="mongodb2",
#     mongo_docker_host="node1",
#     mongo_docker_api_host="tcp://node1:4243",
#     force_reload=True,
# )
kdwd = KDWDDataset(
    context_length=200,
    sequence_length=512,
    tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
    relation_size=200,
    local_root_path="/nfs/vdisk/AI/datasets",
    mongo_docker_name="mongodb2",
    mongo_docker_host="node1",
    mongo_docker_api_host="tcp://node1:4243",
)
# print(kdwd.relation_names)

if __name__ == "__main__":
    # pickle.dumps(kdwd.train_entity_encode_dataset)
    # kdwd.print_sample_of_entity_encode(split="train")
    # kdwd.print_sample_of_entity_encode(item_id=5969138)
    # kdwd.print_sample_of_relation_encode(split="train")

    # t.multiprocessing.set_start_method("spawn", force=True)
    d = DataLoader(
        kdwd.train_entity_encode_dataset,
        batch_size=512,
        collate_fn=collate_function_dict_to_batch_encoding,
        num_workers=6,
    )
    begin = time.time()
    i = 0
    for x in d:
        print(i)
        if i >= 20:
            break
        i += 1
    end = time.time()
    if i != 0:
        print(f"Average {(end-begin)*1000/i} ms")
