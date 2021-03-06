import logging
import torch as t
import time
import pickle
import cProfile

logging.root.setLevel(logging.INFO)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from kb_ae_bert.dataset.kb.kdwd import KDWDBertDataset
from kb_ae_bert.dataset.base import collate_function_dict_to_batch_encoding


# kdwd = KDWDDataset(
#     relation_size=10,
#     local_root_path="/nfs/vdisk/AI/datasets",
#     mongo_docker_name="mongodb2",
#     mongo_docker_host="node1",
#     mongo_docker_api_host="tcp://node1:4243",
#     force_reload=True,
# )
kdwd = KDWDBertDataset(
    graph_depth=1,
    context_length=32,
    sequence_length=64,
    tokenizer=AutoTokenizer.from_pretrained("bert-large-uncased"),
    relation_size=200,
    generate_data=True,
    relation_mask_mode="part",
)
# print(kdwd.relation_names)


def x():
    kdwd.print_sample_of_entity_encode(split="train")
    kdwd.print_sample_of_relation_encode(split="train")


if __name__ == "__main__":
    # pickle.dumps(kdwd.train_entity_encode_dataset)
    # kdwd.print_sample_of_entity_encode(split="train")
    # Apple
    # kdwd.print_sample_of_entity_encode(item_id=89)
    # Apple Inc.
    # kdwd.print_sample_of_entity_encode(item_id=312)
    # kdwd.print_sample_of_relation_encode(edge_id="60bfbf5944245b4296930673")

    kdwd.open_db()
    start = time.time()
    # cProfile.run("x()")
    x()
    # kdwd.print_sample_of_relation_encode(split="train")
    # kdwd.print_sample_of_entity_encode(item_id=5334183)
    end = time.time()
    print((end - start) / 100)

    # t.multiprocessing.set_start_method("spawn", force=True)
    # d = DataLoader(
    #     kdwd.train_entity_encode_dataset,
    #     batch_size=8,
    #     collate_fn=collate_function_dict_to_batch_encoding,
    #     num_workers=1,
    # )
    # begin = time.time()
    # i = 0
    # for x in d:
    #     print(i)
    #     if i >= 100:
    #         break
    #     i += 1
    # end = time.time()
    # if i != 0:
    #     print(f"Average {(end-begin)*1000/i} ms")
