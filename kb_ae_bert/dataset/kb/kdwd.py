import os
import csv
import pprint
import logging
import random
import numpy as np
import pickle
import itertools
import torch as t
from typing import Callable, List
from docker.types import Mount
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING
from transformers import PreTrainedTokenizerBase
from kb_ae_bert.model.kb_ae import get_context_of_masked
from kb_ae_bert.utils.settings import (
    dataset_cache_dir,
    mongo_docker_name as default_mdn,
    preprocess_cache_dir,
)
from kb_ae_bert.utils.file import open_file_with_create_directories
from kb_ae_bert.utils.kaggle import download_dataset
from kb_ae_bert.utils.docker import create_or_reuse_docker, allocate_port
from kb_ae_bert.utils.mongo import load_dataset_files, connect_to_database
from ..base import DynamicIterableDataset


def default_mask_function(seq: List[int], mask_id: int):
    return [mask_id if np.random.rand() < 0.15 else seq_item for seq_item in seq]


class InvalidSampleError(Exception):
    pass


class KDWDDataset:
    def __init__(
        self,
        relation_size: int,
        context_length: int,
        sequence_length: int,
        tokenizer: PreTrainedTokenizerBase,
        graph_depth: int = 1,
        permute_seed: int = 0,
        train_entity_encode_ratio: float = 0.9,
        train_relation_encode_ratio: float = 0.9,
        local_root_path: str = None,
        mongo_docker_name: str = None,
        mongo_docker_host: str = "localhost",
        mongo_docker_api_host: str = None,
        force_reload: bool = False,
        mask_function: Callable[[List[int], int], List[int]] = default_mask_function,
    ):
        """
        Args:
            relation_size: Number of allowed relations, sort by usage, overflowing
                relations will be treated as "Unknown relation".
            context_length: Length of context tokens, should be smaller than
                sequence_length, 1/2 of it is recommended.
            sequence_length: Length of the output sequence to model.
            tokenizer: Tokenizer to use.
            graph_depth: Maximum depth of the BFS algorithm used to generate the graph
                of relations starting from the target entity.
            permute_seed: Permutation seed used to generate splits.
            train_entity_encode_ratio: Ratio of entries used for entity encoding
                training. Remaining are used to validate.
            train_relation_encode_ratio: Ratio of entries used for relation encoding
                training. Remaining are used to validate.
            local_root_path: Local root path of saving the KDWD dataset downloaded
                from Kaggle, if not specified, it is default to
                "<dataset_cache_dir>/kaggle".
            mongo_docker_name: Name of the created / reused mongo docker.
            mongo_docker_host: Host address of the mongo docker.
            mongo_docker_api_host: Host address of the docker API, could be different
                from mongo_docker_host if you are using a cluster, etc.
            force_reload: Whether force relaoding and preprocessing KDWD dataset into
                the mongo database.
            mask_function: The mask function used to mask the relation section
                of the output sequence.
        """
        local_root_path = local_root_path or str(
            os.path.join(dataset_cache_dir, "kaggle")
        )
        mongo_docker_name = mongo_docker_name or default_mdn

        self.relation_size = relation_size
        self.context_length = context_length
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.graph_depth = graph_depth
        self.permute_seed = permute_seed
        self.train_entity_encode_ratio = train_entity_encode_ratio
        self.train_relation_encode_ratio = train_relation_encode_ratio
        self.local_root_path = local_root_path
        self.mongo_docker_name = mongo_docker_name
        self.mongo_docker_host = mongo_docker_host
        self.mongo_docker_api_host = mongo_docker_api_host
        self.force_reload = force_reload
        self.mask_function = mask_function

        if mongo_docker_api_host is not None:
            os.environ["DOCKER_HOST"] = mongo_docker_api_host
        self.db_docker, is_reused = create_or_reuse_docker(
            image="mongo:latest",
            startup_args={
                "ports": {"27017": allocate_port()},
                "mounts": [
                    Mount(
                        target="/mnt/dataset",
                        source=local_root_path,
                        type="bind",
                        read_only=True,
                    )
                ],
            },
            reuse_name=mongo_docker_name,
        )
        self.db_port = int(
            self.db_docker.attrs["HostConfig"]["PortBindings"]["27017/tcp"][0][
                "HostPort"
            ]
        )

        download_dataset(
            "kenshoresearch/kensho-derived-wikimedia-data", local_root_path
        )
        if not is_reused or force_reload:
            # load dataset into the database
            # we don't load statements.csv as we need to preprocess it
            load_dataset_files(
                self.db_docker,
                "kdwd",
                str(os.path.join(local_root_path, "kensho-derived-wikimedia-data")),
                [
                    "item.csv",
                    "item_aliases.csv",
                    "link_annotated_text.jsonl",
                    "page.csv",
                    "property.csv",
                    "property_aliases.csv",
                ],
            )
            self.db = connect_to_database(mongo_docker_host, self.db_port, "kdwd")

            self.insert_statements_with_pages(
                os.path.join(
                    local_root_path, "kensho-derived-wikimedia-data", "statements.csv"
                )
            )

            # create necessary indexes
            logging.info("Creating indexes, this operation will take a LONG time!")
            self.db.item.create_index([("item_id", ASCENDING)])
            self.db.statements.create_index([("source_item_id", ASCENDING)])
            self.db.statements.create_index([("target_item_id", ASCENDING)])
            self.db.page.create_index([("page_id", ASCENDING)])
            self.db.page.create_index([("item_id", ASCENDING)])
            self.db.link_annotated_text.create_index(
                [("sections.target_page_ids", ASCENDING)]
            )
            self.db.link_annotated_text.createIndex([("page_id", ASCENDING)])

            logging.info("Indexes created.")

            # remove disconnected statements
            # self.remove_statements_without_pages()
        else:
            self.db = connect_to_database(mongo_docker_host, self.db_port, "kdwd")

        self.unknown_relation_id = 0
        self.sub_part_relation_id = 1
        self.relation_index_start = 2
        if os.path.exists(os.path.join(preprocess_cache_dir, "kdwd.cache")):
            logging.info("Found states cache for KDWD, skipping generation.")
            self.restore(os.path.join(preprocess_cache_dir, "kdwd.cache"))
        else:
            logging.info("States cache for KDWD not found, generating.")
            # relations
            self.relation_names = ["<unknown relation>", "<is a sub token of>"]
            # maps property id in statements to a relation id
            self.property_to_relation_mapping = {}
            logging.info("Selecting relations by usage.")
            self.select_relations()

            # generate train/validate splits
            self.entity_encode_splits = {}
            self.relation_encode_splits = {}
            logging.info("Generating splits.")
            self.generate_split_for_entity_encode()
            self.generate_split_for_relation_encode()

            self.save(os.path.join(preprocess_cache_dir, "kdwd.cache"))

        # close connection to db, reopen it later, to support forking
        self.db = None

    def get_loss(self, batch, relation_logits):
        ce_loss = t.nn.CrossEntropyLoss()
        direction_loss = ce_loss(relation_logits[:, :2], batch.direction)
        relation_loss = ce_loss(relation_logits[:, 2:], batch.relation)
        return direction_loss, relation_loss

    def validate_relation_encode(self, batch, relation_logits):
        direction_loss, relation_loss = self.get_loss(batch, relation_logits)
        return {
            "direction_loss": direction_loss.item(),
            "relation_loss": relation_loss.item(),
            "total_loss": direction_loss.item() + relation_loss.item(),
        }

    @property
    def train_entity_encode_dataset(self):
        return DynamicIterableDataset(self.generator_of_entity_encode, ("train",))

    @property
    def validate_entity_encode_dataset(self):
        return DynamicIterableDataset(self.generator_of_entity_encode, ("validate",))

    @property
    def train_relation_encode_dataset(self):
        raise DynamicIterableDataset(self.generator_of_relation_encode, ("train",))

    @property
    def validate_relation_encode_dataset(self):
        raise DynamicIterableDataset(self.generator_of_relation_encode, ("validate",))

    def generator_of_entity_encode(self, split: str):
        self.open_db()
        sample = self.generate_sample_of_entity_encode(split=split)

        attention_mask = t.ones(
            sample[0].shape, dtype=t.float32, device=sample[0].device
        )
        token_type_ids = t.ones_like(sample[0])
        token_type_ids[: 2 + self.context_length] = 0
        return {
            "input_ids": sample[1],
            "labels": sample[0],
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def generator_of_relation_encode(self, split: str):
        self.open_db()
        sample = self.generate_sample_of_relation_encode(split=split)

        attention_mask = t.ones(
            sample[1].shape, dtype=t.float32, device=sample[1].device
        )
        token_type_ids = t.ones_like(sample[1])
        token_type_ids[: 2 + self.context_length] = 0
        # randomly switch direction of relation
        if random.choice([0, 1]) == 0:
            return {
                "input_ids_1": sample[1],
                "input_ids_2": sample[2],
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "relation": t.tensor(
                    [sample[0]], dtype=sample[1].dtype, device=sample[1].device
                ),
                "direction": t.tensor(
                    [0], dtype=sample[1].dtype, device=sample[1].device
                ),
            }
        else:
            return {
                "input_ids_1": sample[2],
                "input_ids_2": sample[1],
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "relation": t.tensor(
                    [sample[0]], dtype=sample[1].dtype, device=sample[1].device
                ),
                "direction": t.tensor(
                    [1], dtype=sample[1].dtype, device=sample[1].device
                ),
            }

    def print_sample_of_entity_encode(self, split: str = None, item_id: int = None):
        original, masked = self.generate_sample_of_entity_encode(split, item_id)
        print("Original:")
        print(self.tokenizer.decode(original[0].tolist()))
        print("Masked:")
        print(self.tokenizer.decode(masked[0].tolist()))

    def print_sample_of_relation_encode(self, split: str = None):
        relation, masked_1, masked_2 = self.generate_sample_of_relation_encode(split)

        print("Relation:")
        print(self.relation_names[relation])
        print("Masked 1:")
        print(self.tokenizer.decode(masked_1[0].tolist()))
        print("Masked 2:")
        print(self.tokenizer.decode(masked_2[0].tolist()))

    def generate_sample_of_entity_encode(self, split: str = None, item_id: int = None):
        """
        Args:
            split: Name of split, "train" or "validate".
            item_id: Item id of the page.

        Returns:
            Original token ids (context + relation),
                Long Tensor of shape (1, sequence_length).
            Masked token ids (context + relation),
                Long Tensor of shape (1, sequence_length).
        """
        # first select a page
        if split is not None and item_id is None:
            page_id = random.choice(self.entity_encode_splits[split])
            page = self.db.page.find_one({"page_id": page_id})
            if page is None:
                raise ValueError(f"Cannot find page with page_id={page_id}")
            item_id = page["item_id"]
        elif item_id is not None and split is None:
            # used for generating sample for two given entities in relation sampling
            page = self.db.page.find_one({"item_id": item_id})
            if page is None:
                raise ValueError(f"Cannot find page with item_id={item_id}")
            page_id = page["page_id"]
        else:
            raise ValueError(
                f"You can only set split or item_id, "
                f"but got split={split}, item_id={item_id}."
            )

        entity_name = str(page["title"])
        if len(entity_name) == 0:
            entity_name = "unknown"

        # then find a link_annotated_text where page_id in sections.target_page_ids
        context = next(
            self.db.link_annotated_text.aggregate(
                [
                    {"$match": {"sections.target_page_ids": page_id}},
                    {"$sample": {"size": 1}},
                ]
            ),
            None,
        )
        if context is not None:
            possible_context = []
            for section in context["sections"]:
                if page_id in section["target_page_ids"]:
                    text = section["text"]
                    idx = section["target_page_ids"].index(page_id)
                    link_length = section["link_lengths"][idx]
                    link_offset = section["link_offsets"][idx]
                    possible_context.append((text, link_length, link_offset))
            if len(possible_context) == 0:
                raise ValueError(
                    f"Cannot find page id {page_id} in context {pprint.pformat(context)}"
                )
            text, link_length, link_offset = random.choice(possible_context)
            # replace link with title
            text = (
                text[:link_offset]
                + f" {entity_name} "
                + text[link_offset + link_length :]
            )
            link_length = len(entity_name)
            link_offset = link_offset + 1  # include space before title
        else:
            # fallback to use the page itself, with entity being the title
            context = self.db.link_annotated_text.find_one({"page_id": page_id})
            if context is None:
                # fallback to use title only
                text = entity_name
                link_length = len(entity_name)
                link_offset = 0
            else:
                article_body = [sec["text"] for sec in context["sections"]]
                text = entity_name + " " + " ".join(article_body)
                link_length = len(entity_name)
                link_offset = 0

        # then use graphLookup to find items related to item_id of the page
        graph = next(
            self.db.page.aggregate(
                [
                    {"$match": {"item_id": item_id}},
                    {
                        "$graphLookup": {
                            "from": "statements",
                            "startWith": item_id,
                            "connectFromField": "target_item_id",
                            "connectToField": "source_item_id",
                            "maxDepth": self.graph_depth - 1,
                            "depthField": "depth",
                            "as": "graph",
                        }
                    },
                    {"$limit": 1},
                ]
            ),
            None,
        )
        if graph is None:
            raise ValueError(
                f"Cannot create graph for page_id={page_id}, item_id={item_id}"
            )
        graph = graph["graph"]
        node_label_mapping = {}
        graph = sorted(graph, key=lambda e: e["depth"])
        for edge in graph:
            node_label_mapping[edge["source_item_id"]] = "unknown"
            node_label_mapping[edge["target_item_id"]] = "unknown"

        # first fill entities connected with a page with page title
        for related_page in self.db.page.find(
            {"item_id": {"$in": list(node_label_mapping.keys())}}
        ):
            related_title = str(related_page["title"])
            if len(related_title) > 0:
                node_label_mapping[related_page["item_id"]] = related_title

        # then fill remaining unknown entities with their en_label in item table
        for item in self.db.item.find(
            {"item_id": {"$in": list(node_label_mapping.keys())}}
        ):
            en_label = str(item["en_label"])
            if len(en_label) > 0:
                node_label_mapping[item["item_id"]] = en_label

        # then tokenize, mask, pad, and return result
        entity_tokens = self.tokenizer.tokenize(
            str(entity_name), add_special_tokens=False
        )
        text_tokens = self.tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        )

        # # if entity is comprised of multiple tokens
        # # mask one random token, add relation "is sub part of"
        # # else
        # # mask the only token
        masked_entity_part = len(entity_tokens) != 1

        # # select the token in the entity to mask, if there is only one token,
        # # then it is masked.
        entity_token_indexes = {
            text_tokens.char_to_token(char_idx)
            for char_idx in range(link_offset, link_offset + link_length)
        }
        # # if there are spaces in an entity, their token index would be None
        entity_token_indexes.discard(None)
        entity_token_indexes = list(entity_token_indexes)
        masked_entity_token_index = random.choice(entity_token_indexes)
        masked_entity_token_id = text_tokens.input_ids[
            0, masked_entity_token_index
        ].item()

        masked_entity_context = get_context_of_masked(
            sentence_tokens=text_tokens.input_ids,
            mask_position=t.tensor([masked_entity_token_index], dtype=t.long),
            context_length=self.context_length,
            pad_id=self.tokenizer.pad_token_id,
            mask_id=self.tokenizer.mask_token_id,
        )
        original_entity_context = masked_entity_context.clone()
        # # masked token is always at center, after the left context
        original_entity_context[
            0, int((self.context_length - 1) / 2)
        ] = masked_entity_token_id

        # # generate relation
        # # There is a [PAD] token after each relation tuple.
        original_relation_list = []
        masked_relation_list = []
        for edge in graph:
            masked_source = source = node_label_mapping[edge["source_item_id"]]
            masked_target = target = node_label_mapping[edge["target_item_id"]]
            if not masked_entity_part:
                if edge["source_item_id"] == item_id:
                    masked_source = self.tokenizer.mask_token
                if edge["target_item_id"] == item_id:
                    masked_target = self.tokenizer.mask_token
            relation = self.relation_names[
                self.property_to_relation_mapping[edge["edge_property_id"]]
            ]
            original_relation_list.append(
                self.tokenizer.encode(
                    f"{source} {relation} {target}", add_special_tokens=False
                )
                + [self.tokenizer.pad_token_id]
            )
            masked_relation_list.append(
                self.tokenizer.encode(
                    f"{masked_source} {relation} {masked_target}",
                    add_special_tokens=False,
                )
                + [self.tokenizer.pad_token_id]
            )

        # # If masked token is a subpart of the entity,
        # # add a new relation "[MASK] is a sub part of <entity>"
        if masked_entity_part:
            masked_sub_part = self.tokenizer.encode(
                f"{self.tokenizer.mask_token} "
                f"{self.relation_names[self.sub_part_relation_id]} "
                f"{entity_name}",
                add_special_tokens=False,
            ) + [self.tokenizer.pad_token_id]
            original_sub_part = masked_sub_part.copy()
            original_sub_part[0] = masked_entity_token_id
            masked_relation_list = [masked_sub_part] + masked_relation_list
            original_relation_list = [original_sub_part] + original_relation_list

        # # combine context and relation
        # # Allowed number of tokens with [CLS] context [SEP] ... [SEP] excluded
        max_relation_token_num = self.sequence_length - 3 - self.context_length
        relation_token_num = 0
        relation_tuple_max_idx = 0
        for relation in original_relation_list:
            if relation_token_num + len(relation) > max_relation_token_num:
                break
            relation_token_num += len(relation)
            relation_tuple_max_idx += 1

        relation_padding = [self.tokenizer.pad_token_id] * (
            max_relation_token_num - relation_token_num
        )
        original_relation = t.tensor(
            list(
                itertools.chain(
                    *original_relation_list[:relation_tuple_max_idx], relation_padding
                )
            ),
            dtype=original_entity_context.dtype,
        )
        masked_relation = t.tensor(
            self.mask_function(
                list(
                    itertools.chain(
                        *masked_relation_list[:relation_tuple_max_idx], relation_padding
                    )
                ),
                self.tokenizer.mask_token_id,
            ),
            dtype=masked_entity_context.dtype,
        )
        if masked_relation.shape[0] != original_relation.shape[0]:
            pass
        # # return result
        output_original = t.zeros(
            [1, self.sequence_length], dtype=original_entity_context.dtype
        )
        output_masked = t.zeros(
            [1, self.sequence_length], dtype=masked_entity_context.dtype
        )

        output_original[0, 0] = self.tokenizer.cls_token_id
        output_original[
            0, 1 : 1 + self.context_length
        ] = original_entity_context.squeeze(0)
        output_original[0, 1 + self.context_length] = self.tokenizer.sep_token_id
        output_original[0, 2 + self.context_length : -1] = original_relation
        output_original[0, -1] = self.tokenizer.sep_token_id

        output_masked[0, 0] = self.tokenizer.cls_token_id
        output_masked[0, 1 : 1 + self.context_length] = masked_entity_context.squeeze(0)
        output_masked[0, 1 + self.context_length] = self.tokenizer.sep_token_id
        output_masked[0, 2 + self.context_length : -1] = masked_relation
        output_masked[0, -1] = self.tokenizer.sep_token_id

        return output_original, output_masked

    def generate_sample_of_relation_encode(self, split: str):
        """
        Args:
            split: Name of split, "train" or "validate".

        Returns:
            Relation id from entity 1 to entity 2, int.
            Masked token ids (context + relations), of entity 1.
            Masked token ids (context + relations), of entity 2.
        """
        # first select a random statement(edge)
        edge_id = random.choice(self.relation_encode_splits[split])
        edge = self.db.statements.find_one({"_id": ObjectId(edge_id)})
        source_entity_sample = self.generate_sample_of_entity_encode(
            item_id=edge["source_item_id"]
        )
        target_entity_sample = self.generate_sample_of_entity_encode(
            item_id=edge["target_item_id"]
        )
        relation = edge["edge_property_id"]
        return relation, source_entity_sample[1], target_entity_sample[1]

    def insert_statements_with_pages(self, path):
        logging.info(f"Getting item ids of pages, this will take some time!")
        page_item_ids = {
            p["item_id"] for p in self.db.page.find({}, {"_id": 0, "item_id": 1})
        }
        logging.info(f"Begin inserting statements, this will take a LONG time!")
        if "statements" in self.db.list_collection_names():
            self.db.statements.drop()
        with open(path, "r") as csv_file:
            reader = csv.reader(csv_file)
            # skip header
            next(reader, None)

            insert_count = 0
            documents = []
            for row in reader:
                source_item_id = int(row[0])
                edge_property_id = int(row[1])
                target_item_id = int(row[2])
                if source_item_id in page_item_ids and target_item_id in page_item_ids:
                    documents.append(
                        {
                            "source_item_id": source_item_id,
                            "edge_property_id": edge_property_id,
                            "target_item_id": target_item_id,
                        }
                    )
                if len(documents) >= 1000000:
                    self.db.statements.insert_many(documents)
                    logging.info(f"Inserted {insert_count} statements in total.")
                    insert_count += len(documents)
                    documents = []
            if len(documents) > 0:
                self.db.statements.insert_many(documents)
                insert_count += len(documents)
            logging.info(f"Inserted {insert_count} statements in total.")

    def remove_statements_without_pages(self):
        # DEPRECATED
        pass

        # remove statements whose source_item_id or target_item_id is not
        # an item id of a page.
        # Will remove about 3/2 entries of 141206853 entries in total
        logging.info(
            "Removing statements with one end not connected to a page, "
            "this operation will take a LONG time!"
        )

        # operation batch
        remove_count = 0
        ops = []

        # delete statements whose source_item_id doesn't match
        remove = self.db.statements.aggregate(
            [
                {
                    "$lookup": {
                        "from": "page",
                        "localField": "source_item_id",
                        "foreignField": "item_id",
                        "as": "result",
                    }
                },
                {"$match": {"result.page_id": {"$exists": False}}},
                {"$project": {"_id": 1}},
            ]
        )
        for rem in remove:
            ops.append({"deleteOne": {"filter": {"_id": rem["_id"]}}})
            if len(ops) >= 1000:
                self.db.statements.bulk_write(ops)
                remove_count += len(ops)
                logging.info(f"Removed {remove_count} entries in total.")
                ops = []
        if len(ops) > 0:
            self.db.statements.bulk_write(ops)
            remove_count += len(ops)
            logging.info(f"Removed {remove_count} entries in total.")
            ops = []

        # delete statements whose target_item_id doesn't match
        remove = self.db.statements.aggregate(
            [
                {
                    "$lookup": {
                        "from": "page",
                        "localField": "target_item_id",
                        "foreignField": "item_id",
                        "as": "result",
                    }
                },
                {"$match": {"result.page_id": {"$exists": False}}},
                {"$project": {"_id": 1}},
            ]
        )
        for rem in remove:
            ops.append({"deleteOne": {"filter": {"_id": rem["_id"]}}})
            if len(ops) >= 1000:
                self.db.statements.bulk_write(ops)
                remove_count += len(ops)
                logging.info(f"Removed {remove_count} entries in total.")
                ops = []
        if len(ops) > 0:
            self.db.statements.bulk_write(ops)
            remove_count += len(ops)
            logging.info(f"Removed {remove_count} entries in total.")
        logging.info(f"Finish removing disconnected statements.")

    def select_relations(self):
        all_relations_size = self.db.property.count({})
        all_relations = self.db.property.find({}, {"property_id": 1})
        assert self.relation_size < all_relations_size
        # generate a histogram to select most occurred relations
        top_relations = self.db.statements.aggregate(
            [
                {"$group": {"_id": "$edge_property_id", "count": {"$sum": 1}}},
                {"$sort": {"count": DESCENDING}},
                {"$limit": self.relation_size - self.relation_index_start},
            ]
        )
        top_relations = [r["_id"] for r in top_relations]
        top_relation_names = self.db.property.find(
            {"property_id": {"$in": top_relations}}, {"property_id": 1, "en_label": 1},
        )
        for item in top_relation_names:
            self.property_to_relation_mapping[item["property_id"]] = len(
                self.relation_names
            )
            self.relation_names.append(f"<{item['en_label']}>")

        assert len(self.relation_names) == self.relation_size
        for item in all_relations:
            if item["property_id"] not in top_relations:
                self.property_to_relation_mapping[
                    item["property_id"]
                ] = self.unknown_relation_id

    def generate_split_for_entity_encode(self):
        logging.info(
            f"Generating splits for entity encoding, seed={self.permute_seed}."
        )
        logging.info(
            f"Note: split index of entity encoding may take 100 MiB of memory."
        )
        # split pages into train and validate set
        page_ids = [p["page_id"] for p in self.db.page.find({}, {"page_id": 1})]
        rnd = random.Random(self.permute_seed)
        rnd.shuffle(page_ids)
        split = int(self.train_entity_encode_ratio * len(page_ids))
        self.entity_encode_splits["train"] = page_ids[:split]
        self.entity_encode_splits["validate"] = page_ids[split:]
        logging.info(
            f"Entity encoding split size: "
            f"train={len(self.entity_encode_splits['train'])}, "
            f"validate={len(self.entity_encode_splits['validate'])}"
        )

    def generate_split_for_relation_encode(self):
        logging.info(
            f"Generating splits for relation encoding, seed={self.permute_seed}."
        )
        logging.info(
            f"Note: split index of relation encoding may take 3 GiB of memory."
        )
        # split statements into train and validate set
        statement_ids = [str(p["_id"]) for p in self.db.statements.find({}, {"_id": 1})]
        rnd = random.Random(self.permute_seed)
        rnd.shuffle(statement_ids)
        split = int(self.train_relation_encode_ratio * len(statement_ids))
        self.relation_encode_splits["train"] = statement_ids[:split]
        self.relation_encode_splits["validate"] = statement_ids[split:]
        logging.info(
            f"Relation encoding split size: "
            f"train={len(self.relation_encode_splits['train'])}, "
            f"validate={len(self.relation_encode_splits['validate'])}"
        )

    def restore(self, path):
        save = pickle.load(open(path, "rb"))
        for k, v in save.items():
            setattr(self, k, v)
        assert self.relation_size == len(self.relation_names), (
            f"Relation size changed, "
            f"saved cache is {len(self.relation_names)}, "
            f"input is {self.relation_size}"
        )

    def save(self, path):
        with open_file_with_create_directories(path, "wb") as file:
            pickle.dump(
                {
                    "relation_names": self.relation_names,
                    "property_to_relation_mapping": self.property_to_relation_mapping,
                    "entity_encode_splits": self.entity_encode_splits,
                    "relation_encode_splits": self.relation_encode_splits,
                },
                file,
                protocol=4,
            )

    def open_db(self):
        if self.db is None:
            self.db = connect_to_database(self.mongo_docker_host, self.db_port, "kdwd")

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.relation_size,
                self.context_length,
                self.sequence_length,
                self.tokenizer,
                self.graph_depth,
                self.permute_seed,
                self.train_relation_encode_ratio,
                self.train_relation_encode_ratio,
                self.local_root_path,
                self.mongo_docker_name,
                self.mongo_docker_host,
                self.mongo_docker_api_host,
                self.force_reload,
            ),
        )
