import os
import csv
import logging
import random
import h5py
import numpy as np
import pickle
import pprint
import itertools
import torch as t
import multiprocessing as mp
from typing import Callable, List, Tuple
from tqdm import tqdm
from pymongo import ASCENDING, DESCENDING
from transformers import PreTrainedTokenizerBase
from kb_ae_bert.utils.token import get_context_of_masked
from kb_ae_bert.utils.settings import (
    dataset_cache_dir,
    mongo_config,
    preprocess_cache_dir,
    preprocess_worker_num,
)
from kb_ae_bert.utils.file import open_file_with_create_directories
from kb_ae_bert.utils.kaggle import download_dataset
from kb_ae_bert.utils.mongo import MongoDBHandler
from ..base import StaticIterableDataset

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def default_mask_function(seq: List[int], mask_id: int):
    return [mask_id if np.random.rand() < 0.15 else seq_item for seq_item in seq]


class KDWDBertDataset:
    CACHE_NAME = "kdwd_bert"
    MP_INSTANCE = None

    def __init__(
        self,
        relation_size: int,
        context_length: int,
        sequence_length: int,
        tokenizer: PreTrainedTokenizerBase,
        graph_depth: int = 1,
        permute_seed: int = 2147483647,
        train_entity_encode_ratio: float = 0.9,
        train_relation_encode_ratio: float = 0.9,
        force_reload: bool = False,
        generate_data: bool = True,
        mask_function: Callable[[List[int], int], List[int]] = default_mask_function,
    ):
        """
        Note:
            The result hdf5 file may need 300GiB of disk space to store.

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
            force_reload: Whether force reloading and preprocessing KDWD dataset into
                the mongo database.
            generate_data: Whether generate preprocessed data.
            mask_function: The mask function used to mask the relation section
                of the output sequence.
        """
        self.relation_size = relation_size
        self.context_length = context_length
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.graph_depth = graph_depth
        self.permute_seed = permute_seed
        self.torch_generator = t.Generator().manual_seed(permute_seed)
        self.train_entity_encode_ratio = train_entity_encode_ratio
        self.train_relation_encode_ratio = train_relation_encode_ratio
        self.mongo_db_handler = MongoDBHandler(**mongo_config)
        self.force_reload = force_reload
        self.mask_function = mask_function

        self.db = self.mongo_db_handler.get_database("kdwd")
        collection_names = self.db.list_collection_names()
        is_db_reusable = (
            "item" in collection_names
            and "statements" in collection_names
            and "page" in collection_names
            and "link_annotated_text" in collection_names
        )
        if not is_db_reusable or force_reload:
            kaggle_path = str(os.path.join(dataset_cache_dir, "kaggle"))

            # load dataset into the database
            logging.info(
                f"Existing collections are {collection_names}, "
                f"required are: item, statements, page, link_annotated_text, "
                f"regenerating."
            )
            download_dataset(
                "kenshoresearch/kensho-derived-wikimedia-data", kaggle_path
            )

            # we don't load statements.csv as we need to preprocess it
            self.mongo_db_handler.load_dataset_files(
                "kdwd",
                str(os.path.join(kaggle_path, "kensho-derived-wikimedia-data")),
                [
                    "item.csv",
                    "item_aliases.csv",
                    "link_annotated_text.jsonl",
                    "page.csv",
                    "property.csv",
                    "property_aliases.csv",
                ],
            )

            self.insert_statements_with_pages(
                os.path.join(
                    kaggle_path, "kensho-derived-wikimedia-data", "statements.csv",
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

        self.unknown_relation_id = 0
        self.sub_part_relation_id = 1
        self.relation_index_start = 2

        # Generate states cache
        if os.path.exists(
            os.path.join(preprocess_cache_dir, f"{self.CACHE_NAME}.cache")
        ):
            logging.info("Found states cache for KDWD, skipping generation.")
            self.restore(os.path.join(preprocess_cache_dir, f"{self.CACHE_NAME}.cache"))
        else:
            logging.info("States cache for KDWD not found, generating.")
            # relations
            self.relation_names = ["<unknown relation>", "<is a sub token of>"]
            # maps property id in statements to a relation id
            self.property_to_relation_mapping = {}
            logging.info("Selecting relations by usage.")
            self.select_relations()

            # generate train/validate splits
            # Dict["split_name: str", List[Tuple["page_id: int", "item_id: int"]]]
            self.entity_encode_splits = {}
            # Dict["split_name: str",
            # List[Tuple["source_item_id: int",
            #            "target_item_id: int",
            #            "property_id: int"]]]
            self.relation_encode_splits = {}

            logging.info("Generating splits.")
            self.generate_split_for_entity_encode()
            self.generate_split_for_relation_encode()

            self.save(os.path.join(preprocess_cache_dir, f"{self.CACHE_NAME}.cache"))

        # close connection to db and data file, reopen it later, to support forking
        self.db = None
        self.entity_file = None
        self.relation_file = None

        # Generate data
        self.is_entity_data_generated = False
        self.is_relation_data_generated = False
        if os.path.exists(
            os.path.join(preprocess_cache_dir, f"{self.CACHE_NAME}_entity_data.hdf5")
        ):
            try:
                with h5py.File(
                    os.path.join(
                        preprocess_cache_dir, f"{self.CACHE_NAME}_entity_data.hdf5"
                    ),
                    "r",
                ) as file:
                    assert len(file["entity"]["train"]) > 0
                    assert len(file["entity"]["validate"]) > 0
            except Exception as e:
                logging.info(
                    f"Exception [{str(e)}] occurred while reading entity data file, "
                    f"regenerating data."
                )
                os.remove(
                    os.path.join(
                        preprocess_cache_dir, f"{self.CACHE_NAME}_entity_data.hdf5"
                    )
                )
            else:
                logging.info("Found entity data for KDWD, skipping generation.")
                self.is_entity_data_generated = True
        if os.path.exists(
            os.path.join(preprocess_cache_dir, f"{self.CACHE_NAME}_relation_data.hdf5")
        ):
            try:
                with h5py.File(
                    os.path.join(
                        preprocess_cache_dir, f"{self.CACHE_NAME}_relation_data.hdf5"
                    ),
                    "r",
                ) as file:
                    assert len(file["relation"]["train"]) > 0
                    assert len(file["relation"]["validate"]) > 0
            except Exception as e:
                logging.info(
                    f"Exception [{str(e)}] occurred while reading relation data file, "
                    f"regenerating data."
                )
                os.remove(
                    os.path.join(
                        preprocess_cache_dir, f"{self.CACHE_NAME}_relation_data.hdf5"
                    )
                )
            else:
                logging.info("Found relation data for KDWD, skipping generation.")
                self.is_relation_data_generated = True
        if (
            not self.is_entity_data_generated or not self.is_relation_data_generated
        ) and generate_data:
            self.generate_data()
            self.is_entity_data_generated = True
            self.is_relation_data_generated = True

    @staticmethod
    def get_loss(batch, relation_logits):
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

    def generate_data(self):
        if not self.is_entity_data_generated:
            with h5py.File(
                os.path.join(
                    preprocess_cache_dir, f"{self.CACHE_NAME}_entity_data.hdf5"
                ),
                "w",
                rdcc_nbytes=1024 ** 3,
            ) as file:
                self.generate_preprocessed_data_of_entity_encode(file)
        if not self.is_relation_data_generated:
            with h5py.File(
                os.path.join(
                    preprocess_cache_dir, f"{self.CACHE_NAME}_entity_data.hdf5"
                ),
                "r",
                rdcc_nbytes=1024 ** 3,
            ) as entity_file, h5py.File(
                os.path.join(
                    preprocess_cache_dir, f"{self.CACHE_NAME}_relation_data.hdf5"
                ),
                "w",
                rdcc_nbytes=1024 ** 3,
            ) as file:
                self.generate_preprocessed_data_of_relation_encode(entity_file, file)

    @property
    def train_entity_encode_dataset(self):
        return StaticIterableDataset(
            len(self.entity_encode_splits["train"]),
            self.generator_of_entity_encode,
            ("train",),
        )

    @property
    def validate_entity_encode_dataset(self):
        return StaticIterableDataset(
            len(self.entity_encode_splits["validate"]),
            self.generator_of_entity_encode,
            ("validate",),
        )

    @property
    def train_relation_encode_dataset(self):
        return StaticIterableDataset(
            len(self.relation_encode_splits["train"]),
            self.generator_of_relation_encode,
            ("train",),
        )

    @property
    def validate_relation_encode_dataset(self):
        return StaticIterableDataset(
            len(self.relation_encode_splits["validate"]),
            self.generator_of_relation_encode,
            ("validate",),
        )

    def generator_of_entity_encode(self, index: int, split: str):
        if not self.is_entity_data_generated:
            raise RuntimeError("Data not generated.")
        self.open_file()
        sample = (
            t.from_numpy(self.entity_file["entity"][split][index])
            .to(dtype=t.long)
            .view(2, -1)
        )

        shape = (1, self.sequence_length)
        device = sample.device

        attention_mask = t.ones(shape, dtype=t.float32, device=device)
        token_type_ids = t.ones(shape, dtype=t.long, device=device)
        token_type_ids[:, : 2 + self.context_length] = 0
        return {
            "input_ids": sample[1].view(shape),
            "labels": sample[0].view(shape),
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def generator_of_relation_encode(self, index: int, split: str):
        if not self.is_relation_data_generated:
            raise RuntimeError("Data not generated.")
        self.open_file()
        relation = int(self.relation_file["relation"][split][index][0])
        # original_src, masked_src, original_tar, masked_tar
        sample = (
            t.from_numpy(self.relation_file["relation"][split][index][1:])
            .to(dtype=t.long)
            .view(2, -1)
        )

        shape = (1, self.sequence_length)
        device = sample.device

        attention_mask = t.ones(shape, dtype=t.float32, device=device)
        token_type_ids = t.ones(shape, dtype=t.long, device=device)
        token_type_ids[:, : 2 + self.context_length] = 0
        # randomly switch direction of relation
        if random.choice([0, 1]) == 0:
            return {
                "input_ids_1": sample[0].view(shape),
                "input_ids_2": sample[1].view(shape),
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "relation": t.tensor([relation], dtype=t.long, device=sample.device),
                "direction": t.tensor([0], dtype=t.long, device=sample.device),
            }
        else:
            return {
                "input_ids_1": sample[0].view(shape),
                "input_ids_2": sample[1].view(shape),
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "relation": t.tensor([relation], dtype=t.long, device=sample.device),
                "direction": t.tensor([1], dtype=t.long, device=sample.device),
            }

    def print_sample_of_entity_encode(self, split: str = None, item_id: int = None):
        if split is not None:
            item_id = random.choice(self.entity_encode_splits[split])[1]
        if not self.is_entity_data_generated:
            print("Note: Preprocessed data not generated, generate using database.")
            original, masked = self.generate_sample_of_entity_encode(item_id)
            original, masked = original[0].tolist(), masked[0].tolist()
        else:
            print("Note: Preprocessed data found, generate using preprocessed data.")
            self.open_file()
            split, index = self.find_entity(item_id)
            original, masked = self.entity_file["entity"][split][index].reshape(2, -1)
            original, masked = original.tolist(), masked.tolist()

        print(f"Item id: {item_id}")
        print("Original:")
        print(self.tokenizer.decode(original))
        print("Masked:")
        print(self.tokenizer.decode(masked))

    def find_entity(self, item_id):
        for index, (_, entity_item_id) in enumerate(self.entity_encode_splits["train"]):
            if entity_item_id == item_id:
                return "train", index
        for index, (_, entity_item_id) in enumerate(
            self.entity_encode_splits["validate"]
        ):
            if entity_item_id == item_id:
                return "validate", index
        raise ValueError(f"Item id {item_id} not found in train or validate set.")

    def print_sample_of_relation_encode(
        self, split: str = None, edge: Tuple[int, int, int] = None
    ):
        if edge is None:
            edge = random.choice(self.relation_encode_splits[split])
        if not self.is_relation_data_generated or not self.is_entity_data_generated:
            print("Note: Preprocessed data not generated, generate using database.")
            (
                source_entity_original,
                source_entity_masked,
            ) = self.generate_sample_of_entity_encode(item_id=edge[0])
            (
                target_entity_original,
                target_entity_masked,
            ) = self.generate_sample_of_entity_encode(item_id=edge[1])

            relation = self.property_to_relation_mapping[edge[2]]
            source_entity_original = source_entity_original[0].tolist()
            source_entity_masked = source_entity_masked[0].tolist()
            target_entity_original = target_entity_original[0].tolist()
            target_entity_masked = target_entity_masked[0].tolist()
        else:
            print("Note: Preprocessed data found, generate using preprocessed data.")
            self.open_file()
            src_index, tar_index, edge_index = self.find_relation(*edge)
            source_entity_original, source_entity_masked = self.entity_file["entity"][
                src_index[0]
            ][src_index[1]].reshape(2, -1)
            target_entity_original, target_entity_masked = self.entity_file["entity"][
                tar_index[0]
            ][tar_index[1]].reshape(2, -1)

            edge_data = self.relation_file["relation"][edge_index[0]][edge_index[1]]
            relation = edge_data[0]
            edge_source_masked, edge_target_masked = edge_data[1:].reshape(2, -1)
            assert np.array_equal(edge_source_masked, source_entity_masked), (
                f"Entity and edge source doesn't match: "
                f"\nEdge: {self.tokenizer.decode(edge_source_masked.tolist())}"
                f"\nEntity: {self.tokenizer.decode(source_entity_masked.tolist())}"
            )
            assert np.array_equal(edge_target_masked, target_entity_masked), (
                f"Entity and edge target doesn't match: "
                f"\nEdge: {self.tokenizer.decode(edge_target_masked.tolist())}"
                f"\nEntity: {self.tokenizer.decode(target_entity_masked.tolist())}"
            )

            source_entity_original = source_entity_original.tolist()
            source_entity_masked = source_entity_masked.tolist()
            target_entity_original = target_entity_original.tolist()
            target_entity_masked = target_entity_masked.tolist()

        print("Relation:")
        print(self.relation_names[relation])
        print("Original 1:")
        print(self.tokenizer.decode(source_entity_original))
        print("Masked 1:")
        print(self.tokenizer.decode(source_entity_masked))
        print("Original 2:")
        print(self.tokenizer.decode(target_entity_original))
        print("Masked 2:")
        print(self.tokenizer.decode(target_entity_masked))

    def find_relation(self, src_item_id, tar_item_id, property_id):
        src_index = self.find_entity(src_item_id)
        tar_index = self.find_entity(tar_item_id)
        edge = (src_item_id, tar_item_id, property_id)
        try:
            edge_index = ("train", self.relation_encode_splits["train"].index(edge))
        except ValueError:
            try:
                edge_index = (
                    "validate",
                    self.relation_encode_splits["validate"].index(edge),
                )
            except ValueError:
                raise ValueError(f"Edge {edge} not found in train or validate set.")
        return src_index, tar_index, edge_index

    def generate_preprocessed_data_of_entity_encode(self, hdf5_file: h5py.File):
        group = hdf5_file.create_group("entity")
        length = self.sequence_length * 2
        train_set = list(enumerate(self.entity_encode_splits["train"]))
        validate_set = list(enumerate(self.entity_encode_splits["validate"]))
        total_num = len(train_set) + len(validate_set)
        ctx = mp.get_context("fork")

        logging.info(
            f"Generating entity encoding dataset, size: "
            f"train={len(train_set)}, "
            f"validate={len(validate_set)}"
        )
        with tqdm(
            total=total_num, desc="Processed entities", unit=" entities"
        ) as progress_bar, ctx.Pool(
            processes=preprocess_worker_num,
            initializer=self.multiprocessing_pool_initializer_of_entity_encode,
            initargs=(self,),
        ) as pool:
            train_num = len(self.entity_encode_splits["train"])
            train_dataset = group.create_dataset(
                name="train",
                shape=(train_num, length),
                dtype=np.int32,
                chunks=(1024, length),
            )
            for idx, row_data in pool.imap_unordered(
                self.preprocess_entity_encode_worker, train_set, chunksize=256,
            ):
                train_dataset[idx] = row_data
                progress_bar.update(1)

            validate_num = len(self.entity_encode_splits["validate"])
            validate_dataset = group.create_dataset(
                name="validate",
                shape=(validate_num, length),
                dtype=np.int32,
                chunks=(1024, length),
            )

            for idx, row_data in pool.imap(
                self.preprocess_entity_encode_worker, validate_set, chunksize=256,
            ):
                validate_dataset[idx] = row_data
                progress_bar.update(1)

    @staticmethod
    def multiprocessing_pool_initializer_of_entity_encode(self):
        KDWDBertDataset.MP_INSTANCE = self

    @staticmethod
    def preprocess_entity_encode_worker(args):
        self = KDWDBertDataset.MP_INSTANCE
        idx, (_page_id, item_id) = args
        result = self.generate_sample_of_entity_encode(item_id)
        return idx, t.cat(result, dim=1).flatten().numpy()

    def generate_preprocessed_data_of_relation_encode(
        self, entity_hdf5_file: h5py.File, hdf5_file: h5py.File
    ):
        entity_group = entity_hdf5_file["entity"]
        group = hdf5_file.create_group("relation")
        length = self.sequence_length * 2 + 1
        train_set = list(enumerate(self.relation_encode_splits["train"]))
        validate_set = list(enumerate(self.relation_encode_splits["validate"]))
        total_num = len(train_set) + len(validate_set)

        logging.info(
            f"Generating relation encoding dataset, size: "
            f"train={len(train_set)}, "
            f"validate={len(validate_set)}"
        )
        logging.info("Stopping mongo database for memory reservation.")
        self.mongo_db_handler.stop()

        logging.info("Loading entity data from hdf5 file")
        # Order is original -> masked, only load the masked part for more memory.
        entity_data = {
            "train": entity_group["train"][:, self.sequence_length :],
            "validate": entity_group["validate"][:, self.sequence_length :],
        }

        logging.info("Generating entity index")
        entity_index = {}
        for idx, (_, item_id) in enumerate(self.entity_encode_splits["train"]):
            entity_index[item_id] = ("train", idx)
        for idx, (_, item_id) in enumerate(self.entity_encode_splits["validate"]):
            entity_index[item_id] = ("validate", idx)

        with tqdm(
            total=total_num, desc="Processed relations", unit=" relations"
        ) as progress_bar:
            train_num = len(self.relation_encode_splits["train"])
            train_dataset = group.create_dataset(
                name="train",
                shape=(train_num, length),
                dtype=np.int32,
                chunks=(2048, length),
            )
            for idx, (src_item_id, tar_item_id, property_id) in train_set:
                src_entity_idx = entity_index.get(src_item_id, None)
                tar_entity_idx = entity_index.get(tar_item_id, None)
                if src_entity_idx is not None and tar_entity_idx is not None:
                    train_dataset[idx] = np.concatenate(
                        (
                            np.array(
                                (self.property_to_relation_mapping[property_id],),
                                dtype=np.int32,
                            ),
                            entity_data[src_entity_idx[0]][src_entity_idx[1]],
                            entity_data[tar_entity_idx[0]][tar_entity_idx[1]],
                        ),
                        axis=0,
                    )
                progress_bar.update(1)

            validate_num = len(self.relation_encode_splits["validate"])
            validate_dataset = group.create_dataset(
                name="validate",
                shape=(validate_num, length),
                dtype=np.int32,
                chunks=(2048, length),
            )

            for idx, (src_item_id, tar_item_id, property_id) in validate_set:
                src_entity_idx = entity_index.get(src_item_id, None)
                tar_entity_idx = entity_index.get(tar_item_id, None)
                if src_entity_idx is not None and tar_entity_idx is not None:
                    validate_dataset[idx] = np.concatenate(
                        (
                            np.array(
                                (self.property_to_relation_mapping[property_id],),
                                dtype=np.int32,
                            ),
                            entity_data[src_entity_idx[0]][src_entity_idx[1]],
                            entity_data[tar_entity_idx[0]][tar_entity_idx[1]],
                        ),
                        axis=0,
                    )
                progress_bar.update(1)
        logging.info("Restarting mongo database.")
        self.mongo_db_handler.start()

    def generate_sample_of_entity_encode(self, item_id: int):
        """
        Args:
            item_id: Item id of the entity.

        Returns:
            Original token ids (context + relation),
                Long Tensor of shape (1, sequence_length).
            Masked token ids (context + relation),
                Long Tensor of shape (1, sequence_length).
        """
        self.open_db()
        page = self.db.page.find_one({"item_id": item_id})
        if page is None:
            raise ValueError(f"Cannot find page with item_id={item_id}")

        page_id = page["page_id"]
        entity_name = str(page["title"])
        if len(entity_name) == 0:
            entity_name = "unknown"

        # Phase 1: generate context
        # Generate context according to the following rules:
        # 1. If the entity page is referenced in the one / multiple text sections of
        #    another page, sample one from them, mask the reference position.
        # 2. If the entity page is not referenced, fallback to use the page itself,
        #    mask the title (since title means the entity).
        # 3. Else, use the title only as the context. (Equivalent to no semantic
        #    information.)

        annotated_text = next(
            self.db.link_annotated_text.aggregate(
                [
                    {"$match": {"sections.target_page_ids": page_id}},
                    {"$sample": {"size": 1}},
                ]
            ),
            None,
        )
        if annotated_text is not None:
            possible_sections = []
            for section in annotated_text["sections"]:
                if page_id in section["target_page_ids"]:
                    text = section["text"]
                    idx = section["target_page_ids"].index(page_id)
                    link_length = section["link_lengths"][idx]
                    link_offset = section["link_offsets"][idx]
                    possible_sections.append((text, link_length, link_offset))
            if len(possible_sections) == 0:
                raise ValueError(
                    f"Cannot find page id {page_id} in "
                    f"annotated_text {pprint.pformat(annotated_text)}"
                )
            text, link_length, link_offset = random.choice(possible_sections)

            # replace link with title to generate the raw context to be used next
            context = (
                text[:link_offset]
                + f" {entity_name} "
                + text[link_offset + link_length :]
            )
            link_length = len(entity_name)
            link_offset = link_offset + 1  # include space before title
        else:
            # fallback to use the page itself, with entity being the title
            annotated_text = self.db.link_annotated_text.find_one({"page_id": page_id})
            if annotated_text is not None:
                article_body = [sec["text"] for sec in annotated_text["sections"]]
                context = entity_name + " " + " ".join(article_body)
                link_length = len(entity_name)
                link_offset = 0
            else:
                # fallback to use title only
                context = entity_name
                link_length = len(entity_name)
                link_offset = 0

        # Phase 2: generate relations
        # Generate a tree of items related to the target item
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
        graph = sorted(graph, key=lambda e: e["depth"])

        node_label_mapping = {}

        for edge in graph:
            node_label_mapping[edge["source_item_id"]] = "unknown"
            node_label_mapping[edge["target_item_id"]] = "unknown"

        # # Fill entities with the title of their connected pages,
        # # entities in the statements are guaranteed to be connected with a page
        for related_page in self.db.page.find(
            {"item_id": {"$in": list(node_label_mapping.keys())}}
        ):
            related_title = str(related_page["title"])
            if (
                len(related_title) > 0
                and node_label_mapping[related_page["item_id"]] == "unknown"
            ):
                node_label_mapping[related_page["item_id"]] = related_title

        # Phase 3: tokenize, mask, pad, and return result

        # # if entity is comprised of multiple tokens
        # # mask one random token, add relation "is sub part of"
        # # else
        # # mask the only token
        is_entity_partly_masked = (
            len(self.tokenizer.tokenize(str(entity_name), add_special_tokens=False))
            != 1
        )

        context_tokens = self.tokenizer(
            context, add_special_tokens=False, return_tensors="pt"
        )

        # # select the token in the entity to mask, if there is only one token,
        # # then it is masked.
        entity_token_indexes = {
            context_tokens.char_to_token(char_idx)
            for char_idx in range(link_offset, link_offset + link_length)
        }
        # # if there are spaces in an entity, their token index would be None
        entity_token_indexes.discard(None)
        entity_token_indexes = list(entity_token_indexes)
        masked_entity_token_index = random.choice(entity_token_indexes)
        masked_entity_token_id = context_tokens.input_ids[
            0, masked_entity_token_index
        ].item()

        original_entity_context, masked_entity_context = get_context_of_masked(
            sentence_tokens=context_tokens.input_ids,
            mask_position=t.tensor([masked_entity_token_index], dtype=t.long),
            context_length=self.context_length,
            pad_id=self.tokenizer.pad_token_id,
            mask_id=self.tokenizer.mask_token_id,
            generator=self.torch_generator,
        )

        # # generate relation
        # # There is a [PAD] token after each relation tuple.
        original_relation_list = []
        masked_relation_list = []
        for edge in graph:
            masked_source = source = node_label_mapping[edge["source_item_id"]]
            masked_target = target = node_label_mapping[edge["target_item_id"]]

            # only replace the target entity with [MASK] if it is completely masked.
            if not is_entity_partly_masked:
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
        if is_entity_partly_masked:
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
        # if masked_relation.shape[0] != original_relation.shape[0]:
        #     raise ValueError(
        #         f"""
        #             {masked_relation_list}
        #             {original_relation_list}
        #             {masked_relation}
        #             {original_relation}
        #         """
        #     )
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

    def insert_statements_with_pages(self, path):
        logging.info(f"Getting item ids of pages, this will take some time!")
        page_item_ids = {
            p["item_id"] for p in self.db.page.find({}, {"_id": 0, "item_id": 1})
        }
        logging.info(f"Begin inserting statements, this will take a LONG time!")
        if "statements" in self.db.list_collection_names():
            self.db.statements.drop()
        with open(path, "r") as csv_file, tqdm(
            desc="Inserted ", unit=" statements"
        ) as progress_bar:
            reader = csv.reader(csv_file)
            # skip header
            next(reader, None)

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
                    progress_bar.update(len(documents))
                    documents = []
            if len(documents) > 0:
                self.db.statements.insert_many(documents)
                progress_bar.update(len(documents))

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
            f"Note: split index of entity encoding may take 200 MiB of memory."
        )
        # split pages into train and validate set
        page_ids = [
            (p["page_id"], p["item_id"])
            for p in self.db.page.find({}, {"page_id": 1, "item_id": 1})
        ]
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
        statement_ids = [
            (p["source_item_id"], p["target_item_id"], p["edge_property_id"])
            for p in self.db.statements.find(
                {}, {"source_item_id": 1, "target_item_id": 1, "edge_property_id": 1}
            )
        ]
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
            self.db = self.mongo_db_handler.get_database("kdwd")

    def open_file(self):
        if self.entity_file is None:
            self.entity_file = h5py.File(
                os.path.join(
                    preprocess_cache_dir, f"{self.CACHE_NAME}_entity_data.hdf5"
                ),
                "r",
            )
        if self.relation_file is None:
            self.relation_file = h5py.File(
                os.path.join(
                    preprocess_cache_dir, f"{self.CACHE_NAME}_relation_data.hdf5"
                ),
                "r",
            )

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
                self.train_entity_encode_ratio,
                self.train_relation_encode_ratio,
                self.force_reload,
                # Disable generating data in subprocess, throw exception directly
                False,
                self.mask_function,
            ),
        )
