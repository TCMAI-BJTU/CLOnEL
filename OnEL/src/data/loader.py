# -*- coding: utf-8 -*-
# @Time    : 2024/8/4 16:57
# @Author  : Rui Hua
# @Email   : 
# @File    : loader.py
# @Software: PyCharm
import glob
import os
import numpy as np
from torch.utils.data import DataLoader
from src.data import CandidateDataset
from src.logger import setup_logger


def load_queries(data_dir, dataset_name_or_path, stage="train"):
    def process_concepts(concepts):
        for concept in concepts:
            concept = concept.split("||")
            query_type = concept[2].strip()
            mention = concept[3].strip()
            cui = concept[4].strip()

            if stage == "train":
                for m in mention.replace("+", "|").split("|"):
                    data.append((m, cui, query_type))
            else:
                data.append((mention, cui, query_type))

    data = []
    if dataset_name_or_path in ["cometa-cf", "sympel"]:
        with open(data_dir, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                mention, cui = line.strip().split("||")
                data.append((mention, cui, "cometa2"))
    elif dataset_name_or_path in ['aap']:
        with open(data_dir, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                _, mention, cui = line.strip().split("||")
                data.append((mention, cui, "aap"))
    elif dataset_name_or_path in ["ncbi-disease", "bc5cdr-chemical", "bc5cdr-disease"]:
        concept_files = glob.glob(os.path.join(data_dir, "*.concept"))
        for concept_file in concept_files:
            with open(concept_file, "r", encoding='utf-8') as f:
                concepts = f.readlines()

            process_concepts(concepts)
    else:
        raise ValueError("Invalid dataset name or path")
    if stage == "train":
        data = list(dict.fromkeys(data))
    data = np.array(data)

    return data


def load_dictionary(dictionary_path):
    data = []
    lines = open(dictionary_path, mode='r', encoding='utf-8').readlines()
    for line in lines:
        line = line.strip()
        if line == "": continue
        cui, name = line.split("||")
        data.append((name, cui))
    data = np.array(data)
    return data


def load_data(args):
    train_queries = load_queries(
        args.train_dir,
        args.dataset_name_or_path,
        stage="train"
    )
    train_dictionaries = load_dictionary(args.train_dictionary_path)
    if args.debug:
        train_queries = train_queries[:500]
        train_dictionaries = train_dictionaries[:3000]
    logger = setup_logger(args.log_file)

    logger.info(f"train_queries:{len(train_queries)}")
    logger.info(f"train_dictionaries:{len(train_dictionaries)}")

    train_dataset = CandidateDataset(args, train_queries, train_dictionaries)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    return train_dataset, train_loader
