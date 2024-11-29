# -*- coding: utf-8 -*-
# @Time    : 2024/10/15 17:07
# @Author  : Rui Hua
# @Email   : 
# @File    : base_dataset.py
# @Software: PyCharm
import os
import random
import pandas as pd

random_seed = 2222
import numpy as np
from typing import Dict


class BaseDataset(object):
    def __init__(self, language):
        self.language = language
        self.entities_file = f"data_{self.language}_V2/dictionary.txt"
        self.mentions_file = f"data_{self.language}_V2/mentions.txt"

        self.all_term2node = self._load_all_term2node()
        self.all_mentions = self._load_all_mentions()

        print("all_mentions: ", len(self.all_mentions))

    def _init_dataset(self):
        train_test_split_ratio = 0.8
        shuffle_indices = list(range(len(self.all_mentions)))
        random.seed(random_seed)

        random.shuffle(list(range(len(shuffle_indices))))
        random.shuffle(shuffle_indices)

        train_indices = shuffle_indices[:int(len(shuffle_indices) * train_test_split_ratio)]
        test_indices = shuffle_indices[int(len(shuffle_indices) * train_test_split_ratio):]
        train_dataset = np.array(self.all_mentions)[train_indices]
        test_dataset = np.array(self.all_mentions)[test_indices]

        train_dataset = [(mention, self.all_term2node[mention]) for mention in train_dataset]
        test_dataset = [(mention, self.all_term2node[mention],) for mention in test_dataset]
        return train_dataset, test_dataset

    def _load_all_mentions(self):
        new_entities = []
        with open(self.mentions_file, "r") as f:
            entities = f.read().splitlines()
            for entity in entities:
                node, entity = entity.split("\t")
                new_entities.append(entity)
        new_entities = list(dict.fromkeys(new_entities))
        return new_entities

    def _load_all_term2node(self) -> Dict[str, str]:
        all_entities2node = {}
        with open(self.entities_file, "r") as f:
            entities = f.read().splitlines()
            for entity in entities:
                code, entity = entity.split("\t")
                all_entities2node[entity] = code
        with open(self.mentions_file, "r") as f:
            entities = f.read().splitlines()
            for entity in entities:
                code, entity = entity.split("\t")
                all_entities2node[entity] = code
        return all_entities2node

    def _load_node2cui(self):
        df = pd.read_excel("/home/huarui/pycharmProject/continual_learning/continual_data_process/data/本体_V230511.xlsx")
        node2cui = {}
        for tup in zip(df['Tree Code'], df['CUI']):
            tree_code, cui = tup
            node2cui["T" +tree_code] = cui
        return node2cui

    def main(self):
        self.train_dataset, self.test_dataset = self._init_dataset()
        node2cui_dict = self._load_node2cui()

        res_path = f"base_data_{self.language}_V2"

        if not os.path.exists(res_path):
            os.makedirs(res_path)

        self.train_dataset = sorted(self.train_dataset, key=lambda x: (len(x[1]), x[1]))
        with open(os.path.join(res_path, "train.txt"), "w") as f:
            for mention, node in self.train_dataset:
                cui = node2cui_dict[node]
                f.write(f"{mention}||{cui}\n")

        self.test_dataset = sorted(self.test_dataset, key=lambda x: (len(x[1]), x[1]))
        with open(os.path.join(res_path, "test.txt"), "w") as f:
            for mention, node in self.test_dataset:
                cui = node2cui_dict[node]
                f.write(f"{mention}||{cui}\n")

        lines = open(self.entities_file, "r").readlines()
        with open(os.path.join(res_path, "dictionary.txt"), "w") as f:
            for line in lines:
                node, entity = line.strip().split("\t")
                cui = node2cui_dict[node]
                f.write(f"{cui}||{entity}\n")


if __name__ == '__main__':
    languages = ["EN"]
    for language in languages:
        base_dataset = BaseDataset(language)
        base_dataset.main()
