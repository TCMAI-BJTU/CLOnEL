# -*- coding: utf-8 -*-
# @Time    : 2024/7/25 18:16
# @Author  : Rui Hua
# @Email   : 
# @File    : cl_dataset.py
# @Software: PyCharm
import random
import pandas as pd

random_seed = 2222

import copy
import os
import shutil
from collections import defaultdict
from typing import Dict
import numpy as np
from tqdm import tqdm


class ContinualLearningDataset(object):
    def __init__(self, use_data_strategy, language):
        self.use_data_strategy = use_data_strategy
        self.language = language
        self.entities_file = f"data_{self.language}_V2/dictionary.txt"
        self.mentions_file = f"data_{self.language}_V2/mentions.txt"

        self.all_term2node = self._load_all_term2node()
        self.all_dataset = self._load_all_dataset()
        self.all_nodes = self._load_all_nodes()

        self.current_experience = 0
        self.n_experiences = 10
        self.nodes_init_ratio = 0.1
        self.nodes_update_num = int((1 - self.nodes_init_ratio) * len(self.all_nodes) / (self.n_experiences - 1))

        self.current_nodes = []
        self.current_entities = []
        self.current_train_dataset = []
        self.current_test_dataset = []
        self.history_train_dataset = []

        self.dev_dataset = []

        self.train_dataset, self.test_dataset = self._init_dataset()

    def _init_dataset(self):
        train_test_split_ratio = 0.8
        shuffle_indices = list(range(len(self.all_dataset)))
        random.seed(random_seed)

        random.shuffle(list(range(len(shuffle_indices))))
        random.shuffle(shuffle_indices)

        train_indices = shuffle_indices[:int(len(shuffle_indices) * train_test_split_ratio)]
        test_indices = shuffle_indices[int(len(shuffle_indices) * train_test_split_ratio):]
        train_dataset = np.array(self.all_dataset)[train_indices].tolist()
        test_dataset = np.array(self.all_dataset)[test_indices].tolist()

        return train_dataset, test_dataset

    def _load_all_dataset(self):
        all_dataset = []
        with open(self.mentions_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                node, entity = line.strip().split("\t")
                all_dataset.append((node, entity))
        return all_dataset

    def _load_all_nodes(self):
        all_tree = []
        with open(self.entities_file, "r") as f:
            entities = f.read().splitlines()
            for entity in entities:
                code, entity = entity.split("\t")
                all_tree.append(code)
        all_tree = dict.fromkeys(all_tree).keys()
        root_tree = [node for node in all_tree if len(node.split(".")) <= 2]
        root_tree = sorted(root_tree, key=lambda x: len(x))
        all_tree = [node for node in all_tree if len(node.split(".")) > 2]
        all_tree = sorted(all_tree, key=lambda x: x)
        all_tree = root_tree + all_tree
        return all_tree

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

    def get_parent_node(self, node):
        for i in range(len(node.split(".")), 1, -1):
            parent_node = ".".join(node.split(".")[:i])
            if parent_node in self.current_nodes:
                return parent_node
        raise ValueError(f"Node {node} not in current nodes")

    def _update_nodes(self):
        init_nodes_num = int(len(self.all_nodes) * self.nodes_init_ratio)

        if self.current_experience == 0:
            # 按照entity_init_ratio初始化实体
            self.current_nodes = self.all_nodes[:init_nodes_num]
        elif self.current_experience == self.n_experiences - 1:
            # 最后一轮更新所有实体
            self.current_nodes = self.all_nodes
        else:
            # 按照entity_update_ratio更新实体
            r_pos = init_nodes_num + self.nodes_update_num * self.current_experience
            self.current_nodes = self.all_nodes[:r_pos]

    def _update_all_dataset_with_current_nodes(self):
        '''
        训练集是需要按照experience拆分的，所以需要单独更新，验证集和测试集全部更新，但是需要每轮保存结果
        :return:
        '''
        self._update_current_train_dataset()
        self._update_current_test_dataset()

    def _update_current_train_dataset(self):
        # 保存历史的所有训练集
        self.history_train_dataset = copy.deepcopy(self.current_train_dataset) + self.history_train_dataset
        l_pos = int(len(self.train_dataset) * self.current_experience / self.n_experiences)
        r_pos = int(len(self.train_dataset) * (self.current_experience + 1) / self.n_experiences)
        self.current_train_dataset = copy.deepcopy(self.train_dataset[l_pos:r_pos])

        # 如果当前的字典里还没有加入该节点，先将标签设置为NIL
        for i, (node, mention) in enumerate(self.current_train_dataset):
            gold_node = self.all_term2node[mention]
            current_node = self.get_parent_node(gold_node) if gold_node not in self.current_nodes else gold_node

            self.current_train_dataset[i] = (current_node, mention)

            if self.current_experience == 0 and current_node == gold_node:
                self.dev_dataset.append((current_node, mention))

        if self.use_data_strategy:
            cnt = 0
            self.history_train_dataset = sorted(self.history_train_dataset, key=lambda x: (-len(x[0])))
            for i, (node, mention) in enumerate(self.history_train_dataset):
                gold_node = self.all_term2node[mention]
                current_node = self.get_parent_node(gold_node) if gold_node not in self.current_nodes else gold_node
                if current_node != node:
                    self.history_train_dataset[i] = (current_node, mention)
                    if cnt >= 20:
                        break
                    cnt += 1
                    print(self.current_experience, " update ", mention, "node", node, " to ", gold_node)
                    self.current_train_dataset.append((current_node, mention))


    def _update_current_test_dataset(self):
        self.current_test_dataset = copy.deepcopy(self.test_dataset)

        for i, (node, mention) in enumerate(self.current_test_dataset):
            gold_node = self.all_term2node[mention]
            current_node = self.get_parent_node(node) if gold_node not in self.current_nodes else gold_node
            self.current_test_dataset[i] = (current_node, mention)

            if self.current_experience == 0 and current_node == gold_node:
                self.dev_dataset.append((current_node, mention))

    def _save(self, train_experiences, test_experiences, nodes_experiences):
        node2terms = defaultdict(set)
        with open(self.entities_file, "r") as f:
            entities = f.read().splitlines()
            for entity in entities:
                code, entity = entity.split("\t")
                node2terms[code].add(entity)
        if self.use_data_strategy:
            output_dir = f"cl_data_strategy_{self.language}_{random_seed}_pnode_V2"
        else:
            output_dir = f"cl_no_data_strategy_{self.language}_{random_seed}_pnode_V2"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for i in range(self.n_experiences):
            if os.path.exists(os.path.join(output_dir, "exp_" + str(i))):
                shutil.rmtree(os.path.join(output_dir, "exp_" + str(i)))
            os.mkdir(os.path.join(output_dir, "exp_" + str(i)))

            train_df = pd.DataFrame(sorted(train_experiences[i], key=lambda x: (len(x[0]), x[0])))
            train_df.to_csv(os.path.join(output_dir, "exp_" + str(i), "train.txt"), index=False, header=False, sep="\t")

            dev_df = pd.DataFrame(sorted(self.dev_dataset, key=lambda x: (len(x[0]), x[0])))
            dev_df.to_csv(os.path.join(output_dir, "exp_" + str(i), "dev.txt"), index=False, header=False, sep="\t")

            test_df = pd.DataFrame(sorted(test_experiences[i], key=lambda x: (len(x[0]), x[0])))
            test_df.to_csv(os.path.join(output_dir, "exp_" + str(i), "test.txt"), index=False, header=False, sep="\t")

            with open(os.path.join(output_dir, "exp_" + str(i), "dictionary.txt"), "w") as f:
                for node in nodes_experiences[i]:
                    terms = node2terms[node]
                    for term in terms:
                        f.write(node + "\t" + term + "\n")

    def load_all_experiences(self):
        train_experiences = []
        test_experiences = []
        nodes_experiences = []

        for self.current_experience in tqdm(range(self.n_experiences), desc="Loading experiences"):
            self._update_nodes()
            self._update_all_dataset_with_current_nodes()

            train_experiences.append(self.current_train_dataset)
            test_experiences.append(self.current_test_dataset)

            nodes_experiences.append(self.current_nodes)

        self._save(train_experiences, test_experiences, nodes_experiences)
        return train_experiences, test_experiences, nodes_experiences


if __name__ == '__main__':
    use_data_strategy_ls = [True, False]
    language_ls = ["EN"]
    for use_data_strategy in use_data_strategy_ls:
        for language in language_ls:
            dataset = ContinualLearningDataset(use_data_strategy, language)
            dataset.load_all_experiences()
