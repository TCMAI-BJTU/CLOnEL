# -*- coding: utf-8 -*-
# @Time    : 2024/10/7 17:28
# @Author  : Rui Hua
# @Email   : 
# @File    : medic_tree.py
# @Software: PyCharm
import pandas as pd

from collections import defaultdict
from functools import lru_cache
from tqdm import tqdm
from .tree_similarity import TreeSimilarity


class OntologyTree():
    def __init__(self, dataset_name_or_path, tree_ratio):
        self.dataset_name_or_path = dataset_name_or_path
        self.tree_ratio = tree_ratio

        if dataset_name_or_path in ["bc5cdr-disease", "ncbi-disease"]:
            df = pd.read_excel("./data/ontology/CTD_Disease_Tree.xlsx")
        elif dataset_name_or_path in ["bc5cdr-chemical"]:
            df = pd.read_excel("./data/ontology/CTD_Chemical_Tree.xlsx")
        elif dataset_name_or_path in ["aap", "cometa-cf"]:
            df = pd.read_csv("./data/ontology/SNOMED_CT_Tree.csv")
        elif dataset_name_or_path in ['sympel']:
            df = pd.read_excel("./data/ontology/ISPO_Tree.xlsx")
        else:
            raise ValueError("Invalid dataset_name_or_path.")

        id2tree_codes = defaultdict(list)
        for tup in tqdm(zip(df["ID"], df["TreeNumbers"]), total=len(df)):
            disease_id, tree_numbers = tup
            disease_id = str(disease_id)
            for tree in tree_numbers.split("|"):
                tree = tree.replace("/", ".")
                id2tree_codes[disease_id].append(tree)
        self.id2tree_codes = dict(id2tree_codes)

        tree_code_sets = set()
        for tree_codes in tqdm(id2tree_codes.values()):
            tree_code_sets.update(tree_codes)

        self.tree_similarities = TreeSimilarity(tree_code_sets)

        self.error_cuis = set()

    def get_nearest_similarity(self, tree1, tree2):
        sim = 0
        for t1 in tree1:
            for t2 in tree2:
                sim = max(sim, self.tree_similarities.compute(t1, t2))
        return sim

    def complete_cui(self, cui):
        cui = cui.replace("+", "|").split("|")[0]
        if not cui[0].isdigit():
            cui = "MESH:" + cui
        else:
            cui = "OMIM:" + cui
        return cui

    @lru_cache(maxsize=100000)
    def compute_similarity(self, id1, id2):
        if self.dataset_name_or_path in ["bc5cdr-disease", "ncbi-disease", "bc5cdr-chemical"]:
            id1 = self.complete_cui(id1)
            id2 = self.complete_cui(id2)

        try:
            tree1 = self.id2tree_codes[id1]
        except:
            self.error_cuis.add(id1)
            return 0.0
        try:
            tree2 = self.id2tree_codes[id2]
        except:
            self.error_cuis.add(id2)
            return 0.0

        sim = self.get_nearest_similarity(tree1, tree2)
        return sim * self.tree_ratio
