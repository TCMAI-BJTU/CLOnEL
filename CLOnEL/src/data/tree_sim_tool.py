# -*- coding: utf-8 -*-
# @Time    : 2024/10/9 12:17
# @Author  : Rui Hua
# @Email   : 
# @File    : tree_sim_tool.py
# @Software: PyCharm
import math
import os
from functools import lru_cache


class TreeCodeSimilarity():
    def __init__(self, tree_code_set):
        self.root = "01"

        self.tree_code2child_num = self.get_tree_code2child_num(tree_code_set)

    def get_tree_code2child_num(self, tree_code_set):
        sorted_codes = sorted(tree_code_set)

        tree_code2child_num = {tree_code: 0 for tree_code in tree_code_set}

        for i, tree_code in enumerate(sorted_codes):
            for j in range(i + 1, len(sorted_codes)):
                if sorted_codes[j].startswith(tree_code):
                    tree_code2child_num[tree_code] += 1
                else:
                    break

        tree_code2child_num = {
            f"{self.root}." + k: max(v + 1, 1) for k, v in tree_code2child_num.items()
        }
        tree_code2child_num[self.root] = len(tree_code2child_num)

        return tree_code2child_num

    @lru_cache(maxsize=None)
    def freq(self, c):
        try:
            return self.tree_code2child_num[c]
        except:
            print()

    @lru_cache(maxsize=None)
    def IC(self, c):
        return -math.log(self.freq(c) / self.freq(self.root))

    def lca(self, c1, c2):
        # 最近公共祖先
        c1 = c1.split(".")
        c2 = c2.split(".")
        res = []
        for i, num in enumerate(c1):
            if i < len(c2) and num == c2[i]:
                res.append(num)
            else:
                break
        return ".".join(res)

    @lru_cache(maxsize=None)
    def compute(self, c1, c2):
        c1 = f"{self.root}." + c1
        c2 = f"{self.root}." + c2
        top = 2 * self.IC(self.lca(c1, c2))
        down = self.IC(c1) + self.IC(c2)
        return abs(top / down)


class TreeSimTool():
    def __init__(self, dataset_name_or_path, task_label, tree_ratio):
        self.tree_ratio = tree_ratio
        dictionary_path = os.path.join(dataset_name_or_path, f"exp_{task_label}", "dictionary.txt")

        tree_code_sets = set()
        lines = open(dictionary_path, "r").readlines()
        for line in lines:
            tree_number, dict_name = line.strip().split("\t")
            tree_code_sets.add(tree_number)

        self.tree_similarities = TreeCodeSimilarity(tree_code_sets)

    def compute_similarity(self, tree1, tree2):
        sim = self.tree_similarities.compute(tree1, tree2)
        return sim * self.tree_ratio
