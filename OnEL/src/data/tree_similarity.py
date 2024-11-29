# -*- coding: utf-8 -*-
# @Time    : 2024/6/13
# @Author  : Rui Hua
# @Email   : 
# @File    : tree_similarity.py
# @Software: PyCharm
import math
from functools import lru_cache
from tqdm import tqdm


class TreeSimilarity:
    def __init__(self, tree_code_set):
        self.root = "01"

        self.tree_code2child_num = self.get_tree_code2child_num(tree_code_set)


    def get_tree_code2child_num(self, tree_code_set):
        sorted_codes = sorted(tree_code_set)

        tree_code2child_num = {tree_code: 0 for tree_code in tree_code_set}

        for i, tree_code in tqdm(enumerate(sorted_codes), total=len(sorted_codes), desc='Ontology Computing...'):
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
        return self.tree_code2child_num[c]

    @lru_cache(maxsize=None)
    def IC(self, c):
        return -math.log(self.freq(c) / self.freq(self.root))

    def lca(self, c1, c2):
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

