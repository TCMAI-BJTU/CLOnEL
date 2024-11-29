# -*- coding: utf-8 -*-
# @Time    : 2024/7/19 01:52
# @Author  : Rui Hua
# @Email   : 
# @File    : loader.py
# @Software: PyCharm
import glob
import os
from src.data.candidate_dataset import CandidateDataset
from torch.utils.data import DataLoader


def load_queries(data_dir, stage):
    '''
    :param data_dir:
    :param stage: train or test
    :return:
    '''
    datas = []
    exp_dirs = glob.glob(os.path.join(data_dir, "exp_*"))
    exp_dirs = sorted(exp_dirs)
    for exp_dir in exp_dirs:
        lines = open(os.path.join(exp_dir, f"{stage}.txt"), mode='r', encoding='utf-8').readlines()
        data = []
        for line in lines:
            line = line.strip()
            tui, name = line.split("\t")
            data.append((tui, name))
        datas.append(data)
    return datas


def load_dictionary(data_dir):
    datas = []
    exp_dirs = glob.glob(os.path.join(data_dir, "exp_*"))
    exp_dirs = sorted(exp_dirs)
    for exp_dir in exp_dirs:
        lines = open(os.path.join(exp_dir, "dictionary.txt"), mode='r', encoding='utf-8').readlines()
        data = []
        for line in lines:
            line = line.strip()
            tui, name = line.split("\t")
            data.append((tui, name))
        datas.append(data)
    return datas


def load_data(args, stage, encoder=None):
    queries = load_queries(args.dataset_name_or_path, stage=stage)
    dictionary = load_dictionary(args.dataset_name_or_path)

    candidate_dataset = CandidateDataset(args, queries, dictionary, encoder=encoder)

    dataloader = DataLoader(candidate_dataset, batch_size=args.batch_size, shuffle=True)

    return candidate_dataset, dataloader
