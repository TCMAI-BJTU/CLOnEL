# -*- coding: utf-8 -*-
# @Time    : 2024/6/17
# @Author  : Rui Hua
# @Email   : 
# @File    : parser.py
# @Software: PyCharm
import argparse
import os
from transformers import set_seed
from src.logger import logger


def check_args(args):
    exp_dirs = os.listdir(args.dataset_name_or_path)
    exp_dirs = [exp_dir for exp_dir in exp_dirs if exp_dir.startswith("exp")]
    assert len(exp_dirs) == args.n_experiences, \
        f"Number of experiences should be {len(exp_dirs)}, but got {args.n_experiences}."


def get_args():
    parser = argparse.ArgumentParser("Hyperparameter Setting")

    parser.add_argument('--seed', type=int, default=2222, help='seed')
    parser.add_argument('--max_length', type=int, default=20, help='max_length')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--topk', type=int, default=20, help='topk')
    parser.add_argument('--num_epochs', type=int, default=20, help='num epochs')
    parser.add_argument('--dataset_name_or_path', type=str, default="./data/sympel-cl")
    parser.add_argument('--model_name_or_path', type=str,
                        default="/home/huarui/pycharmProject/symptom_entity_link/症状实体链接/pretrain_model/SapBERT-from-PubMedBERT-fulltext")
    parser.add_argument('--cache_dir', type=str, default="./cache")

    parser.add_argument('--use_cuda', type=bool, default=True)

    parser.add_argument('--n_experiences', type=int, default=10)
    parser.add_argument('--last_layer', type=str, default="mean", choices=["cls", "mean"])
    parser.add_argument('--cl_strategy', type=str, default="naive", choices=["naive", "ewc_separate", "ewc_online"])
    parser.add_argument('--ewc_lambda', type=float, default=100)
    parser.add_argument('--decay_factor', type=float, default=0.95, help='仅在ewc_online中使用')

    parser.add_argument('--retrieve_similarity_func', default="dot", type=str, choices=['dot', 'cosine'])
    parser.add_argument('--train_similarity_func', default="cosine", type=str, choices=['dot', 'cosine'])
    parser.add_argument('--use_tree_similarity', action='store_true')
    parser.add_argument('--tree_ratio', default=0.5, type=float)

    args = parser.parse_args()

    set_seed(args.seed)

    check_args(args)
    logger.info(f"args: {vars(args)}")

    return args
