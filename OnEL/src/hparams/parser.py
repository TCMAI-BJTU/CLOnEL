# -*- coding: utf-8 -*-
# @Time    : 2024/6/17
# @Author  : Rui Hua
# @Email   : 
# @File    : parser.py
# @Software: PyCharm
import argparse
import time
from transformers import set_seed
from src.logger import setup_logger


def update_dictionary_paths(args):
    args.train_dictionary_path = f"./data/{args.dataset_name_or_path}/{args.train_dictionary_path}"
    args.eval_dictionary_path = f"./data/{args.dataset_name_or_path}/{args.eval_dictionary_path}"

    if args.dataset_name_or_path != "aap":
        args.train_dir = f"./data/{args.dataset_name_or_path}/{args.train_dir}"
        args.eval_dir = f"./data/{args.dataset_name_or_path}/{args.eval_dir}"

    return args


def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        '--model_name_or_path',
        default="/home/huarui/pycharmProject/symptom_entity_link/症状实体链接/pretrain_model/SapBERT-from-PubMedBERT-fulltext",
        help='Directory for pretrained model'
    )

    parser.add_argument(
        '--dataset_name_or_path',
        default="ncbi-disease",
        choices=['ncbi-disease', 'bc5cdr-chemical', 'bc5cdr-disease', 'aap','cometa-cf', 'sympel']
    )
    parser.add_argument('--seed', type=int, default=2222)

    parser.add_argument('--train_dir', type=str, default="processed_traindev")
    parser.add_argument('--train_dictionary_path', type=str, default="train_dictionary.txt")

    parser.add_argument('--eval_dir', type=str, default="processed_test")
    parser.add_argument('--eval_dictionary_path', type=str, default="test_dictionary.txt")

    parser.add_argument('--max_length', default=45, type=int)
    parser.add_argument('--topk', default=20, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', help='epoch to train', default=10, type=int)
    parser.add_argument('--learning_rate', help='learning rate', default=1e-5, type=float)
    parser.add_argument('--weight_decay', help='weight decay', default=2e-4, type=float)
    parser.add_argument('--embed_dim', default=768, type=int)
    parser.add_argument('--use_cuda', default=True)

    # Our experiment is to perform iterative training based on epoch. If you want to get better results, you can adjust and reduce the step_ratio. It can be easily achieved by referring to the cross encoder.
    parser.add_argument('--retrieve_step_ratio', default=1.0, type=float)
    parser.add_argument('--eval_step_ratio', default=1.0, type=float)

    parser.add_argument('--debug', default=False, type=bool)

    parser.add_argument('--last_layer', default='mean', type=str, choices=['mean', 'cls'])
    parser.add_argument('--retrieve_similarity_func', default="dot", type=str, choices=['dot', 'cosine'])
    parser.add_argument('--train_similarity_func', default="cosine", type=str, choices=['dot', 'cosine'])

    parser.add_argument('--use_tree_similarity', action='store_true')
    parser.add_argument('--tree_ratio', default=0.5, type=float)

    parser.add_argument('--use_embed_parallel', action='store_true')

    args = parser.parse_args()

    args = update_dictionary_paths(args)

    set_seed(args.seed)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    args.log_file = f"logs/{current_time}.log"

    logger = setup_logger(log_file=args.log_file)
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")

    return args
