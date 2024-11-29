# -*- coding: utf-8 -*-
# @Time    : 2024/8/4 16:50
# @Author  : Rui Hua
# @Email   : 
# @File    : main.py
# @Software: PyCharm

import os
import numpy as np

from src.evaluator import Evaluator
from src.hparams import parse_args
from src.logger import setup_logger


def main():
    args = parse_args()

    logger = setup_logger(args.log_file)

    original_model_path = args.model_name_or_path
    results = []
    for i in range(10):
        args.eval_dir = f"./data/aap/fold{i}/test.txt"
        args.model_name_or_path = os.path.join(original_model_path, f"fold{i}")
        evaluator = Evaluator(args, encoder=None, is_save_checkpoint=False)
        result = evaluator.evaluate(epoch=999, step=999)
        results.append(dict(result))

    average_acc1 = np.mean([result["acc1"] for result in results])
    average_acc5 = np.mean([result["acc5"] for result in results])
    logger.info(f"Average acc1: {average_acc1}")
    logger.info(f"Average acc5: {average_acc5}")



if __name__ == '__main__':
    main()
