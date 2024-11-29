# -*- coding: utf-8 -*-
# @Time    : 2024/8/4 16:50
# @Author  : Rui Hua
# @Email   :
# @File    : train_app.py
# @Software: PyCharm
from src.logger import setup_logger

import numpy as np

from src.data import load_data
from src.model import OnEL
from src.trainer import Trainer
from src.hparams import parse_args
from src.evaluator import Evaluator


def main():
    args = parse_args()


    best_results = []
    train_dir = args.train_dir
    eval_dir = args.eval_dir
    for fold_num in range(3):
        args.train_dir = f"./data/aap/fold{fold_num}/{train_dir}"
        args.eval_dir = f"./data/aap/fold{fold_num}/{eval_dir}"

        train_dataset, train_loader = load_data(args)

        model = OnEL(
            encoder=train_dataset.encoder,
            args=args
        )

        evaluator = Evaluator(args, train_dataset.encoder)

        trainer = Trainer(args, evaluator)

        trainer.train(
            model=model,
            train_dataset=train_dataset,
            train_loader=train_loader
        )

        best_result = trainer.evaluator.best_result
        best_results.append(dict(best_result))

    logger = setup_logger(args.log_file)

    for i, result in enumerate(best_results):
        logger.info(f"Fold {i}: {result}")

    average_acc1 = np.mean([result["acc1"] for result in best_results])
    average_acc5 = np.mean([result["acc5"] for result in best_results])
    logger.info(f"Average acc1: {average_acc1}")
    logger.info(f"Average acc5: {average_acc5}")


if __name__ == '__main__':
    main()
