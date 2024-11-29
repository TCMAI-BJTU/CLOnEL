# -*- coding: utf-8 -*-
# @Time    : 2024/7/19 04:29
# @Author  : Rui Hua
# @Email   : 
# @File    : main.py
# @Software: PyCharm
from src.logger import logger
from src.data import load_data
from src.evaluator import Evaluator
from src.hparams import get_args
from src.model import CLOnEL
from src.trainer import Trainer


def main():
    args = get_args()

    train_dataset, train_loader = load_data(args, stage="train")

    model = CLOnEL(args, encoder=train_dataset.encoder)

    fix_evaluator = Evaluator(args, train_dataset.encoder, stage="dev")
    dynamic_evaluator = Evaluator(args, train_dataset.encoder, stage="test")

    trainer = Trainer(args, fix_evaluator, dynamic_evaluator)

    trainer.train(
        model=model,
        train_dataset=train_dataset,
        train_loader=train_loader
    )

    logger.info(f"fix_best_result: {trainer.fix_evaluator.best_result}")
    logger.info(f"dyn_best_result: {trainer.dynamic_evaluator.best_result}")


if __name__ == '__main__':
    main()
