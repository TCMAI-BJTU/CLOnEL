# -*- coding: utf-8 -*-
# @Time    : 2024/8/4 16:50
# @Author  : Rui Hua
# @Email   : 
# @File    : main.py
# @Software: PyCharm
from src.data import load_data
from src.model import OnEL
from src.trainer import Trainer
from src.hparams import parse_args
from src.evaluator import Evaluator

def main():

    args = parse_args()

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


if __name__ == '__main__':
    main()
