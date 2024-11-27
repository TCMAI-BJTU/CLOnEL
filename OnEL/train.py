# -*- coding: utf-8 -*-
# @Time    : 2024/8/4 16:50
# @Author  : Rui Hua
# @Email   : 
# @File    : main.py
# @Software: PyCharm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from src.evaluator.evaluator import Evaluator
from src.data.loader import load_data
from src.hparams.parser import parse_args
from src.model.onel import OnEL
from src.trainer.trainer import Trainer


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
