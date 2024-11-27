# -*- coding: utf-8 -*-
# @Time    : 2024/8/4 16:50
# @Author  : Rui Hua
# @Email   : 
# @File    : main.py
# @Software: PyCharm
from src.evaluator.evaluator import Evaluator
from src.hparams.parser import parse_args



def main():
    args = parse_args()

    evaluator = Evaluator(args, encoder=None, is_save_checkpoint=False)
    evaluator.evaluate(epoch=999, step=999)


if __name__ == '__main__':
    main()
