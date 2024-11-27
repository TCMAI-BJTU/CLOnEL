# -*- coding: utf-8 -*-
# @Time    : 2024/10/19 18:12
# @Author  : Rui Hua
# @Email   : 
# @File    : main_aap.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# @Time    : 2024/8/4 16:50
# @Author  : Rui Hua
# @Email   :
# @File    : main.py
# @Software: PyCharm
import os
import sys

sys.path.append("/home/huarui/pycharmProject/symptom_entity_link/症状实体链接")
os.environ['TOKENIZERS_PARALLELISM'] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from src.evaluator.evaluator import Evaluator
from src.data.loader import load_data
from src.hparams.parser import parse_args
from src.model.concpet_model import ConceptModel
from src.trainer.trainer import Trainer


def main():
    args = parse_args()

    args.train_dictionary_path = f"{args.root_path}/data/AAP/test_dictionary.txt"
    args.test_dictionary_path = f"{args.root_path}/data/AAP/test_dictionary.txt"

    best_results = []
    for fold_num in range(10):
        print(f"Fold {fold_num}")
        args.train_dir = f"{args.root_path}/data/AAP/fold{fold_num}/train.txt"
        args.test_dir = f"{args.root_path}/data/AAP/fold{fold_num}/test.txt"

        train_dataset, train_loader = load_data(args)

        model = ConceptModel(
            encoder=train_dataset.encoder,
            args=args
        )

        evaluator = Evaluator(args, train_dataset.encoder, dev_or_test="dev")

        trainer = Trainer(args, evaluator)

        trainer.train(
            model=model,
            train_dataset=train_dataset,
            train_loader=train_loader
        )

        best_result = trainer.evaluator.best_result
        best_results.append(dict(best_result))

    for i, result in enumerate(best_results):
        print(f"Fold {i}: {result}")

    res = []
    for i, result in enumerate(best_results):
        res.append(result["acc1"])
    print(f"Average: {sum(res) / len(res)}")


if __name__ == '__main__':
    main()
