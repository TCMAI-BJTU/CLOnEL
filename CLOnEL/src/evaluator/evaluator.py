# -*- coding: utf-8 -*-
# @Time    : 2024/6/28 15:14
# @Author  : Rui Hua
# @Email   : 
# @File    : evalutor.py
# @Software: PyCharm
from collections import defaultdict
import numpy as np
from src.data.loader import load_data
from src.logger.logger import logger
from src.model.clonel import CLOnEL


class Evaluator():
    def __init__(self, args, encoder, stage):
        self.args = args
        self.stage = stage

        self.test_dataset, _ = load_data(self.args, stage=stage, encoder=encoder)

        self.best_result = defaultdict(dict)
        # self.best_encoder = None

    def trainer_state_format(self, trainer_state):
        return f"task_{trainer_state['task_label']}_epoch_{trainer_state['epoch']}_step_{trainer_state['step']}"

    def check_label(self, predicted_cui, golden_cui):
        return int(predicted_cui == golden_cui)

    def evaluate(self, model: CLOnEL, trainer_state: dict):
        model.eval()

        self.test_dataset.update_task(trainer_state['task_label'])
        self.test_dataset.set_candidate_idxs()

        dict_names = np.array(self.test_dataset.dict_names)
        dict_ids = np.array(self.test_dataset.dict_ids)

        queries = []

        for query_idx in range(len(self.test_dataset)):

            mention = self.test_dataset.query_names[query_idx]
            golden_cui = self.test_dataset.query_ids[query_idx]

            pred_candidate_idxs = self.test_dataset.candidate_idxs[query_idx].reshape(-1)
            pred_candidate_scores = self.test_dataset.candidate_scores[query_idx].reshape(-1)

            pred_candidate_names = dict_names[pred_candidate_idxs]
            pred_candidate_ids = dict_ids[pred_candidate_idxs]

            dict_candidates = []
            for pred_candidate in zip(pred_candidate_names, pred_candidate_ids, pred_candidate_scores):
                label = self.check_label(pred_candidate[1], golden_cui)
                dict_candidates.append({
                    'name': pred_candidate[0],
                    'cui': pred_candidate[1],
                    'label': label,
                    'score': f'{pred_candidate[2]:.4f}'
                })
            queries.append({
                'mention': mention,
                'golden_cui': golden_cui,
                'candidates': dict_candidates
            })

        result = self.evaluate_topk_acc({'queries': queries}, trainer_state)
        # 检查异常
        if self.best_result[f"exp_{trainer_state['task_label']}"].get("epoch", 0) > trainer_state['epoch']:
            raise ValueError("epoch is not increasing, please check the model")

        if result['acc1'] >= self.best_result[f"exp_{trainer_state['task_label']}"].get("acc1", 0):
            self.best_result[f"exp_{trainer_state['task_label']}"]["acc1"] = result['acc1']
            self.best_result[f"exp_{trainer_state['task_label']}"]["acc3"] = result['acc3']
            self.best_result[f"exp_{trainer_state['task_label']}"]["acc5"] = result['acc5']
            self.best_result[f"exp_{trainer_state['task_label']}"]["epoch"] = trainer_state['epoch']
            # self.best_encoder = deepcopy(self.test_dataset.encoder)

        # self.test_dataset.encoder.save_pretrained(f"./checkpoints/model_{self.trainer_state_format(trainer_state)}")
        # self.test_dataset.tokenizer.save_pretrained(f"./checkpoints/model_{self.trainer_state_format(trainer_state)}")
        # print(f"Model saved at {self.trainer_state_format(trainer_state)}")

        # pprint(dict(self.best_result))
        for k, v in self.best_result.items():
            logger.info(
                f"{self.stage} Best result in {k}: acc1: {v['acc1']:.4f}, acc3: {v['acc3']:.4f}, acc5: {v['acc5']:.4f}, epoch: {v['epoch']}")
        model.train()
        return result

    def evaluate_topk_acc(self, data, trainer_state):
        """
        evaluate acc@1~acc@k
        """
        queries = data['queries']

        topk = len(queries[0]['candidates'])

        for i in range(0, topk):
            hit = 0
            for mentions in queries:
                candidates = mentions['candidates'][:i + 1]  # to get acc@(i+1)
                hit += int(np.any([candidate['label'] for candidate in candidates]))

            data['acc{}'.format(i + 1)] = round(hit / len(queries), 4) * 100

        output_str = ""
        for k, v in data.items():
            if "acc" in k:
                output_str += f"{k}: {v:.4f}, "

        # print(output_str)
        logger.info(f"{self.stage}_{output_str}")

        # result_file = f"./records/result_{self.trainer_state_format(trainer_state)}.json"
        # json.dump(data, open(result_file, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
        # logger.info(f"Result saved to {result_file}")

        return data
