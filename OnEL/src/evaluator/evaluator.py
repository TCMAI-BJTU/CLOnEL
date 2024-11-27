# -*- coding: utf-8 -*-
# @Time    : 2024/8/6 18:00
# @Author  : Rui Hua
# @Email   : 
# @File    : evaluator.py
# @Software: PyCharm
import json
import os
from collections import defaultdict
import numpy as np
from src.data.candidate_dataset import CandidateDataset
from src.data.loader import load_queries, load_dictionary
from src.logger.logger import setup_logger


class Evaluator():
    def __init__(
            self,
            args,
            encoder,
            is_save_checkpoint=True
    ):

        self.args = args

        self.is_save_checkpoint = is_save_checkpoint

        self.eval_dictionary = load_dictionary(args.eval_dictionary_path, args.dataset_name_or_path)
        self.eval_queries = load_queries(args.eval_dir, args.dataset_name_or_path, stage="eval")

        if args.debug:
            self.eval_queries = self.eval_queries[:120]
            self.eval_dictionary = self.eval_dictionary[:12000]
        # 传入candidateDataset的query和评估循环遍历时候的query不一样，传入candidateDataset的需要拆分复合术语
        eval_queries = []
        for query in self.eval_queries:
            mentions, cuis, query_type = query
            mentions = mentions.split("|")
            for mention in mentions:
                eval_queries.append((mention, cuis))

        self.eval_dataset = CandidateDataset(
            args=args,
            queries=eval_queries,
            dicts=self.eval_dictionary,
            encoder=encoder,
        )

        self.best_result = defaultdict(int)

        self.logger = setup_logger(self.args.log_file)

    def check_label(self, predicted_cui, golden_cui):
        """
        Some composite annotation didn't consider orders
        So, set label '1' if any cui is matched within composite cui (or single cui)
        Otherwise, set label '0'
        """
        return int(len(set(predicted_cui.split("|")).intersection(set(golden_cui.split("|")))) > 0)

    def evaluate(self, epoch, step):
        self.eval_dataset.set_candidate_idxs()

        queries = []

        dict_names = np.array(self.eval_dataset.dict_names)
        dict_ids = np.array(self.eval_dataset.dict_ids)

        for eval_query in self.eval_queries:

            mentions = eval_query[0].replace("+", "|").split("|")
            golden_cui = eval_query[1].replace("+", "|")

            dict_mentions = []
            for mention in mentions:
                query_idx = self.eval_dataset.query_names.index(mention)

                pred_candidate_idxs = self.eval_dataset.candidate_idxs[query_idx].reshape(-1)
                pred_candidate_scores = self.eval_dataset.candidate_scores[query_idx].reshape(-1)

                pred_candidate_names = dict_names[pred_candidate_idxs]
                pred_candidate_ids = dict_ids[pred_candidate_idxs]

                dict_candidates = []
                for pred_candidate in zip(pred_candidate_names, pred_candidate_ids, pred_candidate_scores):
                    dict_candidates.append({
                        'name': pred_candidate[0],
                        'cui': pred_candidate[1],
                        'label': self.check_label(pred_candidate[1], golden_cui),
                        'score': f'{pred_candidate[2]:.4f}'
                    })
                dict_mentions.append({
                    'mention': mention,
                    'golden_cui': golden_cui,
                    'candidates': dict_candidates
                })
            queries.append({
                'mentions': dict_mentions
            })

        result = self.evaluate_topk_acc({'queries': queries}, epoch, step)

        if result['acc1'] >= self.best_result["acc1"]:
            for i in range(len(queries[0]['mentions'][0]['candidates'])):
                self.best_result[f"acc{i + 1}"] = result[f"acc{i + 1}"]
            self.best_result["epoch"] = epoch
            self.save_checkpoint(epoch, step)

        self.logger.info(dict(self.best_result))

    def evaluate_topk_acc(self, data, epoch, step):
        """
        evaluate acc@1~acc@k
        """
        queries = data['queries']

        total = len(queries[0]['mentions'][0]['candidates'])

        for i in range(0, total):
            hit = 0
            for query in queries:
                mentions = query['mentions']
                mention_hit = 0
                for mention in mentions:
                    candidates = mention['candidates'][:i + 1]  # to get acc@(i+1)
                    mention_hit += np.any([candidate['label'] for candidate in candidates])

                # When all mentions in a query are predicted correctly,
                # we consider it as a hit
                if mention_hit == len(mentions):
                    hit += 1

            data['acc{}'.format(i + 1)] = hit / len(queries)

        output_str = ""
        for k, v in data.items():
            if "acc" in k:
                output_str += f"{k}: {v:.4f}, "

        self.logger.info(output_str)

        self.save_records(data, epoch, step)

        return data

    def save_checkpoint(self, epoch, step):
        if self.is_save_checkpoint:
            return
        checkpoint_path = f"./checkpoints/{self.args.dataset_name_or_path}"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint = os.path.join(checkpoint_path, f"model_{str(self.args.learning_rate)}")
        self.eval_dataset.encoder.save_pretrained(checkpoint)
        self.eval_dataset.tokenizer.save_pretrained(checkpoint)
        self.logger.info(f"Model saved at epoch {epoch} step {step}, Best Acc1: {self.best_result['acc1']:.4f}")

    def save_records(self, data, epoch, step):
        result_file = f"./records/result_{epoch}_{step}.json"
        json.dump(data, open(result_file, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
        self.logger.info(f"Result saved to {result_file}")
