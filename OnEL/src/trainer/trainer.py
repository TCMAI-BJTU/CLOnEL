# -*- coding: utf-8 -*-
# @Time    : 2024/8/4 18:16
# @Author  : Rui Hua
# @Email   : 
# @File    : trainer.py
# @Software: PyCharm
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data import CandidateDataset
from src.evaluator import Evaluator
from src.logger import setup_logger
from src.model import OnEL


class Trainer:
    def __init__(self, args, evaluator: Evaluator):
        self.epochs = args.epochs
        self.use_tree_similarity = args.use_tree_similarity
        self.evaluator = evaluator
        self.early_stopping = False

        self.logger = setup_logger(args.log_file)

    def train(
            self,
            model: OnEL,
            train_dataset: CandidateDataset,
            train_loader: DataLoader,
    ):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(model.optimizer, self.epochs * len(train_loader))
        model.eval()
        self.evaluator.evaluate(epoch=0, step=0)
        model.train()

        if train_dataset.dict_names == self.evaluator.eval_dataset.dict_names:
            train_dataset.set_candidate_idxs(
                dict_embeds=self.evaluator.eval_dataset.dict_embeds,
            )
        else:
            train_dataset.set_candidate_idxs()

        for epoch in range(self.epochs):

            total_loss = 0
            model.train()
            progress_bar = tqdm(total=len(train_loader), desc='Training epoch {}'.format(epoch + 1), ncols=100)

            for step, data in enumerate(train_loader):
                model.optimizer.zero_grad()

                batch_x, batch_y = data

                batch_pred = model(batch_x)
                loss = model.compute_loss(batch_pred, batch_y)

                loss.backward()
                model.optimizer.step()
                scheduler.step()

                progress_bar.update(1)
                progress_bar.set_postfix({'loss': loss.item()})
                total_loss += loss.item()

            progress_bar.close()

            model.eval()
            self.evaluator.evaluate(epoch, step)

            if train_dataset.dict_names == self.evaluator.eval_dataset.dict_names:
                train_dataset.set_candidate_idxs(
                    dict_embeds=self.evaluator.eval_dataset.dict_embeds,
                )
            else:
                train_dataset.set_candidate_idxs()

            model.train()

            self.logger.info('Epoch {} average loss: {}'.format(epoch + 1, total_loss / len(train_loader)))

        return model
