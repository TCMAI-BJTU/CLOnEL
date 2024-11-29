# -*- coding: utf-8 -*-
# @Time    : 2024/7/20 17:02
# @Author  : Rui Hua
# @Email   : 
# @File    : trainer.py
# @Software: PyCharm
import json
import torch
from torch import nn, optim
from tqdm import tqdm
from src.evaluator.evaluator import Evaluator
from src.logger.logger import logger
from src.trainer.strategy import Naive, EWC


class Trainer():
    def __init__(self, args, fix_evaluator: Evaluator, dynamic_evaluator: Evaluator):
        self.n_experiences = args.n_experiences
        self.num_epochs = args.num_epochs
        self.use_cuda = args.use_cuda
        self.learning_rate = args.learning_rate
        self.cl_strategy = args.cl_strategy
        self.weight_decay = args.weight_decay

        self.ewc_lambda = args.ewc_lambda
        self.decay_factor = args.decay_factor

        self.fix_evaluator = fix_evaluator
        self.dynamic_evaluator = dynamic_evaluator

    def _train_batch(self, batch, model):
        self.optimizer.zero_grad()

        batch_x, batch_y = batch

        outputs = model(batch_x)

        loss = self.criterion(outputs.float(), batch_y.to(outputs.device).float())
        loss = self.strategy.before_backward(self.trainer_state['task_label'], loss)
        loss.backward()

        self.optimizer.step()

        self.progress_bar.set_postfix({"loss": loss.item()})

    def _train_epoch(self, train_loader, model):
        self.progress_bar = tqdm(
            total=len(train_loader),
            desc=f"Task {self.trainer_state['task_label']}/{self.n_experiences}",
            ncols=100
        )
        for step, batch in enumerate(train_loader):
            self.progress_bar.update(1)

            self.trainer_state["step"] = step

            self._train_batch(batch, model)

            self.progress_bar.set_description(
                f"Task {self.trainer_state['task_label']}/{self.n_experiences} "
                f"Epoch {self.trainer_state['epoch']}/{self.num_epochs}")
        self.progress_bar.close()

    def _train_experience(self, train_loader, model):
        init_result = self.dynamic_evaluator.evaluate(model, self.trainer_state)
        self.trainer_state["dynamic_init"].append({"acc1": init_result["acc1"], "acc5": init_result["acc5"]})

        for epoch in range(self.num_epochs):
            self.trainer_state["epoch"] = epoch

            # 每个任务下的每个epoch需要重新设置候选实体
            train_loader.dataset.set_candidate_idxs()

            self._train_epoch(train_loader, model)

            with torch.no_grad():
                result = self.dynamic_evaluator.evaluate(model, self.trainer_state)
                self.trainer_state["dynamic_state"].append({"acc1": result["acc1"], "acc5": result["acc5"]})


    def train(self, model, train_dataset, train_loader):

        self.criterion = nn.MSELoss()

        if self.cl_strategy == 'ewc_separate':
            self.strategy = EWC(model, mode='separate', ewc_lambda=self.ewc_lambda)
        elif self.cl_strategy == 'ewc_online':
            self.strategy = EWC(model, mode='online', ewc_lambda=self.ewc_lambda, decay_factor=self.decay_factor)
        elif self.cl_strategy == 'naive':
            self.strategy = Naive(model)
        else:
            raise ValueError(f"CL strategy {self.cl_strategy} not recognized")


        logger.info(f"{self.cl_strategy} is used")

        self.trainer_state = {"task_label": 0, "epoch": 0, "step": 0, "fix_state": [], "dynamic_state": [],
                              "dynamic_init": []}

        for task_label in range(self.n_experiences):
            self.trainer_state["task_label"] = task_label
            self.trainer_state["epoch"] = 0
            self.trainer_state["step"] = 0

            # 每个新的任务开始前，需要更新到当前任务的query和dict
            train_dataset.update_task(task_label=task_label)
            self.optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

            self._train_experience(train_loader, model)

            self.strategy.after_training_exp(task_label, train_loader, self.criterion, self.optimizer)

            # logger.info("Update the encoder to the best encoder of the previous task...")
            # train_dataset.encoder = deepcopy(self.dynamic_evaluator.best_encoder)
            logger.info("Update encoder to the last encoder of the current task...")

            logger.info(f"Finished training for task {task_label}/{self.n_experiences}")
            logger.info("Starting evaluation on fixed dataset...")
            fix_result = self.fix_evaluator.evaluate(model, self.trainer_state)
            self.trainer_state["fix_state"].append({"acc1": fix_result["acc1"], "acc5": fix_result["acc5"]})
            logger.info("Evaluation ended...")

        json.dump(self.trainer_state, open(f"./trainer_state_{self.cl_strategy}.json", "w"), indent=4)
