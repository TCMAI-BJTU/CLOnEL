# -*- coding: utf-8 -*-
# @Time    : 2024/8/4 17:56
# @Author  : Rui Hua
# @Email   : 
# @File    : concpet_model.py
# @Software: PyCharm
import torch
from torch import nn, optim
from torch.nn import functional as F


class OnEL(nn.Module):
    def __init__(self, encoder, args):
        super(OnEL, self).__init__()
        self.max_length = args.max_length
        self.use_cuda = args.use_cuda
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.topk = args.topk
        self.encoder = encoder
        self.train_similarity_func = args.train_similarity_func

        self.embed_dim = args.embed_dim

        self.optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.use_cuda:
            self.to("cuda")

    def tensor2cuda(self, query_token, candidate_tokens):
        if self.use_cuda:
            query_token = query_token.to("cuda")
            candidate_tokens = candidate_tokens.to("cuda")
            self.encoder.to("cuda")
        return query_token, candidate_tokens,

    def get_query_embedding(self, query_token):
        query_token = {k: v.squeeze(1) for k, v in query_token.items()}
        query_embeddings = self.encoder.forward(query_token)['sentence_embedding']
        return query_embeddings

    def get_candidate_embedding(self, candidate_tokens):
        candidate_tokens = {k: v.reshape(-1, self.max_length) for k, v in candidate_tokens.items()}
        candidate_embeddings = self.encoder.forward(candidate_tokens)
        candidate_embeddings = candidate_embeddings['sentence_embedding'].view(-1, self.topk, self.embed_dim)
        return candidate_embeddings

    def forward(self, x):
        query_token, candidate_tokens = x

        query_token, candidate_tokens = self.tensor2cuda(query_token, candidate_tokens)

        batch_size = query_token["input_ids"].size(0)

        query_embeddings = self.get_query_embedding(query_token)  # 16, 768

        candidate_embeddings = self.get_candidate_embedding(candidate_tokens)  # (16,7,3,768)

        # dot
        if self.train_similarity_func == "dot":
            score = torch.bmm(query_embeddings.unsqueeze(1), candidate_embeddings.transpose(1, 2)).squeeze(1)
        elif self.train_similarity_func == "cosine":
            score = F.cosine_similarity(query_embeddings.unsqueeze(1), candidate_embeddings, dim=-1)
        else:
            raise ValueError("Unknown similarity function")
        return score

    def compute_loss(self, score, target):
        if self.use_cuda:
            target = target.to("cuda")
        loss = F.mse_loss(score.float(), target.float())
        return loss
