# -*- coding: utf-8 -*-
# @Time    : 2024/10/9 13:11
# @Author  : Rui Hua
# @Email   : 
# @File    : SympModel.py
# @Software: PyCharm
import torch

from torch import nn
from torch.nn import functional as F


class CLOnEL(nn.Module):
    def __init__(self, args, encoder):
        super(CLOnEL, self).__init__()
        self.max_length = args.max_length
        self.use_cuda = args.use_cuda
        self.topk = args.topk
        self.encoder = encoder
        self.train_similarity_func = args.train_similarity_func

        self.embed_dim = 768

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
        query_embeddings = self.encoder(**query_token)[0][:, 0]  # 16, 768
        return query_embeddings

    def get_candidate_embedding(self, candidate_tokens):
        input_ids = candidate_tokens["input_ids"].reshape(-1, self.max_length)
        attention_mask = candidate_tokens["attention_mask"].reshape(-1, self.max_length)
        candidate_embeddings = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        candidate_embeddings = candidate_embeddings[0][:, 0].view(-1, self.topk, self.embed_dim)
        return candidate_embeddings

    def forward(self, x):
        query_token, candidate_tokens = x

        query_token, candidate_tokens = self.tensor2cuda(query_token, candidate_tokens)

        query_embeddings = self.get_query_embedding(query_token)

        candidate_embeddings = self.get_candidate_embedding(candidate_tokens)
        if self.train_similarity_func == "cosine":
            score = F.cosine_similarity(query_embeddings.unsqueeze(1), candidate_embeddings, dim=-1)
        elif self.train_similarity_func == "dot":
            score = torch.bmm(query_embeddings.unsqueeze(1), candidate_embeddings.transpose(1, 2)).squeeze(1)
        else:
            raise ValueError("train_similarity_func must be cosine or dot")
        return score
