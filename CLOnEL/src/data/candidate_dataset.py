# -*- coding: utf-8 -*-
# @Time    : 2024/10/9 11:26
# @Author  : Rui Hua
# @Email   : 
# @File    : candidate_dataset.py
# @Software: PyCharm
import faiss
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, default_data_collator

from src.data.tree_sim_tool import TreeSimTool


class NamesDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


class CandidateDataset(Dataset):
    def __init__(self, args, queries, dicts, encoder=None):
        self.max_length = args.max_length
        self.model_name_or_path = args.model_name_or_path
        self.use_cuda = args.use_cuda
        self.topk = args.topk
        self.last_layer = args.last_layer
        self.dataset_name_or_path = args.dataset_name_or_path
        self.retrieve_similarity_func = args.retrieve_similarity_func
        self.use_tree_similarity = args.use_tree_similarity
        self.tree_ratio = args.tree_ratio

        self.queries = queries
        self.dicts = dicts

        self.candidate_idxs = None

        self.init_encoder_and_tokenizer(encoder)

        self.update_task(task_label=0)

    def update_task(self, task_label):
        self.task_label = task_label

        self.tree_sim_tool = TreeSimTool(self.dataset_name_or_path, task_label=self.task_label,
                                         tree_ratio=self.tree_ratio)

        queries = self.queries[self.task_label]
        dicts = self.dicts[self.task_label]

        self.query_ids, self.query_names = [row[0] for row in queries], [row[1] for row in queries]
        self.dict_ids, self.dict_names = [row[0] for row in dicts], [row[1] for row in dicts]

    def __getitem__(self, query_idx):
        assert (self.candidate_idxs is not None)

        query_name = self.query_names[query_idx]

        query_token = self.tokenizer(query_name, max_length=self.max_length, padding='max_length', truncation=True,
                                     return_tensors='pt')

        topk_candidate_idxs = self.candidate_idxs[query_idx]

        assert len(topk_candidate_idxs) == self.topk

        candidate_names = [self.dict_names[i] for i in topk_candidate_idxs]

        candidate_tokens = self.tokenizer(candidate_names, max_length=self.max_length, padding='max_length',
                                          truncation=True, return_tensors='pt')
        labels = self.get_labels(query_idx, topk_candidate_idxs)

        return (query_token, candidate_tokens), labels

    def __len__(self):
        return len(self.query_ids)

    def check_label(self, query_id, candidate_id):
        if query_id == candidate_id:
            return 1

        if self.use_tree_similarity:
            label = self.tree_sim_tool.compute_similarity(query_id, candidate_id)
        else:
            label = 0
        return label

    def get_labels(self, query_idx, candidate_idxs):
        labels = np.array([])
        query_id = self.query_ids[query_idx]
        candidate_ids = np.array(self.dict_ids)[candidate_idxs]
        for candidate_id in candidate_ids:
            label = self.check_label(query_id, candidate_id)
            labels = np.append(labels, label)
        return labels

    def set_candidate_idxs(self):
        with torch.no_grad():
            self.candidate_idxs, self.candidate_scores = self.retrieve_candidate(topk=self.topk)

    def retrieve_candidate(self, topk):

        queries = self.embed_names(self.query_names)
        candidates = self.embed_names(self.dict_names)

        queries = np.array(queries)
        candidates = np.array(candidates)
        if self.retrieve_similarity_func == "cosine":
            faiss.normalize_L2(queries)
            faiss.normalize_L2(candidates)

        gpu_resources = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = 0

        index = faiss.GpuIndexFlatIP(gpu_resources, 768, flat_config)

        index.add(candidates)

        topk_sim, topk_idx = index.search(queries, topk)

        return topk_idx, topk_sim

    def embed_names(self, names, show_progress=False):
        max_length = self.max_length
        self.encoder.eval()
        batch_size = 1024

        name_encodings = self.tokenizer(names, padding="max_length", max_length=max_length, truncation=True,
                                        return_tensors="pt")

        if self.use_cuda:
            name_encodings.to('cuda')

        name_dataset = NamesDataset(name_encodings)
        name_dataloader = DataLoader(name_dataset, shuffle=False, collate_fn=default_data_collator,
                                     batch_size=batch_size)

        embeddings = []
        with torch.no_grad():
            for batch in tqdm(name_dataloader, disable=not show_progress, desc='embedding'):
                outputs = self.encoder(**batch)
                if self.last_layer == 'cls':
                    batch_dense_embeds = outputs[0][:, 0].cpu().detach().numpy()
                elif self.last_layer == 'mean':
                    batch_dense_embeds = outputs[0].mean(dim=1).cpu().detach().numpy()
                embeddings.append(batch_dense_embeds)
        embeddings = np.concatenate(embeddings, axis=0)
        embeddings = torch.tensor(embeddings)
        return embeddings

    def init_encoder_and_tokenizer(self, encoder):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        if encoder is None:
            self.encoder = AutoModel.from_pretrained(self.model_name_or_path)
        else:
            self.encoder = encoder

        if self.use_cuda:
            self.encoder.to('cuda')
