# -*- coding: utf-8 -*-
# @Time    : 2024/7/31 22:20
# @Author  : Rui Hua
# @Email   : 
# @File    : candidate_dataset.py
# @Software: PyCharm
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.models import Pooling
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from .ontology_tree import OntologyTree
from src.logger import setup_logger

class CandidateDataset(Dataset):
    def __init__(self, args, queries, dicts, encoder=None):
        self.max_length = args.max_length
        self.model_name_or_path = args.model_name_or_path
        self.use_cuda = args.use_cuda
        self.topk = args.topk
        self.last_layer = args.last_layer
        self.use_tree_similarity = args.use_tree_similarity
        self.dataset_name_or_path = args.dataset_name_or_path
        self.retrieve_similarity_func = args.retrieve_similarity_func
        self.embed_dim = args.embed_dim
        self.tree_ratio = args.tree_ratio
        self.use_embed_parallel = args.use_embed_parallel

        self.query_names, self.query_ids = [row[0] for row in queries], [row[1] for row in queries]
        self.dict_names, self.dict_ids = [row[0] for row in dicts], [row[1] for row in dicts]

        self.dict_ids = np.array(self.dict_ids)

        self.candidate_idxs = None

        self.init_encoder(encoder)

        self.init_tree_similarity()

        self.logger = setup_logger(args.log_file)

        self.logger.info("CandidateDataset init finished")

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
        return len(self.query_names)

    def check_label(self, query_id, candidate_id_set):
        query_ids = query_id.split("|")
        label = 0
        for q_id in query_ids:
            if q_id in candidate_id_set:
                label = 1
                continue
            else:
                label = 0
                break
        if label == 0 and self.use_tree_similarity:
            candidate_ids = candidate_id_set.split("|")
            for q_id in query_ids:
                for candidate_id in candidate_ids:
                    label = max(self.tree_sim.compute_similarity(q_id, candidate_id), label)

        return label

    def get_labels(self, query_idx, candidate_idxs):
        labels = np.array([])
        query_id = self.query_ids[query_idx]
        candidate_ids = self.dict_ids[candidate_idxs]
        for candidate_id in candidate_ids:
            label = self.check_label(query_id, candidate_id)
            labels = np.append(labels, label)
        return labels

    def set_candidate_idxs(self, dict_embeds=None):
        self.candidate_idxs, self.candidate_scores = self.retrieve_candidate(dict_embeds, topk=self.topk)

    def retrieve_candidate(self, dict_embeds, topk):
        queries = self.embed_names(self.query_names)
        if dict_embeds is None:
            candidates = self.embed_names(self.dict_names, show_progress_bar=True)
        else:
            candidates = dict_embeds
            self.logger.info("Using pre-computed candidate embeddings")

        self.dict_embeds = candidates

        queries = np.array(queries)
        candidates = np.array(candidates)

        if self.retrieve_similarity_func == "cosine":
            faiss.normalize_L2(queries)
            faiss.normalize_L2(candidates)

        gpu_resources = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = 0

        index = faiss.GpuIndexFlatIP(gpu_resources, self.embed_dim, flat_config)

        index.add(candidates)

        topk_sim, topk_idx = index.search(queries, topk)
        return topk_idx, topk_sim

    def embed_names(self, names, show_progress_bar=False):
        self.encoder.eval()
        if not self.use_embed_parallel:
            embeddings = self.encoder.encode(names, show_progress_bar=show_progress_bar, batch_size=1024)
        else:
            pool = self.encoder.start_multi_process_pool()
            embeddings = self.encoder.encode_multi_process(names, pool, show_progress_bar=show_progress_bar, batch_size=1024)
            self.encoder.stop_multi_process_pool(pool)
        embeddings = torch.tensor(embeddings)
        return embeddings

    def init_encoder(self, encoder):

        if encoder is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            if self.last_layer in ['cls', 'mean']:
                word_embedding_model = models.Transformer(self.model_name_or_path, max_seq_length=self.max_length)
                pooling_model = Pooling(
                    word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
                    pooling_mode=self.last_layer,
                )
                self.encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            else:
                raise NotImplementedError
        else:
            self.encoder = encoder
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        if self.use_cuda:
            self.encoder.to('cuda')

    def init_tree_similarity(self):
        if not self.use_tree_similarity:
            return
        self.tree_sim = OntologyTree(dataset_name_or_path=self.dataset_name_or_path, tree_ratio=self.tree_ratio)
