import os
import torch
import polars as pl
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model_global = None

def fetch_embedding_model():
    global model_global
    if model_global is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_global = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
    return model_global

def create_dense_embeddings(list_texts: list[str], model_embed: SentenceTransformer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matrix_dense = model_embed.encode(
        list_texts,
        batch_size=512,
        device=device,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return np.ascontiguousarray(matrix_dense, dtype=np.float32)

def build_faiss_index(matrix_embed: np.ndarray):
    dim_size = matrix_embed.shape[1]
    index_faiss = faiss.IndexFlatIP(dim_size)
    index_faiss.add(matrix_embed)
    return index_faiss

def retrieve_candidates(matrix_query: np.ndarray, index_faiss: faiss.Index, list_targets: list[str], k: int = 50, **kwargs):
    search_k = kwargs.get("val_k", k)
    _, matrix_indices = index_faiss.search(matrix_query, search_k)
    list_results = []
    
    for array_neighbors in matrix_indices:
        list_matched = [list_targets[idx] for idx in array_neighbors if idx != -1]
        list_results.append(list_matched)
        
    return list_results

def block_by_country(frame_s1: pl.DataFrame, frame_target: pl.DataFrame, label_country: str, k: int = 50, **kwargs):
    search_k = kwargs.get("val_k", k)
    frame_sub_s1 = frame_s1.filter(pl.col("country") == label_country)
    frame_sub_target = frame_target.filter(pl.col("country") == label_country)

    if frame_sub_s1.height == 0 or frame_sub_target.height == 0:
        return pl.DataFrame()

    list_names_s1 = frame_sub_s1["business_name"].fill_null("").to_list()
    list_names_target = frame_sub_target["business_name"].fill_null("").to_list()
    list_ids_target = frame_sub_target["entity_id"].to_list()
    list_ids_s1 = frame_sub_s1["entity_id"].to_list()
    
    model_embed = fetch_embedding_model()

    print(f"  Encoding {len(list_names_target):,} target records for {label_country}...")
    matrix_target = create_dense_embeddings(list_names_target, model_embed)
    index_faiss = build_faiss_index(matrix_target)

    print(f"  Encoding {len(list_names_s1):,} S1 query records for {label_country}...")
    matrix_s1 = create_dense_embeddings(list_names_s1, model_embed)
    list_candidates = retrieve_candidates(matrix_s1, index_faiss, list_ids_target, k=search_k)

    return pl.DataFrame({
        "source1_entity_id": list_ids_s1,
        "matched_entity_ids": list_candidates
    })
