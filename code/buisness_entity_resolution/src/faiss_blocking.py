import os
import time
import polars as pl
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

def create_dense_embeddings(texts: list[str], vectorizer=None, svd=None, fit=False):
    if fit:
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
        svd = TruncatedSVD(n_components=128, random_state=42)
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        dense_matrix = svd.fit_transform(tfidf_matrix)
        dense_matrix = normalize(dense_matrix, norm='l2', axis=1)
        
        return np.ascontiguousarray(dense_matrix, dtype=np.float32), vectorizer, svd
    else:
        tfidf_matrix = vectorizer.transform(texts)
        dense_matrix = svd.transform(tfidf_matrix)
        dense_matrix = normalize(dense_matrix, norm='l2', axis=1)
        
        return np.ascontiguousarray(dense_matrix, dtype=np.float32)

def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def retrieve_candidates(query_embeddings: np.ndarray, index: faiss.Index, target_ids: list[str], k=15):
    distances, indices = index.search(query_embeddings, k)

    results = []
    for row_idx, neighbor_indices in enumerate(indices):
        matched_ids = [target_ids[idx] for idx in neighbor_indices if idx != -1]
        results.append(matched_ids)

    return results

def block_by_country(s1_df: pl.DataFrame, target_df: pl.DataFrame, country: str, k=15):
    s1_subset = s1_df.filter(pl.col("country") == country)
    target_subset = target_df.filter(pl.col("country") == country)

    if s1_subset.height == 0 or target_subset.height == 0:
        return []

    s1_texts = s1_subset["business_name_ipa"].fill_null("").to_list()
    target_texts = target_subset["business_name_ipa"].fill_null("").to_list()
    target_ids = target_subset["entity_id"].to_list()
    s1_ids = s1_subset["entity_id"].to_list()

    all_texts = s1_texts + target_texts
    _, trained_vec, trained_svd = create_dense_embeddings(all_texts, fit=True)

    target_embeddings = create_dense_embeddings(target_texts, trained_vec, trained_svd, fit=False)
    index = build_faiss_index(target_embeddings)

    s1_embeddings = create_dense_embeddings(s1_texts, trained_vec, trained_svd, fit=False)
    candidate_lists = retrieve_candidates(s1_embeddings, index, target_ids, k)

    return pl.DataFrame({
        "source1_entity_id": s1_ids,
        "matched_entity_ids": candidate_lists
    })

if __name__ == "__main__":
    init_s1 = "dataset/train/train_source1.tsv"
    init_s2 = "dataset/train/train_source2.tsv"

    if os.path.isfile(init_s1) and os.path.isfile(init_s2):
        frame_s1 = pl.read_csv(init_s1, separator="\t", infer_schema_length=0).head(1000)
        frame_s2 = pl.read_csv(init_s2, separator="\t", infer_schema_length=0).head(5000)

        frame_s1 = frame_s1.with_columns(pl.col("business_name").alias("business_name_ipa"))
        frame_s2 = frame_s2.with_columns(pl.col("business_name").alias("business_name_ipa"))

        target_country = frame_s1["country"].drop_nulls()[0]

        start_time = time.time()
        print(f"Running FAISS index and retrieval for: {target_country}")

        results_frame = block_by_country(frame_s1, frame_s2, target_country, k=5)

        print(f"Completed in {time.time() - start_time:.2f} seconds.")
        if isinstance(results_frame, pl.DataFrame):
            print(results_frame.head(10))
        else:
            print("No matches found.")
