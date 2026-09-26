import polars as pl
import editdistance

def compute_normalized_levenshtein(string_a: str, string_b: str) -> float:
    if not string_a and not string_b:
        return 1.0
    if not string_a or not string_b:
        return 0.0

    distance = editdistance.eval(string_a, string_b)
    maximum_length = max(len(string_a), len(string_b))
    return 1.0 - (distance / maximum_length)

def compute_jaccard_similarity(string_a: str, string_b: str) -> float:
    if not string_a and not string_b:
        return 1.0
    if not string_a or not string_b:
        return 0.0

    set_a = set(string_a.lower().split())
    set_b = set(string_b.lower().split())

    if not set_a and not set_b:
        return 1.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union

def extract_pairwise_features(frame_candidates: pl.DataFrame, frame_s1: pl.DataFrame, frame_target: pl.DataFrame) -> pl.DataFrame:
    frame_pairs = frame_candidates.explode("matched_entity_ids").drop_nulls()
    frame_pairs = frame_pairs.rename({"matched_entity_ids": "target_entity_id"})

    frame_pairs = frame_pairs.join(frame_s1, left_on="source1_entity_id", right_on="entity_id", how="left")
    frame_pairs = frame_pairs.join(frame_target, left_on="target_entity_id", right_on="entity_id", how="left", suffix="_target")

    list_names_s1 = frame_pairs["business_name"].fill_null("").to_list()
    list_names_target = frame_pairs["business_name_target"].fill_null("").to_list()

    list_ipa_s1 = frame_pairs["business_name_ipa"].fill_null("").to_list()
    list_ipa_target = frame_pairs["business_name_ipa_target"].fill_null("").to_list()

    list_address_s1 = frame_pairs["business_address"].fill_null("").to_list()
    list_address_target = frame_pairs["business_address_target"].fill_null("").to_list()

    score_name_lev = [compute_normalized_levenshtein(a, b) for a, b in zip(list_names_s1, list_names_target)]
    score_ipa_lev = [compute_normalized_levenshtein(a, b) for a, b in zip(list_ipa_s1, list_ipa_target)]
    score_address_lev = [compute_normalized_levenshtein(a, b) for a, b in zip(list_address_s1, list_address_target)]

    score_name_jac = [compute_jaccard_similarity(a, b) for a, b in zip(list_names_s1, list_names_target)]
    score_address_jac = [compute_jaccard_similarity(a, b) for a, b in zip(list_address_s1, list_address_target)]

    frame_features = frame_pairs.select([
        pl.col("source1_entity_id"),
        pl.col("target_entity_id"),
        pl.Series("name_levenshtein", score_name_lev, dtype=pl.Float32),
        pl.Series("ipa_levenshtein", score_ipa_lev, dtype=pl.Float32),
        pl.Series("address_levenshtein", score_address_lev, dtype=pl.Float32),
        pl.Series("name_jaccard", score_name_jac, dtype=pl.Float32),
        pl.Series("address_jaccard", score_address_jac, dtype=pl.Float32)
    ])

    return frame_features

if __name__ == "__main__":
    import os
    import time
    from faiss_blocking import block_by_country

    init_s1 = "dataset/train/train_source1.tsv"
    init_s2 = "dataset/train/train_source2.tsv"

    if os.path.isfile(init_s1) and os.path.isfile(init_s2):
        frame_s1 = pl.read_csv(init_s1, separator="\t", infer_schema_length=0).head(1000)
        frame_s2 = pl.read_csv(init_s2, separator="\t", infer_schema_length=0).head(5000)

        frame_s1 = frame_s1.with_columns(pl.col("business_name").alias("business_name_ipa"))
        frame_s2 = frame_s2.with_columns(pl.col("business_name").alias("business_name_ipa"))

        target_country = frame_s1["country"].drop_nulls()[0]

        print("Running blocking...")
        frame_candidates = block_by_country(frame_s1, frame_s2, target_country, k=5)

        start_time = time.time()
        print(f"Extracting features for {frame_candidates.height} candidate arrays...")
        frame_features = extract_pairwise_features(frame_candidates, frame_s1, frame_s2)

        print(f"Completed in {time.time() - start_time:.2f} seconds.")
        print(frame_features.head(10))
