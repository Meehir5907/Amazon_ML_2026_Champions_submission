import os
import polars as pl
import joblib

from phonetic_normalization import build_phonetic_cache, apply_phonetic_normalization
from faiss_blocking import block_by_country
from feature_extraction import extract_pairwise_features
from extra_features import add_extended_features
from pipeline_cache import load_or_build_cache

def build_candidates_for_target(frame_s1: pl.DataFrame, frame_target: pl.DataFrame, k_val: int = 50) -> pl.DataFrame:
    list_countries = frame_s1["country"].drop_nulls().unique().to_list()
    list_blocks = []
    
    for country in list_countries:
        frame_block = block_by_country(frame_s1, frame_target, country, k=k_val)
        if isinstance(frame_block, pl.DataFrame) and frame_block.height > 0:
            list_blocks.append(frame_block)
            
    if not list_blocks:
        return pl.DataFrame()
        
    return pl.concat(list_blocks)

def execute_inference_pipeline(init_s1: str, init_s2: str, init_s3: str, init_model: str, init_out_cand: str, init_out_match: str):
    print("Loading datasets...")
    frame_s1 = pl.read_csv(init_s1, separator="\t", infer_schema_length=0)
    frame_s2 = pl.read_csv(init_s2, separator="\t", infer_schema_length=0)
    frame_s3 = pl.read_csv(init_s3, separator="\t", infer_schema_length=0)
    
    print("Applying phonetic normalization...")
    frame_combined = pl.concat([
        frame_s1.select("business_name"),
        frame_s2.select("business_name"),
        frame_s3.select("business_name")
    ])
    
    cache_phonetic = load_or_build_cache(frame_combined, build_phonetic_cache)
    
    frame_s1 = apply_phonetic_normalization(frame_s1, "business_name", cache_phonetic)
    frame_s2 = apply_phonetic_normalization(frame_s2, "business_name", cache_phonetic)
    frame_s3 = apply_phonetic_normalization(frame_s3, "business_name", cache_phonetic)
    
    print(f"Generating candidates via FAISS (k=50)...")
    frame_cand_s2 = build_candidates_for_target(frame_s1, frame_s2)
    frame_cand_s3 = build_candidates_for_target(frame_s1, frame_s3)
    
    frame_candidates = pl.concat([f for f in (frame_cand_s2, frame_cand_s3) if f.height > 0])
    
    print(f"Exporting {init_out_cand}...")
    frame_cand_export = (
        frame_candidates.explode("matched_entity_ids")
        .drop_nulls()
        .group_by("source1_entity_id")
        .agg(pl.col("matched_entity_ids").alias("targets"))
        .with_columns(pl.col("targets").list.join(",").alias("matched_entity_ids"))
        .select(["source1_entity_id", "matched_entity_ids"])
    )
    frame_cand_export.write_csv(init_out_cand, separator="\t")
    
    print("Extracting pairwise and extended features...")
    frame_feat_s2 = extract_pairwise_features(frame_cand_s2, frame_s1, frame_s2)
    frame_feat_s2 = add_extended_features(frame_feat_s2, frame_s1, frame_s2)
    
    frame_feat_s3 = extract_pairwise_features(frame_cand_s3, frame_s1, frame_s3)
    frame_feat_s3 = add_extended_features(frame_feat_s3, frame_s1, frame_s3)
    
    frame_features = pl.concat([frame_feat_s2, frame_feat_s3])
    
    print("Running XGBoost inference...")
    model_xgb = joblib.load(init_model)
    
    list_features = [
        "name_levenshtein", "ipa_levenshtein", "address_levenshtein",
        "name_jaccard", "address_jaccard",
        "name_token_sort", "name_partial_ratio", "is_acronym", 
        "name_len_diff", "zip_match"
    ]
    
    matrix_infer = frame_features.select(list_features).to_pandas()
    array_probs = model_xgb.predict_proba(matrix_infer)[:, 1]
    
    frame_preds = frame_features.with_columns(pl.Series("prob", array_probs))
    frame_matches = frame_preds.filter(pl.col("prob") >= 0.5)
    
    print(f"Exporting {init_out_match}...")
    frame_match_export = (
        frame_matches.group_by("source1_entity_id")
        .agg(pl.col("target_entity_id").alias("targets"))
        .with_columns(pl.col("targets").list.join(",").alias("matched_entity_ids"))
    )
    
    frame_final = frame_s1.select(pl.col("entity_id").alias("source1_entity_id")).join(
        frame_match_export, on="source1_entity_id", how="left"
    ).with_columns(pl.col("matched_entity_ids").fill_null(""))
    
    frame_final.select(["source1_entity_id", "matched_entity_ids"]).write_csv(init_out_match, separator="\t")
    print("Pipeline execution complete.")

if __name__ == "__main__":
    execute_inference_pipeline(
        init_s1="dataset/test/test_source1.tsv",
        init_s2="dataset/test/test_source2.tsv",
        init_s3="dataset/test/test_source3.tsv",
        init_model="code/business_entity_resolution/models/xgb_model.pkl",
        init_out_cand="candidate_pairs.tsv",
        init_out_match="matching_results.tsv"
    )
