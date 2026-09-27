import os
import random
import time
import polars as pl
import xgboost as xgb
import joblib

from phonetic_normalization import build_phonetic_cache, apply_phonetic_normalization
from faiss_blocking import block_by_country
from feature_extraction import extract_pairwise_features
from extra_features import add_extended_features
from pipeline_cache import load_or_build_cache

FEATURE_COLS = [
    "name_levenshtein", "ipa_levenshtein", "address_levenshtein",
    "name_jaccard", "address_jaccard",
    "name_token_sort", "name_partial_ratio", "is_acronym",
    "name_len_diff", "zip_match",
    "ipa_token_sort", "ipa_partial_ratio", "addr_token_sort", "street_match"
]

def load_ground_truth(path: str) -> pl.DataFrame:
    frame_raw = pl.read_csv(path, separator="\t", infer_schema_length=0)
    frame_exploded = frame_raw.select(
        pl.col("source1_entity_id"),
        pl.col("matched_entity_ids")
            .fill_null("")
            .str.split(",")
            .list.eval(pl.element().filter(pl.element().str.strip_chars() != ""))
            .alias("target_entity_id")
    ).explode("target_entity_id").drop_nulls()
    return frame_exploded.with_columns(pl.lit(1).alias("label"))

def build_candidates_and_features(frame_s1, frame_target, source_label):
    countries = frame_s1["country"].drop_nulls().unique().to_list()
    list_candidates = []
    for country in countries:
        frame_block = block_by_country(frame_s1, frame_target, country, k=50)
        if isinstance(frame_block, pl.DataFrame) and frame_block.height > 0:
            list_candidates.append(frame_block)
    if not list_candidates:
        print(f"  {source_label}: no candidates generated")
        return None
    frame_candidates = pl.concat(list_candidates)
    print(f"  {source_label}: candidates generated for {frame_candidates.height} S1 entities")
    frame_base = extract_pairwise_features(frame_candidates, frame_s1, frame_target)
    return add_extended_features(frame_base, frame_s1, frame_target)

def prepare_training_data(frame_s1: pl.DataFrame, frame_s2: pl.DataFrame, frame_s3: pl.DataFrame) -> pl.DataFrame:
    print("Step 1: Normalizing phonetics (S1 + S2 + S3 combined)...")
    combined_names = pl.concat([
        frame_s1.select("business_name"),
        frame_s2.select("business_name"),
        frame_s3.select("business_name"),
    ])

    cache_phonetic = load_or_build_cache(combined_names, build_phonetic_cache)

    frame_s1 = apply_phonetic_normalization(frame_s1, "business_name", cache_phonetic)
    frame_s2 = apply_phonetic_normalization(frame_s2, "business_name", cache_phonetic)
    frame_s3 = apply_phonetic_normalization(frame_s3, "business_name", cache_phonetic)

    print("Step 2+3: Blocking + features against S2 and S3...")
    feats_s2 = build_candidates_and_features(frame_s1, frame_s2, "S2")
    feats_s3 = build_candidates_and_features(frame_s1, frame_s3, "S3")
    parts = [f for f in (feats_s2, feats_s3) if f is not None]
    if not parts:
        raise RuntimeError("No candidates generated against either S2 or S3.")
    return pl.concat(parts)

def f_beta_per_entity(predicted_ids: set, actual_ids: set, beta: float = 0.5) -> float:
    if not predicted_ids and not actual_ids:
        return 1.0
    if not predicted_ids or not actual_ids:
        return 0.0
    tp = len(predicted_ids & actual_ids)
    if tp == 0:
        return 0.0
    precision = tp / len(predicted_ids)
    recall = tp / len(actual_ids)
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)

def find_best_threshold(frame_val: pl.DataFrame, frame_truth: pl.DataFrame, val_s1_ids: set, clf) -> float:
    x_val = frame_val.select(FEATURE_COLS).to_pandas()
    probs = clf.predict_proba(x_val)[:, 1]
    frame_val_scored = frame_val.with_columns(pl.Series("prob", probs))

    actual_by_entity = {}
    for row in frame_truth.filter(pl.col("source1_entity_id").is_in(list(val_s1_ids))).iter_rows(named=True):
        actual_by_entity.setdefault(row["source1_entity_id"], set()).add(row["target_entity_id"])

    best_thresh = 0.50
    best_score = 0.0

    print("\n--- Sweeping thresholds for macro F_0.5 ---")
    for thresh_int in range(45, 85, 5):
        thresh = thresh_int / 100.0
        frame_pred = frame_val_scored.filter(pl.col("prob") >= thresh)
        
        predicted_by_entity = {}
        for row in frame_pred.iter_rows(named=True):
            predicted_by_entity.setdefault(row["source1_entity_id"], set()).add(row["target_entity_id"])

        scores = [
            f_beta_per_entity(predicted_by_entity.get(s1, set()), actual_by_entity.get(s1, set()))
            for s1 in val_s1_ids
        ]
        macro_score = sum(scores) / len(scores) if scores else 0.0
        print(f"Threshold: {thresh:.2f} -> Macro F_0.5: {macro_score:.4f}")
        
        if macro_score > best_score:
            best_score = macro_score
            best_thresh = thresh

    print(f"\nOptimal Threshold: {best_thresh:.2f} (Macro F_0.5: {best_score:.4f})")
    return best_thresh

def train_xgboost(frame_features: pl.DataFrame, frame_truth: pl.DataFrame, model_path: str):
    frame_train_labeled = frame_features.join(
        frame_truth, on=["source1_entity_id", "target_entity_id"], how="left"
    ).with_columns(pl.col("label").fill_null(0))

    all_s1_ids = frame_train_labeled["source1_entity_id"].unique().to_list()
    random.seed(42)
    random.shuffle(all_s1_ids)
    split_idx = int(len(all_s1_ids) * 0.8)
    train_ids, val_ids = set(all_s1_ids[:split_idx]), set(all_s1_ids[split_idx:])

    frame_tr = frame_train_labeled.filter(pl.col("source1_entity_id").is_in(train_ids))
    frame_va = frame_train_labeled.filter(pl.col("source1_entity_id").is_in(val_ids))

    x_train = frame_tr.select(FEATURE_COLS).to_pandas()
    y_train = frame_tr["label"].to_pandas()

    pos_sum = y_train.sum()
    pos_weight = (len(y_train) - pos_sum) / pos_sum if pos_sum > 0 else 1.0

    print("Step 5: Training initial XGBoost classifier...")
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    clf.fit(x_train, y_train)

    train_probs = clf.predict_proba(x_train)[:, 1]
    frame_tr_scored = frame_tr.with_columns(pl.Series("prob", train_probs))
    hard_negs = frame_tr_scored.filter((pl.col("label") == 0) & (pl.col("prob") > 0.35))
    
    if hard_negs.height > 0:
        print(f"Mining {hard_negs.height} hard negative samples for second-pass adversarial training...")
        frame_tr_boosted = pl.concat([frame_tr, hard_negs.drop("prob")])
        x_train_boosted = frame_tr_boosted.select(FEATURE_COLS).to_pandas()
        y_train_boosted = frame_tr_boosted["label"].to_pandas()
        
        pos_sum_b = y_train_boosted.sum()
        pos_weight_b = (len(y_train_boosted) - pos_sum_b) / pos_sum_b if pos_sum_b > 0 else 1.0
        
        clf = xgb.XGBClassifier(
            n_estimators=350,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight_b,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )
        clf.fit(x_train_boosted, y_train_boosted)

    best_thresh = find_best_threshold(frame_va, frame_truth, val_ids, clf)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": clf, "threshold": best_thresh}, model_path)
    print(f"Model and optimal threshold successfully saved to {model_path}")

if __name__ == "__main__":
    init_s1 = "dataset/train/train_source1.tsv"
    init_s2 = "dataset/train/train_source2.tsv"
    init_s3 = "dataset/train/train_source3.tsv"
    init_truth = "dataset/train/train_ground_truth.tsv"
    model_output = "code/business_entity_resolution/models/xgb_model.pkl"

    if all(os.path.isfile(p) for p in (init_s1, init_s2, init_s3, init_truth)):
        frame_s1_full = pl.read_csv(init_s1, separator="\t", infer_schema_length=0)
        frame_s2_full = pl.read_csv(init_s2, separator="\t", infer_schema_length=0)
        frame_s3_full = pl.read_csv(init_s3, separator="\t", infer_schema_length=0)
        frame_truth_full = load_ground_truth(init_truth)

        from blocking_diagnostics import build_smart_sample, measure_blocking_recall
        frame_s1, frame_s2, frame_s3 = build_smart_sample(
            frame_s1_full, frame_s2_full, frame_s3_full, frame_truth_full,
            n_s1=2500, n_filler=15000, seed=42
        )

        start_time = time.time()
        frame_features = prepare_training_data(frame_s1, frame_s2, frame_s3)
        measure_blocking_recall(frame_features, frame_truth_full)
        train_xgboost(frame_features, frame_truth_full, model_output)
        print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds.")
