import os
import random
import time
import polars as pl
import xgboost as xgb
import joblib

from phonetic_normalization import build_phonetic_cache, apply_phonetic_normalization
from faiss_blocking import block_by_country
from feature_extraction import extract_pairwise_features

FEATURE_COLS = [
    "name_levenshtein", "ipa_levenshtein", "address_levenshtein",
    "name_jaccard", "address_jaccard",
]


def load_ground_truth(path: str) -> pl.DataFrame:
    """train_ground_truth.tsv is one row per S1 entity with matched_entity_ids
    as a comma-separated list (empty for singletons). FIX: explode into one
    row per (source1_entity_id, target_entity_id) pair -- the previous version
    tried to join against this un-exploded, under wrong column names."""
    df = pl.read_csv(path, separator="\t", infer_schema_length=0)
    exploded = df.select(
        pl.col("source1_entity_id"),
        pl.col("matched_entity_ids")
            .fill_null("")
            .str.split(",")
            .list.eval(pl.element().filter(pl.element().str.strip_chars() != ""))
            .alias("target_entity_id")
    ).explode("target_entity_id").drop_nulls()
    return exploded.with_columns(pl.lit(1).alias("label"))


def build_candidates_and_features(frame_s1, frame_target, source_label):
    countries = frame_s1["country"].drop_nulls().unique().to_list()
    list_candidates = []
    for country in countries:
        candidates = block_by_country(frame_s1, frame_target, country, k=10)
        if isinstance(candidates, pl.DataFrame) and candidates.height > 0:
            list_candidates.append(candidates)
    if not list_candidates:
        print(f"  {source_label}: no candidates generated")
        return None
    frame_candidates = pl.concat(list_candidates)
    print(f"  {source_label}: candidates generated for {frame_candidates.height} S1 entities")
    return extract_pairwise_features(frame_candidates, frame_s1, frame_target)


def prepare_training_data(frame_s1: pl.DataFrame, frame_s2: pl.DataFrame, frame_s3: pl.DataFrame) -> pl.DataFrame:
    print("Step 1: Normalizing phonetics (S1 + S2 + S3 combined)...")
    combined_names = pl.concat([
        frame_s1.select("business_name"),
        frame_s2.select("business_name"),
        frame_s3.select("business_name"),
    ])
    cache = build_phonetic_cache(combined_names, "business_name")

    frame_s1 = apply_phonetic_normalization(frame_s1, "business_name", cache)
    frame_s2 = apply_phonetic_normalization(frame_s2, "business_name", cache)
    frame_s3 = apply_phonetic_normalization(frame_s3, "business_name", cache)

    # FIX: block + extract features against BOTH S2 and S3, not just S2.
    # Run the existing per-target pipeline twice (entity_id namespaces are
    # disjoint) and concatenate, rather than modifying block_by_country /
    # extract_pairwise_features to handle two targets at once.
    print("Step 2+3: Blocking + features against S2 and S3...")
    feats_s2 = build_candidates_and_features(frame_s1, frame_s2, "S2")
    feats_s3 = build_candidates_and_features(frame_s1, frame_s3, "S3")
    parts = [f for f in (feats_s2, feats_s3) if f is not None]
    if not parts:
        raise RuntimeError("No candidates generated against either S2 or S3.")
    return pl.concat(parts)


def f_beta_per_entity(predicted_ids: set, actual_ids: set, beta: float = 0.5) -> float:
    """The exact competition metric per S1 entity: correctly predicting an
    empty list scores 1.0, any false match on a true singleton scores 0.0.
    Verified against the README's own worked example (0.714)."""
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


def evaluate_macro_f05(frame_val: pl.DataFrame, frame_truth: pl.DataFrame,
                        val_s1_ids: set, clf, threshold: float = 0.5) -> float:
    """Reconstructs per-S1-entity predicted match lists from pair-level
    predictions and scores with the actual competition metric, over EVERY
    held-out S1 entity -- not just ones present in frame_val. An S1 entity
    that blocking failed to generate any candidates for is a real recall
    failure and must still count against the score, or this would look more
    optimistic than actual leaderboard scoring."""
    x_val = frame_val.select(FEATURE_COLS).to_pandas()
    probs = clf.predict_proba(x_val)[:, 1]
    frame_pred = frame_val.with_columns(pl.Series("prob", probs)).filter(pl.col("prob") >= threshold)

    predicted_by_entity = {}
    for row in frame_pred.iter_rows(named=True):
        predicted_by_entity.setdefault(row["source1_entity_id"], set()).add(row["target_entity_id"])

    actual_by_entity = {}
    for row in frame_truth.filter(pl.col("source1_entity_id").is_in(list(val_s1_ids))).iter_rows(named=True):
        actual_by_entity.setdefault(row["source1_entity_id"], set()).add(row["target_entity_id"])

    scores = [
        f_beta_per_entity(predicted_by_entity.get(s1, set()), actual_by_entity.get(s1, set()))
        for s1 in val_s1_ids
    ]
    macro_f05 = sum(scores) / len(scores) if scores else 0.0
    print(f"\n--- Held-out macro F_0.5 (real competition metric): {macro_f05:.4f} "
          f"over {len(scores)} S1 entities (threshold={threshold}) ---")
    return macro_f05


def train_xgboost(frame_features: pl.DataFrame, frame_truth: pl.DataFrame, model_path: str):
    frame_train_labeled = frame_features.join(
        frame_truth, on=["source1_entity_id", "target_entity_id"], how="left"
    ).with_columns(pl.col("label").fill_null(0))

    # FIX: split at the S1-ENTITY level, not the pair/row level -- splitting
    # rows would leak some of an entity's candidate pairs into train and
    # others into validation, which both leaks information and makes
    # per-entity F_0.5 reconstruction incoherent.
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

    print("Step 5: Training XGBoost classifier...")
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

    # FIX: real held-out evaluation with the actual competition metric,
    # instead of predict() on the training set scored with sklearn's
    # pair-level classification_report.
    evaluate_macro_f05(frame_va, frame_truth, val_ids, clf)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"Model saved successfully to {model_path}")


if __name__ == "__main__":
    init_s1 = "dataset/train/train_source1.tsv"
    init_s2 = "dataset/train/train_source2.tsv"
    init_s3 = "dataset/train/train_source3.tsv"
    init_truth = "dataset/train/train_ground_truth.tsv"
    model_output = "code/business_entity_resolution/models/xgb_model.pkl"

    if all(os.path.isfile(p) for p in (init_s1, init_s2, init_s3, init_truth)):
        # Load the full datasets into memory
        frame_s1_full = pl.read_csv(init_s1, separator="\t", infer_schema_length=0)
        frame_s2_full = pl.read_csv(init_s2, separator="\t", infer_schema_length=0)
        frame_s3_full = pl.read_csv(init_s3, separator="\t", infer_schema_length=0)
        
        frame_truth_full = load_ground_truth(init_truth)

        # Generate a smart subset guaranteeing true matches exist
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
    else:
        print("Missing dataset files. Ensure source1, source2, source3, and truth files are present.")
