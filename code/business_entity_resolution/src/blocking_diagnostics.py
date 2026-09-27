import polars as pl

def measure_blocking_recall(frame_features: pl.DataFrame, frame_truth: pl.DataFrame) -> None:
    """Isolates blocking failure (recall ceiling) from classifier failure
    (ranking/precision)."""
    
    # Filter the ground truth down to ONLY the S1 entities present in this test run
    s1_ids = frame_features["source1_entity_id"].unique().to_list()
    truth_filtered = frame_truth.filter(pl.col("source1_entity_id").is_in(s1_ids))

    candidate_sets = (
        frame_features.group_by("source1_entity_id")
        .agg(pl.col("target_entity_id").alias("candidates"))
    )
    cand_map = {row["source1_entity_id"]: set(row["candidates"]) for row in candidate_sets.iter_rows(named=True)}

    truth_map = {}
    for row in truth_filtered.iter_rows(named=True):
        truth_map.setdefault(row["source1_entity_id"], set()).add(row["target_entity_id"])

    recalls = []
    zero_recall = 0
    for s1, true_ids in truth_map.items():
        if not true_ids:
            continue  
        candidates = cand_map.get(s1, set())
        found = len(true_ids & candidates)
        r = found / len(true_ids)
        recalls.append(r)
        if r == 0:
            zero_recall += 1

    if not recalls:
        print("No non-singleton entities found in ground truth for this sample.")
        return

    mean_recall = sum(recalls) / len(recalls)
    print(f"\n=== Blocking recall diagnostic ===")
    print(f"  non-singleton S1 entities checked: {len(recalls)}")
    print(f"  mean per-entity recall: {mean_recall:.3f}")
    print(f"  entities with ZERO true matches found in candidates: {zero_recall}/{len(recalls)} "
          f"({zero_recall/len(recalls):.1%})")
    if mean_recall < 0.3:
        print("  -> Recall ceiling is low. This is very likely a SAMPLING problem "
              "(true matches not present in the truncated S2/S3 subset), not a "
              "blocking-algorithm or classifier problem. See build_smart_sample().")


def build_smart_sample(frame_s1_full: pl.DataFrame, frame_s2_full: pl.DataFrame,
                        frame_s3_full: pl.DataFrame, frame_truth_full: pl.DataFrame,
                        n_s1: int = 2500, n_filler: int = 15000, seed: int = 42):
    """Builds a small but HONEST dev sample for fast local iteration: samples
    n_s1 random S1 entities, then guarantees their real true-match rows are
    included in the S2/S3 subset (unlike plain .head(), which essentially
    never includes them at this data scale), plus n_filler random rows as
    realistic negative/distractor noise so blocking still has real work to do.

    This is for fast local dev iteration only -- the real submission still
    needs a full-scale run over the complete dataset.
    """
    s1_sample = frame_s1_full.sample(n=min(n_s1, frame_s1_full.height), seed=seed)
    s1_ids = set(s1_sample["source1_entity_id" if "source1_entity_id" in s1_sample.columns else "entity_id"].to_list())

    truth_subset = frame_truth_full.filter(pl.col("source1_entity_id").is_in(list(s1_ids)))
    required_s2_ids = set(
        truth_subset.filter(pl.col("target_entity_id").str.starts_with("S2-"))["target_entity_id"].to_list()
    )
    required_s3_ids = set(
        truth_subset.filter(pl.col("target_entity_id").str.starts_with("S3-"))["target_entity_id"].to_list()
    )

    s2_required = frame_s2_full.filter(pl.col("entity_id").is_in(list(required_s2_ids)))
    s3_required = frame_s3_full.filter(pl.col("entity_id").is_in(list(required_s3_ids)))

    s2_filler = frame_s2_full.sample(n=min(n_filler, frame_s2_full.height), seed=seed)
    s3_filler = frame_s3_full.sample(n=min(n_filler, frame_s3_full.height), seed=seed)

    frame_s2_sample = pl.concat([s2_required, s2_filler]).unique(subset=["entity_id"])
    frame_s3_sample = pl.concat([s3_required, s3_filler]).unique(subset=["entity_id"])

    print(f"Smart sample: {s1_sample.height} S1 entities, "
          f"{len(required_s2_ids)} required + {s2_filler.height} filler S2 rows "
          f"({frame_s2_sample.height} unique), "
          f"{len(required_s3_ids)} required + {s3_filler.height} filler S3 rows "
          f"({frame_s3_sample.height} unique)")

    return s1_sample, frame_s2_sample, frame_s3_sample
