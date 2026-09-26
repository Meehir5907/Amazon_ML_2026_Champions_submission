import argparse
import os
import unicodedata
from collections import Counter
import concurrent.futures
import polars as pl

DELIM = "\t"

def load_data(path):
    if not os.path.isfile(path):
        print(f"  [skip] {path} not found")
        return None
    
    df = pl.read_csv(path, separator=DELIM, infer_schema_length=0)
    print(f"  loaded {os.path.basename(path)}: {df.height:,} rows, cols={df.columns}")
    return df

def dominant_script(s):
    if not s: return "empty"
    counts = Counter()
    for ch in s:
        if not ch.isalpha(): continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        counts[name.split()[0]] += 1
    return counts.most_common(1)[0][0] if counts else "no_alpha"

def process_scripts(chunk):
    return [dominant_script(s) for s in chunk]

def profile_source(df, label):
    print(f"\n--- {label} ---")
    size = df.height
    if size == 0: return

    for col in ("business_name", "business_address"):
        if col not in df.columns: continue
        
        blank = df.filter(pl.col(col).is_null() | (pl.col(col).str.strip_chars() == "")).height
        lengths = df.select(pl.col(col).str.len_chars().alias("len"))
        
        print(f"  {col}: {blank:,} blank ({blank/size:.1%}), "
              f"len mean={lengths.mean().item():.1f} median={lengths.median().item():.0f} "
              f"max={lengths.max().item()}")

    if "business_name" in df.columns:
        names = df["business_name"].to_list()
        cores = os.cpu_count() or 4
        chunk_size = max(1, len(names) // cores)
        chunks = [names[i:i + chunk_size] for i in range(0, len(names), chunk_size)]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
            results = list(executor.map(process_scripts, chunks))
        
        flat_scripts = [s for res in results for s in res]
        top_scripts = Counter(flat_scripts).most_common(6)
        print(f"  business_name dominant script mix: {top_scripts}")

        dup_names = df.select(pl.col("business_name").str.strip_chars().str.to_lowercase()).n_unique()
        print(f"  exact-lowercased duplicate business_name rate: {1 - dup_names / max(size, 1):.1%}")

    if "country" in df.columns:
        counts = dict(df["country"].value_counts().iter_rows())
        print(f"  country counts: {counts}")

    if "entity_id" in df.columns:
        prefixes = dict(df.select(pl.col("entity_id").str.extract(r"^(S\d)-")).drop_nulls()["entity_id"].value_counts().iter_rows())
        lengths = df.select(pl.col("entity_id").str.len_chars())
        print(f"  entity_id prefixes: {prefixes}, id length range: {lengths.min().item()}-{lengths.max().item()}")

def profile_ground_truth(gt, s1, s2, s3):
    print("\n--- train_ground_truth ---")
    size = gt.height
    print(f"  rows: {size:,}")

    exploded = gt.select(
        pl.col("source1_entity_id"),
        pl.col("matched_entity_ids")
        .fill_null("")
        .str.split(",")
        .list.eval(pl.element().filter(pl.element().str.strip_chars() != ""))
        .alias("matches_list")
    )
    
    counts = exploded.select(pl.col("matches_list").list.len().alias("count"))
    print(f"  matches per S1 entity: mean={counts.mean().item():.2f} median={counts.median().item():.0f} max={counts.max().item()}")
    print(f"  singleton rate (no matches): {counts.filter(pl.col('count') == 0).height / size:.1%}")

    flat_ids_df = exploded.select(pl.col("matches_list").list.explode().drop_nulls()).filter(pl.col("matches_list") != "")
    all_matched = flat_ids_df.n_unique()
    s2_matched = flat_ids_df.filter(pl.col("matches_list").str.starts_with("S2-")).n_unique()
    s3_matched = flat_ids_df.filter(pl.col("matches_list").str.starts_with("S3-")).n_unique()
    print(f"  distinct matched IDs: {all_matched:,} (S2: {s2_matched:,}, S3: {s3_matched:,})")

    flat_unique = set(flat_ids_df["matches_list"].unique().to_list())
    if s2 is not None:
        s2_ids = set(s2["entity_id"].to_list())
        matched_s2 = {x for x in flat_unique if x.startswith("S2-")}
        print(f"  S2 coverage: {len(matched_s2 & s2_ids):,}/{len(s2_ids):,} S2 rows appear as a match ({len(matched_s2 & s2_ids)/max(len(s2_ids),1):.1%})")
        
    if s3 is not None:
        s3_ids = set(s3["entity_id"].to_list())
        matched_s3 = {x for x in flat_unique if x.startswith("S3-")}
        print(f"  S3 coverage: {len(matched_s3 & s3_ids):,}/{len(s3_ids):,} S3 rows appear as a match ({len(matched_s3 & s3_ids)/max(len(s3_ids),1):.1%})")

    if s1 is not None and s2 is not None and s3 is not None:
        s1_map = dict(zip(s1["entity_id"], s1["country"]))
        s2_map = dict(zip(s2["entity_id"], s2["country"]))
        s3_map = dict(zip(s3["entity_id"], s3["country"]))
        
        mismatches = 0
        checked = 0
        for row in exploded.iter_rows(named=True):
            c1 = s1_map.get(row["source1_entity_id"])
            if not c1: continue
            for mid in row["matches_list"]:
                c2 = s2_map.get(mid) if mid.startswith("S2-") else s3_map.get(mid)
                if c1 and c2:
                    checked += 1
                    if c1 != c2: mismatches += 1
        
        if checked:
            print(f"  cross-source country agreement: {checked-mismatches:,}/{checked:,} ({(checked-mismatches)/checked:.1%}) — {'safe to block on country' if mismatches/checked < 0.01 else 'NOT always equal'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    args = ap.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")

    print("=== TRAIN ===")
    train_paths = [
        os.path.join(train_dir, "train_source1.tsv"),
        os.path.join(train_dir, "train_source2.tsv"),
        os.path.join(train_dir, "train_source3.tsv"),
        os.path.join(train_dir, "train_ground_truth.tsv")
    ]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        train_dfs = list(executor.map(load_data, train_paths))
    s1, s2, s3, gt = train_dfs
    
    if s1 is not None: profile_source(s1, "train_source1")
    if s2 is not None: profile_source(s2, "train_source2")
    if s3 is not None: profile_source(s3, "train_source3")
    if gt is not None: profile_ground_truth(gt, s1, s2, s3)

    print("\n=== TEST ===")
    test_paths = [
        os.path.join(test_dir, "test_source1.tsv"),
        os.path.join(test_dir, "test_source2.tsv"),
        os.path.join(test_dir, "test_source3.tsv")
    ]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        test_dfs = list(executor.map(load_data, test_paths))
    t1, t2, t3 = test_dfs
    
    if t1 is not None: profile_source(t1, "test_source1")
    if t2 is not None: profile_source(t2, "test_source2")
    if t3 is not None: profile_source(t3, "test_source3")

    if s1 is not None and t1 is not None:
        train_countries = set(s1["country"].unique()) | set(s2["country"].unique()) | set(s3["country"].unique())
        test_countries = set(t1["country"].unique()) | set(t2["country"].unique()) | set(t3["country"].unique())
        print(f"\ntrain countries: {sorted(list(train_countries))}")
        print(f"test countries: {sorted(list(test_countries))}")
        print(f"test-only countries: {sorted(list(test_countries - train_countries))}")

if __name__ == "__main__":
    main()
