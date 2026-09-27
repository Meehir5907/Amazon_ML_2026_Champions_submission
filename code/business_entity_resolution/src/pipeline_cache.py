import os
import json
import polars as pl

CACHE_FILE = "phonetic_cache.json"

def load_or_build_cache(combined_names_df: pl.DataFrame, build_fn, cache_file: str = CACHE_FILE) -> dict:
    unique_names = set(combined_names_df["business_name"].drop_nulls().unique().to_list())

    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)

    missing = unique_names - cache.keys()
    if missing:
        print(f"  phonetic cache: {len(cache):,} cached, computing {len(missing):,} new name(s)...")
        missing_df = pl.DataFrame({"business_name": list(missing)})
        new_entries = build_fn(missing_df, "business_name")
        cache.update(new_entries)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    else:
        print(f"  phonetic cache: all {len(unique_names):,} name(s) already cached, nothing to compute")

    return cache
