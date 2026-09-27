import os
import json
import polars as pl

CACHE_FILE = "phonetic_cache.json"

def load_or_build_cache(combined_names_df: pl.DataFrame, build_fn) -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    cache = build_fn(combined_names_df, "business_name")

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

    return cache
