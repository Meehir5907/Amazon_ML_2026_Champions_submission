import argparse
import polars as pl
import os

SCRIPT_RANGES = {
    "Tamil": r"[\u0B80-\u0BFF]",
    "Gujarati": r"[\u0A80-\u0AFF]",
    "Telugu": r"[\u0C00-\u0C7F]",
    "Kannada": r"[\u0C80-\u0CFF]",
    "Bengali": r"[\u0980-\u09FF]",
    "Devanagari": r"[\u0900-\u097F]"
}

def sample_script(df, source_label, script_name, regex_pattern, n):
    hits = df.filter(pl.col("business_name").str.contains(regex_pattern))
    
    if hits.height == 0:
        print(f"  [{source_label}] no {script_name} rows found")
        return
        
    print(f"  [{source_label}] {hits.height:,} {script_name} rows found, showing {min(n, hits.height)}:")
    sample = hits.sample(n=min(n, hits.height), seed=42)
    
    for row in sample.iter_rows(named=True):
        print(f"    {row['entity_id']}\t{row['business_name']}\t{row['business_address']}\t{row['country']}")

def main():
    n = 8
    files = [
        ("train_source2", "dataset/train/train_source2.tsv"),
        ("train_source3", "dataset/train/train_source3.tsv"),
        ("test_source2", "dataset/test/test_source2.tsv"),
        ("test_source3", "dataset/test/test_source3.tsv"),
    ]

    for label, path in files:
        if not os.path.isfile(path):
            continue
            
        df = pl.read_csv(path, separator="\t", infer_schema_length=0)
        print(f"\n=== {label} ({df.height:,} rows) ===")
        
        for script_name, regex_pattern in SCRIPT_RANGES.items():
            sample_script(df, label, script_name, regex_pattern, n)

if __name__ == "__main__":
    main()
