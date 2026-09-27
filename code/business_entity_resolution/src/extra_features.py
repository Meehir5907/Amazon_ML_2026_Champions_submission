import re
import polars as pl
from rapidfuzz import fuzz

def extract_zip_code(text: str) -> str:
    if not text:
        return ""
    match = re.search(r'\b\d{5,6}\b', text)
    return match.group(0) if match else ""

def is_acronym_match(name_a: str, name_b: str) -> float:
    if not name_a or not name_b: return 0.0

    tokens_a = [t for t in name_a.split() if len(t) > 0]
    tokens_b = [t for t in name_b.split() if len(t) > 0]

    if not tokens_a or not tokens_b: return 0.0

    acronym_a = "".join(t[0] for t in tokens_a).upper()
    acronym_b = "".join(t[0] for t in tokens_b).upper()

    if len(tokens_a) == 1 and name_a.upper() == acronym_b and len(acronym_b) >= 2: return 1.0
    if len(tokens_b) == 1 and name_b.upper() == acronym_a and len(acronym_a) >= 2: return 1.0
    return 0.0

def add_extended_features(frame_features: pl.DataFrame, frame_s1: pl.DataFrame, frame_target: pl.DataFrame) -> pl.DataFrame:
    merged = frame_features.join(frame_s1, left_on="source1_entity_id", right_on="entity_id", how="left")
    merged = merged.join(frame_target, left_on="target_entity_id", right_on="entity_id", how="left", suffix="_target")

    names_1 = merged["business_name"].fill_null("").to_list()
    names_2 = merged["business_name_target"].fill_null("").to_list()
    addr_1 = merged["business_address"].fill_null("").to_list()
    addr_2 = merged["business_address_target"].fill_null("").to_list()

    tsr = [fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(names_1, names_2)]
    partial_sim = [fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(names_1, names_2)]
    acronym_scores = [is_acronym_match(a, b) for a, b in zip(names_1, names_2)]
    len_diff = [abs(len(a) - len(b)) for a, b in zip(names_1, names_2)]

    zip_match = []
    for a, b in zip(addr_1, addr_2):
        za, zb = extract_zip_code(a), extract_zip_code(b)
        if za and zb:
            zip_match.append(1 if za == zb else 0)
        else:
            zip_match.append(-1)

    return frame_features.with_columns([
        pl.Series("name_token_sort", tsr, dtype=pl.Float32),
        pl.Series("name_partial_ratio", partial_sim, dtype=pl.Float32),
        pl.Series("is_acronym", acronym_scores, dtype=pl.Float32),
        pl.Series("name_len_diff", len_diff, dtype=pl.Int32),
        pl.Series("zip_match", zip_match, dtype=pl.Int8)
    ])
