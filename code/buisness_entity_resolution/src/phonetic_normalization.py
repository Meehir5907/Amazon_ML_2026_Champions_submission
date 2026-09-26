import os
import polars as pl
import concurrent.futures
import epitran
import nltk
from g2p_en import G2p

nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('cmudict', quiet=True)

def clean_unicode_anomalies(text: str) -> str:
    if not text:
        return ""
        
    text = text.replace('\u200c', '').replace('\u200d', '')
    
    nukta_map = {
        '\u0915\u093c': '\u0958',
        '\u0916\u093c': '\u0959',
        '\u0917\u093c': '\u095a',
        '\u091c\u093c': '\u095b',
        '\u0921\u093c': '\u095c',
        '\u0922\u093c': '\u095d',
        '\u092b\u093c': '\u095e',
        '\u092f\u093c': '\u095f'
    }
    for decomp, recomp in nukta_map.items():
        text = text.replace(decomp, recomp)
        
    text = text.replace('\u0b83\u0baa', 'fa')
    text = text.replace('\u0933', 'l') 
    text = text.replace('\u0945', '\u0947') 
    
    gujarati_to_deva = {i: i - 0x0180 for i in range(0x0a80, 0x0b00)}
    text = text.translate(gujarati_to_deva)
    
    return text

def process_epitran_chunk(unique_strings):
    epi_hi = epitran.Epitran('hin-Deva')
    epi_ta = epitran.Epitran('tam-Taml')
    epi_te = epitran.Epitran('tel-Telu')
    epi_kn = epitran.Epitran('kan-Knda')
    epi_be = epitran.Epitran('ben-Beng')
    g2p = G2p()
    
    arpa_to_ipa = {
        'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ',
        'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'EH': 'ɛ', 'ER': 'ɝ',
        'EY': 'eɪ', 'F': 'f', 'G': 'ɡ', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i',
        'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ',
        'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'ʃ',
        'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v', 'W': 'w',
        'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
    }
    
    results = {}
    for s in unique_strings:
        cleaned = clean_unicode_anomalies(s)
        
        if not cleaned.strip():
            results[s] = ""
            continue
            
        if any('\u0900' <= c <= '\u097f' for c in cleaned):
            results[s] = epi_hi.transliterate(cleaned)
        elif any('\u0980' <= c <= '\u09ff' for c in cleaned):
            results[s] = epi_be.transliterate(cleaned)
        elif any('\u0b80' <= c <= '\u0bff' for c in cleaned):
            results[s] = epi_ta.transliterate(cleaned)
        elif any('\u0c00' <= c <= '\u0c7f' for c in cleaned):
            results[s] = epi_te.transliterate(cleaned)
        elif any('\u0c80' <= c <= '\u0cff' for c in cleaned):
            results[s] = epi_kn.transliterate(cleaned)
        else:
            phonemes = g2p(cleaned)
            ipa_str = ""
            for p in phonemes:
                # Strip stress numbers to map correctly
                arpa = "".join(c for c in p if c.isalpha())
                ipa_str += arpa_to_ipa.get(arpa, arpa.lower() if arpa else "")
            results[s] = ipa_str
            
    return results

def build_phonetic_cache(df: pl.DataFrame, col_name: str) -> dict:
    unique_vals = df[col_name].drop_nulls().unique().to_list()
    
    cores = os.cpu_count() or 4
    chunk_size = max(1, len(unique_vals) // cores)
    chunks = [unique_vals[i:i + chunk_size] for i in range(0, len(unique_vals), chunk_size)]
    
    phonetic_cache = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        for result in executor.map(process_epitran_chunk, chunks):
            phonetic_cache.update(result)
            
    return phonetic_cache

def apply_phonetic_normalization(df: pl.DataFrame, col_name: str, cache: dict) -> pl.DataFrame:
    return df.with_columns(
        pl.col(col_name).replace(cache).alias(f"{col_name}_ipa")
    )

if __name__ == "__main__":
    import time
    
    test_path = "dataset/train/train_source2.tsv"
    if os.path.isfile(test_path):
        df = pl.read_csv(test_path, separator="\t", infer_schema_length=0).head(1000)
        
        start_time = time.time()
        print(f"Building cache for {df.height} rows...")
        cache = build_phonetic_cache(df, "business_name")
        
        df_ipa = apply_phonetic_normalization(df, "business_name", cache)
        
        print(f"Completed in {time.time() - start_time:.2f} seconds.")
        print(df_ipa.select(["business_name", "business_name_ipa"]).head(10))
