import os
import re
import polars as pl
import concurrent.futures
import epitran
import nltk
from g2p_en import G2p

nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('cmudict', quiet=True)

ZWNJ, ZWJ = '\u200c', '\u200d'

BLOCKS = {
    "Devanagari": (0x0900, 0x097F),
    "Tamil":      (0x0B80, 0x0BFF),
    "Telugu":     (0x0C00, 0x0C7F),
    "Kannada":    (0x0C80, 0x0CFF),
    "Bengali":    (0x0980, 0x09FF),
    "Gujarati":   (0x0A80, 0x0AFF),
}
GUJARATI_TO_DEVA = {i: i - 0x0180 for i in range(0x0A80, 0x0B00)}

TAMIL_DIGRAPHS = {
    "\u0b83\u0baa": "f",   # ஃப
    "\u0b83\u0b9c": "z",   # ஃஜ
    "\u0b83\u0bb8": "s",   # ஃஸ
    "\u0b83\u0bb9": "h",   # ஃஹ
    "\u0b83\u0bb1": "r",   # ஃற
}

NUKTA_MAP = {
    '\u0915\u093c': '\u0958', '\u0916\u093c': '\u0959', '\u0917\u093c': '\u095a',
    '\u091c\u093c': '\u095b', '\u0921\u093c': '\u095c', '\u0922\u093c': '\u095d',
    '\u092b\u093c': '\u095e', '\u092f\u093c': '\u095f',
}


def clean_devanagari_patches(text: str) -> str:
    """Patches that target Devanagari codepoints specifically. Must run AFTER
    any Gujarati->Devanagari bridging, or Gujarati input never gets them."""
    for decomp, recomp in NUKTA_MAP.items():
        text = text.replace(decomp, recomp)
    text = text.replace('\u0933', 'l')          # ळ (LLA) -> l  (standalone / consonant form)
    text = text.replace('\u0945', '\u0947')      # candra-E SIGN -> E sign
    text = text.replace('\u0911', '\u0913')      # candra-O LETTER (word-initial)
    text = text.replace('\u090d', '\u090f')      # candra-E LETTER (word-initial)
    return text


def clean_unicode_anomalies(text: str) -> str:
    if not text:
        return ""

    text = text.replace(ZWNJ, '').replace(ZWJ, '')

    # FIX: bridge Gujarati -> Devanagari FIRST, so the Devanagari-specific
    # patches below actually get a chance to apply to Gujarati-origin text.
    text = text.translate(GUJARATI_TO_DEVA)

    text = clean_devanagari_patches(text)

    # FIX: match each digraph PLUS an optional trailing virama (pulli, U+0BCD)
    # in one regex, so the vowel-killing virama that's part of a larger
    # conjunct cluster (e.g. இன்ஃப்ராஸ்ட்ரக்சர் -> ...ஃப்ரா...) gets consumed
    # at the substitution site. A blanket strip of all virama would be wrong
    # -- ordinary conjuncts like மேனேஜ்மெண்ட் rely on virama being present
    # for Epitran to read them as consonant clusters, not open syllables.
    for digraph, phoneme in TAMIL_DIGRAPHS.items():
        text = re.sub(re.escape(digraph) + '\u0bcd?', phoneme, text)

    return text


# FIX: token-level script routing, not whole-string

def script_of(token: str):
    for ch in token:
        for script, (lo, hi) in BLOCKS.items():
            if lo <= ord(ch) <= hi:
                # Gujarati already bridged to Devanagari by this point, so
                # this branch only fires for native scripts.
                return script
    if re.search(r'[A-Za-z]', token):
        return "Latin"
    return "Other"


ARPA_TO_IPA = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ',
    'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'EH': 'ɛ', 'ER': 'ɝ',
    'EY': 'eɪ', 'F': 'f', 'G': 'ɡ', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i',
    'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ',
    'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ',
    'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v', 'W': 'w',
    'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
}


def process_epitran_chunk(unique_strings):
    epis = {
        "Devanagari": epitran.Epitran('hin-Deva'),
        "Tamil": epitran.Epitran('tam-Taml'),
        "Telugu": epitran.Epitran('tel-Telu'),
        "Kannada": epitran.Epitran('kan-Knda'),
        "Bengali": epitran.Epitran('ben-Beng'),
    }
    g2p = G2p()

    def word_to_ipa_latin(word):
        phonemes = g2p(word)
        ipa_str = ""
        for p in phonemes:
            arpa = "".join(c for c in p if c.isalpha())
            ipa_str += ARPA_TO_IPA.get(arpa, arpa.lower() if arpa else "")
        return ipa_str

    results = {}
    for s in unique_strings:
        cleaned = clean_unicode_anomalies(s)
        if not cleaned.strip():
            results[s] = ""
            continue

        # FIX: route each whitespace token by its own script, instead of
        # picking one G2P engine for the whole string. This is what makes
        # code-mixed names (e.g. "City மேனேஜ்மெண்ட் Limited") work correctly.
        out_tokens = []
        for token in cleaned.split():
            script = script_of(token)
            if script in epis:
                out_tokens.append(epis[script].transliterate(token))
            elif script == "Latin":
                out_tokens.append(word_to_ipa_latin(token))
            else:
                out_tokens.append(token)
        results[s] = " ".join(out_tokens)

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
    tests = [
        "City மேனேஜ்மெண்ட் Limited",           # code-mixed
        "My કન્સલ્ટિંગ Hardware Private Limited",  # code-mixed, Gujarati
        "મૉડર્ન ફાઇનાન્સ",                        # Gujarati candra-o
        "பர்ஃபெக்ட் இன்டர்நேஷனல்",                 # Tamil f-digraph
        "ஃபார்ச்சூன் இன்ஃப்ராஸ்ட்ரக்சர்",           # more Tamil aytham digraphs
    ]
    for t in tests:
        cache = process_epitran_chunk([t])
        print(f"{t!r}\n  -> {cache[t]!r}\n")
