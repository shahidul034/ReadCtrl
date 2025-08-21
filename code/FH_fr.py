import re
try:
    import pyphen
    _hyph_fr = pyphen.Pyphen(lang='fr')  # or 'fr_FR'
except Exception:
    _hyph_fr = None

# --- Basic French tokenization ---
WORD_RE_FR = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿœŒÆæ]+", re.UNICODE)

def tokenize_words_fr(text: str):
    return WORD_RE_FR.findall(text)

def count_sentences_fr(text: str):
    # Split on ., !, ?, … ; keep it simple
    parts = re.split(r"[.!?…]+", text)
    return max(1, sum(1 for p in parts if p.strip()))

def count_syllables_fr(word: str) -> int:
    if _hyph_fr:
        # Pyphen gives hyphenation points; count pieces as syllables (approx)
        hyph = _hyph_fr.inserted(word)
        return max(1, hyph.count('-') + 1)
    # Fallback: simple vowel-group heuristic (rougher)
    groups = re.findall(r"[aeiouyAEIOUYàâäéèêëîïôöùûüÿœAEIOUYÀÂÄÉÈÊËÎÏÔÖÙÛÜŸŒ]+", word)
    return max(1, len(groups))

# --- FRE-FR (Kandel & Moles) ---
def flesch_kandel_moles_fr(text: str):
    words = tokenize_words_fr(text)
    W = len(words)
    if W == 0:
        return None
    S = count_sentences_fr(text)
    syl = sum(count_syllables_fr(w) for w in words)
    P = (syl / W) * 100.0  # syllables per 100 words
    F = W / S              # words per sentence
    score = 207.0 - 1.015 * F - 0.736 * P
    return round(score, 2)

# --- LIX / RIX ---
def lix(text: str):
    words = tokenize_words_fr(text)
    W = len(words)
    if W == 0:
        return None
    S = count_sentences_fr(text)
    long_words = sum(1 for w in words if len(w) > 6)
    return round((W / S) + (100.0 * long_words / W), 2)

def rix(text: str):
    words = tokenize_words_fr(text)
    W = len(words)
    if W == 0:
        return None
    S = count_sentences_fr(text)
    long_words = sum(1 for w in words if len(w) > 6)
    return round(long_words / S, 2)

# --- Band checks ---
FRE_FR_BANDS = {
    'B1': (70, 100),
    'B2': (60, 70),
    'B3': (45, 60),
}
LIX_BANDS = {
    'B1': (20, 35),
    'B2': (35, 45),
    'B3': (45, 60),
}

def in_band(score, band, bands, delta=0.0):
    if score is None:
        return False
    lo, hi = bands[band]
    return (lo - delta) <= score <= (hi + delta)

# Example
# if __name__ == "__main__":
#     txt = "Le patient se porte bien. Les examens sont rassurants, sans signes d’infection. Un suivi simple est recommandé."
#     fre = flesch_kandel_moles_fr(txt)
#     lx = lix(txt)
#     rx = rix(txt)
#     print("FRE-FR:", fre, "B1?", in_band(fre, 'B1', FRE_FR_BANDS, delta=1.0))
#     print("LIX:", lx, "B1?", in_band(lx, 'B1', LIX_BANDS, delta=2.0))
#     print("RIX:", rx)