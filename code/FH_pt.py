import re
try:
    import pyphen
    _hyph_pt_br = pyphen.Pyphen(lang='pt_BR')
    _hyph_pt_pt = pyphen.Pyphen(lang='pt_PT')
except Exception:
    _hyph_pt_br = _hyph_pt_pt = None

# --- Tokenization ---
WORD_RE_PT = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", re.UNICODE)  # includes áâãà ç éê í óôõ ú ü etc.

def tokenize_words_pt(text: str):
    return WORD_RE_PT.findall(text)

def count_sentences_pt(text: str):
    # Keep it simple: ., !, ?, … as boundaries
    parts = re.split(r"[.!?…]+", text)
    return max(1, sum(1 for p in parts if p.strip()))

def count_syllables_pt(word: str) -> int:
    # Prefer hyphenation dictionaries (pt_BR first, then pt_PT)
    if _hyph_pt_br or _hyph_pt_pt:
        hyph = (_hyph_pt_br or _hyph_pt_pt).inserted(word)
        return max(1, hyph.count('-') + 1)
    # Fallback: vowel-group heuristic (rough)
    groups = re.findall(r"[aeiouyAEIOUYàáâãéêíóôõúüÀÁÂÃÉÊÍÓÔÕÚÜ]+", word)
    return max(1, len(groups))

# --- Flesch Reading Ease (Portuguese adaptation) ---
def flesch_portuguese(text: str):
    words = tokenize_words_pt(text)
    W = len(words)
    if W == 0:
        return None
    S = count_sentences_pt(text)
    syl = sum(count_syllables_pt(w) for w in words)
    F = W / S               # words per sentence
    P = syl / W             # syllables per word
    score = 248.835 - 1.015 * F - 84.6 * P
    return round(score, 2)

# --- LIX / RIX ---
def lix(text: str):
    words = tokenize_words_pt(text)
    W = len(words)
    if W == 0:
        return None
    S = count_sentences_pt(text)
    long_words = sum(1 for w in words if len(w) > 6)
    return round((W / S) + (100.0 * long_words / W), 2)

def rix(text: str):
    words = tokenize_words_pt(text)
    W = len(words)
    if W == 0:
        return None
    S = count_sentences_pt(text)
    long_words = sum(1 for w in words if len(w) > 6)
    return round(long_words / S, 2)

# --- Band checks ---
FRE_PT_BANDS = {
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
if __name__ == "__main__":
    txt = "O paciente está bem. Os exames não mostram sinais de infecção. Recomenda-se apenas acompanhamento."
    fre = flesch_portuguese(txt)
    lx = lix(txt)
    rx = rix(txt)
    print("FRE-PT:", fre, "B1?", in_band(fre, 'B1', FRE_PT_BANDS, delta=1.0))
    print("LIX:", lx, "B1?", in_band(lx, 'B1', LIX_BANDS, delta=2.0))
    print("RIX:", rx)