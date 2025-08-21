import re
import pyphen

# --- Basic Spanish text stats ---
_dic = pyphen.Pyphen(lang='es_ES')

_word_re = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", re.UNICODE)

def _tokenize_words(text):
    return _word_re.findall(text)

def _count_sentences(text):
    # Split on ., !, ?, and Spanish ¡¿ — keep it simple
    parts = re.split(r"[.!?¡¿]+", text)
    return max(1, sum(1 for p in parts if p.strip()))

def _count_syllables_es(word):
    parts = _dic.hyphenate(word)
    return (len(parts) + 1) if parts else 1

def _text_stats_es(text):
    words = _tokenize_words(text)
    W = len(words)
    S = _count_sentences(text)
    syl = sum(_count_syllables_es(w) for w in words) if W else 0
    LW = sum(1 for w in words if len(w) > 6)  # LIX long words (>6 chars)
    return W, S, syl, LW

# --- Szigriszt–Pazos (INFLESZ) ---
def szigriszt_pazos(text):
    W, S, syl, _ = _text_stats_es(text)
    if W == 0 or S == 0:
        return None
    # Reading ease: higher = easier
    return 206.835 - 62.3 * (syl / W) - (W / S)

# --- LIX (language-agnostic) ---
def lix(text):
    W, S, _, LW = _text_stats_es(text)
    if W == 0 or S == 0:
        return None
    return (W / S) + (100.0 * LW / W)

# Example bands (tune to your corpus)
SZ_BANDS = {
    'B1': (65, 100),  # easy to very easy
    'B2': (55, 65),   # normal
    'B3': (40, 55),   # somewhat hard
}

LIX_BANDS = {
    'B1': (20, 35),   # easier
    'B2': (35, 45),   # mid
    'B3': (45, 60),   # harder
}

def in_band(score, band, bands, delta=0.0):
    if score is None:
        return False
    lo, hi = bands[band]
    return (lo - delta) <= score <= (hi + delta)

# Example usage
text = "Las vacunas salvan millones de vidas cada año. Son seguras y eficaces."
sz = szigriszt_pazos(text)
lx = lix(text)
# print("Szigriszt:", sz, "B1?", in_band(sz, 'B1', SZ_BANDS, delta=2))
# print("LIX:", lx, "B1?", in_band(lx, 'B1', LIX_BANDS, delta=2))