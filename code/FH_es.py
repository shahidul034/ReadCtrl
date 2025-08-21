import re

# --- Spanish tokenization ---
WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", re.UNICODE)

def _tokenize_words_es(text: str):
    return WORD_RE.findall(text)

def _count_sentences_es(text: str) -> int:
    # Count sentences via ., !, ?, … and Spanish ¡¿
    sentences = re.split(r"[.!?…]+|[¡¿]", text)
    return max(1, sum(1 for s in sentences if s.strip()))

# --- Syllable counting ---
try:
    import pyphen
    _dic = pyphen.Pyphen(lang='es')  # or 'es_ES'

    def count_syllables_es(word: str) -> int:
        # Use hyphenation positions; count pieces
        hyph = _dic.inserted(word)
        return max(1, hyph.count('-') + 1)
except Exception:
    # Heuristic fallback (handles hiatus and silent 'u' roughly)
    def count_syllables_es(word: str) -> int:
        w = word.lower()

        # Treat final 'y' as vowel 'i'
        w = re.sub(r'y$', 'i', w)

        # Remove silent 'u' before e/i in 'que/qui/gue/gui' (but not 'güe/güi')
        w = re.sub(r'que', 'qe', w)
        w = re.sub(r'qui', 'qi', w)
        w = re.sub(r'gue', 'ge', w)
        w = re.sub(r'gui', 'gi', w)

        vowels = set("aeiouáéíóúü")
        strong = set("aáeéoóíú")  # accented í/ú behave like strong (hiatus)
        n = len(w)
        i = 0
        syll = 0
        while i < n:
            if w[i] not in vowels:
                i += 1
                continue
            # collect contiguous vowels
            j = i + 1
            while j < n and w[j] in vowels:
                j += 1
            seq = w[i:j]
            # one nucleus by default
            nuclei = 1
            # split on strong-strong boundaries (ae, ea, ao, oa, eo, oe, and cases with í/ú)
            for k in range(len(seq) - 1):
                if seq[k] in strong and seq[k + 1] in strong:
                    nuclei += 1
            syll += nuclei
            i = j
        return max(1, syll)

# --- Fernández–Huerta (FH) ---
def fernandez_huerta(text: str) -> float | None:
    """
    Fernández–Huerta readability for Spanish.
    Higher = easier. Typical range ~0–100.
    """
    words = _tokenize_words_es(text)
    n_words = len(words)
    if n_words == 0:
        return None
    n_sentences = _count_sentences_es(text)
    n_syllables = sum(count_syllables_es(w) for w in words)

    # FH = 206.84 - 0.60 * (P) - 1.02 * (F)
    # P = (syllables/words)*100, F = words/sentence
    fh = 206.84 - 0.60 * ((n_syllables / n_words) * 100.0) - 1.02 * (n_words / n_sentences)
    return round(fh, 2)

# --- Quick check ---
# if __name__ == "__main__":
#     text_easy = "El corazón es un órgano que bombea sangre. En este caso, funciona bien."
#     text_medium = "El corazón del paciente muestra una función adecuada, aunque se observaron pequeños cambios que deben revisarse."
#     text_hard = "La evaluación cardiológica indicó una función sistólica preservada, con alteraciones discretas en la relajación diastólica."
#     print("Easy FH:", fernandez_huerta(text_easy))
#     print("Medium FH:", fernandez_huerta(text_medium))
#     print("Hard FH:", fernandez_huerta(text_hard))