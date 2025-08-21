import os
import json
from openai import OpenAI
import tqdm
import re
from FH_es import fernandez_huerta
# Initialize client (ensure you have OPENAI_API_KEY in env vars)
client = OpenAI(api_key=json.load(open('/home/mshahidul/api.json', 'r'))['openai_api_key'])

PROMPTS_ES = {
    "B1": """Eres un asistente que reescribe resúmenes de casos clínicos para niñas y niños de primaria (aprox. 6–11 años).
Escribe SIEMPRE en español claro.

Objetivo de legibilidad (aprox. Fernández–Huerta): 70–100.
Restricciones de forma (cumple todas):
- Longitud total: 45–90 palabras.
- Oraciones: 4–6 oraciones.
- Promedio de palabras por oración: 8–12.
- Palabras: prefiere palabras cortas (1–2 sílabas). Evita tecnicismos. Si un término médico es inevitable, explícalo con 3–8 palabras sencillas.
- Conectores simples: “y”, “pero”, “porque”. Evita oraciones subordinadas largas.
- No inventes información. Sé fiel al artículo y al resumen experto.
- Prohibido: viñetas, listas, emojis, abreviaturas técnicas, explicaciones de pronunciación, títulos/cabeceras.

Tono y contenido:
- Amable, tranquilizador, sin alarmar.
- Destaca 1–3 ideas principales. Explica hallazgos normales con calma; anormalidades con lenguaje sencillo y breve.
Responde solo con el resumen (sin prefacios, sin notas).""",

    "B2": """Eres un asistente que reescribe resúmenes de casos clínicos para estudiantes de secundaria (aprox. 11–17 años).
Escribe SIEMPRE en español claro.

Objetivo de legibilidad (aprox. Fernández–Huerta): 55–65.
Restricciones de forma (cumple todas):
- Longitud total: 90–140 palabras.
- Oraciones: 5–8 oraciones.
- Promedio de palabras por oración: 12–18.
- Palabras: evita jerga innecesaria. Puedes usar términos médicos comunes con una breve explicación (3–10 palabras) la primera vez.
- Conectores permitidos: “porque”, “aunque”, “sin embargo”, “por eso”. Oraciones compuestas moderadas.
- No inventes información. Sé fiel al artículo y al resumen experto.
- Prohibido: viñetas, listas, emojis, explicaciones de pronunciación, títulos/cabeceras.

Tono y contenido:
- Claro y empático.
- Distingue hallazgos normales y anormales, e incluye posibles pasos siguientes cuando sea útil.
Responde solo con el resumen (sin prefacios, sin notas).""",

    "B3": """Eres un asistente que reescribe resúmenes de casos clínicos para lectores con nivel universitario (17+), sin especialización médica.
Escribe SIEMPRE en español claro.

Objetivo de legibilidad (aprox. Fernández–Huerta): 40–55.
Restricciones de forma (cumple todas):
- Longitud total: 140–220 palabras.
- Oraciones: 6–10 oraciones.
- Promedio de palabras por oración: 18–25.
- Palabras: se permiten términos técnicos de uso común; define brevemente solo los poco conocidos. Se aceptan oraciones subordinadas si mantienen claridad.
- Conectores: “sin embargo”, “por lo tanto”, “además”, “no obstante”, “en consecuencia”.
- No inventes información. Sé fiel al artículo y al resumen experto.
- Prohibido: viñetas, listas, emojis, explicaciones de pronunciación, títulos/cabeceras.

Tono y contenido:
- Preciso y empático.
- Estructura más detallada: contexto breve, hallazgos clave, implicaciones y posibles próximos pasos.
Responde solo con el resumen (sin prefacios, sin notas)."""
}


FH_TARGETS = {
    "B1": (70, 100),
    "B2": (55, 65),
    "B3": (40, 55),
}

def count_syllables(word):
    # Simple Spanish syllable counter
    word = word.lower()
    word = re.sub(r'[^a-záéíóúüñ]', '', word)
    return len(re.findall(r'[aeiouáéíóúü]+', word))



def generate_synthetic_summary(article, gold_summary, band, lang='es'):
    prompt_user = f"""Artículo:
{article}

Resumen experto:
{gold_summary}

Tarea:
Genera un resumen en la banda {band} indicada por el sistema. Responde solo con el resumen."""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # <-- Check this model name!
        messages=[
            {"role": "system", "content": PROMPTS_ES[band]},
            {"role": "user", "content": prompt_user}
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()

def revise_to_band(text, band):
    adjustments = {
        "B1": "Acorta oraciones a 8–12 palabras, usa palabras más comunes y evita tecnicismos.",
        "B2": "Ajusta oraciones a 12–18 palabras y limita tecnicismos con breve explicación.",
        "B3": "Usa 18–25 palabras por oración, permite frases subordinadas y vocabulario más técnico.",
    }
    msg = f"""Reescribe el texto para que cumpla la banda {band}:
- {adjustments[band]}
- Mantén fidelidad al contenido.
Devuelve solo el texto revisado, sin comentarios."""
    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": PROMPTS_ES[band]},
            {"role": "user", "content": text},
            {"role": "user", "content": msg}
        ],
        temperature=0.3,
    )
    return r.choices[0].message.content.strip()

def build_synthetic_dataset(input_path, output_path, max_samples=None):
    """Generate synthetic dataset from a JSON file with {fulltext, summary}"""
    results = []
    seen_articles = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            results = json.load(f)
            seen_articles = set(r['article'] for r in results)
    with open(input_path, "r") as f:
        data = json.load(f)
        for item in tqdm.tqdm(data):
            if max_samples and len(results) >= max_samples:
                break
            article, gold = item["fulltext"], item["summary"]
            if article in seen_articles:
                continue
            temp = {}
            for band in ["B1", "B2", "B3"]:
                synthetic = generate_synthetic_summary(article, gold, band)
                fh = fernandez_huerta(synthetic)
                lo, hi = FH_TARGETS[band]
                if fh is None or not (lo <= fh <= hi):
                    synthetic = revise_to_band(synthetic, band)
                temp[band] = synthetic
            results.append({
                "article": article,
                "gold_summary": gold,
                "synthetic_summary": temp
            })
            seen_articles.add(article)
            if len(results) % 5 == 0:
                print(f"Processed {len(results)} samples, saving progress...")
                with open(output_path, "w") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

# Example usage:
lang = "es"
path = f"/home/mshahidul/readctrl/data/testing_data_gs/multiclinsum_gs_train_{lang}.json"
build_synthetic_dataset(path, f"/home/mshahidul/readctrl/generating_data/{lang}_synthetic.json", max_samples=100)