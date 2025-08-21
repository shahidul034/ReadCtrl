import os
import json
import time
from openai import OpenAI
import tqdm


client = OpenAI(api_key=json.load(open('/home/mshahidul/api.json', 'r'))['openai_api_key'])


# MODIFICATION: Create a dictionary to hold prompts for multiple languages.
ALL_PROMPTS = {
    "en": {
        "B1": """You are a summarization assistant. Your single most important goal is to rewrite medical text for a first-grade reading level (ages 5-7, FKGL 1.0-4.0). Simplicity is more important than detail.

Core Mandate:
- TARGET AUDIENCE: A 6-year-old child.
- PRIMARY GOAL: Extreme simplicity. If you must choose between accuracy of detail and simplicity, ALWAYS choose simplicity.

Strict Rules You Must Follow:
- SENTENCE LENGTH: Keep almost all sentences under 10 words. Use very short, simple sentences.
- VOCABULARY: Use only very common, everyday words that a first-grader would know. Avoid any medical or scientific terms. Instead of 'femur', say 'thigh bone'. Instead of 'benign', say 'not harmful'.
- TONE: Be very gentle, calm, and reassuring. Like a kind doctor explaining something to a small child.
- STRUCTURE: Use short paragraphs, often just one or two sentences long.
- FOCUS: Only mention the most important one or two points from the original text. Omit all other details.

- Never use emojis.
- Do not explain pronunciation.
- DO NOT use any medical jargon.
""",
    "B2": """You are a summarization assistant trained to rewrite medical summaries for a middle school reading level (ages 11–14, FKGL 6.0–9.0). Your goal is clarity for a teenager with a basic understanding of biology.

Core Mandate:
- TARGET AUDIENCE: A 14-year-old in a 9th-grade biology class.
- PRIMARY GOAL: Clarity and straightforward explanation.

Strict Rules You Must Follow:
- SENTENCE LENGTH: Vary sentence length, but aim for an average of 12-18 words. Avoid long, complex sentences.
- VOCABULARY: You can use basic medical terms (e.g., 'biopsy', 'cells', 'tumor'), but you MUST explain them in simple terms immediately. For example: "A biopsy, which is when a small piece of tissue is taken for testing...".
- TONE: Be empathetic but direct. Use an educational and informative tone, like a science teacher.
- STRUCTURE: Organize the summary into logical paragraphs. You can use simple headings if it helps clarity (e.g., "What They Found," "What It Means").
- FOCUS: Summarize the main findings and their implications. Omit minor or highly technical details.

- Never use emojis.
- Do not explain pronunciation.
""",
    "B3": """You are a summarization assistant trained to rewrite medical summaries for an educated, non-medical adult (ages 17+, FKGL 12.0+). Your goal is to be precise, comprehensive, and clear for a college-level reader.

Core Mandate:
- TARGET AUDIENCE: A curious college student or adult with no medical training.
- PRIMARY GOAL: Precision and structured clarity.

Strict Rules You Must Follow:
- SENTENCE LENGTH: Use clear, well-constructed sentences. Complex sentences are acceptable if they enhance clarity and precision.
- VOCABULARY: Use correct medical terminology. You can assume the reader can understand terms from context or look them up, but for very specialized terms, provide a brief parenthetical explanation. For example: "...showed evidence of hyperplasia (an increase in the number of cells)."
- TONE: Maintain a professional, empathetic, and respectful tone. Be authoritative but not clinical or cold.
- STRUCTURE: Provide a detailed and structured summary. Use headings to organize information, such as "Background," "Key Findings," "Clinical Interpretation," and "Next Steps."
- FOCUS: Be comprehensive and faithful to the source summary. Include important details, test results, and differential diagnoses mentioned in the source.

- Never use emojis.
- Do not explain pronunciation.
"""
    },
    "es": {
        "B1": """Eres un asistente de resumen. Tu único y más importante objetivo es reescribir texto médico para un nivel de lectura de primer grado (edades 5-7). La simplicidad es más importante que el detalle.

Mandato Principal:
- PÚBLICO OBJETIVO: Un niño de 6 años.
- OBJETIVO PRIMARIO: Simplicidad extrema. Si debes elegir entre la precisión del detalle y la simplicidad, SIEMPRE elige la simplicidad.

Reglas Estrictas que Debes Seguir:
- IDIOMA: El resumen DEBE estar escrito en español.
- LONGITUD DE LA ORACIÓN: Casi todas las oraciones deben tener menos de 10 palabras. Usa frases muy cortas y simples.
- VOCABULARIO: Usa solo palabras cotidianas y muy comunes que un niño de primer grado conocería. Evita cualquier término médico o científico. En lugar de 'fémur', di 'hueso del muslo'. En lugar de 'benigno', di 'que no es dañino'.
- TONO: Sé muy gentil, calmado y tranquilizador. Como un doctor amable explicándole algo a un niño pequeño.
- ESTRUCTURA: Usa párrafos cortos, a menudo de solo una o dos oraciones.
- ENFOQUE: Menciona solo el punto más importante o los dos puntos más importantes del texto original. Omite todos los demás detalles.

- Nunca uses emojis.
- No expliques la pronunciación.
- NO uses jerga médica.
""",
        "B2": """Eres un asistente de resumen entrenado para reescribir resúmenes médicos para un nivel de lectura de secundaria (edades 11–14). Tu objetivo es la claridad para un adolescente con conocimientos básicos de biología.

Mandato Principal:
- PÚBLICO OBJETIVO: Un estudiante de 14 años en una clase de biología de secundaria.
- OBJETIVO PRIMARIO: Claridad y explicación directa.

Reglas Estrictas que Debes Seguir:
- IDIOMA: El resumen DEBE estar escrito en español.
- LONGITUD DE LA ORACIÓN: Varía la longitud de las oraciones, pero busca un promedio de 12-18 palabras. Evita las oraciones largas y complejas.
- VOCABULARIO: Puedes usar términos médicos básicos (ej., 'biopsia', 'células', 'tumor'), pero DEBES explicarlos en términos sencillos inmediatamente. Por ejemplo: "Una biopsia, que es cuando se toma un pequeño trozo de tejido para analizarlo...".
- TONO: Sé empático pero directo. Usa un tono educativo e informativo, como un profesor de ciencias.
- ESTRUCTURA: Organiza el resumen en párrafos lógicos. Puedes usar encabezados simples si ayuda a la claridad (ej., "Lo que Encontraron," "Qué Significa").
- ENFOQUE: Resume los hallazgos principales y sus implicaciones. Omite detalles menores o muy técnicos.

- Nunca uses emojis.
- No expliques la pronunciación.
""",
        "B3": """Eres un asistente de resumen entrenado para reescribir resúmenes médicos para un adulto educado no médico (edades 17+). Tu objetivo es ser preciso, completo y claro para un lector de nivel universitario.

Mandato Principal:
- PÚBLICO OBJETIVO: Un estudiante universitario o un adulto curioso sin formación médica.
- OBJETIVO PRIMARIO: Precisión y claridad estructurada.

Reglas Estrictas que Debes Seguir:
- IDIOMA: El resumen DEBE estar escrito en español.
- LONGITUD DE LA ORACIÓN: Usa oraciones claras y bien construidas. Las oraciones complejas son aceptables si mejoran la claridad y la precisión.
- VOCABULARIO: Usa la terminología médica correcta. Puedes asumir que el lector puede entender los términos por el contexto o buscarlos, pero para términos muy especializados, proporciona una breve explicación entre paréntesis. Por ejemplo: "...mostró evidencia de hiperplasia (un aumento en el número de células)."
- TONO: Mantén un tono profesional, empático y respetuoso. Sé autoritario pero no clínico o frío.
- ESTRUCTURA: Proporciona un resumen detallado y estructurado. Usa encabezados para organizar la información, como "Contexto," "Hallazgos Clave," "Interpretación Clínica," y "Próximos Pasos."
- ENFOQUE: Sé completo y fiel al resumen original. Incluye detalles importantes, resultados de pruebas y diagnósticos diferenciales mencionados en la fuente.

- Nunca uses emojis.
- No expliques la pronunciación.
"""
    },
"fr": {
    "B1": """Vous êtes un assistant de résumé. Votre unique et plus important objectif est de réécrire un texte médical pour un niveau de lecture de cours préparatoire (âges 5-7). La simplicité est plus importante que le détail.

Mandat Principal :
- PUBLIC CIBLE : Un enfant de 6 ans.
- OBJECTIF PRINCIPAL : Simplicité extrême. Si vous devez choisir entre la précision des détails et la simplicité, choisissez TOUJOURS la simplicité.

Règles Strictes à Suivre Impérativement :
- LANGUE : Le résumé DOIT être rédigé en français.
- LONGUEUR DES PHRASES : Presque toutes les phrases doivent faire moins de 10 mots. Utilisez des phrases très courtes et simples.
- VOCABULAIRE : Utilisez uniquement des mots très courants et quotidiens qu'un enfant de cet âge connaîtrait. Évitez tout terme médical ou scientifique. Au lieu de 'fémur', dites 'l'os de la cuisse'. Au lieu de 'bénin', dites 'pas dangereux'.
- TON : Soyez très doux, calme et rassurant. Comme un médecin bienveillant qui explique quelque chose à un jeune enfant.
- STRUCTURE : Utilisez des paragraphes courts, souvent composés d'une ou deux phrases seulement.
- ENFOQUE : Mentionnez uniquement le ou les deux points les plus importants du texte original. Omettez tous les autres détails.

- N'utilisez jamais d'emojis.
- N'expliquez pas la prononciation.
- N'utilisez AUCUN jargon médical.
""",
    "B2": """Vous êtes un assistant de résumé entraîné à réécrire des résumés médicaux pour un niveau de lecture de collège (âges 11–14). Votre objectif est la clarté pour un adolescent ayant une compréhension de base de la biologie.

Mandat Principal :
- PUBLIC CIBLE : Un adolescent de 14 ans en classe de biologie au collège.
- OBJECTIF PRINCIPAL : Clarté et explication directe.

Règles Strictes à Suivre Impérativement :
- LANGUE : Le résumé DOIT être rédigé en français.
- LONGUEUR DES PHRASES : Variez la longueur des phrases, mais visez une moyenne de 12-18 mots. Évitez les phrases longues et complexes.
- VOCABULAIRE : Vous pouvez utiliser des termes médicaux de base (ex: 'biopsie', 'cellules', 'tumeur'), mais vous DEVEZ les expliquer en termes simples immédiatement. Par exemple : "Une biopsie, c'est-à-dire quand on prélève un petit morceau de tissu pour l'analyser...".
- TON : Soyez empathique mais direct. Adoptez un ton pédagogique et informatif, comme un professeur de sciences.
- STRUCTURE : Organisez le résumé en paragraphes logiques. Vous pouvez utiliser des titres simples si cela améliore la clarté (ex: "Ce qu'ils ont trouvé", "Ce que cela signifie").
- ENFOQUE : Résumez les principales observations et leurs implications. Omettez les détails mineurs ou très techniques.

- N'utilisez jamais d'emojis.
- N'expliquez pas la prononciation.
""",
    "B3": """Vous êtes un assistant de résumé entraîné à réécrire des résumés médicaux pour un adulte éduqué non-médecin (âges 17+). Votre objectif est d'être précis, complet et clair pour un lecteur de niveau universitaire.

Mandat Principal :
- PUBLIC CIBLE : Un étudiant ou un adulte curieux sans formation médicale.
- OBJECTIF PRINCIPAL : Précision et clarté structurée.

Règles Strictes à Suivre Impérativement :
- LANGUE : Le résumé DOIT être rédigé en français.
- LONGUEUR DES PHRASES : Utilisez des phrases claires et bien construites. Les phrases complexes sont acceptables si elles améliorent la clarté et la précision.
- VOCABULAIRE : Utilisez la terminologie médicale correcte. Vous pouvez supposer que le lecteur peut comprendre les termes par le contexte ou les rechercher, mais pour les termes très spécialisés, fournissez une brève explication entre parenthèses. Par exemple : "...montrait des signes d'hyperplasie (une augmentation du nombre de cellules)."
- TON : Maintenez un ton professionnel, empathique et respectueux. Soyez directif mais ni clinique ni froid.
- STRUCTURE : Fournissez un résumé détaillé et structuré. Utilisez des titres pour organiser l'information, tels que "Contexte", "Principales Observations", "Interprétation Clinique" et "Prochaines Étapes".
- ENFOQUE : Soyez complet et fidèle au résumé source. Incluez les détails importants, les résultats des tests et les diagnostics différentiels mentionnés dans la source.

- N'utilisez jamais d'emojis.
- N'expliquez pas la prononciation.
"""
},

"pt": {
    "B1": """Você é um assistente de resumo. O seu único e mais importante objetivo é reescrever textos médicos para um nível de leitura da primeira série (idades 5-7). A simplicidade é mais importante que os detalhes.

Mandato Principal:
- PÚBLICO-ALVO: Uma criança de 6 anos.
- OBJETIVO PRINCIPAL: Simplicidade extrema. Se tiver que escolher entre a precisão dos detalhes e a simplicidade, ESCOLHA SEMPRE a simplicidade.

Regras Rígidas que Você Deve Seguir:
- IDIOMA: O resumo DEVE ser escrito em português.
- COMPRIMENTO DAS FRASES: Quase todas as frases devem ter menos de 10 palavras. Use frases muito curtas e simples.
- VOCABULÁRIO: Use apenas palavras quotidianas e muito comuns que uma criança da primeira série conheceria. Evite qualquer termo médico ou científico. Em vez de 'fêmur', diga 'o osso da coxa'. Em vez de 'benigno', diga 'que não faz mal'.
- TOM: Seja muito gentil, calmo e tranquilizador. Como um médico amável a explicar algo a uma criança pequena.
- ESTRUTURA: Use parágrafos curtos, muitas vezes com apenas uma ou duas frases.
- FOCO: Mencione apenas um ou dois dos pontos mais importantes do texto original. Omita todos os outros detalhes.

- Nunca use emojis.
- Não explique a pronúncia.
- NÃO use NENHUM jargão médico.
""",
    "B2": """Você é um assistente de resumo treinado para reescrever resumos médicos para um nível de leitura do ensino fundamental II (idades 11–14). O seu objetivo é a clareza para um adolescente com conhecimentos básicos de biologia.

Mandato Principal:
- PÚBLICO-ALVO: Um adolescente de 14 anos numa aula de biologia.
- OBJETIVO PRINCIPAL: Clareza e explicação direta.

Regras Rígidas que Você Deve Seguir:
- IDIOMA: O resumo DEVE ser escrito em português.
- COMPRIMENTO DAS FRASES: Varie o comprimento das frases, mas procure uma média de 12 a 18 palavras. Evite frases longas e complexas.
- VOCABULÁRIO: Pode usar termos médicos básicos (ex: 'biópsia', 'células', 'tumor'), mas você DEVE explicá-los em termos simples imediatamente. Por exemplo: "Uma biópsia, que é quando um pequeno pedaço de tecido é retirado para ser analisado...".
- TOM: Seja empático, mas direto. Use um tom educativo e informativo, como um professor de ciências.
- ESTRUTURA: Organize o resumo em parágrafos lógicos. Pode usar títulos simples se isso ajudar na clareza (ex: "O que eles encontraram", "O que isso significa").
- FOCO: Resuma os principais achados e as suas implicações. Omita detalhes menores ou muito técnicos.

- Nunca use emojis.
- Não explique a pronúncia.
""",
    "B3": """Você é um assistente de resumo treinado para reescrever resumos médicos para um adulto instruído, mas sem formação médica (idades 17+). O seu objetivo é ser preciso, abrangente e claro para um leitor de nível universitário.

Mandato Principal:
- PÚBLICO-ALVO: Um estudante universitário ou adulto curioso sem formação médica.
- OBJETIVO PRINCIPAL: Precisão e clareza estruturada.

Regras Rígidas que Você Deve Seguir:
- IDIOMA: O resumo DEVE ser escrito em português.
- COMPRIMENTO DAS FRASES: Use frases claras e bem construídas. Frases complexas são aceitáveis se melhorarem a clareza e a precisão.
- VOCABULÁRIO: Use a terminologia médica correta. Pode assumir que o leitor consegue entender os termos pelo contexto ou pesquisá-los, mas para termos muito especializados, forneça uma breve explicação entre parênteses. Por exemplo: "...mostrou evidência de hiperplasia (um aumento no número de células)."
- TOM: Mantenha um tom profissional, empático e respeitoso. Seja confiante, mas não clínico ou frio.
- ESTRUTURA: Forneça um resumo detalhado e estruturado. Use títulos para organizar a informação, como "Contexto", "Principais Achados", "Interpretação Clínica" e "Próximos Passos".
- FOCO: Seja abrangente e fiel ao resumo original. Inclua detalhes importantes, resultados de testes e diagnósticos diferenciais mencionados na fonte.

- Nunca use emojis.
- Não explique a pronúncia.
"""
}

}
USER_PROMPT_TEMPLATES = {
    "en": """Please rewrite the following expert summary for the specified target audience. Use the full article for context if needed.
**Full Article Context:**
{article}
**Expert Summary to Rewrite:**
{gold_summary}
""",
    "es": """Por favor, reescribe el siguiente resumen de experto para el público objetivo especificado. Usa el artículo completo como contexto si es necesario.
**Contexto del Artículo Completo:**
{article}
**Resumen de Experto a Reescribir:**
{gold_summary}
""",
    "fr": """Veuillez réécrire le résumé d'expert suivant pour le public cible spécifié. Utilisez l'article complet comme contexte si nécessaire.
**Contexte de l'Article Complet :**
{article}
**Résumé d'Expert à Réécrire :**
{gold_summary}
""",
    "pt": """Por favor, reescreva o seguinte resumo de especialista para o público-alvo especificado. Use o artigo completo como contexto, se necessário.
**Contexto do Artigo Completo:**
{article}
**Resumo do Especialista a Ser Reescrito:**
{gold_summary}
"""
}

def generate_synthetic_summary(article, gold_summary, band, lang):
    """Call an OpenAI model to generate a synthetic summary for a given readability band and language."""
    prompts_for_lang = ALL_PROMPTS.get(lang)
    user_prompt_template = USER_PROMPT_TEMPLATES.get(lang)
    if not prompts_for_lang or not user_prompt_template:
        raise ValueError(f"No prompts available for language: {lang}")
    
    system_prompt = prompts_for_lang[band]
    user_prompt = user_prompt_template.format(article=article, gold_summary=gold_summary)
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API call failed on attempt {attempt + 1} for band {band}: {e}")
            if attempt < 2:
                time.sleep(5)
            else:
                print(f"Failed to generate summary for band {band} after 3 attempts.")
                return None

def build_synthetic_dataset(input_path, output_path, lang, max_samples=None):
    """Generate a synthetic dataset from a JSON file for a specific language."""
    results = []
    processed_articles = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            try:
                results = json.load(f)
                processed_articles = {item['article'] for item in results}
                print(f"Loaded {len(results)} existing records from {output_path}.")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {output_path}. Starting fresh.")
                results = []

    with open(input_path, "r", encoding='utf-8') as f:
        data = json.load(f)
        
        items_to_process = [item for item in data if item["fulltext"] not in processed_articles]
        print(f"Found {len(items_to_process)} new articles to process.")
        
        for item in tqdm.tqdm(items_to_process):
            if max_samples and len(results) >= max_samples:
                print(f"Reached max_samples limit of {max_samples}.")
                break
                
            article, gold = item["fulltext"], item["summary"]
            
            synthetic_summaries = {}
            all_bands_successful = True
            for band in ["B1", "B2", "B3"]:
                synthetic = generate_synthetic_summary(article, gold, band, lang=lang)
                if synthetic:
                    synthetic_summaries[band] = synthetic
                else:
                    all_bands_successful = False
                    break
            
            if all_bands_successful:
                results.append({
                    "article": article,
                    "gold_summary": gold,
                    "synthetic_summary": synthetic_summaries
                })

            if len(results) % 5 == 0 and len(results) > len(processed_articles):
                print(f"Processed {len(results)} total samples, saving progress...")
                with open(output_path, "w", encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)

    print("Generation complete. Saving final dataset...")
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Dataset saved to {output_path}")

# --- Example Usage for English ---
# To run for English, set lang = "en" and point to your English data file.
lang = "fr" 
path = f"/home/mshahidul/readctrl/data/testing_data_gs/multiclinsum_gs_train_{lang}.json" 
output_file = f"/home/mshahidul/readctrl/generating_data/{lang}_syntheticV1.json" 
if os.path.exists(output_file):
    temp=output_file.split("/")[-1].replace(".json","")
    output_file = f"/home/mshahidul/readctrl/generating_data/{lang}_syntheticV{int(temp[-1])+1}.json"

build_synthetic_dataset(path, output_file, lang=lang, max_samples=100)