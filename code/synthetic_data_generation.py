import os
import json
from openai import OpenAI
import tqdm
# Initialize client (ensure you have OPENAI_API_KEY in env vars)
client = OpenAI(api_key=json.load(open('/home/mshahidul/api.json', 'r'))['openai_api_key'])

# System prompts (from Appendix B in your proposal)
PROMPTS = {
    "B1": """You are a summarization assistant trained to rewrite medical case reports' expert summaries
for readers at an elementary school level (ages 5–11, FKGL 1.0–6.0).

Your job is to generate summaries that are:
* Kind and empathetic
* Clear, simple, and understandable for readers without medical background
* Accurate and faithful to the source

General Instructions:
- Assume the reader is an elementary school student with no medical knowledge.
- Avoid medical jargon. If it must appear, explain it in very simple terms.
- Use short sentences and everyday words.
- Reassure the reader when findings are normal; explain gently if something is abnormal.
- Do not overwhelm with detail; focus on main ideas.
- Never use emojis.
- Do not explain pronunciation.
""",
    "B2": """You are a summarization assistant trained to rewrite medical case reports' expert summaries for readers at a middle or high school level (ages 11–17, FKGL 6.0–12.0).

Your job is to generate summaries that are: 
* Kind and empathetic
* Clear and understandable for readers with only general school-level science
* Accurate and faithful to the source

General Instructions: 
- Assume the reader is a secondary school student with limited medical knowledge.
- Avoid unnecessary jargon. If a medical term is included, provide a brief, clear explanation.
- Write in a style appropriate for middle/high school reading comprehension.
- Present abnormal findings with calm, explanatory language, including possible next steps.
- Keep the tone warm, patient, and caring.
- Never use emojis.
- Do not explain pronunciation.
""",
    "B3": """You are a summarization assistant trained to rewrite medical case reports' expert summaries
for readers at a college or higher education level (ages 17+, FKGL 12.0+).

Your job is to generate summaries that are: 
* Kind and empathetic
* Clear and precise, while remaining faithful to the source
* Appropriate for readers with advanced literacy but no formal medical training

General Instructions:
- Assume the reader is a college-level reader with no medical specialization.
- Medical terms can be used if they are commonly understood or explained briefly.
- Provide a more detailed and structured summary than for younger readers.
- Clearly distinguish between normal and abnormal findings, and outline potential implications or next steps.
- Maintain an empathetic and respectful tone at all times.
- Never use emojis.
- Do not explain pronunciation.
"""
}

def generate_synthetic_summary(article, gold_summary, band):
    """Call GPT-5-mini to generate a synthetic summary for a given readability band"""
    prompt = f"""Article:
{article}

Gold Summary:
{gold_summary}

Task:
Please generate a summary at readability band {band}.
"""

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": PROMPTS[band]},
            {"role": "user", "content": prompt}
        ],
        temperature=1.0
    )

    return response.choices[0].message.content.strip()

def build_synthetic_dataset(input_path, output_path, max_samples=None):
    """Generate synthetic dataset from a JSONL file with {article, gold_summary}"""
    results = []
    if os.path.exists(output_path):
        results = json.load(open(output_path, 'r'))
    with open(input_path, "r") as f:
        data = json.load(f)
        for item in tqdm.tqdm(data):
            if max_samples and len(results) >= max_samples:
                break
            article, gold = item["fulltext"], item["summary"]
            if article in [r['article'] for r in results]:
                continue
            temp={}
            for band in ["B1", "B2", "B3"]:
                synthetic = generate_synthetic_summary(article, gold, band)
                temp[band] = synthetic
            results.append({
                    "article": article,
                    "gold_summary": gold,
                    "synthetic_summary": temp
                })
            if len(results)%5==0:
                print(f"Processed {len(results)} samples, saving progress...")
                with open(output_path, "w") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)

    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

# Example usage:
lang = "es"  # Change to desired language
path=f"/home/mshahidul/readctrl/data/testing_data_gs/multiclinsum_gs_train_{lang}.json"
build_synthetic_dataset(path, f"/home/mshahidul/readctrl/generating_data/{lang}_synthetic.json", max_samples=100)
