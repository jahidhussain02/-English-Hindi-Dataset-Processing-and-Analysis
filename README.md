# -English-Hindi-Dataset-Processing-and-Analysis
 import pandas as pd
from transformers import pipeline
# Load Translation Model

translator = pipeline(
    task="translation",
    model="Helsinki-NLP/opus-mt-en-hi"
)

# Load Cleaned Dataset

df = pd.read_excel("../Assignment1/cleaned_dataset.xlsx")

# Take first 100 sentences
sample = df.head(100).copy()

translations = []

print("Translating...")

for sentence in sample["English Sentences"]:

    try:
        translated = translator(
            sentence,
            max_length=512
        )[0]["translation_text"]

    except Exception as e:
        translated = ""

    translations.append(translated)

sample["Model-generated Hindi translation"] = translations

sample.to_excel(
    "translations.xlsx",
    index=False
)

print("Translation completed.")
print("Saved as translations.xlsx")
import pandas as pd
from sacrebleu.metrics import BLEU
from sacrebleu.metrics import CHRF
from sacrebleu.metrics import TER

# Load Translations

df = pd.read_excel("translations.xlsx")

references = df["Hindi Sentences"].tolist()
hypotheses = df["Model-generated Hindi translation"].tolist()

# Metrics

bleu = BLEU()
chrf = CHRF()
ter = TER()

bleu_score = bleu.corpus_score(
    hypotheses,
    [references]
)

chrf_score = chrf.corpus_score(
    hypotheses,
    [references]
)

ter_score = ter.corpus_score(
    hypotheses,
    [references]
)

# Save Results

with open("scores.txt", "w", encoding="utf-8") as file:

    file.write("Translation Evaluation Metrics\n")
    file.write("=" * 40 + "\n\n")

    file.write(f"BLEU Score : {bleu_score.score:.2f}\n")
    file.write(f"CHRF Score : {chrf_score.score:.2f}\n")
    file.write(f"TER Score  : {ter_score.score:.2f}\n")

print("Scores saved in scores.txt")
pandas
openpyxl
transformers
torch
sentencepiece
sacrebleu
