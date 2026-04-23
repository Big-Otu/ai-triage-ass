import pandas as pd
import numpy as np
import re

# Simple English stopwords (no NLTK needed)
STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more','most',
    'other','some','such','no','nor','not','only','own','same','so','than','too',
    'very','s','t','can','will','just','don','should','now','d','ll','m','o',
    're','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn','haven',
    'isn','ma','mightn','mustn','needn','shan','shouldn','wasn','weren','won','wouldn'
}

# ─── Urgency mapping based on disease severity ───────────────────────────────
# Ghana-clinic relevant urgency levels: HIGH, MEDIUM, LOW
URGENCY_MAP = {
    # HIGH urgency — needs immediate attention
    "heart attack": "HIGH",
    "malaria": "HIGH",
    "typhoid": "HIGH",
    "meningitis": "HIGH",
    "pneumonia": "HIGH",
    "dengue": "HIGH",
    "tuberculosis": "HIGH",
    "hepatitis b": "HIGH",
    "hepatitis c": "HIGH",
    "hepatitis d": "HIGH",
    "paralysis (brain hemorrhage)": "HIGH",
    "brain hemorrhage": "HIGH",
    "alcoholic hepatitis": "HIGH",
    "jaundice": "HIGH",
    "chronic cholestasis": "HIGH",
    "diabetes": "MEDIUM",
    "hypertension": "MEDIUM",
    "peptic ulcer disease": "MEDIUM",
    "gastroenteritis": "MEDIUM",
    "bronchial asthma": "MEDIUM",
    "urinary tract infection": "MEDIUM",
    "psoriasis": "LOW",
    "impetigo": "LOW",
    "chicken pox": "LOW",
    "fungal infection": "LOW",
    "allergy": "LOW",
    "common cold": "LOW",
    "migraine": "LOW",
    "cervical spondylosis": "LOW",
    "hypothyroidism": "MEDIUM",
    "hyperthyroidism": "MEDIUM",
    "hypoglycemia": "HIGH",
    "osteoarthritis": "LOW",
    "arthritis": "LOW",
    "acne": "LOW",
    "drug reaction": "MEDIUM",
    "dimorphic hemorrhoids (piles)": "LOW",
    "varicose veins": "LOW",
    "aids": "HIGH",
    "hepatitis e": "HIGH",
    "hepatitis a": "HIGH",
}

URGENCY_COLORS = {
    "HIGH": "#e74c3c",
    "MEDIUM": "#f39c12",
    "LOW": "#27ae60"
}

URGENCY_ADVICE = {
    "HIGH": "⚠️ This patient needs IMMEDIATE medical attention. Do not delay.",
    "MEDIUM": "🕐 This patient should see a doctor within the next hour.",
    "LOW": "✅ Condition appears non-critical. Patient can wait for routine consultation."
}

def clean_symptom_text(text: str) -> str:
    """Lowercase, remove special chars, strip stopwords."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 2]
    return ' '.join(tokens)


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load dataset, assign urgency labels, clean symptom text."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Combine all symptom columns into one text field
    symptom_cols = [c for c in df.columns if 'symptom' in c]
    df['symptom_text'] = df[symptom_cols].apply(
        lambda row: ' '.join([str(v).replace('_', ' ') for v in row if str(v).strip() not in ['nan', '']]),
        axis=1
    )

    # Normalize disease name
    df['disease'] = df['disease'].str.strip().str.lower()

    # Assign urgency
    df['urgency'] = df['disease'].map(
        lambda d: URGENCY_MAP.get(d, "MEDIUM")  # default MEDIUM if unknown
    )

    # Clean symptom text
    df['clean_symptoms'] = df['symptom_text'].apply(clean_symptom_text)

    return df[['disease', 'symptom_text', 'clean_symptoms', 'urgency']]


if __name__ == "__main__":
    df = load_and_prepare_data("data/dataset.csv")
    print(df.head())
    print("\nUrgency distribution:")
    print(df['urgency'].value_counts())
    df.to_csv("data/processed_dataset.csv", index=False)
    print("\n✅ Saved to data/processed_dataset.csv")
