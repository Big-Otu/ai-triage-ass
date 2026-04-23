import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import load_and_prepare_data, clean_symptom_text

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_PATH   = "data/dataset.csv"
MODEL_DIR   = "model"
MODEL_PATH  = os.path.join(MODEL_DIR, "triage_model.pkl")
DISEASE_MODEL_PATH = os.path.join(MODEL_DIR, "disease_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


def train_urgency_model(df: pd.DataFrame):
    """Train a model to predict HIGH / MEDIUM / LOW urgency."""
    X = df['clean_symptoms']
    y = df['urgency']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("=" * 50)
    print("URGENCY MODEL PERFORMANCE")
    print("=" * 50)
    print(classification_report(y_test, y_pred))

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

    # Save model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n✅ Urgency model saved to {MODEL_PATH}")
    return pipeline


def train_disease_model(df: pd.DataFrame):
    """Train a model to predict the likely disease/condition."""
    X = df['clean_symptoms']
    y = df['disease']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42, C=5))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print("=" * 50)
    print(f"DISEASE MODEL ACCURACY: {accuracy:.2f}")
    print("=" * 50)

    # Save model
    joblib.dump(pipeline, DISEASE_MODEL_PATH)
    print(f"✅ Disease model saved to {DISEASE_MODEL_PATH}")
    return pipeline


def predict(symptoms_text: str):
    """Run inference on raw symptom text input."""
    urgency_model  = joblib.load(MODEL_PATH)
    disease_model  = joblib.load(DISEASE_MODEL_PATH)

    cleaned = clean_symptom_text(symptoms_text)

    urgency  = urgency_model.predict([cleaned])[0]
    disease  = disease_model.predict([cleaned])[0]

    urgency_proba = urgency_model.predict_proba([cleaned])[0]
    urgency_classes = urgency_model.classes_
    confidence = dict(zip(urgency_classes, urgency_proba))

    return {
        "urgency": urgency,
        "likely_condition": disease.title(),
        "confidence": confidence
    }


if __name__ == "__main__":
    print("📂 Loading and preparing data...")
    df = load_and_prepare_data(DATA_PATH)
    print(f"✅ Loaded {len(df)} records with {df['disease'].nunique()} unique conditions\n")

    print("🤖 Training urgency model...")
    train_urgency_model(df)

    print("\n🤖 Training disease prediction model...")
    train_disease_model(df)

    print("\n🧪 Quick test prediction:")
    test_input = "fever headache vomiting chills sweating fatigue"
    result = predict(test_input)
    print(f"Symptoms: {test_input}")
    print(f"Urgency:  {result['urgency']}")
    print(f"Likely Condition: {result['likely_condition']}")
    print(f"Confidence: { {k: f'{v:.0%}' for k, v in result['confidence'].items()} }")
