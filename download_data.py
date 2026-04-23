import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

np.random.seed(42)

diseases = {
    'Malaria':         (['fever','chills','sweating','headache','vomiting','fatigue','body ache','high temperature'], 'HIGH'),
    'Typhoid':         (['fever','headache','abdominal pain','loss of appetite','weakness','diarrhea','vomiting'], 'HIGH'),
    'Pneumonia':       (['cough','chest pain','difficulty breathing','fever','fatigue','chills'], 'HIGH'),
    'Meningitis':      (['stiff neck','fever','headache','vomiting','blurred vision','dizziness'], 'HIGH'),
    'Hepatitis B':     (['yellow eyes','fatigue','abdominal pain','nausea','loss of appetite','fever'], 'HIGH'),
    'Hypoglycemia':    (['dizziness','sweating','weakness','blurred vision','headache','fatigue'], 'HIGH'),
    'Hypertension':    (['headache','dizziness','chest pain','blurred vision','fatigue'], 'MEDIUM'),
    'Diabetes':        (['weight loss','fatigue','blurred vision','weakness','frequent urination','loss of appetite'], 'MEDIUM'),
    'Gastroenteritis': (['diarrhea','vomiting','abdominal pain','nausea','fever','weakness'], 'MEDIUM'),
    'Asthma':          (['cough','difficulty breathing','chest pain','fatigue','wheezing'], 'MEDIUM'),
    'UTI':             (['abdominal pain','fever','back pain','weakness','fatigue'], 'MEDIUM'),
    'Drug Reaction':   (['skin rash','itching','vomiting','fever','weakness'], 'MEDIUM'),
    'Common Cold':     (['runny nose','sore throat','cough','headache','fatigue'], 'LOW'),
    'Allergy':         (['skin rash','itching','runny nose','sneezing','watery eyes'], 'LOW'),
    'Migraine':        (['headache','vomiting','blurred vision','dizziness','sensitivity to light'], 'LOW'),
    'Fungal Infection':(['skin rash','itching','redness','scaling','discomfort'], 'LOW'),
    'Chicken Pox':     (['skin rash','itching','fever','fatigue','loss of appetite'], 'LOW'),
    'Acne':            (['skin rash','redness','itching','swelling'], 'LOW'),
}

rows = []
for disease, (symptoms, urgency) in diseases.items():
    for _ in range(80):
        n = np.random.randint(3, len(symptoms)+1)
        chosen = np.random.choice(symptoms, size=n, replace=False)
        symptom_cols = {f'Symptom_{i+1}': (chosen[i] if i < len(chosen) else '') for i in range(8)}
        rows.append({'disease': disease, 'urgency': urgency, **symptom_cols})

df = pd.DataFrame(rows)
df.to_csv('data/dataset.csv', index=False)
print(f'✅ Dataset generated: {len(df)} records, {df["disease"].nunique()} diseases')
print(df['urgency'].value_counts())