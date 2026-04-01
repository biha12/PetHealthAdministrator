import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load data
df = pd.read_csv("pet_symptoms.csv")

# Collect all unique symptoms
all_symptoms = set()
for row in df['symptoms']:
    for s in row.split():
        all_symptoms.add(s)

all_symptoms = sorted(list(all_symptoms))

# Build feature matrix (binary symptom vectors)
def encode_symptoms(symptom_str, all_symptoms):
    present = set(symptom_str.split())
    return [1 if s in present else 0 for s in all_symptoms]

X = np.array([encode_symptoms(row, all_symptoms) for row in df['symptoms']])
y = df['condition'].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)

# Save everything
with open("model.pkl", "wb") as f:
    pickle.dump((model, all_symptoms), f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Model trained and saved!")
print(f"   Symptoms tracked: {len(all_symptoms)}")
print(f"   Conditions learned: {len(le.classes_)}")