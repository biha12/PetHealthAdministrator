import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("pet_symptoms.csv")
df = df.dropna()
print(f"Total rows: {len(df)}")

# Get all unique symptoms
all_symptoms = set()
for row in df['symptoms']:
    for s in str(row).split():
        all_symptoms.add(s)
all_symptoms = sorted(list(all_symptoms))

# Encode pet_type as 0=Cat 1=Dog
pet_type_encoder = LabelEncoder()
df['pet_type_encoded'] = pet_type_encoder.fit_transform(df['pet_type'])

# Build feature vectors: pet_type + binary symptoms
def encode_row(pet_type_encoded, symptom_str):
    symptom_vec = [1 if s in symptom_str.split() else 0 for s in all_symptoms]
    return [pet_type_encoded] + symptom_vec

X = np.array([encode_row(row['pet_type_encoded'], row['symptoms']) for _, row in df.iterrows()])
y = df['condition'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Model Accuracy: {acc * 100:.2f}%")

with open("model.pkl", "wb") as f:
    pickle.dump((model, all_symptoms, pet_type_encoder), f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Model saved!")