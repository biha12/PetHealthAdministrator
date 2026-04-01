from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and data
with open("model.pkl", "rb") as f:
    model, all_symptoms = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

df = pd.read_csv("pet_symptoms.csv")

def encode_input(selected_symptoms):
    return [1 if s in selected_symptoms else 0 for s in all_symptoms]

@app.route("/")
def index():
    return render_template("index.html", symptoms=all_symptoms)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    selected = data.get("symptoms", [])

    if not selected:
        return jsonify({"error": "No symptoms selected"}), 400

    vector = np.array([encode_input(selected)])
    pred_encoded = model.predict(vector)[0]
    condition = le.inverse_transform([pred_encoded])[0]

    # Get probabilities for top 3
    proba = model.predict_proba(vector)[0]
    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [
        {"condition": le.inverse_transform([i])[0], "confidence": round(float(proba[i]) * 100, 1)}
        for i in top3_idx if proba[i] > 0
    ]

    # Find matching row in CSV
    match = df[df['condition'] == condition].iloc[0]

    return jsonify({
        "condition": condition,
        "urgency": match['urgency'],
        "advice": match['advice'],
        "home_care": match['home_care'],
        "top3": top3
    })

@app.route("/tips")
def tips():
    tips_data = [
        {"icon": "🥗", "title": "Balanced Diet", "tip": "Feed age-appropriate food. Avoid human junk food, chocolate, onions, and grapes — these are toxic to pets."},
        {"icon": "💧", "title": "Fresh Water Daily", "tip": "Always provide clean, fresh water. Dehydration causes kidney failure in cats especially."},
        {"icon": "🏃", "title": "Daily Exercise", "tip": "Dogs need 30–60 min of walks daily. Cats need play sessions. Obesity shortens your pet's life."},
        {"icon": "🦷", "title": "Dental Hygiene", "tip": "Brush teeth 2–3x per week. Dental disease affects 80% of pets over 3 years old."},
        {"icon": "💉", "title": "Vaccinations", "tip": "Keep up with annual vaccines for rabies, distemper, and parvo. Kittens need 3-shot series."},
        {"icon": "🪱", "title": "Deworming", "tip": "Deworm every 3 months. Use vet-prescribed medication, not over-the-counter alternatives."},
        {"icon": "🐾", "title": "Nail Trimming", "tip": "Trim nails every 3–4 weeks. Overgrown nails cause pain and posture problems."},
        {"icon": "🛁", "title": "Grooming", "tip": "Brush your pet weekly to remove dead hair and check for fleas, ticks, or skin issues."},
        {"icon": "🏠", "title": "Safe Environment", "tip": "Keep toxic plants, medicines, and cleaning chemicals out of reach. Check floors for small swallowable objects."},
        {"icon": "❤️", "title": "Mental Wellness", "tip": "Spend quality time daily. Pets suffer from depression and anxiety if neglected or isolated."},
    ]
    return jsonify(tips_data)

if __name__ == "__main__":
    app.run(debug=True)