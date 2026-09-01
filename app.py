from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model, all_symptoms, pet_type_encoder = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

df = pd.read_csv("pet_symptoms.csv")

# Cat and Dog breeds
BREEDS = {
    "Cat": ["Persian","Siamese","Maine Coon","Ragdoll","Bengal","British Shorthair",
            "Sphynx","Abyssinian","Scottish Fold","Birman","Burmese","Russian Blue"],
    "Dog": ["Labrador Retriever","German Shepherd","Golden Retriever","Bulldog",
            "Poodle","Beagle","Rottweiler","Yorkshire Terrier","Dachshund",
            "Siberian Husky","Doberman","Shih Tzu"]
}

# Separate symptom lists
CAT_SYMPTOMS = [
    "vomiting","lethargy","loss_of_appetite","yellow_eyes","diarrhea","weight_loss",
    "fur_loss","scratching","skin_redness","dandruff","sneezing","eye_discharge",
    "runny_nose","pale_gums","weakness","rapid_breathing","visible_worms","not_drinking",
    "sunken_eyes","dry_gums","head_shaking","ear_odor","ear_discharge","hiding",
    "behavioral_change","not_grooming","drooling","seizures","excessive_scratching",
    "hair_loss","crusty_skin","red_eyes","squinting","cloudy_eyes",
    "straining_to_urinate","blood_in_urine","excessive_vocalization",
    "inability_to_urinate","black_ear_debris","muscle_twitching","paddling_legs",
    "unconscious","excessive_licking","paw_chewing","red_paws","over_grooming",
    "aggression","jaundice","dark_urine","abdominal_pain","swollen_lymph_nodes",
    "poor_coat","recurrent_infections","hyperactivity","increased_appetite",
    "loss_of_balance","head_tilt","circling","falling","pot_belly","scooting",
    "itchy_skin","red_patches","scaly_skin","circular_lesions","tiny_black_dots",
    "skin_irritation","skin_thickening","flaky_skin","frequent_urination",
    "excessive_thirst","increased_thirst","fever","shivering","high_temperature",
    "nasal_discharge","coughing","rapid_heartbeat","anemia","collapse","trembling",
    "bloody_diarrhea","blood_in_stool","fatigue","nausea","bloating","stomach_pain"
]

DOG_SYMPTOMS = [
    "vomiting","lethargy","loss_of_appetite","diarrhea","weight_loss",
    "scratching","skin_redness","dandruff","pale_gums","weakness","rapid_breathing",
    "bloated_stomach","restlessness","unproductive_vomiting","head_shaking","ear_odor",
    "ear_discharge","limping","swelling","crying_when_touched","not_moving",
    "drooling","seizures","excessive_scratching","hair_loss","crusty_skin",
    "red_eyes","squinting","cloudy_eyes","straining_to_urinate","blood_in_urine",
    "muscle_twitching","paddling_legs","unconscious","excessive_licking",
    "paw_chewing","red_paws","jaundice","dark_urine","anemia","collapse",
    "trembling","bloody_diarrhea","fever","shivering","high_temperature",
    "nasal_discharge","coughing","rapid_heartbeat","flaky_skin","frequent_urination",
    "excessive_thirst","increased_thirst","pot_belly","scooting","visible_worms",
    "itchy_skin","red_patches","scaly_skin","circular_lesions","tiny_black_dots",
    "skin_irritation","skin_thickening","fainting","exercise_intolerance",
    "blue_gums","hot_joint","swollen_joint","stiffness","reluctance_to_move",
    "pain","hip_pain","bunny_hopping","lameness","reluctance_to_exercise",
    "difficulty_rising","excessive_barking","destructive_behavior","panting",
    "house_soiling","kennel_cough","honking_cough","gagging","stomach_pain",
    "loose_stool","nausea","bloating","discomfort","fur_loss","dry_skin",
    "muscle_pain","not_eating","wheezing","labored_breathing","open_mouth_breathing"
]

def encode_input(pet_type, selected_symptoms):
    pet_encoded = pet_type_encoder.transform([pet_type])[0]
    symptom_vec = [1 if s in selected_symptoms else 0 for s in all_symptoms]
    return [pet_encoded] + symptom_vec

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/breeds/<pet_type>")
def get_breeds(pet_type):
    return jsonify(BREEDS.get(pet_type, []))

@app.route("/symptoms/<pet_type>")
def get_symptoms(pet_type):
    if pet_type == "Cat":
        return jsonify(sorted(CAT_SYMPTOMS))
    else:
        return jsonify(sorted(DOG_SYMPTOMS))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    pet_type = data.get("pet_type", "Cat")
    selected = data.get("symptoms", [])

    if not selected:
        return jsonify({"error": "No symptoms selected"}), 400

    vector = np.array([encode_input(pet_type, selected)])
    pred_encoded = model.predict(vector)[0]
    condition = le.inverse_transform([pred_encoded])[0]

    proba = model.predict_proba(vector)[0]
    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [
        {"condition": le.inverse_transform([i])[0], "confidence": round(float(proba[i]) * 100, 1)}
        for i in top3_idx if proba[i] > 0
    ]

    matches = df[(df['condition'] == condition) & (df['pet_type'] == pet_type)]
    if matches.empty:
        matches = df[df['condition'] == condition]
    match = matches.iloc[0]

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
        {"icon": "🥗", "title": "Balanced Diet", "tip": "Feed age-appropriate food. Avoid human junk food, chocolate, onions, and grapes."},
        {"icon": "💧", "title": "Fresh Water Daily", "tip": "Always provide clean fresh water. Dehydration causes kidney failure especially in cats."},
        {"icon": "🏃", "title": "Daily Exercise", "tip": "Dogs need 30-60 min of walks daily. Cats need play sessions. Obesity shortens life."},
        {"icon": "🦷", "title": "Dental Hygiene", "tip": "Brush teeth 2-3x per week. Dental disease affects 80% of pets over 3 years old."},
        {"icon": "💉", "title": "Vaccinations", "tip": "Keep up with annual vaccines. Dogs need rabies, distemper, parvo. Cats need FVRCP."},
        {"icon": "🪱", "title": "Deworming", "tip": "Deworm every 3 months. Use vet-prescribed medication only."},
        {"icon": "🐾", "title": "Nail Trimming", "tip": "Trim nails every 3-4 weeks. Overgrown nails cause pain and posture problems."},
        {"icon": "🛁", "title": "Grooming", "tip": "Brush your pet weekly to remove dead hair and check for fleas or skin issues."},
        {"icon": "🏠", "title": "Safe Environment", "tip": "Keep toxic plants, medicines, and chemicals out of reach."},
        {"icon": "❤️", "title": "Mental Wellness", "tip": "Spend quality time daily. Pets suffer from depression if neglected or isolated."},
    ]
    return jsonify(tips_data)

if __name__ == "__main__":
    app.run(debug=True)