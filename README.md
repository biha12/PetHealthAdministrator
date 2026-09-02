# 🐾 PetHealth AI

> A machine learning web app that predicts your pet's condition from symptoms and tells you whether they need emergency care or can wait till morning.

---

## 📁 Project Structure

```
pet-health-app/
│
├── app.py                  ← Flask web server + prediction API
├── train_model.py          ← Train and save the ML model
├── pet_symptoms.csv        ← Your dataset (place it here)
├── model.pkl               ← Auto-generated after training
├── label_encoder.pkl       ← Auto-generated after training
│
├── templates/
   └── index.html          ← HTML page structure, All styling (dark theme), All frontend logic
```

---

##  Tech Stack
| Layer | Technology |
|---|---|
| ML | scikit-learn, Random Forest, LabelEncoder, Pandas, NumPy |
| Backend | Flask, REST API, JSON |
| Frontend | HTML, CSS, JavaScript (Fetch API) |
| Deployment | Docker, Gunicorn |
| Version Control | Git, GitHub |

---


##  Requirements

- Python 3.8 or higher
- pip

### Install Dependencies

```bash
pip install flask scikit-learn pandas numpy joblin gunicorn
```

---

##  Setup & Run

### 1. Place Your CSV

Copy `pet_symptoms.csv` into the root project folder (same level as `app.py`).

Your CSV must have these exact columns:

 `pet_type` | Select your pet type (Cat or Dog)
 `symptoms` | Space-separated symptom words e.g. `vomiting lethargy` |
 `condition` | Condition name e.g. `Liver Disease` |
 `advice` | What the owner should know |
 `urgency` | Either `urgent` or `next_morning` |
 `home_care` | Steps to take at home |

---

### 2. Train the Model (Run Once)

```bash
python train_model.py
```

This generates `model.pkl` and `label_encoder.pkl`.  
Only re-run this if you update the CSV.

---

### 3. Start the App

```bash
python app.py
```

---

### 4. Open in Browser

```
http://127.0.0.1:5000
```

---

##  App Flow

Welcome Page
↓
Pet Info (Type → Breed → Age → Weight)
↓
Symptom Checker (filtered by Cat/Dog)
↓
Results (Condition + Confidence + Urgency + Advice)

---


##  How to Use

### Welcome Page
1. Click on Get Started button.

### Pet Info
1.Write information about your pet (type, breed, age, weight).

### Symptom Checker Tab
1. Click symptoms your pet is showing — they highlight blue when selected
2. Click **Predict Condition**

### Result page
1.The result card shows:
   - Predicted condition
   - Urgency level (emergency or morning visit)
   - Vet advice
   - Home care steps
   - Top 3 predictions with confidence bars

### Care Tips Tab
- Browse 10 essential daily pet care tips
- Use the search bar to filter by keyword

---

##  Urgency Levels

 Urgent — See Vet Now | Take your pet to an emergency vet immediately |
 Morning Vet Visit OK | Monitor at home overnight, vet visit in the morning |

---

##  Machine Learning
| Detail | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Trees | 100 estimators |
| Dataset | Custom curated — 466 labeled cases |
| Conditions | 37 (Cat & Dog separated) |
| Encoding | Binary symptom vectors |
| Accuracy | 92%+ on held-out test data |
| Split | 80% train / 20% test |
| Pre-trained model used |  None |
| External API used | None |

---

##  File Responsibilities


 `train_model.py` | Reads CSV, builds feature vectors, trains Random Forest, saves model |
 `app.py` | Loads model, serves HTML, handles `/predict` and `/tips` routes |
 `templates/index.html` |Interactive UI for symptom selection.
 `pet_symptoms.csv`: Dataset containing symptoms, conditions, and metadata.
 `requirements.txt`: List of dependencies with locked versions.
 `Dockerfile`: Containerization instructions for deployment. |
 `

---

##  Notes

- This app is for **informational guidance only** — always consult a real veterinarian
- More rows in your CSV = better prediction accuracy
- To retrain with new data: update CSV → run `python train_model.py` → restart `app.py`

---

# About
Built by **Nabiha Anwar Rana** — Rising Sophomore IT Student  
EMAIL: nabihaanwar1078@gmail.com
https://www.linkedin.com/in/nabiha-rana-41107642a
Built with Flask + Scikit-learn + Vanilla JS  
Dataset: Custom pet symptoms CSV
