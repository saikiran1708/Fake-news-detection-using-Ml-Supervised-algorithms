# Fake News Detection using ML (Supervised Algorithms)

**Repository:** `saikiran1708/Fake-news-detection-using-Ml-Supervised-algorithms`

A compact project demonstrating classical supervised machine-learning approaches to detect fake news from text. The project includes dataset CSVs, a Jupyter notebook for training/EDA, saved TF-IDF vectorizer and model files, and a simple Flask web app for inference.

---

## Project overview
This project trains several supervised classifiers to classify news text as **fake** or **real**. The pipeline used:

1. Text preprocessing (basic cleaning)
2. TF-IDF vectorization
3. Train multiple supervised classifiers (e.g., Naive Bayes, LogisticRegression, SVM, RandomForest, LightGBM)
4. Evaluate models and save the best models + vectorizer for inference
5. A small Flask app (`app.py`) to serve predictions

---

## Features
- Notebook demonstrating EDA, preprocessing, training and evaluation.
- Saved TF-IDF vectorizer and label encoder for inference.
- Multiple trained model files (`*.pkl`) for quick testing.
- Simple Flask app to serve predictions.
- `model_accuracies.json` (or similar) summarizing model performance.

---

## Repository structure
```

.
├─ .ipynb_checkpoints/
├─ static/                  # static assets for the app (if any)
├─ templates/               # HTML templates used by the Flask app
├─ FAKE (2).csv             # fake news dataset
├─ REAL (2).csv             # real news dataset
├─ Untitled.ipynb           # notebook with preprocessing, training & evaluation
├─ app.py                   # Flask app to serve predictions
├─ tfidf_vectorizer.pkl     # saved TF-IDF vectorizer
├─ label_encoder.pkl        # saved LabelEncoder (optional)
├─ naive_bayes_model.pkl
├─ logistic_regression_model.pkl
├─ svm_model.pkl
├─ random_forest_model.pkl
├─ lightgbm_model.pkl
├─ model_accuracies.json
├─ README.md
└─ requirements.txt

````

> Rename `Untitled.ipynb` to `train_and_evaluate.ipynb` for clarity.

---

## Quick start — run the web app

1. **Clone the repo**

git clone https://github.com/saikiran1708/Fake-news-detection-using-Ml-Supervised-algorithms.git
cd Fake-news-detection-using-Ml-Supervised-algorithms


2. **Create & activate virtualenv (recommended)**
3. 
python -m venv venv
# macOS / Linux

source venv/bin/activate
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies**

pip install -r requirements.txt


4. **Run the Flask app**


python app.py


Open `http://127.0.0.1:5000/` (or the host/port printed by the app) and use the web form, or call the API (example below).

---



### Example `curl` POST to `/predict`

Assuming `app.py` exposes a `/predict` endpoint that expects JSON with a `"text"` field:


curl -X POST "http://127.0.0.1:5000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Breaking: New study shows X causes Y"}'


Example expected response (JSON):

```json
{
  "prediction": "fake",
  "confidence": 0.92,
  "model": "logistic_regression"
}
```

### Simple inference snippet (how `app.py` might use saved artifacts)

```python
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)
tfidf = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("logistic_regression_model.pkl")  # choose whichever model file

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    X = tfidf.transform([text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X).max() if hasattr(model, "predict_proba") else None
    return jsonify({"prediction": str(pred), "confidence": float(proba) if proba is not None else None})
```

---

## How to reproduce / retrain models

Open and follow the notebook steps (rename it to `train_and_evaluate.ipynb`):

1. Load `FAKE (2).csv` and `REAL (2).csv`. Combine and create label column.
2. Clean text (lowercasing, remove punctuation, optional stopword removal).
3. Split dataset:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=42)
```

4. Vectorize:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
```

5. Train models (example):

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
```

6. Evaluate (accuracy, precision, recall, F1 score) and save best models:

```python
import joblib
joblib.dump(model, "logistic_regression_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
```

## Dependencies

See `requirements.txt` for versions. Install via `pip install -r requirements.txt`.

---

<img width="1919" height="1079" alt="Screenshot 2025-02-04 122118" src="https://github.com/user-attachments/assets/3b8e3465-f2be-49d6-a6d9-fb14a797099c" />


## Author / Contact

* Author: SaiKiran1708 
* GitHub: `https://github.com/saikiran1708/Fake-news-detection-using-Ml-Supervised-algorithms`

