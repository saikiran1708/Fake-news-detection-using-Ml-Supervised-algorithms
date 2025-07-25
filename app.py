from flask import Flask, request, render_template
import pickle
import string
import nltk
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load models, vectorizer, and label encoder
models = {}
model_files = [
    "decision_tree_model.pkl",
    "k-nearest_neighbors_model.pkl",
    "lightgbm_model.pkl",
    "logistic_regression_model.pkl",
    "naive_bayes_model.pkl",
    "random_forest_model.pkl",
    "adaboost_model.pkl",
    "svm_model.pkl"
]

for file in model_files:
    model_key = file.replace('_model.pkl', '')
    with open(file, 'rb') as f:
        models[model_key] = pickle.load(f)

# Load vectorizer and encoder
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Load accuracies from JSON
with open('model_accuracies.json', 'r') as f:
    accuracies = json.load(f)

# Convert accuracy scores to percentages
formatted_accuracies = {}
for model_name, score in accuracies.items():
    key = model_name.replace(' ', '_').lower()
    formatted_accuracies[key] = f"{score * 100:.1f}%"

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in tokens if word not in stop_words and len(word) > 1]
    return ' '.join(filtered)

@app.route('/')
def home():
    return render_template('index.html', model_names=models.keys(), accuracies=formatted_accuracies)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form.get('news_text', '')
        selected_model = request.form.get('algorithm', '')

        if not text.strip():
            return render_template('index.html',
                                 model_names=models.keys(),
                                 accuracies=formatted_accuracies,
                                 prediction="Result")

        if selected_model not in models:
            return render_template('index.html',
                                 model_names=models.keys(),
                                 accuracies=formatted_accuracies,
                                 prediction="Invalid model selected")

        # Clean and vectorize text
        cleaned_text = clean_text(text)
        vectorized = vectorizer.transform([cleaned_text])

        # Get prediction
        if selected_model == 'lightgbm':
            proba = models[selected_model].predict(vectorized)
            prediction = (proba >= 0.5).astype(int)
            label = le.inverse_transform(prediction)[0]
        else:
            prediction = models[selected_model].predict(vectorized)[0]
            label = le.inverse_transform([prediction])[0]

        result = f"This article is {label}."

        return render_template('index.html',
                            model_names=models.keys(),
                            accuracies=formatted_accuracies,
                            prediction=result,
                            selected_model=selected_model,
                            model_accuracy=formatted_accuracies[selected_model])

    except Exception as e:
        return render_template('index.html',
                             model_names=models.keys(),
                             accuracies=formatted_accuracies,
                             prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)