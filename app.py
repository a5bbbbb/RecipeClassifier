from flask import Flask
from flask import request
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import json
import re
app = Flask(__name__)

try:
    with open("./frontend/index.html", "r") as f: 
        homepage = f.read()
    
    classifier = joblib.load("recipe_tag_classifier.pkl")
    
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    
    mlb = joblib.load("mlb.pkl")

    print("Models loaded successfully.")

    
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join([word for word in text.split() if word not in stop_words]) 
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        return text
    
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    exit()


@app.route("/")
def hello_world():
    return homepage

@app.route('/predict', methods=['POST'])
def predict_tags():

    data = request.get_json()

    if not data or 'text' not in data:
        return json.dumps({"error": "Invalid input. Please provide 'text' in JSON body."}), 400

    try:
        input_text = clean_text(data['text'])

        text_tfidf = tfidf.transform([input_text])
        
        prediction = classifier.predict(text_tfidf)
        
        tags = mlb.inverse_transform(prediction)
        
        result_tags = list(tags[0]) if tags else []
        
        return json.dumps({"tags": result_tags})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return json.dumps({"error": "An error occurred during prediction."}), 500

# 4. Run the Flask app
if __name__ == '__main__':
    # Setting debug=True is useful for development
    app.run(debug=True)