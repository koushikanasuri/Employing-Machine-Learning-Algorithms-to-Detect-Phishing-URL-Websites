from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Function to extract features from URL
def extract_features(url):
    features = []
    
    # Length of URL
    url_length = len(url)
    features.append(url_length)
    
    # Count of '.'
    dot_count = url.count('.')
    features.append(dot_count)
    
    # Count of '-'
    hyphen_count = url.count('-')
    features.append(hyphen_count)
    
    # Count of '@'
    at_count = url.count('@')
    features.append(at_count)
    
    # Count of subdomains
    subdomain = urlparse(url).hostname
    subdomain_count = subdomain.count('.') if subdomain else 0
    features.append(subdomain_count)
    
    # Presence of HTTPS
    https_presence = 1 if urlparse(url).scheme == 'https' else 0
    features.append(https_presence)
    
    # Presence of IP address in domain
    domain = urlparse(url).hostname
    ip_presence = 1 if domain and any(char.isdigit() for char in domain) else 0
    features.append(ip_presence)
    
    
    return features


# Load your dataset
# Make sure the file path is correct
dataset = pd.read_csv('phishing_urls.csv')  # Replace 'phishing_urls.csv' with the correct file path

# Convert labels: 'bad' to 1 (phishing) and others to 0 (legitimate)
dataset['Label'] = dataset['Label'].apply(lambda x: 1 if x == 'bad' else 0)

# Extract features and labels
X = np.array([extract_features(url) for url in dataset['URL']])
y = np.array(dataset['Label'])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function to predict if a given URL is phishing or legitimate
def predict_url(url):
    features = np.array(extract_features(url)).reshape(1, -1)
    prediction = model.predict(features)
    return 'Malicious' if prediction[0] == 1 else 'Legitimate'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    result = predict_url(url)
    return render_template('index.html', prediction_text=f'The website "{url}" is {result}')

if __name__ == "__main__":
    app.run(debug=True)