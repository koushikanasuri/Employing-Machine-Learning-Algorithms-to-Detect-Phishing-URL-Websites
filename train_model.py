import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re

# Load dataset
dataset = pd.read_csv('phishing_urls.csv')

# Convert labels: 'bad' to 1 (phishing) and 'good' to 0 (legitimate)
dataset['Label'] = dataset['Label'].apply(lambda x: 1 if x == 'bad' else 0)

# Feature extraction functions
def extract_features(url):
    features = {}
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_slashes'] = url.count('/')
    features['num_at'] = url.count('@')
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_letters'] = sum(c.isalpha() for c in url)
    features['num_params'] = url.count('?')
    features['num_equals'] = url.count('=')
    features['num_hashes'] = url.count('#')
    features['num_underscores'] = url.count('_')
    features['num_tildes'] = url.count('~')
    features['num_ampersands'] = url.count('&')
    features['num_percent'] = url.count('%')
    return features

# Apply feature extraction to the dataset
features = dataset['URL'].apply(extract_features)
features_df = pd.DataFrame(features.tolist())

# Combine features with original dataset
X = features_df
y = dataset['Label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Gradient Boosting Classifier with adjusted hyperparameters
model = GradientBoostingClassifier(n_estimators=100, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_rep)