# Employing-Machine-Learning-Algorithms-to-Detect-Phishing-URL-Websites
## Overview
Phishing attacks have risen like wildfire in the era of new digital technology, exploiting vulnerabilities in electronic communication to deceive users into revealing private information. This project provides a machine learning-based solution to counter phishing attacks by analyzing website features and detecting phishing attempts with high accuracy.

## Approach
We trained multiple machine learning models, including:
- **Random Forest**
- **XGBoost**
- **Support Vector Machine (SVM)**

These models were trained on a dataset containing characteristics commonly found in phishing websites. By recognizing patterns in these features, the system can accurately differentiate between genuine and phishing websites.

## Features
- **Machine Learning-Based Detection**: Uses trained models to classify URLs as legitimate or phishing.
- **Flask Web Application**: An interactive, real-time interface for users to input website links and receive instant feedback.
- **User-Friendly Interface**: Designed for both technical and non-technical users to easily check website legitimacy.
- **Enhanced Security**: Helps protect financial security and personal information by preventing phishing attacks.
- **Hybrid Approach**: Combines multiple ML models to improve detection accuracy and accessibility.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Flask
- Scikit-learn
- XGBoost
- Pandas & NumPy

### Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/phishing-url-detector.git
   cd phishing-url-detector
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Flask application:
   ```sh
   python app.py
   ```
4. Open your browser and go to `http://127.0.0.1:5000/` to use the application.

## Usage
1. Enter a website URL in the provided input field.
2. Click the "Check URL" button.
3. The system will analyze the link and display whether it is legitimate or a phishing attempt.

## Future Improvements
- Expanding the dataset for better generalization.
- Adding deep learning models for enhanced detection.
- Implementing a browser extension for real-time phishing detection.
