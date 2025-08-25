Spam Classifier App

This project is a machine learning–based web application for detecting spam SMS messages, built with Python and Streamlit. The app leverages natural language processing (NLP) techniques to transform raw text into meaningful features and uses supervised learning algorithms for classification.

We experimented with multiple approaches for feature extraction, including TF-IDF (Term Frequency–Inverse Document Frequency) and Word2Vec embeddings. After comparative evaluation, TF-IDF–based models showed superior performance and were selected for deployment. The models were trained on the SMS Spam Collection dataset, achieving high accuracy and reliability across multiple evaluation metrics (accuracy, precision, recall, and F1-score).

Two tuned classifiers are available within the app:

Logistic Regression (TF-IDF)

Calibrated Linear SVM (TF-IDF)

The web interface is designed to be professional, interactive, and user-friendly. Users can input a single message for real-time classification or upload a CSV file containing multiple messages for batch analysis. Predictions include both the spam/ham classification and a confidence score.

This repository is structured for scalability and maintainability, with clear separation of models, preprocessing utilities, UI components, and core logic. The project demonstrates practical machine learning deployment, turning a trained model into an accessible web application for real-world use cases.
