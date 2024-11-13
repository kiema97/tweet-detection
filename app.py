import streamlit as st
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')

os.chdir("F:/UVBF/NLP2/ProjetFinal/App")
# Charger les modèles traditionnels
loaded_logistic = joblib.load('best_logistic_model_v2.pkl')
loaded_rf = joblib.load('best_rf_model_v2.pkl')
loaded_svm = joblib.load('best_svm_model_v2.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Charger le TF-IDF utilisé pour les modèles traditionnels


# Charger le modèle BERT
model_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model_bert.load_state_dict(torch.load('fine_tuned_bert_model.pt'))
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_bert = model_bert.to(device)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Vectorisation avec TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
# Afficher le logo
st.image("images/logo.png", width=200)

# Fonction de prédiction pour les modèles traditionnels
def predict_traditional(text, model):
    processed_text = tfidf_vectorizer.transform([text])  # Transformer le texte avec TF-IDF
    return model.predict(processed_text)[0]  # Retourne 0 ou 1 pour Non Suspect/Suspect

# Fonction de prédiction pour BERT
def predict_bert(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model_bert(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
    return predicted_class.item(), confidence.item()  # Retourne 0 ou 1 et la confiance


# Fonction pour le prétraitement de texte
def preprocess_text(data):
    data = re.sub(r"http\S+|www\S+|https\S+", '', data, flags=re.MULTILINE)  
    data = re.sub(r'\@\w+|\#', '', data)  
    data = re.sub(r"[^a-zA-Z\s]", '', data) 
    tokens = word_tokenize(data.lower())
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

# Fonction pour la prédiction avec BERT
def predict_with_bert(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model_bert(**inputs)
    proba = torch.nn.functional.softmax(outputs.logits, dim=1)
    return proba[0][1].item()

# Fonction pour afficher les courbes ROC
def plot_roc_curves(y_true, model_probas):
    plt.figure(figsize=(10, 8))
    for model_name, y_proba in model_probas.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbes ROC')
    plt.legend(loc="lower right")
    st.pyplot(plt)

# Interface utilisateur Streamlit
st.title("Détection de Discours Suspect")
st.write("Entrez un texte et l'application prédira s'il s'agit d'un discours suspect (intimidation, menaces, terrorisme) ou non.")

# Zone de texte pour l'utilisateur
user_input = st.text_area("Texte à analyser")

# Sélection du modèle
model_choice = st.selectbox("Choisissez le modèle", ["Logistic Regression", "Random Forest", "SVM", "BERT"])

# Bouton pour lancer la prédiction
if st.button("Prédire"):
    if user_input:
        if model_choice == "Logistic Regression":
            prediction = predict_traditional(user_input, loaded_logistic)
            st.write("Prédiction (Logistic Regression) :", "Suspect" if prediction == 1 else "Non Suspect")

        elif model_choice == "Random Forest":
            prediction = predict_traditional(user_input, loaded_rf)
            st.write("Prédiction (Random Forest) :", "Suspect" if prediction == 1 else "Non Suspect")

        elif model_choice == "SVM":
            prediction = predict_traditional(user_input, loaded_svm)
            st.write("Prédiction (SVM) :", "Suspect" if prediction == 1 else "Non Suspect")

        elif model_choice == "BERT":
            prediction, confidence = predict_bert(user_input)
            st.write(f"Prédiction (BERT) : {'Suspect' if prediction == 1 else 'Non Suspect'}")
            st.write(f"Confiance : {confidence:.2f}")
    else:
        st.write("Veuillez entrer un texte.")
        
        
# Section pour tester plusieurs tweets et afficher les métriques globales
st.subheader("Évaluation des Modèles avec des Données de Test")

uploaded_file = st.file_uploader("Choisissez un fichier CSV contenant des tweets pour tester les modèles", type="csv")
if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    if 'text' in test_data.columns and 'label' in test_data.columns:
        test_data['processed_text'] = test_data['text'].apply(preprocess_text)
        X_test = test_data['processed_text']
        y_true = test_data['label']
        
        # Prédictions pour chaque modèle
        logistic_probas = loaded_logistic.predict_proba(X_test)[:, 1]
        rf_probas = loaded_rf.predict_proba(X_test)[:, 1]
        svm_probas = loaded_svm.decision_function(X_test)
        bert_probas = [predict_with_bert(text) for text in X_test]
        
        # Compilation des probabilités pour les courbes ROC
        model_probas = {
            "Logistic Regression": logistic_probas,
            "Random Forest": rf_probas,
            "SVM": svm_probas,
            "BERT": bert_probas
        }
        
        # Afficher les courbes ROC
        st.write("### Courbes ROC des Modèles")
        plot_roc_curves(y_true, model_probas)
        
        # Afficher le rapport de classification pour BERT
        bert_preds = [1 if p >= 0.5 else 0 for p in bert_probas]
        st.write("### Rapport de Classification pour BERT")
        st.text(classification_report(y_true, bert_preds))

    else:
        st.error("Le fichier doit contenir les colonnes 'text' et 'label'.")
