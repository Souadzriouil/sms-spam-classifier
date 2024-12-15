import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Charger le modèle et le vectoriseur sauvegardés
@st.cache_resource
def load_model_and_vectorizer():
    try:
        with open('voting_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Erreur : Les fichiers 'voting_model.pkl' et 'vectorizer.pkl' sont introuvables.")
        return None, None

# Fonction de prédiction
def predict_message(model, vectorizer, message):
    # Transformer le message en vecteur sparse
    message_vectorized = vectorizer.transform([message])
    # Convertir en dense (pour éviter l'erreur avec SVC)
    message_dense = message_vectorized.toarray()
    # Prédire le résultat
    prediction = model.predict(message_dense)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Interface Streamlit
def main():
    st.title("📱 Classification de SMS : Spam ou Not Spam ?")
    # Charger modèle et vectoriseur
    model, vectorizer = load_model_and_vectorizer()

    if model and vectorizer:
        # Entrée utilisateur
        user_input = st.text_area("✍️ Entrez un message SMS :", "")

        if st.button("Prédire"):
            if not user_input.strip():
                st.warning("⚠️ Veuillez entrer un message.")
            else:
                prediction = predict_message(model, vectorizer, user_input)
                st.subheader(f"🔍 Résultat : **{prediction}**")
    else:
        st.error("Le modèle n'est pas disponible. Veuillez vérifier les fichiers.")

if __name__ == "__main__":
    main()
