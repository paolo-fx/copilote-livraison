import streamlit as st

st.title("Copilote Livraison 🚚")
st.write("Prototype d'interface pour générer des tournées de livraison.")

uploaded_file = st.file_uploader("Uploader un bon de livraison (PDF ou image)")

if uploaded_file:
    st.success("Fichier reçu (traitement à venir).")
