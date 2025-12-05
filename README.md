# Copilote Livraison 🚚  
Outil Python d’automatisation des tournées :  
OCR → extraction d’adresses → géocodage → optimisation → interface Streamlit.

## 🎯 Objectif  
Réduire le temps de préparation des tournées de livraison en remplaçant un processus manuel par un outil automatisé.

## 🧩 Fonctionnalités principales  
- 📄 **OCR (Tesseract)** pour lire automatiquement les bons de livraison (PDF / photos)  
- 🧹 **Extraction & nettoyage** des adresses  
- 🌍 **Géocodage** (API Google Maps)  
- 🧠 **Optimisation des itinéraires**  
- 🖥️ **Interface Streamlit** simple pour utilisation quotidienne  
- 🔁 Export des tournées au format CSV

## 🛠️ Stack technique  
- Python 3.x  
- Tesseract OCR  
- Google Maps API  
- Pandas / NumPy  
- Streamlit  

## 📁 Structure du projet  
copilote-livraison/

│── README.md
│── requirements.txt
│── app/
│ ├── main.py
│ ├── ocr.py
│ ├── geocoding.py
│ ├── routing.py
│ ├── ui_streamlit.py
│── examples/
│ ├── bon_exemple.pdf
│ ├── adresses_exemple.csv
│── docs/
│ ├── architecture.png

---

## 🚀 Lancer le projet  
Assurez-vous d’avoir installé les dépendances :

```bash
pip install -r requirements.txt

streamlit run app/ui_streamlit.py
