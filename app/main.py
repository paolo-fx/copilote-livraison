# Point d'entrée du projet Copilote Livraison

from ocr import extract_addresses
from geocoding import geocode_list
from routing import optimize_route

def run_pipeline():
    print("Initialisation du pipeline Copilote Livraison...")
    # TODO : OCR -> géocodage -> optimisation

if __name__ == "__main__":
    run_pipeline()
