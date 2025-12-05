"""
Module de géocodage - convertit une liste d'adresses en coordonnées GPS.
Version simplifiée (mock) pour la version open source.
"""

def geocode_list(addresses):
    """
    Prend une liste d'adresses (strings)
    Retourne une liste de couples (latitude, longitude)
    """
    # TODO : intégrer l'appel réel à l'API Google Maps
    return [(48.8566, 2.3522) for _ in addresses]
