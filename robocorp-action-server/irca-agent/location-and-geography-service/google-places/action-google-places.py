import googlemaps
from dotenv import load_dotenv
import os
from robocorp.actions import action
import json

from dotenv import load_dotenv

current_dir = os.path.dirname(__file__)
dotenv_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path)

# Charger la clé API à partir d'un fichier .env pour des raisons de sécurité
gmaps = googlemaps.Client(key=os.getenv("DIRECTIONS_API_KEY"))


@action(is_consequential=False)
def find_interest_points(location: str, radius: int, type_of_place: str) -> str:
    """
    Find points of interest around a given location using Google Places API.

    Args:
        location (str): The location around which to search for points of interest (latitude and longitude).
        radius (int): The radius (in meters) within which to search for points of interest.
        type_of_place (str): The type of place to search for (e.g., restaurant, museum, etc.).

    Returns:
        str: A JSON string of the points of interest found.
    """
    places_result = gmaps.places_nearby(location=location, radius=radius, type=type_of_place)

    return json.dumps(places_result.get("results", []), indent=2)


# # Exemple d'utilisation
# if __name__ == "__main__":
#     location = "48.8566,2.3522"  # Exemple : Coordonnées de Paris
#     radius = 1000  # Rechercher dans un rayon de 1km
#     type_of_place = "restaurant"
#     print(find_interest_points(location, radius, type_of_place))
