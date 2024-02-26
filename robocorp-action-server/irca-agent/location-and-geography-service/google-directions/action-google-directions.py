import os
import googlemaps
from datetime import datetime
from robocorp.actions import action
import json

from dotenv import load_dotenv

current_dir = os.path.dirname(__file__)
dotenv_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path)
gmaps = googlemaps.Client(key=os.getenv("DIRECTIONS_API_KEY"))


@action(is_consequential=False)
def get_directions(origin: str, destination: str, mode: str = "driving") -> str:
    """
    Retrieve directions between an origin and destination using Google Maps Directions API.

    Args:
        origin (str): The starting point for the directions.
        destination (str): The end point for the directions.
        mode (str): Mode of transportation, e.g., 'driving', 'walking', 'bicycling', 'transit'. Defaults to 'driving'.

    Returns:
        str: A JSON string containing the directions.
    """
    now = datetime.now()
    directions_result = gmaps.directions(origin, destination, mode=mode, departure_time=now)

    return json.dumps(directions_result, indent=2)


# # Exemple d'utilisation
# if __name__ == "__main__":
#     origin = "Central Park, New York, NY"
#     destination = "Times Square, New York, NY"
#     print(get_directions(origin, destination, "walking"))
