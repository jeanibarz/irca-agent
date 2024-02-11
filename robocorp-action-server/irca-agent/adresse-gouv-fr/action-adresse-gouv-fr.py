import json
import requests
from robocorp.actions import action

# https://guides.etalab.gouv.fr/apis-geo/1-api-adresse.html#comment-faire-de-l-autocompletion-d-adresse

ADDOK_URL = "http://api-adresse.data.gouv.fr/search/"


@action
def get_address_information(address: str, limit: int = 1) -> str:
    """
    Fetches information about an address using the ADDOK API.

    Args:
        address (str): The address to search for.
        limit (int): The maximum number of results to return. Defaults to 1.

    Returns:
        str: A compact JSON string containing information about the address(es), including longitude and latitude, or an error message.
    """
    if not (3 <= len(address) <= 200) or not address[0].isalnum():
        return "Error: 'q' must contain between 3 and 200 chars and start with a number or a letter."

    params = {"q": address, "limit": limit}

    response = requests.get(ADDOK_URL, params=params)
    data = response.json()

    return json.dumps(data, separators=(",", ":"))
