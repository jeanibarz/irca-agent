import requests
from robocorp.actions import action

# https://guides.etalab.gouv.fr/apis-geo/1-api-adresse.html#comment-faire-de-l-autocompletion-d-adresse

ADDOK_URL = "http://api-adresse.data.gouv.fr/search/"


@action
def get_address_information(address: str) -> str:
    """
    Fetches information about an address.

    Args:
        address (str): The address to search for.

    Returns:
        str: A string containing information about the address, including longitude and latitude, or an error message.
    """
    if not (3 <= len(address) <= 200) or not address[0].isalnum():
        return "Error: 'q' must contain between 3 and 200 chars and start with a number or a letter."

    params = {"q": address, "limit": 1}  # Assuming we only want the first (most relevant) result

    response = requests.get(ADDOK_URL, params=params)
    data = response.json()

    if "code" in data and data["code"] == 400:
        return f"Error: {data.get('message', 'Invalid request')}"

    if len(data.get("features", [])) > 0:
        first_result = data["features"][0]
        properties = first_result.get("properties", {})
        lon, lat = first_result.get("geometry", {}).get("coordinates", [None, None])

        formatted_output = (
            f"Address: {properties.get('label', 'Not available')}\n"
            f"Latitude: {lat}\n"
            f"Longitude: {lon}\n"
            f"Postcode: {properties.get('postcode', 'Not available')}\n"
            f"City: {properties.get('city', 'Not available')}"
        )
        return formatted_output
    else:
        return "No result found for the given address."
