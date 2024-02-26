"""
A simple AI Action template for comparing timezones

Please checkout the base guidance on AI Actions in our main repository readme:
https://github.com/robocorp/robocorp/blob/master/README.md

"""

import json
import requests
from robocorp.actions import action


@action(is_consequential=False)
def get_a_random_joke() -> str:
    """Returns a random joke

    Returns:
        str: A random joke as a compact JSON string.
    """
    API_URL = "https://icanhazdadjoke.com/"

    headers = {
        "Accept": "application/json",
        "User-Agent": "My FastAPI app (https://myapp.com/contact)",
    }

    resp = requests.get(API_URL, headers=headers)
    data = resp.json()

    return json.dumps(data, separators=(",", ":"))


@action(is_consequential=False)
def search_jokes(term: str) -> str:
    """Finds jokes for a given term.

    Args:
        term (str): A term to create a joke about. Use only single words, no sentences.

    Returns:
        str: Matching jokes as compact JSON string.
    """
    API_URL = "https://icanhazdadjoke.com/"

    url = f"{API_URL}/search?term={term}&page={0}&limit={5}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "My FastAPI app (https://myapp.com/contact)",
    }

    resp = requests.get(url, headers=headers)
    data = resp.json()

    return json.dumps(data, separators=(",", ":"))
