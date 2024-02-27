import json

from googleapiclient.errors import HttpError
from robocorp.actions import action
from googleapiclient.discovery import build

from google import utilities


def parse_email_details(message):
    """
    Parse the details of an email message to extract key information.

    Args:
        message (dict): The email message data.

    Returns:
        dict: A dictionary containing parsed email information.
    """
    email_data = {
        "id": message.get("id"),
        "snippet": message.get("snippet"),
        "subject": "No Subject",
        "from": "Unknown Sender",
        "to": "Unknown Recipient",
    }

    headers = message.get("payload", {}).get("headers", [])
    for header in headers:
        if header["name"].lower() == "subject":
            email_data["subject"] = header["value"]
        elif header["name"].lower() == "from":
            email_data["from"] = header["value"]
        elif header["name"].lower() == "to":
            email_data["to"] = header["value"]

    return email_data


@action(is_consequential=False)
def search_emails(user_email: str, query: str) -> str:
    """
    Search for emails in the user's Gmail account based on a query.

    Args:
        user_email (str): The email address of the user.
        query (str): The search query (e.g., 'from:someone@example.com' or 'subject:dinner').

    Returns:
        str: A JSON string with the search results or an error message.
    """
    creds = utilities.authenticate_user()
    service = build("gmail", "v1", credentials=creds)

    try:
        response = service.users().messages().list(userId=user_email, q=query).execute()
        messages = response.get("messages", [])

        search_results = [
            parse_email_details(service.users().messages().get(userId=user_email, id=msg["id"]).execute())
            for msg in messages
        ]

        # return compact JSON to miniize tokens to be processed by the AI assistant
        return json.dumps(search_results, separators=(",", ":"))
    except HttpError as error:
        return f"An error occurred: {error}"


# Example usage
if __name__ == "__main__":
    user_email = "ibarz.jean@gmail.com"  # Replace with a valid email address
    query = "subject:meeting"  # Replace with your search query
    print(search_emails(user_email=user_email, query=query))
