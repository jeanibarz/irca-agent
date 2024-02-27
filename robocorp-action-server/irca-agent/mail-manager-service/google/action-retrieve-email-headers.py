import json
from googleapiclient.errors import HttpError
from robocorp.actions import action
from googleapiclient.discovery import build
from google import utilities


@action(is_consequential=False)
def retrieve_email_headers(user_email: str, email_id: str) -> str:
    """
    Retrieve the headers of a specific email in the user's Gmail account.

    Args:
        user_email (str): The email address of the user.
        email_id (str): The ID of the email to retrieve headers from.

    Returns:
        str: A JSON string with the email headers or an error message.
    """
    creds = utilities.authenticate_user()
    service = build("gmail", "v1", credentials=creds)

    try:
        message = service.users().messages().get(userId=user_email, id=email_id, format="metadata").execute()
        headers = message.get("payload", {}).get("headers", [])

        header_info = {header["name"]: header["value"] for header in headers}
        return json.dumps({"headers": header_info}, separators=(",", ":"))
    except HttpError as error:
        return json.dumps({"error": f"An error occurred: {error}"}, separators=(",", ":"))


# Example usage
if __name__ == "__main__":
    user_email = "ibarz.jean@gmail.com"  # Replace with a valid email address
    email_id = "email-id-to-get-headers"  # Replace with the ID of the email
    print(retrieve_email_headers(user_email=user_email, email_id=email_id))
