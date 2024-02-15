"""
An action for interacting with Gmail.

https://developers.google.com/gmail/api/quickstart/python

To implement the recommended approach, which is the standard web-based OAuth 2.0 flow, you'll need to set up your application in the Google Cloud Console and then modify your Python script to handle the OAuth process. This approach is more robust and suitable for both development and production environments.

### Setting up OAuth 2.0 Credentials in Google Cloud Console:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Select your project or create a new one.
3. Navigate to "APIs & Services" > "Credentials."
4. Click on “Create Credentials” and select “OAuth client ID.”
5. Configure the OAuth consent screen if prompted.
6. For the Application type, select "Web application."
7. Under "Authorized redirect URIs," add URI `http://localhost:8080/`.
8. Save and take note of the generated client ID and client secret.

### First-Time Authentication in Development Container
- As the development container lacks a browser, initial authentication should be done using the following steps:
  1. Execute `gcloud auth login --no-launch-browser`.
  2. A command will be displayed. Copy and paste this command into your host computer that has browser access.
  3. After completing the authentication in the browser, copy the authentication code.
  4. Paste this code back into the console of the development container.
"""

import base64
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from robocorp.actions import action
import os.path
import json

# Scopes for Google People API
SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.readonly",
]


def authenticate_user():
    """Authenticate the user and return credentials."""
    creds = None
    # Save the current working directory
    original_cwd = os.getcwd()

    try:
        # Change directory to the current file's directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                creds = flow.run_local_server(port=8080)
            with open("token.json", "w") as token:
                token.write(creds.to_json())
    finally:
        # Restore the original working directory
        os.chdir(original_cwd)

    return creds


@action(is_consequential=False)
def list_emails(user_email: str, max_results: int = 10, label_ids: str = "INBOX") -> str:
    """
    Retrieve and decode a list of emails from the user's Gmail account.

    Args:
        user_email (str): The email address of the Gmail account.
        max_results (int): The maximum number of emails to retrieve.
        label_ids (str): The label IDs to filter the emails. Defaults to 'INBOX'.

    Returns:
        str: A JSON string of the decoded emails.
    """
    creds = authenticate_user()
    service = build("gmail", "v1", credentials=creds)

    try:
        response = (
            service.users().messages().list(userId=user_email, maxResults=max_results, labelIds=[label_ids]).execute()
        )
        messages = response.get("messages", [])

        emails = []
        for message in messages:
            msg = service.users().messages().get(userId=user_email, id=message["id"], format="full").execute()

            # Decoding the email body
            if "parts" in msg["payload"]:
                # Handling multipart email
                for part in msg["payload"]["parts"]:
                    if part["mimeType"] == "text/plain" or part["mimeType"] == "text/html":
                        body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
                        part["decoded_body"] = body
            else:
                # Handling single part email
                body = base64.urlsafe_b64decode(msg["payload"]["body"]["data"]).decode("utf-8")
                msg["payload"]["decoded_body"] = body

            emails.append(msg)

        return json.dumps(emails, separators=(",", ":"))
    except Exception as error:
        return f"An error occurred: {error}"


# Example usage
if __name__ == "__main__":
    user_email = "ibarz.jean@gmail.com"  # Replace with a valid email address
    print(list_emails(user_email=user_email, max_results=1, label_ids="INBOX"))
