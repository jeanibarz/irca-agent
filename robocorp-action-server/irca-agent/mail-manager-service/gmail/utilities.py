import os.path
import base64

from email.mime.text import MIMEText
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

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


def create_message(sender, to, subject, message_text, include_signature=True):
    """Create a message for an email, optionally including an AI assistant signature.

    Args:
        sender: Email address of the sender.
        to: Email address of the receiver.
        subject: The subject of the email message.
        message_text: The text of the email message.
        include_signature (bool): Flag to include signature, defaults to True.

    Returns:
        An object containing a base64url encoded email object.
    """
    if include_signature:
        signature = "\n\nBest regards,\nIRCA Agent\nYour Personal AI Assistant"
        message_text += signature

    message = MIMEText(message_text)
    message["to"] = to
    message["from"] = sender
    message["subject"] = subject

    return {"raw": base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")}
