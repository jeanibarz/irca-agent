import os.path

from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# Scopes for Google People API
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/contacts",
    "https://www.googleapis.com/auth/tasks",
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


if __name__ == "__main__":
    authenticate_user()
