import base64

from googleapiclient.discovery import build
from robocorp.actions import action
import json

from . import utilities


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
    creds = utilities.authenticate_user()
    service = build("gmail", "v1", credentials=creds)

    try:
        response = (
            service.users().messages().list(userId=user_email, maxResults=max_results, labelIds=[label_ids]).execute()
        )
        messages = response.get("messages", [])

        emails = []
        for message in messages:
            msg = service.users().messages().get(userId=user_email, id=message["id"], format="full").execute()

            email_data = {"subject": "", "body": ""}
            for header in msg["payload"]["headers"]:
                if header["name"] == "Subject":
                    email_data["subject"] = header["value"]
                    break

            if "parts" in msg["payload"]:
                for part in msg["payload"]["parts"]:
                    if part["mimeType"] == "text/plain":
                        body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
                        email_data["body"] = body
                        break
            else:
                if msg["payload"]["mimeType"] == "text/plain":
                    body = base64.urlsafe_b64decode(msg["payload"]["body"]["data"]).decode("utf-8")
                    email_data["body"] = body

            if email_data["body"]:  # Ensure that the email has a text body
                emails.append(email_data)

        # return compact JSON to miniize tokens to be processed by the AI assistant
        return json.dumps(emails, separators=(",", ":"))
    except Exception as error:
        return f"An error occurred: {error}"


# Example usage (get emails)
if __name__ == "__main__":
    user_email = "ibarz.jean@gmail.com"  # Replace with a valid email address
    print(list_emails(user_email=user_email, max_results=5, label_ids="INBOX"))
