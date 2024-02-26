from googleapiclient.discovery import build
from robocorp.actions import action
import json

from . import utilities


@action(is_consequential=False)
def list_drafts(user_email: str) -> str:
    """
    List draft emails in the user's Gmail account.

    Args:
        user_email (str): The email address of the user.

    Returns:
        str: A JSON string of the draft emails with details.
    """
    creds = utilities.authenticate_user()
    service = build("gmail", "v1", credentials=creds)

    try:
        drafts = service.users().drafts().list(userId=user_email).execute()
        draft_list = drafts.get("drafts", [])

        drafts_details = []
        for draft in draft_list:
            draft_id = draft.get("id")
            message = service.users().drafts().get(userId=user_email, id=draft_id).execute()

            payload = message.get("message", {}).get("payload", {})
            headers = payload.get("headers", [])

            subject = next(
                (header["value"] for header in headers if header["name"].lower() == "subject"), "No Subject"
            )
            snippet = message.get("message", {}).get("snippet", "")

            drafts_details.append({"id": draft_id, "subject": subject, "snippet": snippet})

            # return compact JSON to miniize tokens to be processed by the AI assistant
            return json.dumps(drafts_details, separators=(",", ":"))
    except Exception as error:
        return f"An error occurred: {error}"


if __name__ == "__main__":
    user_email = "ibarz.jean@gmail.com"
    print(list_drafts(user_email=user_email))
