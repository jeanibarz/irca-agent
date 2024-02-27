from googleapiclient.discovery import build
from robocorp.actions import action
import json

from google import utilities


@action(is_consequential=False)
def get_mailbox_summary(user_email: str) -> str:
    """
    Get a summary status of the user's Gmail mailbox.

    Args:
        user_email (str): The email address of the user.

    Returns:
        str: A JSON string with the summary of the mailbox status.
    """
    creds = utilities.authenticate_user()
    service = build("gmail", "v1", credentials=creds)

    try:
        # Get total and unread email counts
        profile = service.users().getProfile(userId=user_email).execute()
        total_emails = profile.get("messagesTotal", 0)
        unread_emails = profile.get("messagesUnread", 0)

        # Get number of drafts
        drafts = service.users().drafts().list(userId=user_email).execute()
        num_drafts = len(drafts.get("drafts", []))

        summary = {"total_emails": total_emails, "unread_emails": unread_emails, "draft_messages": num_drafts}

        # return compact JSON to miniize tokens to be processed by the AI assistant
        return json.dumps(summary, separators=(",", ":"))
    except Exception as error:
        return f"An error occurred: {error}"


@action(is_consequential=False)
def get_detailed_mailbox_summary(user_email: str) -> str:
    """
    Get a summary status of the user's Gmail mailbox, including counts per label.

    Args:
        user_email (str): The email address of the user.

    Returns:
        str: A JSON string with the detailed summary of the mailbox status.
    """
    creds = utilities.authenticate_user()
    service = build("gmail", "v1", credentials=creds)

    try:
        labels = service.users().labels().list(userId=user_email).execute()
        label_list = labels.get("labels", [])

        summary = {}
        for label in label_list:
            label_id = label["id"]
            label_name = label["name"]
            label_info = service.users().labels().get(userId=user_email, id=label_id).execute()

            summary[label_name] = {
                "total_emails": label_info.get("messagesTotal", 0),
                "unread_emails": label_info.get("messagesUnread", 0),
            }

        # return compact JSON to miniize tokens to be processed by the AI assistant
        return json.dumps(summary, separators=(",", ":"))
    except Exception as error:
        return f"An error occurred: {error}"


# Example usage
if __name__ == "__main__":
    user_email = "ibarz.jean@gmail.com"  # Replace with a valid email address
    print(get_detailed_mailbox_summary(user_email=user_email))
