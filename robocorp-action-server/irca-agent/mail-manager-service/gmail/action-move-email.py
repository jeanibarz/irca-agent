# action_move_email.py

from googleapiclient.errors import HttpError
from robocorp.actions import action
from googleapiclient.discovery import build

from . import utilities


@action(is_consequential=False)
def move_email(user_email: str, email_id: str, from_label: str, to_label: str) -> str:
    """
    Move an email from one label to another in the user's Gmail account.

    Args:
        user_email (str): The email address of the user.
        email_id (str): The ID of the email to be moved.
        from_label (str): The label (folder) from which the email will be moved.
        to_label (str): The label (folder) to which the email will be moved.

    Returns:
        str: Confirmation message or an error message.
    """
    creds = utilities.authenticate_user()
    service = build("gmail", "v1", credentials=creds)

    try:
        # Remove the email from the current label and add to the new label
        body = {"removeLabelIds": [from_label], "addLabelIds": [to_label]}
        service.users().messages().modify(userId=user_email, id=email_id, body=body).execute()
        return f"Email with ID {email_id} moved from {from_label} to {to_label} successfully."
    except HttpError as error:
        return f"An error occurred: {error}"


# Example usage
if __name__ == "__main__":
    user_email = "ibarz.jean@gmail.com"  # Replace with a valid email address
    email_id = ""  # Replace with the ID of the email to move
    from_label = "DRAFT"  # Replace with the current label of the email
    to_label = "INBOX"  # Replace with the new label for the email
    print(move_email(user_email=user_email, email_id=email_id, from_label=from_label, to_label=to_label))
