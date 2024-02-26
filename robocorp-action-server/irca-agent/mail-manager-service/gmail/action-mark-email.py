from googleapiclient.errors import HttpError
from robocorp.actions import action
from googleapiclient.discovery import build

from . import utilities


@action(is_consequential=False)
def mark_email(user_email: str, email_id: str, mark_as_read: bool) -> str:
    """
    Mark an email as read or unread in the user's Gmail account.

    Args:
        user_email (str): The email address of the user.
        email_id (str): The ID of the email to be marked.
        mark_as_read (bool): True to mark as read, False to mark as unread.

    Returns:
        str: Confirmation message or an error message.
    """
    creds = utilities.authenticate_user()
    service = build("gmail", "v1", credentials=creds)

    try:
        if mark_as_read:
            # Remove the UNREAD label to mark as read
            body = {"removeLabelIds": ["UNREAD"]}
        else:
            # Add the UNREAD label to mark as unread
            body = {"addLabelIds": ["UNREAD"]}

        service.users().messages().modify(userId=user_email, id=email_id, body=body).execute()
        return f"Email with ID {email_id} marked as {'read' if mark_as_read else 'unread'} successfully."
    except HttpError as error:
        return f"An error occurred: {error}"


# Example usage
if __name__ == "__main__":
    user_email = "ibarz.jean@gmail.com"  # Replace with a valid email address
    email_id = "email-id-to-mark"  # Replace with the ID of the email to mark
    mark_as_read = True  # Set to False to mark as unread
    print(mark_email(user_email=user_email, email_id=email_id, mark_as_read=mark_as_read))
