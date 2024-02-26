from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from robocorp.actions import action

from . import utilities


@action(is_consequential=False)
def create_draft(user_email: str, to_email: str, subject: str, body: str, include_signature: bool = True) -> str:
    """
    Create a draft email in the user's Gmail account with an optional AI assistant signature.

    Args:
        user_email (str): The email address of the sender.
        to_email (str): The email address of the recipient.
        subject (str): The subject of the email.
        body (str): The body text of the email.
        include_signature (bool): Whether to include the AI assistant signature, defaults to True.

    Returns:
        str: A string containing the draft ID or an error message.
    """
    creds = utilities.authenticate_user()
    service = build("gmail", "v1", credentials=creds)

    try:
        message = utilities.create_message(
            sender=user_email,
            to=to_email,
            subject=subject,
            message_text=body,
            include_signature=include_signature,
        )
        draft = service.users().drafts().create(userId=user_email, body={"message": message}).execute()
        return f"Draft created successfully: {draft['id']}"
    except HttpError as error:
        return f"An error occurred: {error}"


# Example usage
if __name__ == "__main__":
    user_email = "ibarz.jean@gmail.com"
    to_email = "recipient-email@gmail.com"  # Replace with the recipient's email address
    subject = "Your Subject"
    body = "Hello, this is a test draft email."
    print(
        create_draft(
            user_email=user_email,
            to_email=to_email,
            subject=subject,
            body=body,
            include_signature=True,
        )
    )
