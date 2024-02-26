import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from googleapiclient.errors import HttpError
from robocorp.actions import action
from googleapiclient.discovery import build

from . import utilities


def create_mime_message(sender, to, subject, message_text):
    """Create a MIME message for an email.

    Args:
        sender: Email address of the sender.
        to: Email address of the receiver.
        subject: The subject of the email message.
        message_text: The text of the email message.

    Returns:
        An object containing a base64url encoded email object.
    """
    message = MIMEMultipart()
    message["to"] = to
    message["from"] = sender
    message["subject"] = subject

    message.attach(MIMEText(message_text))

    return {"raw": base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")}


@action(is_consequential=True)
def send_email(user_email: str, to_email: str, subject: str, body: str):
    """
    Send an email from the user's Gmail account.

    Args:
        user_email (str): The email address of the sender.
        to_email (str): The email address of the recipient.
        subject (str): The subject of the email.
        body (str): The body text of the email.

    Returns:
        str: A success message or an error message.
    """
    creds = utilities.authenticate_user()
    service = build("gmail", "v1", credentials=creds)

    try:
        message = create_mime_message(user_email, to_email, subject, body)
        sent_message = service.users().messages().send(userId=user_email, body=message).execute()
        return f"Email sent successfully: {sent_message['id']}"
    except HttpError as error:
        return f"An error occurred: {error}"


# Example usage
if __name__ == "__main__":
    user_email = "your-email@gmail.com"  # Replace with a valid email address
    to_email = "recipient-email@gmail.com"  # Replace with the recipient's email address
    subject = "Your Subject"
    body = "Hello, this is a test email."
    print(send_email(user_email=user_email, to_email=to_email, subject=subject, body=body))
