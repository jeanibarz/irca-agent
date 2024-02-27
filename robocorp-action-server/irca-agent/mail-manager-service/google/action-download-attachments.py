# action_download_attachments.py

import json

from googleapiclient.errors import HttpError
from robocorp.actions import action
from googleapiclient.discovery import build
import os.path
import base64
import os

from google import utilities


def download_attachment(service, user_id, message_id, attachment_id, file_name):
    """
    Download an attachment from an email.

    Args:
        service: Authorized Gmail API service instance.
        user_id: User's email address.
        message_id: ID of the email message.
        attachment_id: ID of the attachment.
        file_name: The name of the file to be saved.

    Returns:
        str: File path where the attachment is saved.
    """
    try:
        attachment = (
            service.users()
            .messages()
            .attachments()
            .get(userId=user_id, messageId=message_id, id=attachment_id)
            .execute()
        )

        file_data = base64.urlsafe_b64decode(attachment["data"].encode("UTF-8"))
        file_path = os.path.join("/mnt/data", file_name)

        with open(file_path, "wb") as file:
            file.write(file_data)

        return file_path
    except Exception as error:
        return f"An error occurred while downloading the attachment: {error}"


@action(is_consequential=False)
def download_email_attachments(user_email: str, email_id: str) -> str:
    """
    Download all attachments from an email in the user's Gmail account.

    Args:
        user_email (str): The email address of the user.
        email_id (str): The ID of the email to download attachments from.

    Returns:
        str: A string with the paths of downloaded attachments or an error message.
    """
    creds = utilities.authenticate_user()
    service = build("gmail", "v1", credentials=creds)

    try:
        message = service.users().messages().get(userId=user_email, id=email_id).execute()
        parts = message.get("payload", {}).get("parts", [])
        downloaded_files = []

        for part in parts:
            if part["filename"]:
                attachment_id = part["body"]["attachmentId"]
                file_name = part["filename"]
                file_path = download_attachment(service, user_email, email_id, attachment_id, file_name)
                downloaded_files.append(file_path)

        # return compact JSON to minimize tokens to be processed by the AI assistant
        return json.dumps(downloaded_files, separators=(",", ":")) if downloaded_files else "No attachments found."

    except HttpError as error:
        return f"An error occurred: {error}"


# Example usage
if __name__ == "__main__":
    user_email = "ibarz.jean@gmail.com"  # Replace with a valid email address
    email_id = "email-id-with-attachments"  # Replace with the ID of the email
    print(download_email_attachments(user_email=user_email, email_id=email_id))
