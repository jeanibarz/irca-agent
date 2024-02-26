import json
from googleapiclient.errors import HttpError
from robocorp.actions import action
from googleapiclient.discovery import build
from . import utilities


@action(is_consequential=False)
def empty_folder(user_email: str, folder: str) -> str:
    """
    Empty the Trash or Spam folder in the user's Gmail account.

    Args:
        user_email (str): The email address of the user.
        folder (str): The folder to clear ('trash' or 'spam').

    Returns:
        str: A JSON string with the result or an error message.
    """
    creds = utilities.authenticate_user()
    service = build("gmail", "v1", credentials=creds)

    try:
        if folder.lower() == "trash":
            service.users().messages().delete(userId=user_email, id="trash").execute()
        elif folder.lower() == "spam":
            service.users().messages().delete(userId=user_email, id="spam").execute()
        else:
            raise ValueError("Invalid folder name. Choose 'trash' or 'spam'.")

        return json.dumps({"result": f"The {folder} folder has been emptied."}, separators=(",", ":"))
    except HttpError as error:
        return json.dumps({"error": f"An error occurred: {error}"}, separators=(",", ":"))
    except ValueError as ve:
        return json.dumps({"error": str(ve)}, separators=(",", ":"))


# Example usage
if __name__ == "__main__":
    user_email = "ibarz.jean@gmail.com"  # Replace with a valid email address
    folder = "trash"  # Choose between 'trash' or 'spam'
    print(empty_folder(user_email=user_email, folder=folder))
