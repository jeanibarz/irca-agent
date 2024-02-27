import json
from googleapiclient.errors import HttpError
from robocorp.actions import action
from googleapiclient.discovery import build
from google import utilities


@action(is_consequential=False)
def flag_email(user_email: str, email_id: str, star: bool) -> str:
    """
    Flag or unflag (star/unstar) an email in the user's Gmail account.

    Args:
        user_email (str): The email address of the user.
        email_id (str): The ID of the email to be flagged or unflagged.
        star (bool): True to star the email, False to unstar.

    Returns:
        str: A JSON string with the result or an error message.
    """
    creds = utilities.authenticate_user()
    service = build("gmail", "v1", credentials=creds)

    try:
        # Determine the action based on the 'star' parameter
        body = {"addLabelIds": [], "removeLabelIds": []}
        if star:
            body["addLabelIds"].append("STARRED")
        else:
            body["removeLabelIds"].append("STARRED")

        # Modify the email labels
        service.users().messages().modify(userId=user_email, id=email_id, body=body).execute()
        action_status = "starred" if star else "unstarred"
        return json.dumps({"result": f"Email with ID {email_id} has been {action_status}."}, separators=(",", ":"))
    except HttpError as error:
        return json.dumps({"error": f"An error occurred: {error}"}, separators=(",", ":"))


# Example usage
if __name__ == "__main__":
    user_email = "ibarz.jean@gmail.com"  # Replace with a valid email address
    email_id = "email-id-to-flag"  # Replace with the ID of the email to flag
    star = True  # Set to False to unstar the email
    print(flag_email(user_email=user_email, email_id=email_id, star=star))
