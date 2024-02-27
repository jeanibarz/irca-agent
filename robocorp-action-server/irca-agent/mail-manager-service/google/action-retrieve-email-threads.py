import json
from googleapiclient.errors import HttpError
from robocorp.actions import action
from googleapiclient.discovery import build
from google import utilities


@action(is_consequential=False)
def retrieve_email_threads(user_email: str, thread_id: str) -> str:
    """
    Retrieve all emails from a conversation thread in the user's Gmail account.

    Args:
        user_email (str): The email address of the user.
        thread_id (str): The ID of the email thread to retrieve.

    Returns:
        str: A JSON string with the email thread details or an error message.
    """
    creds = utilities.authenticate_user()
    service = build("gmail", "v1", credentials=creds)

    try:
        thread = service.users().threads().get(userId=user_email, id=thread_id).execute()
        messages = thread.get("messages", [])
        thread_details = [{"id": msg["id"], "snippet": msg.get("snippet", "")} for msg in messages]

        return json.dumps({"thread_details": thread_details}, separators=(",", ":"))
    except HttpError as error:
        return json.dumps({"error": f"An error occurred: {error}"}, separators=(",", ":"))


# Example usage
if __name__ == "__main__":
    user_email = "ibarz.jean@gmail.com"  # Replace with a valid email address
    thread_id = "thread-id-to-retrieve"  # Replace with the ID of the thread
    print(retrieve_email_threads(user_email=user_email, thread_id=thread_id))
