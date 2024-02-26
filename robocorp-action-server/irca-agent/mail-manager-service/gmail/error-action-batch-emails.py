# import json
# from googleapiclient.errors import HttpError
# from robocorp.actions import action
# from googleapiclient.discovery import build
# from . import utilities


# @action(is_consequential=False)
# def batch_email_actions(user_email: str, query: str, actions: dict) -> str:
#     """
#     Perform batch actions (like delete, archive, label) on emails based on a search query.

#     Args:
#         user_email (str): The email address of the user.
#         query (str): The search query to filter emails.
#         actions (dict): The actions to apply (e.g., {'addLabelIds': ['Label_1'], 'removeLabelIds': ['INBOX']}).

#     Returns:
#         str: A JSON string with the result or an error message.
#     """
#     creds = utilities.authenticate_user()
#     service = build("gmail", "v1", credentials=creds)

#     try:
#         # Fetch emails based on the query
#         response = service.users().messages().list(userId=user_email, q=query).execute()
#         messages = response.get("messages", [])

#         # Apply actions to each email
#         for msg in messages:
#             service.users().messages().modify(userId=user_email, id=msg["id"], body=actions).execute()

#         return json.dumps({"result": f"Batch actions applied to {len(messages)} email(s)."}, separators=(",", ":"))
#     except HttpError as error:
#         return json.dumps({"error": f"An error occurred: {error}"}, separators=(",", ":"))


# # Example usage
# if __name__ == "__main__":
#     user_email = "ibarz.jean@gmail.com"  # Replace with a valid email address
#     query = "is:unread"  # Define your search query
#     actions = {"removeLabelIds": ["UNREAD"], "addLabelIds": ["Label_1"]}  # Define your actions
#     print(batch_email_actions(user_email=user_email, query=query, actions=actions))
