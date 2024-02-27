# import json
# from googleapiclient.errors import HttpError
# from robocorp.actions import action
# from googleapiclient.discovery import build
# from google import utilities


# @action(is_consequential=False)
# def create_filter(user_email: str, criteria: dict, action: dict) -> str:
#     """
#     Create a filter (rule) for incoming emails in the user's Gmail account.

#     Args:
#         user_email (str): The email address of the user.
#         criteria (dict): The criteria to filter emails by (e.g., {'from': 'example@example.com'}).
#         action (dict): The action to apply to filtered emails (e.g., {'addLabelIds': ['Label_1'], 'removeLabelIds': ['INBOX']}).

#     Returns:
#         str: A JSON string with the result or an error message.
#     """
#     creds = utilities.authenticate_user()
#     service = build("gmail", "v1", credentials=creds)

#     try:
#         filter = {"criteria": criteria, "action": action}
#         created_filter = service.users().settings().filters().create(userId=user_email, body=filter).execute()
#         return json.dumps(
#             {"result": f"Filter created successfully with ID: {created_filter['id']}."}, separators=(",", ":")
#         )
#     except HttpError as error:
#         return json.dumps({"error": f"An error occurred: {error}"}, separators=(",", ":"))


# # Example usage
# if __name__ == "__main__":
#     user_email = "ibarz.jean@gmail.com"  # Replace with a valid email address
#     criteria = {"from": "example@example.com"}  # Define your criteria
#     action = {"addLabelIds": ["Label_1"], "removeLabelIds": ["INBOX"]}  # Define your action
#     print(create_filter(user_email=user_email, criteria=criteria, action=action))
