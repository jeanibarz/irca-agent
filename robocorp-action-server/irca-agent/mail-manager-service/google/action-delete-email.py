# # action_delete_email.py

# from googleapiclient.errors import HttpError
# from robocorp.actions import action
# from googleapiclient.discovery import build

# from google import utilities


# @action(is_consequential=False)
# def delete_email(user_email: str, email_id: str) -> str:
#     """
#     Delete an email from the user's Gmail account.

#     Args:
#         user_email (str): The email address of the user.
#         email_id (str): The ID of the email to be deleted.

#     Returns:
#         str: Confirmation message or an error message.
#     """
#     creds = utilities.authenticate_user()
#     service = build("gmail", "v1", credentials=creds)

#     try:
#         service.users().messages().delete(userId=user_email, id=email_id).execute()
#         return f"Email with ID {email_id} deleted successfully."
#     except HttpError as error:
#         return f"An error occurred: {error}"


# # Example usage
# if __name__ == "__main__":
#     user_email = "ibarz.jean@gmail.com"  # Replace with a valid email address
#     email_id = "email-id-to-delete"  # Replace with the ID of the email to delete
#     print(delete_email(user_email=user_email, email_id=email_id))
