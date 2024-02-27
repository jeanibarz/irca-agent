"""
An action for interacting with Google Contacts.

https://developers.google.com/people/quickstart/python

To implement the recommended approach, which is the standard web-based OAuth 2.0 flow, you'll need to set up your application in the Google Cloud Console and then modify your Python script to handle the OAuth process. This approach is more robust and suitable for both development and production environments.

### Setting up OAuth 2.0 Credentials in Google Cloud Console:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Select your project or create a new one.
3. Navigate to "APIs & Services" > "Credentials."
4. Click on “Create Credentials” and select “OAuth client ID.”
5. Configure the OAuth consent screen if prompted.
6. For the Application type, select "Web application."
7. Under "Authorized redirect URIs," add URI `http://localhost:8080/`.
8. Save and take note of the generated client ID and client secret.

### First-Time Authentication in Development Container
- As the development container lacks a browser, initial authentication should be done using the following steps:
  1. Execute `gcloud auth login --no-launch-browser`.
  2. A command will be displayed. Copy and paste this command into your host computer that has browser access.
  3. After completing the authentication in the browser, copy the authentication code.
  4. Paste this code back into the console of the development container.
"""

import json

from googleapiclient.discovery import build
from robocorp.actions import action
from fuzzywuzzy import process

from google import utilities


@action(is_consequential=False)
def list_contacts() -> str:
    """
    Retrieve a list of contacts from the user's Google Contacts, handling pagination.

    Returns:
        str: A JSON string of the retrieved contacts.
    """
    creds = utilities.authenticate_user()
    service = build("people", "v1", credentials=creds)

    try:
        contacts = []
        request = (
            service.people()
            .connections()
            .list(resourceName="people/me", pageSize=100, personFields="names,emailAddresses,phoneNumbers")
        )

        while request is not None:
            response = request.execute()
            contacts.extend(response.get("connections", []))
            request = service.people().connections().list_next(previous_request=request, previous_response=response)

        return json.dumps(contacts, separators=(",", ":"))
    except Exception as error:
        return f"An error occurred: {error}"


@action(is_consequential=False)
def create_contact(display_name: str, email: str = "", phone_number: str = "") -> str:
    """
    Create a new contact in the user's Google Contacts.

    Args:
        display_name (str): The display name of the contact.
        email (str, optional): The email address of the contact. Defaults to an empty string.
        phone_number (str, optional): The phone number of the contact. Defaults to an empty string.

    Returns:
        str: A JSON string of the created contact.
    """
    creds = utilities.authenticate_user()
    service = build("people", "v1", credentials=creds)

    contact_body = {
        "names": [{"displayName": display_name}],
        "emailAddresses": [{"value": email}] if email else [],
        "phoneNumbers": [{"value": phone_number}] if phone_number else [],
    }

    try:
        created_contact = service.people().createContact(body=contact_body).execute()
        return json.dumps(created_contact, separators=(",", ":"))
    except Exception as error:
        return f"An error occurred: {error}"


@action(is_consequential=False)
def fuzzy_search_contacts(search_term: str, threshold: int = 80) -> str:
    """
    Executes a fuzzy search in Google Contacts to find matches by scoring each contact field.
    Evaluates contacts based on displayName, familyName, givenName, email, and phone number.
    Contacts with any field scoring at or above the threshold are selected as matches.

    Args:
        search_term (str): The term to search for in the contact's details.
        threshold (int): The similarity threshold for a match to be considered relevant (default 80).

    Returns:
        str: A JSON string representing the list of contacts that match the search term.
             Each match includes the contact's details and a dictionary of scores for each evaluated field.
    """
    contacts_json = list_contacts()
    contacts = json.loads(contacts_json)

    matches = []
    for contact in contacts:
        # Extract individual fields
        displayName = contact.get("names", [{}])[0].get("displayName", "")
        familyName = contact.get("familyName", [{}])[0].get("displayName", "")
        givenName = contact.get("givenName", [{}])[0].get("displayName", "")
        email = contact.get("emailAddresses", [{}])[0].get("value", "")
        phone = contact.get("phoneNumbers", [{}])[0].get("value", "")

        # Calculate scores for each field
        name_score = process.extractOne(query=search_term, choices=[displayName], score_cutoff=threshold)
        family_name_score = process.extractOne(query=search_term, choices=[familyName], score_cutoff=threshold)
        given_name_score = process.extractOne(query=search_term, choices=[givenName], score_cutoff=threshold)
        email_score = process.extractOne(query=search_term, choices=[email], score_cutoff=threshold)
        phone_score = process.extractOne(query=search_term, choices=[phone], score_cutoff=threshold)

        match = {
            "contact": contact,
            "scores": {
                "name": name_score[1] if name_score else 0,
                "familyName": family_name_score[1] if family_name_score else 0,
                "givenName": given_name_score[1] if given_name_score else 0,
                "email": email_score[1] if email_score else 0,
                "phone": phone_score[1] if phone_score else 0,
            },
        }
        # Check if any of the scores meet the threshold
        if any(score >= threshold for score in match["scores"].values()):
            matches.append(match)

    return json.dumps(matches, separators=(",", ":"))


@action(is_consequential=False)
def update_contact(
    resource_name: str, new_display_name: str = "", new_email: str = "", new_phone_number: str = ""
) -> str:
    """
    Update an existing contact in the user's Google Contacts.

    Args:
        resource_name (str): The resource name of the contact to be updated.
        new_display_name (str): The new display name for the contact. If empty, the display name is not updated.
        new_email (str): The new email address for the contact. If empty, the email address is not updated.
        new_phone_number (str): The new phone number for the contact. If empty, the phone number is not updated.

    Returns:
        str: A JSON string of the updated contact.
    """
    creds = utilities.authenticate_user()
    service = build("people", "v1", credentials=creds)

    # First, get the current state of the contact including its etag
    try:
        existing_contact = service.people().get(resourceName=resource_name, personFields="metadata").execute()
    except Exception as error:
        return f"Error fetching contact details: {error}"

    # Build the update payload with the existing etag
    contact_update = {"etag": existing_contact["etag"]}
    update_fields = []
    if new_display_name != "":
        contact_update["names"] = [{"displayName": new_display_name}]
        update_fields.append("names")
    if new_email != "":
        contact_update["emailAddresses"] = [{"value": new_email}]
        update_fields.append("emailAddresses")
    if new_phone_number != "":
        contact_update["phoneNumbers"] = [{"value": new_phone_number}]
        update_fields.append("phoneNumbers")

    try:
        updated_contact = (
            service.people()
            .updateContact(resourceName=resource_name, updatePersonFields=",".join(update_fields), body=contact_update)
            .execute()
        )
        return json.dumps(updated_contact, separators=(",", ":"))
    except Exception as error:
        return f"An error occurred: {error}"


if __name__ == "__main__":
    # search_query = "Sylvia"
    # print(fuzzy_search_contacts(search_query))

    # new_contact = create_contact("John Doe", "john.doe@example.com", "123-456-7890")
    # print(new_contact)

    # Example usage
    contact_resource_name = "people/c3140990601898064338"  # Replace with actual resource name of the contact
    updated_contact = update_contact(
        contact_resource_name, new_display_name="Jane Doe", new_email="jane.doe@example.com"
    )
    print(updated_contact)
