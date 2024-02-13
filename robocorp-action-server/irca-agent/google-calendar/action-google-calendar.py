"""
A action for interacting with Google Calendar

https://developers.google.com/calendar/api/quickstart/python?hl=fr

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
"""

import datetime
import os.path
import json

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from robocorp.actions import action

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
REDIRECT_URI = "http://localhost:8080/"

MAX_RESULTS = 2500  # Maximum value as per Google Calendar API documentation


@action(is_consequential=False)
def get_calendar_events(
    calendar_id: str = "primary",
    time_min: str = "",
    time_max: str = "",
    max_results: int = 2500,
    time_zone: str = "",
    query: str = "",
    show_deleted: bool = False,
    single_events: bool = True,
    order_by: str = "startTime",
    fields: str = "items(id,summary,description,start,end)",
) -> str:
    """
    Retrieve events from a specified Google Calendar with various filtering and sorting options.

    Args:
        calendar_id (str): The ID of the calendar. Defaults to 'primary'.
        time_min (str): The minimum time for events (inclusive) in RFC3339 format. Example: "2024-02-01T00:00:00Z"
        time_max (str): The maximum time for events (exclusive) in RFC3339 format. Example: "2024-03-01T00:00:00Z"
        max_results (int): The maximum number of events to retrieve. Defaults to 2500.
        time_zone (str): The time zone used in the response. Defaults to the calendar's time zone.
        query (str): Free text search terms to find events matching the query.
        show_deleted (bool): Whether to include deleted events. Defaults to False.
        single_events (bool): Whether to expand recurring events into individual events. Defaults to True.
        order_by (str): The order of the events returned. Valid values are 'startTime' or 'updated'.
        fields (str): A comma-separated list specifying a subset of fields to include in the response.
                      Use the syntax "items(field1,field2)" to include specific fields.
                      For example, "items(summary,start)" returns only the summary and start time of each event.
                      Refer to the Google Calendar API documentation for a full list of available fields.
                      Link: https://developers.google.com/calendar/api/v3/reference/events?hl=fr#resource

    Returns:
        str: A JSON string of the retrieved calendar events.
    """
    creds = authenticate_user()
    service = build("calendar", "v3", credentials=creds)

    try:
        events_result = (
            service.events()
            .list(
                calendarId=calendar_id,
                timeMin=time_min if time_min else None,
                timeMax=time_max if time_max else None,
                maxResults=max_results,
                timeZone=time_zone if time_zone else None,
                q=query,
                singleEvents=single_events,
                orderBy=order_by if order_by in ["startTime", "updated"] else None,
                showDeleted=show_deleted,
                fields=fields if fields else None,
            )
            .execute()
        )
        return json.dumps(events_result, separators=(",", ":"))
    except HttpError as error:
        return f"An error occurred: {error}"


@action(is_consequential=False)
def get_events_for_today(calendar_id: str = "primary") -> str:
    """
    Retrieve events for the current day from a specified Google Calendar.

    Args:
        calendar_id (str): The ID of the calendar. Defaults to 'primary'.

    Returns:
        str: A JSON string of the retrieved calendar events for today.
    """
    today_start = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + datetime.timedelta(days=1)

    return get_calendar_events(
        calendar_id=calendar_id, time_min=today_start.isoformat() + "Z", time_max=today_end.isoformat() + "Z"
    )


def authenticate_user():
    """Authenticate the user and return credentials."""
    creds = None

    # Save the current working directory
    original_cwd = os.getcwd()

    try:
        # Change directory to the current file's directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES, redirect_uri=REDIRECT_URI)
                creds = flow.run_local_server(port=8080)
            with open("token.json", "w") as token:
                token.write(creds.to_json())
    finally:
        # Restore the original working directory
        os.chdir(original_cwd)

    return creds


if __name__ == "__main__":
    # events = get_calendar_events("primary", 10, "2024-02-01T00:00:00Z", "2024-03-01T00:00:00Z")
    events = get_events_for_today("primary")
    print(events)
