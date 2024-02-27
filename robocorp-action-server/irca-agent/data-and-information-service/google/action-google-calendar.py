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

### First-Time Authentication in Development Container
- As the development container lacks a browser, initial authentication should be done using the following steps:
  1. Execute `gcloud auth login --no-launch-browser`.
  2. A command will be displayed. Copy and paste this command into your host computer that has browser access.
  3. After completing the authentication in the browser, copy the authentication code.
  4. Paste this code back into the console of the development container.
"""

import json

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from robocorp.actions import action

from google import utilities

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
    creds = utilities.authenticate_user()
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
def create_calendar_event(
    summary: str,
    start_time: str,
    end_time: str,
    calendar_id: str = "primary",
    description: str = "",
    location: str = "",
    attendees: str = "",
    reminders: str = "",
    recurrence: str = "",
) -> str:
    """
    Create a new event in a specified Google Calendar.

    Args:
        calendar_id (str): The ID of the calendar to create the event in. Defaults to 'primary'.
        summary (str): The summary or title of the event.
        start_time (str): The start time of the event in RFC3339 format.
        end_time (str): The end time of the event in RFC3339 format.
        description (str, optional): The description of the event. Defaults to an empty string.
        location (str, optional): The location of the event. Defaults to an empty string.
        attendees (str, optional): A string of email addresses for the attendees, separated by commas. Defaults to an empty string.
        reminders (str, optional): A JSON string specifying reminder settings. Defaults to an empty string.
        recurrence (str, optional): Recurrence rule for the event. Defaults to an empty string.

    Returns:
        str: A JSON string of the created calendar event.
    """
    creds = utilities.authenticate_user()
    service = build("calendar", "v3", credentials=creds)

    # Convert the attendees string to a list of dictionaries
    attendees_list = [{"email": email.strip()} for email in attendees.split(",") if email] if attendees else []

    # Convert the reminders string to a dictionary
    reminders_dict = json.loads(reminders) if reminders else {"useDefault": True}

    event = {
        "summary": summary,
        "location": location,
        "description": description,
        "start": {"dateTime": start_time, "timeZone": "UTC"},
        "end": {"dateTime": end_time, "timeZone": "UTC"},
        "attendees": attendees_list,
        "reminders": reminders_dict,
    }

    # Add recurrence if provided
    if recurrence:
        event["recurrence"] = [recurrence]

    try:
        created_event = service.events().insert(calendarId=calendar_id, body=event).execute()
        return json.dumps(created_event, separators=(",", ":"))
    except HttpError as error:
        return f"An error occurred: {error}"


@action(is_consequential=False)
def update_calendar_event(
    event_id: str,
    calendar_id: str,
    summary: str = "",
    start_time: str = "",
    end_time: str = "",
    description: str = "",
    location: str = "",
    attendees: str = "",
    reminders: str = "",
    recurrence: str = "",
) -> str:
    """
    Update an existing event in a specified Google Calendar.

    Args:
        event_id (str): The ID of the event to be updated.
        calendar_id (str): The ID of the calendar containing the event.
        summary (str): The summary or title of the event. Use an empty string for no update.
        start_time (str): The start time of the event in RFC3339 format. Use an empty string for no update.
        end_time (str): The end time of the event in RFC3339 format. Use an empty string for no update.
        description (str): The description of the event. Use an empty string for no update.
        location (str): The location of the event. Use an empty string for no update.
        attendees (str): A string of email addresses for the attendees, separated by commas. Use an empty string for no update.
        reminders (str): A JSON string specifying reminder settings. Use an empty string for no update.
        recurrence (str): Recurrence rule for the event. Use an empty string for no update.

    Returns:
        str: A JSON string of the updated calendar event.
    """
    creds = utilities.authenticate_user()
    service = build("calendar", "v3", credentials=creds)

    # Retrieve the existing event
    event = service.events().get(calendarId=calendar_id, eventId=event_id).execute()

    # Update event fields based on provided arguments
    if summary != "":
        event["summary"] = summary
    if start_time != "":
        event["start"] = {"dateTime": start_time, "timeZone": "UTC"}
    if end_time != "":
        event["end"] = {"dateTime": end_time, "timeZone": "UTC"}
    if description != "":
        event["description"] = description
    if location != "":
        event["location"] = location
    if attendees != "":
        event["attendees"] = [{"email": email.strip()} for email in attendees.split(",") if email]
    if reminders != "":
        event["reminders"] = json.loads(reminders)
    if recurrence != "":
        event["recurrence"] = [recurrence]

    try:
        updated_event = service.events().update(calendarId=calendar_id, eventId=event_id, body=event).execute()
        return json.dumps(updated_event, separators=(",", ":"))
    except HttpError as error:
        return f"An error occurred: {error}"


# @action(is_consequential=False)
# def get_events_for_today(calendar_id: str = "primary") -> str:
#     """
#     Retrieve events for the current day from a specified Google Calendar.

#     Args:
#         calendar_id (str): The ID of the calendar. Defaults to 'primary'.

#     Returns:
#         str: A JSON string of the retrieved calendar events for today.
#     """
#     today_start = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
#     today_end = today_start + datetime.timedelta(days=1)

#     return get_calendar_events(
#         calendar_id=calendar_id, time_min=today_start.isoformat() + "Z", time_max=today_end.isoformat() + "Z"
#     )


@action(is_consequential=False)
def remove_event(calendar_id: str, event_id: str) -> str:
    """
    Remove an event from a specified Google Calendar.

    Args:
        calendar_id (str): The ID of the calendar containing the event.
        event_id (str): The ID of the event to be removed.

    Returns:
        str: A confirmation message indicating the result of the operation.
    """
    creds = utilities.authenticate_user()
    service = build("calendar", "v3", credentials=creds)

    try:
        service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
        return f"Event with ID {event_id} has been successfully deleted from calendar {calendar_id}."
    except HttpError as error:
        return f"An error occurred: {error}"
