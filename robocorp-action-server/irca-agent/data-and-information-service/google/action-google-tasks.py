"""
An action for interacting with Google Tasks.

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

from google import utilities


@action(is_consequential=False)
def list_tasklists() -> str:
    """
    Retrieve all task lists from the user's Google Tasks.

    Returns:
        str: A JSON string of the retrieved task lists.
    """
    creds = utilities.authenticate_user()
    service = build("tasks", "v1", credentials=creds)

    try:
        result = service.tasklists().list().execute()
        tasklists = result.get("items", [])
        return json.dumps(tasklists, separators=(",", ":"))
    except Exception as error:
        return f"An error occurred: {error}"


@action(is_consequential=False)
def list_tasks(tasklist_id: str) -> str:
    """
    Retrieve all tasks from a specific Google Task list.

    Args:
        tasklist_id (str): The ID of the task list to retrieve tasks from.

    Returns:
        str: A JSON string of the retrieved tasks.
    """
    creds = utilities.authenticate_user()
    service = build("tasks", "v1", credentials=creds)

    try:
        result = service.tasks().list(tasklist=tasklist_id).execute()
        tasks = result.get("items", [])
        return json.dumps(tasks, separators=(",", ":"))
    except Exception as error:
        return f"An error occurred: {error}"


@action(is_consequential=False)
def create_task(tasklist_id: str, title: str, notes: str = "", due_date: str = "") -> str:
    """
    Create a new task in a specific Google Task list.

    Args:
        tasklist_id (str): The ID of the task list where the task will be added.
        title (str): The title of the task.
        notes (str): Additional notes for the task. Defaults to an empty string.
        due_date (str): The due date for the task in RFC3339 format (e.g., '2021-12-31T00:00:00Z'). Defaults to an empty string.

    Returns:
        str: A JSON string of the created task.
    """
    creds = utilities.authenticate_user()
    service = build("tasks", "v1", credentials=creds)

    task = {"title": title}
    if notes:
        task["notes"] = notes
    if due_date:
        task["due"] = due_date

    try:
        created_task = service.tasks().insert(tasklist=tasklist_id, body=task).execute()
        return json.dumps(created_task, separators=(",", ":"))
    except Exception as error:
        return f"An error occurred: {error}"


@action(is_consequential=False)
def update_task(tasklist_id: str, task_id: str, title: str = "", notes: str = "", due_date: str = "") -> str:
    """
    Update an existing task in a specific Google Task list.

    Args:
        tasklist_id (str): The ID of the task list containing the task.
        task_id (str): The ID of the task to be updated.
        title (str): The new title for the task. If empty, the title is not updated.
        notes (str): Additional notes for the task. If empty, the notes are not updated.
        due_date (str): The new due date for the task in RFC3339 format. If empty, the due date is not updated.

    Returns:
        str: A JSON string of the updated task.
    """
    creds = utilities.authenticate_user()
    service = build("tasks", "v1", credentials=creds)

    task = {}
    if title:
        task["title"] = title
    if notes:
        task["notes"] = notes
    if due_date:
        task["due"] = due_date

    try:
        updated_task = service.tasks().update(tasklist=tasklist_id, task=task_id, body=task).execute()
        return json.dumps(updated_task, separators=(",", ":"))
    except Exception as error:
        return f"An error occurred: {error}"


@action(is_consequential=False)
def delete_task(tasklist_id: str, task_id: str) -> str:
    """
    Delete a task from a specific Google Task list.

    Args:
        tasklist_id (str): The ID of the task list containing the task.
        task_id (str): The ID of the task to be deleted.

    Returns:
        str: A confirmation message of the deleted task.
    """
    creds = utilities.authenticate_user()
    service = build("tasks", "v1", credentials=creds)

    try:
        service.tasks().delete(tasklist=tasklist_id, task=task_id).execute()
        return json.dumps({"message": f"Task with ID {task_id} deleted successfully."}, separators=(",", ":"))
    except Exception as error:
        return f"An error occurred: {error}"


@action(is_consequential=False)
def create_tasklist(title: str) -> str:
    """
    Create a new task list in Google Tasks.

    Args:
        title (str): The title of the new task list.

    Returns:
        str: A JSON string of the created task list.
    """
    creds = utilities.authenticate_user()
    service = build("tasks", "v1", credentials=creds)

    tasklist = {"title": title}

    try:
        created_tasklist = service.tasklists().insert(body=tasklist).execute()
        return json.dumps(created_tasklist, separators=(",", ":"))
    except Exception as error:
        return f"An error occurred: {error}"


@action(is_consequential=False)
def delete_tasklist(tasklist_id: str) -> str:
    """
    Delete a task list from Google Tasks.

    Args:
        tasklist_id (str): The ID of the task list to be deleted.

    Returns:
        str: A confirmation message of the deleted task list.
    """
    creds = utilities.authenticate_user()
    service = build("tasks", "v1", credentials=creds)

    try:
        service.tasklists().delete(tasklist=tasklist_id).execute()
        return json.dumps({"message": f"Task list with ID {tasklist_id} deleted successfully."}, separators=(",", ":"))
    except Exception as error:
        return f"An error occurred: {error}"


# # Example usage
if __name__ == "__main__":
    print("Task Lists:")
    print(list_tasklists())
#     # # After obtaining a valid tasklist ID, use it to list tasks
#     # tasklist_id_example = "MDM4MDY0MTc0NTQ5MTQ5NzkwMjI6MDow"  # Replace with a valid tasklist ID
#     # print("Tasks in List:")
#     # print(list_tasks(tasklist_id_example))

#     # print("Creating a new tasklist")
#     # print(create_tasklist("Ma Nouvelle Liste de Tâches"))
