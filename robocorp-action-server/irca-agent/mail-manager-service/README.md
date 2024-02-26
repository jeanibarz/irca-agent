For interacting with Gmail.

https://developers.google.com/gmail/api/quickstart/python

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

### Implemented Gmail Actions

- **Send Email**: Action to send an email directly from the user's Gmail account.
- **Delete Email**: Action to delete emails based on specific email ID.
- **Download Attachments**: Action to download attachments from emails.
- **Batch Email Actions**: Action to perform operations like delete, archive, or label on multiple emails based on a query.
- **Create Draft**: Action to create a draft email in the user's Gmail account.
- **Create Filter**: Action to create filters/rules for incoming emails.
- **Get Summary**: Action to get a summary status of the mailbox (like unread emails count).
- **Flag Email**: Action to flag or unstar an email.
- **Empty Folder**: Action to clear out the trash or spam folders.
- **Mark Email**: Action to mark an email as read or unread.
- **Move Email**: Action to move an email from one label/folder to another.
- **Retrieve Email Headers**: Action to fetch detailed headers of a specific email.
- **Retrieve Email Threads**: Action to fetch all emails from a conversation thread.
- **List Drafts**: Action to list all draft emails.
- **List Emails**: Action to list emails with filtering options.
- **Search Emails**: Action to search for emails based on various criteria.


### Other potential relevant actions

- **Schedule Send Email**: Implementing an action to schedule an email to be sent at a later time or date.
- **Read Receipt Request**: An action to send an email with a request for a read receipt.
- **Export Emails**: An action to export emails to a file format like PDF or plain text.
- **Import Emails**: An action to import emails from a file, useful for backups or migrations.
- **Generate Email Summary Report**: An action to generate a summary report of email activity over a specified period.
- **Manage Email Aliases**: An action to manage email aliases associated with the Gmail account.
- **Email Auto-responder**: Action to set up or modify an auto-responder for the Gmail account.
- **Automated Responses Based on Email Content**: Parsing email content and sending automated responses based on certain criteria or keywords.