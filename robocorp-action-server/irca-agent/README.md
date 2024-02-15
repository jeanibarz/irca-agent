# Action Server

### Running the Action Server

To start the action server, execute the following command:
```
action-server start
```

## Google Cloud Authentication

### Generating OAuth 2.0 Credentials
- Generate OAuth 2.0 user credentials and download the JSON file.

### First-Time Authentication in Development Container
- As the development container lacks a browser, initial authentication should be done using the following steps:
  1. Execute `gcloud auth login --no-launch-browser`.
  2. A command will be displayed. Copy and paste this command into your host computer that has browser access.
  3. After completing the authentication in the browser, copy the authentication code.
  4. Paste this code back into the console of the development container.
```