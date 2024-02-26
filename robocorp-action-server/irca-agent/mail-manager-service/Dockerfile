FROM python:3.10

WORKDIR /app

COPY . /app

# Install Robocorp Action Server
RUN pip install robocorp-action-server

# Copy the start-server script and make it executable
COPY start-server.sh /start-server.sh
RUN chmod +x /start-server.sh

# Expose the default port of the Action Server
EXPOSE 8080

# Use the script as the entry point
CMD ["/start-server.sh"]