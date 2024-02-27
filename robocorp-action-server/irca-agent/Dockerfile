FROM python:3.10

# Argument for specifying service directory
ARG SERVICE_DIR

WORKDIR /app

# Install Robocorp Action Server
RUN pip install robocorp-action-server

# Set the PYTHONPATH to include the SERVICE_DIR
ENV PYTHONPATH="${PYTHONPATH}:/app/${SERVICE_DIR}"

# Copy the shared script
COPY start-server.bash /start-server.bash
RUN chmod +x /start-server.bash

# Copy only the specified service code
COPY ./${SERVICE_DIR}/ /app/

# Expose the default port of the Action Server
EXPOSE 8080

# Start the server using the start-server script
CMD ["/start-server.bash"]
