#!/bin/bash

# Start the Action Server on the specified port
action-server start --expose --expose-allow-reuse --address 0.0.0.0

# Use 'tail -f /dev/null' to keep the script running indefinitely
# tail -f /dev/null
