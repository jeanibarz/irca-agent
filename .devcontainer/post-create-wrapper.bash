#!/bin/bash

# Navigate to the workspace directory
cd /workspace

# Create a virtual environment
python3 -m venv /venv

# Activate the virtual environment
source /venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install llama-cpp-python (build with cuda)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

# Install dependencies
pip install -r requirements.txt
pip install pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context guidance

# Download Robocorp Action Server
curl -o action-server https://downloads.robocorp.com/action-server/releases/latest/linux64/action-server
chmod a+x action-server

# Add to PATH or move to a folder that is in PATH
sudo mv action-server /usr/local/bin/

# To create an action-server, run action-server new

echo "Dev Container successfully set up !"