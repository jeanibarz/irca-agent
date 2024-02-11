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

# Download and install Robocorp Action Server
curl -o action-server https://downloads.robocorp.com/action-server/releases/latest/linux64/action-server
chmod a+x action-server
sudo mv action-server /usr/local/bin/

# Download and install RCC
curl -o rcc https://downloads.robocorp.com/rcc/releases/latest/linux64/rcc
chmod a+x rcc
sudo mv rcc /usr/local/bin/


# Import Google Cloud public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

# Add gloud CLI distribution URL as package source
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Install google-cloud-cli
sudo apt-get update && sudo apt-get install google-cloud-cli

# To create an action-server, run action-server new

echo "Dev Container successfully set up !"