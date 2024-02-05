#!/bin/bash

# Navigate to the workspace directory
cd /workspace

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install llama-cpp-python (build with cuda)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

# Install dependencies
pip install -r requirements.txt
pip install pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context guidance

echo "Dev Container successfully set up !"