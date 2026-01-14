#!/bin/bash
set -e

echo "🚀 Setting up IRCA-Agent development environment..."

# Navigate to the workspace directory
cd /workspace

# Ensure Poetry is in PATH
export PATH="/opt/poetry/bin:$PATH"

# Install all dependencies including dev
echo "📦 Installing dependencies with Poetry..."
poetry install --with dev

# Install llama-cpp-python with CUDA support (optional)
if [ "${INSTALL_LLAMA_CPP:-false}" = "true" ]; then
    echo "🦙 Installing llama-cpp-python with CUDA support..."
    CMAKE_ARGS="-DGGML_CUDA=on" poetry run pip install llama-cpp-python --force-reinstall --upgrade
fi

# Activate virtual environment in .bashrc for convenience
echo 'source /workspace/.venv/bin/activate 2>/dev/null || true' >> ~/.bashrc
echo 'export PYTHONPATH="/workspace/src:$PYTHONPATH"' >> ~/.bashrc

# Set up pre-commit hooks if available
if command -v pre-commit &> /dev/null; then
    echo "🔧 Setting up pre-commit hooks..."
    poetry run pre-commit install || true
fi

# Create necessary directories
mkdir -p models datasets

# Copy .env if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Created .env from .env.example - please update with your tokens"
fi

echo ""
echo "✅ Dev Container successfully set up!"
echo ""
echo "Next steps:"
echo "  1. Update .env with your API tokens"
echo "  2. Run 'poetry shell' to activate the environment"
echo "  3. Run 'pytest' to verify everything works"
echo ""