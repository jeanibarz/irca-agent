# Use an official Python runtime with CUDA as a parent image
FROM huggingface/transformers-pytorch-gpu

ENV TZ=Europe/Paris
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    python3-venv \
    apt-transport-https \
    ca-certificates \
    gnupg \
    curl \
    sudo \
    tree \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VIRTUALENVS_CREATE=true
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV POETRY_NO_INTERACTION=1
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy dependency files first (for better caching)
COPY pyproject.toml ./

# Install dependencies (without dev dependencies for smaller image)
# Dev dependencies will be installed in post-create
RUN poetry install --no-root --only main || true

CMD ["/bin/bash"]

