# Use an official Python runtime with CUDA as a parent image
FROM huggingface/transformers-pytorch-gpu

ENV TZ=Europe/France
ENV DEBIAN_FRONTEND noninteractive

# Switch to the root user for system setup
USER root

# Install depencencies
RUN python3 -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context guidance

# Install llama-cpp-python (build with cuda)
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

# Set the working directory in the container
WORKDIR /workspaces

CMD ["/bin/bash"]
