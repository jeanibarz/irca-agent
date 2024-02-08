# Use an official Python runtime with CUDA as a parent image
FROM huggingface/transformers-pytorch-gpu

ENV TZ=Europe/Paris
ENV DEBIAN_FRONTEND noninteractive

# Set the working directory in the container
WORKDIR /workspace

# Install python3-venv before creating the virtual environment
RUN apt-get update && apt-get install -y python3-venv sudo \
    && rm -rf /var/lib/apt/lists/*

# # Create a vscode group and user
# RUN groupadd vscode && \
#     useradd -rm -d /home/vscode -s /bin/bash -g vscode -G sudo -u 1000 vscode

# # Grant vscode user passwordless sudo privileges
# RUN echo "vscode ALL=(ALL) NOPASSWD: ALL" | tee /etc/sudoers.d/vscode

# Create a virtual environment and activate it
# RUN python3 -m venv .venv
# ENV PATH="/workspace/.venv/bin:$PATH"

# # Install depencencies
# RUN pip install --upgrade pip \
#     && pip install pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context guidance

# # Install llama-cpp-python (build with cuda)
# RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

# # Change ownership of the /workspace and other necessary directories to vscode user
# RUN chown -R vscode:vscode /workspace

# # Append virtual environment activation command to .bashrc
# RUN echo "source /workspace/.venv/bin/activate" >> /home/vscode/.bashrc

# # Switch to vscode user
# USER vscode

CMD ["/bin/bash"]
