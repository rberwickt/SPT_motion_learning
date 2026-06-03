# Use the official AMD ROCm TensorFlow image as the base
FROM rocm/tensorflow:latest

# Set up working directory inside the container
WORKDIR /workspace

# Inform the dynamic linker where to look for the WSL/ROCm bridge libraries
ENV LD_LIBRARY_PATH=/usr/lib:/usr/lib/wsl/lib:/opt/rocm/lib:$LD_LIBRARY_PATH

# Copy your filtered requirements list into the container
COPY requirements.txt /tmp/requirements.txt

# Install the supporting python packages without touching the pre-installed AMD TensorFlow
RUN pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["/bin/bash"]