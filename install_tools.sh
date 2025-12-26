#!/bin/bash
set -e

install_python_deps() {
    echo "Starting Python dependencies installation..."
    pip install uv
    uv sync
    uv pip install seaborn
    echo "Python dependencies installed."
}

install_system_deps() {
    echo "Starting system dependencies installation (nsys)..."
    apt update
    apt install -y --no-install-recommends gnupg
    echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    apt update
    apt install -y nsight-systems-cli
    echo "System dependencies installed."
}

# Run in parallel
install_python_deps &
PYTHON_PID=$!

install_system_deps &
SYSTEM_PID=$!

wait $PYTHON_PID
wait $SYSTEM_PID

echo "Configuring git..."
git config --global user.name "joll"
git config --global user.email "joll"