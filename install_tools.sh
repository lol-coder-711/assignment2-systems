#!/bin/bash
set -e

echo "Installing uv..."
pip install uv

echo "Installing dependencies..."
uv sync

echo "Installing nsys..."
apt update
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt update
apt install -y nsight-systems-cli

echo "Configuring git..."
git config --global user.name "joll"
git config --global user.email "joll"

source .venv/bin/activate