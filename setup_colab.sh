#!/bin/bash
echo "🔧 Setting up LQG CRT-GPU environment..."

apt-get update -qq
apt-get install -y git build-essential cmake libgmp-dev libboost-all-dev

# CGBN install
if [ ! -d "CGBN" ]; then
  git clone https://github.com/NVlabs/CGBN.git
  mkdir -p include/cgbn
  cp -r CGBN/include/cgbn include/
fi

echo "✅ Environment ready."
nvcc --version
nvidia-smi