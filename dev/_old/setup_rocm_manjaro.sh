#!/bin/bash

# ROCm Setup Script for Manjaro Linux (2024)
# For AMD GPU with 20GB VRAM - Educational Question Generation AI

set -e

echo "🚀 Setting up ROCm environment on Manjaro Linux for AI workloads..."

# Update system
echo "📦 Updating system packages..."
sudo pacman -Syu --noconfirm

# Install ROCm packages from official repositories
echo "⚡ Installing ROCm packages..."
sudo pacman -S --noconfirm \
    rocm-dev \
    rocm-libs \
    hip-runtime-amd \
    rocm-smi \
    rocblas \
    rocsparse \
    rocrand \
    rocthrust

# Install Ollama with ROCm support
echo "🦙 Installing Ollama with ROCm support..."
sudo pacman -S --noconfirm ollama-rocm

# Install PyTorch with ROCm support from official repos
echo "🔥 Installing PyTorch with ROCm support..."
sudo pacman -S --noconfirm python-pytorch-opt-rocm python-torchvision-opt-rocm python-torchaudio-opt-rocm

# Install additional Python packages via pip
echo "🐍 Installing Python ML packages..."
pip3 install --user transformers accelerate datasets tokenizers
pip3 install --user huggingface_hub bitsandbytes
pip3 install --user vllm-flash-attn

# Alternative PyTorch installation via pip if needed
echo "🔧 Installing latest PyTorch with ROCm 6.2 support..."
pip3 install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Install additional inference frameworks
echo "📚 Installing inference frameworks..."
pip3 install --user ollama-python
pip3 install --user llama-cpp-python[server]

# Set up environment variables
echo "🔧 Setting up environment variables..."
cat >> ~/.bashrc << 'EOF'

# ROCm Environment Variables
export ROCM_PATH=/opt/rocm
export PATH=$PATH:$ROCM_PATH/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROCM_PATH/lib
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=11.0.0
EOF

# Add user to render and video groups
echo "👥 Adding user to GPU groups..."
sudo usermod -a -G render,video $USER

# Verify ROCm installation
echo "🔍 Verifying ROCm installation..."
if command -v rocm-smi &> /dev/null; then
    echo "✅ ROCm installed successfully!"
    rocm-smi
else
    echo "❌ ROCm installation may have failed"
fi

# Test Ollama
echo "🦙 Testing Ollama ROCm support..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama installed successfully!"
    ollama --version
else
    echo "❌ Ollama installation may have failed"
fi

# Create systemd service for Ollama (optional)
echo "⚙️ Creating Ollama systemd service..."
sudo tee /etc/systemd/system/ollama.service > /dev/null << 'EOF'
[Unit]
Description=Ollama Server
After=network-online.target

[Service]
ExecStart=/usr/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_ORIGINS=*"

[Install]
WantedBy=default.target
EOF

# Create ollama user and group
sudo useradd -r -s /bin/false -m -d /usr/share/ollama ollama || true

echo "🎉 ROCm setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Reboot your system: sudo reboot"
echo "2. Test GPU detection: rocm-smi"
echo "3. Download a small model: ollama pull llama3.1:8b"
echo "4. Test inference: ollama run llama3.1:8b"
echo ""
echo "💡 For the parameter expert system:"
echo "   - Each expert LM will use ~4-6GB VRAM"
echo "   - Your 20GB VRAM can handle 3-4 concurrent models"
echo "   - Sequential processing will maximize efficiency"