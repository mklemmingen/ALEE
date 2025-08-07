#!/bin/bash

# Optimized ROCm Setup Script for Manjaro Linux (2024)
# Educational Question Generation AI with 20GB AMD GPU
# Based on comprehensive multi-LLM research

set -e

echo "Setting up optimized ROCm environment for educational AI..."

# First verify GPU hardware
echo "Detecting AMD GPU hardware..."
lspci | grep VGA
GPU_INFO=$(lspci | grep VGA | grep AMD || echo "No AMD GPU detected")
echo "GPU Info: $GPU_INFO"

# Update system
echo "Updating system packages..."
sudo pacman -Syu --noconfirm

# Install core ROCm packages from official repositories
echo "Installing optimized ROCm packages..."
sudo pacman -S --noconfirm \
    rocm-hip-sdk \
    rocm-opencl-sdk \
    clblast \

yay -S rocm-smi
yay -S opencl-amd

# Install Ollama with ROCm support (primary recommendation)
echo "Installing Ollama with ROCm support..."
sudo pacman -S --noconfirm ollama-rocm

# Add user to GPU groups
echo "Adding user to GPU groups..."
sudo gpasswd -a $USER render
sudo gpasswd -a $USER video

# Set optimized environment variables
echo "Setting up optimized environment variables..."
cat >> ~/.bashrc << 'EOF'
# ROCm Environment Variables - Optimized for Multi-LLM

export ROCM_PATH=/opt/rocm
export PATH=$PATH:$ROCM_PATH/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROCM_PATH/lib
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

# GPU Architecture override (adjust based on your GPU)
# For RX 7000 series:
export HSA_OVERRIDE_GFX_VERSION=11.0.0
# For RX 6000 series, use: export HSA_OVERRIDE_GFX_VERSION=10.3.0
# For older cards, use: export HSA_OVERRIDE_GFX_VERSION=9.0.0

# Ollama optimizations
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_FLASH_ATTENTION=1
EOF

# System optimizations for large model inference
echo "Applying system optimizations..."
sudo tee -a /etc/sysctl.conf << 'EOF'

# Memory optimizations for large LLMs
vm.max_map_count=2097152
vm.swappiness=10
vm.dirty_ratio=5
vm.dirty_background_ratio=2
EOF

# GPU performance optimization script
echo "Creating GPU optimization script..."
cat > optimize_gpu.sh << 'EOF'
#!/bin/bash
# Set compute mode for optimal LLM performance
sudo sh -c 'echo "manual" > /sys/class/drm/card0/device/power_dpm_force_performance_level'
sudo sh -c 'echo "5" > /sys/class/drm/card0/device/pp_power_profile_mode'  # COMPUTE mode

# Set memory clocks to maximum
sudo sh -c 'echo "1" > /sys/class/drm/card0/device/pp_dpm_mclk'

echo "GPU optimized for compute workloads"
rocm-smi --showtemp
EOF
chmod +x optimize_gpu.sh

# Install Python dependencies for the orchestration system
echo "Installing Python dependencies..."
pip3 install fastapi uvicorn aiohttp pydantic
pip3 install ollama-python
pip3 install python-multipart

# Create Ollama systemd service
echo "Setting up Ollama systemd service..."
sudo tee /etc/systemd/system/ollama.service > /dev/null << 'EOF'
[Unit]
Description=Ollama Server for Educational AI
After=network-online.target

[Service]
ExecStart=/usr/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_ORIGINS=*"
Environment="OLLAMA_NUM_PARALLEL=2"
Environment="OLLAMA_MAX_LOADED_MODELS=2"
Environment="ROCM_PATH=/opt/rocm"

[Install]
WantedBy=default.target
EOF

# Create ollama user
sudo useradd -r -s /bin/false -m -d /usr/share/ollama ollama 2>/dev/null || true

# Install monitoring tools
echo "Installing monitoring tools..."
sudo pacman -S --noconfirm nvtop btop

# Create model download script
echo "Creating model download script..."
cat > download_models.sh << 'EOF'
#!/bin/bash
# Download optimized models for educational question generation

echo "Downloading optimized models for 20GB VRAM setup..."

# Primary models (Q4_K_M quantization for best balance)
ollama pull llama3.1:8b      # Main generator (~5.5GB)
ollama pull mistral:7b       # Validator (~5.0GB) 
ollama pull qwen2.5:7b       # Math/logic expert (~5.0GB)

# Specialized models for parameter validation
ollama pull llama3.2:3b      # Lightweight parameter expert (~2.5GB)

echo "Models downloaded successfully!"
echo "Total VRAM usage with 2 concurrent models: ~10-11GB"
echo "Remaining VRAM for KV cache and operations: ~9-10GB"

# List installed models
ollama list
EOF
chmod +x download_models.sh

# Create startup validation script
echo "Creating validation script..."
cat > validate_setup.sh << 'EOF'
#!/bin/bash
echo "Validating ROCm setup..."

# Check ROCm installation
if command -v rocm-smi &> /dev/null; then
    echo "ROCm installed successfully"
    rocm-smi
else
    echo "ROCm not found"
    exit 1
fi

# Check Ollama
if command -v ollama &> /dev/null; then
    echo "Ollama installed successfully"
    ollama --version
else
    echo "Ollama not found"
    exit 1
fi

# Check GPU detection
echo "Testing GPU detection..."
rocm-smi | grep "GPU" && echo "GPU detected" || echo "GPU not detected"

# Check OpenCL
if command -v clinfo &> /dev/null; then
    echo "OpenCL devices:"
    clinfo -l
fi

echo "Setup validation complete!"
EOF
chmod +x validate_setup.sh

# Apply sysctl changes
sudo sysctl -p

echo ""
echo "ROCm setup complete!"
echo ""
echo "Next steps:"
echo "1. REBOOT REQUIRED: sudo reboot"
echo "2. After reboot, run: ./validate_setup.sh"
echo "3. Optimize GPU: ./optimize_gpu.sh"
echo "3.5 Start the Ollama server in the background: ollama serve | or enable the systemd service: sudo systemctl enable --now ollama"
echo "4. Download models: ./download_models.sh"
echo "5. Start Ollama service: sudo systemctl enable --now ollama"
echo "6. Start Ollama server: ./start_ollama_servers.sh"
echo ""
echo "Architecture Summary:"
echo "   - Primary: Ollama with ROCm acceleration"
echo "   - Memory: 2 concurrent models max (~10-11GB VRAM)"
echo "   - API: OpenAI-compatible endpoints"
echo "   - Models: Q4_K_M quantized for optimal balance"
echo ""
echo "Important: Verify your GPU architecture and adjust HSA_OVERRIDE_GFX_VERSION if needed"