#!/bin/bash

# Educational AI Orchestrator - Ollama Server Startup Script
# Starts multiple ollama servers on different ports with specific models

set -e

echo "Starting Educational AI Orchestrator Ollama Servers..."

# Function to start ollama server on specific port with model
start_ollama_server() {
    local port=$1
    local model=$2
    local expert_name=$3
    
    echo "Starting $expert_name ($model) on port $port..."
    
    # Start ollama server in background
    OLLAMA_HOST=0.0.0.0:$port ollama serve &
    local server_pid=$!
    
    # Wait a moment for server to start
    sleep 3
    
    # Test if server is responding
    if curl -s "http://localhost:$port/api/tags" > /dev/null; then
        echo "Server on port $port is running"
        
        # Load the model by making a test request
        echo "Loading model $model..."
        curl -s -X POST "http://localhost:$port/api/generate" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$model\",\"prompt\":\"test\",\"stream\":false}" > /dev/null
        
        echo "Model $model loaded on port $port"
    else
        echo "Failed to start server on port $port"
        kill $server_pid 2>/dev/null || true
        return 1
    fi
    
    return 0
}

# Kill any existing ollama processes
echo "Stopping any existing ollama servers..."
pkill -f "ollama serve" || echo "No existing ollama servers found"
sleep 2

# Start each expert server
echo "Starting expert servers..."

start_ollama_server 8001 "llama3.1:8b" "variation_expert" || exit 1
start_ollama_server 8002 "mistral:7b" "taxonomy_expert" || exit 1  
start_ollama_server 8003 "qwen2.5:7b" "math_expert" || exit 1
start_ollama_server 8004 "llama3.2:3b" "text_reference_expert" || exit 1
start_ollama_server 8005 "llama3.2:3b" "obstacle_expert" || exit 1
start_ollama_server 8006 "mistral:7b" "instruction_expert" || exit 1
start_ollama_server 8007 "llama3.1:8b" "content_expert" || exit 1

echo ""
echo "All ollama servers started successfully!"
echo ""
echo "Running servers:"
echo "  - Port 8001: llama3.1:8b (variation_expert)"
echo "  - Port 8002: mistral:7b (taxonomy_expert)" 
echo "  - Port 8003: qwen2.5:7b (math_expert)"
echo "  - Port 8004: llama3.2:3b (text_reference_expert)"
echo "  - Port 8005: llama3.2:3b (obstacle_expert)"
echo "  - Port 8006: mistral:7b (instruction_expert)"
echo "  - Port 8007: llama3.1:8b (content_expert)"
echo ""
echo "You can now start the main orchestrator:"
echo "  python ALEE_Agent/educational_ai_orchestrator.py"
echo ""
echo "To stop all servers, run:"
echo "  pkill -f 'ollama serve'"

# Keep script running and show process info
echo "Press Ctrl+C to stop all servers..."
trap 'echo "Stopping all ollama servers..."; pkill -f "ollama serve"; exit 0' INT

# Monitor servers
while true; do
    sleep 10
    active_servers=$(pgrep -f "ollama serve" | wc -l)
    echo "$(date): $active_servers ollama servers running"
done