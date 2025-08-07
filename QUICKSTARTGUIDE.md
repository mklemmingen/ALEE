# Quick Start Guide - Educational AI System

This guide provides step-by-step instructions to quickly start, test, and stop the Educational AI Parameter-Expert LLM System.

## Starting the System

### Step 1: Start OLLAMA Servers
Start all 7 expert LLM servers on ports 8001-8007:
```bash
./start_ollama_servers.sh
```

**What this does:**
- Starts variation_expert (llama3.1:8b) on port 8001
- Starts taxonomy_expert (mistral:7b) on port 8002
- Starts math_expert (qwen2.5:7b) on port 8003
- Starts text_reference_expert (llama3.2:3b) on port 8004
- Starts obstacle_expert (llama3.2:3b) on port 8005
- Starts instruction_expert (mistral:7b) on port 8006
- Starts content_expert (llama3.1:8b) on port 8007

**Expected output:**
```
Starting Educational AI Orchestrator Ollama Servers...
Server on port 8001 is running
Model llama3.1:8b loaded on port 8001
[... continues for all servers ...]
All ollama servers started successfully!
```

### Step 2: Start ALEE_Agent (Main Orchestrator)
In a **new terminal**, start the main FastAPI orchestrator:
```bash
python3 ALEE_Agent/educational_ai_orchestrator.py
```

**Alternative with uvicorn:**
```bash
uvicorn ALEE_Agent.educational_ai_orchestrator:app --host 0.0.0.0 --port 8000
```

**Expected output:**
```
INFO: Started server process
INFO: Waiting for application startup.
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Step 3: Verify System Health
In a **third terminal**, run the comprehensive test suite:
```bash
python3 CallersWithTexts/test_system.py
```

**Expected output:**
```
INFO:__main__:Testing health endpoint...
INFO:__main__:Health check passed: healthy
INFO:__main__:Testing model status endpoint...
INFO:__main__:Model status check passed
[... continues with all tests ...]
```

## Access Points

Once running, you can access:

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Model Status**: http://localhost:8000/models/status

## Stopping the System

### Quick Stop - All Services
Kill everything at once:
```bash
pkill -f "ollama serve"
pkill -f "educational_ai_orchestrator"
```

### Individual Components

#### Stop OLLAMA Servers Only
```bash
pkill -f "ollama serve"
```

#### Stop ALEE_Agent Only
- **If running in terminal**: Press `Ctrl+C`
- **If running as background process**: 
```bash
pkill -f "educational_ai_orchestrator"
```

#### Graceful Shutdown
1. Stop ALEE_Agent first (`Ctrl+C` in its terminal)
2. Stop OLLAMA servers: `pkill -f "ollama serve"`

## 📊 Monitoring & Troubleshooting

### Check Running Processes
```bash
# Check what's running
ps aux | grep -E "(ollama|educational)"

# Check specific ports
netstat -tlnp | grep -E "(800[0-7])"

# Monitor GPU usage
watch -n 1 rocm-smi
```

### System Status
```bash
# Quick health check
curl http://localhost:8000/health

# Model status and VRAM usage
curl http://localhost:8000/models/status
```

### Common Issues

#### OLLAMA Servers Won't Start
```bash
# Check if ports are already in use
netstat -tlnp | grep -E "(800[0-7])"

# Kill any existing ollama processes
pkill -f "ollama serve"

# Wait a moment, then restart
sleep 5
./start_ollama_servers.sh
```

#### ALEE_Agent Can't Connect
```bash
# Verify OLLAMA servers are running
curl -s http://localhost:8001/api/tags

# Check if orchestrator port is available
netstat -tlnp | grep 8000
```

#### High VRAM Usage
```bash
# Monitor memory usage in real-time
watch -n 1 'rocm-smi && echo "---" && curl -s http://localhost:8000/models/status'
```

## Testing Workflow

### 1. Basic Health Test
```bash
curl http://localhost:8000/health
```

### 2. Generate a Test Question
```bash
curl -X POST http://localhost:8000/generate-question \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Bedürfnisse und Güter",
    "difficulty": "leicht",
    "age_group": "9. Klasse"
  }'
```

### 3. Run Full Test Suite
```bash
python3 CallersWithTexts/test_system.py
```

## Result Management

Test results are automatically saved in timestamped folders:
```
CallersWithTexts/results/
└── YYYY-MM-DD_HH-MM-SS/
    ├── prompts/        # Prompts used
    └── results/        # Generated CSV
```

To create a new result session:
```bash
python3 -c "from CallersWithTexts.result_manager import ResultManager; rm = ResultManager(); print(rm.create_timestamp_folder())"
```

## Restart Procedure

If you need to restart the system:

1. **Stop everything:**
```bash
pkill -f "ollama serve"
pkill -f "educational_ai_orchestrator"
```

2. **Wait for processes to terminate:**
```bash
sleep 5
```

3. **Start fresh:**
```bash
./start_ollama_servers.sh
# Wait for servers to start, then in new terminal:
python3 ALEE_Agent/educational_ai_orchestrator.py
```

## Pro Tips

- **Keep terminals organized**: Use 3 terminals (OLLAMA servers, ALEE_Agent, testing)
- **Monitor VRAM**: Run `watch -n 1 rocm-smi` in a fourth terminal
- **Save logs**: Redirect output to files for debugging
- **Background mode**: Add `&` to run servers in background, but logs become harder to monitor

## Emergency Stop

If the system becomes unresponsive:
```bash
# Nuclear option - stop everything AI-related
pkill -f "ollama"
pkill -f "educational"
pkill -f "python.*AI"

# Check if anything is still running
ps aux | grep -i -E "(ollama|educational|ai)"
```

---