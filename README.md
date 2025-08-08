# Multimodal Educational AI System for Parameter-Specific Question Generation through multiple Expertised data-backed LLM refinement iterations

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![ROCm](https://img.shields.io/badge/ROCm-6.2+-red.svg)
![Ollama](https://img.shields.io/badge/Ollama-Latest-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![SYSARCH](https://img.shields.io/badge/SYSARCH-Compliant-brightgreen.svg)

Educational AI system that uses multilayered small language model bouncing to generate expert-refined questions 
(currently set to 3 per Request)

The system follows a three-layered server architecture (Caller -> Generator -> Experts) defined in SYSARCH.md 
with modular prompt construction from ALEE Tübingen defined educational question parameters
and intelligent CSV conversion fallbacks.

## Three-Layer Architecture

```mermaid
graph TD
    A["1. CALLER<br/>HTTP Request with SYSARCH Parameters<br/>c_id, text, p.variation, p.taxonomy_level, etc."] --> B["2. ORCHESTRATOR<br/>Educational AI Orchestrator"]
    
    B --> B1["Health Check Expert Servers<br/>Ports 8001-8007"]
    B1 --> B2["Clean Expert Sessions<br/>(Clear history, no restarts)"]
    B2 --> B3["Modular Prompt Construction<br/>Parameter → TXT File Mapping"]
    
    B3 --> B3A["p.variation → variationPrompts/*.txt"]
    B3 --> B3B["p.taxonomy_level → taxonomyLevelPrompt/*.txt"]  
    B3 --> B3C["p.mathematical_requirement_level → mathematicalRequirementLevel/*.txt"]
    B3 --> B3D["p.*_obstacle_* → itemXObstacle/ + instructionObstacle/"]
    B3 --> B3E["p.root_text_* → rootTextParameterTextPrompts/"]
    
    B3A --> B4["Build Master Generation Prompt"]
    B3B --> B4
    B3C --> B4
    B3D --> B4
    B3E --> B4
    
    B4 --> B5["Main Generator LLM<br/>Generate Exactly 3 Questions"]
    B5 --> B6["Question 1, Question 2, Question 3"]
    
    B6 --> C["3. EXPERT LLMs<br/>Sequential Parameter Validation"]
    
    C --> C1["For Each Question (1,2,3):<br/>Send to All Expert Validators"]
    
    C1 --> D["Variation Expert<br/>Port 8001<br/>p.variation validation"]
    C1 --> E["Taxonomy Expert<br/>Port 8002<br/>p.taxonomy_level validation"] 
    C1 --> F["Math Expert<br/>Port 8003<br/>p.mathematical_requirement_level"]
    C1 --> G["Obstacle Expert<br/>Port 8004<br/>p.*_obstacle_* validation"]
    C1 --> H["Instruction Expert<br/>Port 8005<br/>p.instruction_* validation"]
    C1 --> I["Content Expert<br/>Port 8006<br/>p.root_text_contains_irrelevant_info"]
    C1 --> J["Reference Expert<br/>Port 8007<br/>p.root_text_reference_explanatory_text"]
    
    D --> K{"Expert Consensus Check<br/>Rating 1-5 + Suggestions<br/>(Max 3 iterations per SYSARCH)"}
    E --> K
    F --> K
    G --> K
    H --> K
    I --> K
    J --> K
    
    K -->|"Rating < 3<br/>Need Refinement"| L["Apply Expert Suggestions<br/>Regenerate Question"]
    L --> C1
    K -->|"All Experts Rating ≥ 3<br/>OR Max 3 Iterations Reached"| M["Build SYSARCH CSV Format"]
    
    M --> M1["Convert to Required CSV Columns:<br/>c_id, subject, type, text, p.instruction_*, p.item_*, etc."]
    M1 --> M2["Intelligent CSV Fallbacks<br/>LM-Assisted Format Correction"]
    M2 --> M3["Save Results with result_manager<br/>ISO timestamp + prompts snapshot"]
    
    M3 --> N["Return HTTP Response:<br/>question_1, question_2, question_3<br/>+ complete CSV data"]
    
    style A fill:#e3f2fd
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style N fill:#e8f5e8
    style K fill:#ffecb3
    style B3 fill:#f0f8ff
    style M fill:#f5f5dc
```

### Expert LLM Specialists

| Expert | Focus Parameters | Model | Expertise |
|--------|------------------|-------|-----------|
| **Variation Expert** | `p.variation` | llama3.1:8b | Difficulty level assessment (leicht/stammaufgabe/schwer) |
| **Taxonomy Expert** | `p.taxonomy_level` | mistral:7b | Bloom's taxonomy classification (Stufe 1/2) |
| **Math Expert** | `p.mathematical_requirement_level` | qwen2.5:7b | Mathematical complexity analysis (0-2 scale) |
| **Obstacle Expert** | `p.*_obstacle_*` | llama3.2:3b | Linguistic barriers (passive, negation, complex NP) |
| **Instruction Expert** | `p.instruction_*` | mistral:7b | Clarity and explicitness analysis |
| **Content Expert** | `p.root_text_contains_irrelevant_information` | llama3.1:8b | Content relevance validation |
| **Reference Expert** | `p.root_text_reference_explanatory_text` | llama3.2:3b | Text reference analysis |

## Quick Start

### Prerequisites (as tested on, similiar capable or better Hardware and Software configurations possible!)

- **Hardware**: AMD GPU with 20GB VRAM (RX 6000/7000 series recommended)
- **OS**: Manjaro Linux (Arch-based)
- **Python**: 3.8+
- **ROCm**: 6.2+ compatible

### Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd educational-ai-system
   ```

2. **Install ROCm and Dependencies**
   ```bash
   chmod +x setup_rocm_ManjArch_AMD7kSeries.sh
   ./setup_rocm_ManjArch_AMD7kSeries.sh
   
   # Reboot required for ROCm
   sudo reboot
   ```

3. **Verify Installation**
   ```bash
   # Check GPU detection
   rocm-smi
   
   # Optimize GPU for compute workloads
   ./optimize_gpu.sh
   
   # Validate complete setup
   ./validate_setup.sh
   ```

4. **Download AI Models**
   ```bash
   # Download optimized models for 20GB VRAM
   ./download_models.sh
   ```

5. **Start the System**
   ```bash
   # Start Ollama servers first
   ./start_ollama_servers.sh
   
   # Then start the main orchestrator
   python3 ALEE_Agent/educational_ai_orchestrator.py
   ```

6. **Verify System Health**
   ```bash
   # Run comprehensive tests
   python3 CallersWithTexts/test_system.py
   
   # Access API documentation
   # http://localhost:8000/docs
   ```

## Technical Specifications

### Memory Management

```python
MEMORY_CONFIGURATION = {
    "total_vram": "20GB",
    "max_concurrent_models": 2,
    "memory_buffer": "2GB",
    "model_swapping": "dynamic",
    "vram_monitoring": "real-time"
}

MODEL_MEMORY_USAGE = {
    "llama3.1NutzenMathematischerDarstellungen:8b": "5.5GB (Q4_K_M)",
    "mistral:7b": "5.0GB (Q4_K_M)", 
    "qwen2.5:7b": "5.0GB (Q4_K_M)",
    "llama3.2:3b": "2.5GB (Q4_K_M)"
}
```

### Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Concurrent Models** | Max 2 | Memory-optimized via semaphore |
| **VRAM Efficiency** | ~55% utilization | 11GB usage, 9GB for operations |
| **Processing Speed** | 15-25 tokens/sec | Per active model |
| **Parameter Validation** | 7+ experts | Specialized domain knowledge |
| **Iteration Limit** | 3 cycles | SYSARCH-specified limit |
| **Question Output** | Exactly 3 | Per SYSARCH requirements |
| **CSV Compliance** | 100% | All SYSARCH columns implemented |
| **Parameter Coverage** | 16 parameters | Complete SYSARCH implementation |

## API Reference

### Core Endpoints

#### Question Generation (PRIMARY)
```http
POST /generate-validation-plan
Content-Type: application/json

{
  "c_id": "41-1-4",
  "text": "Die Inflation beschreibt einen allgemeinen Anstieg des Preisniveaus...",
  "p_variation": "stammaufgabe",
  "p_taxonomy_level": "Stufe 1 (Wissen/Reproduktion)",
  "p_mathematical_requirement_level": "0",
  "p_root_text_obstacle_passive": "Nicht Enthalten",
  "p_root_text_obstacle_negation": "Nicht Enthalten",
  "p_root_text_obstacle_complex_np": "Nicht Enthalten",
  "p_root_text_contains_irrelevant_information": "Nicht Enthalten",
  "p_item_X_obstacle_passive": "Nicht Enthalten",
  "p_item_X_obstacle_negation": "Nicht Enthalten",
  "p_item_X_obstacle_complex_np": "Nicht Enthalten",
  "p_instruction_obstacle_passive": "Nicht Enthalten",
  "p_instruction_obstacle_complex_np": "Nicht Enthalten",
  "p_instruction_explicitness_of_instruction": "Implizit"
}
```

**Response:**
```json
{
  "question_1": "First generated question text...",
  "question_2": "Second generated question text...",
  "question_3": "Third generated question text...",
  "c_id": "41-1-4",
  "processing_time": 23.7,
  "csv_data": {
    "c_id": "41-1-4",
    "subject": "stammaufgabe",
    "type": "multiple-choice",
    "text": "1. Question1 2. Question2 3. Question3",
    "p.instruction_explicitness_of_instruction": "Implizit",
    "p.mathematical_requirement_level": "0 (Kein Bezug)",
    "p.taxanomy_level": "Stufe 1 (Wissen/Reproduktion)",
    "p.variation": "stammaufgabe",
    "answers": "Extracted from questions with LM assistance"
  }
}
```

**Response:**
```json
{
  "question_content": {
    "aufgabenstellung": "...",
    "antwortoptionen": ["A", "B", "C", "D"],
    "korrekte_antwort": "A"
  },
  "parameter_validations": [
    {
      "parameter": "p.variation",
      "status": "approved",
      "score": 8.5,
      "expert_used": "variation_expert"
    }
  ],
  "iterations": 2,
  "total_processing_time": 12.3,
  "final_status": "approved",
  "csv_ready": {...}
}
```

#### System Monitoring
```http
GET /health
GET /models/status
```

## Project Structure

```
/
├── _dev/                                   # Development files
│   ├── README_DEPLOYMENT.md               # System documentation
│   ├── _old/                              # Legacy files
│   └── providedProjectFromStakeHolder/    # Stakeholder data
│       └── explanation_metadata.csv       # Real educational texts (16 entries)
├── ALEE_Agent/                            # Main AI system
│   ├── educational_ai_orchestrator.py     # FastAPI server (SYSARCH.md compliant, see that doc for details to layered system)
│   ├── variationPrompts/                  # Difficulty-specific prompts
│   ├── taxonomyLevelPrompt/               # Cognitive level prompts
│   ├── mathematicalRequirementLevel/      # Math complexity prompts
│   ├── rootTextParameterTextPrompts/      # Text analysis prompts
│   ├── itemXObstacle/                     # Item obstacle prompts
│   ├── instructionObstacle/               # Instruction obstacle prompts
│   └── instructionExplicitnessOfInstruction/ # Instruction explicitness prompts
├── CallersWithTexts/                      # Testing & results
│   ├── stakeholder_test_system.py         # Comprehensive stakeholder testing
│   ├── test_system.py                     # Basic system tests
│   ├── result_manager.py                  # Result organization
│   └── results/                           # Timestamped outputs
│       └── YYYY-MM-DD_HH-MM-SS/          # Session folders
│           ├── prompts/                   # Snapshot of prompts used
│           ├── results.csv                # Generated test results
│           └── session_metadata.json     # Session metadata & statistics
└── *.sh                                   # Setup scripts
```

### Python SDK Example

```python
import asyncio
import aiohttp

class ValidationPlanClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    async def generate_validation_plan_questions(self, c_id, text, p_variation, 
                                               p_taxonomy_level, **kwargs):
        """Generate 3 questions according to SYSARCH specifications"""
        async with aiohttp.ClientSession() as session:
            request = {
                "c_id": c_id,
                "text": text,
                "p_variation": p_variation,
                "p_taxonomy_level": p_taxonomy_level,
                "p_mathematical_requirement_level": kwargs.get("p_mathematical_requirement_level", "0"),
                "p_root_text_reference_explanatory_text": kwargs.get("p_root_text_reference_explanatory_text", "Nicht vorhanden"),
                "p_root_text_obstacle_passive": kwargs.get("p_root_text_obstacle_passive", "Nicht Enthalten"),
                "p_root_text_obstacle_negation": kwargs.get("p_root_text_obstacle_negation", "Nicht Enthalten"),
                "p_root_text_obstacle_complex_np": kwargs.get("p_root_text_obstacle_complex_np", "Nicht Enthalten"),
                "p_root_text_contains_irrelevant_information": kwargs.get("p_root_text_contains_irrelevant_information", "Nicht Enthalten"),
                "p_item_X_obstacle_passive": kwargs.get("p_item_X_obstacle_passive", "Nicht Enthalten"),
                "p_item_X_obstacle_negation": kwargs.get("p_item_X_obstacle_negation", "Nicht Enthalten"),
                "p_item_X_obstacle_complex_np": kwargs.get("p_item_X_obstacle_complex_np", "Nicht Enthalten"),
                "p_instruction_obstacle_passive": kwargs.get("p_instruction_obstacle_passive", "Nicht Enthalten"),
                "p_instruction_obstacle_complex_np": kwargs.get("p_instruction_obstacle_complex_np", "Nicht Enthalten"),
                "p_instruction_explicitness_of_instruction": kwargs.get("p_instruction_explicitness_of_instruction", "Implizit")
            }
            
            async with session.post(
                f"{self.base_url}/generate-validation-plan", 
                json=request,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                return await response.json()
    
    async def convert_to_csv(self, result, save_path=None):
        """Convert SYSARCH result to CSV format"""
        csv_data = result.get('csv_data', {})
        
        if save_path:
            import csv
            with open(save_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_data.keys())
                writer.writeheader()
                writer.writerow(csv_data)
        
        return csv_data
    
    async def get_system_status(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                health = await response.json()
            
            async with session.get(f"{self.base_url}/models/status") as response:
                models = await response.json()
                
            return {"health": health, "models": models}

# SYSARCH Usage Example
async def main():
    client = ValidationPlanClient()
    
    # Generate SYSARCH-compliant questions
    result = await client.generate_validation_plan_questions(
        c_id="42-2-3",
        text="Die Marktwirtschaft funktioniert über Angebot und Nachfrage. Preise entstehen durch das Zusammenspiel von Anbietern und Nachfragern auf dem Markt.",
        p_variation="leicht",
        p_taxonomy_level="Stufe 2 (Anwendung/Transfer)",
        p_mathematical_requirement_level="1",
        p_root_text_obstacle_passive="Enthalten"
    )
    
    print(f"Generated {len([q for q in ['question_1', 'question_2', 'question_3'] if result.get(q)])} questions")
    print(f"Processing time: {result['processing_time']:.2f}s")
    print(f"C_ID: {result['c_id']}")
    print(f"Question 1: {result['question_1'][:100]}...")
    print(f"Question 2: {result['question_2'][:100]}...")
    print(f"Question 3: {result['question_3'][:100]}...")
    
    # Convert to CSV and save
    csv_data = await client.convert_to_csv(result, "questions_output.csv")
    print(f"CSV columns: {len(csv_data)} - {list(csv_data.keys())[:5]}...")
    
    # Check system status
    status = await client.get_system_status()
    print(f"VRAM usage: {status['models']['vram_usage_gb']:.1f}GB")

# Run
asyncio.run(main())
```

## Parameter Validation System

### Parameter Implementation

The system implements stakeholder (ALEE) defined parameters to perform modular at request prompt construction.
The prompts are stored in ALEE_agent subfolders that mimic the parameter structure defined in SYSARCH.md.
Each txt features an explanation as well as human-oversight tested data to refine LM pattern and expertise without overloading
a token limit of smaller models.

#### Request Parameters
- `c_id` - Question ID format: question_number-difficulty-version (e.g., "41-1-4")
- `text` - Informational text about the system's pre-configured topic  
- `p_variation` - Difficulty: "Stammaufgabe", "schwer", "leicht"
- `p_taxonomy_level` - "Stufe 1 (Wissen/Reproduktion)", "Stufe 2 (Anwendung/Transfer)"
- `p_mathematical_requirement_level` - "0", "1", "2" with descriptions

#### Root Text Parameters
- `p_root_text_reference_explanatory_text` - "Nicht vorhanden", "Explizit", "Implizit"
- `p_root_text_obstacle_passive` - "Enthalten", "Nicht Enthalten" 
- `p_root_text_obstacle_negation` - "Enthalten", "Nicht Enthalten"
- `p_root_text_obstacle_complex_np` - "Enthalten", "Nicht Enthalten"
- `p_root_text_contains_irrelevant_information` - "Enthalten", "Nicht Enthalten"

#### Item-X Parameters
- `p_item_X_obstacle_passive` - "Enthalten", "Nicht Enthalten"
- `p_item_X_obstacle_negation` - "Enthalten", "Nicht Enthalten"
- `p_item_X_obstacle_complex_np` - "Enthalten", "Nicht Enthalten"

#### Instruction Parameters
- `p_instruction_obstacle_passive` - "Enthalten", "Nicht Enthalten"
- `p_instruction_obstacle_complex_np` - "Enthalten", "Nicht Enthalten"
- `p_instruction_explicitness_of_instruction` - "Explizit", "Implizit"

#### CSV Output Columns
The system generates a CSV as:
- `c_id`, `subject`, `type`, `text`, `answers`
- `p.instruction_explicitness_of_instruction` through `p.instruction_number_of_sentences`
- `p.item_1_answer_verbatim_explanatory_text` through `p.item_8_sentence_length`
- `p.mathematical_requirement_level`, `p.taxanomy_level`, `p.variation`
- Plus all obstacle and reference parameters as specified

## Expert LLM Prompts in detail

### Variation Expert
Specializes in difficulty assessment using cognitive load theory and educational psychology principles.

**Key Evaluation Criteria:**
- Cognitive demand analysis
- Complexity level assessment  
- Context familiarity evaluation
- Distractor quality analysis

### Taxonomy Expert  
Focuses on Bloom's taxonomy classification and learning objective alignment.

**Key Evaluation Criteria:**
- Cognitive operation identification
- Transfer requirement analysis
- Situational novelty assessment
- Multi-step thinking evaluation

### Math Expert
Analyzes mathematical complexity and quantitative requirements.

**Key Evaluation Criteria:**
- Calculation difficulty assessment
- Mathematical representation usage
- Age-appropriate complexity
- Economic integration quality

### Obstacle Expert
Evaluates linguistic barriers and text accessibility.

**Key Evaluation Criteria:**
- Passive voice detection
- Negation complexity analysis
- Noun phrase structure evaluation
- Cumulative cognitive load assessment

## System Reliability Features

### Robust JSON Parsing
The system includes advanced error handling for LLM responses that may contain malformed JSON:

**Key Features:**
- **Markdown Extraction**: Automatically removes ````json` code blocks from responses
- **Boundary Detection**: Finds JSON objects even in mixed content responses  
- **Feedback Normalization**: Converts dict/list feedback to strings for consistency
- **Fallback Defaults**: Provides safe defaults when parsing fails completely
- **Pydantic Compatibility**: Prevents serialization warnings in FastAPI

**Implementation:** `parse_expert_response()` function in `educational_ai_orchestrator.py:29-101`

### Result Management System
Modular system for organizing test results and documentation:

**Key Features:**
- **Timestamped Sessions**: ISO format timestamps for result organization
- **Prompt Snapshots**: Automatic backup of prompts used for each session
- **Multiple CSV Formats**: Handles various input formats (list of dicts, files, strings)
- **Session Metadata**: Comprehensive statistics and processing information

**Implementation:** `result_manager.py` in `CallersWithTexts/` directory

## Configuration

### Model Configuration
```python
# ALEE_Agent/educational_ai_orchestrator.py
PARAMETER_EXPERTS = {
    "variation_expert": ParameterExpertConfig(
        name="variation_expert",
        model="llama3.1NutzenMathematischerDarstellungen:8b",
        port=8001,
        parameters=["p.variation"],
        expertise="Difficulty level assessment",
        temperature=0.2
    ),
    # ... additional experts
}
```

### Memory Management
```python
class ModelManager:
    def __init__(self):
        self.model_memory_usage = {
            "llama3.1NutzenMathematischerDarstellungen:8b": 5.5,    # GB
            "mistral:7b": 5.0,
            "qwen2.5:7b": 5.0,
            "llama3.2:3b": 2.5
        }
        self.max_vram_gb = 18  # Leave 2GB buffer
```

### Environment Variables
```bash
# ROCm Configuration
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # RX 7000 series
export HIP_VISIBLE_DEVICES=0

# Ollama Optimization
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_FLASH_ATTENTION=1
```

## Testing

### Stakeholder Data Testing System

The system includes comprehensive stakeholder testing using real educational data:

```bash
# Run stakeholder data tests with all 16 texts from explanation_metadata.csv
# Each text tested with 5 randomized parameter combinations (80 total tests)
python3 CallersWithTexts/stakeholder_test_system.py

# Features:
# - Uses all texts from _dev/providedProjectFromStakeHolder/explanation_metadata.csv
# - 5 randomized parameter combinations per text (80 total tests)
# - Incremental saving after each text completion
# - Complete SYSARCH parameter coverage
# - Real production simulation with timestamped results
```

### Stakeholder Test Coverage

**Comprehensive Stakeholder Testing (`stakeholder_test_system.py`):**
- **Real Educational Data**: Uses all 16 texts from `explanation_metadata.csv`
- **80 Total Tests**: 16 texts × 5 randomized parameter combinations each
- **Incremental Saving**: Results saved after each text completion (simulates production)
- **Complete Parameter Randomization**: All 16+ SYSARCH parameters randomized
- **Production Simulation**: Real-world testing environment with full result tracking
- **CSV Export**: Complete results with parameters, questions, and metadata
- **Timestamped Results**: Organized in `CallersWithTexts/results/YYYY-MM-DD_HH-MM-SS/`

## Troubleshooting

### Common Issues

#### ROCm Not Detected
```bash
# Check GPU architecture
lspci | grep VGA

# Set correct GFX version
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # RX 7000
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # RX 6000
```

#### Models Not Loading
```bash
# Check Ollama service
sudo systemctl status ollama

# Verify model downloads
ollama list

# Test model manually
ollama run llama3.1NutzenMathematischerDarstellungen:8b "Hello"
```

#### VRAM Overflow
```bash
# Monitor memory usage
watch -n 1 'rocm-smi && echo "---" && curl -s http://localhost:8000/models/status'

# Reduce concurrent models
# Edit ALEE_Agent/educational_ai_orchestrator.py: model_semaphore = asyncio.Semaphore(1NutzenMathematischerDarstellungen)
```

#### Slow Response Times
- Verify GPU compute mode: `./optimize_gpu.sh`
- Check model quantization: Ensure Q4_K_M variants
- Monitor system resources: `btop`
- Review network connectivity to Ollama

### Debug Mode
```bash
# Run with verbose logging
export PYTHONPATH=.
python3 -m uvicorn ALEE_Agent.educational_ai_orchestrator:app --log-level debug

# Enable detailed ROCm logging
export ROCM_DEBUG=1
export HIP_DEBUG=1
```

### Model Information
- **Ollama Models**: [https://ollama.com/library](https://ollama.com/library)
- **ROCm Documentation**: [https://rocm.docs.amd.com/](https://rocm.docs.amd.com/)
- **FastAPI Documentation**: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Usage Examples

### Stakeholder Testing (Production Simulation)

```bash
# Start system components
./start_ollama_servers.sh
python3 ALEE_Agent/educational_ai_orchestrator.py &

# Run comprehensive stakeholder tests
# - Uses all 16 texts from explanation_metadata.csv
# - 5 randomized parameter combinations per text (80 total tests)
# - Incremental saves after each text completion
python3 CallersWithTexts/stakeholder_test_system.py

# Results saved to:
# CallersWithTexts/results/2024-08-08_14-23-45/
# ├── results.csv              # All 80 test results
# ├── session_metadata.json    # Test statistics
# └── prompts/                 # Prompt snapshots
```

### Individual Request

```bash
# Single SYSARCH request
curl -X POST http://localhost:8000/generate-validation-plan \
  -H "Content-Type: application/json" \
  -d '{
    "c_id": "181-1-3",
    "text": "Bedürfnisse sind Wünsche Menschen haben...",
    "p_variation": "leicht",
    "p_taxonomy_level": "Stufe 1 (Wissen/Reproduktion)",
    "p_mathematical_requirement_level": "0",
    "p_root_text_obstacle_passive": "Nicht Enthalten",
    "p_instruction_explicitness_of_instruction": "Implizit"
  }'

# Returns:
# {
#   "question_1": "First generated question...",
#   "question_2": "Second generated question...",
#   "question_3": "Third generated question...",
#   "c_id": "181-1-3",
#   "processing_time": 23.4,
#   "csv_data": { ... complete SYSARCH CSV ... }
# }
```

## Acknowledgments

- **AMD ROCm Team** - For excellent GPU compute support
- **Ollama Project** - For simplified local LLM deployment  
- **Hugging Face** - For transformer models and tokenizers
- **FastAPI** - For high-performance async API framework
- **Educational Research Community** - For parameter frameworks and validation methods
- **Stakeholder Data Contributors** - For providing real educational content for testing

---