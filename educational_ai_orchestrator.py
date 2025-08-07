"""
Educational Question Generation AI Orchestrator
Parameter-Specific Expert LLM System for Manjaro Linux with AMD GPU

Architecture:
- Main Generator LLM: Creates initial questions
- Parameter Expert LLMs: Validate and refine specific parameters
- Memory Management: Max 2 concurrent models (20GB VRAM optimization)
- Async Processing: Sequential parameter validation with feedback loops
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from enum import Enum
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParameterStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing" 
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REFINEMENT = "needs_refinement"

class QuestionDifficulty(Enum):
    LEICHT = "leicht"
    STAMMAUFGABE = "stammaufgabe"
    SCHWER = "schwer"

@dataclass
class ParameterExpertConfig:
    name: str
    model: str
    port: int
    parameters: List[str]
    expertise: str
    temperature: float = 0.3
    max_tokens: int = 500

@dataclass
class QuestionRequest:
    topic: str
    difficulty: QuestionDifficulty
    age_group: str = "9. Klasse"
    context: Optional[str] = None
    target_parameters: Optional[Dict[str, str]] = None

@dataclass
class ParameterValidation:
    parameter: str
    status: ParameterStatus
    score: float
    feedback: str
    expert_used: str
    processing_time: float

@dataclass
class QuestionResult:
    question_content: Dict[str, Any]
    parameter_validations: List[ParameterValidation]
    iterations: int
    total_processing_time: float
    final_status: str
    csv_ready: Dict[str, str]

# Parameter Expert Configurations
PARAMETER_EXPERTS = {
    "variation_expert": ParameterExpertConfig(
        name="variation_expert",
        model="llama3.1:8b",
        port=8001,
        parameters=["p.variation"],
        expertise="Difficulty level assessment (leicht/Stammaufgabe/schwer)",
        temperature=0.2
    ),
    "taxonomy_expert": ParameterExpertConfig(
        name="taxonomy_expert", 
        model="mistral:7b",
        port=8002,
        parameters=["p.taxanomy_level"],
        expertise="Bloom's taxonomy classification (Stufe 1: Wissen/Reproduktion, Stufe 2: Anwendung/Transfer)",
        temperature=0.2
    ),
    "math_expert": ParameterExpertConfig(
        name="math_expert",
        model="qwen2.5:7b", 
        port=8003,
        parameters=["p.mathematical_requirement_level"],
        expertise="Mathematical requirement assessment (0-2 scale)",
        temperature=0.2
    ),
    "text_reference_expert": ParameterExpertConfig(
        name="text_reference_expert",
        model="llama3.2:3b",
        port=8004, 
        parameters=["p.root_text_reference_explanatory_text"],
        expertise="Reference text analysis (Nicht vorhanden/Explizit/Implizit)",
        temperature=0.2
    ),
    "obstacle_expert": ParameterExpertConfig(
        name="obstacle_expert",
        model="llama3.2:3b",
        port=8005,
        parameters=[
            "p.root_text_obstacle_passive",
            "p.root_text_obstacle_negation", 
            "p.root_text_obstacle_complex_np",
            "p.item_X_obstacle_passive",
            "p.item_X_obstacle_negation",
            "p.item_X_obstacle_complex_np",
            "p.instruction_obstacle_passive",
            "p.instruction_obstacle_complex_np"
        ],
        expertise="Linguistic obstacle detection (passive voice, negation, complex noun phrases)",
        temperature=0.2
    ),
    "instruction_expert": ParameterExpertConfig(
        name="instruction_expert",
        model="mistral:7b",
        port=8006,
        parameters=["p.instruction_explicitness_of_instruction"],
        expertise="Instruction clarity and explicitness analysis",
        temperature=0.2
    ),
    "content_expert": ParameterExpertConfig(
        name="content_expert",
        model="llama3.1:8b", 
        port=8007,
        parameters=["p.root_text_contains_irrelevant_information"],
        expertise="Content relevance and distractor analysis",
        temperature=0.3
    )
}

# Global resources
model_semaphore = None
active_models = {}
session_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_semaphore, active_models, session_pool
    
    # Initialize semaphore for max 2 concurrent models
    model_semaphore = asyncio.Semaphore(2)
    active_models = {}
    
    # Create persistent HTTP session pool
    connector = aiohttp.TCPConnector(
        limit=50,
        limit_per_host=10,
        keepalive_timeout=60,
        enable_cleanup_closed=True
    )
    session_pool = aiohttp.ClientSession(connector=connector)
    
    logger.info("Educational AI Orchestrator initialized")
    logger.info(f"Parameter experts configured: {len(PARAMETER_EXPERTS)}")
    logger.info("Memory management: Max 2 concurrent models")
    
    yield
    
    # Cleanup
    await session_pool.close()
    logger.info("Educational AI Orchestrator shutdown complete")

app = FastAPI(
    title="Educational Question Generation AI",
    description="Parameter-specific expert LLM system for educational content",
    version="1.0.0",
    lifespan=lifespan
)

class ModelManager:
    """Manages model loading/unloading with VRAM optimization"""
    
    def __init__(self):
        self.model_memory_usage = {
            "llama3.1:8b": 5.5,    # GB
            "mistral:7b": 5.0,
            "qwen2.5:7b": 5.0,
            "llama3.2:3b": 2.5
        }
        self.max_vram_gb = 18  # Leave 2GB buffer from 20GB
        
    async def ensure_model_loaded(self, expert_config: ParameterExpertConfig):
        """Ensure specific model is loaded, managing memory constraints"""
        model_name = expert_config.model
        port = expert_config.port
        
        # Check if model is already active
        if port in active_models:
            return port
            
        async with model_semaphore:
            # Calculate current VRAM usage
            current_usage = sum(
                self.model_memory_usage.get(active_models[p], 0) 
                for p in active_models
            )
            
            required_memory = self.model_memory_usage.get(model_name, 5.0)
            
            # Unload models if needed
            while current_usage + required_memory > self.max_vram_gb and active_models:
                oldest_port = min(active_models.keys())
                await self.unload_model(oldest_port)
                current_usage = sum(
                    self.model_memory_usage.get(active_models[p], 0) 
                    for p in active_models
                )
            
            # Load the required model
            await self.load_model(expert_config)
            active_models[port] = model_name
            
            logger.info(f"Loaded {model_name} on port {port} ({required_memory}GB)")
            return port
    
    async def load_model(self, expert_config: ParameterExpertConfig):
        """Load model via Ollama API"""
        try:
            # Ollama automatically handles model loading when first request is made
            # We just need to make a test request to ensure it's ready
            test_payload = {
                "model": expert_config.model,
                "messages": [{"role": "user", "content": "test"}],
                "stream": False
            }
            
            async with session_pool.post(
                f"http://localhost:{expert_config.port}/v1/chat/completions",
                json=test_payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    logger.info(f"Model {expert_config.model} ready on port {expert_config.port}")
                else:
                    logger.warning(f"Model loading may have issues: {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to load model {expert_config.model}: {e}")
            raise
    
    async def unload_model(self, port: int):
        """Unload model to free VRAM"""
        if port in active_models:
            model_name = active_models[port]
            del active_models[port]
            logger.info(f"Unloaded {model_name} from port {port}")

model_manager = ModelManager()

class EducationalAISystem:
    """Main orchestrator for educational question generation"""
    
    def __init__(self):
        self.max_iterations = 5
        self.min_approval_score = 7.0
        
    async def generate_question(self, request: QuestionRequest) -> QuestionResult:
        """Main pipeline: Generate question with parameter-specific expert validation"""
        start_time = time.time()
        iterations = 0
        parameter_validations = []
        
        # Step 1: Generate initial question
        logger.info(f"Generating question for topic: {request.topic}")
        question_content = await self._generate_initial_question(request)
        
        # Step 2: Sequential parameter validation with expert LLMs
        for iteration in range(self.max_iterations):
            iterations += 1
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            # Validate parameters sequentially (max 2 models at once via semaphore)
            current_validations = await self._validate_all_parameters(
                question_content, request
            )
            parameter_validations.extend(current_validations)
            
            # Check if all parameters are approved
            rejected_params = [v for v in current_validations if v.status == ParameterStatus.REJECTED]
            refinement_params = [v for v in current_validations if v.status == ParameterStatus.NEEDS_REFINEMENT]
            
            if not rejected_params and not refinement_params:
                logger.info("All parameters approved! Question generation complete.")
                break
                
            # Step 3: Refine question based on expert feedback
            if rejected_params or refinement_params:
                feedback = [v.feedback for v in rejected_params + refinement_params]
                question_content = await self._refine_question(
                    question_content, feedback, request
                )
        
        # Step 4: Generate CSV-ready format
        csv_ready = self._prepare_csv_output(question_content, parameter_validations)
        
        total_time = time.time() - start_time
        final_status = "approved" if not rejected_params and not refinement_params else "needs_work"
        
        return QuestionResult(
            question_content=question_content,
            parameter_validations=parameter_validations,
            iterations=iterations,
            total_processing_time=total_time,
            final_status=final_status,
            csv_ready=csv_ready
        )
    
    async def _generate_initial_question(self, request: QuestionRequest) -> Dict[str, Any]:
        """Generate initial question using main generator LLM"""
        generator_config = PARAMETER_EXPERTS["variation_expert"]  # Use main model
        
        await model_manager.ensure_model_loaded(generator_config)
        
        prompt = f"""Du bist ein Experte für die Erstellung von Bildungsaufgaben. Erstelle eine deutsche Wirtschaftsaufgabe mit folgenden Spezifikationen:

Thema: {request.topic}
Schwierigkeitsgrad: {request.difficulty.value}
Zielgruppe: {request.age_group}
Kontext: {request.context or 'Wirtschaftliche Grundbegriffe'}

Die Aufgabe soll alle notwendigen Parameter für eine vollständige Multiple-Choice-Frage enthalten:
- Aufgabenstellung (klar und präzise)
- 6-8 Antwortoptionen (eine korrekte, 5-7 Distraktoren)
- Bezug zum Referenztext wenn relevant
- Angemessene sprachliche Komplexität für das Niveau

Antworte im JSON-Format mit allen Komponenten der Aufgabe."""

        response = await self._call_expert_llm(generator_config, prompt)
        
        try:
            question_data = json.loads(response)
            logger.info("Initial question generated successfully")
            return question_data
        except json.JSONDecodeError:
            # Fallback parsing
            logger.warning("JSON parsing failed, using fallback structure")
            return {
                "aufgabenstellung": response[:500],
                "antwortoptionen": ["Option A", "Option B", "Option C", "Option D"],
                "korrekte_antwort": "Option A",
                "parameter_settings": {
                    "variation": request.difficulty.value,
                    "taxonomy_level": "Stufe 1",
                    "mathematical_requirement_level": "0"
                }
            }
    
    async def _validate_all_parameters(self, question: Dict[str, Any], request: QuestionRequest) -> List[ParameterValidation]:
        """Validate all parameters using appropriate expert LLMs"""
        validation_tasks = []
        
        # Create validation tasks for each parameter expert
        for expert_name, expert_config in PARAMETER_EXPERTS.items():
            if expert_name != "variation_expert":  # Skip main generator
                validation_tasks.append(
                    self._validate_with_expert(expert_config, question, request)
                )
        
        # Execute validations (semaphore limits concurrent models)
        validations = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        valid_validations = [v for v in validations if isinstance(v, ParameterValidation)]
        
        logger.info(f"Completed {len(valid_validations)} parameter validations")
        return valid_validations
    
    async def _validate_with_expert(self, expert_config: ParameterExpertConfig, 
                                   question: Dict[str, Any], 
                                   request: QuestionRequest) -> ParameterValidation:
        """Validate specific parameters with expert LLM"""
        start_time = time.time()
        
        await model_manager.ensure_model_loaded(expert_config)
        
        # Create specialized prompt for this expert
        prompt = f"""Du bist ein Experte für {expert_config.expertise}.

Analysiere diese Bildungsaufgabe und bewerte die folgenden Parameter:
{', '.join(expert_config.parameters)}

Aufgabe: {json.dumps(question, indent=2, ensure_ascii=False)}
Zielgruppe: {request.age_group}
Schwierigkeit: {request.difficulty.value}

Bewerte jeden Parameter auf einer Skala von 1-10 und gib spezifisches Feedback.
Antworte im JSON-Format mit 'parameter_scores', 'overall_score', 'status' (approved/rejected/needs_refinement), 'feedback'."""

        response = await self._call_expert_llm(expert_config, prompt)
        
        try:
            validation_data = json.loads(response)
            overall_score = validation_data.get("overall_score", 5.0)
            status_text = validation_data.get("status", "needs_refinement")
            
            # Map status to enum
            status_mapping = {
                "approved": ParameterStatus.APPROVED,
                "rejected": ParameterStatus.REJECTED,
                "needs_refinement": ParameterStatus.NEEDS_REFINEMENT
            }
            status = status_mapping.get(status_text, ParameterStatus.NEEDS_REFINEMENT)
            
            processing_time = time.time() - start_time
            
            return ParameterValidation(
                parameter=", ".join(expert_config.parameters),
                status=status,
                score=overall_score,
                feedback=validation_data.get("feedback", "No specific feedback"),
                expert_used=expert_config.name,
                processing_time=processing_time
            )
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse validation from {expert_config.name}")
            return ParameterValidation(
                parameter=", ".join(expert_config.parameters),
                status=ParameterStatus.NEEDS_REFINEMENT,
                score=5.0,
                feedback=response[:200],
                expert_used=expert_config.name,
                processing_time=time.time() - start_time
            )
    
    async def _refine_question(self, question: Dict[str, Any], 
                              feedback: List[str], 
                              request: QuestionRequest) -> Dict[str, Any]:
        """Refine question based on expert feedback"""
        generator_config = PARAMETER_EXPERTS["variation_expert"]
        
        await model_manager.ensure_model_loaded(generator_config)
        
        prompt = f"""Du bist ein Experte für Bildungsaufgaben. Verbessere diese Aufgabe basierend auf dem Expertenfeedback:

Aktuelle Aufgabe: {json.dumps(question, indent=2, ensure_ascii=False)}

Expertenfeedback:
{chr(10).join(f"- {fb}" for fb in feedback)}

Überarbeite die Aufgabe unter Berücksichtigung aller Verbesserungsvorschläge.
Antworte im JSON-Format mit der verbesserten Aufgabe."""

        response = await self._call_expert_llm(generator_config, prompt)
        
        try:
            refined_question = json.loads(response)
            logger.info("Question refined based on expert feedback")
            return refined_question
        except json.JSONDecodeError:
            logger.warning("Refinement parsing failed, keeping original")
            return question
    
    async def _call_expert_llm(self, expert_config: ParameterExpertConfig, prompt: str) -> str:
        """Call specific expert LLM via Ollama API"""
        payload = {
            "model": expert_config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": expert_config.temperature,
            "max_tokens": expert_config.max_tokens,
            "stream": False
        }
        
        try:
            async with session_pool.post(
                f"http://localhost:{expert_config.port}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    raise HTTPException(status_code=500, detail=f"LLM API error: {response.status}")
                
                data = await response.json()
                return data["choices"][0]["message"]["content"]
                
        except Exception as e:
            logger.error(f"❌ Failed to call {expert_config.name}: {e}")
            raise HTTPException(status_code=500, detail=f"Expert LLM error: {e}")
    
    def _prepare_csv_output(self, question: Dict[str, Any], 
                           validations: List[ParameterValidation]) -> Dict[str, str]:
        """Prepare question in CSV format matching stakeholder requirements"""
        
        # Extract basic information
        csv_data = {
            "c_id": f"ai-gen-{int(time.time())}",
            "subject": "Wirtschaft",
            "type": "multiple_choice",
            "text": question.get("aufgabenstellung", ""),
            "answers": json.dumps(question.get("antwortoptionen", [])),
        }
        
        # Map parameter validations to CSV columns
        param_settings = question.get("parameter_settings", {})
        
        # Fill in parameter columns based on validations and settings
        csv_data.update({
            "p.variation": param_settings.get("variation", "Stammaufgabe"),
            "p.taxanomy_level": param_settings.get("taxonomy_level", "Stufe 1"),
            "p.mathematical_requirement_level": param_settings.get("mathematical_requirement_level", "0"),
            "p.root_text_reference_explanatory_text": param_settings.get("root_text_reference", "Nicht vorhanden"),
            "p.root_text_obstacle_passive": "Nicht Enthalten",
            "p.root_text_obstacle_negation": "Nicht Enthalten", 
            "p.root_text_obstacle_complex_np": "Nicht Enthalten",
            "p.root_text_contains_irrelevant_information": "Nicht Enthalten",
            "p.instruction_explicitness_of_instruction": "Explizit",
            "p.instruction_obstacle_passive": "Nicht Enthalten",
            "p.instruction_obstacle_complex_np": "Nicht Enthalten"
        })
        
        return csv_data

# Initialize the AI system
ai_system = EducationalAISystem()

# API Endpoints
@app.post("/generate-question", response_model=QuestionResult)
async def generate_question(request: QuestionRequest):
    """Generate educational question with parameter-specific expert validation"""
    try:
        logger.info(f"New question request: {request.topic} ({request.difficulty.value})")
        result = await ai_system.generate_question(request)
        logger.info(f"Question generated in {result.total_processing_time:.2f}s with {result.iterations} iterations")
        return result
        
    except Exception as e:
        logger.error(f"❌ Question generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_models": len(active_models),
        "available_experts": list(PARAMETER_EXPERTS.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models/status")
async def model_status():
    """Get current model status and VRAM usage"""
    vram_usage = sum(
        model_manager.model_memory_usage.get(active_models[port], 0)
        for port in active_models
    )
    
    return {
        "active_models": active_models,
        "vram_usage_gb": vram_usage,
        "max_vram_gb": model_manager.max_vram_gb,
        "available_vram_gb": model_manager.max_vram_gb - vram_usage
    }

@app.post("/batch-generate")
async def batch_generate_questions(requests: List[QuestionRequest]):
    """Generate multiple questions in batch with proper memory management"""
    try:
        logger.info(f"📦 Batch generation request: {len(requests)} questions")
        
        results = []
        for i, request in enumerate(requests):
            logger.info(f"Processing question {i+1}/{len(requests)}")
            result = await ai_system.generate_question(request)
            results.append(result)
            
            # Brief pause between questions for memory management
            await asyncio.sleep(0.5)
        
        logger.info(f"Batch generation complete: {len(results)} questions")
        return {"results": results, "total_questions": len(results)}
        
    except Exception as e:
        logger.error(f"❌ Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "educational_ai_orchestrator:app",
        host="0.0.0.0", 
        port=8000,
        reload=False,
        log_level="info"
    )