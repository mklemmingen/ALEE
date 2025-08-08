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
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_expert_response(response_text: str) -> dict:
    """Robust JSON parsing for expert responses with error handling"""
    try:
        # Remove any leading/trailing whitespace
        response_text = response_text.strip()
        
        # Extract JSON from markdown code blocks if present
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end == -1:  # No closing ```
                json_text = response_text[start:].strip()
            else:
                json_text = response_text[start:end].strip()
        elif "```" in response_text and "{" in response_text:
            # Handle cases with ``` but no json marker
            start = response_text.find("{")
            end = response_text.rfind("}")
            if end > start:
                json_text = response_text[start:end+1]
            else:
                json_text = response_text
        else:
            # Try to find JSON object boundaries
            start = response_text.find("{")
            end = response_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_text = response_text[start:end+1]
            else:
                json_text = response_text
        
        # Parse JSON
        parsed = json.loads(json_text)
        
        # Ensure feedback is string, not dict or list
        if "feedback" in parsed:
            if isinstance(parsed["feedback"], dict):
                # Convert dict feedback to string
                feedback_parts = []
                for key, value in parsed["feedback"].items():
                    feedback_parts.append(f"{key}: {value}")
                parsed["feedback"] = "; ".join(feedback_parts)
            elif isinstance(parsed["feedback"], list):
                # Convert list feedback to string
                if parsed["feedback"] and isinstance(parsed["feedback"][0], dict):
                    feedback_parts = []
                    for item in parsed["feedback"]:
                        if isinstance(item, dict):
                            feedback_parts.append(str(item))
                        else:
                            feedback_parts.append(item)
                    parsed["feedback"] = "; ".join(feedback_parts)
                else:
                    parsed["feedback"] = "; ".join(map(str, parsed["feedback"]))
        
        # Ensure required fields exist with defaults
        if "overall_score" not in parsed:
            parsed["overall_score"] = 5.0
        if "status" not in parsed:
            parsed["status"] = "needs_refinement"
        if "feedback" not in parsed:
            parsed["feedback"] = "No feedback provided"
            
        return parsed
        
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing failed: {e}")
        logger.debug(f"Response text: {response_text[:500]}")
        return None
    except Exception as e:
        logger.warning(f"Response parsing failed: {e}")
        return None

# Data classes defined before functions that use them
@dataclass
class ParameterExpertConfig:
    name: str
    model: str
    port: int
    parameters: List[str]
    expertise: str
    temperature: float = 0.3
    max_tokens: int = 500

async def verify_model_health(expert_config: ParameterExpertConfig) -> bool:
    """Check if model server is responsive before calling"""
    try:
        url = f"http://localhost:{expert_config.port}/api/tags"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status != 200:
                    logger.warning(f"Model health check failed for {expert_config.name}: HTTP {response.status}")
                    return False
                
                # Also check if the specific model is loaded
                data = await response.json()
                models = data.get("models", [])
                model_names = [model.get("name", "") for model in models]
                
                if not any(expert_config.model in name for name in model_names):
                    logger.warning(f"Model {expert_config.model} not found in {expert_config.name} server")
                    return False
                    
                return True
    except asyncio.TimeoutError:
        logger.warning(f"Model health check timeout for {expert_config.name}")
        return False
    except Exception as e:
        logger.warning(f"Model health check error for {expert_config.name}: {e}")
        return False

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

# ValidationPlan Request Model
class ValidationPlanRequest(BaseModel):
    c_id: str = Field(..., description="Question ID in format: question_number-difficulty-version (e.g., 41-1-4)")
    text: str = Field(..., description="The informational text about the system's pre-configured topic")
    p_variation: str = Field(..., description="Stammaufgabe, schwer, leicht")
    p_taxonomy_level: str = Field(..., description="Stufe 1 (Wissen/Reproduktion), Stufe 2 (Anwendung/Transfer)")
    p_root_text_reference_explanatory_text: str = Field("Nicht vorhanden", description="Nicht vorhanden, Explizit, Implizit")
    p_root_text_obstacle_passive: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_root_text_obstacle_negation: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_root_text_obstacle_complex_np: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_root_text_contains_irrelevant_information: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_mathematical_requirement_level: str = Field("0", description="0 (Kein Bezug), 1 (Nutzen mathematischer Darstellungen), 2 (Mathematische Operation)")
    p_item_X_obstacle_passive: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_X_obstacle_negation: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_X_obstacle_complex_np: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_instruction_obstacle_passive: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_instruction_obstacle_complex_np: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_instruction_explicitness_of_instruction: str = Field("Implizit", description="Explizit, Implizit")

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

# ValidationPlan Result Model
class ValidationPlanResult(BaseModel):
    question_1: str = Field(..., description="First generated question")
    question_2: str = Field(..., description="Second generated question") 
    question_3: str = Field(..., description="Third generated question")
    c_id: str = Field(..., description="Original question ID")
    processing_time: float = Field(..., description="Total processing time")
    csv_data: Dict[str, Any] = Field(..., description="CSV-ready data with all parameters")

# Parameter Expert Configurations
PARAMETER_EXPERTS = {
    "variation_expert": ParameterExpertConfig(
        name="variation_expert",
        model="llama3.1NutzenMathematischerDarstellungen:8b",
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
        expertise="Bloom's taxonomy classification (Stufe 1NutzenMathematischerDarstellungen: Wissen/Reproduktion, Stufe 2: Anwendung/Transfer)",
        temperature=0.2
    ),
    "math_expert": ParameterExpertConfig(
        name="math_expert",
        model="qwen2.5:7b", 
        port=8003,
        parameters=["p.mathematical_requirement_level"],
        expertise="Mathematical requirement assessment (0KeinBezug-2 scale)",
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
        model="llama3.1NutzenMathematischerDarstellungen:8b",
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
    version="1NutzenMathematischerDarstellungen.0KeinBezug.0KeinBezug",
    lifespan=lifespan
)

class ModelManager:
    """Manages model loading/unloading with VRAM optimization"""
    
    def __init__(self):
        self.model_memory_usage = {
            "llama3.1NutzenMathematischerDarstellungen:8b": 5.5,    # GB
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
                "prompt": "test",
                "stream": False
            }
            
            async with session_pool.post(
                f"http://localhost:{expert_config.port}/api/generate",
                json=test_payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    logger.info(f"Model {expert_config.model} ready on port {expert_config.port}")
                else:
                    logger.warning(f"Model loading may have issues: {response.status}")
                    
        except Exception as e:
            logger.error(f"[ERROR] Failed to load model {expert_config.model}: {e}")
            raise
    
    async def unload_model(self, port: int):
        """Unload model to free VRAM"""
        if port in active_models:
            model_name = active_models[port]
            del active_models[port]
            logger.info(f"Unloaded {model_name} from port {port}")

model_manager = ModelManager()

class ValidationPlanPromptBuilder:
    """Builds modular prompts based on parameter values using txt files"""
    
    def __init__(self):
        self.base_path = Path("/home/mklemmingen/PycharmProjects/PythonProject/ALEE_Agent")
        
    def load_prompt_txt(self, file_path: str) -> str:
        """Load prompt text from file with error handling"""
        try:
            full_path = self.base_path / file_path
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.warning(f"Prompt file not found: {file_path}")
            return ""
        except Exception as e:
            logger.error(f"Error loading prompt {file_path}: {e}")
            return ""
    
    def build_variation_prompt(self, variation: str) -> str:
        """Build variation-specific prompt"""
        variation_map = {
            "stammaufgabe": "variationPrompts/multiple-choice.txt",
            "schwer": "variationPrompts/true-false.txt", 
            "leicht": "variationPrompts/single-choice.txt"
        }
        file_path = variation_map.get(variation.lower(), "variationPrompts/multiple-choice.txt")
        return self.load_prompt_txt(file_path)
    
    def build_taxonomy_prompt(self, level: str) -> str:
        """Build taxonomy-specific prompt"""
        if "Stufe 1" in level or "1" in level:
            return self.load_prompt_txt("taxonomyLevelPrompt/stufe1WissenReproduktion.txt")
        elif "Stufe 2" in level or "2" in level:
            return self.load_prompt_txt("taxonomyLevelPrompt/stufe2AnwendungTransfer.txt")
        else:
            return self.load_prompt_txt("taxonomyLevelPrompt/stufe1WissenReproduktion.txt")
    
    def build_mathematical_prompt(self, level: str) -> str:
        """Build mathematical requirement prompt"""
        level_map = {
            "0": "mathematicalRequirementLevel/0KeinBezug",
            "1": "mathematicalRequirementLevel/1NutzenMathematischerDarstellungen", 
            "2": "mathematicalRequirementLevel/2MathematischeOperationen"
        }
        file_path = level_map.get(level.strip(), "mathematicalRequirementLevel/0KeinBezug")
        return self.load_prompt_txt(file_path)
    
    def build_obstacle_prompt(self, obstacle_type: str, value: str) -> str:
        """Build obstacle-specific prompts"""
        value_file = "enthalten.txt" if "Enthalten" in value and "Nicht" not in value else "nichtEnthalten.txt"
        
        obstacle_map = {
            "passive": f"rootTextParameterTextPrompts/obstaclePassivePrompts/{value_file}",
            "negation": f"rootTextParameterTextPrompts/obstacleNegationPrompts/{value_file}", 
            "complex_np": f"rootTextParameterTextPrompts/obstacleComplexPrompts/{value_file}",
            "item_passive": f"itemXObstacle/passive/{value_file}",
            "item_negation": f"itemXObstacle/negation/{value_file}",
            "item_complex": f"itemXObstacle/complex/{value_file}",
            "instruction_passive": f"instructionObstacle/passive/{value_file}",
            "instruction_complex_np": f"instructionObstacle/complex_np/{value_file}"
        }
        
        file_path = obstacle_map.get(obstacle_type, "")
        return self.load_prompt_txt(file_path) if file_path else ""
    
    def build_irrelevant_info_prompt(self, value: str) -> str:
        """Build irrelevant information prompt"""
        value_file = "enthalten.txt" if "Enthalten" in value and "Nicht" not in value else "nichtEnthalten.txt"
        return self.load_prompt_txt(f"rootTextParameterTextPrompts/containsIrrelevantInformationPrompt/{value_file}")
    
    def build_explicitness_prompt(self, value: str) -> str:
        """Build instruction explicitness prompt"""
        if "Explizit" in value:
            return self.load_prompt_txt("instructionExplicitnessOfInstruction/explizit")
        else:
            return self.load_prompt_txt("instructionExplicitnessOfInstruction/implizit")
    
    def build_master_prompt(self, request: ValidationPlanRequest) -> str:
        """Build the complete master prompt from modular components"""
        components = []
        
        # Base instruction
        components.append("Du bist ein Experte für die Erstellung von Bildungsaufgaben. Erstelle basierend auf dem gegebenen Text GENAU DREI deutsche Bildungsfragen mit den spezifizierten Parametern.")
        components.append(f"\nReferenztext:\n{request.text}\n")
        
        # Variation-specific component
        variation_prompt = self.build_variation_prompt(request.p_variation)
        if variation_prompt:
            components.append(f"Variationsanforderungen:\n{variation_prompt}")
        
        # Taxonomy component
        taxonomy_prompt = self.build_taxonomy_prompt(request.p_taxonomy_level)
        if taxonomy_prompt:
            components.append(f"Taxonomie-Level:\n{taxonomy_prompt}")
        
        # Mathematical requirements
        math_prompt = self.build_mathematical_prompt(request.p_mathematical_requirement_level)
        if math_prompt:
            components.append(f"Mathematische Anforderungen:\n{math_prompt}")
        
        # Obstacle requirements
        if request.p_root_text_obstacle_passive != "Nicht Enthalten":
            passive_prompt = self.build_obstacle_prompt("passive", request.p_root_text_obstacle_passive)
            if passive_prompt:
                components.append(f"Passive Konstruktionen (Referenztext):\n{passive_prompt}")
        
        if request.p_root_text_obstacle_negation != "Nicht Enthalten":
            negation_prompt = self.build_obstacle_prompt("negation", request.p_root_text_obstacle_negation)
            if negation_prompt:
                components.append(f"Negationen (Referenztext):\n{negation_prompt}")
        
        if request.p_root_text_obstacle_complex_np != "Nicht Enthalten":
            complex_prompt = self.build_obstacle_prompt("complex_np", request.p_root_text_obstacle_complex_np)
            if complex_prompt:
                components.append(f"Komplexe Nominalphrasen (Referenztext):\n{complex_prompt}")
        
        # Irrelevant information
        if request.p_root_text_contains_irrelevant_information != "Nicht Enthalten":
            irrelevant_prompt = self.build_irrelevant_info_prompt(request.p_root_text_contains_irrelevant_information)
            if irrelevant_prompt:
                components.append(f"Irrelevante Informationen:\n{irrelevant_prompt}")
        
        # Item-specific obstacles
        if request.p_item_X_obstacle_passive != "Nicht Enthalten":
            item_passive = self.build_obstacle_prompt("item_passive", request.p_item_X_obstacle_passive)
            if item_passive:
                components.append(f"Item-Passive:\n{item_passive}")
        
        if request.p_item_X_obstacle_negation != "Nicht Enthalten":
            item_negation = self.build_obstacle_prompt("item_negation", request.p_item_X_obstacle_negation)
            if item_negation:
                components.append(f"Item-Negation:\n{item_negation}")
        
        if request.p_item_X_obstacle_complex_np != "Nicht Enthalten":
            item_complex = self.build_obstacle_prompt("item_complex", request.p_item_X_obstacle_complex_np)
            if item_complex:
                components.append(f"Item-Komplexe NP:\n{item_complex}")
        
        # Instruction-specific obstacles  
        if request.p_instruction_obstacle_passive != "Nicht Enthalten":
            instr_passive = self.build_obstacle_prompt("instruction_passive", request.p_instruction_obstacle_passive)
            if instr_passive:
                components.append(f"Instruktion-Passive:\n{instr_passive}")
        
        if request.p_instruction_obstacle_complex_np != "Nicht Enthalten":
            instr_complex = self.build_obstacle_prompt("instruction_complex_np", request.p_instruction_obstacle_complex_np)
            if instr_complex:
                components.append(f"Instruktion-Komplexe NP:\n{instr_complex}")
        
        # Explicitness
        explicit_prompt = self.build_explicitness_prompt(request.p_instruction_explicitness_of_instruction)
        if explicit_prompt:
            components.append(f"Instruktions-Explizitheit:\n{explicit_prompt}")
        
        components.append("\nWICHTIG: Antworte mit GENAU DREI Fragen im JSON-Format: {\"question_1\": \"...\", \"question_2\": \"...\", \"question_3\": \"...\", \"answers_1\": [...], \"answers_2\": [...], \"answers_3\": [...]}")
        
        return "\n\n".join(components)

class EducationalAISystem:
    """Main orchestrator for educational question generation - ValidationPlan aligned"""
    
    def __init__(self):
        self.max_iterations = 3  # ValidationPlan specifies max 3 iterations
        self.min_approval_score = 7.0
        self.prompt_builder = ValidationPlanPromptBuilder()
        
    async def clean_expert_sessions(self):
        """Clean all expert sessions for fresh start as specified in ValidationPlan"""
        logger.info("Cleaning expert sessions for fresh start...")
        
        # Check health of all experts and clear any existing conversations
        for expert_name, expert_config in PARAMETER_EXPERTS.items():
            try:
                # Make a simple call to reset the conversation context
                reset_payload = {
                    "model": expert_config.model,
                    "prompt": "Reset conversation context. Reply with 'Ready'.",
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 10}
                }
                
                async with session_pool.post(
                    f"http://localhost:{expert_config.port}/api/generate",
                    json=reset_payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Reset {expert_name} session")
                    else:
                        logger.warning(f"Failed to reset {expert_name}: HTTP {response.status}")
                        
            except Exception as e:
                logger.warning(f"Error resetting {expert_name}: {e}")
        
        logger.info("Expert session cleanup complete")
    
    async def generate_validation_plan_questions(self, request: ValidationPlanRequest) -> ValidationPlanResult:
        """Generate exactly 3 questions according to ValidationPlan specifications"""
        start_time = time.time()
        
        # Step 1: Clean expert sessions for fresh start
        await self.clean_expert_sessions()
        
        # Step 2: Build master prompt using modular components
        master_prompt = self.prompt_builder.build_master_prompt(request)
        logger.info("Built master prompt with modular components")
        
        # Step 3: Generate initial 3 questions using main generator
        generator_config = PARAMETER_EXPERTS["variation_expert"]
        await model_manager.ensure_model_loaded(generator_config)
        
        questions_response = await self._call_expert_llm(generator_config, master_prompt)
        questions_data = parse_expert_response(questions_response)
        
        if not questions_data:
            # Fallback parsing
            try:
                questions_data = json.loads(questions_response)
            except:
                logger.error("Failed to parse questions from generator")
                questions_data = {
                    "question_1": "Fallback question 1",
                    "question_2": "Fallback question 2", 
                    "question_3": "Fallback question 3",
                    "answers_1": ["A", "B", "C"],
                    "answers_2": ["A", "B", "C"],
                    "answers_3": ["A", "B", "C"]
                }
        
        # Step 4: Expert validation for each question (max 3 iterations)
        validated_questions = []
        
        for i in range(3):
            question_key = f"question_{i+1}"
            answer_key = f"answers_{i+1}"
            
            question_text = questions_data.get(question_key, f"Question {i+1}")
            question_answers = questions_data.get(answer_key, ["A", "B", "C"])
            
            # Validate this question through expert iterations
            validated_question = await self._validate_question_with_experts(
                question_text, question_answers, request, i+1
            )
            validated_questions.append(validated_question)
        
        # Step 5: Build CSV data according to ValidationPlan format
        initial_csv_data = self._build_validation_plan_csv(request, validated_questions)
        
        # Step 6: Apply intelligent fallback system for CSV issues
        csv_data = await self._intelligent_csv_fallback(validated_questions, request, initial_csv_data)
        
        total_time = time.time() - start_time
        
        return ValidationPlanResult(
            question_1=validated_questions[0],
            question_2=validated_questions[1],
            question_3=validated_questions[2],
            c_id=request.c_id,
            processing_time=total_time,
            csv_data=csv_data
        )
    
    async def _validate_question_with_experts(self, question: str, answers: List[str], 
                                            request: ValidationPlanRequest, question_num: int) -> str:
        """Validate single question through expert system (max 3 iterations)"""
        current_question = question
        
        for iteration in range(self.max_iterations):
            logger.info(f"Question {question_num}, Iteration {iteration + 1}/{self.max_iterations}")
            
            # Get expert validations
            validations = await self._get_expert_validations(current_question, request)
            
            # Check if approved by all experts
            failed_validations = [v for v in validations 
                                if v.status in [ParameterStatus.REJECTED, ParameterStatus.NEEDS_REFINEMENT]]
            
            if not failed_validations:
                logger.info(f"Question {question_num} approved by all experts")
                break
                
            if iteration < self.max_iterations - 1:  # Not the last iteration
                # Refine question based on feedback
                feedback = [v.feedback for v in failed_validations]
                current_question = await self._refine_single_question(current_question, feedback, request)
        
        return current_question
        
    async def _get_expert_validations(self, question: str, request: ValidationPlanRequest) -> List[ParameterValidation]:
        """Get validations from all relevant experts for a single question"""
        validation_tasks = []
        
        # Create validation tasks for each relevant expert
        for expert_name, expert_config in PARAMETER_EXPERTS.items():
            if expert_name != "variation_expert":  # Skip main generator
                validation_tasks.append(
                    self._validate_question_with_expert(expert_config, question, request)
                )
        
        # Execute validations (semaphore limits concurrent models)
        validations = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        valid_validations = [v for v in validations if isinstance(v, ParameterValidation)]
        
        return valid_validations
    
    async def _validate_question_with_expert(self, expert_config: ParameterExpertConfig,
                                           question: str, request: ValidationPlanRequest) -> ParameterValidation:
        """Validate question with specific expert using parameter configuration"""
        start_time = time.time()
        
        # Health check
        if not await verify_model_health(expert_config):
            return ParameterValidation(
                parameter=",".join(expert_config.parameters),
                status=ParameterStatus.NEEDS_REFINEMENT,
                score=5.0,
                feedback="Model server unavailable",
                expert_used=expert_config.name,
                processing_time=time.time() - start_time
            )
        
        await model_manager.ensure_model_loaded(expert_config)
        
        # Build expert prompt with parameter configuration
        expert_prompt = f"""Du bist ein Experte für {expert_config.expertise}.

Analysiere diese Bildungsfrage bezüglich der spezifizierten Parameter:

Frage: {question}

Parameter-Konfiguration:
- c_id: {request.c_id}
- Variation: {request.p_variation}
- Taxonomie-Level: {request.p_taxonomy_level}
- Mathematisches Niveau: {request.p_mathematical_requirement_level}
- Root-Text Passive: {request.p_root_text_obstacle_passive}
- Root-Text Negation: {request.p_root_text_obstacle_negation}
- Root-Text Komplexe NP: {request.p_root_text_obstacle_complex_np}
- Item-X Passive: {request.p_item_X_obstacle_passive}
- Item-X Negation: {request.p_item_X_obstacle_negation}
- Item-X Komplexe NP: {request.p_item_X_obstacle_complex_np}
- Instruktion Passive: {request.p_instruction_obstacle_passive}
- Instruktion Komplexe NP: {request.p_instruction_obstacle_complex_np}
- Instruktion Explizitheit: {request.p_instruction_explicitness_of_instruction}
- Irrelevante Informationen: {request.p_root_text_contains_irrelevant_information}

Bewerte die Frage auf einer Skala von 1-10 und gib spezifisches Feedback.

Antworte im JSON-Format:
{{
  "overall_score": numeric_score,
  "status": "approved" | "needs_refinement" | "rejected",
  "feedback": "Detaillierte Bewertung und Verbesserungsvorschläge"
}}"""
        
        response = await self._call_expert_llm(expert_config, expert_prompt)
        validation_data = parse_expert_response(response)
        
        if validation_data:
            status_mapping = {
                "approved": ParameterStatus.APPROVED,
                "rejected": ParameterStatus.REJECTED,
                "needs_refinement": ParameterStatus.NEEDS_REFINEMENT
            }
            status = status_mapping.get(validation_data.get("status", "needs_refinement"), ParameterStatus.NEEDS_REFINEMENT)
            
            return ParameterValidation(
                parameter=",".join(expert_config.parameters),
                status=status,
                score=validation_data.get("overall_score", 5.0),
                feedback=validation_data.get("feedback", "No feedback"),
                expert_used=expert_config.name,
                processing_time=time.time() - start_time
            )
        else:
            return ParameterValidation(
                parameter=",".join(expert_config.parameters),
                status=ParameterStatus.NEEDS_REFINEMENT,
                score=5.0,
                feedback="Response parsing failed",
                expert_used=expert_config.name,
                processing_time=time.time() - start_time
            )
    
    async def _refine_single_question(self, question: str, feedback: List[str], 
                                     request: ValidationPlanRequest) -> str:
        """Refine a single question based on expert feedback"""
        generator_config = PARAMETER_EXPERTS["variation_expert"]
        await model_manager.ensure_model_loaded(generator_config)
        
        refinement_prompt = f"""Du bist ein Experte für Bildungsaufgaben. Verbessere diese Frage basierend auf dem Expertenfeedback:

Aktuelle Frage: {question}

Expertenfeedback:
{chr(10).join(f"- {fb}" for fb in feedback)}

Parameter-Vorgaben:
- Variation: {request.p_variation}
- Taxonomie: {request.p_taxonomy_level}
- Alle anderen Parameter wie ursprünglich konfiguriert

Verbessere die Frage unter Berücksichtigung des Feedbacks. Antworte nur mit der verbesserten Frage."""
        
        response = await self._call_expert_llm(generator_config, refinement_prompt)
        
        # Extract refined question from response
        refined_data = parse_expert_response(response)
        if refined_data and "question" in refined_data:
            return refined_data["question"]
        else:
            # Fallback: return response text directly
            return response.strip()
    
    def _build_validation_plan_csv(self, request: ValidationPlanRequest, questions: List[str]) -> Dict[str, Any]:
        """Build CSV data according to ValidationPlan specifications"""
        
        # Extract difficulty from c_id (format: question_number-difficulty-version)
        c_id_parts = request.c_id.split("-")
        difficulty_num = c_id_parts[1] if len(c_id_parts) > 1 else "1"
        difficulty_map = {"1": "stammaufgabe", "2": "leicht", "3": "schwer"}
        subject = difficulty_map.get(difficulty_num, "stammaufgabe")
        
        # Determine question type based on variation
        type_map = {
            "stammaufgabe": "multiple-choice",
            "schwer": "true-false",
            "leicht": "single-choice"
        }
        question_type = type_map.get(request.p_variation.lower(), "multiple-choice")
        
        # Build the main question text (combining all 3 questions)
        combined_text = f"1. {questions[0]} 2. {questions[1]} 3. {questions[2]}"
        
        # Build CSV data with all required columns from ValidationPlan
        csv_data = {
            "c_id": request.c_id,
            "subject": subject,
            "type": question_type,
            "text": combined_text,
            "p.instruction_explicitness_of_instruction": request.p_instruction_explicitness_of_instruction,
            "p.instruction_obstacle_complex_np": request.p_instruction_obstacle_complex_np,
            "p.instruction_obstacle_negation": "Nicht enthalten",  # Default as per example
            "p.instruction_obstacle_passive": request.p_instruction_obstacle_passive,
            # Item parameters (8 items as per ValidationPlan CSV format)
            "p.item_1_answer_verbatim_explanatory_text": "Nicht enthalten",
            "p.item_1_obstacle_complex_np": request.p_item_X_obstacle_complex_np,
            "p.item_1_obstacle_negation": request.p_item_X_obstacle_negation,
            "p.item_1_obstacle_passive": request.p_item_X_obstacle_passive,
            "p.item_1_sentence_length": "",
            "p.item_2_answer_verbatim_explanatory_text": "Nicht enthalten",
            "p.item_2_obstacle_complex_np": request.p_item_X_obstacle_complex_np,
            "p.item_2_obstacle_negation": request.p_item_X_obstacle_negation,
            "p.item_2_obstacle_passive": request.p_item_X_obstacle_passive,
            "p.item_2_sentence_length": "",
            "p.item_3_answer_verbatim_explanatory_text": "Nicht enthalten",
            "p.item_3_obstacle_complex_np": request.p_item_X_obstacle_complex_np,
            "p.item_3_obstacle_negation": request.p_item_X_obstacle_negation,
            "p.item_3_obstacle_passive": request.p_item_X_obstacle_passive,
            "p.item_3_sentence_length": "",
            "p.item_4_answer_verbatim_explanatory_text": "Nicht enthalten",
            "p.item_4_obstacle_complex_np": request.p_item_X_obstacle_complex_np,
            "p.item_4_obstacle_negation": request.p_item_X_obstacle_negation,
            "p.item_4_obstacle_passive": request.p_item_X_obstacle_passive,
            "p.item_4_sentence_length": "",
            "p.item_5_answer_verbatim_explanatory_text": "Nicht enthalten",
            "p.item_5_obstacle_complex_np": request.p_item_X_obstacle_complex_np,
            "p.item_5_obstacle_negation": request.p_item_X_obstacle_negation,
            "p.item_5_obstacle_passive": request.p_item_X_obstacle_passive,
            "p.item_5_sentence_length": "",
            "p.item_6_answer_verbatim_explanatory_text": "Nicht enthalten",
            "p.item_6_obstacle_complex_np": request.p_item_X_obstacle_complex_np,
            "p.item_6_obstacle_negation": request.p_item_X_obstacle_negation,
            "p.item_6_obstacle_passive": request.p_item_X_obstacle_passive,
            "p.item_6_sentence_length": "",
            "p.item_7_answer_verbatim_explanatory_text": "Nicht enthalten",
            "p.item_7_obstacle_complex_np": request.p_item_X_obstacle_complex_np,
            "p.item_7_obstacle_negation": request.p_item_X_obstacle_negation,
            "p.item_7_obstacle_passive": request.p_item_X_obstacle_passive,
            "p.item_7_sentence_length": "",
            "p.item_8_answer_verbatim_explanatory_text": "Nicht enthalten",
            "p.item_8_obstacle_complex_np": request.p_item_X_obstacle_complex_np,
            "p.item_8_obstacle_negation": request.p_item_X_obstacle_negation,
            "p.item_8_obstacle_passive": request.p_item_X_obstacle_passive,
            "p.item_8_sentence_length": "",
            "p.mathematical_requirement_level": f"{request.p_mathematical_requirement_level} (Kein Bezug)" if request.p_mathematical_requirement_level == "0" else request.p_mathematical_requirement_level,
            "p.root_text_contains_irrelevant_information": request.p_root_text_contains_irrelevant_information,
            "p.root_text_obstacle_complex_np": request.p_root_text_obstacle_complex_np,
            "p.root_text_obstacle_negation": request.p_root_text_obstacle_negation,
            "p.root_text_obstacle_passive": request.p_root_text_obstacle_passive,
            "p.root_text_reference_explanatory_text": request.p_root_text_reference_explanatory_text,
            "p.taxanomy_level": request.p_taxonomy_level,
            "p.variation": request.p_variation,
            "answers": "Question-specific answers",  # Will be filled by intelligent fallback system
            "p.instruction_number_of_sentences": "1"  # Default value
        }
        
        return csv_data
    
    async def _intelligent_csv_fallback(self, questions: List[str], request: ValidationPlanRequest, 
                                       initial_csv: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligent fallback system with LM help for CSV conversion warnings/errors"""
        logger.info("Applying intelligent CSV fallback system...")
        
        # Check for potential issues and warnings
        issues = []
        
        # Check for missing answers
        if not initial_csv.get("answers") or initial_csv["answers"] == "Question-specific answers":
            issues.append("Missing specific answer content")
        
        # Check for empty item fields
        empty_items = []
        for i in range(1, 9):
            if not initial_csv.get(f"p.item_{i}_sentence_length"):
                empty_items.append(f"item_{i}_sentence_length")
        
        if empty_items:
            issues.append(f"Empty item fields: {', '.join(empty_items[:3])}...")
        
        # Check for parameter consistency
        if request.p_variation.lower() not in ["stammaufgabe", "leicht", "schwer"]:
            issues.append(f"Invalid variation: {request.p_variation}")
        
        if issues:
            logger.warning(f"CSV issues detected: {issues}")
            
            # Use LM to help fix issues
            try:
                corrected_csv = await self._llm_csv_correction(questions, request, initial_csv, issues)
                return corrected_csv
            except Exception as e:
                logger.error(f"LM CSV correction failed: {e}")
                return self._apply_fallback_defaults(initial_csv, issues)
        
        return initial_csv
    
    async def _llm_csv_correction(self, questions: List[str], request: ValidationPlanRequest,
                                 initial_csv: Dict[str, Any], issues: List[str]) -> Dict[str, Any]:
        """Use LM to help correct CSV conversion issues"""
        
        # Use a lightweight expert for CSV correction
        helper_config = PARAMETER_EXPERTS["content_expert"]  # Reuse existing expert
        await model_manager.ensure_model_loaded(helper_config)
        
        correction_prompt = f"""Du bist ein Experte für CSV-Datenkonvertierung von Bildungsfragen.

AUFGABE: Korrigiere die folgenden CSV-Konvertierungsprobleme:

Fragen:
1. {questions[0]}
2. {questions[1]}  
3. {questions[2]}

Aktuelle CSV-Daten:
{json.dumps(initial_csv, indent=2, ensure_ascii=False)}

Erkannte Probleme:
{chr(10).join(f"- {issue}" for issue in issues)}

Parameter-Konfiguration:
- c_id: {request.c_id}
- Variation: {request.p_variation}
- Typ: {initial_csv.get('type', 'unknown')}

KORREKTUR-ANFORDERUNGEN:
1. Wenn "Missing specific answer content": Extrahiere realistische Antworten aus den Fragen
2. Wenn "Empty item fields": Schätze Satzlängen basierend auf den Fragen (kurz/mittel/lang)
3. Wenn "Invalid variation": Korrigiere auf stammaufgabe/leicht/schwer

Antworte mit korrigierter CSV als JSON im exakt gleichen Format."""

        response = await self._call_expert_llm(helper_config, correction_prompt)
        correction_data = parse_expert_response(response)
        
        if correction_data and isinstance(correction_data, dict):
            # Merge corrections with original data
            corrected_csv = initial_csv.copy()
            corrected_csv.update(correction_data)
            
            logger.info("LM CSV correction applied successfully")
            return corrected_csv
        else:
            raise Exception("LM correction parsing failed")
    
    def _apply_fallback_defaults(self, initial_csv: Dict[str, Any], issues: List[str]) -> Dict[str, Any]:
        """Apply intelligent fallback defaults when LM correction fails"""
        logger.info("Applying fallback defaults for CSV issues...")
        
        fallback_csv = initial_csv.copy()
        
        # Fix missing answers
        if "Missing specific answer content" in str(issues):
            fallback_csv["answers"] = "Fallback: Multiple choice answers based on question content"
        
        # Fill empty sentence lengths with defaults
        for i in range(1, 9):
            if not fallback_csv.get(f"p.item_{i}_sentence_length"):
                fallback_csv[f"p.item_{i}_sentence_length"] = "mittel"  # Default to medium length
        
        # Fix invalid variations
        if any("Invalid variation" in issue for issue in issues):
            fallback_csv["p.variation"] = "stammaufgabe"  # Safe default
            fallback_csv["type"] = "multiple-choice"
        
        logger.info("Fallback defaults applied")
        return fallback_csv
    
    async def generate_question(self, request: QuestionRequest) -> QuestionResult:
        """Main pipeline: Generate question with parameter-specific expert validation"""
        start_time = time.time()
        iterations = 0
        parameter_validations = []
        
        # Step 1NutzenMathematischerDarstellungen: Generate initial question
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
                    "taxonomy_level": "Stufe 1NutzenMathematischerDarstellungen",
                    "mathematical_requirement_level": "0KeinBezug"
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
        
        # Health check before proceeding
        if not await verify_model_health(expert_config):
            logger.error(f"Model health check failed for {expert_config.name}")
            return ParameterValidation(
                parameter=", ".join(expert_config.parameters),
                status=ParameterStatus.NEEDS_REFINEMENT,
                score=5.0,
                feedback="Model server unavailable",
                expert_used=expert_config.name,
                processing_time=time.time() - start_time
            )
        
        await model_manager.ensure_model_loaded(expert_config)
        
        # Create specialized prompt for this expert
        prompt = f"""Du bist ein Experte für {expert_config.expertise}.

        Analysiere diese Bildungsaufgabe und bewerte die folgenden Parameter:
        {', '.join(expert_config.parameters)}
        
        Aufgabe: {json.dumps(question, indent=2, ensure_ascii=False)}
        Zielgruppe: {request.age_group}
        Schwierigkeit: {request.difficulty.value}
        
        Bewerte jeden Parameter auf einer Skala von 1NutzenMathematischerDarstellungen-10 und gib spezifisches Feedback.
        
        WICHTIG: Antworte in diesem EXAKTEN JSON-Format:
        {{
          "parameter_scores": {{
            "parameter_name": numeric_score
          }},
          "overall_score": numeric_score,
          "status": "approved" | "needs_refinement" | "rejected",
          "feedback": "Single string describing the assessment"
        }}"""

        response = await self._call_expert_llm(expert_config, prompt)
        
        # Use robust JSON parsing
        validation_data = parse_expert_response(response)
        
        if validation_data is not None:
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
        else:
            # Fallback when parsing completely fails
            logger.warning(f"Complete parsing failure from {expert_config.name}")
            return ParameterValidation(
                parameter=", ".join(expert_config.parameters),
                status=ParameterStatus.NEEDS_REFINEMENT,
                score=5.0,
                feedback=f"Response parsing failed: {response[:200]}...",
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
        
        WICHTIG: Antworte im JSON-Format mit der vollständigen verbesserten Aufgabe.
        Behalte die gleiche Struktur wie die ursprüngliche Aufgabe bei."""

        response = await self._call_expert_llm(generator_config, prompt)
        
        # Use robust JSON parsing for refinement too
        refined_data = parse_expert_response(response)
        
        if refined_data is not None:
            logger.info("Question refined based on expert feedback")
            return refined_data
        else:
            # Try standard JSON parsing as fallback
            try:
                refined_question = json.loads(response)
                logger.info("Question refined using fallback JSON parsing")
                return refined_question
            except json.JSONDecodeError:
                logger.warning("Refinement parsing failed completely, keeping original")
                return question
    
    async def _call_expert_llm(self, expert_config: ParameterExpertConfig, prompt: str) -> str:
        """Call specific expert LLM via Ollama API"""
        logger.info(f"[EXPERT_CALL] Calling {expert_config.name} ({expert_config.model}) on port {expert_config.port}")
        logger.info(f"[PROMPT] TO {expert_config.name}:\n{'-'*50}\n{prompt[:300]}{'...' if len(prompt) > 300 else ''}\n{'-'*50}")
        
        payload = {
            "model": expert_config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": expert_config.temperature,
                "num_predict": expert_config.max_tokens
            }
        }
        
        try:
            async with session_pool.post(
                f"http://localhost:{expert_config.port}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    raise HTTPException(status_code=500, detail=f"LLM API error: {response.status}")
                
                data = await response.json()
                response_text = data["response"]
                
                logger.info(f"[RESPONSE] FROM {expert_config.name}:\n{'-'*50}\n{response_text[:300]}{'...' if len(response_text) > 300 else ''}\n{'-'*50}")
                
                return response_text
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to call {expert_config.name}: {e}")
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
            "p.taxanomy_level": param_settings.get("taxonomy_level", "Stufe 1NutzenMathematischerDarstellungen"),
            "p.mathematical_requirement_level": param_settings.get("mathematical_requirement_level", "0KeinBezug"),
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

@app.post("/generate-validation-plan", response_model=ValidationPlanResult)
async def generate_validation_plan_questions(request: ValidationPlanRequest):
    """Generate exactly 3 questions according to ValidationPlan specifications"""
    try:
        logger.info(f"ValidationPlan request: {request.c_id} - {request.p_variation}")
        result = await ai_system.generate_validation_plan_questions(request)
        logger.info(f"ValidationPlan questions generated in {result.processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"ValidationPlan generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-question", response_model=QuestionResult)
async def generate_question(request: QuestionRequest):
    """Generate educational question with parameter-specific expert validation (legacy endpoint)"""
    try:
        logger.info(f"Legacy question request: {request.topic} ({request.difficulty.value})")
        result = await ai_system.generate_question(request)
        logger.info(f"Question generated in {result.total_processing_time:.2f}s with {result.iterations} iterations")
        return result
        
    except Exception as e:
        logger.error(f"Question generation failed: {e}")
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
        logger.info(f"Batch generation request: {len(requests)} questions")
        
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
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "educational_ai_orchestrator:app",
        host="0KeinBezug.0KeinBezug.0KeinBezug.0KeinBezug",
        port=8000,
        reload=False,
        log_level="info"
    )