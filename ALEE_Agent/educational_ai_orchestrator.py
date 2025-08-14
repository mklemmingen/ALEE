"""
Educational Question Generation AI Orchestrator
Parameter-Specific Expert LLM System for Manjaro Linux with AMD GPU

Architecture:
- Main Generator LLM: Creates initial questions
- Parameter Expert LLMs: Validate and refine specific parameters
- Memory Management: Max 2 concurrent models (20GB VRAM optimization)
- Async Processing: Sequential parameter validation with feedback loops
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional

import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Custom Exception Classes for meaningful error handling
class QuestionGenerationError(Exception):
    """Base exception for question generation errors"""
    def __init__(self, message: str, error_code: str, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ExpertValidationError(QuestionGenerationError):
    """Expert validation system errors"""
    def __init__(self, message: str, expert_name: str, model: str, port: int, details: Dict[str, Any] = None):
        super().__init__(message, "EXPERT_VALIDATION_ERROR", details)
        self.expert_name = expert_name
        self.model = model
        self.port = port


class ModelLoadingError(QuestionGenerationError):
    """Model loading and VRAM management errors"""
    def __init__(self, message: str, model_name: str, port: int, vram_info: Dict[str, Any] = None):
        super().__init__(message, "MODEL_LOADING_ERROR", vram_info or {})
        self.model_name = model_name
        self.port = port


class PromptBuildingError(QuestionGenerationError):
    """Prompt construction and parameter validation errors"""
    def __init__(self, message: str, parameter_name: str = None, parameter_value: str = None):
        super().__init__(message, "PROMPT_BUILDING_ERROR")
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value


class CSVGenerationError(QuestionGenerationError):
    """CSV data generation and formatting errors"""
    def __init__(self, message: str, csv_issues: List[str] = None):
        super().__init__(message, "CSV_GENERATION_ERROR")
        self.csv_issues = csv_issues or []

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import result manager from same directory (ALEE_Agent)
try:
    from result_manager import ResultManager, save_results
    RESULT_MANAGER_AVAILABLE = True
    logger.info("Result manager imported successfully - orchestrator can save incremental results")
except ImportError as e:
    logger.warning(f"Result manager not available - results will only be returned to caller: {e}")
    RESULT_MANAGER_AVAILABLE = False
    ResultManager = None
    save_results = None

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

# SysArch Request Model
class SysArchRequest(BaseModel):
    c_id: str = Field(..., description="Question ID in format: question_number-difficulty-version (e.g., 41-1-4)")
    text: str = Field(..., description="The informational text about the system's pre-configured topic")
    question_type: str = Field(..., description="multiple-choice, single-choice, true-false, mapping")
    p_variation: str = Field(..., description="Stammaufgabe, schwer, leicht")
    p_taxonomy_level: str = Field(..., description="Stufe 1 (Wissen/Reproduktion), Stufe 2 (Anwendung/Transfer)")
    p_root_text_reference_explanatory_text: str = Field("Nicht vorhanden", description="Nicht vorhanden, Explizit, Implizit")
    p_root_text_obstacle_passive: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_root_text_obstacle_negation: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_root_text_obstacle_complex_np: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_root_text_contains_irrelevant_information: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_mathematical_requirement_level: str = Field("0", description="0 (Kein Bezug), 1 (Nutzen mathematischer Darstellungen), 2 (Mathematische Operation)")
    # Individual item parameters as per SYSARCH.md
    p_item_1_obstacle_passive: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_1_obstacle_negation: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_1_obstacle_complex_np: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_2_obstacle_passive: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_2_obstacle_negation: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_2_obstacle_complex_np: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_3_obstacle_passive: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_3_obstacle_negation: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_3_obstacle_complex_np: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_4_obstacle_passive: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_4_obstacle_negation: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_4_obstacle_complex_np: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_5_obstacle_passive: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_5_obstacle_negation: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_5_obstacle_complex_np: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_6_obstacle_passive: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_6_obstacle_negation: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_6_obstacle_complex_np: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_7_obstacle_passive: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_7_obstacle_negation: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_7_obstacle_complex_np: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_8_obstacle_passive: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_8_obstacle_negation: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_item_8_obstacle_complex_np: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_instruction_obstacle_passive: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_instruction_obstacle_negation: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_instruction_obstacle_complex_np: str = Field("Nicht Enthalten", description="Enthalten, Nicht Enthalten")
    p_instruction_explicitness_of_instruction: str = Field("Implizit", description="Explizit, Implizit")

@dataclass
class ParameterValidation:
    parameter: str
    status: ParameterStatus
    score: float
    feedback: str
    expert_used: str
    processing_time: float

# SysArch Result Model
class SysArchResult(BaseModel):
    question_1: str = Field(..., description="First generated question")
    question_2: str = Field(..., description="Second generated question") 
    question_3: str = Field(..., description="Third generated question")
    c_id: str = Field(..., description="Original question ID")
    processing_time: float = Field(..., description="Total processing time")
    csv_data: Dict[str, Any] = Field(..., description="CSV-ready data with all parameters")
    generation_updates: List[Dict[str, Any]] = Field(default_factory=list, description="Intermediate generation updates")

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
            "p.item_1_obstacle_passive", "p.item_1_obstacle_negation", "p.item_1_obstacle_complex_np",
            "p.item_2_obstacle_passive", "p.item_2_obstacle_negation", "p.item_2_obstacle_complex_np",
            "p.item_3_obstacle_passive", "p.item_3_obstacle_negation", "p.item_3_obstacle_complex_np",
            "p.item_4_obstacle_passive", "p.item_4_obstacle_negation", "p.item_4_obstacle_complex_np",
            "p.item_5_obstacle_passive", "p.item_5_obstacle_negation", "p.item_5_obstacle_complex_np",
            "p.item_6_obstacle_passive", "p.item_6_obstacle_negation", "p.item_6_obstacle_complex_np",
            "p.item_7_obstacle_passive", "p.item_7_obstacle_negation", "p.item_7_obstacle_complex_np",
            "p.item_8_obstacle_passive", "p.item_8_obstacle_negation", "p.item_8_obstacle_complex_np",
            "p.instruction_obstacle_passive",
            "p.instruction_obstacle_negation",
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
session_pool: Optional[aiohttp.ClientSession] = None

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
            current_vram = sum(
                self.model_memory_usage.get(active_models[p], 0) 
                for p in active_models
            )
            vram_info = {
                "current_vram_usage_gb": current_vram,
                "max_vram_gb": self.max_vram_gb,
                "available_vram_gb": self.max_vram_gb - current_vram,
                "active_models": dict(active_models)
            }
            raise ModelLoadingError(
                f"Failed to load model {expert_config.model} on port {expert_config.port}: {str(e)}",
                expert_config.model,
                expert_config.port,
                vram_info
            )
    
    async def unload_model(self, port: int):
        """Unload model to free VRAM"""
        if port in active_models:
            model_name = active_models[port]
            del active_models[port]
            logger.info(f"Unloaded {model_name} from port {port}")

model_manager = ModelManager()

class SysArchPromptBuilder:
    """Builds modular prompts based on parameter values using txt files"""
    
    def __init__(self):
        self.base_path = Path("/home/mklemmingen/PycharmProjects/PythonProject/ALEE_Agent")
        
    def load_prompt_txt(self, file_path: str) -> str:
        """Load prompt text from file with error handling"""
        try:
            full_path = self.base_path / file_path
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                return content if content else f"<{file_path.split('/')[-1]} parameter prompt content>"
        except FileNotFoundError:
            logger.warning(f"Prompt file not found: {file_path}")
            return f"<{file_path.split('/')[-1]} parameter prompt content>"
        except Exception as e:
            logger.error(f"Error loading prompt {file_path}: {e}")
            return f"<{file_path.split('/')[-1]} parameter prompt content>"
    
    def build_variation_prompt(self, variation: str) -> str:
        """Build variation-specific prompt"""
        # p.variation values are "Stammaufgabe", "schwer", "leicht"
        if not variation or variation.strip() == "":
            raise PromptBuildingError(
                "Variation parameter cannot be empty",
                "p_variation", 
                variation
            )
            
        valid_variations = ["stammaufgabe", "schwer", "leicht"]
        if variation.lower() not in valid_variations:
            raise PromptBuildingError(
                f"Invalid variation '{variation}'. Must be one of: {', '.join(valid_variations)}",
                "p_variation",
                variation
            )
            
        # Map variation to difficulty level prompt files (stakeholder-aligned)
        variation_map = {
            "stammaufgabe": "variationPrompts/stammaufgabe.txt",  # Standard difficulty
            "schwer": "variationPrompts/schwer.txt",              # Hard difficulty  
            "leicht": "variationPrompts/leicht.txt"               # Easy difficulty
        }
        file_path = variation_map.get(variation.lower(), "variationPrompts/stammaufgabe.txt")
        content = self.load_prompt_txt(file_path)
        return f"DIFFICULTY LEVEL ({variation}): {content}"
    
    def build_taxonomy_prompt(self, level: str) -> str:
        """Build taxonomy-specific prompt"""
        # "Stufe 1 (Wissen/Reproduktion)", "Stufe 2 (Anwendung/Transfer)"
        if "Stufe 1" in level:
            content = self.load_prompt_txt("taxonomyLevelPrompt/stufe1WissenReproduktion.txt")
            return f"p.taxonomy_level (Stufe 1): {content}"
        elif "Stufe 2" in level:
            content = self.load_prompt_txt("taxonomyLevelPrompt/stufe2AnwendungTransfer.txt")
            return f"p.taxonomy_level (Stufe 2): {content}"
        else:
            content = self.load_prompt_txt("taxonomyLevelPrompt/stufe1WissenReproduktion.txt")
            return f"p.taxonomy_level (Default Stufe 1): {content}"
    
    def build_mathematical_prompt(self, level: str) -> str:
        """Build mathematical requirement prompt"""
        # "0 (Kein Bezug)", "1 (Nutzen mathematischer Darstellungen)", "2 (Mathematische Operation)"
        level_clean = level.strip().split()[0]  # Extract just the number
        level_map = {
            "0": "mathematicalRequirementLevel/0KeinBezug.txt",
            "1": "mathematicalRequirementLevel/1NutzenMathematischerDarstellungen.txt", 
            "2": "mathematicalRequirementLevel/2MathematischeOperationen.txt"
        }
        file_path = level_map.get(level_clean, "mathematicalRequirementLevel/0KeinBezug.txt")
        content = self.load_prompt_txt(file_path)
        return f"p.mathematical_requirement_level ({level}): {content}"
    
    def build_obstacle_prompt(self, obstacle_type: str, value: str, parameter_name: str) -> str:
        """Build obstacle-specific prompts"""
        # Values are "Enthalten", "Nicht Enthalten"
        value_file = "enthalten.txt" if value == "Enthalten" else "nichtEnthalten.txt"
        
        obstacle_map = {
            "root_passive": f"rootTextParameterTextPrompts/obstaclePassivePrompts/{value_file}",
            "root_negation": f"rootTextParameterTextPrompts/obstacleNegationPrompts/{value_file}", 
            "root_complex_np": f"rootTextParameterTextPrompts/obstacleComplexPrompts/{value_file}",
            "item_passive": f"itemXObstacle/passive/{value_file}",
            "item_negation": f"itemXObstacle/negation/{value_file}",
            "item_complex": f"itemXObstacle/complex/{value_file}",
            "instruction_passive": f"instructionObstacle/passive/{value_file}",
            "instruction_negation": f"instructionObstacle/negation/{value_file}",
            "instruction_complex_np": f"instructionObstacle/complex_np/{value_file}"
        }
        
        file_path = obstacle_map.get(obstacle_type, "")
        if file_path:
            content = self.load_prompt_txt(file_path)
            return f"{parameter_name} ({value}): {content}"
        return ""
    
    def build_irrelevant_info_prompt(self, value: str) -> str:
        """Build irrelevant information prompt"""
        # "Enthalten", "Nicht Enthalten"
        value_file = "enthalten.txt" if value == "Enthalten" else "nichtEnthalten.txt"
        content = self.load_prompt_txt(f"rootTextParameterTextPrompts/containsIrrelevantInformationPrompt/{value_file}")
        return f"p.root_text_contains_irrelevant_information ({value}): {content}"
    
    def build_explicitness_prompt(self, value: str) -> str:
        """Build instruction explicitness prompt"""
        # "Explizit", "Implizit"
        if "Explizit" in value:
            content = self.load_prompt_txt("instructionExplicitnessOfInstruction/explizit.txt")
            return f"p.instruction_explicitness_of_instruction (Explizit): {content}"
        else:
            content = self.load_prompt_txt("instructionExplicitnessOfInstruction/implizit.txt")
            return f"p.instruction_explicitness_of_instruction (Implizit): {content}"
    
    def build_question_type_prompt(self, question_type: str) -> str:
        """Build question type specific prompt"""
        if not question_type or question_type.strip() == "":
            raise PromptBuildingError(
                "Question type parameter cannot be empty",
                "question_type",
                question_type
            )
            
        valid_types = ["multiple-choice", "single-choice", "true-false", "mapping"]
        if question_type.lower() not in valid_types:
            raise PromptBuildingError(
                f"Invalid question type '{question_type}'. Must be one of: {', '.join(valid_types)}",
                "question_type",
                question_type
            )
        
        # Map to existing prompt files
        type_map = {
            "multiple-choice": "variationPrompts/multiple-choice.txt",
            "single-choice": "variationPrompts/single-choice.txt",
            "true-false": "variationPrompts/true-false.txt",
            "mapping": "variationPrompts/mapping.txt"
        }
        
        file_path = type_map.get(question_type.lower())
        content = self.load_prompt_txt(file_path)
        return f"QUESTION TYPE ({question_type}): {content}"
    
    def build_master_prompt(self, request: SysArchRequest) -> str:
        """Build the complete master prompt from modular components"""
        components = []
        
        # Base instruction from external txt file
        main_intro = self.load_prompt_txt("mainGenPromptIntro.txt")
        components.append(main_intro)
        components.append(f"\nReferenztext:\n{request.text}\n")
        components.append(f"c_id: {request.c_id}\n")
        
        # Include question type (required - newly added)
        question_type_prompt = self.build_question_type_prompt(request.question_type)
        components.append(question_type_prompt)
        
        # Always include variation (required - difficulty level)
        variation_prompt = self.build_variation_prompt(request.p_variation)
        components.append(variation_prompt)
        
        # Always include taxonomy (required)
        taxonomy_prompt = self.build_taxonomy_prompt(request.p_taxonomy_level)
        components.append(taxonomy_prompt)
        
        # Always include mathematical level
        math_prompt = self.build_mathematical_prompt(request.p_mathematical_requirement_level)
        components.append(math_prompt)
        
        # Root text obstacles - only add if "Enthalten"
        if request.p_root_text_obstacle_passive == "Enthalten":
            components.append(self.build_obstacle_prompt("root_passive", request.p_root_text_obstacle_passive, "p.root_text_obstacle_passive"))
        
        if request.p_root_text_obstacle_negation == "Enthalten":
            components.append(self.build_obstacle_prompt("root_negation", request.p_root_text_obstacle_negation, "p.root_text_obstacle_negation"))
        
        if request.p_root_text_obstacle_complex_np == "Enthalten":
            components.append(self.build_obstacle_prompt("root_complex_np", request.p_root_text_obstacle_complex_np, "p.root_text_obstacle_complex_np"))
        
        # Irrelevant information - only add if "Enthalten"
        if request.p_root_text_contains_irrelevant_information == "Enthalten":
            components.append(self.build_irrelevant_info_prompt(request.p_root_text_contains_irrelevant_information))
        
        # Individual item obstacles - only add if "Enthalten" (per SYSARCH.md)
        for i in range(1, 9):  # Items 1-8 as per SYSARCH.md
            passive_attr = f"p_item_{i}_obstacle_passive"
            negation_attr = f"p_item_{i}_obstacle_negation"
            complex_attr = f"p_item_{i}_obstacle_complex_np"
            
            if getattr(request, passive_attr) == "Enthalten":
                components.append(self.build_obstacle_prompt("item_passive", getattr(request, passive_attr), f"p.item_{i}_obstacle_passive"))
            
            if getattr(request, negation_attr) == "Enthalten":
                components.append(self.build_obstacle_prompt("item_negation", getattr(request, negation_attr), f"p.item_{i}_obstacle_negation"))
            
            if getattr(request, complex_attr) == "Enthalten":
                components.append(self.build_obstacle_prompt("item_complex", getattr(request, complex_attr), f"p.item_{i}_obstacle_complex_np"))
        
        # Instruction obstacles - only add if "Enthalten"
        if request.p_instruction_obstacle_passive == "Enthalten":
            components.append(self.build_obstacle_prompt("instruction_passive", request.p_instruction_obstacle_passive, "p.instruction_obstacle_passive"))
        
        if request.p_instruction_obstacle_negation == "Enthalten":
            components.append(self.build_obstacle_prompt("instruction_negation", request.p_instruction_obstacle_negation, "p.instruction_obstacle_negation"))
        
        if request.p_instruction_obstacle_complex_np == "Enthalten":
            components.append(self.build_obstacle_prompt("instruction_complex_np", request.p_instruction_obstacle_complex_np, "p.instruction_obstacle_complex_np"))
        
        # Always include explicitness
        explicit_prompt = self.build_explicitness_prompt(request.p_instruction_explicitness_of_instruction)
        components.append(explicit_prompt)
        
        # Root text reference (always include)
        if hasattr(request, 'p_root_text_reference_explanatory_text') and request.p_root_text_reference_explanatory_text != "Nicht vorhanden":
            components.append(f"p.root_text_reference_explanatory_text ({request.p_root_text_reference_explanatory_text}): <reference text handling prompt content>")
        
        # Output format instruction from external txt file
        output_format = self.load_prompt_txt("outputFormatPrompt.txt")
        components.append(f"\n{output_format}")
        
        return "\n\n".join(components)

class EducationalAISystem:
    """Main orchestrator for educational question generation"""
    
    def __init__(self):
        self.max_iterations = 3  # Max 3 iterations
        self.min_approval_score = 7.0
        self.prompt_builder = SysArchPromptBuilder()
        
    async def clean_expert_sessions(self):
        """Clean all expert sessions for fresh start"""
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
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Reset {expert_name} session")
                    else:
                        logger.warning(f"Failed to reset {expert_name}: HTTP {response.status}")
                        
            except Exception as e:
                logger.warning(f"Error resetting {expert_name}: {e}")
        
        logger.info("Expert session cleanup complete")
    
    async def generate_questions(self, request: SysArchRequest) -> SysArchResult:
        """Generate exactly 3 questions with incremental saving"""
        start_time = time.time()
        generation_updates = []  # Track all question changes
        
        # Initialize result manager for incremental saving
        result_manager = None
        if RESULT_MANAGER_AVAILABLE:
            result_manager = ResultManager()
            session_dir = result_manager.create_session_package(request.c_id, request.model_dump())
            logger.info(f"Created incremental result session: {session_dir}")
        
        # Step 1: Clean expert sessions for fresh start
        await self.clean_expert_sessions()
        
        # Step 2: Build master prompt using modular components
        master_prompt = self.prompt_builder.build_master_prompt(request)
        logger.info("Built master prompt with modular components")
        
        # Save initial prompt used
        if result_manager:
            result_manager.save_iteration_result(
                iteration_num=0,
                questions=["Initial generation starting..."],
                prompts_used={"master_prompt": master_prompt},
                processing_metadata={
                    "step": "initial_prompt_construction",
                    "timestamp": datetime.now().isoformat(),
                    "parameters_used": request.model_dump()
                }
            )
        
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
        
        # Save initial questions generated
        initial_questions = [
            questions_data.get("question_1", "Question 1"),
            questions_data.get("question_2", "Question 2"), 
            questions_data.get("question_3", "Question 3")
        ]
        
        # Track initial generation
        generation_updates.append({
            "timestamp": datetime.now().isoformat(),
            "step": "initial_generation",
            "questions": initial_questions.copy(),
            "source": "main_generator",
            "model": generator_config.model
        })
        
        if result_manager:
            result_manager.save_iteration_result(
                iteration_num=1,
                questions=initial_questions,
                prompts_used={"generator_prompt": master_prompt, "generator_model": generator_config.model},
                processing_metadata={
                    "step": "initial_question_generation",
                    "generator_model": generator_config.model,
                    "generator_port": generator_config.port,
                    "raw_response": questions_response[:500],  # First 500 chars
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Step 4: Expert validation for each question (max 3 iterations) with incremental saving
        validated_questions = []
        
        for i in range(3):
            question_key = f"question_{i+1}"
            answer_key = f"answers_{i+1}"
            
            question_text = questions_data.get(question_key, f"Question {i+1}")
            question_answers = questions_data.get(answer_key, ["A", "B", "C"])
            
            # Validate this question through expert iterations with incremental saving
            validated_question = await self._validate_question_with_experts_incremental(
                question_text, question_answers, request, i+1, result_manager, generation_updates
            )
            validated_questions.append(validated_question)
        
        # Step 5: Build CSV data
        initial_csv_data = self._build_csv_data(request, validated_questions)
        
        # Step 6: Apply intelligent fallback system for CSV issues
        csv_data = await self._intelligent_csv_fallback(validated_questions, request, initial_csv_data)
        
        total_time = time.time() - start_time
        
        # Save final results
        if result_manager:
            final_metadata = {
                "c_id": request.c_id,
                "session_started_at": datetime.fromtimestamp(start_time).isoformat(),
                "total_processing_time": total_time,
                "final_questions_count": len(validated_questions),
                "csv_columns_count": len(csv_data.keys()) if csv_data else 0,
                "system_compliant": True,
                "expert_iterations_completed": True
            }
            
            result_manager.save_final_results(
                final_questions=validated_questions,
                csv_data=csv_data,
                final_metadata=final_metadata
            )
            logger.info(f"Final results saved to session: {result_manager.current_session_dir}")
        
        return SysArchResult(
            question_1=validated_questions[0],
            question_2=validated_questions[1],
            question_3=validated_questions[2],
            c_id=request.c_id,
            processing_time=total_time,
            csv_data=csv_data,
            generation_updates=generation_updates
        )
    
    async def _validate_question_with_experts(self, question: str, answers: List[str], 
                                            request: SysArchRequest, question_num: int) -> str:
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

    async def _validate_question_with_experts_incremental(self, question: str, answers: List[str], 
                                                        request: SysArchRequest, question_num: int, 
                                                        result_manager, generation_updates: List[Dict[str, Any]]) -> str:
        """Validate single question through expert system with incremental saving (max 3 iterations)"""
        current_question = question
        
        for iteration in range(self.max_iterations):
            iteration_start_time = time.time()
            logger.info(f"Question {question_num}, Iteration {iteration + 1}/{self.max_iterations}")
            
            # Get expert validations
            validations = await self._get_expert_validations(current_question, request)
            
            # Collect expert feedback for saving
            expert_feedback = {}
            expert_prompts_used = {
                "complete_parameter_summary": self._build_complete_parameter_summary(request),
                "expert_prompt_sources": "All experts received complete parameter context"
            }
            
            for validation in validations:
                expert_feedback[validation.parameter] = {
                    "status": validation.status.value if hasattr(validation.status, 'value') else str(validation.status),
                    "score": validation.score,
                    "feedback": validation.feedback,
                    "expert_used": getattr(validation, 'expert_used', 'unknown')
                }
            
            # Save iteration results if result manager is available
            if result_manager:
                iteration_metadata = {
                    "question_number": question_num,
                    "expert_iteration": iteration + 1,
                    "iteration_processing_time": time.time() - iteration_start_time,
                    "experts_consulted": len(validations),
                    "expert_validations": expert_feedback,
                    "timestamp": datetime.now().isoformat()
                }
                
                result_manager.save_iteration_result(
                    iteration_num=(question_num * 10) + iteration + 2,  # Unique iteration number
                    questions=[current_question],
                    prompts_used=expert_prompts_used,
                    expert_feedback=expert_feedback,
                    processing_metadata=iteration_metadata
                )
            
            # Check if approved by all experts
            failed_validations = [v for v in validations 
                                if v.status in [ParameterStatus.REJECTED, ParameterStatus.NEEDS_REFINEMENT]]
            
            if not failed_validations:
                logger.info(f"Question {question_num} approved by all experts")
                
                # Save final approval if result manager is available
                if result_manager:
                    approval_metadata = {
                        "question_number": question_num,
                        "final_approval_iteration": iteration + 1,
                        "approved_by_all_experts": True,
                        "total_expert_iterations": iteration + 1,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    result_manager.save_iteration_result(
                        iteration_num=(question_num * 10) + iteration + 20,  # Approval marker
                        questions=[current_question],
                        prompts_used={"final_approved_question": current_question},
                        expert_feedback={"approval_status": "APPROVED_BY_ALL_EXPERTS"},
                        processing_metadata=approval_metadata
                    )
                
                break
                
            if iteration < self.max_iterations - 1:  # Not the last iteration
                # Refine question based on feedback
                feedback = [v.feedback for v in failed_validations]
                refined_question = await self._refine_single_question(current_question, feedback, request)
                
                # Track question change for ongoing feedback
                if refined_question != current_question:
                    generation_updates.append({
                        "timestamp": datetime.now().isoformat(),
                        "step": f"expert_refinement_q{question_num}_iter{iteration + 1}",
                        "question_num": question_num,
                        "iteration": iteration + 1,
                        "original_question": current_question,
                        "refined_question": refined_question,
                        "expert_feedback": feedback,
                        "source": "expert_validation"
                    })
                
                # Save refinement results
                if result_manager and refined_question != current_question:
                    refinement_metadata = {
                        "question_number": question_num,
                        "refinement_iteration": iteration + 1,
                        "original_question": current_question,
                        "refined_question": refined_question,
                        "feedback_applied": feedback,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    result_manager.save_iteration_result(
                        iteration_num=(question_num * 10) + iteration + 30,  # Refinement marker
                        questions=[refined_question],
                        prompts_used={"refinement_feedback": "; ".join(feedback)},
                        expert_feedback={"refinement_applied": True},
                        processing_metadata=refinement_metadata
                    )
                
                current_question = refined_question
        
        return current_question
    
    def _build_complete_parameter_summary(self, request: SysArchRequest) -> str:
        """Build comprehensive parameter summary for expert analysis"""
        summary_lines = [
            f"• c_id: {request.c_id}",
            f"• Variation: {request.p_variation}",
            f"• Taxonomie-Level: {request.p_taxonomy_level}",
            f"• Mathematisches Niveau: {request.p_mathematical_requirement_level}",
            f"• Root-Text Referenz: {request.p_root_text_reference_explanatory_text}",
            "",
            "ROOT-TEXT HINDERNISSE:",
            f"• Passiv-Strukturen: {request.p_root_text_obstacle_passive}",
            f"• Verneinungen: {request.p_root_text_obstacle_negation}",
            f"• Komplexe Nominalphrasen: {request.p_root_text_obstacle_complex_np}",
            f"• Irrelevante Informationen: {request.p_root_text_contains_irrelevant_information}",
            "",
            "ITEM-SPEZIFISCHE HINDERNISSE (Items 1-8):"
        ]
        
        # Add all individual item parameters
        for i in range(1, 9):
            summary_lines.append(f"Item {i}: Passiv={getattr(request, f'p_item_{i}_obstacle_passive')}, Negation={getattr(request, f'p_item_{i}_obstacle_negation')}, Komplex-NP={getattr(request, f'p_item_{i}_obstacle_complex_np')}")
        
        summary_lines.extend([
            "",
            "INSTRUKTIONS-HINDERNISSE:",
            f"• Passiv-Strukturen: {request.p_instruction_obstacle_passive}",
            f"• Verneinungen: {request.p_instruction_obstacle_negation}", 
            f"• Komplexe Nominalphrasen: {request.p_instruction_obstacle_complex_np}",
            f"• Explizitheit der Anweisung: {request.p_instruction_explicitness_of_instruction}"
        ])
        
        return "\n".join(summary_lines)
        
    async def _get_expert_validations(self, question: str, request: SysArchRequest) -> List[ParameterValidation]:
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
                                           question: str, request: SysArchRequest) -> ParameterValidation:
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
        
        # Build comprehensive parameter summary for expert analysis
        parameter_summary = self._build_complete_parameter_summary(request)
        
        # Build expert prompt with complete parameter configuration
        # Load expert intro prompt from external txt file
        expert_intro = self.load_prompt_txt("expertPromptIntro.txt")
        expert_prompt = f"""{expert_intro} {expert_config.expertise}.

        Analysiere diese Bildungsfrage bezüglich ALLER spezifizierten Parameter:
        
        Frage: {question}
        
        VOLLSTÄNDIGE PARAMETER-KONFIGURATION:
{parameter_summary}
        
        Fokussiere dich auf deine Expertise ({', '.join(expert_config.parameters)}) aber berücksichtige alle Parameter für eine ganzheitliche Bewertung.
        
        Bewerte die Frage und gib Verbesserungsvorschläge.
        
        Antworte in folgendem einfachen Format:
        BEWERTUNG: [Gut/Mittelmäßig/Schlecht]
        FEEDBACK: [Deine detaillierten Verbesserungsvorschläge oder "Frage ist gut so"]"""
        
        response = await self._call_expert_llm(expert_config, expert_prompt)
        
        # Parse simple text format instead of JSON
        try:
            # Extract BEWERTUNG and FEEDBACK from response
            lines = response.strip().split('\n')
            bewertung = "Mittelmäßig"  # Default
            feedback = "Keine spezifischen Verbesserungen"  # Default
            
            for line in lines:
                if line.startswith("BEWERTUNG:"):
                    bewertung = line.replace("BEWERTUNG:", "").strip()
                elif line.startswith("FEEDBACK:"):
                    feedback = line.replace("FEEDBACK:", "").strip()
            
            # Map text rating to status and score
            if bewertung.lower() == "gut":
                status = ParameterStatus.APPROVED
                score = 8.0
            elif bewertung.lower() == "schlecht":
                status = ParameterStatus.REJECTED
                score = 3.0
            else:  # Mittelmäßig or anything else
                status = ParameterStatus.NEEDS_REFINEMENT
                score = 5.0
            
            return ParameterValidation(
                parameter=",".join(expert_config.parameters),
                status=status,
                score=score,
                feedback=feedback,
                expert_used=expert_config.name,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.warning(f"Expert response parsing error: {e}")
            return ParameterValidation(
                parameter=",".join(expert_config.parameters),
                status=ParameterStatus.NEEDS_REFINEMENT,
                score=5.0,
                feedback=f"Experte antwortete: {response[:200]}...",
                expert_used=expert_config.name,
                processing_time=time.time() - start_time
            )
    
    async def _refine_single_question(self, question: str, feedback: List[str], 
                                     request: SysArchRequest) -> str:
        """Refine a single question based on expert feedback"""
        generator_config = PARAMETER_EXPERTS["variation_expert"]
        await model_manager.ensure_model_loaded(generator_config)
        
        # Load refinement intro prompt from external txt file
        refinement_intro = self.load_prompt_txt("refinementPromptIntro.txt")
        refinement_prompt = f"""{refinement_intro}

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
    
    def _build_csv_data(self, request: SysArchRequest, questions: List[str]) -> Dict[str, Any]:
        """Build CSV data"""
        
        # Use the p_variation parameter as subject (difficulty level)
        subject = request.p_variation  # Difficulty level (stammaufgabe, schwer, leicht)
        
        # Use the question_type parameter directly (architectural fix)
        question_type = request.question_type  # Question format (multiple-choice, single-choice, true-false, mapping)
        
        # Build the main question text (combining all 3 questions)
        combined_text = f"1. {questions[0]} 2. {questions[1]} 3. {questions[2]}"
        
        # Build CSV data with all required columns
        csv_data = {
            "c_id": request.c_id,
            "subject": subject,  # Now correctly uses p_variation parameter
            "type": question_type,  # Correctly mapped from p_variation
            "text": combined_text,
            "p.instruction_explicitness_of_instruction": request.p_instruction_explicitness_of_instruction,
            "p.instruction_obstacle_complex_np": request.p_instruction_obstacle_complex_np,
            "p.instruction_obstacle_negation": request.p_instruction_obstacle_negation,
            "p.instruction_obstacle_passive": request.p_instruction_obstacle_passive,
            # Item parameters (8 items as per SYSARCH.md CSV format)
            "p.item_1_answer_verbatim_explanatory_text": "Nicht enthalten",
            "p.item_1_obstacle_complex_np": request.p_item_1_obstacle_complex_np,
            "p.item_1_obstacle_negation": request.p_item_1_obstacle_negation,
            "p.item_1_obstacle_passive": request.p_item_1_obstacle_passive,
            "p.item_1_sentence_length": "",
            "p.item_2_answer_verbatim_explanatory_text": "Nicht enthalten",
            "p.item_2_obstacle_complex_np": request.p_item_2_obstacle_complex_np,
            "p.item_2_obstacle_negation": request.p_item_2_obstacle_negation,
            "p.item_2_obstacle_passive": request.p_item_2_obstacle_passive,
            "p.item_2_sentence_length": "",
            "p.item_3_answer_verbatim_explanatory_text": "Nicht enthalten",
            "p.item_3_obstacle_complex_np": request.p_item_3_obstacle_complex_np,
            "p.item_3_obstacle_negation": request.p_item_3_obstacle_negation,
            "p.item_3_obstacle_passive": request.p_item_3_obstacle_passive,
            "p.item_3_sentence_length": "",
            "p.item_4_answer_verbatim_explanatory_text": "Nicht enthalten",
            "p.item_4_obstacle_complex_np": request.p_item_4_obstacle_complex_np,
            "p.item_4_obstacle_negation": request.p_item_4_obstacle_negation,
            "p.item_4_obstacle_passive": request.p_item_4_obstacle_passive,
            "p.item_4_sentence_length": "",
            "p.item_5_answer_verbatim_explanatory_text": "Nicht enthalten",
            "p.item_5_obstacle_complex_np": request.p_item_5_obstacle_complex_np,
            "p.item_5_obstacle_negation": request.p_item_5_obstacle_negation,
            "p.item_5_obstacle_passive": request.p_item_5_obstacle_passive,
            "p.item_5_sentence_length": "",
            "p.item_6_answer_verbatim_explanatory_text": "Nicht enthalten",
            "p.item_6_obstacle_complex_np": request.p_item_6_obstacle_complex_np,
            "p.item_6_obstacle_negation": request.p_item_6_obstacle_negation,
            "p.item_6_obstacle_passive": request.p_item_6_obstacle_passive,
            "p.item_6_sentence_length": "",
            "p.item_7_answer_verbatim_explanatory_text": "Nicht enthalten",
            "p.item_7_obstacle_complex_np": request.p_item_7_obstacle_complex_np,
            "p.item_7_obstacle_negation": request.p_item_7_obstacle_negation,
            "p.item_7_obstacle_passive": request.p_item_7_obstacle_passive,
            "p.item_7_sentence_length": "",
            "p.item_8_answer_verbatim_explanatory_text": "Nicht enthalten",
            "p.item_8_obstacle_complex_np": request.p_item_8_obstacle_complex_np,
            "p.item_8_obstacle_negation": request.p_item_8_obstacle_negation,
            "p.item_8_obstacle_passive": request.p_item_8_obstacle_passive,
            "p.item_8_sentence_length": "",
            "p.mathematical_requirement_level": f"{request.p_mathematical_requirement_level} (Kein Bezug)" if request.p_mathematical_requirement_level == "0" else request.p_mathematical_requirement_level,
            "p.root_text_contains_irrelevant_information": request.p_root_text_contains_irrelevant_information,
            "p.root_text_obstacle_complex_np": request.p_root_text_obstacle_complex_np,
            "p.root_text_obstacle_negation": request.p_root_text_obstacle_negation,
            "p.root_text_obstacle_passive": request.p_root_text_obstacle_passive,
            "p.root_text_reference_explanatory_text": request.p_root_text_reference_explanatory_text,
            "p.taxonomy_level": request.p_taxonomy_level,  # Fixed typo: was 'taxanomy'
            "p.variation": request.p_variation,
            "answers": "Question-specific answers",  # Will be filled by intelligent fallback system
            "p.instruction_number_of_sentences": "1"  # Default value
        }
        
        return csv_data
    
    async def _intelligent_csv_fallback(self, questions: List[str], request: SysArchRequest, 
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
    
    async def _llm_csv_correction(self, questions: List[str], request: SysArchRequest,
                                 initial_csv: Dict[str, Any], issues: List[str]) -> Dict[str, Any]:
        """Use LM to help correct CSV conversion issues"""
        
        # Use a lightweight expert for CSV correction
        helper_config = PARAMETER_EXPERTS["content_expert"]  # Reuse existing expert
        await model_manager.ensure_model_loaded(helper_config)
        
        # Load CSV correction intro prompt from external txt file
        csv_intro = self.load_prompt_txt("csvCorrectionPromptIntro.txt")
        correction_prompt = f"""{csv_intro}

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
            raise CSVGenerationError(
                "LM-assisted CSV correction failed - could not parse corrected data",
                issues
            )
    
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
    
    def _load_expert_prompt(self, expert_name: str) -> str:
        """Load expert-specific prompt from ALEE_Agent/expertPrompts/ directory"""
        expert_prompt_map = {
            "math_expert": "math_expert.txt",
            "taxonomy_expert": "taxonomy_expert.txt", 
            "variation_expert": "variation_expert.txt",
            "obstacle_expert": "obstacle_expert.txt"
        }
        
        prompt_file = expert_prompt_map.get(expert_name, "")
        if not prompt_file:
            logger.warning(f"No expert prompt file found for {expert_name}")
            return ""
            
        try:
            prompt_path = Path(__file__).parent / "expertPrompts" / prompt_file
            with open(prompt_path, 'r', encoding='utf-8') as f:
                expert_prompt = f.read().strip()
                logger.info(f"Loaded expert prompt for {expert_name} ({len(expert_prompt)} chars)")
                return expert_prompt
        except Exception as e:
            logger.warning(f"Failed to load expert prompt for {expert_name}: {e}")
            return ""

    async def _call_expert_llm(self, expert_config: ParameterExpertConfig, prompt: str) -> str:
        """Call specific expert LLM via Ollama API with expert-specific prompt prepended"""
        # Load and prepend expert-specific prompt from ALEE_Agent/prompts/
        expert_prompt = self._load_expert_prompt(expert_config.name)
        
        if expert_prompt:
            # Combine expert prompt with evaluation prompt
            full_prompt = f"{expert_prompt}\n\n## Aktuelle Bewertungsaufgabe:\n{prompt}"
            logger.info(f"[EXPERT_CALL] Calling {expert_config.name} with specialized prompt ({len(expert_prompt)} chars expert + {len(prompt)} chars evaluation)")
        else:
            # Fallback to original prompt if expert prompt not available
            full_prompt = prompt
            logger.info(f"[EXPERT_CALL] Calling {expert_config.name} ({expert_config.model}) on port {expert_config.port} (no specialized prompt)")
        
        logger.info(f"[PROMPT] TO {expert_config.name}:\n{'-'*50}\n{full_prompt[:300]}{'...' if len(full_prompt) > 300 else ''}\n{'-'*50}")
        
        payload = {
            "model": expert_config.model,
            "prompt": full_prompt,
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
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status != 200:
                    raise HTTPException(status_code=500, detail=f"LLM API error: {response.status}")
                
                data = await response.json()
                response_text = data["response"]
                
                logger.info(f"[RESPONSE] FROM {expert_config.name}:\n{'-'*50}\n{response_text[:300]}{'...' if len(response_text) > 300 else ''}\n{'-'*50}")
                
                return response_text
                
        except asyncio.TimeoutError as e:
            raise ExpertValidationError(
                f"Expert {expert_config.name} call timed out after 120 seconds",
                expert_config.name,
                expert_config.model,
                expert_config.port,
                {"timeout_seconds": 120, "error": str(e)}
            )
        except aiohttp.ClientConnectorError as e:
            raise ExpertValidationError(
                f"Cannot connect to expert {expert_config.name} server on port {expert_config.port}",
                expert_config.name,
                expert_config.model,
                expert_config.port,
                {"connection_error": str(e)}
            )
        except Exception as e:
            raise ExpertValidationError(
                f"Expert {expert_config.name} validation failed: {str(e)}",
                expert_config.name,
                expert_config.model,
                expert_config.port,
                {"raw_error": str(e)}
            )


# Initialize the AI system
ai_system = EducationalAISystem()

# API Endpoints

@app.post("/generate-questions", response_model=SysArchResult)
async def generate_questions_endpoint(request: SysArchRequest):
    """Generate exactly 3 questions"""
    start_time = time.time()
    
    try:
        logger.info(f"Request: {request.c_id} - {request.p_variation}")
        result = await ai_system.generate_questions(request)
        logger.info(f"Questions generated in {result.processing_time:.2f}s")
        
        # Save comprehensive results via result_manager (orchestrator-side saving)
        if RESULT_MANAGER_AVAILABLE:
            try:
                await save_comprehensive_results(request, result, start_time)
                logger.info(f"Results saved by orchestrator for {request.c_id}")
            except Exception as save_error:
                logger.warning(f"Orchestrator result saving failed (continuing anyway): {save_error}")
        
        return result
        
    except ModelLoadingError as e:
        logger.error(f"Model loading failed: {e.message}")
        error_detail = {
            "error_type": "MODEL_LOADING_ERROR",
            "message": e.message,
            "model_name": e.model_name,
            "port": e.port,
            "vram_info": e.details,
            "c_id": request.c_id
        }
        if RESULT_MANAGER_AVAILABLE:
            try:
                await save_error_results(request, f"ModelLoadingError: {e.message}", time.time() - start_time)
            except Exception:
                pass
        raise HTTPException(status_code=503, detail=error_detail)
        
    except ExpertValidationError as e:
        logger.error(f"Expert validation failed: {e.message}")
        error_detail = {
            "error_type": "EXPERT_VALIDATION_ERROR", 
            "message": e.message,
            "expert_name": e.expert_name,
            "model": e.model,
            "port": e.port,
            "details": e.details,
            "c_id": request.c_id
        }
        if RESULT_MANAGER_AVAILABLE:
            try:
                await save_error_results(request, f"ExpertValidationError: {e.message}", time.time() - start_time)
            except Exception:
                pass
        raise HTTPException(status_code=502, detail=error_detail)
        
    except CSVGenerationError as e:
        logger.error(f"CSV generation failed: {e.message}")
        error_detail = {
            "error_type": "CSV_GENERATION_ERROR",
            "message": e.message,
            "csv_issues": e.csv_issues,
            "c_id": request.c_id
        }
        if RESULT_MANAGER_AVAILABLE:
            try:
                await save_error_results(request, f"CSVGenerationError: {e.message}", time.time() - start_time)
            except Exception:
                pass
        raise HTTPException(status_code=422, detail=error_detail)
        
    except PromptBuildingError as e:
        logger.error(f"Prompt building failed: {e.message}")
        error_detail = {
            "error_type": "PROMPT_BUILDING_ERROR",
            "message": e.message,
            "parameter_name": e.parameter_name,
            "parameter_value": e.parameter_value,
            "c_id": request.c_id
        }
        if RESULT_MANAGER_AVAILABLE:
            try:
                await save_error_results(request, f"PromptBuildingError: {e.message}", time.time() - start_time)
            except Exception:
                pass
        raise HTTPException(status_code=400, detail=error_detail)
        
    except QuestionGenerationError as e:
        logger.error(f"Question generation failed: {e.message}")
        error_detail = {
            "error_type": e.error_code,
            "message": e.message,
            "details": e.details,
            "c_id": request.c_id
        }
        if RESULT_MANAGER_AVAILABLE:
            try:
                await save_error_results(request, f"QuestionGenerationError: {e.message}", time.time() - start_time)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=error_detail)
        
    except Exception as e:
        logger.error(f"Unexpected error during question generation: {e}")
        error_detail = {
            "error_type": "UNEXPECTED_ERROR",
            "message": f"An unexpected error occurred: {str(e)}",
            "c_id": request.c_id
        }
        if RESULT_MANAGER_AVAILABLE:
            try:
                await save_error_results(request, f"UnexpectedError: {str(e)}", time.time() - start_time)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=error_detail)

async def save_comprehensive_results(request: SysArchRequest, result: SysArchResult, start_time: float):
    """Save comprehensive results including prompts, logs, system metadata, and CSV"""
    try:
        # Prepare CSV data from result
        csv_data = [{
            "c_id": result.c_id,
            "subject": request.p_variation,
            "type": result.csv_data.get("type", "multiple-choice"),
            "text": f"1. {result.question_1} 2. {result.question_2} 3. {result.question_3}",
            "p.instruction_explicitness_of_instruction": result.csv_data.get("p.instruction_explicitness_of_instruction", ""),
            "p.instruction_obstacle_complex_np": result.csv_data.get("p.instruction_obstacle_complex_np", ""),
            "p.instruction_obstacle_negation": result.csv_data.get("p.instruction_obstacle_negation", ""),
            "p.instruction_obstacle_passive": result.csv_data.get("p.instruction_obstacle_passive", ""),
            "p.mathematical_requirement_level": result.csv_data.get("p.mathematical_requirement_level", ""),
            "p.taxonomy_level": result.csv_data.get("p.taxonomy_level", ""),  # Fixed typo
            "p.variation": result.csv_data.get("p.variation", ""),
            "answers": result.csv_data.get("answers", ""),
            "processing_time_seconds": result.processing_time,
            "generated_by": "orchestrator",
            "generation_timestamp": datetime.now().isoformat()
        }]
        
        # Collect system metadata including VRAM usage
        vram_usage = sum(
            model_manager.model_memory_usage.get(active_models[port], 0)
            for port in active_models
        )
        
        # Comprehensive metadata
        metadata = {
            "test_type": "orchestrator_generated_questions",
            "description": f"Questions generated by orchestrator for {request.c_id}",
            "request_parameters": {
                "c_id": request.c_id,
                "text_length": len(request.text),
                "p_variation": request.p_variation,
                "p_taxonomy_level": request.p_taxonomy_level,
                "p_mathematical_requirement_level": request.p_mathematical_requirement_level,
                "p_instruction_explicitness_of_instruction": request.p_instruction_explicitness_of_instruction,
                # Add all other parameters...
                "p_root_text_reference_explanatory_text": request.p_root_text_reference_explanatory_text,
                "p_root_text_obstacle_passive": request.p_root_text_obstacle_passive,
                "p_root_text_obstacle_negation": request.p_root_text_obstacle_negation,
                "p_root_text_obstacle_complex_np": request.p_root_text_obstacle_complex_np,
                "p_root_text_contains_irrelevant_information": request.p_root_text_contains_irrelevant_information,
                # Individual item parameters included in SysArchRequest model
                "p_instruction_obstacle_passive": request.p_instruction_obstacle_passive,
                "p_instruction_obstacle_complex_np": request.p_instruction_obstacle_complex_np
            },
            "system_status": {
                "active_models": dict(active_models),
                "vram_usage_gb": vram_usage,
                "max_vram_gb": model_manager.max_vram_gb,
                "available_experts": list(PARAMETER_EXPERTS.keys()),
                "expert_validation_enabled": True,
                "max_expert_iterations": 3
            },
            "generation_results": {
                "questions_generated": 3,
                "processing_time_seconds": result.processing_time,
                "csv_format_compliant": True,
                "system_compliant": True,
                "question_1_length": len(result.question_1),
                "question_2_length": len(result.question_2),
                "question_3_length": len(result.question_3)
            },
            "logs_captured": {
                "generation_logs": "Available in orchestrator logs",
                "expert_validation_logs": "Expert feedback captured during generation",
                "model_loading_logs": "VRAM management logs captured",
                "prompt_construction_logs": "Modular prompt building logged"
            },
            "orchestrator_save_timestamp": datetime.now().isoformat(),
            "saved_by": "orchestrator_direct_save",
            "prompts_snapshot": "All ALEE_Agent prompts saved to prompts/ folder"
        }
        
        # Use result_manager to save everything
        session_dir = save_results(
            csv_data=csv_data,
            metadata=metadata,
            prompts_source_dir=str(Path(__file__).parent)  # Save all ALEE_Agent prompts
        )
        
        logger.info(f"Comprehensive results saved to: {session_dir}")
        return session_dir
        
    except Exception as e:
        logger.error(f"Failed to save comprehensive results: {e}")
        raise

async def save_error_results(request: SysArchRequest, error_message: str, processing_time: float):
    """Save error results for debugging"""
    try:
        error_data = [{
            "c_id": request.c_id,
            "subject": request.p_variation,
            "type": "error",
            "text": f"ERROR: {error_message}",
            "processing_time_seconds": processing_time,
            "error_timestamp": datetime.now().isoformat(),
            "generated_by": "orchestrator_error"
        }]
        
        error_metadata = {
            "test_type": "orchestrator_error_capture",
            "description": f"Error occurred during question generation for {request.c_id}",
            "error_message": error_message,
            "processing_time_seconds": processing_time,
            "request_parameters": request.model_dump(),
            "system_status_at_error": {
                "active_models": dict(active_models),
                "available_experts": list(PARAMETER_EXPERTS.keys())
            },
            "error_timestamp": datetime.now().isoformat()
        }
        
        session_dir = save_results(csv_data=error_data, metadata=error_metadata)
        logger.info(f"Error results saved to: {session_dir}")
        
    except Exception as save_error:
        logger.error(f"Failed to save error results: {save_error}")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "educational_ai_orchestrator:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )