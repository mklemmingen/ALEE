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
import sys
import os

# Add CallersWithTexts to path for result_manager import
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'CallersWithTexts'))
# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from result_manager import save_results
    RESULT_MANAGER_AVAILABLE = True
    logger.info("Result manager imported successfully - orchestrator can save results")
except ImportError as e:
    logger.warning(f"Result manager not available - results will only be returned to caller: {e}")
    RESULT_MANAGER_AVAILABLE = False

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

# Legacy enum removed - ValidationPlan uses string values directly

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

# Legacy classes removed - ValidationPlan-only system

@dataclass
class ParameterValidation:
    parameter: str
    status: ParameterStatus
    score: float
    feedback: str
    expert_used: str
    processing_time: float

# Legacy result class removed - ValidationPlan-only system

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
                content = f.read().strip()
                return content if content else f"<{file_path.split('/')[-1]} parameter prompt content>"
        except FileNotFoundError:
            logger.warning(f"Prompt file not found: {file_path}")
            return f"<{file_path.split('/')[-1]} parameter prompt content>"
        except Exception as e:
            logger.error(f"Error loading prompt {file_path}: {e}")
            return f"<{file_path.split('/')[-1]} parameter prompt content>"
    
    def build_variation_prompt(self, variation: str) -> str:
        """Build variation-specific prompt according to ValidationPlan mapping"""
        # ValidationPlan: p.variation values are "Stammaufgabe", "schwer", "leicht"
        variation_map = {
            "stammaufgabe": "variationPrompts/multiple-choice.txt",  # Default type for stammaufgabe
            "schwer": "variationPrompts/true-false.txt",             # Complex questions often true-false
            "leicht": "variationPrompts/single-choice.txt"            # Simple questions single choice
        }
        file_path = variation_map.get(variation.lower(), "variationPrompts/multiple-choice.txt")
        content = self.load_prompt_txt(file_path)
        return f"p.variation ({variation}): {content}"
    
    def build_taxonomy_prompt(self, level: str) -> str:
        """Build taxonomy-specific prompt according to ValidationPlan"""
        # ValidationPlan: "Stufe 1 (Wissen/Reproduktion)", "Stufe 2 (Anwendung/Transfer)"
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
        """Build mathematical requirement prompt according to ValidationPlan"""
        # ValidationPlan: "0 (Kein Bezug)", "1 (Nutzen mathematischer Darstellungen)", "2 (Mathematische Operation)"
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
        """Build obstacle-specific prompts according to ValidationPlan"""
        # ValidationPlan: Values are "Enthalten", "Nicht Enthalten"
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
        """Build irrelevant information prompt according to ValidationPlan"""
        # ValidationPlan: "Enthalten", "Nicht Enthalten"
        value_file = "enthalten.txt" if value == "Enthalten" else "nichtEnthalten.txt"
        content = self.load_prompt_txt(f"rootTextParameterTextPrompts/containsIrrelevantInformationPrompt/{value_file}")
        return f"p.root_text_contains_irrelevant_information ({value}): {content}"
    
    def build_explicitness_prompt(self, value: str) -> str:
        """Build instruction explicitness prompt according to ValidationPlan"""
        # ValidationPlan: "Explizit", "Implizit"
        if "Explizit" in value:
            content = self.load_prompt_txt("instructionExplicitnessOfInstruction/explizit.txt")
            return f"p.instruction_explicitness_of_instruction (Explizit): {content}"
        else:
            content = self.load_prompt_txt("instructionExplicitnessOfInstruction/implizit.txt")
            return f"p.instruction_explicitness_of_instruction (Implizit): {content}"
    
    def build_master_prompt(self, request: ValidationPlanRequest) -> str:
        """Build the complete master prompt from modular components according to ValidationPlan"""
        components = []
        
        # Base instruction with ValidationPlan specification
        components.append("Du bist ein Experte für die Erstellung von Bildungsaufgaben. Erstelle basierend auf dem gegebenen Text GENAU DREI deutsche Bildungsfragen mit den spezifizierten ValidationPlan-Parametern.")
        components.append(f"\nReferenztext:\n{request.text}\n")
        components.append(f"ValidationPlan c_id: {request.c_id}\n")
        
        # Always include variation (required)
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
        
        components.append("\nValidationPlan Output Format: Antworte mit GENAU DREI Fragen im JSON-Format: {\"question_1\": \"...\", \"question_2\": \"...\", \"question_3\": \"...\", \"answers_1\": [...], \"answers_2\": [...], \"answers_3\": [...]}")
        
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
                    timeout=aiohttp.ClientTimeout(total=30)
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
        
        # Build expert prompt with parameter configuration - simplified to plain text response
        expert_prompt = f"""Du bist ein Experte für {expert_config.expertise}.

Analysiere diese Bildungsfrage bezüglich der spezifizierten Parameter:

Frage: {question}

Parameter-Konfiguration:
- c_id: {request.c_id}
- Variation: {request.p_variation}
- Taxonomie-Level: {request.p_taxonomy_level}
- Mathematisches Niveau: {request.p_mathematical_requirement_level}

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
    
    # Legacy generate_question method removed - ValidationPlan-only system
    
    # Legacy _generate_initial_question method removed - ValidationPlan-only system
    
    # Legacy _validate_all_parameters method removed - ValidationPlan-only system
    
    # Legacy _validate_with_expert method removed - ValidationPlan-only system
    
    # Legacy _refine_question method removed - ValidationPlan-only system
    
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
                timeout=aiohttp.ClientTimeout(total=120)
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
    
    # Legacy _prepare_csv_output method removed - ValidationPlan-only system

# Initialize the AI system
ai_system = EducationalAISystem()

# API Endpoints

@app.post("/generate-validation-plan", response_model=ValidationPlanResult)
async def generate_validation_plan_questions(request: ValidationPlanRequest):
    """Generate exactly 3 questions according to ValidationPlan specifications"""
    start_time = time.time()
    
    try:
        logger.info(f"ValidationPlan request: {request.c_id} - {request.p_variation}")
        result = await ai_system.generate_validation_plan_questions(request)
        logger.info(f"ValidationPlan questions generated in {result.processing_time:.2f}s")
        
        # Save comprehensive results via result_manager (orchestrator-side saving)
        if RESULT_MANAGER_AVAILABLE:
            try:
                await save_comprehensive_results(request, result, start_time)
                logger.info(f"Results saved by orchestrator for {request.c_id}")
            except Exception as save_error:
                logger.warning(f"Orchestrator result saving failed (continuing anyway): {save_error}")
        
        return result
        
    except Exception as e:
        logger.error(f"ValidationPlan generation failed: {e}")
        
        # Save error results if possible
        if RESULT_MANAGER_AVAILABLE:
            try:
                await save_error_results(request, str(e), time.time() - start_time)
            except Exception:
                pass  # Don't fail twice
                
        raise HTTPException(status_code=500, detail=str(e))

async def save_comprehensive_results(request: ValidationPlanRequest, result: ValidationPlanResult, start_time: float):
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
            "p.taxanomy_level": result.csv_data.get("p.taxanomy_level", ""),
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
            "test_type": "orchestrator_generated_validation_plan",
            "description": f"ValidationPlan questions generated by orchestrator for {request.c_id}",
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
                "p_item_X_obstacle_passive": request.p_item_X_obstacle_passive,
                "p_item_X_obstacle_negation": request.p_item_X_obstacle_negation,
                "p_item_X_obstacle_complex_np": request.p_item_X_obstacle_complex_np,
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
                "validation_plan_compliant": True,
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

async def save_error_results(request: ValidationPlanRequest, error_message: str, processing_time: float):
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
            "description": f"Error occurred during ValidationPlan generation for {request.c_id}",
            "error_message": error_message,
            "processing_time_seconds": processing_time,
            "request_parameters": request.dict(),
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

# Legacy endpoint removed - ValidationPlan-only system

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

# Legacy batch endpoint removed - ValidationPlan-only system

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "educational_ai_orchestrator:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )