"""
DSPy-Enhanced Educational AI Orchestrator
Single-pass consensus architecture using modular .txt prompts
Replaces complex iteration system with intelligent expert consensus
"""

import logging
import time
from typing import Dict, List, Any

from fastapi import FastAPI, HTTPException
# Define request/response models locally
from pydantic import BaseModel, Field

from educational_modules import GermanEducationalPipeline, ModularPromptBuilder
from result_manager import ResultManager
# Import existing components
from system_configuration import configure_dspy_with_ollama


class SysArchRequest(BaseModel):
    c_id: str = Field(..., description="Question ID")
    text: str = Field(..., description="Educational text")
    question_type: str = Field(..., description="Question format")
    p_variation: str = Field(..., description="Difficulty level")
    p_taxonomy_level: str = Field(..., description="Bloom's taxonomy level")
    p_mathematical_requirement_level: str = Field("0", description="Math complexity")
    p_root_text_reference_explanatory_text: str = Field("Nicht vorhanden")
    p_root_text_obstacle_passive: str = Field("Nicht Enthalten")
    p_root_text_obstacle_negation: str = Field("Nicht Enthalten")
    p_root_text_obstacle_complex_np: str = Field("Nicht Enthalten")
    p_root_text_contains_irrelevant_information: str = Field("Nicht Enthalten")
    # Item parameters
    p_item_1_obstacle_passive: str = Field("Nicht Enthalten")
    p_item_1_obstacle_negation: str = Field("Nicht Enthalten")
    p_item_1_obstacle_complex_np: str = Field("Nicht Enthalten")
    p_item_2_obstacle_passive: str = Field("Nicht Enthalten")
    p_item_2_obstacle_negation: str = Field("Nicht Enthalten")
    p_item_2_obstacle_complex_np: str = Field("Nicht Enthalten")
    p_item_3_obstacle_passive: str = Field("Nicht Enthalten")
    p_item_3_obstacle_negation: str = Field("Nicht Enthalten")
    p_item_3_obstacle_complex_np: str = Field("Nicht Enthalten")
    p_item_4_obstacle_passive: str = Field("Nicht Enthalten")
    p_item_4_obstacle_negation: str = Field("Nicht Enthalten")
    p_item_4_obstacle_complex_np: str = Field("Nicht Enthalten")
    p_item_5_obstacle_passive: str = Field("Nicht Enthalten")
    p_item_5_obstacle_negation: str = Field("Nicht Enthalten")
    p_item_5_obstacle_complex_np: str = Field("Nicht Enthalten")
    p_item_6_obstacle_passive: str = Field("Nicht Enthalten")
    p_item_6_obstacle_negation: str = Field("Nicht Enthalten")
    p_item_6_obstacle_complex_np: str = Field("Nicht Enthalten")
    p_item_7_obstacle_passive: str = Field("Nicht Enthalten")
    p_item_7_obstacle_negation: str = Field("Nicht Enthalten")
    p_item_7_obstacle_complex_np: str = Field("Nicht Enthalten")
    p_item_8_obstacle_passive: str = Field("Nicht Enthalten")
    p_item_8_obstacle_negation: str = Field("Nicht Enthalten")
    p_item_8_obstacle_complex_np: str = Field("Nicht Enthalten")
    # Instruction parameters
    p_instruction_obstacle_passive: str = Field("Nicht Enthalten")
    p_instruction_obstacle_complex_np: str = Field("Nicht Enthalten")
    p_instruction_explicitness_of_instruction: str = Field("Implizit")

class SysArchResult(BaseModel):
    question_1: str = Field(..., description="First generated question")
    question_2: str = Field(..., description="Second generated question") 
    question_3: str = Field(..., description="Third generated question")
    c_id: str = Field(..., description="Original question ID")
    processing_time: float = Field(..., description="Total processing time")
    csv_data: Dict[str, Any] = Field(..., description="CSV-ready data")
    generation_updates: List[Dict[str, Any]] = Field(default_factory=list)

logger = logging.getLogger(__name__)

class DSPyEducationalSystem:
    """DSPy-enhanced educational system using modular .txt prompts"""
    
    def __init__(self):
        self.prompt_builder = ModularPromptBuilder()
        self.main_lm = None
        self.expert_lms = None
        self.pipeline = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize DSPy system with OLLAMA multi-server setup"""
        if self.initialized:
            return
        
        logger.info("Initializing DSPy educational system")
        
        try:
            # Configure DSPy with existing OLLAMA servers
            self.main_lm, self.expert_lms = configure_dspy_with_ollama()
            
            # Create German educational pipeline using modular prompts
            self.pipeline = GermanEducationalPipeline(
                expert_lms=self.expert_lms,
                prompt_builder=self.prompt_builder
            )
            
            self.initialized = True
            logger.info("DSPy system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DSPy system: {e}")
            raise RuntimeError(f"DSPy initialization failed: {e}")
    
    async def generate_questions_dspy(self, request: SysArchRequest) -> SysArchResult:
        """
        Generate questions using DSPy single-pass consensus architecture
        Preserves existing result_manager integration and SYSARCH compliance
        """
        start_time = time.time()
        
        if not self.initialized:
            await self.initialize()
        
        logger.info(f"DSPy generation starting for {request.c_id}")
        
        try:
            # Convert request to parameter dictionary
            request_params = request.model_dump()
            
            # Run DSPy pipeline (single-pass: generation → experts → consensus)
            pipeline_result = self.pipeline(request_params)
            
            # Extract validated questions
            validated_questions = pipeline_result['validated_questions']
            
            if len(validated_questions) != 3:
                raise ValueError(f"Expected 3 questions, got {len(validated_questions)}")
            
            # Build SYSARCH-compliant CSV data
            csv_data = self._build_sysarch_csv(validated_questions, request_params)
            
            # Create generation updates for compatibility
            generation_updates = [
                {
                    "iteration": 1,
                    "questions": [q['question'] for q in validated_questions],
                    "expert_consensus": [q['consensus']['approved'] for q in validated_questions],
                    "dspy_metadata": {
                        "modular_prompts_used": True,
                        "single_pass_consensus": True,
                        "all_approved": pipeline_result['all_approved']
                    }
                }
            ]
            
            processing_time = time.time() - start_time
            
            result = SysArchResult(
                question_1=validated_questions[0]['question'],
                question_2=validated_questions[1]['question'],
                question_3=validated_questions[2]['question'],
                c_id=request.c_id,
                processing_time=processing_time,
                csv_data=csv_data,
                generation_updates=generation_updates
            )
            
            logger.info(f"DSPy generation completed for {request.c_id} in {processing_time:.2f}s")
            logger.info(f"All questions approved: {pipeline_result['all_approved']}")
            
            return result
            
        except Exception as e:
            logger.error(f"DSPy generation failed for {request.c_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error_type": "DSPY_GENERATION_ERROR",
                    "message": str(e),
                    "c_id": request.c_id,
                    "processing_time": time.time() - start_time
                }
            )
    
    def _build_sysarch_csv(self, validated_questions: List[Dict], request_params: Dict[str, Any]) -> Dict[str, Any]:
        """Build SYSARCH-compliant CSV data from DSPy results"""
        
        # Extract questions and answers
        questions_text = []
        all_answers = []
        
        for i, q in enumerate(validated_questions, 1):
            questions_text.append(f"{i}. {q['question']}")
            if q['answers']:
                all_answers.extend(q['answers'])
        
        combined_questions = " ".join(questions_text)
        combined_answers = ", ".join(all_answers) if all_answers else "Multiple choice options generated"
        
        # Build complete SYSARCH CSV structure
        csv_data = {
            "c_id": request_params['c_id'],
            "subject": request_params.get('p_variation', 'stammaufgabe'),
            "type": request_params.get('question_type', 'multiple-choice'),
            "text": combined_questions,
            
            # All SYSARCH parameters mapped correctly
            "p.instruction_explicitness_of_instruction": request_params.get('p_instruction_explicitness_of_instruction', 'Implizit'),
            "p.mathematical_requirement_level": f"{request_params.get('p_mathematical_requirement_level', '0')} ({'Kein Bezug' if request_params.get('p_mathematical_requirement_level', '0') == '0' else 'Mit Bezug'})",
            "p.taxanomy_level": request_params.get('p_taxonomy_level', 'Stufe 1'),
            "question_type": request_params.get('question_type', 'multiple-choice'),
            "p.variation": request_params.get('p_variation', 'stammaufgabe'),
            "answers": combined_answers,
            
            # Root text parameters
            "p.root_text_reference_explanatory_text": request_params.get('p_root_text_reference_explanatory_text', 'Nicht vorhanden'),
            "p.root_text_obstacle_passive": request_params.get('p_root_text_obstacle_passive', 'Nicht Enthalten'),
            "p.root_text_obstacle_negation": request_params.get('p_root_text_obstacle_negation', 'Nicht Enthalten'),
            "p.root_text_obstacle_complex_np": request_params.get('p_root_text_obstacle_complex_np', 'Nicht Enthalten'),
            "p.root_text_contains_irrelevant_information": request_params.get('p_root_text_contains_irrelevant_information', 'Nicht Enthalten'),
            
            # Instruction parameters
            "p.instruction_obstacle_passive": request_params.get('p_instruction_obstacle_passive', 'Nicht Enthalten'),
            "p.instruction_obstacle_complex_np": request_params.get('p_instruction_obstacle_complex_np', 'Nicht Enthalten'),
            
            # DSPy-specific metadata
            "dspy_consensus_used": True,
            "dspy_modular_prompts": True,
            "dspy_expert_count": 6,
            "dspy_all_approved": all(q['approved'] for q in validated_questions)
        }
        
        # Add individual item parameters (1-8)
        for i in range(1, 9):
            csv_data[f"p.item_{i}_obstacle_passive"] = request_params.get(f'p_item_{i}_obstacle_passive', 'Nicht Enthalten')
            csv_data[f"p.item_{i}_obstacle_negation"] = request_params.get(f'p_item_{i}_obstacle_negation', 'Nicht Enthalten')
            csv_data[f"p.item_{i}_obstacle_complex_np"] = request_params.get(f'p_item_{i}_obstacle_complex_np', 'Nicht Enthalten')
        
        return csv_data


# Global DSPy system instance
dspy_system = DSPyEducationalSystem()

# FastAPI app integration
app = FastAPI(title="DSPy Educational Question Generator")

@app.on_event("startup")
async def startup_event():
    """Initialize DSPy system on startup"""
    try:
        await dspy_system.initialize()
        logger.info("DSPy orchestrator startup completed")
    except Exception as e:
        logger.error(f"DSPy orchestrator startup failed: {e}")
        raise

@app.post("/generate-questions-dspy", response_model=SysArchResult)
async def generate_questions_dspy_endpoint(request: SysArchRequest):
    """
    Generate questions using DSPy single-pass consensus architecture
    Uses modular .txt prompts, preserves all existing functionality
    """
    start_time = time.time()
    
    try:
        logger.info(f"DSPy Request: {request.c_id} - {request.p_variation} ({request.question_type})")
        
        # Generate using DSPy pipeline
        result = await dspy_system.generate_questions_dspy(request)
        
        logger.info(f"DSPy Questions generated in {result.processing_time:.2f}s")
        
        # Preserve existing result_manager integration
        try:
            # Initialize result manager for this session
            result_manager = ResultManager(base_dir="ALEE_Agent/results")
            
            # Create session package 
            session_id = result_manager.create_session_package(
                session_metadata={
                    "dspy_enabled": True,
                    "single_pass_consensus": True,
                    "modular_prompts_used": True,
                    "request_c_id": request.c_id,
                    "processing_mode": "dspy_consensus"
                }
            )
            
            # Save DSPy results
            final_questions = [result.question_1, result.question_2, result.question_3]
            result_manager.save_final_results(
                session_id=session_id,
                final_questions=final_questions,
                csv_data=result.csv_data,
                final_metadata={
                    "dspy_pipeline_used": True,
                    "expert_consensus_achieved": result.csv_data.get('dspy_all_approved', False),
                    "processing_time": result.processing_time,
                    "generation_updates": result.generation_updates
                }
            )
            
            logger.info(f"DSPy results saved for {request.c_id} in session {session_id}")
            
        except Exception as save_error:
            logger.warning(f"DSPy result saving failed (continuing anyway): {save_error}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DSPy endpoint failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "DSPY_ENDPOINT_ERROR", 
                "message": str(e),
                "c_id": request.c_id,
                "processing_time": time.time() - start_time
            }
        )

@app.get("/health-dspy")
async def health_check_dspy():
    """Health check for DSPy system"""
    try:
        if not dspy_system.initialized:
            return {"status": "initializing", "dspy_ready": False}
        
        return {
            "status": "healthy",
            "dspy_ready": True,
            "expert_lms_configured": len(dspy_system.expert_lms) if dspy_system.expert_lms else 0,
            "modular_prompts_enabled": True,
            "consensus_architecture": "single_pass"
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "dspy_ready": False}

@app.get("/dspy-info")
async def dspy_system_info():
    """Get DSPy system information and configuration"""
    if not dspy_system.initialized:
        return {"error": "DSPy system not initialized"}
    
    return {
        "dspy_version": "3.0.1",
        "architecture": "single_pass_consensus",
        "expert_lms": list(dspy_system.expert_lms.keys()) if dspy_system.expert_lms else [],
        "modular_prompts": {
            "enabled": True,
            "builder_class": "ModularPromptBuilder",
            "txt_files_used": True,
            "hardcoded_prompts": False
        },
        "pipeline_modules": [
            "GermanQuestionGenerator",
            "VariationExpertGerman", 
            "TaxonomyExpertGerman",
            "MathExpertGerman",
            "ObstacleExpertGerman",
            "InstructionExpertGerman",
            "ContentExpertGerman",
            "GermanExpertConsensus"
        ],
        "sysarch_compliance": True,
        "result_manager_integration": True
    }

if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting DSPy Educational Question Generator")
    
    # Run server
    uvicorn.run(
        "dspy_orchestrator:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )