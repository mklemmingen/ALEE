"""
DSPy Configuration for Educational Question Generation System
Configures DSPy to work with existing OLLAMA multiserver setup (ports 8001-8007)
"""

import logging

import dspy

logger = logging.getLogger(__name__)

def create_ollama_lm(port: int, model: str = "llama3.1:8b", temperature: float = 0.2, role: str = "expert") -> dspy.LM:
    """Create OLLAMA LM instance for specific port"""
    base_url = f"http://localhost:{port}"
    
    # Use LiteLLM format for OLLAMA
    model_name = f"ollama_chat/{model}"
    
    return dspy.LM(
        model=model_name,
        api_base=base_url,
        temperature=temperature,
        max_tokens=2000,
        # Additional OLLAMA-specific settings
        num_predict=2000,
        repeat_penalty=1.1,
        top_k=40,
        top_p=0.9
    )

def configure_dspy_with_ollama():
    """Configure DSPy with sequential server reuse pattern"""
    
    logger.info("Configuring DSPy with sequential server reuse (Generation -> Experts -> Refinement)")
    
    # Main generator & refiner LM (sequential reuse of port 8001)
    # Used for: 1) Initial generation, 2) Question refinement with expert feedback
    main_lm = create_ollama_lm(
        port=8001,
        model="llama3.1:8b", 
        temperature=0.7  # Higher creativity for generation and refinement
    )
    
    # Expert LMs configuration - ports 8002-8007 for parallel validation
    # Note: Port 8001 is used by main_lm for generation/refinement
    expert_lms = {
        'variation_expert': create_ollama_lm(
            port=8002,  # Changed from 8001 to avoid conflict
            model="mistral:7b",  # Updated to match start_ollama_servers.sh
            temperature=0.2  # Lower temperature for validation
        ),
        'taxonomy_expert': create_ollama_lm(
            port=8003,  # Sequential assignment
            model="qwen2.5:7b",  # Updated to match start_ollama_servers.sh
            temperature=0.2
        ),
        'math_expert': create_ollama_lm(
            port=8004,  # Sequential assignment
            model="llama3.2:3b",  # Updated to match start_ollama_servers.sh
            temperature=0.2
        ),
        'text_reference_expert': create_ollama_lm(
            port=8005,  # Sequential assignment
            model="llama3.2:3b",
            temperature=0.2
        ),
        'obstacle_expert': create_ollama_lm(
            port=8006,  # Sequential assignment
            model="mistral:7b",  # Updated to match start_ollama_servers.sh
            temperature=0.2
        ),
        'instruction_expert': create_ollama_lm(
            port=8007,  # Final expert server
            model="llama3.1:8b",  # Updated to match start_ollama_servers.sh
            temperature=0.2
        )
        # Note: Removed 'content_expert' to avoid using more servers than available
        # content validation will be handled by instruction_expert
    }
    
    # Set main LM as default for DSPy
    dspy.configure(lm=main_lm)
    
    # Configure DSPy settings for better performance
    dspy.settings.configure(
        # trace=False,  # Disable tracing to avoid len() issues
        cache=True,  # Cache LLM calls for efficiency
        experimental=True,  # Enable experimental features
        max_retry=3  # Retry failed calls
    )
    
    # Add refinement LM (same as main_lm for sequential reuse)
    expert_lms['question_refiner'] = main_lm  # Sequential reuse of port 8001
    
    logger.info(f"Configured DSPy with sequential server reuse:")
    logger.info(f"  - Main Generator/Refiner: Port 8001 (llama3.1:8b) - Sequential reuse")
    logger.info(f"  - Expert Validators: {len(expert_lms)-1} servers (ports 8002-8007)")
    logger.info(f"  - Processing flow: Generation -> Parallel Validation -> Refinement")
    
    return main_lm, expert_lms

def get_question_refiner_lm():
    """Get the question refiner LM (same as main generator for sequential reuse)"""
    return create_ollama_lm(
        port=8001,
        model="llama3.1:8b",
        temperature=0.8,  # Higher temperature for creative refinement
        role="refiner"
    )

def get_expert_model_mapping():
    """Map expert names to their specialized models with sequential server allocation"""
    return {
        'main_generator_refiner': {
            'model': 'llama3.1:8b',
            'port': 8001,
            'expertise': 'Question generation and refinement with expert feedback',
            'parameters': 'all',
            'specialization': 'Sequential reuse: Initial generation -> Final refinement',
            'usage': 'Sequential (Generation phase, then Refinement phase)'
        },
        'variation_expert': {
            'model': 'mistral:7b',
            'port': 8002,
            'expertise': 'German difficulty level assessment (leicht/stammaufgabe/schwer)',
            'parameters': ['p_variation'],
            'specialization': 'Educational content complexity analysis'
        },
        'taxonomy_expert': {
            'model': 'qwen2.5:7b',
            'port': 8003,
            'expertise': "Bloom's taxonomy level validation", 
            'parameters': ['p_taxanomy_level'],
            'specialization': 'Cognitive complexity classification'
        },
        'math_expert': {
            'model': 'llama3.2:3b',
            'port': 8004,
            'expertise': 'Mathematical requirement assessment',
            'parameters': ['p_mathematical_requirement_level'],
            'specialization': 'Mathematical complexity analysis'
        },
        'text_reference_expert': {
            'model': 'llama3.2:3b',
            'port': 8005,
            'expertise': 'Text reference and explanatory content validation',
            'parameters': ['p_root_text_reference_explanatory_text'],
            'specialization': 'Text coherence and reference analysis'
        },
        'obstacle_expert': {
            'model': 'mistral:7b',
            'port': 8006,
            'expertise': 'Linguistic obstacle detection and validation',
            'parameters': [
                'p_root_text_obstacle_passive',
                'p_root_text_obstacle_negation', 
                'p_root_text_obstacle_complex_np',
                'p_item_*_obstacle_*'  # All item obstacles
            ],
            'specialization': 'German linguistic complexity analysis'
        },
        'instruction_expert': {
            'model': 'llama3.1:8b',
            'port': 8007,
            'expertise': 'Instruction clarity, content relevance, and overall quality',
            'parameters': [
                'p_instruction_explicitness_of_instruction',
                'p_instruction_obstacle_passive',
                'p_instruction_obstacle_complex_np',
                'p_root_text_contains_irrelevant_information'  # Combined content validation
            ],
            'specialization': 'Educational instruction design and content quality'
        }
    }

async def test_dspy_connection():
    """Test connection to all configured LLMs"""
    
    main_lm, expert_lms = configure_dspy_with_ollama()
    
    # Test main generator
    try:
        test_prompt = "Was ist Bildung?"
        response = main_lm("Test connection. Respond with: DSPy verbunden")
        logger.info(f"Main generator test: {response}")
    except Exception as e:
        logger.error(f"Main generator connection failed: {e}")
    
    # Test expert LMs  
    for expert_name, expert_lm in expert_lms.items():
        try:
            with dspy.context(lm=expert_lm):
                response = expert_lm("Test. Antwort: 'Verbunden'")
                logger.info(f"{expert_name} test: {response}")
        except Exception as e:
            logger.error(f"{expert_name} connection failed: {e}")
    
    return main_lm, expert_lms

if __name__ == "__main__":
    # Test configuration
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_dspy_connection())