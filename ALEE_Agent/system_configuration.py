"""
DSPy Configuration for Educational Question Generation System
Configures DSPy to work with existing OLLAMA multiserver setup (ports 8001-8007)
"""

import logging

import dspy

logger = logging.getLogger(__name__)

def create_ollama_lm(port: int, model: str = "llama3.1:8b", temperature: float = 0.2) -> dspy.LM:
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
    """Configure DSPy to work with existing OLLAMA servers"""
    
    logger.info("Configuring DSPy with OLLAMA multi-server setup")
    
    # Main generator LM on default OLLAMA port (11434)
    main_lm = create_ollama_lm(
        port=11434,
        model="llama3.1:8b", 
        temperature=0.7  # Higher creativity for generation
    )
    
    # Expert LMs configuration - matching your existing server setup
    expert_lms = {
        'variation_expert': create_ollama_lm(
            port=8001,
            model="llama3.1:8b",
            temperature=0.2  # Lower temperature for validation
        ),
        'taxonomy_expert': create_ollama_lm(
            port=8002,
            model="mistral:7b",
            temperature=0.2
        ),
        'math_expert': create_ollama_lm(
            port=8003,
            model="qwen2.5:7b", 
            temperature=0.2
        ),
        'text_reference_expert': create_ollama_lm(
            port=8004,
            model="llama3.2:3b",
            temperature=0.2
        ),
        'obstacle_expert': create_ollama_lm(
            port=8005,
            model="llama3.2:3b",
            temperature=0.2
        ),
        'instruction_expert': create_ollama_lm(
            port=8006,
            model="mistral:7b",
            temperature=0.2
        ),
        'content_expert': create_ollama_lm(
            port=8007,
            model="llama3.1:8b",
            temperature=0.2
        )
    }
    
    # Set main LM as default for DSPy
    dspy.configure(lm=main_lm)
    
    # Configure DSPy settings for better performance
    dspy.settings.configure(
        trace=True,  # Enable detailed tracing
        cache=True,  # Cache LLM calls for efficiency
        experimental=True,  # Enable experimental features
        max_retry=3  # Retry failed calls
    )
    
    logger.info(f"Configured DSPy with main generator on port 11434 and {len(expert_lms)} expert servers")
    
    return main_lm, expert_lms

def get_expert_model_mapping():
    """Map expert names to their specialized models and capabilities"""
    return {
        'variation_expert': {
            'model': 'llama3.1:8b',
            'port': 8001,
            'expertise': 'German difficulty level assessment (leicht/stammaufgabe/schwer)',
            'parameters': ['p_variation'],
            'specialization': 'Educational content complexity analysis'
        },
        'taxonomy_expert': {
            'model': 'mistral:7b',
            'port': 8002,
            'expertise': "Bloom's taxonomy level validation", 
            'parameters': ['p_taxanomy_level'],
            'specialization': 'Cognitive complexity classification'
        },
        'math_expert': {
            'model': 'qwen2.5:7b',
            'port': 8003,
            'expertise': 'Mathematical requirement assessment',
            'parameters': ['p_mathematical_requirement_level'],
            'specialization': 'Mathematical complexity analysis'
        },
        'text_reference_expert': {
            'model': 'llama3.2:3b',
            'port': 8004,
            'expertise': 'Text reference and explanatory content validation',
            'parameters': ['p_root_text_reference_explanatory_text'],
            'specialization': 'Text coherence and reference analysis'
        },
        'obstacle_expert': {
            'model': 'llama3.2:3b',
            'port': 8005,
            'expertise': 'Linguistic obstacle detection and validation',
            'parameters': [
                'p_root_text_obstacle_passive',
                'p_root_text_obstacle_negation', 
                'p_root_text_obstacle_complex_np',
                'p_item_1_obstacle_passive',
                'p_item_1_obstacle_negation',
                'p_item_1_obstacle_complex_np'
                # ... (extends to all item obstacles)
            ],
            'specialization': 'German linguistic complexity analysis'
        },
        'instruction_expert': {
            'model': 'mistral:7b',
            'port': 8006,
            'expertise': 'Instruction clarity and explicitness validation',
            'parameters': [
                'p_instruction_explicitness_of_instruction',
                'p_instruction_obstacle_passive',
                'p_instruction_obstacle_complex_np'
            ],
            'specialization': 'Educational instruction design'
        },
        'content_expert': {
            'model': 'llama3.1:8b',
            'port': 8007,
            'expertise': 'Content relevance and irrelevant information detection',
            'parameters': [
                'p_root_text_contains_irrelevant_information'
            ],
            'specialization': 'Educational content quality assessment'
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