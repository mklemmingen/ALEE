#!/usr/bin/env python3
"""
Configuration Management for CallersWithTexts
Centralized settings for orchestrator communication, timeouts, and validation rules
"""

import os
from pathlib import Path
from typing import Dict, Any, List


class CallerConfig:
    """Centralized configuration for all caller scripts"""
    
    # Orchestrator Configuration
    DEFAULT_BASE_URL = "http://localhost:8000"
    DEFAULT_TIMEOUT_SECONDS = 300  # 5 minutes for question generation
    HEALTH_CHECK_TIMEOUT = 30     # 30 seconds for health checks
    
    # Request Configuration
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 2
    MAX_TEXT_LENGTH = 10000
    MIN_TEXT_LENGTH = 10
    
    # Validation Rules
    VALID_QUESTION_TYPES = ["multiple-choice", "single-choice", "true-false", "mapping"]
    VALID_VARIATIONS = ["stammaufgabe", "schwer", "leicht"]
    VALID_TAXONOMY_LEVELS = ["Stufe 1 (Wissen/Reproduktion)", "Stufe 2 (Anwendung/Transfer)"]
    VALID_MATH_LEVELS = ["0", "1", "2"]
    VALID_BINARY_VALUES = ["Enthalten", "Nicht Enthalten"]
    VALID_REFERENCE_VALUES = ["Nicht vorhanden", "Explizit", "Implizit"]
    VALID_EXPLICITNESS_VALUES = ["Explizit", "Implizit"]
    
    # File Paths
    STAKEHOLDER_CSV_PATH = "../.dev/providedProjectFromStakeHolder/explanation_metadata.csv"
    LOG_DIR = "../_logs"
    
    # Logging Configuration
    LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "SUCCESS"]
    MAX_LOG_FILE_SIZE_MB = 50
    LOG_RETENTION_DAYS = 30
    
    # Performance Thresholds
    FAST_RESPONSE_THRESHOLD = 10.0   # Seconds - responses faster than this are "fast"
    SLOW_RESPONSE_THRESHOLD = 60.0   # Seconds - responses slower than this are "slow"
    SUCCESS_RATE_WARNING_THRESHOLD = 85.0  # Percent - below this triggers warnings
    
    @classmethod
    def get_base_url(cls) -> str:
        """Get orchestrator base URL from environment or default"""
        return os.getenv("ORCHESTRATOR_URL", cls.DEFAULT_BASE_URL)
    
    @classmethod
    def get_timeout(cls) -> int:
        """Get timeout from environment or default"""
        try:
            return int(os.getenv("ORCHESTRATOR_TIMEOUT", cls.DEFAULT_TIMEOUT_SECONDS))
        except ValueError:
            return cls.DEFAULT_TIMEOUT_SECONDS
    
    @classmethod
    def get_stakeholder_csv_path(cls) -> str:
        """Get stakeholder CSV path from environment or default"""
        return os.getenv("STAKEHOLDER_CSV_PATH", cls.STAKEHOLDER_CSV_PATH)
    
    @classmethod
    def get_log_dir(cls) -> Path:
        """Get log directory path"""
        base_path = Path(__file__).parent
        log_dir = base_path / cls.LOG_DIR
        log_dir.mkdir(exist_ok=True)
        return log_dir
    
    @classmethod
    def validate_question_type(cls, question_type: str) -> bool:
        """Validate question type"""
        return question_type in cls.VALID_QUESTION_TYPES
    
    @classmethod
    def validate_variation(cls, variation: str) -> bool:
        """Validate difficulty variation"""
        return variation in cls.VALID_VARIATIONS
    
    @classmethod
    def validate_taxonomy_level(cls, level: str) -> bool:
        """Validate taxonomy level"""
        return level in cls.VALID_TAXONOMY_LEVELS
    
    @classmethod
    def validate_math_level(cls, level: str) -> bool:
        """Validate mathematical requirement level"""
        return str(level) in cls.VALID_MATH_LEVELS
    
    @classmethod
    def validate_binary_value(cls, value: str) -> bool:
        """Validate binary parameter values"""
        return value in cls.VALID_BINARY_VALUES
    
    @classmethod
    def validate_c_id_format(cls, c_id: str) -> bool:
        """Validate c_id format (number-number-number)"""
        if not c_id or not isinstance(c_id, str):
            return False
        
        parts = c_id.split('-')
        if len(parts) < 2:
            return False
        
        try:
            for part in parts:
                int(part)
            return True
        except ValueError:
            return False
    
    @classmethod
    def get_required_parameters(cls) -> List[str]:
        """Get list of required parameters for SYSARCH compliance"""
        return [
            "c_id",
            "text", 
            "question_type",
            "p_variation",
            "p_taxonomy_level",
            "p_mathematical_requirement_level",
            "p_root_text_reference_explanatory_text",
            "p_root_text_obstacle_passive",
            "p_root_text_obstacle_negation", 
            "p_root_text_obstacle_complex_np",
            "p_root_text_contains_irrelevant_information",
            "p_instruction_obstacle_passive",
            "p_instruction_obstacle_negation",
            "p_instruction_obstacle_complex_np", 
            "p_instruction_explicitness_of_instruction"
        ] + [
            f"p_item_{i}_obstacle_{obstacle}"
            for i in range(1, 9)
            for obstacle in ["passive", "negation", "complex_np"]
        ]
    
    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """Get default parameter values for testing"""
        defaults = {
            "question_type": "multiple-choice",
            "p_variation": "stammaufgabe", 
            "p_taxonomy_level": "Stufe 1 (Wissen/Reproduktion)",
            "p_mathematical_requirement_level": "0",
            "p_root_text_reference_explanatory_text": "Nicht vorhanden",
            "p_root_text_obstacle_passive": "Nicht Enthalten",
            "p_root_text_obstacle_negation": "Nicht Enthalten",
            "p_root_text_obstacle_complex_np": "Nicht Enthalten", 
            "p_root_text_contains_irrelevant_information": "Nicht Enthalten",
            "p_instruction_obstacle_passive": "Nicht Enthalten",
            "p_instruction_obstacle_negation": "Nicht Enthalten",
            "p_instruction_obstacle_complex_np": "Nicht Enthalten",
            "p_instruction_explicitness_of_instruction": "Explizit"
        }
        
        # Add item-specific parameters
        for i in range(1, 9):
            for obstacle in ["passive", "negation", "complex_np"]:
                defaults[f"p_item_{i}_obstacle_{obstacle}"] = "Nicht Enthalten"
        
        return defaults
    
    @classmethod
    def get_validation_rules(cls) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive validation rules for all parameters"""
        return {
            "c_id": {
                "type": "string",
                "required": True,
                "validator": cls.validate_c_id_format,
                "description": "Unique identifier in format: number-number-number"
            },
            "text": {
                "type": "string", 
                "required": True,
                "min_length": cls.MIN_TEXT_LENGTH,
                "max_length": cls.MAX_TEXT_LENGTH,
                "description": "Educational text content for question generation"
            },
            "question_type": {
                "type": "choice",
                "required": True,
                "choices": cls.VALID_QUESTION_TYPES,
                "description": "Format of questions to generate"
            },
            "p_variation": {
                "type": "choice",
                "required": True, 
                "choices": cls.VALID_VARIATIONS,
                "description": "Difficulty level for questions"
            },
            "p_taxonomy_level": {
                "type": "choice",
                "required": True,
                "choices": cls.VALID_TAXONOMY_LEVELS,
                "description": "Cognitive complexity level (Bloom's taxonomy)"
            },
            "p_mathematical_requirement_level": {
                "type": "choice",
                "required": True,
                "choices": cls.VALID_MATH_LEVELS, 
                "description": "Mathematical complexity requirement"
            },
            "p_instruction_explicitness_of_instruction": {
                "type": "choice",
                "required": True,
                "choices": cls.VALID_EXPLICITNESS_VALUES,
                "description": "How explicit task instructions should be"
            }
        }
    
    @classmethod
    def get_endpoint_config(cls) -> Dict[str, Dict[str, Any]]:
        """Get configuration for different orchestrator endpoints"""
        return {
            "health": {
                "path": "/system-health",
                "method": "GET",
                "timeout": cls.HEALTH_CHECK_TIMEOUT,
                "expected_fields": ["status", "dspy_ready"]
            },
            "capabilities": {
                "path": "/system-capabilities", 
                "method": "GET",
                "timeout": cls.HEALTH_CHECK_TIMEOUT,
                "expected_fields": ["modules", "endpoints"]
            },
            "generate": {
                "path": "/generate-educational-questions",
                "method": "POST",
                "timeout": cls.DEFAULT_TIMEOUT_SECONDS,
                "expected_fields": ["question_1", "question_2", "question_3", "c_id", "processing_time"]
            }
        }
    
    @classmethod
    def get_performance_config(cls) -> Dict[str, Any]:
        """Get performance monitoring configuration"""
        return {
            "response_time_thresholds": {
                "fast": cls.FAST_RESPONSE_THRESHOLD,
                "slow": cls.SLOW_RESPONSE_THRESHOLD
            },
            "success_rate_thresholds": {
                "warning": cls.SUCCESS_RATE_WARNING_THRESHOLD,
                "critical": 50.0
            },
            "monitoring_enabled": True,
            "alert_on_failures": True,
            "log_performance_metrics": True
        }


class TestConfig:
    """Configuration specific to testing scenarios"""
    
    # Test execution settings
    MAX_CONCURRENT_REQUESTS = 3
    REQUEST_DELAY_SECONDS = 2
    SYSTEMATIC_TEST_COVERAGE = True
    
    # Test data configuration
    SAMPLE_C_IDS = ["181-1-3", "182-1-1", "183-1-4", "194-1-2"]
    SAMPLE_TEXTS = [
        "Bedürfnisse sind Wünsche Menschen haben...",
        "Wirtschaftliche Güter sind knapp und müssen verteilt werden...",
        "Märkte entstehen durch Angebot und Nachfrage..."
    ]
    
    # Test coverage requirements
    REQUIRED_QUESTION_TYPES = CallerConfig.VALID_QUESTION_TYPES
    REQUIRED_VARIATIONS = CallerConfig.VALID_VARIATIONS
    REQUIRED_TAXONOMY_LEVELS = CallerConfig.VALID_TAXONOMY_LEVELS
    
    @classmethod
    def get_systematic_test_matrix(cls) -> List[Dict[str, Any]]:
        """Generate systematic test parameter combinations"""
        test_combinations = []
        
        for i, question_type in enumerate(cls.REQUIRED_QUESTION_TYPES):
            for j, variation in enumerate(cls.REQUIRED_VARIATIONS):
                for k, taxonomy in enumerate(cls.REQUIRED_TAXONOMY_LEVELS):
                    combination = {
                        "test_id": f"sys_{i}_{j}_{k}",
                        "question_type": question_type,
                        "p_variation": variation,
                        "p_taxonomy_level": taxonomy,
                        "p_mathematical_requirement_level": str(k % 3),  # Cycle through 0,1,2
                        "obstacle_pattern": (i + j + k) % 8  # For systematic obstacle variation
                    }
                    test_combinations.append(combination)
        
        return test_combinations
    
    @classmethod
    def get_test_text(cls, index: int = 0) -> str:
        """Get sample text for testing"""
        return cls.SAMPLE_TEXTS[index % len(cls.SAMPLE_TEXTS)]
    
    @classmethod
    def get_test_c_id(cls, index: int = 0, suffix: str = "test") -> str:
        """Generate test c_id"""
        base_id = cls.SAMPLE_C_IDS[index % len(cls.SAMPLE_C_IDS)]
        return f"{base_id}-{suffix}"


# Environment-specific configuration
class EnvironmentConfig:
    """Environment-specific settings"""
    
    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development environment"""
        return os.getenv("ENVIRONMENT", "development").lower() == "development"
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production environment"""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    @classmethod
    def get_log_level(cls) -> str:
        """Get appropriate log level for environment"""
        if cls.is_development():
            return os.getenv("LOG_LEVEL", "DEBUG")
        else:
            return os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def get_config_overrides(cls) -> Dict[str, Any]:
        """Get environment-specific configuration overrides"""
        overrides = {}
        
        if cls.is_production():
            overrides.update({
                "timeout_seconds": 600,  # Longer timeout in production
                "max_retries": 5,
                "log_level": "INFO"
            })
        elif cls.is_development():
            overrides.update({
                "timeout_seconds": 180,  # Shorter timeout for dev
                "max_retries": 2,
                "log_level": "DEBUG"
            })
        
        return overrides


# Export main configuration classes
__all__ = ['CallerConfig', 'TestConfig', 'EnvironmentConfig']