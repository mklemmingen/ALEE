#!/usr/bin/env python3
"""
Type-specific Pydantic models for educational questions
Provides format validation and expert suggestion integration
"""

import re
from typing import List, Optional, Union, Literal, Dict, Any
from pydantic import BaseModel, field_validator, Field


class ExpertSuggestion(BaseModel):
    """Individual expert suggestion with metadata"""
    expert_type: str
    rating: int = Field(ge=1, le=5)
    feedback: str
    suggestions: List[str]
    reasoning: str
    target_value: str
    applied: bool = False


class BaseQuestionModel(BaseModel):
    """Base model for all educational questions with expert integration"""
    question_text: str
    question_type: Literal["multiple-choice", "single-choice", "true-false", "mapping"]
    answers: List[str]
    correct_answer: Union[str, List[str]]
    explanation: Optional[str] = ""
    
    # Expert integration
    expert_suggestions: List[ExpertSuggestion] = []
    expert_ratings: Dict[str, int] = {}
    refinement_applied: bool = False
    refinement_reasoning: Optional[str] = ""
    
    # Validation tracking
    format_preserved: bool = True
    parameters_maintained: bool = True
    original_question: Optional[str] = None
    
    def add_expert_suggestion(self, suggestion: ExpertSuggestion) -> None:
        """Add expert suggestion and update ratings"""
        self.expert_suggestions.append(suggestion)
        self.expert_ratings[suggestion.expert_type] = suggestion.rating
    
    def get_average_rating(self) -> float:
        """Calculate average expert rating"""
        if not self.expert_ratings:
            return 0.0
        return sum(self.expert_ratings.values()) / len(self.expert_ratings)
    
    def needs_refinement(self, threshold: float = 3.0) -> bool:
        """Determine if question needs refinement based on expert ratings"""
        return self.get_average_rating() < threshold
    
    def to_csv_dict(self) -> Dict[str, Any]:
        """Convert to CSV-compatible dictionary"""
        return {
            'question_text': self.question_text,
            'question_type': self.question_type,
            'answers': '; '.join(self.answers),
            'correct_answer': str(self.correct_answer),
            'expert_average_rating': self.get_average_rating(),
            'refinement_applied': self.refinement_applied,
            'format_preserved': self.format_preserved
        }


class MultipleChoiceModel(BaseQuestionModel):
    """Multiple choice question with <option> markup validation"""
    question_type: Literal["multiple-choice"] = "multiple-choice"
    correct_answer: List[str]  # Multiple correct answers allowed
    
    @field_validator('question_text')
    @classmethod
    def validate_option_markup(cls, v: str) -> str:
        """Ensure proper <option> markup and sufficient options"""
        option_pattern = r'<option>(.*?)</option>'
        options = re.findall(option_pattern, v, re.DOTALL)
        
        if len(options) < 6:
            raise ValueError(f"Multiple choice needs at least 6 options, found {len(options)}")
        if len(options) > 8:
            raise ValueError(f"Multiple choice should have max 8 options, found {len(options)}")
        
        # Check for meaningful content in options
        for i, option in enumerate(options):
            if not option.strip():
                raise ValueError(f"Option {i+1} is empty")
        
        return v
    
    @field_validator('correct_answer')
    @classmethod
    def validate_multiple_correct(cls, v: Union[str, List[str]]) -> List[str]:
        """Ensure multiple correct answers for multiple choice"""
        if isinstance(v, str):
            v = [v]
        if len(v) < 2:
            raise ValueError("Multiple choice must have at least 2 correct answers")
        if len(v) > 4:
            raise ValueError("Multiple choice should have max 4 correct answers")
        return v
    
    def extract_options(self) -> List[str]:
        """Extract option text from markup"""
        option_pattern = r'<option>(.*?)</option>'
        return re.findall(option_pattern, self.question_text, re.DOTALL)
    
    def validate_format_compliance(self) -> bool:
        """Check if question maintains proper multiple choice format"""
        try:
            options = self.extract_options()
            return (6 <= len(options) <= 8 and 
                   len(self.correct_answer) >= 2 and 
                   all(opt.strip() for opt in options))
        except:
            return False


class SingleChoiceModel(BaseQuestionModel):
    """Single choice question with <option> markup validation"""
    question_type: Literal["single-choice"] = "single-choice"
    correct_answer: str  # Only one correct answer
    
    @field_validator('question_text')
    @classmethod
    def validate_option_markup(cls, v: str) -> str:
        """Ensure proper <option> markup with exactly 4 options"""
        option_pattern = r'<option>(.*?)</option>'
        options = re.findall(option_pattern, v, re.DOTALL)
        
        if len(options) != 4:
            raise ValueError(f"Single choice needs exactly 4 options, found {len(options)}")
        
        # Check for meaningful content in options
        for i, option in enumerate(options):
            if not option.strip():
                raise ValueError(f"Option {i+1} is empty")
        
        return v
    
    @field_validator('correct_answer')
    @classmethod
    def validate_single_correct(cls, v: Union[str, List[str]]) -> str:
        """Ensure only one correct answer for single choice"""
        if isinstance(v, list):
            if len(v) != 1:
                raise ValueError("Single choice must have exactly 1 correct answer")
            return v[0]
        return v
    
    def extract_options(self) -> List[str]:
        """Extract option text from markup"""
        option_pattern = r'<option>(.*?)</option>'
        return re.findall(option_pattern, self.question_text, re.DOTALL)
    
    def validate_format_compliance(self) -> bool:
        """Check if question maintains proper single choice format"""
        try:
            options = self.extract_options()
            return (len(options) == 4 and 
                   isinstance(self.correct_answer, str) and
                   all(opt.strip() for opt in options))
        except:
            return False


class TrueFalseModel(BaseQuestionModel):
    """True/False question with <true-false> markup validation"""
    question_type: Literal["true-false"] = "true-false"
    correct_answer: str  # "Richtig" or "Falsch"
    
    @field_validator('question_text')
    @classmethod
    def validate_true_false_markup(cls, v: str) -> str:
        """Ensure proper <true-false> markup"""
        tf_pattern = r'<true-false>(.*?)</true-false>'
        statements = re.findall(tf_pattern, v, re.DOTALL)
        
        if len(statements) != 2:
            raise ValueError(f"True-false needs exactly 2 <true-false> statements, found {len(statements)}")
        
        # Check typical true-false statement patterns
        expected_patterns = ["richtig", "falsch", "wahr", "unwahr"]
        statements_text = ' '.join(statements).lower()
        
        if not any(pattern in statements_text for pattern in expected_patterns):
            raise ValueError("True-false statements should contain richtig/falsch indicators")
        
        return v
    
    @field_validator('correct_answer')
    @classmethod
    def validate_true_false_answer(cls, v: str) -> str:
        """Ensure correct answer is Richtig or Falsch"""
        valid_answers = ["Richtig", "Falsch", "richtig", "falsch"]
        if v not in valid_answers:
            raise ValueError(f"True-false answer must be 'Richtig' or 'Falsch', got '{v}'")
        return v.capitalize()  # Normalize to "Richtig"/"Falsch"
    
    def extract_statements(self) -> List[str]:
        """Extract true-false statements from markup"""
        tf_pattern = r'<true-false>(.*?)</true-false>'
        return re.findall(tf_pattern, self.question_text, re.DOTALL)
    
    def validate_format_compliance(self) -> bool:
        """Check if question maintains proper true-false format"""
        try:
            statements = self.extract_statements()
            return (len(statements) == 2 and 
                   self.correct_answer in ["Richtig", "Falsch"])
        except:
            return False


class MappingModel(BaseQuestionModel):
    """Mapping question with <start-option>/<end-option> markup validation"""
    question_type: Literal["mapping"] = "mapping"
    correct_answer: Dict[str, str]  # Mapping pairs
    
    @field_validator('question_text')
    @classmethod
    def validate_mapping_markup(cls, v: str) -> str:
        """Ensure proper <start-option>/<end-option> markup"""
        # Pattern matches <start-option> followed by content until next <start-option> or <end-option>
        start_pattern = r'<start-option>\s*([^<]+?)(?=\s*<(?:start-option|end-option))'
        end_pattern = r'<end-option>\s*([^<]+?)(?=\s*<(?:start-option|end-option)|$)'
        
        start_options = re.findall(start_pattern, v, re.DOTALL)
        end_options = re.findall(end_pattern, v, re.DOTALL)
        
        if len(start_options) < 3 or len(start_options) > 5:
            raise ValueError(f"Mapping needs 3-5 start options, found {len(start_options)}")
        
        if len(end_options) < 3 or len(end_options) > 5:
            raise ValueError(f"Mapping needs 3-5 end options, found {len(end_options)}")
        
        if len(start_options) != len(end_options):
            raise ValueError(f"Start options ({len(start_options)}) must equal end options ({len(end_options)})")
        
        return v
    
    @field_validator('correct_answer')
    @classmethod
    def validate_mapping_pairs(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Ensure correct answer is proper mapping dictionary"""
        if not isinstance(v, dict):
            raise ValueError("Mapping correct_answer must be a dictionary")
        
        if len(v) < 3:
            raise ValueError("Mapping must have at least 3 pairs")
        
        return v
    
    def extract_mapping_options(self) -> tuple[List[str], List[str]]:
        """Extract start and end options from markup"""
        start_pattern = r'<start-option>\s*([^<]+?)(?=\s*<(?:start-option|end-option))'
        end_pattern = r'<end-option>\s*([^<]+?)(?=\s*<(?:start-option|end-option)|$)'
        
        start_options = re.findall(start_pattern, self.question_text, re.DOTALL)
        end_options = re.findall(end_pattern, self.question_text, re.DOTALL)
        
        return start_options, end_options
    
    def validate_format_compliance(self) -> bool:
        """Check if question maintains proper mapping format"""
        try:
            start_options, end_options = self.extract_mapping_options()
            return (3 <= len(start_options) <= 5 and 
                   len(start_options) == len(end_options) and
                   isinstance(self.correct_answer, dict) and
                   len(self.correct_answer) >= 3)
        except:
            return False


class QuestionFactory:
    """Factory for creating typed question models"""
    
    MODEL_MAP = {
        "multiple-choice": MultipleChoiceModel,
        "single-choice": SingleChoiceModel,
        "true-false": TrueFalseModel,
        "mapping": MappingModel
    }
    
    @classmethod
    def create_question(cls, question_type: str, question_data: Dict[str, Any]) -> BaseQuestionModel:
        """Create appropriate question model based on type"""
        if question_type not in cls.MODEL_MAP:
            raise ValueError(f"Unknown question type: {question_type}")
        
        model_class = cls.MODEL_MAP[question_type]
        
        # Ensure question_type is set in data
        question_data["question_type"] = question_type
        
        try:
            return model_class(**question_data)
        except Exception as e:
            raise ValueError(f"Failed to create {question_type} question: {e}")
    
    @classmethod
    def from_dspy_output(cls, question_type: str, frage: str, antworten: List[str], 
                        correct_answer: Union[str, List[str]], explanation: str = "") -> BaseQuestionModel:
        """Create typed question from DSPy generation output"""
        question_data = {
            "question_text": frage,
            "answers": antworten,
            "correct_answer": correct_answer,
            "explanation": explanation
        }
        
        return cls.create_question(question_type, question_data)
    
    @classmethod
    def validate_question_type(cls, question: BaseQuestionModel) -> bool:
        """Validate that question maintains its type format"""
        return question.validate_format_compliance()
    
    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Get list of supported question types"""
        return list(cls.MODEL_MAP.keys())


# Export main classes
__all__ = [
    'BaseQuestionModel', 
    'MultipleChoiceModel', 
    'SingleChoiceModel', 
    'TrueFalseModel', 
    'MappingModel',
    'ExpertSuggestion',
    'QuestionFactory'
]