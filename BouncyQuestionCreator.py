import json
import asyncio
import aiohttp
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    ACCEPT = "accept"
    REJECT = "reject" 
    PASS = "pass"
    FAIL = "fail"
    UNIQUE = "unique"
    SIMILAR = "similar"
    DUPLICATE = "duplicate"
    ADVANCE = "advance"
    REVISE = "revise"


@dataclass
class LMConfig:
    name: str
    endpoint: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000


@dataclass
class QuestionRequest:
    topic: str
    age_group: str
    difficulty_level: str
    context: Optional[str] = None


@dataclass
class ValidationScore:
    score: float
    decision: ValidationResult
    feedback: str
    details: Dict[str, Any]


class LanguageModelClient:
    def __init__(self, config: LMConfig):
        self.config = config
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        try:
            async with self.session.post(
                f"{self.config.endpoint}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    raise Exception(f"API request failed: {response.status}")
                
                result = await response.json()
                return result["choices"][0]["message"]["content"]
                
        except Exception as e:
            logger.error(f"Error calling {self.config.name}: {e}")
            raise


class BouncyQuestionCreator:
    def __init__(self, config_path: str = "config.json"):
        self.load_config(config_path)
        self.load_prompts()
        self.question_database = []  # In-memory storage for uniqueness checking
    
    def load_config(self, config_path: str):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            self.lm_configs = {
                name: LMConfig(**config) 
                for name, config in config_data.get("language_models", {}).items()
            }
            
            self.orchestrator_config = config_data.get("orchestrator", {})
            self.max_iterations = self.orchestrator_config.get("max_iterations", 3)
            self.min_score_threshold = self.orchestrator_config.get("min_score_threshold", 7.0)
            
        except FileNotFoundError:
            logger.warning("Config file not found. Using default configuration.")
            self.create_default_config()
    
    def create_default_config(self):
        default_config = {
            "language_models": {
                "generator": {
                    "name": "generator",
                    "endpoint": "http://localhost:8001",
                    "model": "llama-3.1-8b-instruct",
                    "temperature": 0.8,
                    "max_tokens": 1000
                },
                "meaningfulness_checker": {
                    "name": "meaningfulness_checker", 
                    "endpoint": "http://localhost:8002",
                    "model": "llama-3.1-8b-instruct",
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                "robustness_validator": {
                    "name": "robustness_validator",
                    "endpoint": "http://localhost:8003", 
                    "model": "llama-3.1-8b-instruct",
                    "temperature": 0.2,
                    "max_tokens": 500
                },
                "uniqueness_checker": {
                    "name": "uniqueness_checker",
                    "endpoint": "http://localhost:8004",
                    "model": "llama-3.1-8b-instruct", 
                    "temperature": 0.3,
                    "max_tokens": 400
                },
                "progress_assessor": {
                    "name": "progress_assessor",
                    "endpoint": "http://localhost:8005",
                    "model": "llama-3.1-8b-instruct",
                    "temperature": 0.4,
                    "max_tokens": 500
                }
            },
            "orchestrator": {
                "max_iterations": 3,
                "min_score_threshold": 7.0
            }
        }
        
        with open("config.json", 'w') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info("Created default configuration file: config.json")
        self.lm_configs = {name: LMConfig(**config) for name, config in default_config["language_models"].items()}
        self.orchestrator_config = default_config["orchestrator"]
        self.max_iterations = self.orchestrator_config["max_iterations"]
        self.min_score_threshold = self.orchestrator_config["min_score_threshold"]
    
    def load_prompts(self):
        self.prompts = {}
        prompt_files = {
            "generator": "startPrompt.txt",
            "meaningfulness": "chooserPrompt.txt", 
            "robustness": "suggestorPrompt.txt",
            "uniqueness": "childPrompt.txt",
            "progress": "rectorPrompt.txt"
        }
        
        for role, filename in prompt_files.items():
            try:
                with open(filename, 'r') as f:
                    self.prompts[role] = f.read().strip()
            except FileNotFoundError:
                logger.warning(f"Prompt file {filename} not found for role {role}")
                self.prompts[role] = f"You are a {role} assistant for educational content."
    
    async def generate_question(self, request: QuestionRequest) -> str:
        generator_config = self.lm_configs.get("generator")
        if not generator_config:
            raise ValueError("Generator configuration not found")
        
        async with LanguageModelClient(generator_config) as client:
            prompt = f"""
            Topic: {request.topic}
            Age Group: {request.age_group} 
            Difficulty Level: {request.difficulty_level}
            Context: {request.context or 'None'}
            
            Please generate an educational question following the requirements.
            """
            
            return await client.generate(prompt, self.prompts["generator"])
    
    async def validate_meaningfulness(self, question: str, request: QuestionRequest) -> ValidationScore:
        config = self.lm_configs.get("meaningfulness_checker")
        if not config:
            return ValidationScore(5.0, ValidationResult.REJECT, "Validator not configured", {})
        
        async with LanguageModelClient(config) as client:
            prompt = f"""
            Question to evaluate: {question}
            Target age group: {request.age_group}
            Topic: {request.topic}
            Difficulty: {request.difficulty_level}
            
            Please evaluate this question and provide your assessment in JSON format.
            """
            
            try:
                response = await client.generate(prompt, self.prompts["meaningfulness"])
                result = self._parse_validation_response(response)
                
                avg_score = sum(result["scores"].values()) / len(result["scores"])
                decision = ValidationResult.ACCEPT if avg_score >= self.min_score_threshold else ValidationResult.REJECT
                
                return ValidationScore(avg_score, decision, result.get("reasoning", ""), result)
                
            except Exception as e:
                logger.error(f"Meaningfulness validation failed: {e}")
                return ValidationScore(0.0, ValidationResult.REJECT, f"Validation error: {e}", {})
    
    async def validate_robustness(self, question: str) -> ValidationScore:
        config = self.lm_configs.get("robustness_validator")
        if not config:
            return ValidationScore(5.0, ValidationResult.FAIL, "Validator not configured", {})
        
        async with LanguageModelClient(config) as client:
            prompt = f"""
            Question to validate: {question}
            
            Please perform robustness validation and provide your assessment in JSON format.
            """
            
            try:
                response = await client.generate(prompt, self.prompts["robustness"])
                result = self._parse_validation_response(response)
                
                avg_score = sum(result["scores"].values()) / len(result["scores"])
                decision = ValidationResult.PASS if avg_score >= self.min_score_threshold else ValidationResult.FAIL
                
                return ValidationScore(avg_score, decision, result.get("feedback", ""), result)
                
            except Exception as e:
                logger.error(f"Robustness validation failed: {e}")
                return ValidationScore(0.0, ValidationResult.FAIL, f"Validation error: {e}", {})
    
    async def validate_uniqueness(self, question: str) -> ValidationScore:
        config = self.lm_configs.get("uniqueness_checker")
        if not config:
            return ValidationScore(5.0, ValidationResult.SIMILAR, "Validator not configured", {})
        
        async with LanguageModelClient(config) as client:
            # Provide context of existing questions for comparison
            existing_questions = "\n".join(self.question_database[-10:])  # Last 10 questions
            
            prompt = f"""
            New question: {question}
            
            Existing questions for comparison:
            {existing_questions}
            
            Please evaluate uniqueness and provide your assessment in JSON format.
            """
            
            try:
                response = await client.generate(prompt, self.prompts["uniqueness"])
                result = self._parse_validation_response(response)
                
                avg_score = sum(result["scores"].values()) / len(result["scores"])
                
                if avg_score >= 8.0:
                    decision = ValidationResult.UNIQUE
                elif avg_score >= 6.0:
                    decision = ValidationResult.SIMILAR
                else:
                    decision = ValidationResult.DUPLICATE
                
                return ValidationScore(avg_score, decision, result.get("assessment", ""), result)
                
            except Exception as e:
                logger.error(f"Uniqueness validation failed: {e}")
                return ValidationScore(5.0, ValidationResult.SIMILAR, f"Validation error: {e}", {})
    
    async def validate_progress(self, question: str, request: QuestionRequest) -> ValidationScore:
        config = self.lm_configs.get("progress_assessor")
        if not config:
            return ValidationScore(5.0, ValidationResult.REVISE, "Validator not configured", {})
        
        async with LanguageModelClient(config) as client:
            prompt = f"""
            Question: {question}
            Topic: {request.topic}
            Age Group: {request.age_group}
            Difficulty Level: {request.difficulty_level}
            
            Please assess learning progression value and provide your assessment in JSON format.
            """
            
            try:
                response = await client.generate(prompt, self.prompts["progress"])
                result = self._parse_validation_response(response)
                
                avg_score = sum(result["scores"].values()) / len(result["scores"])
                
                if avg_score >= self.min_score_threshold:
                    decision = ValidationResult.ADVANCE
                elif avg_score >= 5.0:
                    decision = ValidationResult.REVISE
                else:
                    decision = ValidationResult.REJECT
                
                return ValidationScore(avg_score, decision, result.get("recommendations", ""), result)
                
            except Exception as e:
                logger.error(f"Progress validation failed: {e}")
                return ValidationScore(0.0, ValidationResult.REJECT, f"Validation error: {e}", {})
    
    def _parse_validation_response(self, response: str) -> Dict[str, Any]:
        try:
            # Extract JSON from response if it contains additional text
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback: try to parse entire response
                return json.loads(response)
                
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse validation response as JSON: {response[:200]}...")
            # Return a basic structure
            return {
                "scores": {"overall": 5.0},
                "feedback": response[:500],
                "decision": "uncertain"
            }
    
    async def create_perfect_question(self, request: QuestionRequest) -> Dict[str, Any]:
        logger.info(f"Starting question creation for topic: {request.topic}")
        
        iteration = 0
        best_question = None
        best_scores = None
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{self.max_iterations}")
            
            try:
                # Generate question
                question = await self.generate_question(request)
                logger.info(f"Generated question: {question[:100]}...")
                
                # Run all validations in parallel
                validation_tasks = [
                    self.validate_meaningfulness(question, request),
                    self.validate_robustness(question), 
                    self.validate_uniqueness(question),
                    self.validate_progress(question, request)
                ]
                
                scores = await asyncio.gather(*validation_tasks)
                meaningfulness, robustness, uniqueness, progress = scores
                
                # Calculate overall score
                overall_score = (
                    meaningfulness.score * 0.3 +
                    robustness.score * 0.25 +
                    uniqueness.score * 0.25 +
                    progress.score * 0.2
                )
                
                logger.info(f"Scores - M:{meaningfulness.score:.1f} R:{robustness.score:.1f} U:{uniqueness.score:.1f} P:{progress.score:.1f} Overall:{overall_score:.1f}")
                
                # Check if all validations passed
                all_passed = (
                    meaningfulness.decision == ValidationResult.ACCEPT and
                    robustness.decision == ValidationResult.PASS and
                    uniqueness.decision in [ValidationResult.UNIQUE, ValidationResult.SIMILAR] and
                    progress.decision in [ValidationResult.ADVANCE, ValidationResult.REVISE]
                )
                
                current_result = {
                    "question": question,
                    "overall_score": overall_score,
                    "validation_results": {
                        "meaningfulness": asdict(meaningfulness),
                        "robustness": asdict(robustness), 
                        "uniqueness": asdict(uniqueness),
                        "progress": asdict(progress)
                    },
                    "iteration": iteration,
                    "passed_all_checks": all_passed
                }
                
                # Update best question if this one is better
                if best_question is None or overall_score > best_scores["overall_score"]:
                    best_question = question
                    best_scores = current_result
                
                # If all checks passed and score is high enough, we have perfection
                if all_passed and overall_score >= self.min_score_threshold:
                    logger.info(f"Perfect question achieved in iteration {iteration}!")
                    self.question_database.append(question)
                    return current_result
                
                # Log feedback for improvement
                feedback_items = []
                if meaningfulness.decision == ValidationResult.REJECT:
                    feedback_items.append(f"Meaningfulness: {meaningfulness.feedback}")
                if robustness.decision == ValidationResult.FAIL:
                    feedback_items.append(f"Robustness: {robustness.feedback}")
                if uniqueness.decision == ValidationResult.DUPLICATE:
                    feedback_items.append(f"Uniqueness: {uniqueness.feedback}")
                if progress.decision == ValidationResult.REJECT:
                    feedback_items.append(f"Progress: {progress.feedback}")
                
                if feedback_items:
                    logger.info(f"Improvement needed: {'; '.join(feedback_items[:2])}")
                
            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                continue
        
        # If we didn't achieve perfection, return the best attempt
        logger.info(f"Max iterations reached. Returning best question with score {best_scores['overall_score']:.1f}")
        self.question_database.append(best_question)
        return best_scores
    
    async def batch_create_questions(self, requests: List[QuestionRequest]) -> List[Dict[str, Any]]:
        logger.info(f"Creating {len(requests)} questions in batch")
        
        tasks = [self.create_perfect_question(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to create question {i+1}: {result}")
                processed_results.append({
                    "error": str(result),
                    "request": asdict(requests[i])
                })
            else:
                processed_results.append(result)
        
        return processed_results


async def main():
    creator = BouncyQuestionCreator()
    
    # Example usage
    request = QuestionRequest(
        topic="Basic Mathematics", 
        age_group="8-10 years",
        difficulty_level="beginner",
        context="Introduction to addition and subtraction"
    )
    
    try:
        result = await creator.create_perfect_question(request)
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        logger.error(f"Failed to create question: {e}")


if __name__ == "__main__":
    asyncio.run(main())