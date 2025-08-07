import json
import asyncio
import aiohttp
import logging
import csv
import pandas as pd
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
        self.max_iterations = 5  # Mother LM iteration limit
        self.principal_validation_rounds = 3  # Principal checks each question 3x
    
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
            "mother_llm": "mother_llm.txt",
            "principal": "principal_validator.txt",
            "kim_leicht": "persona_kim_leicht.txt",
            "alex_stammaufgabe": "persona_alex_stammaufgabe.txt", 
            "lisa_schwer": "persona_lisa_schwer.txt"
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
    
    async def hierarchical_question_creation(self, request: QuestionRequest) -> Dict[str, Any]:
        """Main hierarchical workflow: Mother -> Principal -> Students -> Iteration"""
        logger.info(f"Starting hierarchical question creation for topic: {request.topic}")
        
        best_question = None
        all_iterations = []
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"=== Iteration {iteration}/{self.max_iterations} ===")
            
            # Step 1: Mother LM creates question
            mother_result = await self.mother_llm_generate(request, iteration, 
                                                         previous_feedback=all_iterations[-1] if all_iterations else None)
            
            if not mother_result:
                logger.error(f"Mother LM failed in iteration {iteration}")
                continue
            
            # Step 2: Principal validates 3x
            principal_results = []
            for round_num in range(1, self.principal_validation_rounds + 1):
                logger.info(f"Principal validation round {round_num}/3")
                principal_result = await self.principal_validate(mother_result, round_num)
                principal_results.append(principal_result)
                
                # If Principal rejects, Mother LM must fix immediately
                if principal_result.get("gesamtbewertung") in ["NACHBESSERUNG_ERFORDERLICH", "ABLEHNUNG"]:
                    logger.info(f"Principal rejected question, requesting Mother LM fix")
                    mother_result = await self.mother_llm_fix(mother_result, principal_result)
            
            # Step 3: Student personas select best fit
            if principal_results[-1].get("gesamtbewertung") == "FREIGEGEBEN":
                student_feedback = await self.student_persona_evaluation(mother_result, request)
                
                iteration_result = {
                    "iteration": iteration,
                    "mother_result": mother_result,
                    "principal_results": principal_results,
                    "student_feedback": student_feedback,
                    "overall_score": self.calculate_iteration_score(principal_results[-1], student_feedback)
                }
                
                all_iterations.append(iteration_result)
                
                # Check if this is the best so far
                if not best_question or iteration_result["overall_score"] > best_question["overall_score"]:
                    best_question = iteration_result
                
                # If student personas are satisfied, we may have perfection
                if self.check_perfection_achieved(student_feedback):
                    logger.info(f"Perfection achieved in iteration {iteration}!")
                    break
            else:
                logger.warning(f"Principal never approved question in iteration {iteration}")
        
        if best_question:
            self.question_database.append(best_question["mother_result"]["frage_inhalt"]["aufgabenstellung"])
            
        return {
            "best_result": best_question,
            "all_iterations": all_iterations,
            "total_iterations": len(all_iterations),
            "perfection_achieved": best_question and best_question.get("perfection_achieved", False)
        }

    async def mother_llm_generate(self, request: QuestionRequest, iteration: int, previous_feedback=None) -> Dict[str, Any]:
        """Mother LM generates question based on parameters and previous feedback"""
        config = self.lm_configs.get("generator")  # Reuse generator config for mother
        if not config:
            raise ValueError("Mother LM configuration not found")
        
        async with LanguageModelClient(config) as client:
            feedback_context = ""
            if previous_feedback:
                feedback_context = f"""
                Previous iteration feedback:
                Principal: {previous_feedback.get('principal_results', [{}])[-1].get('principal_feedback', 'N/A')}
                Students: {previous_feedback.get('student_feedback', {}).get('combined_feedback', 'N/A')}
                
                Please improve the question based on this feedback.
                """
            
            prompt = f"""
            Iteration: {iteration}
            Topic: {request.topic}
            Age Group: {request.age_group}
            Difficulty Level: {request.difficulty_level}
            Context: {request.context or 'None'}
            
            {feedback_context}
            
            Create a German economics question following all parameter requirements.
            """
            
            try:
                response = await client.generate(prompt, self.prompts["mother_llm"])
                return self._parse_mother_response(response)
            except Exception as e:
                logger.error(f"Mother LM generation failed: {e}")
                return None

    async def principal_validate(self, mother_result: Dict[str, Any], round_num: int) -> Dict[str, Any]:
        """Principal performs 3-round technical validation"""
        config = self.lm_configs.get("meaningfulness_checker")  # Reuse for principal
        if not config:
            return {"gesamtbewertung": "ABLEHNUNG", "principal_feedback": "Principal not configured"}
        
        async with LanguageModelClient(config) as client:
            prompt = f"""
            Validation Round: {round_num}/3
            Question to validate: {json.dumps(mother_result, indent=2, ensure_ascii=False)}
            
            Perform your {round_num} validation check as school principal.
            """
            
            try:
                response = await client.generate(prompt, self.prompts["principal"])
                return self._parse_validation_response(response)
            except Exception as e:
                logger.error(f"Principal validation failed: {e}")
                return {"gesamtbewertung": "ABLEHNUNG", "principal_feedback": f"Validation error: {e}"}

    async def student_persona_evaluation(self, mother_result: Dict[str, Any], request: QuestionRequest) -> Dict[str, Any]:
        """Student personas evaluate question appropriateness"""
        difficulty = mother_result.get("parameter_settings", {}).get("variation", "Stammaufgabe")
        
        # Select appropriate persona based on difficulty
        persona_mapping = {
            "leicht": "kim_leicht",
            "Stammaufgabe": "alex_stammaufgabe", 
            "schwer": "lisa_schwer"
        }
        
        primary_persona = persona_mapping.get(difficulty, "alex_stammaufgabe")
        
        # Get feedback from primary persona
        config = self.lm_configs.get("uniqueness_checker")  # Reuse for personas
        if not config:
            return {"error": "Student persona not configured"}
        
        async with LanguageModelClient(config) as client:
            prompt = f"""
            Question for evaluation: {json.dumps(mother_result, indent=2, ensure_ascii=False)}
            Target difficulty: {difficulty}
            
            Evaluate this question from your student perspective.
            """
            
            try:
                response = await client.generate(prompt, self.prompts[primary_persona])
                return {
                    "primary_persona": primary_persona,
                    "persona_feedback": self._parse_validation_response(response),
                    "difficulty_match": difficulty
                }
            except Exception as e:
                logger.error(f"Student persona evaluation failed: {e}")
                return {"error": f"Evaluation failed: {e}"}

    def calculate_iteration_score(self, principal_result: Dict[str, Any], student_feedback: Dict[str, Any]) -> float:
        """Calculate overall score for iteration"""
        principal_scores = principal_result.get("principal_pruefung_1_parameter", {}).get("parameter_score", 0)
        principal_scores += principal_result.get("principal_pruefung_2_distraktoren", {}).get("distraktoren_score", 0)  
        principal_scores += principal_result.get("principal_pruefung_3_curriculum", {}).get("curriculum_score", 0)
        principal_avg = principal_scores / 3 if principal_scores > 0 else 0
        
        student_score = 50  # Default if parsing fails
        persona_feedback = student_feedback.get("persona_feedback", {})
        if isinstance(persona_feedback, dict):
            # Extract numeric scores from student feedback
            student_score = persona_feedback.get("gesamtbewertung_score", 50)
        
        return (principal_avg * 0.6 + student_score * 0.4)

    def check_perfection_achieved(self, student_feedback: Dict[str, Any]) -> bool:
        """Check if student personas indicate perfection"""
        persona_feedback = student_feedback.get("persona_feedback", {})
        if isinstance(persona_feedback, dict):
            return persona_feedback.get("gesamtbewertung") == "GEEIGNET"
        return False

    def _parse_mother_response(self, response: str) -> Dict[str, Any]:
        """Parse Mother LM JSON response"""
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse Mother LM response: {response[:200]}...")
            return {"error": "Parse failed", "raw_response": response[:500]}

    async def mother_llm_fix(self, mother_result: Dict[str, Any], principal_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Mother LM fixes question based on Principal feedback"""
        config = self.lm_configs.get("generator")
        if not config:
            return mother_result  # Return unchanged if no config
        
        async with LanguageModelClient(config) as client:
            prompt = f"""
            Original question: {json.dumps(mother_result, indent=2, ensure_ascii=False)}
            
            Principal feedback requiring fixes:
            {json.dumps(principal_feedback, indent=2, ensure_ascii=False)}
            
            Please fix the question immediately based on the Principal's feedback.
            Address all parameter compliance issues and distractor problems.
            """
            
            try:
                response = await client.generate(prompt, self.prompts["mother_llm"])
                fixed_result = self._parse_mother_response(response)
                return fixed_result if fixed_result and "error" not in fixed_result else mother_result
            except Exception as e:
                logger.error(f"Mother LM fix failed: {e}")
                return mother_result

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

    def export_questions_to_csv(self, results: List[Dict[str, Any]], filename: str = "generated_questions.csv"):
        """Export questions to CSV in exact stakeholder format (58 columns)"""
        
        # Define all 58 columns in exact order
        csv_columns = [
            "c_id", "subject", "type", "text", "answers",
            "p.variation", "p.taxanomy_level", "p.mathematical_requirement_level",
            "p.instruction_explicitness_of_instruction", "p.instruction_obstacle_complex_np",
            "p.instruction_obstacle_negation", "p.instruction_obstacle_passive",
            "p.instruction_number_of_sentences",
            "p.item_1_answer_verbatim_explanatory_text", "p.item_1_obstacle_complex_np", 
            "p.item_1_obstacle_negation", "p.item_1_obstacle_passive", "p.item_1_sentence_length",
            "p.item_2_answer_verbatim_explanatory_text", "p.item_2_obstacle_complex_np",
            "p.item_2_obstacle_negation", "p.item_2_obstacle_passive", "p.item_2_sentence_length",
            "p.item_3_answer_verbatim_explanatory_text", "p.item_3_obstacle_complex_np",
            "p.item_3_obstacle_negation", "p.item_3_obstacle_passive", "p.item_3_sentence_length",
            "p.item_4_answer_verbatim_explanatory_text", "p.item_4_obstacle_complex_np",
            "p.item_4_obstacle_negation", "p.item_4_obstacle_passive", "p.item_4_sentence_length",
            "p.item_5_answer_verbatim_explanatory_text", "p.item_5_obstacle_complex_np",
            "p.item_5_obstacle_negation", "p.item_5_obstacle_passive", "p.item_5_sentence_length",
            "p.item_6_answer_verbatim_explanatory_text", "p.item_6_obstacle_complex_np",
            "p.item_6_obstacle_negation", "p.item_6_obstacle_passive", "p.item_6_sentence_length",
            "p.item_7_answer_verbatim_explanatory_text", "p.item_7_obstacle_complex_np",
            "p.item_7_obstacle_negation", "p.item_7_obstacle_passive", "p.item_7_sentence_length",
            "p.item_8_answer_verbatim_explanatory_text", "p.item_8_obstacle_complex_np",
            "p.item_8_obstacle_negation", "p.item_8_obstacle_passive", "p.item_8_sentence_length",
            "p.root_text_contains_irrelevant_information", "p.root_text_obstacle_complex_np",
            "p.root_text_obstacle_negation", "p.root_text_obstacle_passive",
            "p.root_text_reference_explanatory_text"
        ]
        
        csv_data = []
        
        for result in results:
            if "best_result" in result and result["best_result"]:
                best_result = result["best_result"]
                mother_result = best_result.get("mother_result", {})
                csv_ready = mother_result.get("csv_ready_question", {})
                
                if csv_ready:
                    # Create row with all columns, filling missing values with empty strings
                    row = {}
                    for col in csv_columns:
                        row[col] = csv_ready.get(col, "")
                    
                    csv_data.append(row)
        
        # Write to CSV file
        if csv_data:
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                    writer.writeheader()
                    writer.writerows(csv_data)
                
                logger.info(f"Successfully exported {len(csv_data)} questions to {filename}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to export CSV: {e}")
                return False
        else:
            logger.warning("No valid questions to export")
            return False

    def load_reference_texts(self) -> Dict[str, str]:
        """Load reference texts from explanation_metadata.csv"""
        try:
            df = pd.read_csv("../providedProjectFromStakeHolder/explanation_metadata.csv")
            
            reference_texts = {}
            for _, row in df.iterrows():
                reference_texts[row['c_id']] = row['text']
            
            logger.info(f"Loaded {len(reference_texts)} reference texts")
            return reference_texts
            
        except Exception as e:
            logger.error(f"Failed to load reference texts: {e}")
            return {}


async def main():
    creator = BouncyQuestionCreator()
    
    # German economics question
    request = QuestionRequest(
        topic="Bedürfnisse und Güter", 
        age_group="9. Klasse",
        difficulty_level="leicht",
        context="Wirtschaftliche Grundbegriffe nach Referenztext"
    )
    
    try:
        # Use new hierarchical workflow: Generator -> Principal -> Students (5 iterations max)
        result = await creator.hierarchical_question_creation(request)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        logger.error(f"Failed to create question: {e}")


if __name__ == "__main__":
    asyncio.run(main())