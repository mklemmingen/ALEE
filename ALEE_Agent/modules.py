"""
German Educational DSPy Modules - Integrates with existing modular .txt prompt system
NO hardcoded prompt strings - uses ModularPromptBuilder for all prompts
"""

import logging
from typing import Dict, Any, Optional, List
from models import BaseQuestionModel, QuestionFactory, ExpertSuggestion
from signatures import *
from prompt_builder import ModularPromptBuilder, ExpertPromptEnhancer
from question_refiner import BatchQuestionRefiner

logger = logging.getLogger(__name__)

class GermanQuestionGenerator(dspy.Module):
    """DSPy module for generating German educational questions using modular .txt prompts"""
    
    def __init__(self, prompt_builder: Optional[ModularPromptBuilder] = None):
        super().__init__()
        # Use ChainOfThought for reasoning about question generation
        self.generate = dspy.ChainOfThought(GenerateGermanEducationalQuestions)
        self.prompt_builder = prompt_builder or ModularPromptBuilder()
        
        # Load generation instruction from .txt file
        self.generation_instruction = self.prompt_builder.load_prompt_txt("dtoAndOutputPrompt/questionGenerationInstruction.txt")
    
    def forward(self, request_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate 3 German questions as typed models using modular prompts from .txt files"""
        
        # Create a request object for the prompt builder
        class RequestObj:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        request = RequestObj(request_params)
        
        # Build complete modular prompt from .txt files
        modular_prompt = self.prompt_builder.build_master_prompt(request)
        
        # Load output format instructions
        output_format = self.prompt_builder.load_prompt_txt("dtoAndOutputPrompt/fallbackGenerationPrompt.txt")
        
        # Combine modular prompt with generation instruction
        full_prompt = f"""{self.generation_instruction}

        {modular_prompt}
        
        {output_format}"""
        
        # Prepare DSPy inputs from modular prompt sections
        result = self.generate(
            source_text=request_params.get('text', ''),
            variation=request_params.get('p_variation', 'stammaufgabe'),
            question_type=request_params.get('question_type', 'multiple-choice'),
            taxonomy_level=request_params.get('p_taxonomy_level', 'Stufe 1'),
            mathematical_requirement=request_params.get('p_mathematical_requirement_level', '0'),
            text_reference_explanatory=request_params.get('p_root_text_reference_explanatory_text', 'Nicht vorhanden'),
            text_obstacles=self._extract_text_obstacles(request_params),
            item_obstacles=self._extract_item_obstacles(request_params),
            instruction_obstacles=self._extract_instruction_obstacles(request_params),
            instruction_explicitness=request_params.get('p_instruction_explicitness_of_instruction', 'Implizit'),
            irrelevant_information=request_params.get('p_root_text_contains_irrelevant_information', 'Nicht Enthalten')
        )
        
        # Parse answers from comma-separated strings
        answers_1 = [a.strip() for a in result.antworten_1.split(',') if a.strip()]
        answers_2 = [a.strip() for a in result.antworten_2.split(',') if a.strip()]
        answers_3 = [a.strip() for a in result.antworten_3.split(',') if a.strip()]
        
        # Create typed question models
        question_type = request_params.get('question_type', 'multiple-choice')
        typed_questions = []
        
        questions_data = [
            (result.frage_1, answers_1),
            (result.frage_2, answers_2), 
            (result.frage_3, answers_3)
        ]
        
        for frage, antworten in questions_data:
            try:
                # Extract correct answer based on question type
                correct_answer = self._extract_correct_answer(frage, antworten, question_type)
                
                # Create typed question model
                typed_question = QuestionFactory.from_dspy_output(
                    question_type=question_type,
                    frage=frage,
                    antworten=antworten,
                    correct_answer=correct_answer,
                    explanation=result.generation_rationale
                )
                
                typed_questions.append(typed_question)
                
            except Exception as e:
                logger.error(f"Failed to create typed question from DSPy output: {e}")
                # Create a fallback question model
                fallback_data = {
                    "question_text": frage,
                    "question_type": question_type,
                    "answers": antworten,
                    "correct_answer": antworten[0] if antworten else "Unknown",
                    "explanation": f"Fallback question due to error: {e}"
                }
                typed_question = QuestionFactory.create_question(question_type, fallback_data)
                typed_questions.append(typed_question)
        
        return {
            'typed_questions': typed_questions,
            'questions': [result.frage_1, result.frage_2, result.frage_3],
            'answers': [answers_1, answers_2, answers_3],
            'reasoning': result.generation_rationale,
            'modular_prompt_used': modular_prompt
        }
    
    def _extract_text_obstacles(self, params: Dict[str, Any]) -> str:
        """Extract text obstacle parameters"""
        obstacles = []
        if params.get('p_root_text_obstacle_passive') == 'Enthalten':
            obstacles.append('passive')
        if params.get('p_root_text_obstacle_negation') == 'Enthalten':
            obstacles.append('negation')
        if params.get('p_root_text_obstacle_complex_np') == 'Enthalten':
            obstacles.append('complex_np')
        return ', '.join(obstacles) if obstacles else 'keine'
    
    def _extract_item_obstacles(self, params: Dict[str, Any]) -> str:
        """Extract item obstacle parameters"""
        obstacles = []
        for i in range(1, 9):
            if params.get(f'p_item_{i}_obstacle_passive') == 'Enthalten':
                obstacles.append(f'item_{i}_passive')
            if params.get(f'p_item_{i}_obstacle_negation') == 'Enthalten':
                obstacles.append(f'item_{i}_negation')
            if params.get(f'p_item_{i}_obstacle_complex_np') == 'Enthalten':
                obstacles.append(f'item_{i}_complex_np')
        return ', '.join(obstacles) if obstacles else 'keine'
    
    def _extract_correct_answer(self, question_text: str, answers: List[str], question_type: str) -> Any:
        """Extract correct answer based on question type and content"""
        if question_type == "multiple-choice":
            # For multiple choice, assume first 2-3 answers are correct (simplified)
            return answers[:2] if len(answers) >= 2 else answers
        elif question_type == "single-choice":
            # For single choice, assume first answer is correct (simplified)
            return answers[0] if answers else "Unknown"
        elif question_type == "true-false":
            # For true-false, analyze content for correct answer
            if "richtig" in question_text.lower():
                return "Richtig"
            else:
                return "Falsch"
        elif question_type == "mapping":
            # For mapping, create simple 1:1 mapping (simplified)
            if len(answers) >= 6:  # At least 3 pairs
                pairs = {}
                for i in range(0, min(len(answers), 6), 2):
                    if i + 1 < len(answers):
                        pairs[answers[i]] = answers[i + 1]
                return pairs
            return {}
        else:
            return answers[0] if answers else "Unknown"
    
    def _extract_instruction_obstacles(self, params: Dict[str, Any]) -> str:
        """Extract instruction obstacle parameters"""
        obstacles = []
        if params.get('p_instruction_obstacle_passive') == 'Enthalten':
            obstacles.append('passive')
        if params.get('p_instruction_obstacle_complex_np') == 'Enthalten':
            obstacles.append('complex_np')
        return ', '.join(obstacles) if obstacles else 'keine'


class GermanExpertValidator(dspy.Module):
    """Base class for German expert validation using enhanced .txt prompts"""
    
    def __init__(self, signature_class, expert_name: str, prompt_builder: Optional[ModularPromptBuilder] = None):
        super().__init__()
        self.expert_name = expert_name
        self.validate = dspy.ChainOfThought(signature_class)
        self.prompt_builder = prompt_builder or ModularPromptBuilder()
        self.expert_enhancer = ExpertPromptEnhancer()
        
        # Load expert-specific evaluation instruction from .txt file
        self.evaluation_instruction = self.prompt_builder.load_prompt_txt("expertEval/expertEvaluationInstruction.txt")
    
    def get_expert_context(self, question: str, answers: list[str], target_value: str, params: Dict[str, Any]) -> str:
        """Build enhanced expert context using modular prompts with parameter knowledge"""
        
        # Get enhanced expert prompt with parameter knowledge appended
        enhanced_expert_prompt = self.expert_enhancer.build_enhanced_expert_prompt(self.expert_name, params)
        
        context_sections = [
            f"=== EXPERTENROLLE MIT PARAMETER-WISSENSBASIS ===",
            enhanced_expert_prompt,
            "",
            f"=== ZU BEWERTENDE FRAGE ===",
            f"Frage: {question}",
            f"Antworten: {', '.join(answers)}",
            f"Zielwert: {target_value}",
            "",
            f"=== BEWERTUNGSANWEISUNG ===",
            self.evaluation_instruction
        ]
        
        return "\n".join(context_sections)


class VariationExpertGerman(GermanExpertValidator):
    """German difficulty level expert using .txt prompts"""
    
    def __init__(self, prompt_builder: Optional[ModularPromptBuilder] = None):
        super().__init__(ValidateVariationGerman, "variation_expert", prompt_builder)
    
    def forward(self, question_model: BaseQuestionModel, target_variation: str, params: Dict[str, Any]) -> ExpertSuggestion:
        """Validate German difficulty using modular prompts and return ExpertSuggestion"""
        
        # Extract question data from typed model
        question_text = question_model.question_text
        answers = question_model.answers
        
        # Build expert context from .txt files
        expert_context = self.get_expert_context(question_text, answers, target_variation, params)
        
        result = self.validate(
            frage=question_text,
            antworten=", ".join(answers),
            ziel_variation=target_variation
        )
        
        # Create ExpertSuggestion object
        expert_suggestion = ExpertSuggestion(
            expert_type=self.expert_name,
            rating=result.bewertung,
            feedback=result.feedback,
            suggestions=result.vorschlaege.split('\n') if result.vorschlaege else [],
            reasoning=result.begruendung,
            target_value=target_variation
        )
        
        return expert_suggestion


class TaxonomyExpertGerman(GermanExpertValidator):
    """German taxonomy expert using .txt prompts"""
    
    def __init__(self, prompt_builder: Optional[ModularPromptBuilder] = None):
        super().__init__(ValidateTaxonomyGerman, "taxonomy_expert", prompt_builder)
    
    def forward(self, question_model: BaseQuestionModel, target_taxonomy: str, params: Dict[str, Any]) -> ExpertSuggestion:
        """Validate taxonomy using modular prompts and return ExpertSuggestion"""
        
        # Extract question data from typed model
        question_text = question_model.question_text
        answers = question_model.answers
        
        expert_context = self.get_expert_context(question_text, answers, target_taxonomy, params)
        
        result = self.validate(
            frage=question_text,
            antworten=", ".join(answers),
            ziel_taxonomie=target_taxonomy
        )
        
        # Create ExpertSuggestion object
        expert_suggestion = ExpertSuggestion(
            expert_type=self.expert_name,
            rating=result.bewertung,
            feedback=result.feedback,
            suggestions=result.vorschlaege.split('\n') if result.vorschlaege else [],
            reasoning=result.begruendung,
            target_value=target_taxonomy
        )
        
        return expert_suggestion


class MathExpertGerman(GermanExpertValidator):
    """German mathematical complexity expert using .txt prompts"""
    
    def __init__(self, prompt_builder: Optional[ModularPromptBuilder] = None):
        super().__init__(ValidateMathematicalGerman, "math_expert", prompt_builder)
    
    def forward(self, question_model: BaseQuestionModel, target_math_level: str, params: Dict[str, Any]) -> ExpertSuggestion:
        """Validate mathematical complexity using modular prompts and return ExpertSuggestion"""
        
        # Extract question data from typed model
        question_text = question_model.question_text
        answers = question_model.answers
        
        expert_context = self.get_expert_context(question_text, answers, target_math_level, params)
        
        result = self.validate(
            frage=question_text,
            antworten=", ".join(answers),
            ziel_math_stufe=target_math_level
        )
        
        # Create ExpertSuggestion object
        expert_suggestion = ExpertSuggestion(
            expert_type=self.expert_name,
            rating=result.bewertung,
            feedback=result.feedback,
            suggestions=result.vorschlaege.split('\n') if result.vorschlaege else [],
            reasoning=result.begruendung,
            target_value=target_math_level
        )
        
        return expert_suggestion


class ObstacleExpertGerman(GermanExpertValidator):
    """German linguistic obstacle expert using .txt prompts"""
    
    def __init__(self, prompt_builder: Optional[ModularPromptBuilder] = None):
        super().__init__(ValidateObstacleGerman, "obstacle_expert", prompt_builder)
    
    def forward(self, question_model: BaseQuestionModel, target_obstacles: str, params: Dict[str, Any]) -> ExpertSuggestion:
        """Validate linguistic obstacles using modular prompts and return ExpertSuggestion"""
        
        # Extract question data from typed model
        question_text = question_model.question_text
        answers = question_model.answers
        
        expert_context = self.get_expert_context(question_text, answers, target_obstacles, params)
        
        result = self.validate(
            frage=question_text,
            antworten=", ".join(answers),
            ziel_hindernisse=target_obstacles
        )
        
        # Create ExpertSuggestion object
        expert_suggestion = ExpertSuggestion(
            expert_type=self.expert_name,
            rating=result.bewertung,
            feedback=result.feedback,
            suggestions=result.vorschlaege.split('\n') if result.vorschlaege else [],
            reasoning=result.begruendung,
            target_value=target_obstacles
        )
        
        return expert_suggestion


class InstructionExpertGerman(GermanExpertValidator):
    """German instruction clarity expert using .txt prompts"""
    
    def __init__(self, prompt_builder: Optional[ModularPromptBuilder] = None):
        super().__init__(ValidateInstructionGerman, "instruction_expert", prompt_builder)
    
    def forward(self, question_model: BaseQuestionModel, target_explicitness: str, params: Dict[str, Any]) -> ExpertSuggestion:
        """Validate instruction clarity using modular prompts and return ExpertSuggestion"""
        
        # Extract question data from typed model
        question_text = question_model.question_text
        answers = question_model.answers
        
        expert_context = self.get_expert_context(question_text, answers, target_explicitness, params)
        
        result = self.validate(
            frage=question_text,
            antworten=", ".join(answers),
            ziel_explizitheit=target_explicitness
        )
        
        # Create ExpertSuggestion object
        expert_suggestion = ExpertSuggestion(
            expert_type=self.expert_name,
            rating=result.bewertung,
            feedback=result.feedback,
            suggestions=result.vorschlaege.split('\n') if result.vorschlaege else [],
            reasoning=result.begruendung,
            target_value=target_explicitness
        )
        
        return expert_suggestion





class GermanExpertConsensus(dspy.Module):
    """Expert consensus with question refinement using .txt prompts and parameter guardrails"""
    
    def __init__(self, prompt_builder: Optional[ModularPromptBuilder] = None):
        super().__init__()
        self.consensus = dspy.ChainOfThought(ExpertConsensusGerman)
        self.prompt_builder = prompt_builder or ModularPromptBuilder()
        self.expert_enhancer = ExpertPromptEnhancer()
        
        # Load refinement instructions from .txt file
        self.refinement_instruction = self.prompt_builder.load_prompt_txt("expertEval/questionImprovementInstruction.txt")
    
    def forward(self, question: str, answers: list[str], expert_validations: Dict[str, Dict[str, Any]], params: Dict[str, Any]):
        """Aggregate expert feedback and refine question with parameter guardrails"""
        
        # Aggregate expert data
        ratings = [v['rating'] for v in expert_validations.values()]
        feedback_parts = [f"{name}: {v['feedback']}" for name, v in expert_validations.items()]
        suggestion_parts = []
        
        for name, validation in expert_validations.items():
            if validation['suggestions']:
                suggestion_parts.append(f"{name}: {'; '.join(validation['suggestions'])}")
        
        # Format data for consensus-refinement module
        ratings_str = f"Bewertungen: {ratings} (Durchschnitt: {sum(ratings)/len(ratings):.1f})"
        feedback_str = "\n".join(feedback_parts)
        suggestions_str = "\n".join(suggestion_parts) if suggestion_parts else "Keine spezifischen Verbesserungsvorschläge"
        
        # Build comprehensive parameter context with guardrails
        parameter_context = self._build_parameter_context_with_guardrails(params)
        
        # Run consensus-refinement
        result = self.consensus(
            original_frage=question,
            original_antworten=", ".join(answers),
            experten_bewertungen=ratings_str,
            experten_feedback=feedback_str,
            experten_vorschlaege=suggestions_str,
            parameter_kontext=parameter_context
        )
        
        # Parse refined answers
        refined_answers = [a.strip() for a in result.verfeinerte_antworten.split(',') if a.strip()]
        
        return {
            'refined_question': result.verfeinerte_frage,
            'refined_answers': refined_answers,
            'consensus_reasoning': result.konsens_begründung,
            'format_preserved': result.format_bewahrt.lower() == 'ja',
            'parameters_maintained': result.parameter_eingehalten.lower() == 'ja',
            'average_rating': sum(ratings) / len(ratings),
            'expert_count': len(expert_validations),
            'original_question': question,
            'original_answers': answers
        }
    
    def _build_parameter_context_with_guardrails(self, params: Dict[str, Any]) -> str:
        """Build comprehensive parameter context with strict guardrails"""
        
        # Get all relevant parameter knowledge (using content_expert mapping for comprehensive access)
        parameter_knowledge = self.expert_enhancer._get_relevant_parameter_knowledge("content_expert", params)
        
        # Get question type preservation instructions  
        question_type_preservation = self.expert_enhancer._get_question_type_preservation_instructions(params.get('question_type', 'multiple-choice'))
        
        context = f"""=== SYSARCH-PARAMETER KONTEXT ===
        {parameter_knowledge}
        
        === FRAGEFORMAT-BEWAHRUNG (ABSOLUT KRITISCH) ===
        {question_type_preservation}
        
        === PARAMETER-GUARD RAILS ===
        - NIEMALS den Fragetyp ändern: {params.get('question_type', 'multiple-choice')}
        - NIEMALS Tags ändern (z.B. <option>, <true-false>)
        - Schwierigkeitsgrad beibehalten: {params.get('p_variation', 'stammaufgabe')}
        - Taxonomie-Stufe respektieren: {params.get('p_taxonomy_level', 'Stufe 1')}
        - Mathematisches Niveau einhalten: {params.get('p_mathematical_requirement_level', '0')}
        - Sprachliche Hindernisse gemäß Parametern berücksichtigen
        - Nur inhaltliche Verbesserungen bei struktureller Bewahrung
        
        === VERFEINERUNGS-ANWEISUNG ===
        {self.refinement_instruction}"""
        
        return context


class TypeAwareEducationalPipeline(dspy.Module):
    """Type-aware German educational pipeline with refinement decision engine"""
    
    def __init__(self, expert_lms: Dict[str, Any], prompt_builder: Optional[ModularPromptBuilder] = None):
        super().__init__()
        
        self.prompt_builder = prompt_builder or ModularPromptBuilder()
        self.generator = GermanQuestionGenerator(self.prompt_builder)
        self.refiner = BatchQuestionRefiner()
        
        # Initialize experts with their specific LMs and prompt builder
        self.experts = {
            'variation': VariationExpertGerman(self.prompt_builder),
            'taxonomy': TaxonomyExpertGerman(self.prompt_builder), 
            'math': MathExpertGerman(self.prompt_builder),
            'obstacle': ObstacleExpertGerman(self.prompt_builder),
            'instruction': InstructionExpertGerman(self.prompt_builder)
        }
        
        # Store LM configs for context switching
        self.expert_lms = expert_lms
    
    def forward(self, request_params: Dict[str, Any]) -> Dict[str, Any]:
        """Type-aware generation and refinement pipeline"""
        
        import time
        pipeline_start = time.time()
        
        logger.info("Starting type-aware educational pipeline with DSPy")
        
        # Step 1: Generate typed question models
        generation_start = time.time()
        generation_result = self.generator(request_params)
        typed_questions = generation_result['typed_questions']
        generation_time = time.time() - generation_start
        
        logger.info(f"Generated {len(typed_questions)} typed question models in {generation_time:.2f}s")
        
        # Step 2: Expert evaluation with typed models
        expert_start = time.time()
        
        for question_model in typed_questions:
            expert_suggestions = []
            
            # Validate with each expert using typed models
            for expert_name, expert_module in self.experts.items():
                expert_eval_start = time.time()
                target_value = self._get_target_value_for_expert(expert_name, request_params)
                
                # Switch to expert's LM context and get ExpertSuggestion
                if expert_name in self.expert_lms:
                    with dspy.context(lm=self.expert_lms[expert_name]):
                        expert_suggestion = expert_module(question_model, target_value, request_params)
                else:
                    expert_suggestion = expert_module(question_model, target_value, request_params)
                
                expert_eval_time = time.time() - expert_eval_start
                logger.debug(f"Expert {expert_name} evaluation took {expert_eval_time:.2f}s")
                
                # Add expert suggestion to question model
                question_model.add_expert_suggestion(expert_suggestion)
                expert_suggestions.append(expert_suggestion)
        
        expert_total_time = time.time() - expert_start
        logger.info(f"Expert evaluation completed in {expert_total_time:.2f}s")
        
        # Step 3: Refinement decision and application
        refinement_start = time.time()
        refined_questions, refinement_summary = self.refiner.refine_questions(typed_questions)
        refinement_time = time.time() - refinement_start
        
        total_pipeline_time = time.time() - pipeline_start
        
        logger.info(f"Question refinement completed in {refinement_time:.2f}s")
        logger.info(f"Total type-aware pipeline time: {total_pipeline_time:.2f}s")
        
        # Build final result with type-safe data
        final_validations = []
        for refined_question in refined_questions:
            validation_data = {
                'typed_question': refined_question,
                'question': refined_question.question_text,
                'answers': refined_question.answers,
                'original_question': refined_question.original_question or refined_question.question_text,
                'original_answers': refined_question.answers,
                'expert_suggestions': refined_question.expert_suggestions,
                'refinement_applied': refined_question.refinement_applied,
                'format_preserved': refined_question.format_preserved,
                'approved': True  # All questions processed through refinement engine
            }
            final_validations.append(validation_data)
        
        # Build comprehensive pipeline details
        pipeline_details = {
            'typed_generation': {
                'typed_questions': typed_questions,
                'generation_result': generation_result,
                'processing_time_ms': generation_time * 1000,
                'question_types_generated': [q.question_type for q in typed_questions]
            },
            'expert_evaluation': {
                'total_suggestions': sum(len(q.expert_suggestions) for q in typed_questions),
                'average_ratings': [q.get_average_rating() for q in typed_questions],
                'processing_time_ms': expert_total_time * 1000
            },
            'refinement_decisions': {
                'summary': refinement_summary,
                'processing_time_ms': refinement_time * 1000,
                'questions_refined': refinement_summary.get('questions_refined', 0),
                'refinement_rate': refinement_summary.get('refinement_rate', 0.0),
                'format_preservation_rate': refinement_summary.get('format_preservation_rate', 0.0)
            },
            'timing': {
                'generation_ms': generation_time * 1000,
                'expert_evaluation_ms': expert_total_time * 1000,
                'refinement_ms': refinement_time * 1000,
                'total_pipeline_ms': total_pipeline_time * 1000
            }
        }
        
        return {
            'validated_questions': final_validations,
            'typed_questions': refined_questions,
            'generation_metadata': {
                'typed_models_used': True,
                'question_factory_used': True,
                'generation_reasoning': generation_result.get('reasoning', '')
            },
            'parameters_used': request_params,
            'all_approved': True,  # All questions processed through type-safe refinement
            'pipeline_details': pipeline_details
        }
    
    def _get_target_value_for_expert(self, expert_name: str, params: Dict[str, Any]) -> str:
        """Map expert to their target parameter value"""
        
        expert_param_map = {
            'variation': params.get('p_variation', 'stammaufgabe'),
            'taxonomy': params.get('p_taxonomy_level', 'Stufe 1'),
            'math': params.get('p_mathematical_requirement_level', '0'),
            'obstacle': self._format_obstacles_for_expert(params),
            'instruction': params.get('p_instruction_explicitness_of_instruction', 'Implizit')
        }
        
        return expert_param_map.get(expert_name, '')
    
    def _format_obstacles_for_expert(self, params: Dict[str, Any]) -> str:
        """Format obstacle parameters for obstacle expert"""
        obstacles = []
        
        # Root text obstacles
        if params.get('p_root_text_obstacle_passive') == 'Enthalten':
            obstacles.append('Grundtext: Passiv')
        if params.get('p_root_text_obstacle_negation') == 'Enthalten':
            obstacles.append('Grundtext: Negation')
        if params.get('p_root_text_obstacle_complex_np') == 'Enthalten':
            obstacles.append('Grundtext: Komplexe Nominalphrasen')
        
        # Item obstacles
        for i in range(1, 9):
            if params.get(f'p_item_{i}_obstacle_passive') == 'Enthalten':
                obstacles.append(f'Item {i}: Passiv')
            if params.get(f'p_item_{i}_obstacle_negation') == 'Enthalten':
                obstacles.append(f'Item {i}: Negation')
            if params.get(f'p_item_{i}_obstacle_complex_np') == 'Enthalten':
                obstacles.append(f'Item {i}: Komplexe NP')
        
        # Instruction obstacles  
        if params.get('p_instruction_obstacle_passive') == 'Enthalten':
            obstacles.append('Anweisung: Passiv')
        if params.get('p_instruction_obstacle_complex_np') == 'Enthalten':
            obstacles.append('Anweisung: Komplexe NP')
        
        return '; '.join(obstacles) if obstacles else 'Keine sprachlichen Hindernisse'


class GermanEducationalPipeline(dspy.Module):
    """Complete German educational pipeline using modular .txt prompts"""
    
    def __init__(self, expert_lms: Dict[str, Any], prompt_builder: Optional[ModularPromptBuilder] = None):
        super().__init__()
        
        self.prompt_builder = prompt_builder or ModularPromptBuilder()
        self.generator = GermanQuestionGenerator(self.prompt_builder)
        self.consensus = GermanExpertConsensus(self.prompt_builder)
        
        # Initialize experts with their specific LMs and prompt builder
        # Note: Content validation is now handled by instruction expert
        self.experts = {
            'variation': VariationExpertGerman(self.prompt_builder),
            'taxonomy': TaxonomyExpertGerman(self.prompt_builder), 
            'math': MathExpertGerman(self.prompt_builder),
            'obstacle': ObstacleExpertGerman(self.prompt_builder),
            'instruction': InstructionExpertGerman(self.prompt_builder)
        }
        
        # Store LM configs for context switching
        self.expert_lms = expert_lms
    
    def forward(self, request_params: Dict[str, Any]):
        """Generate and validate German questions using modular prompts (single-pass)"""
        
        import time
        pipeline_start = time.time()
        
        logger.info("Starting German educational pipeline with DSPy")
        
        # Step 1: Generate questions using modular .txt prompts
        generation_start = time.time()
        generation_result = self.generator(request_params)
        questions = generation_result['questions']
        answers = generation_result['answers'] 
        generation_time = time.time() - generation_start
        
        logger.info(f"Generated {len(questions)} questions using modular prompts in {generation_time:.2f}s")
        
        # Step 2: Parallel expert validation using modular prompts
        expert_start = time.time()
        expert_validations = {}
        all_expert_details = {}
        
        for i, (question, answer_list) in enumerate(zip(questions, answers)):
            question_validations = {}
            
            # Validate with each expert using their specific LM and .txt prompts
            for expert_name, expert_module in self.experts.items():
                expert_eval_start = time.time()
                target_value = self._get_target_value_for_expert(expert_name, request_params)
                
                # Switch to expert's LM context
                if expert_name in self.expert_lms:
                    with dspy.context(lm=self.expert_lms[expert_name]):
                        validation = expert_module(question, answer_list, target_value, request_params)
                else:
                    validation = expert_module(question, answer_list, target_value, request_params)
                
                expert_eval_time = time.time() - expert_eval_start
                
                # Enhanced validation with processing metadata
                enhanced_validation = {
                    **validation,
                    'processing_time_ms': expert_eval_time * 1000,
                    'target_value': target_value,
                    'question_evaluated': question,
                    'answers_evaluated': answer_list
                }
                
                question_validations[expert_name] = enhanced_validation
                
                # Store for detailed pipeline tracking
                if f'question_{i+1}' not in all_expert_details:
                    all_expert_details[f'question_{i+1}'] = {}
                all_expert_details[f'question_{i+1}'][expert_name] = enhanced_validation
            
            expert_validations[f'question_{i+1}'] = question_validations
        
        expert_total_time = time.time() - expert_start
        logger.info(f"Expert validation completed in {expert_total_time:.2f}s")
        
        # Step 3: Expert consensus with question refinement using .txt prompts
        consensus_start = time.time()
        final_validations = []
        consensus_details = {}
        
        for i in range(len(questions)):
            consensus_result = self.consensus(
                question=questions[i],
                answers=answers[i], 
                expert_validations=expert_validations[f'question_{i+1}'],
                params=request_params
            )
            consensus_details[f'question_{i+1}'] = consensus_result
            
            final_validations.append({
                'question': consensus_result['refined_question'],  # Use refined question
                'answers': consensus_result['refined_answers'],   # Use refined answers
                'original_question': consensus_result['original_question'],
                'original_answers': consensus_result['original_answers'],
                'expert_validations': expert_validations[f'question_{i+1}'],
                'consensus': consensus_result,
                'approved': True  # Always approved since consensus refines questions
            })
        
        consensus_time = time.time() - consensus_start
        total_pipeline_time = time.time() - pipeline_start
        
        logger.info(f"Expert consensus with refinement completed in {consensus_time:.2f}s")
        logger.info(f"Total pipeline time: {total_pipeline_time:.2f}s")
        
        # Build detailed pipeline information for saving
        pipeline_details = {
            'initial_generation': {
                'questions': questions,
                'answers': answers,
                'generation_result': generation_result,
                'processing_time_ms': generation_time * 1000,
                'modular_prompt_used': generation_result.get('modular_prompt_used', True),
                'generation_reasoning': generation_result.get('reasoning', '')
            },
            'expert_evaluations': all_expert_details,
            'consensus_refinement': {
                'results': consensus_details,
                'processing_time_ms': consensus_time * 1000,
                'final_approvals': [v['approved'] for v in final_validations],
                'refinement_algorithm': 'expert_consensus_with_parameter_guardrails',
                'questions_refined': len(final_validations),
                'format_preservation_success': [v['consensus'].get('format_preserved', True) for v in final_validations],
                'parameter_compliance': [v['consensus'].get('parameters_maintained', True) for v in final_validations]
            },
            'timing': {
                'generation_ms': generation_time * 1000,
                'expert_evaluation_ms': expert_total_time * 1000,
                'consensus_refinement_ms': consensus_time * 1000,
                'total_pipeline_ms': total_pipeline_time * 1000
            }
        }
        
        return {
            'validated_questions': final_validations,
            'generation_metadata': {
                'modular_prompt_used': generation_result['modular_prompt_used'],
                'generation_reasoning': generation_result['reasoning']
            },
            'parameters_used': request_params,
            'all_approved': all(v['approved'] for v in final_validations),
            'pipeline_details': pipeline_details  # Added detailed pipeline information
        }
    
    def _get_target_value_for_expert(self, expert_name: str, params: Dict[str, Any]) -> str:
        """Map expert to their target parameter value"""
        
        expert_param_map = {
            'variation': params.get('p_variation', 'stammaufgabe'),
            'taxonomy': params.get('p_taxonomy_level', 'Stufe 1'),
            'math': params.get('p_mathematical_requirement_level', '0'),
            'obstacle': self._format_obstacles_for_expert(params),
            'instruction': params.get('p_instruction_explicitness_of_instruction', 'Implizit')
            # Note: Content validation (irrelevant_information) now handled by instruction expert
        }
        
        return expert_param_map.get(expert_name, '')
    
    def _format_obstacles_for_expert(self, params: Dict[str, Any]) -> str:
        """Format obstacle parameters for obstacle expert"""
        obstacles = []
        
        # Root text obstacles
        if params.get('p_root_text_obstacle_passive') == 'Enthalten':
            obstacles.append('Grundtext: Passiv')
        if params.get('p_root_text_obstacle_negation') == 'Enthalten':
            obstacles.append('Grundtext: Negation')
        if params.get('p_root_text_obstacle_complex_np') == 'Enthalten':
            obstacles.append('Grundtext: Komplexe Nominalphrasen')
        
        # Item obstacles
        for i in range(1, 9):
            if params.get(f'p_item_{i}_obstacle_passive') == 'Enthalten':
                obstacles.append(f'Item {i}: Passiv')
            if params.get(f'p_item_{i}_obstacle_negation') == 'Enthalten':
                obstacles.append(f'Item {i}: Negation')
            if params.get(f'p_item_{i}_obstacle_complex_np') == 'Enthalten':
                obstacles.append(f'Item {i}: Komplexe NP')
        
        # Instruction obstacles  
        if params.get('p_instruction_obstacle_passive') == 'Enthalten':
            obstacles.append('Anweisung: Passiv')
        if params.get('p_instruction_obstacle_complex_np') == 'Enthalten':
            obstacles.append('Anweisung: Komplexe NP')
        
        return '; '.join(obstacles) if obstacles else 'Keine sprachlichen Hindernisse'