"""
German Educational DSPy Modules - Integrates with existing modular .txt prompt system
NO hardcoded prompt strings - uses ModularPromptBuilder for all prompts
"""

import logging
from typing import Dict, Any, Optional

from .educational_signatures import *
from .prompt_builder import ModularPromptBuilder

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
    
    def forward(self, request_params: Dict[str, Any]):
        """Generate 3 German questions using modular prompts from .txt files"""
        
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
        
        return {
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
    
    def _extract_instruction_obstacles(self, params: Dict[str, Any]) -> str:
        """Extract instruction obstacle parameters"""
        obstacles = []
        if params.get('p_instruction_obstacle_passive') == 'Enthalten':
            obstacles.append('passive')
        if params.get('p_instruction_obstacle_complex_np') == 'Enthalten':
            obstacles.append('complex_np')
        return ', '.join(obstacles) if obstacles else 'keine'


class GermanExpertValidator(dspy.Module):
    """Base class for German expert validation using .txt prompts"""
    
    def __init__(self, signature_class, expert_name: str, prompt_builder: Optional[ModularPromptBuilder] = None):
        super().__init__()
        self.expert_name = expert_name
        self.validate = dspy.ChainOfThought(signature_class)
        self.prompt_builder = prompt_builder or ModularPromptBuilder()
        
        # Load expert-specific prompt from .txt file
        self.expert_prompt = self.prompt_builder.load_prompt_txt(f"expertEval/expertPrompts/{expert_name}.txt")
        self.evaluation_instruction = self.prompt_builder.load_prompt_txt("expertEval/expertEvaluationInstruction.txt")
    
    def get_expert_context(self, question: str, answers: List[str], target_value: str, params: Dict[str, Any]) -> str:
        """Build expert context using modular prompts"""
        
        context_sections = [
            f"=== EXPERTENROLLE ===",
            self.expert_prompt,
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
    
    def forward(self, question: str, answers: List[str], target_variation: str, params: Dict[str, Any]):
        """Validate German difficulty using modular prompts"""
        
        # Build expert context from .txt files
        expert_context = self.get_expert_context(question, answers, target_variation, params)
        
        result = self.validate(
            frage=question,
            antworten=", ".join(answers),
            ziel_variation=target_variation
        )
        
        return {
            'expert': self.expert_name,
            'rating': result.bewertung,
            'feedback': result.feedback,
            'suggestions': result.vorschlaege.split('\n') if result.vorschlaege else [],
            'reasoning': result.begruendung,
            'expert_context': expert_context
        }


class TaxonomyExpertGerman(GermanExpertValidator):
    """German taxonomy expert using .txt prompts"""
    
    def __init__(self, prompt_builder: Optional[ModularPromptBuilder] = None):
        super().__init__(ValidateTaxonomyGerman, "taxonomy_expert", prompt_builder)
    
    def forward(self, question: str, answers: List[str], target_taxonomy: str, params: Dict[str, Any]):
        """Validate taxonomy using modular prompts"""
        
        expert_context = self.get_expert_context(question, answers, target_taxonomy, params)
        
        result = self.validate(
            frage=question,
            antworten=", ".join(answers),
            ziel_taxonomie=target_taxonomy
        )
        
        return {
            'expert': self.expert_name,
            'rating': result.bewertung,
            'feedback': result.feedback,
            'suggestions': result.vorschlaege.split('\n') if result.vorschlaege else [],
            'reasoning': result.begruendung,
            'expert_context': expert_context
        }


class MathExpertGerman(GermanExpertValidator):
    """German mathematical complexity expert using .txt prompts"""
    
    def __init__(self, prompt_builder: Optional[ModularPromptBuilder] = None):
        super().__init__(ValidateMathematicalGerman, "math_expert", prompt_builder)
    
    def forward(self, question: str, answers: List[str], target_math_level: str, params: Dict[str, Any]):
        """Validate mathematical complexity using modular prompts"""
        
        expert_context = self.get_expert_context(question, answers, target_math_level, params)
        
        result = self.validate(
            frage=question,
            antworten=", ".join(answers),
            ziel_math_stufe=target_math_level
        )
        
        return {
            'expert': self.expert_name,
            'rating': result.bewertung,
            'feedback': result.feedback,
            'suggestions': result.vorschlaege.split('\n') if result.vorschlaege else [],
            'reasoning': result.begruendung,
            'expert_context': expert_context
        }


class ObstacleExpertGerman(GermanExpertValidator):
    """German linguistic obstacle expert using .txt prompts"""
    
    def __init__(self, prompt_builder: Optional[ModularPromptBuilder] = None):
        super().__init__(ValidateObstacleGerman, "obstacle_expert", prompt_builder)
    
    def forward(self, question: str, answers: List[str], target_obstacles: str, params: Dict[str, Any]):
        """Validate linguistic obstacles using modular prompts"""
        
        expert_context = self.get_expert_context(question, answers, target_obstacles, params)
        
        result = self.validate(
            frage=question,
            antworten=", ".join(answers),
            ziel_hindernisse=target_obstacles
        )
        
        return {
            'expert': self.expert_name,
            'rating': result.bewertung,
            'feedback': result.feedback,
            'suggestions': result.vorschlaege.split('\n') if result.vorschlaege else [],
            'reasoning': result.begruendung,
            'expert_context': expert_context
        }


class InstructionExpertGerman(GermanExpertValidator):
    """German instruction clarity expert using .txt prompts"""
    
    def __init__(self, prompt_builder: Optional[ModularPromptBuilder] = None):
        super().__init__(ValidateInstructionGerman, "instruction_expert", prompt_builder)
    
    def forward(self, question: str, answers: List[str], target_explicitness: str, params: Dict[str, Any]):
        """Validate instruction clarity using modular prompts"""
        
        expert_context = self.get_expert_context(question, answers, target_explicitness, params)
        
        result = self.validate(
            frage=question,
            antworten=", ".join(answers),
            ziel_explizitheit=target_explicitness
        )
        
        return {
            'expert': self.expert_name,
            'rating': result.bewertung,
            'feedback': result.feedback,
            'suggestions': result.vorschlaege.split('\n') if result.vorschlaege else [],
            'reasoning': result.begruendung,
            'expert_context': expert_context
        }


class ContentExpertGerman(GermanExpertValidator):
    """German content relevance expert using .txt prompts"""
    
    def __init__(self, prompt_builder: Optional[ModularPromptBuilder] = None):
        super().__init__(ValidateContentGerman, "content_expert", prompt_builder)
    
    def forward(self, question: str, answers: List[str], target_relevance: str, params: Dict[str, Any]):
        """Validate content relevance using modular prompts"""
        
        expert_context = self.get_expert_context(question, answers, target_relevance, params)
        
        result = self.validate(
            frage=question,
            antworten=", ".join(answers),
            ziel_relevanz=target_relevance
        )
        
        return {
            'expert': self.expert_name,
            'rating': result.bewertung,
            'feedback': result.feedback,
            'suggestions': result.vorschlaege.split('\n') if result.vorschlaege else [],
            'reasoning': result.begruendung,
            'expert_context': expert_context
        }


class GermanExpertConsensus(dspy.Module):
    """Consensus module for German expert validation using .txt prompts"""
    
    def __init__(self, prompt_builder: Optional[ModularPromptBuilder] = None):
        super().__init__()
        self.consensus = dspy.ChainOfThought(ExpertConsensusGerman)
        self.prompt_builder = prompt_builder or ModularPromptBuilder()
        
        # Load consensus instructions from .txt file
        self.consensus_instruction = self.prompt_builder.load_prompt_txt("expertEval/questionImprovementInstruction.txt")
    
    def forward(self, expert_validations: Dict[str, Dict[str, Any]]):
        """Determine consensus from all German expert validations"""
        
        # Aggregate expert data
        ratings = [v['rating'] for v in expert_validations.values()]
        feedback_parts = [f"{name}: {v['feedback']}" for name, v in expert_validations.items()]
        suggestion_parts = []
        
        for name, validation in expert_validations.items():
            if validation['suggestions'] and validation['rating'] < 3:
                suggestion_parts.append(f"{name}: {'; '.join(validation['suggestions'])}")
        
        # Format data for consensus module
        ratings_str = f"Bewertungen: {ratings} (Durchschnitt: {sum(ratings)/len(ratings):.1f})"
        feedback_str = "\n".join(feedback_parts)
        suggestions_str = "\n".join(suggestion_parts) if suggestion_parts else "Keine Verbesserungsvorschläge"
        
        result = self.consensus(
            experten_bewertungen=ratings_str,
            experten_feedback=feedback_str,
            experten_vorschlaege=suggestions_str
        )
        
        return {
            'approved': result.genehmigt.lower() == 'ja',
            'improvement_priority': result.verbesserungspriorität,
            'consensus_reasoning': result.konsens_begründung,
            'synthesized_suggestions': result.zusammengefasste_vorschlaege,
            'average_rating': sum(ratings) / len(ratings),
            'expert_count': len(expert_validations),
            'consensus_instruction_used': self.consensus_instruction
        }


class GermanEducationalPipeline(dspy.Module):
    """Complete German educational pipeline using modular .txt prompts"""
    
    def __init__(self, expert_lms: Dict[str, Any], prompt_builder: Optional[ModularPromptBuilder] = None):
        super().__init__()
        
        self.prompt_builder = prompt_builder or ModularPromptBuilder()
        self.generator = GermanQuestionGenerator(self.prompt_builder)
        self.consensus = GermanExpertConsensus(self.prompt_builder)
        
        # Initialize experts with their specific LMs and prompt builder
        self.experts = {
            'variation': VariationExpertGerman(self.prompt_builder),
            'taxonomy': TaxonomyExpertGerman(self.prompt_builder), 
            'math': MathExpertGerman(self.prompt_builder),
            'obstacle': ObstacleExpertGerman(self.prompt_builder),
            'instruction': InstructionExpertGerman(self.prompt_builder),
            'content': ContentExpertGerman(self.prompt_builder)
        }
        
        # Store LM configs for context switching
        self.expert_lms = expert_lms
    
    def forward(self, request_params: Dict[str, Any]):
        """Generate and validate German questions using modular prompts (single-pass)"""
        
        logger.info("Starting German educational pipeline with DSPy")
        
        # Step 1: Generate questions using modular .txt prompts
        generation_result = self.generator(request_params)
        questions = generation_result['questions']
        answers = generation_result['answers'] 
        
        logger.info(f"Generated {len(questions)} questions using modular prompts")
        
        # Step 2: Parallel expert validation using modular prompts
        expert_validations = {}
        
        for i, (question, answer_list) in enumerate(zip(questions, answers)):
            question_validations = {}
            
            # Validate with each expert using their specific LM and .txt prompts
            for expert_name, expert_module in self.experts.items():
                target_value = self._get_target_value_for_expert(expert_name, request_params)
                
                # Switch to expert's LM context
                if expert_name in self.expert_lms:
                    with dspy.context(lm=self.expert_lms[expert_name]):
                        validation = expert_module(question, answer_list, target_value, request_params)
                else:
                    validation = expert_module(question, answer_list, target_value, request_params)
                
                question_validations[expert_name] = validation
            
            expert_validations[f'question_{i+1}'] = question_validations
        
        # Step 3: Consensus determination using .txt prompts
        final_validations = []
        for i in range(len(questions)):
            consensus_result = self.consensus(expert_validations[f'question_{i+1}'])
            
            final_validations.append({
                'question': questions[i],
                'answers': answers[i],
                'expert_validations': expert_validations[f'question_{i+1}'],
                'consensus': consensus_result,
                'approved': consensus_result['approved']
            })
        
        return {
            'validated_questions': final_validations,
            'generation_metadata': {
                'modular_prompt_used': generation_result['modular_prompt_used'],
                'generation_reasoning': generation_result['reasoning']
            },
            'parameters_used': request_params,
            'all_approved': all(v['approved'] for v in final_validations)
        }
    
    def _get_target_value_for_expert(self, expert_name: str, params: Dict[str, Any]) -> str:
        """Map expert to their target parameter value"""
        
        expert_param_map = {
            'variation': params.get('p_variation', 'stammaufgabe'),
            'taxonomy': params.get('p_taxonomy_level', 'Stufe 1'),
            'math': params.get('p_mathematical_requirement_level', '0'),
            'obstacle': self._format_obstacles_for_expert(params),
            'instruction': params.get('p_instruction_explicitness_of_instruction', 'Implizit'),
            'content': params.get('p_root_text_contains_irrelevant_information', 'Nicht Enthalten')
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