#!/usr/bin/env python3
"""
Modular Prompt Builder - Handles modular prompt construction from parameters
Separated from orchestrator for better debugging and modularity
"""
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

class ExpertPromptEnhancer:
    """Enhances expert prompts by appending relevant parameter txt files"""
    
    def __init__(self, base_path: str = "/home/mklemmingen/PycharmProjects/PythonProject/ALEE_Agent"):
        self.base_path = Path(base_path)
        self.prompt_builder = ModularPromptBuilder(base_path)
        
        # Expert-to-parameter mapping for relevant txt files
        self.expert_parameter_mappings = {
            "variation_expert": {
                "difficulty_prompts": ["mainGen/variationPrompts/stammaufgabe.txt", 
                                     "mainGen/variationPrompts/schwer.txt", 
                                     "mainGen/variationPrompts/leicht.txt"],
                "taxonomy_prompts": ["mainGen/taxonomyLevelPrompt/stufe1WissenReproduktion.txt",
                                   "mainGen/taxonomyLevelPrompt/stufe2AnwendungTransfer.txt"],
                "question_type_prompts": ["mainGen/variationPrompts/multiple-choice.txt",
                                        "mainGen/variationPrompts/single-choice.txt",
                                        "mainGen/variationPrompts/true-false.txt",
                                        "mainGen/variationPrompts/mapping.txt"]
            },
            "math_expert": {
                "math_level_prompts": ["mainGen/mathematicalRequirementLevel/0KeinBezug.txt",
                                     "mainGen/mathematicalRequirementLevel/1NutzenMathematischerDarstellungen.txt",
                                     "mainGen/mathematicalRequirementLevel/2MathematischeOperationen.txt"]
            },
            "obstacle_expert": {
                "root_obstacle_prompts": ["mainGen/rootTextParameterTextPrompts/obstaclePassivePrompts/enthalten.txt",
                                        "mainGen/rootTextParameterTextPrompts/obstaclePassivePrompts/nichtEnthalten.txt",
                                        "mainGen/rootTextParameterTextPrompts/obstacleNegationPrompts/enthalten.txt",
                                        "mainGen/rootTextParameterTextPrompts/obstacleNegationPrompts/nichtEnthalten.txt",
                                        "mainGen/rootTextParameterTextPrompts/obstacleComplexPrompts/enthalten.txt",
                                        "mainGen/rootTextParameterTextPrompts/obstacleComplexPrompts/nichtEnthalten.txt"],
                "item_obstacle_prompts": ["mainGen/itemXObstacle/passive/enthalten.txt",
                                        "mainGen/itemXObstacle/passive/nichtEnthalten.txt",
                                        "mainGen/itemXObstacle/negation/enthalten.txt", 
                                        "mainGen/itemXObstacle/negation/nichtEnthalten.txt",
                                        "mainGen/itemXObstacle/complex/enthalten.txt",
                                        "mainGen/itemXObstacle/complex/nichtEnthalten.txt"],
                "instruction_obstacle_prompts": ["mainGen/instructionObstacle/passive/enthalten.txt",
                                               "mainGen/instructionObstacle/passive/nichtEnthalten.txt",
                                               "mainGen/instructionObstacle/negation/enthalten.txt",
                                               "mainGen/instructionObstacle/negation/nichtEnthalten.txt",
                                               "mainGen/instructionObstacle/complex_np/enthalten.txt",
                                               "mainGen/instructionObstacle/complex_np/nichtEnthalten.txt"]
            },
            "taxonomy_expert": {
                "taxonomy_prompts": ["mainGen/taxonomyLevelPrompt/stufe1WissenReproduktion.txt",
                                   "mainGen/taxonomyLevelPrompt/stufe2AnwendungTransfer.txt"],
                "question_type_prompts": ["mainGen/variationPrompts/multiple-choice.txt",
                                        "mainGen/variationPrompts/single-choice.txt", 
                                        "mainGen/variationPrompts/true-false.txt",
                                        "mainGen/variationPrompts/mapping.txt"]
            },
            "instruction_expert": {
                "explicitness_prompts": ["mainGen/instructionExplicitnessOfInstruction/explizit.txt",
                                       "mainGen/instructionExplicitnessOfInstruction/implizit.txt"],
                "question_type_prompts": ["mainGen/variationPrompts/multiple-choice.txt",
                                        "mainGen/variationPrompts/single-choice.txt",
                                        "mainGen/variationPrompts/true-false.txt", 
                                        "mainGen/variationPrompts/mapping.txt"]
            },
            "content_expert": {
                "all_difficulty_prompts": ["mainGen/variationPrompts/stammaufgabe.txt",
                                         "mainGen/variationPrompts/schwer.txt",
                                         "mainGen/variationPrompts/leicht.txt"],
                "all_taxonomy_prompts": ["mainGen/taxonomyLevelPrompt/stufe1WissenReproduktion.txt",
                                       "mainGen/taxonomyLevelPrompt/stufe2AnwendungTransfer.txt"],
                "question_type_prompts": ["mainGen/variationPrompts/multiple-choice.txt",
                                        "mainGen/variationPrompts/single-choice.txt",
                                        "mainGen/variationPrompts/true-false.txt",
                                        "mainGen/variationPrompts/mapping.txt"],
                "irrelevant_info_prompts": ["mainGen/rootTextParameterTextPrompts/containsIrrelevantInformationPrompt/enthalten.txt",
                                          "mainGen/rootTextParameterTextPrompts/containsIrrelevantInformationPrompt/nichtEnthalten.txt"]
            }
        }
    
    def build_enhanced_expert_prompt(self, expert_name: str, request_params: dict) -> str:
        """Build enhanced expert prompt with relevant parameter knowledge appended"""
        # Load base expert prompt
        base_expert_prompt = self.prompt_builder.load_prompt_txt(f"expertEval/expertPrompts/{expert_name}.txt")
        
        # Get relevant parameter knowledge for this expert
        parameter_knowledge = self._get_relevant_parameter_knowledge(expert_name, request_params)
        
        # Add question type preservation instructions
        question_type_preservation = self._get_question_type_preservation_instructions(request_params.get('question_type', 'multiple-choice'))
        
        # Build enhanced prompt
        enhanced_prompt = f"""{base_expert_prompt}

        === RELEVANTE PARAMETER-WISSENSBASIS ===
        {parameter_knowledge}
        
        === FRAGEFORMAT-BEWAHRUNG (KRITISCH) ===
        {question_type_preservation}
        
        === ZUSÄTZLICHE GUARD RAILS ===
        - NIEMALS den Fragetyp ändern ({request_params.get('question_type', 'multiple-choice')})
        - Nur Verbesserungen innerhalb des bestehenden Formats
        - Behalte die strukturelle Integrität der Frage bei
        - Verbessere nur Inhalt, Klarheit und Genauigkeit
        - Achte auf die spezifischen Parameter-Anforderungen dieses Experten-Bereichs"""

        return enhanced_prompt
    
    def _get_relevant_parameter_knowledge(self, expert_name: str, request_params: dict) -> str:
        """Extract and combine relevant parameter txt file contents for expert"""
        if expert_name not in self.expert_parameter_mappings:
            return f"<Keine spezifischen Parameter für {expert_name}>"
        
        knowledge_sections = []
        mappings = self.expert_parameter_mappings[expert_name]
        
        for category, file_paths in mappings.items():
            category_knowledge = []
            for file_path in file_paths:
                content = self.prompt_builder.load_prompt_txt(file_path)
                if content and not content.startswith('<'):  # Skip missing files
                    file_name = file_path.split('/')[-1].replace('.txt', '')
                    category_knowledge.append(f"**{file_name.upper()}**: {content}")
            
            if category_knowledge:
                knowledge_sections.append(f"### {category.upper().replace('_', ' ')}\n" + "\n\n".join(category_knowledge))
        
        return "\n\n".join(knowledge_sections) if knowledge_sections else f"<Keine verfügbaren Parameter für {expert_name}>"
    
    def _get_question_type_preservation_instructions(self, question_type: str) -> str:
        """Get specific instructions for preserving question type format"""
        type_instructions = {
            "multiple-choice": """
            MULTIPLE-CHOICE FORMAT BEWAHREN:
            - Behalte die <option> Tags für alle Antwortmöglichkeiten bei
            - Stelle sicher, dass MEHRERE richtige Antworten möglich sind
            - Verwende Formulierungen wie "Wähle alle zutreffenden Aussagen aus"
            - Mindestens 2, maximal 4 richtige Antworten aus den Optionen""",

                        "single-choice": """
            SINGLE-CHOICE FORMAT BEWAHREN:
            - Behalte die <option> Tags für alle Antwortmöglichkeiten bei
            - GENAU EINE richtige Antwort unter allen Optionen
            - Verwende Formulierungen wie "Wähle die richtige Antwort aus"
            - Alle anderen Optionen müssen eindeutig falsch sein""",

                        "true-false": """
            TRUE-FALSE FORMAT BEWAHREN:
            - Behalte die <true-false> Tags für alle Aussagen bei
            - Jede Aussage einzeln bewertbar als richtig oder falsch
            - Verwende "Entscheide, ob die Aussagen richtig oder falsch sind"
            - Keine Option-Tags verwenden""",

                        "mapping": """
            MAPPING FORMAT BEWAHREN:
            - Behalte die Zuordnungsstruktur bei
            - Links stehen Begriffe/Konzepte, rechts stehen Definitionen/Erklärungen
            - Verwende "Ordne zu" oder ähnliche Formulierungen
            - Eindeutige 1:1 Zuordnungen"""
                    }
        
        return type_instructions.get(question_type.lower(), 
                                   f"UNBEKANNTER FRAGETYP ({question_type}) - Behalte das bestehende Format exakt bei")

class ModularPromptBuilder:
    """Builds modular prompts based on parameter values using external txt files"""
    
    def __init__(self, base_path: str = "/home/mklemmingen/PycharmProjects/PythonProject/ALEE_Agent"):
        self.base_path = Path(base_path)
        
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
        """Build variation-specific prompt (difficulty level)"""
        if not variation or variation.strip() == "":
            logger.error("Variation parameter cannot be empty")
            return "<Missing variation parameter>"
            
        valid_variations = ["stammaufgabe", "schwer", "leicht"]
        if variation.lower() not in valid_variations:
            logger.error(f"Invalid variation '{variation}'. Must be one of: {', '.join(valid_variations)}")
            return f"<Invalid variation: {variation}>"
            
        # Map variation to difficulty level prompt files (stakeholder-aligned)
        variation_map = {
            "stammaufgabe": "mainGen/variationPrompts/stammaufgabe.txt",  # Standard difficulty
            "schwer": "mainGen/variationPrompts/schwer.txt",              # Hard difficulty  
            "leicht": "mainGen/variationPrompts/leicht.txt"               # Easy difficulty
        }
        file_path = variation_map.get(variation.lower(), "mainGen/variationPrompts/stammaufgabe.txt")
        content = self.load_prompt_txt(file_path)
        return f"SCHWIERIGKEITSSTUFE ({variation}): {content}"
    
    def build_taxonomy_prompt(self, level: str) -> str:
        """Build taxonomy-specific prompt (cognitive level)"""
        if "Stufe 1" in level:
            content = self.load_prompt_txt("mainGen/taxonomyLevelPrompt/stufe1WissenReproduktion.txt")
            return f"p.taxonomy_level (Stufe 1): {content}"
        elif "Stufe 2" in level:
            content = self.load_prompt_txt("mainGen/taxonomyLevelPrompt/stufe2AnwendungTransfer.txt")
            return f"p.taxonomy_level (Stufe 2): {content}"
        else:
            content = self.load_prompt_txt("mainGen/taxonomyLevelPrompt/stufe1WissenReproduktion.txt")
            return f"p.taxonomy_level (Default Stufe 1): {content}"
    
    def build_mathematical_prompt(self, level: str) -> str:
        """Build mathematical requirement prompt"""
        level_clean = level.strip().split()[0]  # Extract just the number
        level_map = {
            "0": "mainGen/mathematicalRequirementLevel/0KeinBezug.txt",
            "1": "mainGen/mathematicalRequirementLevel/1NutzenMathematischerDarstellungen.txt", 
            "2": "mainGen/mathematicalRequirementLevel/2MathematischeOperationen.txt"
        }
        file_path = level_map.get(level_clean, "mainGen/mathematicalRequirementLevel/0KeinBezug.txt")
        content = self.load_prompt_txt(file_path)
        return f"p.mathematical_requirement_level ({level}): {content}"
    
    def build_obstacle_prompt(self, obstacle_type: str, value: str, parameter_name: str) -> str:
        """Build obstacle-specific prompts"""
        value_file = "enthalten.txt" if value == "Enthalten" else "nichtEnthalten.txt"
        
        obstacle_map = {
            "root_passive": f"mainGen/rootTextParameterTextPrompts/obstaclePassivePrompts/{value_file}",
            "root_negation": f"mainGen/rootTextParameterTextPrompts/obstacleNegationPrompts/{value_file}", 
            "root_complex_np": f"mainGen/rootTextParameterTextPrompts/obstacleComplexPrompts/{value_file}",
            "item_passive": f"mainGen/itemXObstacle/passive/{value_file}",
            "item_negation": f"mainGen/itemXObstacle/negation/{value_file}",
            "item_complex": f"mainGen/itemXObstacle/complex/{value_file}",
            "instruction_passive": f"mainGen/instructionObstacle/passive/{value_file}",
            "instruction_negation": f"mainGen/instructionObstacle/negation/{value_file}",
            "instruction_complex_np": f"mainGen/instructionObstacle/complex_np/{value_file}"
        }
        
        file_path = obstacle_map.get(obstacle_type, "")
        if file_path:
            content = self.load_prompt_txt(file_path)
            return f"{parameter_name} ({value}): {content}"
        return ""
    
    def build_irrelevant_info_prompt(self, value: str) -> str:
        """Build irrelevant information prompt"""
        value_file = "enthalten.txt" if value == "Enthalten" else "nichtEnthalten.txt"
        content = self.load_prompt_txt(f"mainGen/rootTextParameterTextPrompts/containsIrrelevantInformationPrompt/{value_file}")
        return f"p.root_text_contains_irrelevant_information ({value}): {content}"
    
    def build_explicitness_prompt(self, value: str) -> str:
        """Build instruction explicitness prompt"""
        if "Explizit" in value:
            content = self.load_prompt_txt("mainGen/instructionExplicitnessOfInstruction/explizit.txt")
            return f"p.instruction_explicitness_of_instruction (Explizit): {content}"
        else:
            content = self.load_prompt_txt("mainGen/instructionExplicitnessOfInstruction/implizit.txt")
            return f"p.instruction_explicitness_of_instruction (Implizit): {content}"
    
    def build_question_type_prompt(self, question_type: str) -> str:
        """Build question type specific prompt"""
        if not question_type or question_type.strip() == "":
            logger.error("Question type parameter cannot be empty")
            return "<Missing question_type parameter>"
            
        valid_types = ["multiple-choice", "single-choice", "true-false", "mapping"]
        if question_type.lower() not in valid_types:
            logger.error(f"Invalid question type '{question_type}'. Must be one of: {', '.join(valid_types)}")
            return f"<Invalid question_type: {question_type}>"
            
        # Map question type to format prompt files
        type_map = {
            "multiple-choice": "mainGen/variationPrompts/multiple-choice.txt",
            "single-choice": "mainGen/variationPrompts/single-choice.txt",
            "true-false": "mainGen/variationPrompts/true-false.txt",
            "mapping": "mainGen/variationPrompts/mapping.txt"  # If exists
        }
        file_path = type_map.get(question_type.lower(), "mainGen/variationPrompts/multiple-choice.txt")
        content = self.load_prompt_txt(file_path)
        return f"FRAGEFORMAT ({question_type}): {content}"
    
    def build_master_prompt(self, request: Any) -> str:
        """Build the complete modular master prompt from request parameters"""
        prompt_sections = []
        
        # Add educational text
        prompt_sections.append(f"=== BILDUNGSTEXT ===\\n{request.text}")
        
        # Add question format (question_type parameter)
        if hasattr(request, 'question_type') and request.question_type:
            prompt_sections.append(self.build_question_type_prompt(request.question_type))
        
        # Add difficulty level (p_variation parameter)  
        prompt_sections.append(self.build_variation_prompt(request.p_variation))
        
        # Add cognitive taxonomy level
        prompt_sections.append(self.build_taxonomy_prompt(request.p_taxonomy_level))
        
        # Add mathematical requirements
        prompt_sections.append(self.build_mathematical_prompt(request.p_mathematical_requirement_level))
        
        # Add root text obstacles
        if request.p_root_text_obstacle_passive != "Nicht Enthalten":
            prompt_sections.append(self.build_obstacle_prompt("root_passive", request.p_root_text_obstacle_passive, "p.root_text_obstacle_passive"))
        
        if request.p_root_text_obstacle_negation != "Nicht Enthalten":
            prompt_sections.append(self.build_obstacle_prompt("root_negation", request.p_root_text_obstacle_negation, "p.root_text_obstacle_negation"))
            
        if request.p_root_text_obstacle_complex_np != "Nicht Enthalten":
            prompt_sections.append(self.build_obstacle_prompt("root_complex_np", request.p_root_text_obstacle_complex_np, "p.root_text_obstacle_complex_np"))
        
        # Add irrelevant information
        if request.p_root_text_contains_irrelevant_information != "Nicht Enthalten":
            prompt_sections.append(self.build_irrelevant_info_prompt(request.p_root_text_contains_irrelevant_information))
        
        # Add item obstacles for individual items (1-8)
        for i in range(1, 9):
            passive_attr = f"p_item_{i}_obstacle_passive"
            negation_attr = f"p_item_{i}_obstacle_negation"
            complex_attr = f"p_item_{i}_obstacle_complex_np"
            
            if hasattr(request, passive_attr):
                value = getattr(request, passive_attr)
                if value != "Nicht Enthalten":
                    prompt_sections.append(self.build_obstacle_prompt("item_passive", value, f"p.item_{i}_obstacle_passive"))
            
            if hasattr(request, negation_attr):
                value = getattr(request, negation_attr)
                if value != "Nicht Enthalten":
                    prompt_sections.append(self.build_obstacle_prompt("item_negation", value, f"p.item_{i}_obstacle_negation"))
                    
            if hasattr(request, complex_attr):
                value = getattr(request, complex_attr)
                if value != "Nicht Enthalten":
                    prompt_sections.append(self.build_obstacle_prompt("item_complex", value, f"p.item_{i}_obstacle_complex_np"))
        
        # Add instruction obstacles
        if request.p_instruction_obstacle_passive != "Nicht Enthalten":
            prompt_sections.append(self.build_obstacle_prompt("instruction_passive", request.p_instruction_obstacle_passive, "p.instruction_obstacle_passive"))
            
        if request.p_instruction_obstacle_complex_np != "Nicht Enthalten":
            prompt_sections.append(self.build_obstacle_prompt("instruction_complex_np", request.p_instruction_obstacle_complex_np, "p.instruction_obstacle_complex_np"))
        
        # Add instruction explicitness
        prompt_sections.append(self.build_explicitness_prompt(request.p_instruction_explicitness_of_instruction))
        
        # Combine all sections
        master_prompt = "\\n\\n".join([section for section in prompt_sections if section.strip()])
        
        logger.info(f"Built master prompt with {len(prompt_sections)} sections")
        return master_prompt

# CLI interface for standalone testing
def main():
    """Test the prompt builder standalone"""
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python prompt_builder.py <request_json_file>")
        print("Example: python prompt_builder.py test_request.json")
        sys.exit(1)
    
    request_file = sys.argv[1]
    
    try:
        with open(request_file, 'r', encoding='utf-8') as f:
            request_data = json.load(f)
        
        # Create a simple object from the dict
        class RequestObj:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        request = RequestObj(request_data)
        
        builder = ModularPromptBuilder()
        master_prompt = builder.build_master_prompt(request)
        
        print("=== MODULAR MASTER PROMPT ===")
        print(master_prompt)
        print(f"\\n=== STATISTICS ===")
        print(f"Total length: {len(master_prompt)} characters")
        print(f"Lines: {len(master_prompt.split(chr(10)))}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()