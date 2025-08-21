"""
Result to Graphical Feedback Visualization Tool
Generates visual representations of the DSPy question generation pipeline
Shows the evolution of questions from initial generation through expert feedback to final output
"""

import json
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Rectangle
import textwrap
import numpy as np

class PipelineVisualizer:
    """Visualizes the DSPy question generation pipeline for each question"""
    
    def __init__(self, results_dir: str = None, overwrite: bool = True):
        """Initialize the visualizer with the results directory
        
        Args:
            results_dir: Directory containing result folders
            overwrite: Whether to overwrite existing visualizations (default: True)
        """
        if results_dir is None:
            # Default to results directory in the same folder as this script
            self.results_dir = Path(__file__).parent
        else:
            self.results_dir = Path(results_dir)
        
        self.overwrite = overwrite
            
        # Color scheme for visualization
        self.colors = {
            'input': '#E8F4FD',      # Light blue for input
            'generation': '#FFF9E6',  # Light yellow for generation
            'expert_good': '#E8F5E9', # Light green for good ratings (4-5)
            'expert_ok': '#FFF3E0',   # Light orange for ok ratings (3)
            'expert_bad': '#FFEBEE',  # Light red for poor ratings (1-2)
            'refinement': '#F3E5F5',  # Light purple for refinement
            'final': '#E1F5FE',       # Light cyan for final output
            'arrow': '#757575',       # Gray for arrows
            'text': '#212121'         # Dark gray for text
        }
        
    def find_result_folders(self):
        """Find all correctly named result folders matching the pattern"""
        folders = []
        pattern = r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_.*'
        
        for item in self.results_dir.iterdir():
            if item.is_dir() and item.name != 'visualizations':
                # Check if it matches the timestamp pattern
                parts = item.name.split('_')
                if len(parts) >= 3:
                    try:
                        # Validate timestamp format
                        timestamp_str = f"{parts[0]}_{parts[1]}"
                        datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
                        folders.append(item)
                    except ValueError:
                        continue
                        
        return sorted(folders)
    
    def load_session_data(self, session_dir: Path):
        """Load all relevant JSON data from a session directory"""
        data = {}
        
        # Load request parameters
        params_file = session_dir / "parameters" / "request_parameters.json"
        if params_file.exists():
            try:
                with open(params_file, 'r', encoding='utf-8') as f:
                    data['parameters'] = json.load(f)
            except json.JSONDecodeError as e:
                print(f"    Warning: Could not parse {params_file.name}: {e}")
        
        # Load DSPy pipeline data
        pipeline_dir = session_dir / "dspy_pipeline"
        if pipeline_dir.exists():
            # Load typed generation data
            for file in pipeline_dir.glob("01_typed_generation_*.json"):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Try to parse the JSON, handle truncated files
                        try:
                            data['generation'] = json.loads(content)
                        except json.JSONDecodeError:
                            # Try to extract the typed_questions array even from truncated JSON
                            if '"typed_questions": [' in content:
                                # Extract the typed_questions portion
                                start = content.find('"typed_questions": [') + len('"typed_questions": ')
                                # Find the end of the array by counting brackets
                                bracket_count = 0
                                end = start
                                in_string = False
                                escape_next = False
                                
                                for i, char in enumerate(content[start:], start):
                                    if escape_next:
                                        escape_next = False
                                        continue
                                    if char == '\\':
                                        escape_next = True
                                        continue
                                    if char == '"' and not escape_next:
                                        in_string = not in_string
                                    if not in_string:
                                        if char == '[':
                                            bracket_count += 1
                                        elif char == ']':
                                            bracket_count -= 1
                                            if bracket_count == 0:
                                                end = i + 1
                                                break
                                
                                if end > start:
                                    try:
                                        typed_questions_json = content[start:end]
                                        typed_questions = json.loads(typed_questions_json)
                                        data['generation'] = {
                                            'step_data': {
                                                'typed_questions': typed_questions
                                            }
                                        }
                                    except:
                                        print(f"    Warning: Could not extract typed_questions from truncated JSON")
                except Exception as e:
                    print(f"    Warning: Error reading {file.name}: {e}")
                break
            
            # Load refinement decisions
            for file in pipeline_dir.glob("03_refinement_decisions_*.json"):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data['refinement'] = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"    Warning: Could not parse {file.name}: {e}")
                break
        
        # Load final questions
        final_file = session_dir / "final_questions.json"
        if final_file.exists():
            try:
                with open(final_file, 'r', encoding='utf-8') as f:
                    data['final'] = json.load(f)
            except json.JSONDecodeError as e:
                print(f"    Warning: Could not parse {final_file.name}: {e}")
                
        return data
    
    def wrap_text(self, text: str, width: int = 50) -> str:
        """Wrap text to specified width"""
        return '\n'.join(textwrap.wrap(text, width=width))
    
    def format_question_text(self, question_text: str) -> str:
        """Format question text with proper line breaks for readability"""
        formatted = question_text.strip()
        
        # Handle multiple choice with <option> tags and <start-option>
        if '<option>' in formatted.lower() or '<start-option>' in formatted.lower():
            # Replace all variations of option tags including <start-option>
            import re
            formatted = re.sub(r'<\s*option\s*>', '\n  A) ', formatted, flags=re.IGNORECASE)
            formatted = re.sub(r'<\s*start-option\s*>', '\n  A) ', formatted, flags=re.IGNORECASE)
            
            # Fix option letters A, B, C, D, E
            lines = formatted.split('\n')
            option_count = 0
            option_letters = ['A) ', 'B) ', 'C) ', 'D) ', 'E) ', 'F) ']
            
            for i, line in enumerate(lines):
                if line.strip().startswith('A) '):
                    if option_count < len(option_letters):
                        lines[i] = line.replace('A) ', option_letters[option_count])
                        option_count += 1
            formatted = '\n'.join(lines)
        
        # Handle true/false questions (various patterns)
        elif any(word in formatted.lower() for word in ['true', 'false', 'richtig', 'falsch', 'wahr', '<true-false>']):
            # Handle <true-false> tags first
            formatted = re.sub(r'<\s*true-false\s*>', '\n  ', formatted, flags=re.IGNORECASE)
            # Handle German true/false
            formatted = re.sub(r'\b(richtig|falsch)\b', r'\n  \1', formatted, flags=re.IGNORECASE)
            # Handle English true/false
            formatted = re.sub(r'\b(true|false)\b', r'\n  \1', formatted, flags=re.IGNORECASE)
            # Handle "wahr/falsch" pattern
            formatted = re.sub(r'\b(wahr)\b', r'\n  \1', formatted, flags=re.IGNORECASE)
            
        # Handle numbered options (1. 2. 3. etc.)
        elif re.search(r'\b[1-9]\.\s', formatted):
            formatted = re.sub(r'\s+([1-9]\.\s)', r'\n  \1', formatted)
            
        # Handle lettered options (a) b) c) etc.)
        elif re.search(r'\b[a-f]\)\s', formatted):
            formatted = re.sub(r'\s+([a-f]\)\s)', r'\n  \1', formatted)
            
        # Handle dash-separated options (- option1 - option2)
        elif formatted.count(' - ') >= 2:
            formatted = formatted.replace(' - ', '\n  - ')
            
        # Handle semicolon-separated options
        elif formatted.count(';') >= 2 and len(formatted.split(';')) <= 6:
            parts = formatted.split(';')
            if len(parts) > 1:
                question_part = parts[0].strip()
                options = [f"  {part.strip()}" for part in parts[1:] if part.strip()]
                formatted = question_part + '\n' + '\n'.join(options)
        
        # Handle questions with "oder" (German "or") patterns
        elif ' oder ' in formatted.lower():
            formatted = re.sub(r'\s+oder\s+', r'\n  oder ', formatted, flags=re.IGNORECASE)
            
        # Handle questions with explicit choice indicators
        elif any(indicator in formatted.lower() for indicator in ['wähle', 'choose', 'select', 'mark']):
            # Look for patterns like "A: text B: text" or "1: text 2: text"
            if re.search(r'[A-F]:\s', formatted):
                formatted = re.sub(r'\s+([A-F]:\s)', r'\n  \1', formatted)
            elif re.search(r'[1-9]:\s', formatted):
                formatted = re.sub(r'\s+([1-9]:\s)', r'\n  \1', formatted)
        
        # Clean up formatting
        while formatted.startswith('\n'):
            formatted = formatted[1:]
        
        # Remove excessive whitespace
        formatted = re.sub(r'\n\s*\n', '\n', formatted)
        formatted = re.sub(r'  +', '  ', formatted)  # Normalize multiple spaces to two
        
        return formatted
    
    def get_expert_color(self, rating: int) -> str:
        """Get color based on expert rating"""
        if rating >= 4:
            return self.colors['expert_good']
        elif rating == 3:
            return self.colors['expert_ok']
        else:
            return self.colors['expert_bad']
    
    def create_question_visualization(self, question_data: dict, question_num: int, 
                                    parameters: dict, final_question: str, 
                                    output_path: Path):
        """Create a visualization for a single question's pipeline"""
        # A4 format - optimized for readability
        fig, ax = plt.subplots(figsize=(8.3, 11.7))  # A4 dimensions in inches
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 14)
        ax.axis('off')
        
        # Title
        plt.title(f"Question {question_num} Generation Pipeline", 
                 fontsize=12, fontweight='bold', pad=0)
        
        # A4 working area - leave margins
        available_height = 12.5  # Total usable height
        current_y = 13.2  # Start right below title
        
        # Calculate content sizes first to determine scaling
        content_sections = self._prepare_content_sections(question_data, parameters, final_question)
        
        # Calculate total content with base spacing
        base_line_spacing = 0.22  # Increased for better readability
        base_layer_spacing = 0.25  # Consistent spacing between sections
        total_lines = sum(len(section['lines']) for section in content_sections)
        total_content_height = (total_lines * base_line_spacing + 
                              len(content_sections) * (base_layer_spacing + 0.5))  # 0.5 for title+box padding
        
        # Dynamic scaling based on content
        if total_content_height > available_height:
            scale_factor = available_height / total_content_height * 0.95  # 5% safety margin
            line_spacing = max(0.12, base_line_spacing * scale_factor)
            font_size_base = max(7, int(9 * scale_factor))
            layer_spacing = max(0.15, base_layer_spacing * scale_factor)
        else:
            line_spacing = base_line_spacing
            font_size_base = 9
            layer_spacing = base_layer_spacing
        
        # Render all sections with dynamic scaling
        for i, section in enumerate(content_sections):
            # Add minimal padding above layer titles (except for the first one)
            if i > 0:
                current_y -= 0.05  # Minimal padding above title
            
            # Calculate actual height with scaled spacing
            actual_height = len(section['lines']) * line_spacing + 0.5  # title + box padding
            self._render_section(ax, section, current_y, line_spacing, font_size_base)
            current_y -= actual_height + layer_spacing

        # Add watermark at bottom center
        ax.text(5.0, 0.3, "Kateryna Lauterbach, 2025, github.com/mklemmingen/ALEE", 
                fontsize=8, ha='center', va='bottom', alpha=0.5, 
                color='gray', style='italic')
        
        # Save the figure with proper margins for A4 printing
        plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05)
        plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.1, 
                   facecolor='white', edgecolor='none')
        plt.close()
    
    def _prepare_content_sections(self, question_data: dict, parameters: dict, final_question: str):
        """Prepare all content sections and calculate their heights"""
        sections = []
        
        # Section 1: Request Parameters
        req_params = parameters['request_parameters']
        params_lines = [
            f"[KEY] C_ID: {req_params.get('c_id', 'N/A')}",
            f"[KEY] Question Type: {req_params.get('question_type', 'N/A')}",
            f"[KEY] Variation: {req_params.get('p_variation', 'N/A')}",
            f"  Taxonomy: {req_params.get('p_taxonomy_level', 'N/A')}",
            f"  Math Level: {req_params.get('p_mathematical_requirement_level', 'N/A')}",
            f"  Instruction: {req_params.get('p_instruction_explicitness_of_instruction', 'N/A')}"
        ]
        
        obstacle_count = sum(1 for key, value in req_params.items() 
                           if 'obstacle' in key and value == 'Enthalten')
        params_lines.append(f"  Linguistic Obstacles: {obstacle_count}")
        
        # Add text with dynamic wrapping
        full_text = req_params['text']
        text_wrapped = textwrap.fill(full_text, width=90)
        text_lines = text_wrapped.split('\n')
        params_lines.append("[KEY] Text:")
        params_lines.extend([f"  {line}" for line in text_lines])
        
        sections.append({
            'type': 'params',
            'title': 'LAYER 1: REQUEST RECEIVED',
            'lines': params_lines,
            'color': self.colors['input'],
            'height': len(params_lines) * 0.22 + 0.5  # Updated for better spacing
        })
        
        # Section 2: Generation
        formatted_question = self.format_question_text(question_data['question_text'])
        question_lines = formatted_question.split('\n')
        
        sections.append({
            'type': 'generation',
            'title': 'LAYER 2: GENERATION BY DEDICATED GENERATOR',
            'lines': question_lines,
            'color': self.colors['generation'],
            'height': len(question_lines) * 0.22 + 0.5
        })
        
        # Section 3: Expert Judging
        expert_types = ['variation_expert', 'taxonomy_expert', 'math_expert', 
                       'obstacle_expert', 'instruction_expert']
        expert_names = ['Variation', 'Taxonomy', 'Math', 'Obstacle', 'Instruction']
        
        expert_lines = []
        for expert_type, expert_name in zip(expert_types, expert_names):
            expert_data = next((e for e in question_data.get('expert_suggestions', []) 
                              if e['expert_type'] == expert_type), None)
            if expert_data:
                rating = expert_data['rating']
                feedback = expert_data['feedback']
                feedback_wrapped = textwrap.fill(feedback, width=70)
                expert_lines.append(f"[{expert_name}] Rating: {rating}/5")
                for line in feedback_wrapped.split('\n'):
                    expert_lines.append(f"  {line}")
        
        sections.append({
            'type': 'experts',
            'title': 'LAYER 3: EXPERT JUDGING',
            'lines': expert_lines,
            'color': self.colors['expert_ok'],
            'height': len(expert_lines) * 0.22 + 0.5
        })
        
        # Section 4: Refinement
        refinement_lines = [
            f"[RESULT] Refinement Applied: {'Yes' if question_data.get('refinement_applied', False) else 'No'}"
        ]
        
        expert_ratings = question_data.get('expert_ratings', {})
        if expert_ratings:
            avg_rating = sum(expert_ratings.values()) / len(expert_ratings)
        else:
            avg_rating = 0
        refinement_lines.append(f"[RESULT] Average Rating: {avg_rating:.1f}/5")
        
        reasoning = question_data.get('refinement_reasoning', 'N/A')
        reasoning_wrapped = textwrap.fill(reasoning, width=70)
        refinement_lines.append("[RESULT] Reasoning:")
        for line in reasoning_wrapped.split('\n'):
            refinement_lines.append(f"  {line}")
        
        sections.append({
            'type': 'refinement',
            'title': 'LAYER 4: EXPERT AGGREGATED REFINEMENT',
            'lines': refinement_lines,
            'color': self.colors['refinement'],
            'height': len(refinement_lines) * 0.22 + 0.5
        })
        
        # Section 5: Final Question
        formatted_final = self.format_question_text(final_question)
        final_lines = formatted_final.split('\n')
        
        sections.append({
            'type': 'final',
            'title': 'LAYER 5: SAVING',
            'lines': final_lines,
            'color': self.colors['final'],
            'height': len(final_lines) * 0.22 + 0.5
        })
        
        return sections
    
    def _render_section(self, ax, section, current_y, line_spacing, font_size):
        """Render a single section with proper formatting"""
        # Draw title
        title_font_size = max(8, font_size + 1)
        ax.text(0.2, current_y, section['title'], 
                fontsize=title_font_size, fontweight='bold', ha='left')
        current_y -= 0.03  # Tiny spacing between title and box
        
        # Calculate box height using actual line spacing
        box_height = len(section['lines']) * line_spacing + 0.25  # Reduced box padding
        
        # Create box
        linewidth = 2 if section['type'] == 'final' else 1.5
        box = Rectangle((0.2, current_y - box_height), 9.6, box_height,
                       facecolor=section['color'], edgecolor='black', linewidth=linewidth)
        ax.add_patch(box)
        
        # Add text
        text_y = current_y - 0.15
        for line in section['lines']:
            # Determine if line should be bold
            is_bold = (line.startswith('[KEY]') or 
                      line.startswith('[RESULT]') or 
                      (line.startswith('[') and '] Rating:' in line))
            
            weight = 'bold' if is_bold else 'normal'
            ax.text(0.3, text_y, line, fontsize=font_size, ha='left', va='top', fontweight=weight)
            text_y -= line_spacing
    
    def process_session(self, session_dir: Path):
        """Process a single session and create visualizations"""
        print(f"Processing session: {session_dir.name}")
        
        # Load session data
        data = self.load_session_data(session_dir)
        
        if not all(key in data for key in ['parameters', 'generation', 'final']):
            print(f"  Skipping - missing required data files")
            return
        
        # Create visualizations directory
        viz_dir = session_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Get typed questions from generation data
        typed_questions = data['generation']['step_data'].get('typed_questions', [])
        final_questions = data['final']['final_questions']
        
        # Create visualization for each question
        for i, (question_data, final_q) in enumerate(zip(typed_questions, final_questions)):
            output_path = viz_dir / f"question_{i+1}_pipeline.png"
            
            # Check if we should skip existing visualizations
            if output_path.exists() and not self.overwrite:
                print(f"  Skipping existing visualization: {output_path.name}")
                continue
                
            self.create_question_visualization(
                question_data, i+1, data['parameters'], final_q, output_path
            )
            print(f"  Created visualization: {output_path.name}")
    
    def run(self):
        """Main method to process all result folders"""
        print("Starting Result to Graphical Feedback Visualization")
        print(f"Scanning directory: {self.results_dir}")
        
        folders = self.find_result_folders()
        print(f"Found {len(folders)} result folders to process")
        
        for folder in folders:
            try:
                self.process_session(folder)
            except Exception as e:
                print(f"  Error processing {folder.name}: {str(e)}")
                continue
        
        print("\nVisualization complete!")


if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    overwrite = True  # Default to overwriting
    if len(sys.argv) > 1:
        if sys.argv[1] == '--no-overwrite':
            overwrite = False
            print("Running in no-overwrite mode - existing visualizations will be skipped")
    
    # Run the visualizer
    visualizer = PipelineVisualizer(overwrite=overwrite)
    visualizer.run()