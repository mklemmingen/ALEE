#!/usr/bin/env python3
"""
Prompt Tree Visualization using Graphviz
Creates a tree visualization with expert coverage using graphviz for layout
"""

import json
from pathlib import Path
import graphviz
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Ellipse
import subprocess

class PromptTreeGraphviz:
    def __init__(self):
        self.load_structure()
        self.setup_expert_mapping()
        
    def load_structure(self):
        """Load directory structure from JSON"""
        json_path = Path(__file__).parent / "prompt_directory_structure.json"
        with open(json_path, 'r') as f:
            self.data = json.load(f)["ALEE_Agent_Prompt_Structure"]
    
    def setup_expert_mapping(self):
        """Setup expert to prompt mapping based on prompt_builder.py"""
        self.expert_colors = {
            'variation_expert': '#FF6B6B',      # Red
            'math_expert': '#4ECDC4',           # Teal
            'obstacle_expert': '#45B7D1',       # Blue
            'taxonomy_expert': '#96CEB4',       # Green  
            'instruction_expert': '#FECA57',    # Yellow
            'content_expert': '#DDA0DD',        # Purple
            'generator': '#FFB347'              # Orange
        }
        
        # Define which prompts each expert accesses - based on README table
        self.expert_access = {
            'variation_expert': {
                'dirs': ['variationPrompts', 'taxonomyLevelPrompt'],
                'files': ['stammaufgabe.txt', 'schwer.txt', 'leicht.txt', 'multiple-choice.txt', 
                         'single-choice.txt', 'true-false.txt', 'mapping.txt',
                         'stufe1WissenReproduktion.txt', 'stufe2AnwendungTransfer.txt']
            },
            'taxonomy_expert': {
                'dirs': ['taxonomyLevelPrompt'],
                'files': ['stufe1WissenReproduktion.txt', 'stufe2AnwendungTransfer.txt']
            },
            'math_expert': {
                'dirs': ['mathematicalRequirementLevel'],
                'files': ['0KeinBezug.txt', '1NutzenMathematischerDarstellungen.txt', 
                         '2MathematischeOperationen.txt']
            },
            'text_reference_expert': {
                'dirs': ['rootTextParameterTextPrompts'],
                'files': ['enthalten.txt', 'nichtEnthalten.txt']  # For text reference parameters
            },
            'obstacle_expert': {
                'dirs': ['obstaclePassivePrompts', 'obstacleNegationPrompts', 'obstacleComplexPrompts',
                        'itemXObstacle', 'instructionObstacle'],
                'files': ['enthalten.txt', 'nichtEnthalten.txt']  # These appear in multiple obstacle dirs
            },
            'instruction_expert': {
                'dirs': ['instructionExplicitnessOfInstruction', 'instructionObstacle'],
                'files': ['explizit.txt', 'implizit.txt', 'enthalten.txt', 'nichtEnthalten.txt']
            },
            'generator': {
                'dirs': ['dtoAndOutputPrompt', 'variationPrompts', 'taxonomyLevelPrompt', 
                        'mathematicalRequirementLevel', 'instructionExplicitnessOfInstruction',
                        'obstaclePassivePrompts', 'obstacleNegationPrompts', 'obstacleComplexPrompts',
                        'itemXObstacle', 'instructionObstacle', 'containsIrrelevantInformationPrompt'],
                'files': ['questionGenerationInstruction.txt', 'outputFormatPrompt.txt',
                         'multipleChoiceAnnotatedExamples.txt', 'singleChoiceAnnotatedExamples.txt',
                         'trueFalseAnnotatedExamples.txt', 'mappingAnnotatedExamples.txt',
                         'stammaufgabe.txt', 'schwer.txt', 'leicht.txt', 'multiple-choice.txt',
                         'single-choice.txt', 'true-false.txt', 'mapping.txt',
                         'stufe1WissenReproduktion.txt', 'stufe2AnwendungTransfer.txt',
                         '0KeinBezug.txt', '1NutzenMathematischerDarstellungen.txt', 
                         '2MathematischeOperationen.txt', 'explizit.txt', 'implizit.txt',
                         'enthalten.txt', 'nichtEnthalten.txt']
            }
        }
    
    def get_file_expert_color(self, filename, dir_path):
        """Determine which expert(s) access a file and return appropriate color"""
        accessing_experts = []
        
        for expert, access_info in self.expert_access.items():
            # Check if file is directly listed
            if filename in access_info['files']:
                accessing_experts.append(expert)
            
            # Check if directory is accessed by expert - improved matching
            for dir_name in access_info['dirs']:
                if (dir_name in dir_path or 
                    dir_path.endswith(dir_name) or
                    # Special obstacle directory matching
                    (dir_name == 'itemXObstacle' and 'itemXObstacle' in dir_path) or
                    (dir_name == 'instructionObstacle' and 'instructionObstacle' in dir_path) or
                    (dir_name.startswith('obstacle') and dir_name in dir_path)):
                    accessing_experts.append(expert)
        
        # Remove duplicates but preserve order
        accessing_experts = list(dict.fromkeys(accessing_experts))
        
        # Return color based on number of accessing experts
        if len(accessing_experts) == 0:
            return '#FFFFFF'  # White for no access
        elif len(accessing_experts) == 1:
            return self.expert_colors[accessing_experts[0]]  # Expert's color
        elif len(accessing_experts) == 2:
            # Two experts - show primary expert color but lighter
            primary_expert = accessing_experts[0] if accessing_experts[0] != 'generator' else accessing_experts[1]
            return self.expert_colors[primary_expert] + '80'  # Add transparency
        else:
            # Multiple experts - use mixed color
            return '#E0E0E0'  # Light gray for shared access
    
    def get_file_expert_label(self, filename, dir_path):
        """Get expert labels for a file"""
        accessing_experts = []
        
        for expert, access_info in self.expert_access.items():
            # Check if file is directly listed
            if filename in access_info['files']:
                accessing_experts.append(expert)
            
            # Check if directory is accessed by expert - improved matching
            for dir_name in access_info['dirs']:
                if (dir_name in dir_path or 
                    dir_path.endswith(dir_name) or
                    # Special obstacle directory matching
                    (dir_name == 'itemXObstacle' and 'itemXObstacle' in dir_path) or
                    (dir_name == 'instructionObstacle' and 'instructionObstacle' in dir_path) or
                    (dir_name.startswith('obstacle') and dir_name in dir_path)):
                    accessing_experts.append(expert)
        
        # Remove duplicates
        accessing_experts = list(dict.fromkeys(accessing_experts))
        
        if accessing_experts:
            # Create short labels
            expert_labels = [e.split('_')[0][0].upper() for e in accessing_experts]
            return f" [{','.join(expert_labels)}]"
        return ""
    
    def _get_subdir_color(self, subdir_name, dir_path):
        """Get color for subdirectory based on expert access"""
        accessing_experts = []
        
        for expert, access_info in self.expert_access.items():
            for dir_name in access_info['dirs']:
                # More flexible matching - check if any part matches
                if (dir_name in dir_path or 
                    subdir_name in dir_name or 
                    dir_name in subdir_name or
                    dir_path.endswith(dir_name) or
                    # Special cases for obstacle directories
                    (dir_name == 'itemXObstacle' and subdir_name in ['passive', 'negation', 'complex']) or
                    (dir_name == 'instructionObstacle' and subdir_name in ['passive', 'negation', 'complex_np'])):
                    accessing_experts.append(expert)
        
        # Remove duplicates
        accessing_experts = list(dict.fromkeys(accessing_experts))
        
        if len(accessing_experts) == 0:
            return '#F0F0F0'  # Default gray
        elif len(accessing_experts) == 1:
            # Use expert color directly (not lighter)
            return self.expert_colors[accessing_experts[0]]
        else:
            return '#E0E0E0'  # Mixed access
    
    def create_graphviz_tree(self):
        """Create tree using graphviz"""
        dot = graphviz.Digraph('ALEE_Prompts', format='png')
        dot.attr(rankdir='TB', size='20,24', dpi='300')
        dot.attr('graph', ranksep='1.5', nodesep='0.5')
        dot.attr('node', shape='box', style='rounded,filled', fontsize='12')
        dot.attr('edge', fontsize='10')
        
        # Add legend as a subgraph
        with dot.subgraph(name='cluster_legend') as legend:
            legend.attr(label='Expert Legend', fontsize='14', style='filled', color='lightgrey')
            legend.attr('node', shape='box', style='filled')
            
            # Create legend nodes
            for expert, color in self.expert_colors.items():
                expert_short = expert.split('_')[0][0].upper()
                expert_label = f"{expert_short}: {expert.replace('_', ' ').title()}"
                legend.node(f"legend_{expert}", expert_label, fillcolor=color, fontsize='10')
        
        # Root node
        dot.node('root', 'ALEE_Agent', fillcolor='#333333', fontcolor='white')
        
        # Main directories - force vertical layout
        main_colors = {
            'mainGen': '#FFE5B4',
            'expertEval': '#E6E6FA',
            'dtoAndOutputPrompt': '#F0E68C'
        }
        
        for main_dir in ['mainGen', 'expertEval', 'dtoAndOutputPrompt']:
            dot.node(main_dir, main_dir, fillcolor=main_colors[main_dir])
            dot.edge('root', main_dir)
        
        # Add directory contents
        for main_dir in ['mainGen', 'expertEval', 'dtoAndOutputPrompt']:
            if main_dir in self.data:
                self._add_directory_to_graph(dot, main_dir, self.data[main_dir], main_dir)
        
        # Render to file
        dot_path = Path(__file__).parent / 'prompt_tree_graphviz'
        dot.render(dot_path, cleanup=True)
        
        return f"{dot_path}.png"
    
    def _add_directory_to_graph(self, dot, parent_id, dir_data, path_prefix):
        """Recursively add directory contents to graph"""
        # Handle direct files with expert color coding
        if 'files' in dir_data and dir_data['files']:
            # Create individual file nodes instead of grouped
            for i, filename in enumerate(dir_data['files']):
                file_id = f"{parent_id}_file_{i}"
                file_color = self.get_file_expert_color(filename, path_prefix)
                expert_label = self.get_file_expert_label(filename, path_prefix)
                file_label = f"{filename}{expert_label}"
                
                dot.node(file_id, file_label, shape='note', fillcolor=file_color, 
                        fontsize='10', style='filled')
                dot.edge(parent_id, file_id)
        
        # Handle subdirectories - force vertical layout with rank constraints
        if 'subdirectories' in dir_data:
            subdir_names = list(dir_data['subdirectories'].keys())
            
            # Group subdirectories by level to control layout
            for i, (subdir_name, subdir_data) in enumerate(dir_data['subdirectories'].items()):
                subdir_id = f"{parent_id}_{subdir_name}"
                # Color subdirectory based on expert access
                subdir_color = self._get_subdir_color(subdir_name, f"{path_prefix}/{subdir_name}")
                dot.node(subdir_id, subdir_name, fillcolor=subdir_color)
                dot.edge(parent_id, subdir_id)
                
                self._process_subdirectory(dot, subdir_id, subdir_data, f"{path_prefix}/{subdir_name}")
    
    def _process_subdirectory(self, dot, parent_id, subdir_data, path_prefix):
        """Process subdirectory recursively"""
        if isinstance(subdir_data, dict):
            if 'files' in subdir_data:
                # Create individual file nodes with expert color coding
                for i, filename in enumerate(subdir_data['files']):
                    file_id = f"{parent_id}_file_{i}"
                    file_color = self.get_file_expert_color(filename, path_prefix)
                    expert_label = self.get_file_expert_label(filename, path_prefix)
                    file_label = f"{filename}{expert_label}"
                    
                    dot.node(file_id, file_label, shape='note', fillcolor=file_color, 
                            fontsize='8', style='filled')
                    dot.edge(parent_id, file_id)
            
            if 'subdirectories' in subdir_data:
                for sub_name, sub_data in subdir_data['subdirectories'].items():
                    sub_id = f"{parent_id}_{sub_name}"
                    subdir_color = self._get_subdir_color(sub_name, f"{path_prefix}/{sub_name}")
                    dot.node(sub_id, sub_name, fillcolor=subdir_color)
                    dot.edge(parent_id, sub_id)
                    self._process_subdirectory(dot, sub_id, sub_data, f"{path_prefix}/{sub_name}")
        elif isinstance(subdir_data, list):
            # Handle direct file list
            for i, filename in enumerate(subdir_data):
                file_id = f"{parent_id}_file_{i}"
                file_color = self.get_file_expert_color(filename, path_prefix)
                expert_label = self.get_file_expert_label(filename, path_prefix)
                file_label = f"{filename}{expert_label}"
                
                dot.node(file_id, file_label, shape='note', fillcolor=file_color, 
                        fontsize='10', style='filled')
                dot.edge(parent_id, file_id)
    
    def add_expert_overlays(self, tree_image_path):
        """Add expert coverage circles as overlay on the tree image"""
        # This would require parsing the graphviz output to get node positions
        # For now, let's create a separate visualization showing expert mappings
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
        
        # Left side: Directory structure summary
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.axis('off')
        ax1.set_title('Prompt Directory Structure', fontsize=16, fontweight='bold')
        
        # Draw main directories as boxes
        main_dirs = {
            'mainGen': (5, 8, '#FFE5B4'),
            'expertEval': (2, 5, '#E6E6FA'),
            'dtoAndOutputPrompt': (8, 5, '#F0E68C')
        }
        
        for dir_name, (x, y, color) in main_dirs.items():
            rect = patches.FancyBboxPatch((x-1.5, y-0.3), 3, 0.6,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color,
                                        edgecolor='black')
            ax1.add_patch(rect)
            ax1.text(x, y, dir_name, ha='center', va='center', fontweight='bold')
        
        # Right side: Expert coverage
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.axis('off')
        ax2.set_title('Expert Access Patterns', fontsize=16, fontweight='bold')
        
        # Draw expert coverage areas
        expert_positions = {
            'variation_expert': (2, 8),
            'taxonomy_expert': (4, 8),
            'math_expert': (6, 8),
            'text_reference_expert': (8, 8),
            'obstacle_expert': (2, 5),
            'instruction_expert': (5, 5),
            'generator': (8, 5)
        }
        
        for expert, (x, y) in expert_positions.items():
            # Draw circle for expert
            circle = Circle((x, y), 1.2, facecolor=self.expert_colors[expert],
                          alpha=0.3, edgecolor=self.expert_colors[expert],
                          linewidth=2)
            ax2.add_patch(circle)
            
            # Add expert label
            ax2.text(x, y, expert.replace('_', '\n'), ha='center', va='center',
                    fontsize=9, fontweight='bold')
            
            # Add accessed directories below
            access_info = self.expert_access[expert]
            dirs_text = '\n'.join(access_info['dirs'][:3])
            if len(access_info['dirs']) > 3:
                dirs_text += '\n...'
            
            ax2.text(x, y-1.8, dirs_text, ha='center', va='top',
                    fontsize=7, style='italic', alpha=0.7)
        
        # Save combined visualization
        output_path = Path(__file__).parent / "prompt_tree_expert_mapping.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path
    
    def create_full_visualization(self):
        """Create complete visualization"""
        print("Creating tree structure with graphviz...")
        tree_path = self.create_graphviz_tree()
        print(f"Tree saved to: {tree_path}")
        
        print("Creating expert mapping visualization...")
        expert_path = self.add_expert_overlays(tree_path)
        print(f"Expert mapping saved to: {expert_path}")
        
        return tree_path, expert_path


def main():
    visualizer = PromptTreeGraphviz()
    tree_path, expert_path = visualizer.create_full_visualization()
    print("\nVisualization complete!")
    print(f"Tree structure: {tree_path}")
    print(f"Expert mapping: {expert_path}")


if __name__ == "__main__":
    main()