#!/usr/bin/env python3
"""
ALEE Agent UML Diagram Generator
Creates a comprehensive UML-style architecture diagram of the DSPy-enhanced educational question generation system
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_alee_uml_diagram():
    """Create a UML-style architecture diagram of the ALEE system"""
    
    # Create figure with A4 landscape dimensions (11.7 x 8.3 inches)
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.set_xlim(0, 11.7)
    ax.set_ylim(0, 8.3)
    ax.axis('off')
    
    # Title with proper spacing
    ax.text(5.85, 7.9, "ALEE Agent - DSPy Educational Question Generation System", 
            fontsize=14, fontweight='bold', ha='center')
    ax.text(5.85, 7.5, "Three-Layer Architecture with Expert Consensus Validation", 
            fontsize=10, ha='center', style='italic')
    
    # Color scheme
    colors = {
        'layer1': '#E3F2FD',      # Light blue for Layer 1 (HTTP Interface)
        'layer2': '#FFF3E0',      # Light orange for Layer 2 (DSPy Orchestrator)
        'layer3': '#E8F5E9',      # Light green for Layer 3 (Expert LLMs)
        'storage': '#F3E5F5',     # Light purple for Storage
        'external': '#FFEBEE',    # Light red for External Systems
        'border': '#424242'       # Dark gray for borders
    }
    
    # Layer 1: HTTP Interface & Callers (with proper spacing from title)
    layer1_box = FancyBboxPatch((0.3, 6.2), 11.1, 1.0, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['layer1'], 
                               edgecolor=colors['border'], 
                               linewidth=2)
    ax.add_patch(layer1_box)
    ax.text(5.85, 7.0, "LAYER 1: HTTP INTERFACE & CALLERS", 
            fontsize=11, fontweight='bold', ha='center')
    
    # Caller components
    caller_boxes = [
        ("Stakeholder\nTest System", 1.3, 6.6),
        ("Single Request\nTest", 3.2, 6.6),
        ("Interactive\nInterface", 5.1, 6.6),
        ("Orchestor\nCaller", 7.0, 6.6),
        ("FastAPI\nServer", 8.9, 6.6),
        ("Result\nVisualizer", 10.8, 6.6)
    ]
    
    for name, x, y in caller_boxes:
        box = FancyBboxPatch((x-0.5, y-0.2), 1.0, 0.4, 
                            boxstyle="round,pad=0.05", 
                            facecolor='white', 
                            edgecolor=colors['border'])
        ax.add_patch(box)
        ax.text(x, y, name, fontsize=7, ha='center', va='center')
    
    # Layer 2: DSPy Orchestrator (with more spacing)
    layer2_box = FancyBboxPatch((0.3, 4.5), 11.1, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['layer2'], 
                               edgecolor=colors['border'], 
                               linewidth=2)
    ax.add_patch(layer2_box)
    ax.text(5.85, 5.5, "LAYER 2: DSPY ORCHESTRATOR", 
            fontsize=11, fontweight='bold', ha='center')
    
    # DSPy components
    dspy_components = [
        ("TypeAware\nPipeline", 1.5, 5.0),
        ("Question\nGenerator\n(OLLAMA 8001)", 3.0, 5.0),
        ("Prompt\nBuilder", 4.5, 5.0),
        ("Expert\nEnhancer", 6.0, 5.0),
        ("Question\nRefiner", 7.5, 5.0),
        ("Result\nManager", 9.0, 5.0),
        ("Pipeline\nVisualizer", 10.5, 5.0)
    ]
    
    for name, x, y in dspy_components:
        box = FancyBboxPatch((x-0.4, y-0.2), 0.8, 0.4, 
                            boxstyle="round,pad=0.05", 
                            facecolor='white', 
                            edgecolor=colors['border'])
        ax.add_patch(box)
        ax.text(x, y, name, fontsize=7, ha='center', va='center')
    
    # Layer 3: Expert LLMs (with more spacing)
    layer3_box = FancyBboxPatch((0.3, 2.8), 11.1, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['layer3'], 
                               edgecolor=colors['border'], 
                               linewidth=2)
    ax.add_patch(layer3_box)
    ax.text(5.85, 3.8, "LAYER 3: EXPERT LLM VALIDATION (OLLAMA Ports 8001-8007)", 
            fontsize=11, fontweight='bold', ha='center')
    
    # Expert components
    experts = [
        ("Variation\nExpert", 1.3, 3.3),
        ("Taxonomy\nExpert", 2.7, 3.3),
        ("Math\nExpert", 4.1, 3.3),
        ("Obstacle\nExpert", 5.5, 3.3),
        ("Instruction\nExpert", 6.9, 3.3),
        ("Content\nExpert", 8.3, 3.3),
        ("Expert\nConsensus", 9.7, 3.3)
    ]
    
    for name, x, y in experts:
        box = FancyBboxPatch((x-0.4, y-0.2), 0.8, 0.4, 
                            boxstyle="round,pad=0.05", 
                            facecolor='white', 
                            edgecolor=colors['border'])
        ax.add_patch(box)
        ax.text(x, y, name, fontsize=7, ha='center', va='center')
    
    # Storage Systems
    storage_box = FancyBboxPatch((0.3, 0.5), 5.5, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['storage'], 
                                edgecolor=colors['border'], 
                                linewidth=2)
    ax.add_patch(storage_box)
    ax.text(3.05, 1.5, "STORAGE & PERSISTENCE", 
            fontsize=10, fontweight='bold', ha='center')
    
    storage_components = [
        ("Session Results\n(timestamped)", 1.5, 1.0),
        ("CSV Export\n(compliant)", 3.05, 1.0),
        ("Pipeline\nTracking", 4.6, 1.0)
    ]
    
    for name, x, y in storage_components:
        box = FancyBboxPatch((x-0.6, y-0.15), 1.2, 0.3, 
                            boxstyle="round,pad=0.05", 
                            facecolor='white', 
                            edgecolor=colors['border'])
        ax.add_patch(box)
        ax.text(x, y, name, fontsize=7, ha='center', va='center')
    
    # External Prompt System
    external_box = FancyBboxPatch((6.0, 0.5), 5.4, 1.2, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['external'], 
                                 edgecolor=colors['border'], 
                                 linewidth=2)
    ax.add_patch(external_box)
    ax.text(8.7, 1.5, "EXTERNAL PROMPT SYSTEM", 
            fontsize=10, fontweight='bold', ha='center')
    
    external_components = [
        ("45+ .txt Files\n(Parameter-specific)", 7.2, 1.0),
        ("Expert Prompts\n(6 specialists)", 8.7, 1.0),
        ("Parameter\nTemplates (PYdantic!)", 10.2, 1.0)
    ]
    
    for name, x, y in external_components:
        box = FancyBboxPatch((x-0.6, y-0.15), 1.2, 0.3, 
                            boxstyle="round,pad=0.05", 
                            facecolor='white', 
                            edgecolor=colors['border'])
        ax.add_patch(box)
        ax.text(x, y, name, fontsize=7, ha='center', va='center')
    
    # Data Flow Arrows with proper spacing
    # Layer 1 to Layer 2
    arrow1 = ConnectionPatch((5.85, 6.2), (5.85, 5.7), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, 
                           mutation_scale=15, fc=colors['border'], lw=2)
    ax.add_patch(arrow1)
    
    # Layer 2 to Layer 3  
    arrow2 = ConnectionPatch((5.85, 4.5), (5.85, 4.0), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, 
                           mutation_scale=15, fc=colors['border'], lw=2)
    ax.add_patch(arrow2)
    
    # Storage connections
    arrow4 = ConnectionPatch((3.05, 4.5), (3.05, 1.7), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, 
                           mutation_scale=15, fc=colors['border'], lw=2)
    ax.add_patch(arrow4)
    
    # External prompt connections
    arrow5 = ConnectionPatch((8.7, 4.5), (8.7, 1.7), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, 
                           mutation_scale=15, fc=colors['border'], lw=2)
    ax.add_patch(arrow5)
    
    # Add flow labels with proper positioning
    ax.text(6.2, 5.8, "HTTP Request\n(39 parameters)", fontsize=7, ha='left')
    ax.text(6.2, 4.15, "Expert Validation\n(Parallel)", fontsize=7, ha='left')
    
    # Add watermark
    ax.text(5.85, 0.05, "Kateryna Lauterbach, 2025, github.com/mklemmingen/ALEE", 
            fontsize=8, ha='center', va='bottom', alpha=0.5, 
            color='gray', style='italic')
    
    # Save the diagram
    plt.tight_layout()
    plt.savefig('/home/mklemmingen/PycharmProjects/PythonProject/ALEE_Agent_UML_Architecture.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print("ALEE Agent UML Architecture diagram saved to:")
    print("- ALEE_Agent_UML_Architecture.png")

if __name__ == "__main__":
    create_alee_uml_diagram()