# BouncyQuestionCreator: Educational AI Question Generation System

## Project Overview

This project implements an advanced multi-agent AI system for generating high-quality educational questions, with support for both general educational content and specialized German economics curricula.

## Current Implementation Status

### Completed: General Educational Question System

**Architecture**: Multi-agent validation system with specialized LMs
- **Generator** (`startPrompt.txt`): Creates educational questions
- **Meaningfulness Checker** (`chooserPrompt.txt`): Validates educational value 
- **Robustness Validator** (`suggestorPrompt.txt`): Ensures quality and accuracy
- **Uniqueness Checker** (`childPrompt.txt`): Prevents duplicates
- **Progress Assessor** (`rectorPrompt.txt`): Evaluates learning progression

**Key Features**:
- Async architecture for parallel validation
- Iterative refinement (up to 3 attempts)
- Weighted scoring system (meaningfulness 30%, robustness 25%, uniqueness 25%, progress 20%)
- Configurable endpoints for multiple local LM instances
- Batch processing capabilities

## Stakeholder Requirements Analysis

### Target System: German 9th-Grade Economics Question Bank

Based on analysis of `providedProjectFromStakeHolder/`, stakeholders require a specialized system with:

#### Core Requirements
1. **Language**: German (9th-grade level)
2. **Subject**: Economics/Business Studies curriculum
3. **Question Types**: Single-choice, multiple-choice, true-false, mapping, fill-in-the-blank
4. **Output**: Structured CSV format with detailed parameter columns

#### Sophisticated Generation Process
**Two-Step Approach**:
1. **Base Questions (Stammaufgabe)**: Foundation questions for Taxonomy Levels 1 & 2
2. **Variations**: Generate "leicht" (easier) and "schwer" (harder) variants

#### Parameter-Driven Difficulty System
- **Easier variants**: Reduce distractors, remove linguistic obstacles, use verbatim text
- **Harder variants**: Increase distractors (up to 8), add linguistic obstacles, avoid verbatim answers

#### Advanced Linguistic Parameters
- `p.instruction_obstacle_passive/negation/complex_np` and `p.item_X_obstacle_passive/negation/complex_np`
  - X stands for the item number (e.g., 1, 2, 3)
  - passive example: Wähle die eine richtige Verteilungssituation aus, in denen die Güter unter anderem über einen Markt verteilt werden.
  - negation example: Entscheide, ob es sich bei den folgenden Dingen um ein Gut handelt oder nicht.
  - complex noun phrase example: Ordne den links stehenden Kategorien von Bedürfnissen Jugendlicher die richtigen Beispiele zu.
- `p.item_X_answer_verbatim_explanatory_text`
- `p.mathematical_requirement_level`
  - Werte: 0 (Kein Bezug), 1 (Nutzen mathematischer Darstellungen), 2 (Mathematische Operation)
- `p.taxonomy_level` (Stufe 1: Knowledge, Stufe 2: Application)
  - Stufe 1: Bloom's taxonomy Remembering level, Stufe 2: Understanding and Application levels

#### Quality Control Requirements
- Plausible distractors based on student misconceptions
- Similar length between attractors and distractors
- Maximum 2 attractors per question
- Strict adherence to reference text content

## Reference Data Analysis & System Compliance

### Provided Assets
1. **`ProvidedStuff.MD`**: Detailed generation prompts and guidelines
2. **`explanation_metadata.csv`**: 17 reference texts on economic concepts (source material for questions)
3. **`task_metadata_with_answers_final2_colcleaned.csv`**: Reference question database with **58-column structure**
4. **`tasks_parameters.csv`**: Parameter definitions and specifications
5. **`Did Annotationen/` Excel files**: Categorized question types and formatting examples

### **System Compliance Analysis**

#### **CSV Output Format Compliance**
Our system generates questions in exact compliance with `task_metadata_with_answers_final2_colcleaned.csv` format:

**Complete 58-Column Structure:**
- **Core**: `c_id`, `subject`, `type`, `text`, `answers`
- **Variation**: `p.variation` ("Stammaufgabe", "schwer", "leicht")
- **Taxonomy**: `p.taxanomy_level` ("Stufe 1", "Stufe 2")  
- **Mathematical**: `p.mathematical_requirement_level` (0-2)
- **Instruction Parameters**: `p.instruction_obstacle_passive/negation/complex_np`
- **Item Parameters**: `p.item_X_answer_verbatim_explanatory_text` (X = 1-8)
- **Root Text Parameters**: `p.root_text_*` (reference adherence tracking)

#### **Question Type Format Compliance**

**True-False Questions**
```
Format: "Entscheide, ob die Aussagen falsch oder richtig sind. <true-false> [Statement1] <true-false> [Statement2]..."
Answers: "Semicolon-separated TRUE statements only"
```

**Multiple-Choice Questions** 
```
Format: "[Question stem] <option> [Option1] <option> [Option2] <option> [Option3]..."
Answers: "Exact text of correct option(s)"
```

**Single-Choice Questions**
```
Format: "[Question stem] <option> [Option1] <option> [Option2] <option> [Option3]"
Answers: "Single correct option text"
```

**Mapping Questions**
```
Format: "Ordne... <start-option> [Left1] <start-option> [Left2] <end-option> [Right1] <end-option> [Right2]"
Answers: "Left1 -> Right1; Left2 -> Right2"
```

#### **ID Format Compliance**
**c_id Pattern**: `{question_number}-{difficulty_level}-{version}`
- Question number: Sequential (35, 36, 37...)
- Difficulty: 1=Stammaufgabe, 2=leicht, 3=schwer
- Version: Numeric identifier
- **Example**: "35-3-7" (Question 35, schwer difficulty, version 7)

#### **Parameter Value Compliance**
All parameter values use exact German strings:
- **Binary Parameters**: "Enthalten" / "Nicht enthalten"
- **Explicitness**: "Explizit" / "Implizit"
- **Reference Levels**: "Nicht vorhanden" / "Explizit" / "Implizit"
- **Taxonomy**: "Stufe 1" / "Stufe 2"
- **Mathematical**: "0 (Kein Bezug)" / "1 (Nutzen mathematischer Darstellungen)" / "2 (Mathematische Operation)"

#### **Reference Text Integration**
Questions are generated strictly from `explanation_metadata.csv` reference texts:
- **17 Economic Topics**: Bedürfnisse, Güter, Märkte, etc.
- **Content Adherence**: All questions must be derivable from reference text
- **Text IDs**: Match reference material to generated questions

## System Evolution Path

### Phase 1: Hierarchical Architecture ✅
- **Multi-agent validation**: Mother LM → Principal → Student Personas
- **German educational content**: 9th-grade economics questions
- **CSV output compliance**: Exact 58-column format matching stakeholder data
- **Persona-based difficulty validation**: Kim (leicht), Alex (Stammaufgabe), Lisa (schwer)
- **5-iteration refinement**: Continuous improvement until perfection
- **Parameter compliance**: All German economics parameters validated

### Phase 2: Complete Integration ✅
**Implemented Features**:
1. **58-Column CSV Export**: Exact format compliance with `task_metadata_with_answers_final2_colcleaned.csv`
2. **Reference Text Integration**: Questions generated from `explanation_metadata.csv`
3. **Question Type Support**: True-false, multiple-choice, single-choice, mapping
4. **Parameter Engine**: Complete German parameter validation system
5. **Linguistic Obstacle System**: Passive voice, negation, complex noun phrases
6. **Taxonomy Validation**: Stufe 1 (Knowledge) and Stufe 2 (Application) compliance
7. **Student Persona Validation**: Authentic 9th-grader feedback system

## Architecture Comparison

### Current vs. Target System

| Aspect | Current System | Stakeholder Requirements |
|--------|---------------|-------------------------|
| Language | German | German |
| Subject | General educational content | 9th-grade economics curriculum |
| Output | CSV | Structured CSV with detailed parameters |
| Difficulty | Basic levels | Parameter-driven variations (leicht/schwer) |
| Content Source | Generated from scratch | Strict reference text adherence |
| Question Structure | Simple validation | Complex parameter manipulation |

## Implementation Strategy

### Recommended Approach: Hybrid System
1. **Preserve Current Architecture**: Multi-agent validation concept remains valuable
2. **Add German Module**: Specialized German economics question generator
3. **Parameter Framework**: Sophisticated difficulty manipulation engine
4. **Enhanced CSV Output**: Structured CSV format with all required parameter columns

### Technical Requirements
- German language models (Llama-3.1 or similar with German training)
- Economics domain knowledge integration
- Advanced parameter manipulation system
- CSV generation pipeline
- Reference text compliance validator

## Project Structure

```
PythonProject/
├── BouncyQuestionCreator.py      # Main orchestrator system
├── config.json                   # LM endpoint configuration
├── startPrompt.txt               # Question generator prompt
├── chooserPrompt.txt             # Meaningfulness validator prompt
├── suggestorPrompt.txt           # Robustness validator prompt
├── childPrompt.txt               # Uniqueness checker prompt
├── rectorPrompt.txt              # Progress assessor prompt
├── providedProjectFromStakeHolder/
│   ├── ProvidedStuff.MD          # Stakeholder requirements
│   ├── explanation_metadata.csv  # Reference texts
│   ├── task_metadata_*.csv       # Question database
│   └── tasks_parameters.csv      # Parameter definitions
└── README.md                     # This documentation
```

## Next Steps

1. **System Architecture Decision**: Extend current system vs. build specialized module
2. **German Model Integration**: Set up German language model endpoints
3. **Parameter Engine Development**: Build sophisticated difficulty manipulation
4. **Reference Text System**: Implement strict content adherence validation
5. **CSV Pipeline**: Replace JSON output with structured CSV format

## Development Environment

- **Platform**: Linux (Manjaro)
- **GPU**: 20GB VRAM (suitable for multiple local LM instances)
- **Python**: Async/await architecture with aiohttp
- **Models**: Llama-3.1-8b-instruct (configurable endpoints)

---

**Status**: Analysis complete, ready for stakeholder alignment on implementation approach.