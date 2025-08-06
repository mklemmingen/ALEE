+ bei der Stammaufgabe: The answers should not be verbatim, but in leicht doch.
+ two prompts for questions generating and parameters?
+ the answer shouldn’t be obvious from the length
+ provide examples for EVERYTHING (esp the linguistic variation)

# Single-Choice Question Generation Prompt with Variation Approach

You are creating {question_type} questions for a German 9th-grade economics question bank. All questions must be based strictly on the provided reference text and follow specific parameter requirements from "tasks_parameters" file. The manually crafted reference questions are in the "task_metadata…" file.

## IMPORTANT: Generation Approach

You will follow a two-step process:

1. **First**: Generate base questions (Stammaufgabe) for both Stufe 1 and Stufe 2
2. **Then**: Create variations (leicht and schwer) of EACH base question by modifying specific parameters

## Reference Text

{reference_text}

## Key Parameter Variations for Difficulty Adjustment

When creating variations from base questions, modify these parameters:

### For EASIER (leicht) variations:

- **Reduce number of distractors** (e.g., from 3 to 2 options)
- **Remove linguistic obstacles**:
    - p.instruction_obstacle_passive = "Nicht enthalten"
    - p.instruction_obstacle_negation = "Nicht enthalten"
    - p.instruction_obstacle_complex_np = "Nicht enthalten"
- **Use verbatim answers from text** where possible (p.item_X_answer_verbatim_explanatory_text = "Enthalten")

### For HARDER (schwer) variations:

- **Increase number of distractors** (e.g., from 3 to 5 options, at most to 8)
- **Add linguistic obstacles** (at least one of):
    - p.instruction_obstacle_passive = "Enthalten"
    - p.instruction_obstacle_negation = "Enthalten"
    - p.instruction_obstacle_complex_np = "Enthalten"
- **Avoid verbatim answers** (p.item_X_answer_verbatim_explanatory_text = "Nicht enthalten")

## Distractor Generation Guidelines

- **Plausibility**: Each distractor must be plausible enough that a student might reasonably select it
- **Misconception-based**: Draw on common student misconceptions about economic concepts
- **Similarity**: Distractors should be related to the correct answer but clearly incorrect
- **Diversity**: Generate varied types of incorrect options, not just minor variations
- **Length**: Distractors and attractors should have similar length; the attractors shouldn’t be longer than the distractors.

## Generation Instructions

### Step 1: Generate Base Questions (Stammaufgabe)

Create 2 base questions:

1. **Base Question 1** - Taxonomy Level 1 (Stufe 1)
    - Type: {question-type}
    - Distractors: 4 (total 5 options)
    - Focus: Basic recall
    - All linguistic obstacle parameters = "Nicht enthalten"
    - Have at most 2 attractors
2. **Base Question 2** - Taxonomy Level 2 (Stufe 2)
	- Same as Base Question 2, but with Taxonomy Level 2 (Stufe 2)

### Step 2: Create Variations

For EACH base question, create:

#### Easy Variation (leicht):

- Reduce distractors by 1
- Keep all parameters simple
- Ensure at least one answer option uses verbatim text from reference

#### Difficult Variation (schwer):

- Increase distractors by 2 (6 total options)
- Add at least one linguistic obstacle to the instruction
- Ensure no answer options use verbatim text

## Output Format

Create a CSV file with the following structure:

- Include headers for all parameters
- Use c_id format: QuestionNumber-Difficulty-Version
    - Difficulty codes: 1 = Stammaufgabe, 2 = leicht, 3 = schwer
- Each variation of a question should have the same QuestionNumber
- Format questions exactly as in reference file:
    - `<option>` for single-choice and single-choice (NO closing tags)
    - `<true-false>` for true-false
    - Include question text with all options inline


Remember:

- The CORE QUESTION CONTENT remains the same across variations
- Only modify the specified parameters to adjust difficulty
- Stay strictly within the content of the reference text