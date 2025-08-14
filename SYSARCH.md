The System should be three-layered.

1. The caller: He sends a http request to the orchestrator.
The http request contains the following parameters:

- c_id,"Question number - difficulty (1 = Stammaufgabe, 2 = leicht, 3 = schwer) - question version" | GIVEN DIRECTLY BY USER!
Explanation: If the c_id is 41-1-4, then 41 is the question number, 1 is the difficulty, and 4 is the version of the question."
- text, "The text from which the question should be generated. It is a string that contains a Informational text about the systems pre-configured topic." | GIVEN DIRECTLY BY USER!
- question_type,"Werte: multiple-choice, single-choice, true-false, mapping" | GIVEN BY USER AND MODULAR CONFIGURED BY ORCHESTRATOR INTO CORRECT TXT PROMPT TO ADD FOR MASTER PROMPT!
Explanation: User request for a specific question format/type.
- p.variation,"Werte: Stammaufgabe, schwer, leicht" | GIVEN BY USER AND MODULAR CONFIGURED BY ORCHESTRATOR INTO CORRECT TXT PROMPT TO ADD FOR MASTER PROMPT!
Explanation: User request for a specific difficulty level of questions.
- p.taxanomy_level,"Werte: Stufe 1 (Wissen/Reproduktion), Stufe 2 (Anwendung/Transfer)" | GIVEN BY USER AND MODULAR CONFIGURED BY ORCHESTRATOR INTO CORRECT TXT PROMPT TO ADD FOR MASTER PROMPT!
Exlanation: The taxonomy level of the question the user is requesting.

Explanation of the following parameters: They are used to request a specific variation of the question. The main generator uses different modular parts in his main generation prompt depending on the values of these parameters.

IMPORTANT ARCHITECTURE RULE: The orchestrator contains NO hardcoded prompt strings. All prompt text must be loaded from external .txt files:
- mainGenPromptIntro.txt: Main generator introduction
- expertPromptIntro.txt: Expert validation introduction  
- refinementPromptIntro.txt: Question refinement introduction
- csvCorrectionPromptIntro.txt: CSV correction introduction
- outputFormatPrompt.txt: Output format instructions
- expertPrompts/: Expert-specific prompts
- All parameter-specific prompts in respective folders (variationPrompts/, taxonomyLevelPrompt/, etc.)

- p.root_text_reference_explanatory_text,"Werte: Nocht vorhanden, Explizit, Implizit" | GIVEN BY USER AND MODULAR CONFIGURED BY ORCHESTRATOR INTO CORRECT TXT PROMPT TO ADD FOR MASTER PROMPT!
- p.root_text_obstacle_passive,"Werte: Enthalten, Nicht Enthalten" | GIVEN BY USER AND MODULAR CONFIGURED BY ORCHESTRATOR INTO CORRECT TXT PROMPT TO ADD FOR MASTER PROMPT!
- p.root_text_obstacle_negation,"Werte: Enthalten, Nicht Enthalten" | GIVEN BY USER AND MODULAR CONFIGURED BY ORCHESTRATOR INTO CORRECT TXT PROMPT TO ADD FOR MASTER PROMPT!
- p.root_text_obstacle_complex_np,"Werte: Enthalten, Nicht Enthalten" | GIVEN BY USER AND MODULAR CONFIGURED BY ORCHESTRATOR INTO CORRECT TXT PROMPT TO ADD FOR MASTER PROMPT!
- p.root_text_contains_irrelevant_information,"Werte: Enthalten, Nicht Enthalten" | GIVEN BY USER AND MODULAR CONFIGURED BY ORCHESTRATOR INTO CORRECT TXT PROMPT TO ADD FOR MASTER PROMPT!
- p.mathematical_requirement_level,"Werte: 0 (Kein Bezug), 1 (Nutzen mathematischer Darstellungen), 2 (Mathematische Operation) "
- p.item_X_obstacle_passive,"Werte: Enthalten, Nicht Enthalten"
- p.item_X_obstacle_negation,"Werte: Enthalten, Nicht Enthalten"
- p.item_X_obstacle_complex_np,"Werte: Enthalten, Nicht Enthalten"
- p.instruction_obstacle_passive,"Instruction incudes a passive contstruction. Werte: Enthalten, Nicht Enthalten"
- p.instruction_obstacle_complex_np,"Instruction includes a complex noun phrase. Werte: Enthalten, Nicht Enthalten"
- p.instruction_explicitness_of_instruction,"Specifies the exact number of correct answers. Werte: Explizit, Implizit (= number not specified)"

The caller receives a simple HTTP response confirming successful generation. The orchestrator handles all result saving internally.

The orchestrator automatically:
1. Converts the three questions into a csv file with the complete SYSARCH format containing all required columns:

c_id,subject,type,text,p.instruction_explicitness_of_instruction,p.instruction_obstacle_complex_np,p.instruction_obstacle_negation,p.instruction_obstacle_passive,p.item_1_answer_verbatim_explanatory_text,p.item_1_obstacle_complex_np,p.item_1_obstacle_negation,p.item_1_obstacle_passive,p.item_1_sentence_length,p.item_2_answer_verbatim_explanatory_text,p.item_2_obstacle_complex_np,p.item_2_obstacle_negation,p.item_2_obstacle_passive,p.item_2_sentence_length,p.item_3_answer_verbatim_explanatory_text,p.item_3_obstacle_complex_np,p.item_3_obstacle_negation,p.item_3_obstacle_passive,p.item_3_sentence_length,p.item_4_answer_verbatim_explanatory_text,p.item_4_obstacle_complex_np,p.item_4_obstacle_negation,p.item_4_obstacle_passive,p.item_4_sentence_length,p.item_5_answer_verbatim_explanatory_text,p.item_5_obstacle_complex_np,p.item_5_obstacle_negation,p.item_5_obstacle_passive,p.item_5_sentence_length,p.item_6_answer_verbatim_explanatory_text,p.item_6_obstacle_complex_np,p.item_6_obstacle_negation,p.item_6_obstacle_passive,p.item_6_sentence_length,p.item_7_answer_verbatim_explanatory_text,p.item_7_obstacle_complex_np,p.item_7_obstacle_negation,p.item_7_obstacle_passive,p.item_7_sentence_length,p.item_8_answer_verbatim_explanatory_text,p.item_8_obstacle_complex_np,p.item_8_obstacle_negation,p.item_8_obstacle_passive,p.item_8_sentence_length,p.mathematical_requirement_level,p.root_text_contains_irrelevant_information,p.root_text_obstacle_complex_np,p.root_text_obstacle_negation,p.root_text_obstacle_passive,p.root_text_reference_explanatory_text,p.taxanomy_level,p.variation,answers,p.instruction_number_of_sentences

Example: 

35-3-7,1,true-false,"Entscheide, ob die Aussagen falsch oder richtig sind. <true-false> Alle Menschen haben die gleichen Bedürfnisse. <true-false> Die Bedürfnisse von Menschen sind nicht begrenzt. <true-false> Bedürfnisse beschreiben Wünsche, die durch Geld befriedigt werden können. <true-false> Bedürfnisse beschreiben Wünsche, die durch einen empfundenen Mangel entstehen. <true-false> Es gibt Bedürfnisse, die nicht mit Geld befriedigt werden können. <true-false> Wenn man alle seine Bedürfnisse befriedig hat, können keine neuen Bedürfnisse entstehen. ",Implizit,Nicht enthalten,Nicht enthalten,Nicht enthalten,Nicht enthalten,Nicht enthalten,Nicht enthalten,Nicht enthalten,,Nicht enthalten,Nicht enthalten,Enthalten,Nicht enthalten,,Nicht enthalten,Enthalten,Nicht enthalten,Enthalten,,Nicht enthalten,Enthalten,Nicht enthalten,Nicht enthalten,,Nicht enthalten,Enthalten,Enthalten,Enthalten,,Nicht enthalten,Nicht enthalten,Enthalten,Nicht enthalten,,,,,,,,,,,,0 (Kein Bezug),,,,,,Stufe 1,schwer,"Die Bedürfnisse von Menschen sind nicht begrenzt.; Bedürfnisse beschreiben Wünsche, die durch einen empfundenen Mangel entstehen.; Es gibt Bedürfnisse, die nicht mit Geld befriedigt werden können.",1

2. Uses intelligent fallbacks and LM assistance for CSV conversion when needed
3. Saves comprehensive results directly to CallersWithTexts/results/ using the integrated result_manager
4. Creates timestamped ISO date folders with seconds precision
5. Includes snapshot of all prompts .txt files from ALEE_AGENT
6. Saves complete system metadata: VRAM usage, model status, processing times, expert validation logs
7. Handles both successful generations and error cases with full logging

The caller is simplified to focus only on HTTP requests without data handling responsibilities.

2. The orchestrator: 
- He receives the http request,
- he calls upon all of his known local servers to check their health, then cleans them for a fresh start (Since new user request!) - they are ollama servers, so we need to carefully clean them without unnecessary restarts and complete reloads. just clear history. No resets or restarts if possible to minimise runtime. 
- constructs the main generator prompt using modular strings dependend on each parameters value. 
The constructor uses exclusively prompt txt he finds in the folders corresponding to the parameters, e.g. p.variation is variationPrompts, 
and there the txt corresponding to what the user has given as value for p.variation.
When the user has provided a number, for example in p.mathematical_requirement_level, the orchestrator uses the prompt txt that corresponds to the number, e.g. 0, 1, or 2.
Or when a number is specified in explicitness of instruction, the orchestrator uses the prompt txt that corresponds to the number, e.g. > 0 for "Explizit" (as well as the amount specified as well into the master prompt beside the txt!), or 0 for "Implizit".
- he also saves the configuration and gives them to the expert LMs at each call of check, so they can use their specific expertise precisely on the needed parameters. 
- The orchestrator then calls the main generator with the constructed prompt.
- He receives the three questions from the main generator.
- One by one, consecutively, he calls the expert LMs with the questions and the configuration.
- At each of them, he receives their judgment, and accordingly, he decides whether to change the question with the experts suggestion and rating. 
- He does this, until all the questions have been checked as passing by all the experts, or until the maximum number of 3 expert iterations has been reached.
- Finally, he returns the three questions to the caller in an http response of 1. "" 2. "" 3. ""

3. The expert LMs:
- Each has one master prompt that contains the instructions for the expert LM. He receives the question and the configuration parameters.
- He checks the question for the parameters that are relevant for his expertise, takes them seriously, and gives a rating of 1-5 for the question
- as well as a suggestion for improvement, if he thinks the question needs to be changed.
- He returns the rating and the suggestion to the orchestrator.