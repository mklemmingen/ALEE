#!/usr/bin/env python3
"""
Interactive Question Generator - User-Guided Orchestrator Interface
Guides users through all parameters with colorful validation and real-time feedback
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import new modular components
from sharedTools.caller import OrchestorCaller
from sharedTools.logger import CallerLogger
from sharedTools.config import CallerConfig

try:
    from colorama import Fore, Back, Style, init
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    print("Installing colorama for colorful output...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "colorama"])
    from colorama import Fore, Back, Style, init
    init(autoreset=True)
    COLORAMA_AVAILABLE = True

# Color definitions - clean white-on-black and black-on-white scheme
class Colors:
    SUCCESS = Fore.GREEN + Style.BRIGHT
    ERROR = Fore.RED + Style.BRIGHT 
    WARNING = Fore.YELLOW + Style.BRIGHT
    INFO = Fore.WHITE + Style.BRIGHT     # Clean white text for info
    PROMPT = Fore.WHITE + Style.BRIGHT   # Clean white text for prompts
    INPUT = Fore.WHITE + Style.BRIGHT    # Clean white text for input
    HEADER = Fore.BLACK + Back.WHITE + Style.BRIGHT  # Black text on white background
    PARAMETER = Fore.WHITE + Style.BRIGHT  # Clean white text for parameters
    VALUE = Fore.WHITE + Style.NORMAL      # Slightly dimmed white for values
    RESET = Style.RESET_ALL

def print_header(text: str):
    """Print a colorful header"""
    print(f"\n{Colors.HEADER} {text} {Colors.RESET}")
    print(f"{Colors.INFO}{'=' * (len(text) + 2)}{Colors.RESET}")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.SUCCESS}[SUCCESS] {text}{Colors.RESET}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.ERROR}[ERROR] {text}{Colors.RESET}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}[WARNING] {text}{Colors.RESET}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.INFO}[INFO] {text}{Colors.RESET}")

def print_progress(text: str):
    """Print progress indicator"""
    print(f"{Colors.PROMPT}[PROGRESS] {text}{Colors.RESET}")

def get_user_input(prompt: str, default: str = None, validator=None) -> str:
    """Get validated user input with colorful prompts"""
    while True:
        if default:
            display_prompt = f"{Colors.PROMPT}{prompt} [{Colors.VALUE}{default}{Colors.PROMPT}]: {Colors.INPUT}"
        else:
            display_prompt = f"{Colors.PROMPT}{prompt}: {Colors.INPUT}"
        
        try:
            user_input = input(display_prompt).strip()
            if not user_input and default:
                user_input = default
            
            if validator:
                is_valid, message = validator(user_input)
                if not is_valid:
                    print_error(message)
                    continue
            
            print(f"{Colors.RESET}", end="")  # Reset color after input
            return user_input
            
        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING}Operation cancelled by user{Colors.RESET}")
            sys.exit(0)

def validate_c_id(c_id: str) -> tuple[bool, str]:
    """Validate c_id format (e.g., 181-1-3)"""
    if not c_id:
        return False, "c_id cannot be empty"
    
    parts = c_id.split('-')
    if len(parts) != 3:
        return False, "c_id must be in format: number-number-number (e.g., 181-1-3)"
    
    try:
        for part in parts:
            int(part)
        return True, ""
    except ValueError:
        return False, "c_id parts must be numbers (e.g., 181-1-3)"

def validate_text(text: str) -> tuple[bool, str]:
    """Validate educational text"""
    if not text or len(text.strip()) < 10:
        return False, "Educational text must be at least 10 characters long"
    
    if len(text) > 5000:
        return False, "Educational text should be under 5000 characters for better processing"
    
    return True, ""

def validate_choice(value: str, valid_choices: List[str]) -> tuple[bool, str]:
    """Validate choice against valid options"""
    if value not in valid_choices:
        return False, f"Must be one of: {', '.join(valid_choices)}"
    return True, ""

def get_parameter_choices() -> Dict[str, Dict[str, Any]]:
    """Define all parameter choices with descriptions"""
    return {
        "question_type": {
            "description": "Question format/type",
            "choices": ["multiple-choice", "single-choice", "true-false", "mapping"],
            "details": {
                "multiple-choice": "Questions with multiple correct answers",
                "single-choice": "Questions with exactly one correct answer", 
                "true-false": "True/false statements to evaluate",
                "mapping": "Match terms to definitions or concepts"
            }
        },
        "p_variation": {
            "description": "Question difficulty level",
            "choices": ["stammaufgabe", "schwer", "leicht"],
            "details": {
                "stammaufgabe": "Standard difficulty for average 9th grade students",
                "schwer": "Hard difficulty for advanced students", 
                "leicht": "Easy difficulty for struggling students"
            }
        },
        "p_taxonomy_level": {
            "description": "Cognitive complexity level (Bloom's taxonomy)",
            "choices": ["Stufe 1 (Wissen/Reproduktion)", "Stufe 2 (Anwendung/Transfer)"],
            "details": {
                "Stufe 1 (Wissen/Reproduktion)": "Knowledge recall and reproduction",
                "Stufe 2 (Anwendung/Transfer)": "Application and transfer to new situations"
            }
        },
        "p_mathematical_requirement_level": {
            "description": "Mathematical complexity required",
            "choices": ["0", "1", "2"],
            "details": {
                "0": "No mathematical content required",
                "1": "Use of mathematical representations/diagrams",
                "2": "Active mathematical calculations required"
            }
        },
        "p_root_text_reference_explanatory_text": {
            "description": "Reference to explanatory text in root content",
            "choices": ["Nicht vorhanden", "Explizit", "Implizit"],
            "details": {
                "Nicht vorhanden": "No reference to explanatory text",
                "Explizit": "Explicit reference to explanatory content",
                "Implizit": "Implicit reference to explanatory content"
            }
        },
        "p_instruction_explicitness_of_instruction": {
            "description": "How explicit the task instructions should be",
            "choices": ["Explizit", "Implizit"],
            "details": {
                "Explizit": "Clear, detailed step-by-step instructions",
                "Implizit": "Brief, less detailed instructions"
            }
        }
    }

def get_obstacle_parameters() -> Dict[str, str]:
    """Get obstacle parameter descriptions"""
    return {
        "passive": "Passive voice constructions",
        "negation": "Negation/negative formulations", 
        "complex_np": "Complex noun phrases"
    }

def display_parameter_info(param_name: str, param_info: Dict[str, Any]):
    """Display colorful parameter information"""
    print(f"\n{Colors.PARAMETER}Parameter: {param_name}{Colors.RESET}")
    print(f"{Colors.INFO}Description: {param_info['description']}{Colors.RESET}")
    
    if 'choices' in param_info:
        print(f"{Colors.INFO}Available options:{Colors.RESET}")
        for i, choice in enumerate(param_info['choices'], 1):
            detail = param_info['details'].get(choice, "")
            print(f"  {Colors.VALUE}{i}. {choice}{Colors.RESET} - {detail}")

def collect_basic_parameters() -> Dict[str, Any]:
    """Collect basic required parameters"""
    print_header("BASIC PARAMETERS")
    
    # c_id
    print_info("First, we need a unique identifier for your question set")
    c_id = get_user_input(
        "Enter c_id (format: number-number-number, e.g., 181-1-3)",
        validator=validate_c_id
    )
    
    # Educational text
    print_info("\nNow provide the educational text that questions will be based on")
    print_warning("This should be comprehensive educational content (economics, business, etc.)")
    text = get_user_input(
        "Enter educational text (min 10 chars)",
        validator=validate_text
    )
    
    return {"c_id": c_id, "text": text}

def collect_core_parameters() -> Dict[str, Any]:
    """Collect core educational parameters"""
    print_header("CORE EDUCATIONAL PARAMETERS")
    
    params = {}
    param_info = get_parameter_choices()
    
    for param_name, info in param_info.items():
        display_parameter_info(param_name, info)
        
        choices = info['choices']
        while True:
            choice = get_user_input(f"Select {param_name}", choices[0])
            is_valid, message = validate_choice(choice, choices)
            if is_valid:
                params[param_name] = choice
                print_success(f"Set {param_name} = {choice}")
                break
            else:
                print_error(message)
    
    return params

def collect_obstacle_parameters() -> Dict[str, Any]:
    """Collect obstacle parameters for text, items, and instructions"""
    print_header("OBSTACLE PARAMETERS")
    print_info("These parameters control linguistic complexity obstacles")
    
    params = {}
    obstacle_types = get_obstacle_parameters()
    
    # Root text obstacles
    print(f"\n{Colors.PARAMETER}Root Text Obstacles:{Colors.RESET}")
    for obstacle, description in obstacle_types.items():
        param_name = f"p_root_text_obstacle_{obstacle}"
        print_info(f"Setting {description} in root text")
        choice = get_user_input(
            f"{param_name} [Enthalten/Nicht Enthalten]",
            "Nicht Enthalten",
            lambda x: validate_choice(x, ["Enthalten", "Nicht Enthalten"])
        )
        params[param_name] = choice
    
    # Irrelevant information
    params["p_root_text_contains_irrelevant_information"] = get_user_input(
        "Root text contains irrelevant information [Enthalten/Nicht Enthalten]",
        "Nicht Enthalten",
        lambda x: validate_choice(x, ["Enthalten", "Nicht Enthalten"])
    )
    
    # Item obstacles (1-8)
    print(f"\n{Colors.PARAMETER}Item-Specific Obstacles (Items 1-8):{Colors.RESET}")
    print_info("Setting obstacles for individual answer items")
    
    use_same_for_all = get_user_input(
        "Use same obstacle settings for all items 1-8? [y/N]",
        "y"
    ).lower() == 'y'
    
    if use_same_for_all:
        print_info("Setting obstacles for all items...")
        item_obstacles = {}
        for obstacle, description in obstacle_types.items():
            choice = get_user_input(
                f"All items - {description} [Enthalten/Nicht Enthalten]",
                "Nicht Enthalten",
                lambda x: validate_choice(x, ["Enthalten", "Nicht Enthalten"])
            )
            item_obstacles[obstacle] = choice
        
        # Apply to all items 1-8
        for i in range(1, 9):
            for obstacle in obstacle_types.keys():
                params[f"p_item_{i}_obstacle_{obstacle}"] = item_obstacles[obstacle]
    else:
        # Individual item settings
        for i in range(1, 9):
            print(f"\n{Colors.PARAMETER}Item {i} obstacles:{Colors.RESET}")
            for obstacle, description in obstacle_types.items():
                param_name = f"p_item_{i}_obstacle_{obstacle}"
                choice = get_user_input(
                    f"Item {i} - {description} [Enthalten/Nicht Enthalten]",
                    "Nicht Enthalten",
                    lambda x: validate_choice(x, ["Enthalten", "Nicht Enthalten"])
                )
                params[param_name] = choice
    
    # Instruction obstacles
    print(f"\n{Colors.PARAMETER}Instruction Obstacles:{Colors.RESET}")
    for obstacle, description in obstacle_types.items():
        param_name = f"p_instruction_obstacle_{obstacle}"
        choice = get_user_input(
            f"Instructions - {description} [Enthalten/Nicht Enthalten]",
            "Nicht Enthalten",
            lambda x: validate_choice(x, CallerConfig.VALID_BINARY_VALUES)
        )
        params[param_name] = choice
    
    return params

def display_request_summary(request_data: Dict[str, Any]):
    """Display a colorful summary of the request"""
    print_header("REQUEST SUMMARY")
    
    print(f"{Colors.PARAMETER}Basic Info:{Colors.RESET}")
    print(f"  {Colors.INFO}c_id:{Colors.RESET} {Colors.VALUE}{request_data['c_id']}{Colors.RESET}")
    print(f"  {Colors.INFO}Text length:{Colors.RESET} {Colors.VALUE}{len(request_data['text'])} characters{Colors.RESET}")
    print(f"  {Colors.INFO}Text preview:{Colors.RESET} {Colors.VALUE}{request_data['text'][:100]}...{Colors.RESET}")
    
    print(f"\n{Colors.PARAMETER}Core Parameters:{Colors.RESET}")
    core_params = ["p_variation", "p_taxonomy_level", "p_mathematical_requirement_level", 
                   "p_instruction_explicitness_of_instruction"]
    for param in core_params:
        if param in request_data:
            print(f"  {Colors.INFO}{param}:{Colors.RESET} {Colors.VALUE}{request_data[param]}{Colors.RESET}")
    
    # Count obstacles
    obstacle_counts = {"Enthalten": 0, "Nicht Enthalten": 0}
    for key, value in request_data.items():
        if "obstacle" in key or "irrelevant" in key:
            obstacle_counts[value] = obstacle_counts.get(value, 0) + 1
    
    print(f"\n{Colors.PARAMETER}Obstacles Summary:{Colors.RESET}")
    print(f"  {Colors.INFO}Active obstacles:{Colors.RESET} {Colors.VALUE}{obstacle_counts.get('Enthalten', 0)}{Colors.RESET}")
    print(f"  {Colors.INFO}Inactive obstacles:{Colors.RESET} {Colors.VALUE}{obstacle_counts.get('Nicht Enthalten', 0)}{Colors.RESET}")

def send_request_to_orchestrator(request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Send request to orchestrator with progress feedback"""
    orchestrator_url = "http://localhost:8000/generate-educational-questions"
    
    print_header("SENDING REQUEST TO ORCHESTRATOR")
    print_info(f"Endpoint: {orchestrator_url}")
    
    try:
        print_progress("Connecting to orchestrator...")
        start_time = time.time()
        
        response = requests.post(
            orchestrator_url,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minutes timeout
        )
        
        elapsed = time.time() - start_time
        print_info(f"Request completed in {elapsed:.2f} seconds")
        
        if response.status_code == 200:
            print_success("Request successful!")
            return response.json()
        else:
            print_error(f"Request failed with status {response.status_code}")
            try:
                error_data = response.json()
                print_error(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print_error(f"Error response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to orchestrator!")
        print_warning("Make sure the orchestrator is running on http://localhost:8000")
        print_info("Start it with: python3 ALEE_Agent/_server_question_generator.py")
        return None
        
    except requests.exceptions.Timeout:
        print_error("Request timed out after 5 minutes")
        print_warning("The orchestrator might be overloaded or processing slowly")
        return None
        
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        return None

def display_generation_progress(result_data: Dict[str, Any]):
    """Display the generation progress and updates"""
    if not result_data or 'generation_updates' not in result_data:
        return
        
    print_header("GENERATION PROGRESS")
    
    updates = result_data.get('generation_updates', [])
    if not updates:
        print_warning("No generation updates available")
        return
    
    print_info(f"Tracked {len(updates)} generation steps:")
    
    for i, update in enumerate(updates, 1):
        step = update.get('step', 'unknown')
        timestamp = update.get('timestamp', '')
        source = update.get('source', 'unknown')
        
        if step == 'initial_generation':
            print(f"\n{Colors.SUCCESS}Step {i}: Initial Generation{Colors.RESET}")
            print(f"  {Colors.INFO}Source:{Colors.RESET} {source}")
            print(f"  {Colors.INFO}Time:{Colors.RESET} {timestamp}")
            questions = update.get('questions', [])
            for j, q in enumerate(questions, 1):
                print(f"  {Colors.VALUE}Q{j}:{Colors.RESET} {q[:80]}...")
                
        elif 'refinement' in step:
            print(f"\n{Colors.WARNING}Step {i}: Expert Refinement{Colors.RESET}")
            print(f"  {Colors.INFO}Question:{Colors.RESET} {update.get('question_num', 'N/A')}")
            print(f"  {Colors.INFO}Iteration:{Colors.RESET} {update.get('iteration', 'N/A')}")
            print(f"  {Colors.INFO}Time:{Colors.RESET} {timestamp}")
            
            original = update.get('original_question', '')
            refined = update.get('refined_question', '')
            if original != refined:
                print(f"  {Colors.ERROR}Before:{Colors.RESET} {original[:60]}...")
                print(f"  {Colors.SUCCESS}After:{Colors.RESET} {refined[:60]}...")
                
                feedback = update.get('expert_feedback', [])
                if feedback:
                    print(f"  {Colors.INFO}Feedback:{Colors.RESET}")
                    for fb in feedback[:2]:  # Show first 2 feedback items
                        print(f"    • {fb[:100]}...")

def display_final_results(result_data: Dict[str, Any], request_data: Dict[str, Any]):
    """Display the final generated questions and metadata"""
    print_header("FINAL RESULTS")
    
    if not result_data:
        print_error("No results to display")
        return
    
    # Display questions
    questions = [
        result_data.get('question_1', ''),
        result_data.get('question_2', ''),
        result_data.get('question_3', '')
    ]
    
    print_success(f"Successfully generated {len([q for q in questions if q])} questions!")
    print_info(f"Processing time: {result_data.get('processing_time', 0):.2f} seconds")
    print_info(f"c_id: {result_data.get('c_id', 'N/A')}")
    
    print(f"\n{Colors.PARAMETER}Generated Questions:{Colors.RESET}")
    for i, question in enumerate(questions, 1):
        if question:
            print(f"\n{Colors.VALUE}Question {i}:{Colors.RESET}")
            print(f"  {question}")
    
    # CSV data summary
    csv_data = result_data.get('csv_data', {})
    if csv_data:
        print(f"\n{Colors.PARAMETER}CSV Data Summary:{Colors.RESET}")
        print(f"  {Colors.INFO}Type:{Colors.RESET} {csv_data.get('type', 'N/A')}")
        print(f"  {Colors.INFO}Subject:{Colors.RESET} {csv_data.get('subject', 'N/A')}")
        print(f"  {Colors.INFO}CSV columns:{Colors.RESET} {len(csv_data)}")

def find_result_files(c_id: str) -> List[Path]:
    """Find result files for the given c_id"""
    results_dir = Path(__file__).parent.parent / "ALEE_Agent" / "results"
    if not results_dir.exists():
        return []
    
    result_files = []
    for session_dir in results_dir.iterdir():
        if session_dir.is_dir() and c_id in session_dir.name:
            result_files.extend(session_dir.rglob("*"))
    
    return [f for f in result_files if f.is_file()]

def display_saved_results_info(c_id: str):
    """Display information about saved results"""
    print_header("SAVED RESULTS")
    
    result_files = find_result_files(c_id)
    
    if not result_files:
        print_warning(f"No saved result files found for c_id: {c_id}")
        print_info("Results might be saved in the orchestrator's internal result system")
        return
    
    print_success(f"Found {len(result_files)} result files for c_id: {c_id}")
    
    # Group by directory
    directories = {}
    for file_path in result_files:
        dir_name = file_path.parent.name
        if dir_name not in directories:
            directories[dir_name] = []
        directories[dir_name].append(file_path)
    
    for dir_name, files in directories.items():
        print(f"\n{Colors.PARAMETER}{dir_name}:{Colors.RESET}")
        for file_path in sorted(files):
            file_size = file_path.stat().st_size if file_path.exists() else 0
            print(f"  {Colors.VALUE}{file_path.name}{Colors.RESET} ({file_size} bytes)")
        
        # Show full path to first directory
        if files:
            full_path = files[0].parent
            print(f"  {Colors.INFO}Location:{Colors.RESET} {full_path}")

def main():
    """Main interactive function"""
    print_header("INTERACTIVE QUESTION GENERATOR")
    print_info("Welcome! This tool will guide you through generating educational questions.")
    print_warning("Make sure the orchestrator is running on http://localhost:8000")
    
    try:
        # Collect parameters step by step
        basic_params = collect_basic_parameters()
        core_params = collect_core_parameters()
        obstacle_params = collect_obstacle_parameters()
        
        # Combine all parameters
        request_data = {**basic_params, **core_params, **obstacle_params}
        
        # Show summary and confirm
        display_request_summary(request_data)
        
        confirm = get_user_input(
            f"\n{Colors.PROMPT}Proceed with request? [Y/n]",
            "Y"
        ).lower()
        
        if confirm not in ['y', 'yes', '']:
            print_warning("Request cancelled by user")
            return
        
        # Send request
        result_data = send_request_to_orchestrator(request_data)
        
        if result_data:
            # Display progress and results
            display_generation_progress(result_data)
            display_final_results(result_data, request_data)
            
            # Show where results are saved
            display_saved_results_info(request_data['c_id'])
            
            print_success("\nQuestion generation completed successfully!")
            print_info("You can now use the generated questions for your educational purposes.")
            
        else:
            print_error("Question generation failed. Check the error messages above.")
            
    except KeyboardInterrupt:
        print_warning("\n\nOperation cancelled by user. Goodbye!")
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        import traceback
        print_error("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()