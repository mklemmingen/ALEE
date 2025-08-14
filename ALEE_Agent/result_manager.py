"""
Modular Result Management System for Educational AI System
Manages timestamped result folders and saves prompts + CSV results

Usage:
    from result_manager import save_results
    save_results(csv_data, metadata={"test": "batch_1"})
    
    # Or use class for multiple operations
    from result_manager import ResultManager
    rm = ResultManager()
    session_dir = rm.save_results(csv_data)
"""
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

class ResultManager:
    """Modular result manager for educational AI system results with incremental saving support"""
    
    def __init__(self, base_results_dir: str = None):
        if base_results_dir is None:
            # Default to results directory in same folder as this script
            self.base_results_dir = Path(__file__).parent / "results"
        else:
            self.base_results_dir = Path(base_results_dir)
        self.base_results_dir.mkdir(exist_ok=True)
        self.current_session_dir = None  # Track current active session
        
    def save_results(self, 
                    csv_data: Union[List[Dict[str, Any]], str, Path], 
                    metadata: Optional[Dict[str, Any]] = None,
                    custom_timestamp: Optional[str] = None,
                    prompts_source_dir: Optional[str] = None) -> Path:
        """
        Main method: Save CSV data with prompts and metadata to timestamped folder
        
        Args:
            csv_data: List of dicts, CSV file path, or CSV content string
            metadata: Optional metadata dict to save
            custom_timestamp: Optional custom timestamp (ISO format)
            prompts_source_dir: Optional custom prompts directory path
            
        Returns:
            Path to created session directory
        """
        # Create timestamped session folder
        if custom_timestamp:
            timestamp = custom_timestamp
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
        session_dir = self.base_results_dir / timestamp
        session_dir.mkdir(exist_ok=True)
        
        # Save CSV data
        csv_path = self._save_csv_data(session_dir, csv_data)
        
        # Save prompts snapshot
        self._save_prompts_snapshot(session_dir, prompts_source_dir)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            "session_timestamp": timestamp,
            "created_at": datetime.now().isoformat(),
            "csv_file": csv_path.name if csv_path else None,
            "prompts_saved": len(list((session_dir / "prompts").rglob("*.txt"))) if (session_dir / "prompts").exists() else 0
        })
        self._save_metadata(session_dir, metadata)
        
        return session_dir

    def create_session_package(self, c_id: str, request_data: Dict[str, Any], 
                             custom_timestamp: Optional[str] = None) -> Path:
        """
        Create a new session package for incremental result saving during question generation
        
        Args:
            c_id: Question ID for this session
            request_data: Initial request parameters
            custom_timestamp: Optional custom timestamp
            
        Returns:
            Path to created session directory
        """
        # Create timestamped session folder
        if custom_timestamp:
            timestamp = custom_timestamp
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        session_dir = self.base_results_dir / f"{timestamp}_{c_id}"
        session_dir.mkdir(exist_ok=True)
        self.current_session_dir = session_dir
        
        # Create subdirectories for incremental saving
        (session_dir / "iterations").mkdir(exist_ok=True)
        (session_dir / "prompts").mkdir(exist_ok=True)
        (session_dir / "parameters").mkdir(exist_ok=True)
        
        # Save prompts snapshot at session creation
        self._save_prompts_snapshot(session_dir)
        
        # Save initial request parameters
        self._save_request_parameters(session_dir, request_data)
        
        # Save initial metadata
        initial_metadata = {
            "c_id": c_id,
            "session_timestamp": timestamp,
            "session_started_at": datetime.now().isoformat(),
            "request_parameters": request_data,
            "session_type": "incremental_question_generation",
            "iterations_saved": 0,
            "final_results_saved": False
        }
        self._save_metadata(session_dir, initial_metadata)
        
        return session_dir

    def save_iteration_result(self, iteration_num: int, questions: List[str], 
                            prompts_used: Dict[str, str], expert_feedback: Dict[str, Any] = None,
                            processing_metadata: Dict[str, Any] = None) -> bool:
        """
        Save results from a specific iteration of question generation
        
        Args:
            iteration_num: Current iteration number (1, 2, 3, etc.)
            questions: List of questions generated in this iteration
            prompts_used: Dictionary of prompts used {prompt_type: prompt_content}
            expert_feedback: Optional expert feedback received
            processing_metadata: Optional processing information (timing, models used, etc.)
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.current_session_dir:
            print("Warning: No active session package. Call create_session_package() first.")
            return False
            
        try:
            iteration_dir = self.current_session_dir / "iterations" / f"iteration_{iteration_num:02d}"
            iteration_dir.mkdir(exist_ok=True)
            
            # Save questions for this iteration
            questions_file = iteration_dir / "questions.json"
            with open(questions_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "iteration": iteration_num,
                    "timestamp": datetime.now().isoformat(),
                    "questions": questions,
                    "question_count": len(questions)
                }, f, indent=2, ensure_ascii=False)
            
            # Save prompts used in this iteration
            prompts_file = iteration_dir / "prompts_used.json"
            with open(prompts_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "iteration": iteration_num,
                    "timestamp": datetime.now().isoformat(),
                    "prompts": prompts_used
                }, f, indent=2, ensure_ascii=False)
            
            # Save expert feedback if provided
            if expert_feedback:
                feedback_file = iteration_dir / "expert_feedback.json"
                with open(feedback_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "iteration": iteration_num,
                        "timestamp": datetime.now().isoformat(),
                        "expert_feedback": expert_feedback
                    }, f, indent=2, ensure_ascii=False)
            
            # Save processing metadata if provided
            if processing_metadata:
                metadata_file = iteration_dir / "processing_metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "iteration": iteration_num,
                        "timestamp": datetime.now().isoformat(),
                        "processing_metadata": processing_metadata
                    }, f, indent=2, ensure_ascii=False)
            
            # Update session metadata with iteration count
            self._update_session_metadata({"iterations_saved": iteration_num})
            
            return True
            
        except Exception as e:
            print(f"Warning: Failed to save iteration {iteration_num} results: {e}")
            return False

    def save_final_results(self, final_questions: List[str], csv_data: Dict[str, Any], 
                         final_metadata: Dict[str, Any]) -> bool:
        """
        Save the final results of the question generation process
        
        Args:
            final_questions: Final approved questions
            csv_data: Complete CSV data for SYSARCH compliance
            final_metadata: Final session metadata
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.current_session_dir:
            print("Warning: No active session package. Call create_session_package() first.")
            return False
            
        try:
            print(f"Saving final results to session: {self.current_session_dir}")
            print(f"Final questions count: {len(final_questions)}")
            print(f"CSV data keys: {list(csv_data.keys()) if csv_data else 'No CSV data'}")
            
            # Save final questions
            final_questions_file = self.current_session_dir / "final_questions.json"
            with open(final_questions_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "final_questions": final_questions,
                    "question_count": len(final_questions),
                    "finalized_at": datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            print(f"Successfully saved final questions to: {final_questions_file}")
            
            # Save CSV data
            csv_file = self.current_session_dir / "results.csv"
            if csv_data:
                with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_data.keys())
                    writer.writeheader()
                    writer.writerow(csv_data)
                print(f"Successfully saved CSV data to: {csv_file}")
            else:
                print("Warning: No CSV data provided to save")
            
            # Update final metadata
            final_metadata.update({
                "session_completed_at": datetime.now().isoformat(),
                "final_results_saved": True,
                "total_processing_time": (datetime.now() - datetime.fromisoformat(final_metadata.get("session_started_at", datetime.now().isoformat()))).total_seconds()
            })
            
            self._save_metadata(self.current_session_dir, final_metadata)
            print(f"Successfully updated session metadata with final_results_saved=True")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to save final results: {e}")
            print(f"ERROR: Session dir: {self.current_session_dir}")
            print(f"ERROR: Session dir exists: {self.current_session_dir.exists() if self.current_session_dir else 'N/A'}")
            import traceback
            print(f"ERROR: Full traceback: {traceback.format_exc()}")
            return False

    def _save_request_parameters(self, session_dir: Path, request_data: Dict[str, Any]):
        """Save the original request parameters"""
        params_file = session_dir / "parameters" / "request_parameters.json"
        try:
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "request_parameters": request_data,
                    "saved_at": datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save request parameters: {e}")

    def _update_session_metadata(self, updates: Dict[str, Any]):
        """Update session metadata with new information"""
        if not self.current_session_dir:
            return
            
        metadata_file = self.current_session_dir / "session_metadata.json"
        try:
            # Read existing metadata
            existing_metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)
            
            # Update with new information
            existing_metadata.update(updates)
            existing_metadata["last_updated"] = datetime.now().isoformat()
            
            # Save updated metadata
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(existing_metadata, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Warning: Failed to update session metadata: {e}")
        
    def _save_csv_data(self, session_dir: Path, csv_data: Union[List[Dict[str, Any]], str, Path]) -> Optional[Path]:
        """Save CSV data in various formats"""
        if csv_data is None:
            return None
            
        csv_path = session_dir / "results.csv"
        
        try:
            if isinstance(csv_data, (str, Path)):
                # Handle file path or CSV string
                csv_input = Path(csv_data) if isinstance(csv_data, str) and Path(csv_data).exists() else csv_data
                
                if isinstance(csv_input, Path) and csv_input.exists():
                    # Copy existing CSV file
                    shutil.copy2(csv_input, csv_path)
                else:
                    # Treat as CSV string content
                    with open(csv_path, 'w', encoding='utf-8') as f:
                        f.write(str(csv_data))
                        
            elif isinstance(csv_data, list) and csv_data:
                # Handle list of dictionaries
                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    if isinstance(csv_data[0], dict):
                        fieldnames = csv_data[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(csv_data)
                    else:
                        # Simple list of rows
                        writer = csv.writer(csvfile)
                        writer.writerows(csv_data)
            else:
                return None
                
            return csv_path
            
        except Exception as e:
            print(f"Warning: Failed to save CSV data: {e}")
            return None
            
    def _save_prompts_snapshot(self, session_dir: Path, prompts_source_dir: Optional[str] = None):
        """Save snapshot of all ALEE_Agent prompt directories (mainGen, expertEval, dtoAndOutputPrompt)"""
        if prompts_source_dir is None:
            # Default: ALEE_Agent directory (same directory as this result_manager.py file)
            alee_agent_dir = Path(__file__).parent
        else:
            alee_agent_dir = Path(prompts_source_dir)
            
        prompts_dest = session_dir / "prompts"
        prompts_dest.mkdir(exist_ok=True)
        
        # Define the new organized prompt directories to copy
        prompt_directories = [
            "mainGen",      # Contains mainGenPromptIntro.txt and all parameter subdirectories
            "expertEval",   # Contains expertPromptIntro.txt, refinementPromptIntro.txt, and expertPrompts/
            "dtoAndOutputPrompt"  # Contains csvCorrectionPromptIntro.txt and outputFormatPrompt.txt
        ]
        
        total_files_copied = 0
        
        for dir_name in prompt_directories:
            source_dir = alee_agent_dir / dir_name
            if source_dir.exists() and source_dir.is_dir():
                dest_subdir = prompts_dest / dir_name
                try:
                    # Copy entire directory structure
                    shutil.copytree(source_dir, dest_subdir, dirs_exist_ok=True)
                    
                    # Count files copied for logging
                    files_in_dir = len(list(dest_subdir.rglob("*.txt")))
                    total_files_copied += files_in_dir
                    print(f"Copied {files_in_dir} prompt files from {dir_name}/")
                    
                except Exception as e:
                    print(f"Warning: Failed to copy directory {dir_name}: {e}")
            else:
                print(f"Warning: Prompt directory not found: {source_dir}")
        
        print(f"Prompts snapshot complete: {total_files_copied} total files copied to {prompts_dest}")
            
    def _save_metadata(self, session_dir: Path, metadata: Dict[str, Any]):
        """Save session metadata as JSON"""
        metadata_path = session_dir / "session_metadata.json"
        
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Failed to save metadata: {e}")
            
    def get_latest_session(self) -> Optional[Path]:
        """Get the most recent session folder"""
        try:
            session_folders = [d for d in self.base_results_dir.iterdir() if d.is_dir()]
            if session_folders:
                return max(session_folders, key=lambda x: x.stat().st_mtime)
        except Exception:
            pass
        return None
        
    def list_sessions(self) -> List[str]:
        """List all session folders sorted by date (newest first)"""
        try:
            return sorted([d.name for d in self.base_results_dir.iterdir() if d.is_dir()], reverse=True)
        except Exception:
            return []


# Convenience function for simple usage
def save_results(csv_data: Union[List[Dict[str, Any]], str, Path], 
                metadata: Optional[Dict[str, Any]] = None,
                results_dir: Optional[str] = None,
                custom_timestamp: Optional[str] = None) -> Path:
    """
    Convenience function to save results in one call
    
    Args:
        csv_data: CSV data in various formats (list of dicts, file path, or string)
        metadata: Optional metadata dictionary
        results_dir: Optional custom results directory
        custom_timestamp: Optional custom timestamp string
        
    Returns:
        Path to created session directory
        
    Example:
        from result_manager import save_results
        
        # Save list of results
        results = [{"question": "What is GDP?", "status": "approved"}]
        session_dir = save_results(results, {"test": "economics_batch"})
        
        # Save existing CSV file
        session_dir = save_results("/path/to/results.csv", {"source": "manual_test"})
    """
    rm = ResultManager(results_dir)
    return rm.save_results(csv_data, metadata, custom_timestamp)


# Example usage and testing
if __name__ == "__main__":
    print("Testing ResultManager...")
    
    # Test 1NutzenMathematischerDarstellungen: List of dictionaries
    test_results = [
        {"question": "What is economics?", "difficulty": "easy", "status": "approved", "score": 8.5},
        {"question": "Explain market equilibrium", "difficulty": "hard", "status": "needs_refinement", "score": 6.2}
    ]
    
    test_metadata = {
        "test_type": "automated_test",
        "model_used": "llama3.1NutzenMathematischerDarstellungen:8b",
        "total_questions": len(test_results),
        "processing_time": "45.2s"
    }
    
    # Using convenience function
    print("Test 1NutzenMathematischerDarstellungen: Using convenience function...")
    session_dir = save_results(test_results, test_metadata)
    print(f"Created session: {session_dir}")
    
    # Test 2: Using class directly
    print("\nTest 2: Using ResultManager class...")
    rm = ResultManager()
    session_dir2 = rm.save_results(test_results, {"test": "class_method"})
    print(f"Created session: {session_dir2}")
    
    # List all sessions
    print(f"\nAll sessions: {rm.list_sessions()}")
    print(f"Latest session: {rm.get_latest_session()}")
    
    print("\nResultManager test complete!")