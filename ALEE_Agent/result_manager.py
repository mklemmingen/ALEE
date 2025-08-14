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
    """Modular result manager for educational AI system results"""
    
    def __init__(self, base_results_dir: str = None):
        if base_results_dir is None:
            # Default to results directory in same folder as this script
            self.base_results_dir = Path(__file__).parent / "results"
        else:
            self.base_results_dir = Path(base_results_dir)
        self.base_results_dir.mkdir(exist_ok=True)
        
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
            "prompts_saved": len(list((session_dir / "prompts").glob("*.txt"))) if (session_dir / "prompts").exists() else 0
        })
        self._save_metadata(session_dir, metadata)
        
        return session_dir
        
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
        """Save snapshot of all ALEE_Agent prompts"""
        if prompts_source_dir is None:
            # Default: relative path to ALEE_Agent/prompts from CallersWithTexts
            prompts_source_dir = Path(__file__).parent.parent / "ALEE_Agent" / "prompts"
        else:
            prompts_source_dir = Path(prompts_source_dir)
            
        prompts_dest = session_dir / "prompts"
        prompts_dest.mkdir(exist_ok=True)
        
        if prompts_source_dir.exists():
            # Copy all prompt files
            for prompt_file in prompts_source_dir.glob("*.txt"):
                try:
                    shutil.copy2(prompt_file, prompts_dest / prompt_file.name)
                except Exception as e:
                    print(f"Warning: Failed to copy {prompt_file.name}: {e}")
        else:
            print(f"Warning: Prompts directory not found: {prompts_source_dir}")
            
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