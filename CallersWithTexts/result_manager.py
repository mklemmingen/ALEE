"""
Result Management System for Educational AI System
Manages timestamped result folders and saves prompts + CSV results
"""
import os
import shutil
from datetime import datetime
from pathlib import Path
import csv


class ResultManager:
    def __init__(self, base_results_dir: str = "./results"):
        self.base_results_dir = Path(base_results_dir)
        self.base_results_dir.mkdir(exist_ok=True)
        
    def create_timestamp_folder(self) -> Path:
        """Create a new timestamped folder for this session"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_dir = self.base_results_dir / timestamp
        session_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (session_dir / "prompts").mkdir(exist_ok=True)
        (session_dir / "results").mkdir(exist_ok=True)
        
        return session_dir
        
    def save_prompts_snapshot(self, session_dir: Path, prompts_source_dir: str = "../ALEE_Agent/prompts"):
        """Save a snapshot of all prompts used for this session"""
        prompts_source = Path(prompts_source_dir)
        prompts_dest = session_dir / "prompts"
        
        if prompts_source.exists():
            # Copy all prompt files
            for prompt_file in prompts_source.glob("*.txt"):
                shutil.copy2(prompt_file, prompts_dest)
                
    def save_results_csv(self, session_dir: Path, results_data: list, filename: str = "parsed_results.csv"):
        """Save results data as CSV"""
        csv_path = session_dir / "results" / filename
        
        if results_data:
            # Assuming results_data is a list of dictionaries
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                if isinstance(results_data[0], dict):
                    fieldnames = results_data[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results_data)
                else:
                    # Simple list of rows
                    writer = csv.writer(csvfile)
                    writer.writerows(results_data)
                    
        return csv_path
        
    def save_session_metadata(self, session_dir: Path, metadata: dict):
        """Save metadata about this session"""
        metadata_path = session_dir / "session_metadata.json"
        import json
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
            
    def get_latest_session(self) -> Path:
        """Get the most recent session folder"""
        session_folders = [d for d in self.base_results_dir.iterdir() if d.is_dir()]
        if session_folders:
            return max(session_folders, key=lambda x: x.stat().st_mtime)
        return None
        
    def list_sessions(self) -> list:
        """List all session folders"""
        return sorted([d.name for d in self.base_results_dir.iterdir() if d.is_dir()], reverse=True)


# Example usage
if __name__ == "__main__":
    # Example of how to use the ResultManager
    rm = ResultManager()
    
    # Create a new session
    session = rm.create_timestamp_folder()
    print(f"Created session: {session}")
    
    # Save prompts snapshot
    rm.save_prompts_snapshot(session)
    
    # Example results data
    example_results = [
        {"question": "What is economics?", "difficulty": "easy", "parameters": {"variation": "leicht"}},
        {"question": "Explain market equilibrium", "difficulty": "hard", "parameters": {"variation": "schwer"}}
    ]
    
    # Save results
    csv_path = rm.save_results_csv(session, example_results)
    print(f"Saved results to: {csv_path}")
    
    # Save session metadata
    metadata = {
        "timestamp": datetime.now(),
        "model_used": "llama3.1:8b",
        "total_questions": len(example_results),
        "processing_time": "45.2s"
    }
    rm.save_session_metadata(session, metadata)