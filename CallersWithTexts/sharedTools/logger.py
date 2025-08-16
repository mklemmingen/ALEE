#!/usr/bin/env python3
"""
Caller Logger Module - Centralized logging system with ISO8601 timestamps and analysis tools
Provides structured logging for all CallersWithTexts scripts and analysis capabilities
"""

import json
import logging
import statistics
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd


class CallerLogger:
    """Centralized logging system with ISO8601 timestamps and structured data"""
    
    def __init__(self, script_name: str, log_dir: str = "../_logs"):
        self.script_name = script_name
        self.log_dir = Path(__file__).parent / log_dir
        self.log_dir.mkdir(exist_ok=True)
        
        # Generate ISO8601 timestamp for log file
        self.session_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.log_file_path = self.log_dir / f"{self.session_timestamp}_{script_name}_log.jsonl"
        
        # Session tracking
        self.session_id = f"{script_name}_{self.session_timestamp}"
        self.log_entries = []
        
        # Set up structured logging
        self._setup_logging()
        
        # Log session start
        self.log_session_start()
    
    def _setup_logging(self):
        """Set up Python logging to work with our structured system"""
        # Create formatter for console output
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        root_logger.addHandler(console_handler)
    
    def _create_log_entry(self, level: str, message: str, **kwargs) -> Dict[str, Any]:
        """Create structured log entry"""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "script": self.script_name,
            "level": level,
            "message": message,
            **kwargs
        }
        return entry
    
    def _write_log_entry(self, entry: Dict[str, Any]):
        """Write log entry to JSONL file and memory"""
        self.log_entries.append(entry)
        
        # Write to file immediately for persistence
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    def log_session_start(self):
        """Log session start with metadata"""
        entry = self._create_log_entry(
            "INFO",
            f"Session started for {self.script_name}",
            event_type="session_start",
            log_file=str(self.log_file_path)
        )
        self._write_log_entry(entry)
        logging.info(f"[{self.session_id}] Session started - logging to {self.log_file_path}")
    
    def log_session_end(self, summary: Optional[Dict[str, Any]] = None):
        """Log session end with summary statistics"""
        entry = self._create_log_entry(
            "INFO",
            f"Session ended for {self.script_name}",
            event_type="session_end",
            summary=summary or {},
            total_log_entries=len(self.log_entries)
        )
        self._write_log_entry(entry)
        logging.info(f"[{self.session_id}] Session ended - {len(self.log_entries)} log entries")
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        entry = self._create_log_entry("INFO", message, **kwargs)
        self._write_log_entry(entry)
        logging.info(f"[{self.session_id}] {message}")
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        entry = self._create_log_entry("WARNING", message, **kwargs)
        self._write_log_entry(entry)
        logging.warning(f"[{self.session_id}] {message}")
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        entry = self._create_log_entry("ERROR", message, **kwargs)
        self._write_log_entry(entry)
        logging.error(f"[{self.session_id}] {message}")
    
    def success(self, message: str, **kwargs):
        """Log success message"""
        entry = self._create_log_entry("SUCCESS", message, **kwargs)
        self._write_log_entry(entry)
        logging.info(f"[{self.session_id}] ✅ {message}")
    
    def log_request_start(self, correlation_id: str, request_type: str, **kwargs):
        """Log start of a request"""
        entry = self._create_log_entry(
            "INFO",
            f"Starting {request_type} request",
            event_type="request_start",
            correlation_id=correlation_id,
            request_type=request_type,
            **kwargs
        )
        self._write_log_entry(entry)
        logging.info(f"[{correlation_id}] Starting {request_type} request")
    
    def log_request_end(self, correlation_id: str, request_type: str, 
                       success: bool, elapsed_time: float, **kwargs):
        """Log end of a request"""
        level = "SUCCESS" if success else "ERROR"
        status = "completed" if success else "failed"
        
        entry = self._create_log_entry(
            level,
            f"{request_type} request {status}",
            event_type="request_end",
            correlation_id=correlation_id,
            request_type=request_type,
            success=success,
            elapsed_time=elapsed_time,
            **kwargs
        )
        self._write_log_entry(entry)
        
        status_icon = "✅" if success else "❌"
        logging.info(f"[{correlation_id}] {status_icon} {request_type} {status} in {elapsed_time:.2f}s")
    
    def log_parameter_validation(self, correlation_id: str, is_valid: bool, 
                               errors: List[str] = None, **kwargs):
        """Log parameter validation results"""
        level = "INFO" if is_valid else "ERROR"
        message = "Parameter validation passed" if is_valid else f"Parameter validation failed: {errors}"
        
        entry = self._create_log_entry(
            level,
            message,
            event_type="parameter_validation",
            correlation_id=correlation_id,
            is_valid=is_valid,
            validation_errors=errors or [],
            **kwargs
        )
        self._write_log_entry(entry)
    
    def log_system_health(self, correlation_id: str, is_healthy: bool, 
                         health_data: Dict[str, Any], elapsed_time: float):
        """Log system health check results"""
        level = "SUCCESS" if is_healthy else "ERROR"
        message = "System health check passed" if is_healthy else "System health check failed"
        
        entry = self._create_log_entry(
            level,
            message,
            event_type="health_check",
            correlation_id=correlation_id,
            is_healthy=is_healthy,
            health_data=health_data,
            elapsed_time=elapsed_time
        )
        self._write_log_entry(entry)
    
    def log_question_generation(self, correlation_id: str, c_id: str, 
                              question_type: str, success: bool, 
                              result_data: Dict[str, Any], elapsed_time: float):
        """Log question generation attempt and results"""
        level = "SUCCESS" if success else "ERROR"
        message = f"Question generation for {c_id}"
        
        # Extract key metrics from result
        metrics = {}
        if success and 'csv_data' in result_data:
            csv_data = result_data['csv_data']
            metrics = {
                "questions_generated": sum(1 for i in range(1, 4) 
                                         if f'question_{i}' in result_data and result_data[f'question_{i}']),
                "processing_time": result_data.get('processing_time', 0),
                "dspy_consensus_used": csv_data.get('dspy_consensus_used', False),
                "dspy_expert_count": csv_data.get('dspy_expert_count', 0),
                "dspy_all_approved": csv_data.get('dspy_all_approved', False)
            }
        
        entry = self._create_log_entry(
            level,
            message,
            event_type="question_generation",
            correlation_id=correlation_id,
            c_id=c_id,
            question_type=question_type,
            success=success,
            elapsed_time=elapsed_time,
            metrics=metrics,
            result_summary=self._summarize_result(result_data) if success else {}
        )
        self._write_log_entry(entry)
    
    def _summarize_result(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of result data for logging"""
        summary = {
            "c_id": result_data.get('c_id'),
            "processing_time": result_data.get('processing_time'),
            "questions_present": [f'question_{i}' for i in range(1, 4) 
                                if f'question_{i}' in result_data and result_data[f'question_{i}']]
        }
        
        if 'csv_data' in result_data:
            csv_data = result_data['csv_data']
            summary["csv_fields"] = len(csv_data)
            summary["question_type"] = csv_data.get('type')
            summary["subject"] = csv_data.get('subject')
        
        return summary
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for current session"""
        if not self.log_entries:
            return {"message": "No log entries in session"}
        
        # Count by level
        level_counts = Counter(entry['level'] for entry in self.log_entries)
        
        # Count by event type
        event_counts = Counter(entry.get('event_type', 'unknown') for entry in self.log_entries)
        
        # Request statistics
        request_entries = [e for e in self.log_entries if e.get('event_type') == 'request_end']
        successful_requests = sum(1 for e in request_entries if e.get('success', False))
        total_requests = len(request_entries)
        
        # Timing statistics for requests
        timing_stats = {}
        if request_entries:
            elapsed_times = [e.get('elapsed_time', 0) for e in request_entries if e.get('elapsed_time')]
            if elapsed_times:
                timing_stats = {
                    "min_time": min(elapsed_times),
                    "max_time": max(elapsed_times),
                    "avg_time": statistics.mean(elapsed_times),
                    "median_time": statistics.median(elapsed_times)
                }
        
        return {
            "session_id": self.session_id,
            "script": self.script_name,
            "session_start": self.log_entries[0]['timestamp'] if self.log_entries else None,
            "session_end": self.log_entries[-1]['timestamp'] if self.log_entries else None,
            "total_log_entries": len(self.log_entries),
            "level_counts": dict(level_counts),
            "event_counts": dict(event_counts),
            "request_statistics": {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": total_requests - successful_requests,
                "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 0
            },
            "timing_statistics": timing_stats,
            "log_file": str(self.log_file_path)
        }


class LogAnalyzer:
    """Analysis tools for caller logs"""
    
    def __init__(self, log_dir: str = "_logs"):
        self.log_dir = Path(__file__).parent / log_dir
    
    def find_log_files(self, script_name: Optional[str] = None, 
                      start_date: Optional[str] = None) -> List[Path]:
        """Find log files matching criteria"""
        if not self.log_dir.exists():
            return []
        
        pattern = "*.jsonl"
        log_files = list(self.log_dir.glob(pattern))
        
        # Filter by script name if specified
        if script_name:
            log_files = [f for f in log_files if script_name in f.name]
        
        # Filter by date if specified (YYYYMMDD format)
        if start_date:
            log_files = [f for f in log_files if f.name.startswith(start_date)]
        
        return sorted(log_files)
    
    def load_log_entries(self, log_file: Path) -> List[Dict[str, Any]]:
        """Load log entries from JSONL file"""
        entries = []
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
        except Exception as e:
            logging.error(f"Failed to load log file {log_file}: {e}")
        
        return entries
    
    def analyze_session_performance(self, session_id: str) -> Dict[str, Any]:
        """Analyze performance metrics for a specific session"""
        # Find log file for session
        log_files = self.find_log_files()
        session_entries = []
        
        for log_file in log_files:
            entries = self.load_log_entries(log_file)
            session_entries.extend([e for e in entries if e.get('session_id') == session_id])
        
        if not session_entries:
            return {"error": f"No entries found for session {session_id}"}
        
        # Analyze request performance
        request_entries = [e for e in session_entries if e.get('event_type') == 'request_end']
        generation_entries = [e for e in session_entries if e.get('event_type') == 'question_generation']
        
        analysis = {
            "session_id": session_id,
            "total_entries": len(session_entries),
            "request_analysis": self._analyze_requests(request_entries),
            "generation_analysis": self._analyze_generations(generation_entries),
            "error_analysis": self._analyze_errors(session_entries),
            "timeline": self._create_timeline(session_entries)
        }
        
        return analysis
    
    def _analyze_requests(self, request_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze request performance data"""
        if not request_entries:
            return {"message": "No request data available"}
        
        successful = [e for e in request_entries if e.get('success', False)]
        failed = [e for e in request_entries if not e.get('success', False)]
        
        # Timing analysis
        timing_data = [e.get('elapsed_time', 0) for e in request_entries if e.get('elapsed_time')]
        timing_stats = {}
        if timing_data:
            timing_stats = {
                "count": len(timing_data),
                "min": min(timing_data),
                "max": max(timing_data),
                "avg": statistics.mean(timing_data),
                "median": statistics.median(timing_data),
                "std_dev": statistics.stdev(timing_data) if len(timing_data) > 1 else 0
            }
        
        # Request type breakdown
        request_types = Counter(e.get('request_type', 'unknown') for e in request_entries)
        
        return {
            "total_requests": len(request_entries),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(request_entries) * 100,
            "timing_statistics": timing_stats,
            "request_types": dict(request_types),
            "failure_reasons": [e.get('error', 'unknown') for e in failed]
        }
    
    def _analyze_generations(self, generation_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze question generation specific metrics"""
        if not generation_entries:
            return {"message": "No generation data available"}
        
        successful = [e for e in generation_entries if e.get('success', False)]
        
        # Question type analysis
        question_types = Counter(e.get('question_type', 'unknown') for e in generation_entries)
        
        # DSPy metrics analysis
        dspy_stats = {
            "consensus_used": sum(1 for e in successful if e.get('metrics', {}).get('dspy_consensus_used', False)),
            "expert_counts": [e.get('metrics', {}).get('dspy_expert_count', 0) for e in successful],
            "all_approved": sum(1 for e in successful if e.get('metrics', {}).get('dspy_all_approved', False))
        }
        
        # Processing time analysis (from DSPy, not network time)
        processing_times = [e.get('metrics', {}).get('processing_time', 0) for e in successful]
        processing_stats = {}
        if processing_times:
            processing_stats = {
                "min": min(processing_times),
                "max": max(processing_times),
                "avg": statistics.mean(processing_times),
                "median": statistics.median(processing_times)
            }
        
        return {
            "total_generations": len(generation_entries),
            "successful_generations": len(successful),
            "question_type_distribution": dict(question_types),
            "dspy_statistics": dspy_stats,
            "processing_time_stats": processing_stats
        }
    
    def _analyze_errors(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error patterns"""
        error_entries = [e for e in entries if e.get('level') == 'ERROR']
        
        if not error_entries:
            return {"message": "No errors found"}
        
        # Extract error patterns
        error_messages = [e.get('message', '') for e in error_entries]
        error_types = Counter()
        
        for msg in error_messages:
            if 'timeout' in msg.lower():
                error_types['timeout'] += 1
            elif 'connection' in msg.lower():
                error_types['connection'] += 1
            elif 'validation' in msg.lower():
                error_types['validation'] += 1
            elif 'http' in msg.lower():
                error_types['http_error'] += 1
            else:
                error_types['other'] += 1
        
        return {
            "total_errors": len(error_entries),
            "error_types": dict(error_types),
            "recent_errors": [e.get('message', '') for e in error_entries[-5:]]  # Last 5 errors
        }
    
    def _create_timeline(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create timeline of key events"""
        key_events = []
        
        for entry in entries:
            if entry.get('event_type') in ['session_start', 'session_end', 'request_end', 'question_generation']:
                key_events.append({
                    "timestamp": entry.get('timestamp'),
                    "event_type": entry.get('event_type'),
                    "message": entry.get('message'),
                    "success": entry.get('success'),
                    "correlation_id": entry.get('correlation_id')
                })
        
        # Sort by timestamp
        key_events.sort(key=lambda x: x.get('timestamp', ''))
        
        return key_events[-20:]  # Return last 20 key events
    
    def generate_summary_report(self, script_name: Optional[str] = None, 
                               days_back: int = 7) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        # Find relevant log files
        log_files = self.find_log_files(script_name=script_name)
        
        # Load all entries
        all_entries = []
        for log_file in log_files:
            entries = self.load_log_entries(log_file)
            # Filter by date if needed (basic filtering)
            all_entries.extend(entries)
        
        if not all_entries:
            return {"message": "No log data found"}
        
        # Group by session
        sessions = defaultdict(list)
        for entry in all_entries:
            session_id = entry.get('session_id', 'unknown')
            sessions[session_id].append(entry)
        
        # Analyze each session
        session_summaries = {}
        for session_id, entries in sessions.items():
            request_entries = [e for e in entries if e.get('event_type') == 'request_end']
            generation_entries = [e for e in entries if e.get('event_type') == 'question_generation']
            
            session_summaries[session_id] = {
                "total_entries": len(entries),
                "requests": len(request_entries),
                "successful_requests": sum(1 for e in request_entries if e.get('success')),
                "generations": len(generation_entries),
                "successful_generations": sum(1 for e in generation_entries if e.get('success')),
                "start_time": min(e.get('timestamp', '') for e in entries),
                "end_time": max(e.get('timestamp', '') for e in entries)
            }
        
        # Overall statistics
        total_requests = sum(s['requests'] for s in session_summaries.values())
        total_successful = sum(s['successful_requests'] for s in session_summaries.values())
        total_generations = sum(s['generations'] for s in session_summaries.values())
        successful_generations = sum(s['successful_generations'] for s in session_summaries.values())
        
        return {
            "report_generated": datetime.now(timezone.utc).isoformat(),
            "script_filter": script_name,
            "total_sessions": len(sessions),
            "total_log_entries": len(all_entries),
            "overall_statistics": {
                "total_requests": total_requests,
                "successful_requests": total_successful,
                "request_success_rate": (total_successful / total_requests * 100) if total_requests > 0 else 0,
                "total_generations": total_generations,
                "successful_generations": successful_generations,
                "generation_success_rate": (successful_generations / total_generations * 100) if total_generations > 0 else 0
            },
            "session_summaries": session_summaries,
            "log_files_analyzed": [str(f) for f in log_files]
        }
    
    def export_to_csv(self, output_path: str, script_name: Optional[str] = None) -> bool:
        """Export log analysis to CSV format"""
        try:
            log_files = self.find_log_files(script_name=script_name)
            all_entries = []
            
            for log_file in log_files:
                entries = self.load_log_entries(log_file)
                all_entries.extend(entries)
            
            if not all_entries:
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(all_entries)
            df.to_csv(output_path, index=False)
            
            logging.info(f"Exported {len(all_entries)} log entries to {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to export to CSV: {e}")
            return False


# Export main classes
__all__ = ['CallerLogger', 'LogAnalyzer']