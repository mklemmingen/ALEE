#!/usr/bin/env python3
"""
Orchestrator Caller Module - Centralized communication with DSPy orchestrator
Provides standardized methods for health checks, question generation, and system queries
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)


class OrchestorCaller:
    """Centralized orchestrator communication with standardized error handling and logging"""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout_seconds: int = 300):
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        # Request correlation tracking
        self.request_counter = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
    def _get_correlation_id(self) -> str:
        """Generate unique correlation ID for request tracking"""
        self.request_counter += 1
        return f"{self.session_id}_{self.request_counter:04d}"
    
    async def test_health_endpoint(self, correlation_id: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Test DSPy system health endpoint
        
        Returns:
            Tuple[bool, Dict]: (is_healthy, health_data)
        """
        if not correlation_id:
            correlation_id = self._get_correlation_id()
            
        logger.info(f"[{correlation_id}] Testing DSPy health endpoint: {self.base_url}/system-health")
        
        self.total_requests += 1
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.base_url}/system-health",
                    timeout=aiohttp.ClientTimeout(total=30)  # Short timeout for health checks
                ) as response:
                    elapsed = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        is_healthy = data.get('dspy_ready', False)
                        
                        logger.info(f"[{correlation_id}] Health check passed in {elapsed:.2f}s: "
                                  f"status={data.get('status', 'unknown')}, dspy_ready={is_healthy}")
                        
                        self.successful_requests += 1
                        return is_healthy, data
                    else:
                        logger.error(f"[{correlation_id}] Health check failed: HTTP {response.status} in {elapsed:.2f}s")
                        self.failed_requests += 1
                        return False, {"error": f"HTTP {response.status}", "elapsed_time": elapsed}
                        
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                logger.error(f"[{correlation_id}] Health check timeout after {elapsed:.2f}s")
                self.failed_requests += 1
                return False, {"error": "timeout", "elapsed_time": elapsed}
                
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"[{correlation_id}] Health check error: {e} (elapsed: {elapsed:.2f}s)")
                self.failed_requests += 1
                return False, {"error": str(e), "elapsed_time": elapsed}
    
    async def get_system_info(self, correlation_id: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Get system capabilities and configuration
        
        Returns:
            Tuple[bool, Dict]: (success, system_info)
        """
        if not correlation_id:
            correlation_id = self._get_correlation_id()
            
        logger.info(f"[{correlation_id}] Getting system info: {self.base_url}/system-capabilities")
        
        self.total_requests += 1
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.base_url}/system-capabilities",
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    elapsed = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"[{correlation_id}] System info retrieved in {elapsed:.2f}s")
                        self.successful_requests += 1
                        return True, data
                    else:
                        logger.error(f"[{correlation_id}] System info failed: HTTP {response.status} in {elapsed:.2f}s")
                        self.failed_requests += 1
                        return False, {"error": f"HTTP {response.status}", "elapsed_time": elapsed}
                        
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"[{correlation_id}] System info error: {e}")
                self.failed_requests += 1
                return False, {"error": str(e), "elapsed_time": elapsed}
    
    def validate_request_parameters(self, request_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate request parameters for completeness and correctness
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []
        
        # Required fields
        required_fields = ["c_id", "text", "question_type"]
        for field in required_fields:
            if field not in request_data or not request_data[field]:
                errors.append(f"Missing required field: {field}")
        
        # Validate c_id format
        if "c_id" in request_data:
            c_id = request_data["c_id"]
            if not isinstance(c_id, str) or len(c_id.split('-')) < 2:
                errors.append(f"Invalid c_id format: {c_id} (expected format: number-number-number)")
        
        # Validate question_type
        valid_question_types = ["multiple-choice", "single-choice", "true-false", "mapping"]
        if "question_type" in request_data:
            if request_data["question_type"] not in valid_question_types:
                errors.append(f"Invalid question_type: {request_data['question_type']} "
                            f"(must be one of: {', '.join(valid_question_types)})")
        
        # Validate text length
        if "text" in request_data:
            text = request_data["text"]
            if not isinstance(text, str) or len(text.strip()) < 10:
                errors.append("Text must be at least 10 characters long")
            elif len(text) > 10000:
                errors.append("Text should be under 10,000 characters for optimal processing")
        
        # Validate variation if present
        if "p_variation" in request_data:
            valid_variations = ["stammaufgabe", "schwer", "leicht"]
            if request_data["p_variation"] not in valid_variations:
                errors.append(f"Invalid p_variation: {request_data['p_variation']} "
                            f"(must be one of: {', '.join(valid_variations)})")
        
        # Validate mathematical requirement level
        if "p_mathematical_requirement_level" in request_data:
            valid_math_levels = ["0", "1", "2"]
            if str(request_data["p_mathematical_requirement_level"]) not in valid_math_levels:
                errors.append(f"Invalid p_mathematical_requirement_level: {request_data['p_mathematical_requirement_level']} "
                            f"(must be one of: {', '.join(valid_math_levels)})")
        
        return len(errors) == 0, errors
    
    async def generate_questions(self, request_data: Dict[str, Any], 
                               correlation_id: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Generate educational questions via DSPy orchestrator
        
        Args:
            request_data: Complete request parameters including all SYSARCH fields
            correlation_id: Optional correlation ID for tracking
            
        Returns:
            Tuple[bool, Dict]: (success, result_data)
        """
        if not correlation_id:
            correlation_id = self._get_correlation_id()
        
        # Validate parameters first
        is_valid, validation_errors = self.validate_request_parameters(request_data)
        if not is_valid:
            logger.error(f"[{correlation_id}] Request validation failed: {validation_errors}")
            return False, {"error": "validation_failed", "details": validation_errors}
        
        c_id = request_data.get('c_id', 'unknown')
        question_type = request_data.get('question_type', 'unknown')
        text_length = len(request_data.get('text', ''))
        
        logger.info(f"[{correlation_id}] Starting question generation for {c_id}")
        logger.info(f"[{correlation_id}] Parameters: type={question_type}, text_length={text_length}")
        
        self.total_requests += 1
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/generate-educational-questions",
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
                ) as response:
                    
                    elapsed = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Basic result validation
                        questions_generated = sum(1 for i in range(1, 4) 
                                                if f'question_{i}' in result and result[f'question_{i}'])
                        
                        logger.info(f"[{correlation_id}] ✅ Question generation successful for {c_id}")
                        logger.info(f"[{correlation_id}] Generated {questions_generated}/3 questions in {elapsed:.2f}s")
                        logger.info(f"[{correlation_id}] Processing time: {result.get('processing_time', 0):.2f}s")
                        
                        # Log DSPy metadata if available
                        if 'csv_data' in result and result['csv_data']:
                            csv_data = result['csv_data']
                            logger.info(f"[{correlation_id}] DSPy consensus: {csv_data.get('dspy_consensus_used', False)}")
                            logger.info(f"[{correlation_id}] Expert count: {csv_data.get('dspy_expert_count', 0)}")
                            logger.info(f"[{correlation_id}] All approved: {csv_data.get('dspy_all_approved', False)}")
                        
                        self.successful_requests += 1
                        return True, result
                        
                    else:
                        error_text = await response.text()
                        logger.error(f"[{correlation_id}] ❌ Question generation failed for {c_id}: "
                                   f"HTTP {response.status} in {elapsed:.2f}s")
                        logger.error(f"[{correlation_id}] Error details: {error_text}")
                        
                        self.failed_requests += 1
                        return False, {
                            "error": f"HTTP {response.status}",
                            "details": error_text,
                            "elapsed_time": elapsed
                        }
                        
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                logger.error(f"[{correlation_id}] ❌ Question generation timeout for {c_id} after {elapsed:.2f}s")
                self.failed_requests += 1
                return False, {"error": "timeout", "elapsed_time": elapsed}
                
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"[{correlation_id}] ❌ Question generation exception for {c_id}: {e}")
                self.failed_requests += 1
                return False, {"error": str(e), "elapsed_time": elapsed}
    
    def load_stakeholder_texts(self, csv_path: str) -> Optional[pd.DataFrame]:
        """
        Load stakeholder texts from CSV file with error handling
        
        Args:
            csv_path: Path to explanation_metadata.csv or similar
            
        Returns:
            DataFrame or None if loading fails
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} texts from {csv_path}")
            logger.info(f"CSV columns: {list(df.columns)}")
            
            # Basic validation
            required_columns = ['c_id', 'text']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing required columns in CSV: {missing_columns}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load stakeholder CSV from {csv_path}: {e}")
            return None
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get current session statistics"""
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            "session_id": self.session_id,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate_percent": success_rate,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds
        }
    
    def reset_statistics(self):
        """Reset request statistics for new test session"""
        self.request_counter = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        logger.info(f"Reset statistics for new session: {self.session_id}")


# Convenience functions for backwards compatibility
async def test_health(base_url: str = "http://localhost:8000") -> bool:
    """Simple health check function for backwards compatibility"""
    caller = OrchestorCaller(base_url=base_url)
    is_healthy, _ = await caller.test_health_endpoint()
    return is_healthy


async def generate_questions_simple(request_data: Dict[str, Any], 
                                  base_url: str = "http://localhost:8000",
                                  timeout: int = 300) -> Optional[Dict[str, Any]]:
    """Simple question generation function for backwards compatibility"""
    caller = OrchestorCaller(base_url=base_url, timeout_seconds=timeout)
    success, result = await caller.generate_questions(request_data)
    return result if success else None


# Export main classes and functions
__all__ = ['OrchestorCaller', 'test_health', 'generate_questions_simple']