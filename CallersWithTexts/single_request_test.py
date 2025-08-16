#!/usr/bin/env python3
"""
Single Request Test - Enhanced with Modular Components
Tests DSPy system with one random text and fixed realistic parameters
Includes structured logging, parameter validation, and performance analysis
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional

# Import modular components
from sharedTools.caller import OrchestorCaller
from sharedTools.logger import CallerLogger
from sharedTools.config import CallerConfig, TestConfig

class SingleRequestTest:
    """Enhanced single request test with modular components"""
    
    def __init__(self):
        # Initialize modular components
        self.caller = OrchestorCaller(
            base_url=CallerConfig.get_base_url(),
            timeout_seconds=CallerConfig.get_timeout()
        )
        self.logger = CallerLogger("single_request_test")
        self.stakeholder_csv_path = CallerConfig.get_stakeholder_csv_path()
        
        # Log initialization
        self.logger.info("Initialized SingleRequestTest",
                        base_url=self.caller.base_url,
                        timeout=self.caller.timeout_seconds,
                        csv_path=self.stakeholder_csv_path)
        
    async def test_health_endpoint(self) -> bool:
        """Test DSPy health endpoint using modular caller"""
        correlation_id = self.caller._get_correlation_id()
        self.logger.log_request_start(correlation_id, "health_check")
        
        start_time = time.time()
        is_healthy, health_data = await self.caller.test_health_endpoint(correlation_id)
        elapsed_time = time.time() - start_time
        
        self.logger.log_system_health(correlation_id, is_healthy, health_data, elapsed_time)
        self.logger.log_request_end(correlation_id, "health_check", is_healthy, elapsed_time)
        
        return is_healthy
    
    def load_random_text(self) -> Optional[Dict[str, str]]:
        """Load a random text using modular caller"""
        self.logger.info("Loading random text from stakeholder CSV")
        
        df = self.caller.load_stakeholder_texts(self.stakeholder_csv_path)
        if df is None:
            self.logger.error("Failed to load stakeholder texts")
            return None
        
        # Select random row
        random_row = df.sample(n=1).iloc[0]
        
        text_data = {
            'c_id': random_row['c_id'],
            'subject': random_row['subject'],
            'text': random_row['text']
        }
        
        self.logger.info("Selected random text",
                        c_id=text_data['c_id'],
                        subject=text_data['subject'],
                        text_length=len(text_data['text']))
        
        return text_data
    
    def create_fixed_realistic_parameters(self, text_data: Dict[str, str]) -> Dict[str, Any]:
        """Create fixed realistic parameters using CallerConfig defaults"""
        
        # Start with default parameters from config
        parameters = CallerConfig.get_default_parameters().copy()
        
        # Override with test-specific values
        parameters.update({
            "c_id": f"{text_data['c_id']}-test",
            "text": text_data['text']
        })
        
        self.logger.info("Created test parameters",
                        total_params=len(parameters),
                        c_id=parameters['c_id'],
                        question_type=parameters['question_type'],
                        difficulty=parameters['p_variation'])
        
        return parameters
    
    async def execute_single_request(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute single request using modular caller"""
        correlation_id = self.caller._get_correlation_id()
        c_id = request_data.get('c_id', 'unknown')
        question_type = request_data.get('question_type', 'unknown')
        
        self.logger.info("Sending request to DSPy endpoint",
                        c_id=c_id,
                        question_type=question_type,
                        total_params=len(request_data))
        
        self.logger.log_request_start(correlation_id, "single_test_generation",
                                     c_id=c_id, question_type=question_type)
        
        start_time = time.time()
        success, result = await self.caller.generate_questions(request_data, correlation_id)
        elapsed_time = time.time() - start_time
        
        self.logger.log_question_generation(correlation_id, c_id, question_type,
                                           success, result, elapsed_time)
        self.logger.log_request_end(correlation_id, "single_test_generation", 
                                   success, elapsed_time, c_id=c_id)
        
        if success:
            self.logger.success(f"Request successful in {elapsed_time:.2f}s")
            return result
        else:
            error_details = result.get('error', 'Unknown error')
            self.logger.error(f"Request failed: {error_details}")
            return None
    
    def validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate the result structure and content with enhanced logging"""
        correlation_id = self.caller._get_correlation_id()
        
        self.logger.info("Starting result validation", 
                        event_type="validation_start",
                        correlation_id=correlation_id)
        
        validations = []
        validation_results = {}
        
        # Check required fields using config
        endpoint_config = CallerConfig.get_endpoint_config()['generate']
        required_fields = endpoint_config['expected_fields'] + ['csv_data']
        
        for field in required_fields:
            is_present = field in result
            validation_results[f"field_{field}"] = is_present
            if is_present:
                validations.append(f"✅ Field '{field}' present")
            else:
                validations.append(f"❌ Field '{field}' missing")
                
        # Check questions are not empty
        for i in range(1, 4):
            q_field = f'question_{i}'
            if q_field in result and result[q_field] and len(result[q_field]) > 10:
                validations.append(f"✅ {q_field} has content ({len(result[q_field])} chars)")
            else:
                validations.append(f"❌ {q_field} is empty or too short")
        
        # Check CSV data structure
        if 'csv_data' in result:
            csv_data = result['csv_data']
            csv_required = ['c_id', 'subject', 'type', 'text', 'answers']
            for field in csv_required:
                if field in csv_data:
                    validations.append(f"✅ CSV field '{field}' present")
                else:
                    validations.append(f"❌ CSV field '{field}' missing")
                    
            # Check DSPy-specific metadata
            if csv_data.get('dspy_consensus_used'):
                validations.append("✅ DSPy consensus architecture used")
            else:
                validations.append("❌ DSPy consensus not indicated")
                
            if csv_data.get('dspy_all_approved'):
                validations.append("✅ All questions approved by experts")
            else:
                validations.append("⚠️ Not all questions approved by experts")
        
        # Log all validations
        for validation in validations:
            if "✅" in validation:
                self.logger.info(validation)
            else:
                self.logger.error(validation)
            
        # Calculate success
        success_count = sum(1 for v in validations if v.startswith("✅"))
        total_count = len(validations)
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        
        # Log validation summary
        validation_summary = {
            "passed_validations": success_count,
            "total_validations": total_count,
            "success_rate": success_rate,
            "validation_details": validation_results
        }
        
        self.logger.info(f"Validation Summary: {success_count}/{total_count} passed ({success_rate:.1f}%)",
                        event_type="validation_summary",
                        correlation_id=correlation_id,
                        validation_summary=validation_summary)
        
        # Log generated questions
        if all(f'question_{i}' in result for i in range(1, 4)):
            questions_data = {}
            for i in range(1, 4):
                question_text = result[f'question_{i}']
                questions_data[f"question_{i}"] = {
                    "text": question_text,
                    "length": len(question_text),
                    "has_content": bool(question_text and len(question_text) > 10)
                }
                self.logger.info(f"Question {i}: {question_text[:100]}...")
            
            self.logger.info("Generated questions logged",
                           event_type="questions_generated",
                           correlation_id=correlation_id,
                           questions_summary=questions_data)
                
        # Log processing metadata
        if 'generation_updates' in result and result['generation_updates']:
            update = result['generation_updates'][0]
            dspy_metadata = update.get('dspy_metadata', {})
            
            processing_metadata = {
                "single_pass_consensus": dspy_metadata.get('single_pass_consensus', False),
                "modular_prompts_used": dspy_metadata.get('modular_prompts_used', False),
                "all_approved": dspy_metadata.get('all_approved', False),
                "expert_consensus": update.get('expert_consensus', [])
            }
            
            self.logger.info("DSPy processing metadata",
                           event_type="dspy_metadata",
                           correlation_id=correlation_id,
                           processing_metadata=processing_metadata)
        
        return success_count == total_count
    
    async def run_single_test(self) -> bool:
        """Run single request test with enhanced logging and validation"""
        self.logger.info("Starting single request test",
                        event_type="test_start")
        
        # Check health first
        if not await self.test_health_endpoint():
            self.logger.error("DSPy health check failed - aborting test",
                            event_type="test_abort")
            return False
            
        # Load random text
        text_data = self.load_random_text()
        if not text_data:
            self.logger.error("Failed to load text data", event_type="test_abort")
            return False
        
        # Create request parameters
        request_data = self.create_fixed_realistic_parameters(text_data)
        
        # Execute request
        result = await self.execute_single_request(request_data)
        
        if not result:
            self.logger.error("Request failed - no result received",
                            event_type="request_failed")
            return False
            
        # Validate result
        validation_passed = self.validate_result(result)
        
        # Final summary
        test_summary = {
            "test_passed": validation_passed,
            "processing_time": result.get('processing_time', 0),
            "c_id": result.get('c_id', 'N/A'),
            "text_c_id": text_data['c_id'],
            "subject": text_data['subject']
        }
        
        if 'csv_data' in result:
            csv_data = result['csv_data']
            test_summary.update({
                "expert_count": csv_data.get('dspy_expert_count', 0),
                "all_approved": csv_data.get('dspy_all_approved', False),
                "consensus_used": csv_data.get('dspy_consensus_used', False)
            })
        
        status = "✅ PASSED" if validation_passed else "❌ FAILED"
        self.logger.info(f"Test Status: {status}",
                        event_type="test_complete",
                        test_summary=test_summary)
            
        return validation_passed

async def main():
    """Main test function with enhanced error handling"""
    tester = None
    
    try:
        tester = SingleRequestTest()
        
        # Run the test
        success = await tester.run_single_test()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        if tester:
            tester.logger.warning("Test interrupted by user", event_type="user_interrupt")
        return 1
    except Exception as e:
        if tester:
            tester.logger.error("Test failed with exception", error=str(e), event_type="system_error")
        else:
            print(f"Failed to initialize test system: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if tester:
            # Ensure session is properly closed
            session_summary = tester.logger.get_session_summary()
            tester.logger.log_session_end(session_summary)
            print(f"\nSession completed. Log file: {session_summary.get('log_file')}")
            print(f"Total log entries: {session_summary.get('total_log_entries', 0)}")

if __name__ == "__main__":
    import sys
    
    print("=" * 80)
    print("SINGLE REQUEST TEST - Enhanced with Modular Components")
    print("=" * 80)
    print("Features: Parameter validation, structured logging, comprehensive result analysis")
    print("Log files saved to CallersWithTexts/_logs/ with ISO8601 timestamps")
    print("=" * 80)
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)