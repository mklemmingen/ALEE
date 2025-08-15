#!/usr/bin/env python3
"""
Single Request Test - Tests DSPy system with one random text and fixed realistic parameters
Based on stakeholder_test_system.py but runs only one request for quick validation
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional

import aiohttp
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SingleRequestTest:
    """Single request test for DSPy educational question generator"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.stakeholder_csv_path = "../.dev/providedProjectFromStakeHolder/explanation_metadata.csv"
        self.timeout_seconds = 180  # Timeout for single request (increased for DSPy processing)
        
    async def test_health_endpoint(self) -> bool:
        """Test DSPy health endpoint"""
        logger.info("Testing DSPy health endpoint...")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/system-health") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"DSPy health check: {data}")
                        return data.get('dspy_ready', False)
                    else:
                        logger.error(f"Health check failed: HTTP {response.status}")
                        return False
            except Exception as e:
                logger.error(f"Health check error: {e}")
                return False
    
    def load_random_text(self) -> Optional[Dict[str, str]]:
        """Load a random text from explanation_metadata.csv"""
        try:
            df = pd.read_csv(self.stakeholder_csv_path)
            logger.info(f"Loaded {len(df)} texts from CSV")
            
            # Select random row
            random_row = df.sample(n=1).iloc[0]
            
            return {
                'c_id': random_row['c_id'],
                'subject': random_row['subject'],
                'text': random_row['text']
            }
        except Exception as e:
            logger.error(f"Failed to load stakeholder CSV: {e}")
            return None
    
    def create_fixed_realistic_parameters(self, text_data: Dict[str, str]) -> Dict[str, Any]:
        """Create fixed realistic parameters for testing"""
        
        # Fixed realistic parameters for a standard multiple-choice question
        parameters = {
            "c_id": f"{text_data['c_id']}-test",
            "text": text_data['text'],
            "question_type": "multiple-choice",  # Fixed question format
            "p_variation": "stammaufgabe",       # Standard difficulty
            "p_taxonomy_level": "Stufe 1 (Wissen/Reproduktion)",  # Knowledge level
            "p_mathematical_requirement_level": "0",  # No math requirement
            "p_root_text_reference_explanatory_text": "Nicht vorhanden",
            "p_root_text_obstacle_passive": "Nicht Enthalten",
            "p_root_text_obstacle_negation": "Nicht Enthalten",
            "p_root_text_obstacle_complex_np": "Nicht Enthalten",
            "p_root_text_contains_irrelevant_information": "Nicht Enthalten",
        }
        
        # Add all 8 item parameters (all set to "Nicht Enthalten" for simplicity)
        for i in range(1, 9):
            parameters[f"p_item_{i}_obstacle_passive"] = "Nicht Enthalten"
            parameters[f"p_item_{i}_obstacle_negation"] = "Nicht Enthalten"
            parameters[f"p_item_{i}_obstacle_complex_np"] = "Nicht Enthalten"
        
        # Instruction parameters
        parameters["p_instruction_obstacle_passive"] = "Nicht Enthalten"
        parameters["p_instruction_obstacle_negation"] = "Nicht Enthalten"
        parameters["p_instruction_obstacle_complex_np"] = "Nicht Enthalten"
        parameters["p_instruction_explicitness_of_instruction"] = "Explizit"  # Explicit instructions
        
        return parameters
    
    async def execute_single_request(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute single request to DSPy endpoint"""
        start_time = time.time()
        
        logger.info(f"Sending request to DSPy endpoint...")
        logger.info(f"Request parameters: {json.dumps(request_data, indent=2, ensure_ascii=False)}")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/generate-educational-questions",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
                ) as response:
                    
                    processing_time = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ Request successful in {processing_time:.2f}s")
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Request failed: HTTP {response.status}")
                        logger.error(f"Error: {error_text}")
                        return None
                        
            except asyncio.TimeoutError:
                logger.error(f"❌ Request timeout after {self.timeout_seconds}s")
                return None
            except Exception as e:
                logger.error(f"❌ Request exception: {e}")
                return None
    
    def validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate the result structure and content"""
        logger.info("\n" + "="*60)
        logger.info("VALIDATING RESULT")
        logger.info("="*60)
        
        validations = []
        
        # Check required fields
        required_fields = ['question_1', 'question_2', 'question_3', 'c_id', 'processing_time', 'csv_data']
        for field in required_fields:
            if field in result:
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
            logger.info(validation)
            
        # Calculate success
        success_count = sum(1 for v in validations if v.startswith("✅"))
        total_count = len(validations)
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        
        logger.info(f"\nValidation Summary: {success_count}/{total_count} passed ({success_rate:.1f}%)")
        
        # Log generated questions
        if all(f'question_{i}' in result for i in range(1, 4)):
            logger.info("\n" + "="*60)
            logger.info("GENERATED QUESTIONS")
            logger.info("="*60)
            for i in range(1, 4):
                logger.info(f"\nQuestion {i}:")
                logger.info(result[f'question_{i}'])
                
        # Log processing metadata
        if 'generation_updates' in result and result['generation_updates']:
            update = result['generation_updates'][0]
            logger.info("\n" + "="*60)
            logger.info("DSPY PROCESSING METADATA")
            logger.info("="*60)
            logger.info(f"Single-pass consensus: {update.get('dspy_metadata', {}).get('single_pass_consensus', False)}")
            logger.info(f"Modular prompts used: {update.get('dspy_metadata', {}).get('modular_prompts_used', False)}")
            logger.info(f"All approved: {update.get('dspy_metadata', {}).get('all_approved', False)}")
            logger.info(f"Expert consensus: {update.get('expert_consensus', [])}")
        
        return success_count == total_count
    
    async def run_single_test(self) -> bool:
        """Run single request test with random text and fixed parameters"""
        logger.info("="*60)
        logger.info("SINGLE REQUEST TEST - DSPy EDUCATIONAL QUESTION GENERATOR")
        logger.info("="*60)
        
        # Check health first
        if not await self.test_health_endpoint():
            logger.error("DSPy health check failed - aborting test")
            return False
            
        # Load random text
        text_data = self.load_random_text()
        if not text_data:
            logger.error("Failed to load text data")
            return False
            
        logger.info(f"\nSelected text: {text_data['c_id']} - {text_data['subject']}")
        logger.info(f"Text preview: {text_data['text'][:200]}...")
        
        # Create request parameters
        request_data = self.create_fixed_realistic_parameters(text_data)
        
        # Execute request
        result = await self.execute_single_request(request_data)
        
        if not result:
            logger.error("Request failed - no result received")
            return False
            
        # Validate result
        validation_passed = self.validate_result(result)
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Test Status: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
        logger.info(f"Processing Time: {result.get('processing_time', 0):.2f}s")
        logger.info(f"C_ID: {result.get('c_id', 'N/A')}")
        
        if 'csv_data' in result:
            logger.info(f"Expert Count: {result['csv_data'].get('dspy_expert_count', 0)}")
            logger.info(f"All Approved: {result['csv_data'].get('dspy_all_approved', False)}")
            
        return validation_passed

async def main():
    """Main test function"""
    tester = SingleRequestTest()
    
    try:
        # First start the DSPy server
        logger.info("Starting DSPy server...")
        logger.info("Please ensure the DSPy orchestrator is running on port 8000")
        logger.info("Run: python3 ALEE_Agent/_server_question_generator.py")
        
        # Give user time to start server if needed
        await asyncio.sleep(2)
        
        # Run the test
        success = await tester.run_single_test()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)