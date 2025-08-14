#!/usr/bin/env python3
"""
ValidationPlan Stakeholder Testing System - Pure Caller Mode
Uses all rows from explanation_metadata.csv to make HTTP requests with randomized parameters
Orchestrator handles all result saving - caller only monitors process completion
"""

import asyncio
import logging
import random
import time
from typing import Dict, Any

import aiohttp
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StakeholderTestSystem:
    """Pure caller for ValidationPlan testing using stakeholder explanation_metadata.csv"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.stakeholder_csv_path = "../.dev/providedProjectFromStakeHolder/explanation_metadata.csv" # confidential path, please change if needed to a csv of your choice
        # No result storage - orchestrator handles everything
        
    async def test_health_endpoint(self):
        """Test system health endpoint"""
        logger.info("Testing health endpoint...")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Health check passed: {data['status']}")
                        return True
                    else:
                        logger.error(f"Health check failed: {response.status}")
                        return False
            except Exception as e:
                logger.error(f"Health check error: {e}")
                return False
    
    def load_stakeholder_texts(self) -> pd.DataFrame:
        """Load all texts from explanation_metadata.csv"""
        try:
            df = pd.read_csv(self.stakeholder_csv_path)
            logger.info(f"Loaded {len(df)} texts from explanation_metadata.csv")
            logger.info(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Failed to load stakeholder CSV: {e}")
            raise
    
    @staticmethod
    def generate_randomized_parameters(base_c_id: str, text_content: str, run_num: int) -> Dict[str, Any]:
        """Generate randomized ValidationPlan parameters for testing"""
        
        # Create new c_id with run number
        c_id = f"{base_c_id.split('-')[0]}-{random.choice(['1', '2', '3'])}-{run_num}"
        
        # Randomize all parameters according to ValidationPlan specifications
        variations = ["stammaufgabe", "schwer", "leicht"]
        taxonomy_levels = ["Stufe 1 (Wissen/Reproduktion)", "Stufe 2 (Anwendung/Transfer)"]
        math_levels = ["0", "1", "2"]
        binary_values = ["Enthalten", "Nicht Enthalten"]
        reference_values = ["Nicht vorhanden", "Explizit", "Implizit"]
        explicit_values = ["Explizit", "Implizit"]
        
        # Generate complete parameter set with all individual item parameters (1-8) as per SYSARCH.md
        parameters = {
            "c_id": c_id,
            "text": text_content,
            "p_variation": random.choice(variations),
            "p_taxonomy_level": random.choice(taxonomy_levels),
            "p_mathematical_requirement_level": random.choice(math_levels),
            "p_root_text_reference_explanatory_text": random.choice(reference_values),
            "p_root_text_obstacle_passive": random.choice(binary_values),
            "p_root_text_obstacle_negation": random.choice(binary_values),
            "p_root_text_obstacle_complex_np": random.choice(binary_values),
            "p_root_text_contains_irrelevant_information": random.choice(binary_values),
            # Individual item parameters for all 8 items as per SYSARCH.md
            "p_item_1_obstacle_passive": random.choice(binary_values),
            "p_item_1_obstacle_negation": random.choice(binary_values),
            "p_item_1_obstacle_complex_np": random.choice(binary_values),
            "p_item_2_obstacle_passive": random.choice(binary_values),
            "p_item_2_obstacle_negation": random.choice(binary_values),
            "p_item_2_obstacle_complex_np": random.choice(binary_values),
            "p_item_3_obstacle_passive": random.choice(binary_values),
            "p_item_3_obstacle_negation": random.choice(binary_values),
            "p_item_3_obstacle_complex_np": random.choice(binary_values),
            "p_item_4_obstacle_passive": random.choice(binary_values),
            "p_item_4_obstacle_negation": random.choice(binary_values),
            "p_item_4_obstacle_complex_np": random.choice(binary_values),
            "p_item_5_obstacle_passive": random.choice(binary_values),
            "p_item_5_obstacle_negation": random.choice(binary_values),
            "p_item_5_obstacle_complex_np": random.choice(binary_values),
            "p_item_6_obstacle_passive": random.choice(binary_values),
            "p_item_6_obstacle_negation": random.choice(binary_values),
            "p_item_6_obstacle_complex_np": random.choice(binary_values),
            "p_item_7_obstacle_passive": random.choice(binary_values),
            "p_item_7_obstacle_negation": random.choice(binary_values),
            "p_item_7_obstacle_complex_np": random.choice(binary_values),
            "p_item_8_obstacle_passive": random.choice(binary_values),
            "p_item_8_obstacle_negation": random.choice(binary_values),
            "p_item_8_obstacle_complex_np": random.choice(binary_values),
            # Instruction parameters including the missing negation parameter
            "p_instruction_obstacle_passive": random.choice(binary_values),
            "p_instruction_obstacle_negation": random.choice(binary_values),
            "p_instruction_obstacle_complex_np": random.choice(binary_values),
            "p_instruction_explicitness_of_instruction": random.choice(explicit_values)
        }
        
        return parameters
    
    # Result saving removed - orchestrator handles all result management
    
    async def test_validation_plan_generation(self, request_data: Dict[str, Any]):
        """Test ValidationPlan question generation - pure caller mode"""
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/generate-validation-plan",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    
                    processing_time = time.time() - start_time
                    
                    if response.status == 200:
                        # Don't parse result data - orchestrator handles saving
                        logger.info(f"   Request successful for {request_data['c_id']} in {processing_time:.2f}s")
                        logger.info(f"   Results saved by orchestrator automatically")
                        
                        return True
                        
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ ValidationPlan generation failed: HTTP {response.status}")
                        logger.error(f"   Error: {error_text}")
                        return False
                        
            except Exception as e:
                logger.error(f"❌ ValidationPlan generation exception: {e}")
                return False
    
    @staticmethod
    def count_parameters_in_request() -> int:
        """Count total number of parameters being sent to verify SYSARCH compliance"""
        # This method helps verify we're sending all required parameters
        sample_params = StakeholderTestSystem.generate_randomized_parameters("test-1-1", "sample text", 1)
        return len(sample_params)
    
    async def run_comprehensive_stakeholder_test(self):
        """Run comprehensive test using all stakeholder texts with 5 randomized parameter sets each"""
        logger.info("=" * 80)
        logger.info("   STARTING STAKEHOLDER TEST - FULL SYSARCH COMPLIANCE")
        logger.info("=" * 80)
        
        # Verify parameter completeness
        param_count = self.count_parameters_in_request()
        logger.info(f"Using {param_count} ValidationPlan parameters per request (includes all individual item parameters 1-8)")
        logger.info("All SYSARCH-defined parameters implemented: Core, Root-text, Mathematical, Items 1-8, Instructions")
        
        # Health check first
        if not await self.test_health_endpoint():
            logger.error("Health check failed - aborting tests")
            return False
        
        # Load stakeholder texts
        try:
            df = self.load_stakeholder_texts()
        except Exception as e:
            logger.error(f"Failed to load stakeholder data: {e}")
            return False
        
        # Initialize counters
        total_runs = 0
        successful_runs = 0
        failed_runs = 0
        start_time = time.time()
        
        # Process each text in the CSV
        for idx, row in df.iterrows():
            original_c_id = row['c_id']
            subject = row['subject'] 
            text_content = row['text']
            
            # Convert idx to int for mathematical operations
            idx_int = int(idx)
            logger.info(f"\n Processing text {idx_int+1}/{len(df)}: {original_c_id}")
            logger.info(f"   Subject: {subject}")
            logger.info(f"   Text length: {len(text_content)} chars")
            
            text_successful = 0
            
            for run_num in range(1, 6):  # 1 to 5
                total_runs += 1
                
                # Generate randomized parameters
                request_data = self.generate_randomized_parameters(original_c_id, text_content, run_num)
                
                logger.info(f"\n   Run {run_num}/5 for {original_c_id}")
                logger.info(f"     c_id: {request_data['c_id']}")
                logger.info(f"     Variation: {request_data['p_variation']}")
                logger.info(f"     Taxonomy: {request_data['p_taxonomy_level']}")
                logger.info(f"     Math Level: {request_data['p_mathematical_requirement_level']}")
                
                # Log sample of individual item parameters to verify completeness
                logger.info(f"     Item Parameters Sample: Item1_Passive={request_data['p_item_1_obstacle_passive']}, Item4_Negation={request_data['p_item_4_obstacle_negation']}, Item8_Complex={request_data['p_item_8_obstacle_complex_np']}")
                logger.info(f"     Instruction Parameters: Passive={request_data['p_instruction_obstacle_passive']}, Negation={request_data['p_instruction_obstacle_negation']}, Explicitness={request_data['p_instruction_explicitness_of_instruction']}")
                logger.info(f"     Total Parameters Sent: {len(request_data)} (SYSARCH-compliant with all individual item parameters)")
                
                # Execute the test - orchestrator handles all result saving
                success = await self.test_validation_plan_generation(request_data)
                
                if success:
                    successful_runs += 1
                    text_successful += 1
                else:
                    failed_runs += 1
                
                # Brief pause between runs
                await asyncio.sleep(1)
            
            # Log text completion summary
            logger.info(f"   Text {original_c_id} completed: {text_successful}/5 successful runs")
            logger.info(f"   Overall progress: {idx_int + 1}/{len(df)} texts, {successful_runs}/{total_runs} successful calls")
            
            # Longer pause between texts
            await asyncio.sleep(2)
        
        # Calculate final statistics
        total_time = time.time() - start_time
        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
        
        logger.info("\n" + "=" * 80)
        logger.info("FINAL CALLER SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total texts processed: {len(df)}")
        logger.info(f"Total HTTP calls made: {total_runs}")
        logger.info(f"Successful calls: {successful_runs}")
        logger.info(f"Failed calls: {failed_runs}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"⏱Total caller time: {total_time:.2f}s")
        logger.info(f"Average time per call: {total_time/total_runs:.2f}s")
        logger.info("")
        logger.info("All results automatically saved by orchestrator to:")
        logger.info("   CallersWithTexts/results/YYYY-MM-DD_HH-MM-SS/")
        logger.info("   • Complete CSV with all SYSARCH columns")
        logger.info("   • System metadata and VRAM usage")
        logger.info("   • All ALEE_Agent prompts snapshot")
        logger.info("   • Expert validation logs")
        
        logger.info("=" * 80)
        logger.info("STAKEHOLDER CALLING TEST COMPLETE")
        logger.info("=" * 80)
        
        return successful_runs > 0

async def main():
    """Main function to run stakeholder testing"""
    tester = StakeholderTestSystem()
    
    try:
        success = await tester.run_comprehensive_stakeholder_test()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Testing failed with exception: {e}")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)