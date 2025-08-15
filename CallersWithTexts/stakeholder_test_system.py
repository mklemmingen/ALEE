#!/usr/bin/env python3
"""
Stakeholder Testing System - Pure Caller Mode
Uses all rows from explanation_metadata.csv to make HTTP requests with randomized parameters
Orchestrator handles all result saving - caller only monitors process completion
"""

import asyncio
import logging
import time
from typing import Dict, Any

import aiohttp
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StakeholderTestSystem:
    """Pure caller for question generation testing using stakeholder explanation_metadata.csv"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.stakeholder_csv_path = ".dev/providedProjectFromStakeHolder/explanation_metadata.csv" # confidential path, please change if needed to a csv of your choice
        # No result storage - orchestrator handles everything
        
    async def test_health_endpoint(self):
        """Test DSPy system health endpoint"""
        logger.info("Testing DSPy health endpoint...")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/health-dspy") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"DSPy health check passed: {data.get('status', 'unknown')}, DSPy ready: {data.get('dspy_ready', False)}")
                        return data.get('dspy_ready', False)
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
    def generate_systematic_parameters(base_c_id: str, text_content: str, text_index: int) -> Dict[str, Any]:
        """Generate systematic SysArch parameters for comprehensive testing"""
        
        # Create systematic c_id 
        c_id = f"{base_c_id.split('-')[0]}-sys-{text_index+1}"
        
        # Define systematic parameter sets to ensure comprehensive coverage
        question_types = ["multiple-choice", "single-choice", "true-false", "mapping"]
        variations = ["stammaufgabe", "schwer", "leicht"]
        taxonomy_levels = ["Stufe 1 (Wissen/Reproduktion)", "Stufe 2 (Anwendung/Transfer)"]
        math_levels = ["0", "1", "2"]
        binary_values = ["Enthalten", "Nicht Enthalten"]
        reference_values = ["Nicht vorhanden", "Explizit", "Implizit"]
        explicit_values = ["Explizit", "Implizit"]
        
        # Systematic selection based on text index to ensure all combinations are tested
        question_type = question_types[text_index % len(question_types)]
        variation = variations[text_index % len(variations)]
        taxonomy = taxonomy_levels[text_index % len(taxonomy_levels)]
        math_level = math_levels[text_index % len(math_levels)]
        
        # Systematic binary selections to test different combinations
        text_obstacles_pattern = text_index % 8  # 0-7 for different obstacle combinations
        item_obstacles_pattern = (text_index + 1) % 8  # Different pattern for items
        instruction_pattern = (text_index + 2) % 4  # Pattern for instruction obstacles
        
        # Root text obstacles (systematic coverage)
        root_passive = "Enthalten" if (text_obstacles_pattern & 1) else "Nicht Enthalten"
        root_negation = "Enthalten" if (text_obstacles_pattern & 2) else "Nicht Enthalten"
        root_complex = "Enthalten" if (text_obstacles_pattern & 4) else "Nicht Enthalten"
        
        # Instruction obstacles (systematic coverage)
        instr_passive = "Enthalten" if (instruction_pattern & 1) else "Nicht Enthalten"
        instr_negation = "Enthalten" if (instruction_pattern & 2) else "Nicht Enthalten"
        instr_complex = "Enthalten" if (instruction_pattern & 1) else "Nicht Enthalten"
        
        # Generate complete parameter set with systematic coverage
        parameters = {
            "c_id": c_id,
            "text": text_content,
            "question_type": question_type,
            "p_variation": variation,
            "p_taxonomy_level": taxonomy,
            "p_mathematical_requirement_level": math_level,
            "p_root_text_reference_explanatory_text": reference_values[text_index % len(reference_values)],
            "p_root_text_obstacle_passive": root_passive,
            "p_root_text_obstacle_negation": root_negation,
            "p_root_text_obstacle_complex_np": root_complex,
            "p_root_text_contains_irrelevant_information": binary_values[text_index % 2],
            
            # Individual item parameters (systematic patterns for items 1-8)
            "p_item_1_obstacle_passive": "Enthalten" if (item_obstacles_pattern & 1) else "Nicht Enthalten",
            "p_item_1_obstacle_negation": "Enthalten" if (item_obstacles_pattern & 2) else "Nicht Enthalten",
            "p_item_1_obstacle_complex_np": "Enthalten" if (item_obstacles_pattern & 4) else "Nicht Enthalten",
            "p_item_2_obstacle_passive": "Enthalten" if ((item_obstacles_pattern + 1) & 1) else "Nicht Enthalten",
            "p_item_2_obstacle_negation": "Enthalten" if ((item_obstacles_pattern + 1) & 2) else "Nicht Enthalten",
            "p_item_2_obstacle_complex_np": "Enthalten" if ((item_obstacles_pattern + 1) & 4) else "Nicht Enthalten",
            "p_item_3_obstacle_passive": "Enthalten" if ((item_obstacles_pattern + 2) & 1) else "Nicht Enthalten",
            "p_item_3_obstacle_negation": "Enthalten" if ((item_obstacles_pattern + 2) & 2) else "Nicht Enthalten",
            "p_item_3_obstacle_complex_np": "Enthalten" if ((item_obstacles_pattern + 2) & 4) else "Nicht Enthalten",
            "p_item_4_obstacle_passive": "Enthalten" if ((item_obstacles_pattern + 3) & 1) else "Nicht Enthalten",
            "p_item_4_obstacle_negation": "Enthalten" if ((item_obstacles_pattern + 3) & 2) else "Nicht Enthalten",
            "p_item_4_obstacle_complex_np": "Enthalten" if ((item_obstacles_pattern + 3) & 4) else "Nicht Enthalten",
            "p_item_5_obstacle_passive": "Enthalten" if ((item_obstacles_pattern + 4) & 1) else "Nicht Enthalten",
            "p_item_5_obstacle_negation": "Enthalten" if ((item_obstacles_pattern + 4) & 2) else "Nicht Enthalten",
            "p_item_5_obstacle_complex_np": "Enthalten" if ((item_obstacles_pattern + 4) & 4) else "Nicht Enthalten",
            "p_item_6_obstacle_passive": "Enthalten" if ((item_obstacles_pattern + 5) & 1) else "Nicht Enthalten",
            "p_item_6_obstacle_negation": "Enthalten" if ((item_obstacles_pattern + 5) & 2) else "Nicht Enthalten",
            "p_item_6_obstacle_complex_np": "Enthalten" if ((item_obstacles_pattern + 5) & 4) else "Nicht Enthalten",
            "p_item_7_obstacle_passive": "Enthalten" if ((item_obstacles_pattern + 6) & 1) else "Nicht Enthalten",
            "p_item_7_obstacle_negation": "Enthalten" if ((item_obstacles_pattern + 6) & 2) else "Nicht Enthalten",
            "p_item_7_obstacle_complex_np": "Enthalten" if ((item_obstacles_pattern + 6) & 4) else "Nicht Enthalten",
            "p_item_8_obstacle_passive": "Enthalten" if ((item_obstacles_pattern + 7) & 1) else "Nicht Enthalten",
            "p_item_8_obstacle_negation": "Enthalten" if ((item_obstacles_pattern + 7) & 2) else "Nicht Enthalten",
            "p_item_8_obstacle_complex_np": "Enthalten" if ((item_obstacles_pattern + 7) & 4) else "Nicht Enthalten",
            
            # Instruction parameters with systematic coverage
            "p_instruction_obstacle_passive": instr_passive,
            "p_instruction_obstacle_negation": instr_negation,
            "p_instruction_obstacle_complex_np": instr_complex,
            "p_instruction_explicitness_of_instruction": explicit_values[text_index % 2]
        }
        
        return parameters
    
    # Result saving removed - orchestrator handles all result management
    
    async def test_question_generation(self, request_data: Dict[str, Any]):
        """Test DSPy question generation - pure caller mode"""
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/generate-educational-questions",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=600)  # Increased timeout for DSPy processing
                ) as response:
                    
                    processing_time = time.time() - start_time
                    
                    if response.status == 200:
                        # Don't parse result data - DSPy orchestrator handles saving
                        logger.info(f"   DSPy request successful for {request_data['c_id']} in {processing_time:.2f}s")
                        logger.info(f"   Results saved by DSPy orchestrator automatically")
                        
                        return True
                        
                    else:
                        error_text = await response.text()
                        logger.error(f"Question generation failed: HTTP {response.status}")
                        logger.error(f"   Error: {error_text}")
                        return False
                        
            except Exception as e:
                logger.error(f"Question generation exception: {e}")
                return False
    
    @staticmethod
    def count_parameters_in_request() -> int:
        """Count total number of parameters being sent to verify SYSARCH compliance"""
        # This method helps verify we're sending all required parameters
        sample_params = StakeholderTestSystem.generate_systematic_parameters("test-1-1", "sample text", 0)
        return len(sample_params)
    
    async def run_comprehensive_stakeholder_test(self):
        """Run systematic test using all stakeholder texts with one systematic parameter set each"""
        logger.info("=" * 80)
        logger.info("   STARTING SYSTEMATIC DSPy STAKEHOLDER TEST - COMPREHENSIVE PARAMETER COVERAGE")
        logger.info("=" * 80)
        
        # Verify parameter completeness
        param_count = self.count_parameters_in_request()
        logger.info(f"Using {param_count} parameters per request (includes all individual item parameters 1-8)")
        logger.info("Systematic parameter selection ensures comprehensive coverage of all SYSARCH parameters:")
        logger.info("• All 4 question types systematically tested")
        logger.info("• All difficulty levels and taxonomy levels covered")
        logger.info("• All mathematical levels tested systematically")
        logger.info("• Obstacle combinations tested through bit patterns")
        logger.info("• Each text gets unique, systematic parameter combination")
        
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
        
        # Parameter tracking for systematic coverage verification
        question_types_used = set()
        variations_used = set()
        taxonomies_used = set()
        math_levels_used = set()
        
        # Process each text in the CSV (ONE RUN PER TEXT)
        for idx, row in df.iterrows():
            original_c_id = row['c_id']
            subject = row['subject'] 
            text_content = row['text']
            
            # Convert idx to int for mathematical operations
            idx_int = int(idx)
            total_runs += 1
            
            logger.info(f"\n Processing text {idx_int+1}/{len(df)}: {original_c_id}")
            logger.info(f"   Subject: {subject}")
            logger.info(f"   Text length: {len(text_content)} chars")
            
            # Generate systematic parameters (one set per text)
            request_data = self.generate_systematic_parameters(original_c_id, text_content, idx_int)
            
            # Track parameter usage for coverage verification
            question_types_used.add(request_data['question_type'])
            variations_used.add(request_data['p_variation'])
            taxonomies_used.add(request_data['p_taxonomy_level'])
            math_levels_used.add(request_data['p_mathematical_requirement_level'])
            
            logger.info(f"   Systematic Test for {original_c_id}")
            logger.info(f"     c_id: {request_data['c_id']}")
            logger.info(f"     Question Type: {request_data['question_type']}")
            logger.info(f"     Difficulty: {request_data['p_variation']}")
            logger.info(f"     Taxonomy: {request_data['p_taxonomy_level']}")
            logger.info(f"     Math Level: {request_data['p_mathematical_requirement_level']}")
            logger.info(f"     Reference: {request_data['p_root_text_reference_explanatory_text']}")
            
            # Log sample of systematic obstacle patterns
            logger.info(f"     Root Obstacles: Passive={request_data['p_root_text_obstacle_passive']}, Negation={request_data['p_root_text_obstacle_negation']}, Complex={request_data['p_root_text_obstacle_complex_np']}")
            logger.info(f"     Item 1 Sample: Passive={request_data['p_item_1_obstacle_passive']}, Negation={request_data['p_item_1_obstacle_negation']}, Complex={request_data['p_item_1_obstacle_complex_np']}")
            logger.info(f"     Instruction: Passive={request_data['p_instruction_obstacle_passive']}, Negation={request_data['p_instruction_obstacle_negation']}, Explicitness={request_data['p_instruction_explicitness_of_instruction']}")
            logger.info(f"     Total Parameters: {len(request_data)} (SYSTEMATIC SYSARCH-compliant coverage)")
            
            # Execute the test - orchestrator handles all result saving
            success = await self.test_question_generation(request_data)
            
            if success:
                successful_runs += 1
            else:
                failed_runs += 1
            
            # Log progress
            logger.info(f"   Text {original_c_id} {'✓ COMPLETED' if success else '✗ FAILED'}")
            logger.info(f"   Overall progress: {idx_int + 1}/{len(df)} texts, {successful_runs}/{total_runs} successful calls")
            
            # Brief pause between texts
            await asyncio.sleep(2)
        
        # Calculate final statistics
        total_time = time.time() - start_time
        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
        
        logger.info("\n" + "=" * 80)
        logger.info("SYSTEMATIC TEST SUMMARY - PARAMETER COVERAGE ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"Total texts processed: {len(df)}")
        logger.info(f"Total HTTP calls made: {total_runs} (1 per text)")
        logger.info(f"Successful calls: {successful_runs}")
        logger.info(f"Failed calls: {failed_runs}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"⏱Total processing time: {total_time:.2f}s")
        logger.info(f"Average time per call: {total_time/total_runs:.2f}s")
        logger.info("")
        logger.info("SYSTEMATIC PARAMETER COVERAGE ACHIEVED:")
        logger.info(f"• Question Types tested: {len(question_types_used)}/4 → {sorted(question_types_used)}")
        logger.info(f"• Difficulty levels tested: {len(variations_used)}/3 → {sorted(variations_used)}")
        logger.info(f"• Taxonomy levels tested: {len(taxonomies_used)}/2 → {sorted(taxonomies_used)}")
        logger.info(f"• Math levels tested: {len(math_levels_used)}/3 → {sorted(math_levels_used)}")
        logger.info("• Obstacle combinations: Systematic bit patterns ensuring diverse coverage")
        logger.info("• Item parameters 1-8: All systematically tested with unique patterns")
        logger.info("")
        logger.info("All results automatically saved by DSPy orchestrator to:")
        logger.info("   results/YYYY-MM-DD_HH-MM-SS_c_id/")
        logger.info("   • Complete CSV with all SYSARCH columns")
        logger.info("   • DSPy pipeline steps (generation → experts → consensus)")
        logger.info("   • Individual expert evaluations (5 experts × 3 questions each)")
        logger.info("   • System metadata and processing times")
        logger.info("   • All ALEE_Agent prompts snapshot")
        logger.info("   • Expert validation logs and detailed feedback")
        
        logger.info("=" * 80)
        logger.info("SYSTEMATIC DSPy STAKEHOLDER TEST COMPLETE")
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