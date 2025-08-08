#!/usr/bin/env python3
"""
ValidationPlan Stakeholder Data Testing System
Uses all rows from explanation_metadata.csv to test with randomized parameters
Saves complete results and parameters to CallersWithTexts/results/
"""

import asyncio
import aiohttp
import json
import time
import csv
import random
import logging
from pathlib import Path
from typing import List, Dict, Any
from result_manager import save_results, ResultManager
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StakeholderTestSystem:
    """Comprehensive ValidationPlan testing using stakeholder explanation_metadata.csv"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.stakeholder_csv_path = "/home/mklemmingen/PycharmProjects/PythonProject/_dev/providedProjectFromStakeHolder/explanation_metadata.csv"
        self.all_results = []
        
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
    
    def generate_randomized_parameters(self, base_c_id: str, text_content: str, run_num: int) -> Dict[str, Any]:
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
        
        return {
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
            "p_item_X_obstacle_passive": random.choice(binary_values),
            "p_item_X_obstacle_negation": random.choice(binary_values),
            "p_item_X_obstacle_complex_np": random.choice(binary_values),
            "p_instruction_obstacle_passive": random.choice(binary_values),
            "p_instruction_obstacle_complex_np": random.choice(binary_values),
            "p_instruction_explicitness_of_instruction": random.choice(explicit_values)
        }
    
    async def save_incremental_results(self, texts_completed: int, total_texts: int, total_runs: int, successful_runs: int):
        """Save incremental results after each text completion (simulating real production environment)"""
        
        # Create incremental metadata
        incremental_metadata = {
            "test_type": "stakeholder_incremental_save",
            "description": f"Incremental save after {texts_completed}/{total_texts} texts completed",
            "source_file": "explanation_metadata.csv",
            "texts_completed": texts_completed,
            "total_texts": total_texts,
            "runs_per_text": 5,
            "total_runs_so_far": total_runs,
            "successful_runs_so_far": successful_runs,
            "failed_runs_so_far": total_runs - successful_runs,
            "success_rate_percent": (successful_runs / total_runs * 100) if total_runs > 0 else 0,
            "completion_percentage": (texts_completed / total_texts * 100),
            "incremental_save_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "validation_plan_compliant": True,
            "is_incremental_save": True
        }
        
        try:
            # Save current results
            session_dir = save_results(self.all_results, incremental_metadata)
            
            logger.info(f"      Incremental save completed: {session_dir}")
            logger.info(f"      Progress: {texts_completed}/{total_texts} texts ({incremental_metadata['completion_percentage']:.1f}%)")
            logger.info(f"      Runs: {total_runs} total, {successful_runs} successful ({incremental_metadata['success_rate_percent']:.1f}%)")
            
        except Exception as e:
            logger.warning(f"   Incremental save failed (continuing anyway): {e}")
    
    async def test_validation_plan_generation(self, request_data: Dict[str, Any]):
        """Test ValidationPlan question generation"""
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
                        result = await response.json()
                        
                        logger.info(f"   Generated 3 questions for {request_data['c_id']} in {processing_time:.2f}s")
                        logger.info(f"   Q1: {result.get('question_1', '')[:60]}...")
                        logger.info(f"   Q2: {result.get('question_2', '')[:60]}...")
                        logger.info(f"   Q3: {result.get('question_3', '')[:60]}...")
                        
                        return True, result
                        
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ ValidationPlan generation failed: HTTP {response.status}")
                        logger.error(f"   Error: {error_text}")
                        return False, {"error": f"HTTP {response.status}: {error_text}"}
                        
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"❌ ValidationPlan generation exception: {e}")
                return False, {"error": str(e), "processing_time": processing_time}
    
    async def run_comprehensive_stakeholder_test(self):
        """Run comprehensive test using all stakeholder texts with 5 randomized parameter sets each"""
        logger.info("=" * 80)
        logger.info("   STARTING COMPREHENSIVE STAKEHOLDER TEST")
        logger.info("=" * 80)
        
        # Health check first
        if not await self.test_health_endpoint():
            logger.error("❌ Health check failed - aborting tests")
            return False
        
        # Load stakeholder texts
        try:
            df = self.load_stakeholder_texts()
        except Exception as e:
            logger.error(f"❌ Failed to load stakeholder data: {e}")
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
            
            logger.info(f"\n Processing text {idx+1}/{len(df)}: {original_c_id}")
            logger.info(f"   Subject: {subject}")
            logger.info(f"   Text length: {len(text_content)} chars")
            
            # Generate 5 randomized parameter runs for this text
            text_results = []
            
            for run_num in range(1, 6):  # 1 to 5
                total_runs += 1
                
                # Generate randomized parameters
                request_data = self.generate_randomized_parameters(original_c_id, text_content, run_num)
                
                logger.info(f"\n   Run {run_num}/5 for {original_c_id}")
                logger.info(f"     c_id: {request_data['c_id']}")
                logger.info(f"     Variation: {request_data['p_variation']}")
                logger.info(f"     Taxonomy: {request_data['p_taxonomy_level']}")
                logger.info(f"     Math Level: {request_data['p_mathematical_requirement_level']}")
                
                # Execute the test
                success, result = await self.test_validation_plan_generation(request_data)
                
                # Prepare result record with ALL information
                result_record = {
                    # Original stakeholder data
                    "original_c_id": original_c_id,
                    "original_subject": subject,
                    "text_row_index": idx + 1,
                    "run_number": run_num,
                    "total_run_number": total_runs,
                    
                    # All parameter values used
                    "parameters_used": request_data,
                    
                    # Test results
                    "success": success,
                    "processing_time": result.get("processing_time", 0),
                }
                
                if success:
                    successful_runs += 1
                    result_record.update({
                        "question_1": result.get("question_1", ""),
                        "question_2": result.get("question_2", ""),
                        "question_3": result.get("question_3", ""),
                        "csv_data": result.get("csv_data", {}),
                        "generation_c_id": result.get("c_id", ""),
                    })
                    logger.info(f"       SUCCESS - 3 questions generated")
                else:
                    failed_runs += 1
                    result_record.update({
                        "error": result.get("error", "Unknown error"),
                        "question_1": "",
                        "question_2": "",
                        "question_3": "",
                        "csv_data": {},
                    })
                    logger.info(f"     ❌ FAILED - {result.get('error', 'Unknown error')}")
                
                # Add to results
                self.all_results.append(result_record)
                text_results.append(result_record)
                
                # Brief pause between runs
                await asyncio.sleep(1)
            
            # Log text completion
            text_success_count = sum(1 for r in text_results if r["success"])
            logger.info(f"   📊 Text {original_c_id} completed: {text_success_count}/5 successful runs")
            
            # Save incremental results after each text (simulating real production runs)
            await self.save_incremental_results(idx + 1, len(df), total_runs, successful_runs)
            
            # Longer pause between texts
            await asyncio.sleep(2)
        
        # Calculate final statistics
        total_time = time.time() - start_time
        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 FINAL RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total texts processed: {len(df)}")
        logger.info(f"Total runs executed: {total_runs}")
        logger.info(f"Successful runs: {successful_runs}")
        logger.info(f"Failed runs: {failed_runs}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Total processing time: {total_time:.2f}s")
        logger.info(f"Average time per run: {total_time/total_runs:.2f}s")
        
        # Save results using result_manager
        metadata = {
            "test_type": "stakeholder_comprehensive_validation_plan_test",
            "description": "Complete test using all explanation_metadata.csv texts with 5 randomized parameter sets each",
            "source_file": "explanation_metadata.csv",
            "total_texts": len(df),
            "runs_per_text": 5,
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "success_rate_percent": success_rate,
            "total_processing_time_seconds": total_time,
            "avg_time_per_run_seconds": total_time/total_runs if total_runs > 0 else 0,
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "validation_plan_compliant": True,
            "parameter_randomization": "All 16+ ValidationPlan parameters randomized per run",
            "results_include": "Original text data, all parameters used, all generated questions, CSV data"
        }
        
        try:
            session_dir = save_results(self.all_results, metadata)
            logger.info(f"\n    RESULTS SAVED TO: {session_dir}")
            logger.info(f"      CSV results: {session_dir}/results.csv")
            logger.info(f"      Prompts snapshot: {session_dir}/prompts/")
            logger.info(f"      Metadata: {session_dir}/session_metadata.json")
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
        
        logger.info("=" * 80)
        logger.info("   STAKEHOLDER TEST COMPLETE")
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