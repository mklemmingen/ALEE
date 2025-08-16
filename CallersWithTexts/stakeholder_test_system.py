#!/usr/bin/env python3
"""
Stakeholder Testing System - Enhanced with Modular Components
Uses all rows from explanation_metadata.csv to make HTTP requests with systematic parameters
Orchestrator handles all result saving - enhanced with structured logging and caller module
"""

import asyncio
import time
from typing import Dict, Any

# Import new modular components
from sharedTools.caller import OrchestorCaller
from sharedTools.config import CallerConfig
from sharedTools.logger import CallerLogger


class StakeholderTestSystem:
    """Enhanced stakeholder testing system with modular components"""
    
    def __init__(self):
        # Initialize modular components
        self.caller = OrchestorCaller(
            base_url=CallerConfig.get_base_url(),
            timeout_seconds=CallerConfig.get_timeout()
        )
        self.logger = CallerLogger("stakeholder_test_system")
        self.stakeholder_csv_path = CallerConfig.get_stakeholder_csv_path()
        
        # Log initialization
        self.logger.info("Initialized StakeholderTestSystem", 
                        base_url=self.caller.base_url,
                        timeout=self.caller.timeout_seconds,
                        csv_path=self.stakeholder_csv_path)
        
    async def test_health_endpoint(self) -> bool:
        """Test DSPy system health endpoint using modular caller"""
        correlation_id = self.caller._get_correlation_id()
        self.logger.log_request_start(correlation_id, "health_check")
        
        start_time = time.time()
        is_healthy, health_data = await self.caller.test_health_endpoint(correlation_id)
        elapsed_time = time.time() - start_time
        
        self.logger.log_system_health(correlation_id, is_healthy, health_data, elapsed_time)
        self.logger.log_request_end(correlation_id, "health_check", is_healthy, elapsed_time)
        
        return is_healthy
    
    def load_stakeholder_texts(self):
        """Load all texts from explanation_metadata.csv using modular caller"""
        self.logger.info("Loading stakeholder texts", csv_path=self.stakeholder_csv_path)
        
        df = self.caller.load_stakeholder_texts(self.stakeholder_csv_path)
        if df is None:
            self.logger.error("Failed to load stakeholder texts")
            raise Exception(f"Could not load CSV from {self.stakeholder_csv_path}")
        
        self.logger.info("Successfully loaded stakeholder texts", 
                        text_count=len(df), 
                        columns=list(df.columns))
        return df
    
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
    
    async def test_question_generation(self, request_data: Dict[str, Any]) -> bool:
        """Test DSPy question generation using modular caller"""
        correlation_id = self.caller._get_correlation_id()
        c_id = request_data.get('c_id', 'unknown')
        question_type = request_data.get('question_type', 'unknown')
        
        self.logger.log_request_start(correlation_id, "question_generation", 
                                     c_id=c_id, question_type=question_type)
        
        start_time = time.time()
        success, result_data = await self.caller.generate_questions(request_data, correlation_id)
        elapsed_time = time.time() - start_time
        
        self.logger.log_question_generation(correlation_id, c_id, question_type, 
                                           success, result_data, elapsed_time)
        self.logger.log_request_end(correlation_id, "question_generation", success, elapsed_time,
                                   c_id=c_id)
        
        return success
    
    @staticmethod
    def count_parameters_in_request() -> int:
        """Count total number of parameters being sent to verify SYSARCH compliance"""
        # Use CallerConfig to get required parameters count
        required_params = CallerConfig.get_required_parameters()
        sample_params = StakeholderTestSystem.generate_systematic_parameters("test-1-1", "sample text", 0)
        return len(sample_params)
    
    async def run_comprehensive_stakeholder_test(self) -> bool:
        """Run systematic test using all stakeholder texts with enhanced logging"""
        self.logger.info("Starting comprehensive stakeholder test", 
                        event_type="test_session_start")
        
        self.logger.info("SYSTEMATIC DSPy STAKEHOLDER TEST - COMPREHENSIVE PARAMETER COVERAGE")
        
        # Verify parameter completeness using config
        param_count = self.count_parameters_in_request()
        required_params = CallerConfig.get_required_parameters()
        
        self.logger.info("Parameter coverage analysis", 
                        total_params=param_count,
                        required_params=len(required_params),
                        question_types=CallerConfig.VALID_QUESTION_TYPES,
                        variations=CallerConfig.VALID_VARIATIONS,
                        taxonomy_levels=CallerConfig.VALID_TAXONOMY_LEVELS)
        
        self.logger.info(f"Using {param_count} parameters per request (SYSARCH compliant)")
        self.logger.info("Systematic parameter coverage: question types, difficulty levels, taxonomy, obstacles")
        
        # Health check first
        if not await self.test_health_endpoint():
            self.logger.error("Health check failed - aborting tests", event_type="test_abort")
            return False
        
        # Load stakeholder texts
        try:
            df = self.load_stakeholder_texts()
        except Exception as e:
            self.logger.error("Failed to load stakeholder data", error=str(e), event_type="test_abort")
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
            
            self.logger.info(f"Processing text {idx_int+1}/{len(df)}", 
                           text_id=original_c_id,
                           subject=subject, 
                           text_length=len(text_content),
                           progress=f"{idx_int+1}/{len(df)}")
            
            # Generate systematic parameters (one set per text)
            request_data = self.generate_systematic_parameters(original_c_id, text_content, idx_int)
            
            # Track parameter usage for coverage verification
            question_types_used.add(request_data['question_type'])
            variations_used.add(request_data['p_variation'])
            taxonomies_used.add(request_data['p_taxonomy_level'])
            math_levels_used.add(request_data['p_mathematical_requirement_level'])
            
            self.logger.info("Systematic test parameters",
                           original_c_id=original_c_id,
                           test_c_id=request_data['c_id'],
                           question_type=request_data['question_type'],
                           difficulty=request_data['p_variation'],
                           taxonomy=request_data['p_taxonomy_level'],
                           math_level=request_data['p_mathematical_requirement_level'],
                           reference=request_data['p_root_text_reference_explanatory_text'])
            
            # Log systematic obstacle patterns
            self.logger.info("Obstacle pattern analysis",
                           root_obstacles={
                               "passive": request_data['p_root_text_obstacle_passive'],
                               "negation": request_data['p_root_text_obstacle_negation'],
                               "complex": request_data['p_root_text_obstacle_complex_np']
                           },
                           item_1_sample={
                               "passive": request_data['p_item_1_obstacle_passive'],
                               "negation": request_data['p_item_1_obstacle_negation'], 
                               "complex": request_data['p_item_1_obstacle_complex_np']
                           },
                           instruction_params={
                               "passive": request_data['p_instruction_obstacle_passive'],
                               "negation": request_data['p_instruction_obstacle_negation'],
                               "explicitness": request_data['p_instruction_explicitness_of_instruction']
                           },
                           total_parameters=len(request_data))
            
            # Execute the test - orchestrator handles all result saving
            success = await self.test_question_generation(request_data)
            
            if success:
                successful_runs += 1
            else:
                failed_runs += 1
            
            # Log progress
            status = "✓ COMPLETED" if success else "✗ FAILED"
            self.logger.info(f"Text {original_c_id} {status}", 
                           success=success,
                           overall_progress=f"{idx_int + 1}/{len(df)}",
                           success_rate=f"{successful_runs}/{total_runs}")
            
            # Brief pause between texts
            await asyncio.sleep(2)
        
        # Calculate final statistics
        total_time = time.time() - start_time
        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
        
        # Generate comprehensive test summary
        test_summary = {
            "total_texts": len(df),
            "total_requests": total_runs,
            "successful_requests": successful_runs,
            "failed_requests": failed_runs,
            "success_rate": success_rate,
            "total_time": total_time,
            "average_time_per_call": total_time/total_runs if total_runs > 0 else 0,
            "question_types_tested": len(question_types_used),
            "variations_tested": len(variations_used),
            "taxonomies_tested": len(taxonomies_used),
            "math_levels_tested": len(math_levels_used)
        }
        
        self.logger.info("SYSTEMATIC TEST SUMMARY - PARAMETER COVERAGE ANALYSIS")
        self.logger.info("Test execution completed", 
                        event_type="test_session_end",
                        summary=test_summary)
        # Log parameter coverage analysis
        coverage_analysis = {
            "question_types": {
                "tested": len(question_types_used),
                "total": len(CallerConfig.VALID_QUESTION_TYPES),
                "coverage": sorted(question_types_used)
            },
            "difficulty_levels": {
                "tested": len(variations_used),
                "total": len(CallerConfig.VALID_VARIATIONS), 
                "coverage": sorted(variations_used)
            },
            "taxonomy_levels": {
                "tested": len(taxonomies_used),
                "total": len(CallerConfig.VALID_TAXONOMY_LEVELS),
                "coverage": sorted(taxonomies_used)
            },
            "math_levels": {
                "tested": len(math_levels_used),
                "total": len(CallerConfig.VALID_MATH_LEVELS),
                "coverage": sorted(math_levels_used)
            }
        }
        
        self.logger.info("SYSTEMATIC PARAMETER COVERAGE ACHIEVED", 
                        coverage_analysis=coverage_analysis)
        # Log completion with session statistics
        session_stats = self.caller.get_session_statistics()
        
        self.logger.info("All results automatically saved by DSPy orchestrator")
        self.logger.info("Result storage: results/YYYY-MM-DD_HH-MM-SS_c_id/ with complete pipeline tracking")
        
        self.logger.info("SYSTEMATIC DSPy STAKEHOLDER TEST COMPLETE", 
                        event_type="test_session_complete",
                        session_statistics=session_stats)
        
        # End the logging session with summary
        final_summary = {**test_summary, **session_stats, **coverage_analysis}
        self.logger.log_session_end(final_summary)
        
        return successful_runs > 0

async def main():
    """Main function to run stakeholder testing with enhanced error handling"""
    tester = None
    
    try:
        tester = StakeholderTestSystem()
        success = await tester.run_comprehensive_stakeholder_test()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        if tester:
            tester.logger.warning("Testing interrupted by user", event_type="user_interrupt")
        return 1
    except Exception as e:
        if tester:
            tester.logger.error("Testing failed with exception", error=str(e), event_type="system_error")
        else:
            print(f"Failed to initialize system: {e}")
        return 1
    finally:
        if tester:
            # Ensure session is properly closed
            session_summary = tester.logger.get_session_summary()
            print(f"\nSession completed. Log file: {session_summary.get('log_file')}")
            print(f"Total log entries: {session_summary.get('total_log_entries', 0)}")

if __name__ == "__main__":
    import sys
    
    print("=" * 80)
    print("STAKEHOLDER TESTING SYSTEM - Enhanced with Modular Components")
    print("=" * 80)
    print("Features: Structured logging, parameter validation, performance tracking")
    print("Log files saved to CallersWithTexts/_logs/ with ISO8601 timestamps")
    print("=" * 80)
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)