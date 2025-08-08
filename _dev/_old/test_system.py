#!/usr/bin/env python3
"""
Educational AI System Testing Suite - ValidationPlan Aligned
Comprehensive testing of parameter-expert LLM functionality with ValidationPlan compliance
"""

import asyncio
import aiohttp
import json
import time
import csv
from pathlib import Path
import logging
from typing import List, Dict, Any
from result_manager import save_results, ResultManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemTester:
    """Comprehensive testing suite for the Educational AI system - ValidationPlan aligned"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_results = []
        self.validation_plan_results = []  # Track ValidationPlan specific results
        
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
    
    async def test_model_status_endpoint(self):
        """Test model status endpoint"""
        logger.info("Testing model status endpoint...")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/models/status") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Model status: {data['vram_usage_gb']:.1f}GB used of {data['max_vram_gb']}GB")
                        return True, data
                    else:
                        logger.error(f"Model status failed: {response.status}")
                        return False, {}
            except Exception as e:
                logger.error(f"Model status error: {e}")
                return False, {}
    
    async def test_validation_plan_generation(self, request_data: Dict[str, Any]):
        """Test ValidationPlan question generation with detailed analysis"""
        logger.info(f"Testing ValidationPlan: {request_data['c_id']} ({request_data['p_variation']})")
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/generate-validation-plan",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=300)  # Longer timeout for 3 questions
                ) as response:
                    
                    processing_time = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        test_result = {
                            "c_id": request_data["c_id"],
                            "p_variation": request_data["p_variation"],
                            "p_taxonomy_level": request_data["p_taxonomy_level"],
                            "success": True,
                            "processing_time": processing_time,
                            "questions_generated": 3,
                            "csv_data_complete": bool(result.get("csv_data")),
                            "question_1_length": len(result.get("question_1", "")),
                            "question_2_length": len(result.get("question_2", "")),
                            "question_3_length": len(result.get("question_3", ""))
                        }
                        
                        logger.info(f"ValidationPlan questions generated successfully!")
                        logger.info(f"   - Processing time: {processing_time:.2f}s")
                        logger.info(f"   - Question 1: {result.get('question_1', '')[:80]}...")
                        logger.info(f"   - Question 2: {result.get('question_2', '')[:80]}...")
                        logger.info(f"   - Question 3: {result.get('question_3', '')[:80]}...")
                        logger.info(f"   - CSV data complete: {bool(result.get('csv_data'))}")
                        
                        self.validation_plan_results.append(test_result)
                        return True, result
                        
                    else:
                        error_text = await response.text()
                        logger.error(f"ValidationPlan generation failed: HTTP {response.status}")
                        logger.error(f"   Error: {error_text}")
                        
                        test_result = {
                            "c_id": request_data["c_id"],
                            "p_variation": request_data["p_variation"],
                            "p_taxonomy_level": request_data["p_taxonomy_level"],
                            "success": False,
                            "processing_time": processing_time,
                            "error": f"HTTP {response.status}: {error_text}"
                        }
                        self.validation_plan_results.append(test_result)
                        return False, {}
                        
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"ValidationPlan generation exception: {e}")
                
                test_result = {
                    "c_id": request_data["c_id"],
                    "p_variation": request_data["p_variation"],
                    "p_taxonomy_level": request_data["p_taxonomy_level"],
                    "success": False,
                    "processing_time": processing_time,
                    "error": str(e)
                }
                self.validation_plan_results.append(test_result)
                return False, {}

    async def test_single_question_generation(self, request_data: Dict[str, Any]):
        """Test single question generation with detailed analysis"""
        logger.info(f"Testing question: {request_data['topic']} ({request_data['difficulty']})")
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/generate-question",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=180)
                ) as response:
                    
                    processing_time = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        test_result = {
                            "topic": request_data["topic"],
                            "difficulty": request_data["difficulty"],
                            "success": True,
                            "processing_time": processing_time,
                            "iterations": result["iterations"],
                            "final_status": result["final_status"],
                            "parameter_validations": len(result["parameter_validations"]),
                            "csv_ready": bool(result.get("csv_ready"))
                        }
                        
                        logger.info(f"Question generated successfully!")
                        logger.info(f"   - Processing time: {processing_time:.2f}s")
                        logger.info(f"   - Iterations: {result['iterations']}")
                        logger.info(f"   - Final status: {result['final_status']}")
                        logger.info(f"   - Parameter validations: {len(result['parameter_validations'])}")
                        
                        # Analyze parameter validations
                        approved = sum(1 for v in result['parameter_validations'] if v['status'] == 'approved')
                        rejected = sum(1 for v in result['parameter_validations'] if v['status'] == 'rejected')
                        needs_refinement = sum(1 for v in result['parameter_validations'] if v['status'] == 'needs_refinement')
                        
                        logger.info(f"   - Parameter status: appro: {approved} rej: {rejected} need refine: {needs_refinement}")
                        
                        # Show question content summary
                        question_content = result.get("question_content", {})
                        if isinstance(question_content, dict):
                            aufgabe = question_content.get("aufgabenstellung", "N/A")
                            logger.info(f"   - Question preview: {aufgabe[:100]}...")
                        
                        self.test_results.append(test_result)
                        return True, result
                        
                    else:
                        error_text = await response.text()
                        logger.error(f"Question generation failed: HTTP {response.status}")
                        logger.error(f"   Error: {error_text}")
                        
                        test_result = {
                            "topic": request_data["topic"],
                            "difficulty": request_data["difficulty"], 
                            "success": False,
                            "processing_time": processing_time,
                            "error": f"HTTP {response.status}: {error_text}"
                        }
                        self.test_results.append(test_result)
                        return False, {}
                        
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"Question generation exception: {e}")
                
                test_result = {
                    "topic": request_data["topic"],
                    "difficulty": request_data["difficulty"],
                    "success": False, 
                    "processing_time": processing_time,
                    "error": str(e)
                }
                self.test_results.append(test_result)
                return False, {}
    
    async def test_parameter_coverage(self):
        """Test coverage of all parameter types"""
        logger.info("Testing parameter expert coverage...")
        
        parameter_test_cases = [
            {
                "topic": "Inflation und Deflation",
                "difficulty": "leicht",
                "focus": "variation_expert",
                "description": "Testing difficulty level assessment"
            },
            {
                "topic": "Wirtschaftskreislauf berechnen",
                "difficulty": "stammaufgabe", 
                "focus": "taxonomy_expert",
                "description": "Testing Bloom's taxonomy classification"
            },
            {
                "topic": "Zinssätze kalkulieren",
                "difficulty": "schwer",
                "focus": "math_expert", 
                "description": "Testing mathematical requirements"
            },
            {
                "topic": "Komplexe Marktanalyse mit verschiedenen nicht unwesentlichen Faktoren",
                "difficulty": "schwer",
                "focus": "obstacle_expert",
                "description": "Testing linguistic obstacle detection"
            }
        ]
        
        results = {}
        
        for test_case in parameter_test_cases:
            logger.info(f"🔍 Testing {test_case['focus']}: {test_case['description']}")
            
            success, result = await self.test_single_question_generation({
                "topic": test_case["topic"],
                "difficulty": test_case["difficulty"],
                "age_group": "9. Klasse"
            })
            
            if success:
                # Analyze which experts were called
                validations = result.get("parameter_validations", [])
                expert_usage = {}
                for validation in validations:
                    expert = validation.get("expert_used", "unknown")
                    expert_usage[expert] = expert_usage.get(expert, 0) + 1
                
                results[test_case['focus']] = {
                    "success": True,
                    "expert_usage": expert_usage,
                    "total_validations": len(validations)
                }
                
                logger.info(f"   Expert usage: {expert_usage}")
            else:
                results[test_case['focus']] = {"success": False}
                logger.error(f"   Failed to test {test_case['focus']}")
            
            await asyncio.sleep(2)  # Brief pause between tests
        
        return results
    
    async def test_batch_generation(self):
        """Test batch question generation"""
        logger.info("Testing batch generation...")
        
        batch_requests = [
            {"topic": "Angebot und Nachfrage", "difficulty": "leicht", "age_group": "9. Klasse"},
            {"topic": "Marktformen", "difficulty": "stammaufgabe", "age_group": "9. Klasse"},
            {"topic": "Wirtschaftspolitik", "difficulty": "schwer", "age_group": "9. Klasse"}
        ]
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/batch-generate",
                    json=batch_requests,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    
                    processing_time = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Batch generation successful!")
                        logger.info(f"   - Total time: {processing_time:.2f}s")
                        logger.info(f"   - Questions generated: {result['total_questions']}")
                        logger.info(f"   - Avg time per question: {processing_time/len(batch_requests):.2f}s")
                        return True
                    else:
                        logger.error(f"Batch generation failed: HTTP {response.status}")
                        return False
                        
            except Exception as e:
                logger.error(f"Batch generation exception: {e}")
                return False
    
    async def test_memory_efficiency(self):
        """Test memory management under load"""
        logger.info("Testing memory efficiency...")
        
        # Get initial memory state
        success, initial_status = await self.test_model_status_endpoint()
        if not success:
            return False
            
        initial_vram = initial_status.get("vram_usage_gb", 0)
        
        # Generate multiple questions to test memory management
        test_requests = [
            {"topic": f"Economics Topic {i}", "difficulty": ["leicht", "stammaufgabe", "schwer"][i % 3], "age_group": "9. Klasse"}
            for i in range(5)
        ]
        
        max_vram_seen = initial_vram
        
        for i, request in enumerate(test_requests):
            success, result = await self.test_single_question_generation(request)
            
            if success:
                # Check memory usage
                _, status = await self.test_model_status_endpoint()
                current_vram = status.get("vram_usage_gb", 0)
                max_vram_seen = max(max_vram_seen, current_vram)
                
                logger.info(f"   Question {i+1}: {current_vram:.1f}GB VRAM used")
            
            await asyncio.sleep(1)
        
        logger.info(f"Memory efficiency results:")
        logger.info(f"   - Initial VRAM: {initial_vram:.1f}GB")
        logger.info(f"   - Max VRAM seen: {max_vram_seen:.1f}GB") 
        logger.info(f"   - Memory overhead: {max_vram_seen - initial_vram:.1f}GB")
        
        # Check if we stayed within limits (18GB max)
        if max_vram_seen <= 18:
            logger.info("Memory management successful - stayed within limits")
            return True
        else:
            logger.warning(f"Memory usage exceeded limit: {max_vram_seen:.1f}GB > 18GB")
            return False
    
    async def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("Generating test report...")
        
        report = {
            "test_summary": {
                "total_tests": len(self.test_results),
                "successful_tests": sum(1 for t in self.test_results if t["success"]),
                "failed_tests": sum(1 for t in self.test_results if not t["success"]),
                "avg_processing_time": sum(t.get("processing_time", 0) for t in self.test_results) / len(self.test_results) if self.test_results else 0
            },
            "detailed_results": self.test_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to JSON
        report_path = Path("test_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save to CSV for easy analysis
        csv_path = Path("test_results.csv")
        if self.test_results:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.test_results[0].keys())
                writer.writeheader()
                writer.writerows(self.test_results)
        
        logger.info(f"Test report saved to: {report_path}")
        logger.info(f"Test results CSV saved to: {csv_path}")
        
        # Print summary
        summary = report["test_summary"]
        success_rate = (summary["successful_tests"] / summary["total_tests"] * 100) if summary["total_tests"] > 0 else 0
        
        print(f"""
            TEST SUMMARY:
           - Total tests: {summary['total_tests']}
           - Successful: {summary['successful_tests']}
           - Failed: {summary['failed_tests']}
           - Success rate: {success_rate:.1f}%
           - Avg processing time: {summary['avg_processing_time']:.2f}s
        """)
        
        return report
    
    def save_test_results(self, report: Dict[str, Any]):
        """Save test results using the result manager"""
        logger.info("Saving test results with result manager...")
        
        try:
            # Prepare CSV data from test results
            csv_data = self.test_results if self.test_results else [{"message": "No test results available"}]
            
            # Create metadata
            metadata = {
                "test_type": "comprehensive_system_test",
                "total_tests": report["test_summary"]["total_tests"],
                "successful_tests": report["test_summary"]["successful_tests"],
                "failed_tests": report["test_summary"]["failed_tests"],
                "success_rate_percent": (report["test_summary"]["successful_tests"] / report["test_summary"]["total_tests"] * 100) if report["test_summary"]["total_tests"] > 0 else 0,
                "avg_processing_time": report["test_summary"]["avg_processing_time"],
                "test_timestamp": report["timestamp"],
                "system_endpoint": self.base_url
            }
            
            # Save results with timestamped folder
            session_dir = save_results(csv_data, metadata)
            
            logger.info(f"Test results saved to: {session_dir}")
            logger.info(f"   - CSV results: {session_dir}/results.csv")
            logger.info(f"   - Prompts snapshot: {session_dir}/prompts/")
            logger.info(f"   - Metadata: {session_dir}/session_metadata.json")
            
        except Exception as e:
            logger.warning(f"Failed to save results with result_manager: {e}")
            logger.info("Continuing with traditional file saving...")
    
    async def run_comprehensive_tests(self):
        """Run all test suites"""
        logger.info("Starting comprehensive test suite...")
        
        # Test 1NutzenMathematischerDarstellungen: Health checks
        if not await self.test_health_endpoint():
            logger.error("Health check failed - aborting tests")
            return False
        
        # Test 2: Model status
        model_status_result = await self.test_model_status_endpoint()
        if not model_status_result[0]:
            logger.error("Model status check failed - aborting tests")
            return False
        
        # Test 3: Basic question generation
        basic_test_cases = [
            {"topic": "Grundbedürfnisse", "difficulty": "leicht", "age_group": "9. Klasse"},
            {"topic": "Marktmechanismen", "difficulty": "stammaufgabe", "age_group": "9. Klasse"},
            {"topic": "Makroökonomische Indikatoren", "difficulty": "schwer", "age_group": "9. Klasse"}
        ]
        
        logger.info("Running basic question generation tests...")
        for test_case in basic_test_cases:
            await self.test_single_question_generation(test_case)
            await asyncio.sleep(2)
        
        # Test 4: Parameter coverage
        await self.test_parameter_coverage()
        
        # Test 5: Batch generation
        await self.test_batch_generation()
        
        # Test 6: Memory efficiency
        await self.test_memory_efficiency()
        
        # Generate final report
        report = await self.generate_test_report()
        
        # Save results using result_manager
        self.save_test_results(report)
        
        logger.info("Comprehensive testing complete!")
        return report["test_summary"]["successful_tests"] > 0

async def main():
    """Main testing function"""
    tester = SystemTester()
    
    try:
        success = await tester.run_comprehensive_tests()
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