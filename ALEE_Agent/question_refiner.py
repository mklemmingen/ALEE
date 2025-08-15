#!/usr/bin/env python3
"""
Question Refinement Decision Engine
Determines whether to apply expert suggestions and applies them safely
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple

from models import BaseQuestionModel, ExpertSuggestion

logger = logging.getLogger(__name__)


class RefinementDecision:
    """Result of refinement decision analysis"""
    def __init__(self, should_refine: bool, reasoning: str, confidence: float, 
                 priority_suggestions: List[ExpertSuggestion] = None):
        self.should_refine = should_refine
        self.reasoning = reasoning
        self.confidence = confidence
        self.priority_suggestions = priority_suggestions or []


class QuestionRefiner:
    """Refines questions based on expert suggestions with format preservation"""
    
    def __init__(self, rating_threshold: float = 3.0, consensus_threshold: float = 0.6):
        self.rating_threshold = rating_threshold
        self.consensus_threshold = consensus_threshold
        
    def analyze_refinement_need(self, question: BaseQuestionModel) -> RefinementDecision:
        """Analyze if question needs refinement based on expert feedback"""
        
        if not question.expert_suggestions:
            return RefinementDecision(
                should_refine=False,
                reasoning="No expert suggestions available",
                confidence=0.0
            )
        
        # Calculate metrics
        avg_rating = question.get_average_rating()
        low_ratings = [s for s in question.expert_suggestions if s.rating < self.rating_threshold]
        total_suggestions = len([s for s in question.expert_suggestions if s.suggestions])
        
        # Decision logic
        if avg_rating < self.rating_threshold:
            if len(low_ratings) >= 3:  # Majority of experts agree
                return RefinementDecision(
                    should_refine=True,
                    reasoning=f"Average rating {avg_rating:.1f} below threshold, {len(low_ratings)} experts suggest improvements",
                    confidence=0.8,
                    priority_suggestions=low_ratings
                )
            elif len(low_ratings) >= 2:  # Some expert consensus
                return RefinementDecision(
                    should_refine=True,
                    reasoning=f"Average rating {avg_rating:.1f} below threshold, {len(low_ratings)} experts suggest improvements",
                    confidence=0.6,
                    priority_suggestions=low_ratings
                )
            else:
                return RefinementDecision(
                    should_refine=False,
                    reasoning=f"Average rating {avg_rating:.1f} below threshold but insufficient expert consensus",
                    confidence=0.3
                )
        
        # Check for specific improvement suggestions even with good ratings
        critical_suggestions = [s for s in question.expert_suggestions 
                              if any(keyword in ' '.join(s.suggestions).lower() 
                                   for keyword in ['format', 'struktur', 'fehler', 'falsch'])]
        
        if critical_suggestions:
            return RefinementDecision(
                should_refine=True,
                reasoning=f"Critical format/structure issues identified by {len(critical_suggestions)} experts",
                confidence=0.7,
                priority_suggestions=critical_suggestions
            )
        
        return RefinementDecision(
            should_refine=False,
            reasoning=f"Average rating {avg_rating:.1f} meets threshold, no critical issues",
            confidence=0.9
        )
    
    def refine_question(self, question: BaseQuestionModel) -> BaseQuestionModel:
        """Apply expert suggestions to question with format preservation"""
        
        # Store original for rollback
        original_question_text = question.question_text
        question.original_question = original_question_text
        
        decision = self.analyze_refinement_need(question)
        
        if not decision.should_refine:
            question.refinement_reasoning = decision.reasoning
            return question
        
        try:
            # Apply suggestions by category
            refined_question = self._apply_content_suggestions(question, decision.priority_suggestions)
            refined_question = self._apply_format_suggestions(refined_question, decision.priority_suggestions)
            
            # Validate format preservation
            if not refined_question.validate_format_compliance():
                logger.warning(f"Refinement broke format compliance for {question.question_type}, reverting")
                question.question_text = original_question_text
                question.refinement_reasoning = "Refinement reverted due to format compliance failure"
                question.format_preserved = False
                return question
            
            # Mark successful refinement
            refined_question.refinement_applied = True
            refined_question.refinement_reasoning = decision.reasoning
            refined_question.format_preserved = True
            
            # Mark applied suggestions
            for suggestion in decision.priority_suggestions:
                suggestion.applied = True
            
            logger.info(f"Successfully refined {question.question_type} question based on {len(decision.priority_suggestions)} expert suggestions")
            return refined_question
            
        except Exception as e:
            logger.error(f"Error during refinement: {e}, reverting to original")
            question.question_text = original_question_text
            question.refinement_reasoning = f"Refinement failed due to error: {e}"
            question.format_preserved = True
            return question
    
    def _apply_content_suggestions(self, question: BaseQuestionModel, 
                                 suggestions: List[ExpertSuggestion]) -> BaseQuestionModel:
        """Apply content-related suggestions without changing format"""
        
        content_suggestions = []
        for suggestion in suggestions:
            for sug in suggestion.suggestions:
                if any(keyword in sug.lower() for keyword in 
                      ['inhalt', 'klarheit', 'verständlich', 'begriff', 'wording', 'formulierung']):
                    content_suggestions.append(sug)
        
        if not content_suggestions:
            return question
        
        # Apply content improvements (simplified approach)
        # In a full implementation, this would use NLP to apply specific content changes
        logger.info(f"Content suggestions available for {question.question_type}: {len(content_suggestions)}")
        
        # For now, we log the suggestions but don't automatically apply them
        # This ensures format preservation while tracking improvement opportunities
        
        return question
    
    def _apply_format_suggestions(self, question: BaseQuestionModel, 
                                suggestions: List[ExpertSuggestion]) -> BaseQuestionModel:
        """Apply format-related suggestions with strict validation"""
        
        format_suggestions = []
        for suggestion in suggestions:
            for sug in suggestion.suggestions:
                if any(keyword in sug.lower() for keyword in 
                      ['format', 'struktur', 'markup', 'tag', 'option']):
                    format_suggestions.append(sug)
        
        if not format_suggestions:
            return question
        
        # Apply safe format improvements
        question_text = question.question_text
        
        # Fix common markup issues while preserving structure
        if question.question_type in ["multiple-choice", "single-choice"]:
            # Ensure proper option spacing
            question_text = re.sub(r'<option>\s*', '<option>', question_text)
            question_text = re.sub(r'\s*</option>', '</option>', question_text)
            
        elif question.question_type == "true-false":
            # Ensure proper true-false spacing
            question_text = re.sub(r'<true-false>\s*', '<true-false>', question_text)
            question_text = re.sub(r'\s*</true-false>', '</true-false>', question_text)
            
        elif question.question_type == "mapping":
            # Ensure proper mapping tag spacing
            question_text = re.sub(r'<start-option>\s*', '<start-option>', question_text)
            question_text = re.sub(r'<end-option>\s*', '<end-option>', question_text)
        
        question.question_text = question_text
        
        logger.info(f"Applied format improvements for {question.question_type}")
        return question
    
    def get_refinement_summary(self, questions: List[BaseQuestionModel]) -> Dict[str, Any]:
        """Generate summary of refinement decisions and outcomes"""
        
        total_questions = len(questions)
        refined_questions = [q for q in questions if q.refinement_applied]
        format_preserved = [q for q in questions if q.format_preserved]
        
        expert_suggestion_counts = {}
        for question in questions:
            for suggestion in question.expert_suggestions:
                expert_type = suggestion.expert_type
                expert_suggestion_counts[expert_type] = expert_suggestion_counts.get(expert_type, 0) + 1
        
        avg_ratings_by_type = {}
        for question in questions:
            q_type = question.question_type
            if q_type not in avg_ratings_by_type:
                avg_ratings_by_type[q_type] = []
            avg_ratings_by_type[q_type].append(question.get_average_rating())
        
        # Calculate averages
        for q_type in avg_ratings_by_type:
            ratings = avg_ratings_by_type[q_type]
            avg_ratings_by_type[q_type] = sum(ratings) / len(ratings) if ratings else 0.0
        
        return {
            'total_questions': total_questions,
            'questions_refined': len(refined_questions),
            'refinement_rate': len(refined_questions) / total_questions if total_questions > 0 else 0.0,
            'format_preservation_rate': len(format_preserved) / total_questions if total_questions > 0 else 0.0,
            'expert_suggestion_counts': expert_suggestion_counts,
            'average_ratings_by_type': avg_ratings_by_type,
            'questions_needing_attention': [
                q.question_type for q in questions 
                if q.get_average_rating() < self.rating_threshold and not q.refinement_applied
            ]
        }


class BatchQuestionRefiner:
    """Handles refinement of multiple questions with batch optimization"""
    
    def __init__(self, refiner: Optional[QuestionRefiner] = None):
        self.refiner = refiner or QuestionRefiner()
    
    def refine_questions(self, questions: List[BaseQuestionModel]) -> Tuple[List[BaseQuestionModel], Dict[str, Any]]:
        """Refine multiple questions and return results with summary"""
        
        refined_questions = []
        
        for question in questions:
            try:
                refined_question = self.refiner.refine_question(question)
                refined_questions.append(refined_question)
            except Exception as e:
                logger.error(f"Failed to refine question {question.question_type}: {e}")
                # Add original question if refinement fails
                question.refinement_reasoning = f"Refinement failed: {e}"
                refined_questions.append(question)
        
        # Generate comprehensive summary
        summary = self.refiner.get_refinement_summary(refined_questions)
        
        return refined_questions, summary


# Export main classes
__all__ = ['QuestionRefiner', 'BatchQuestionRefiner', 'RefinementDecision']