"""
Test Questions Loader

Utilities for loading test questions for evaluation.
"""

import json
import os
from typing import List, Dict, Any


def load_test_questions() -> List[Dict[str, str]]:
    """
    Load test questions from test_questions.json.
    
    Returns:
        List of question dictionaries with 'category' and 'question' keys
    """
    base_dir = os.path.dirname(__file__)
    test_file = os.path.join(base_dir, "test_questions.json")
    
    with open(test_file, "r") as f:
        data = json.load(f)
    
    questions = []
    for category, q_list in data.items():
        for question in q_list:
            questions.append({
                "category": category,
                "question": question,
            })
    
    return questions


def get_all_queries() -> List[str]:
    """
    Get all test queries from test_questions.json.
    
    Returns:
        List of query strings
    """
    questions = load_test_questions()
    return [q["question"] for q in questions]


def get_queries_by_category(category: str) -> List[str]:
    """
    Get test queries for a specific category.
    
    Args:
        category: Category name
    
    Returns:
        List of query strings for the category
    """
    questions = load_test_questions()
    return [q["question"] for q in questions if q["category"] == category]