"""
Evaluation Runner

Orchestrates various evaluation metrics including accuracy and retrieval speed.
"""

from .retrieval_speed import run_retrieval_speed_evaluation
from typing import List, Dict, Any, Optional


def run_full_evaluation(
    test_queries: List[str],
    top_k: int = 3,
    category: Optional[str] = None,
    run_name: str = "full-evaluation",
    tags: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Run full evaluation suite including retrieval speed metrics.
    
    Args:
        test_queries: List of test queries
        top_k: Number of results to return
        category: Optional category filter
        run_name: MLflow run name
        tags: Optional tags for MLflow run
    
    Returns:
        Dictionary containing all evaluation results
    """
    results = {
        "retrieval_speed": run_retrieval_speed_evaluation(
            test_queries=test_queries,
            top_k=top_k,
            category=category,
            run_name=run_name,
            tags=tags,
        )
    }
    
    return results\n
