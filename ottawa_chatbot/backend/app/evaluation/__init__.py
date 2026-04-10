"""
Evaluation Module

Provides evaluation metrics and reporting for the Ottawa Chatbot RAG system.

Main Components:
- retrieval_speed: Measure and log retrieval latency metrics to MLflow
- evaluator: Orchestrate full evaluation suite
- test_questions_loader: Load test questions for evaluation
- accuracy_report: Generate evaluation reports
"""

from .retrieval_speed import (
    run_retrieval_speed_evaluation,
    measure_embedding_latency,
    measure_vector_search_latency,
    measure_end_to_end_retrieval_latency,
    setup_mlflow,
)

from .evaluator import run_full_evaluation

from .test_questions_loader import (
    load_test_questions,
    get_all_queries,
    get_queries_by_category,
)

from .accuracy_report import generate_retrieval_speed_report

__all__ = [
    "run_retrieval_speed_evaluation",
    "measure_embedding_latency",
    "measure_vector_search_latency",
    "measure_end_to_end_retrieval_latency",
    "setup_mlflow",
    "run_full_evaluation",
    "load_test_questions",
    "get_all_queries",
    "get_queries_by_category",
    "generate_retrieval_speed_report",
]
