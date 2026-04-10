"""
Evaluation Reporting

Generates evaluation reports including accuracy metrics and retrieval speed performance.

See retrieval_speed.py for retrieval speed evaluation metrics.
See evaluator.py to run full evaluation suite.
"""

import json
from typing import Dict, List, Any
from datetime import datetime


def generate_retrieval_speed_report(metrics: Dict[str, Any]) -> str:
    """
    Generate a formatted report for retrieval speed metrics.
    
    Args:
        metrics: Dictionary of retrieval speed metrics from run_retrieval_speed_evaluation
    
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 70)
    report.append("RETRIEVAL SPEED EVALUATION REPORT")
    report.append("=" * 70)
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("")
    
    report.append("END-TO-END RETRIEVAL LATENCY")
    report.append("-" * 70)
    report.append(f"  Number of Queries: {metrics.get('num_queries', 'N/A')}")
    report.append(f"  Mean Latency: {metrics.get('mean_e2e_latency_ms', 'N/A'):.2f} ms")
    report.append(f"  Median Latency: {metrics.get('median_e2e_latency_ms', 'N/A'):.2f} ms")
    report.append(f"  Min Latency: {metrics.get('min_e2e_latency_ms', 'N/A'):.2f} ms")
    report.append(f"  Max Latency: {metrics.get('max_e2e_latency_ms', 'N/A'):.2f} ms")
    report.append(f"  Std Dev: {metrics.get('stdev_e2e_latency_ms', 'N/A'):.2f} ms")
    report.append(f"  Total Latency: {metrics.get('total_e2e_latency_ms', 'N/A'):.2f} ms")
    report.append("")
    
    report.append("EMBEDDING GENERATION LATENCY")
    report.append("-" * 70)
    report.append(f"  Mean Latency: {metrics.get('mean_embedding_latency_ms', 'N/A'):.2f} ms")
    report.append(f"  Total Latency: {sum(metrics.get('embedding_latencies', [])):.2f} ms")
    report.append("")
    
    report.append("VECTOR SEARCH LATENCY")
    report.append("-" * 70)
    report.append(f"  Mean Latency: {metrics.get('mean_search_latency_ms', 'N/A'):.2f} ms")
    report.append(f"  Top-K Results: {metrics.get('top_k', 'N/A')}")
    report.append(f"  Total Latency: {sum(metrics.get('search_latencies', [])):.2f} ms")
    report.append("")
    
    report.append("=" * 70)
    
    return "\n".join(report)\n
