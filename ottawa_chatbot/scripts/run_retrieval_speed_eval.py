#!/usr/bin/env python
"""
Retrieval Speed Evaluation Script

Run retrieval speed evaluation on test queries and log results to MLflow.

Usage:
    python -m backend.app.evaluation.run_retrieval_speed_eval
    python -m backend.app.evaluation.run_retrieval_speed_eval --category housing
    python -m backend.app.evaluation.run_retrieval_speed_eval --top-k 5 --all-categories
"""

import argparse
import sys
from typing import List, Dict, Any

from backend.app.evaluation.retrieval_speed import run_retrieval_speed_evaluation
from backend.app.evaluation.test_questions_loader import (
    get_all_queries,
    get_queries_by_category,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run retrieval speed evaluation"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieval results (default: 3)"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Category to evaluate (default: all)"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="retrieval-speed-eval",
        help="MLflow run name"
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Run evaluation separately for each category"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.all_categories:
        # Run evaluation for each category separately
        categories = [
            "housing",
            "transportation",
            "jobs",
            "study",
            "healthcare",
            "community_events",
        ]
        
        all_results = {}
        for category in categories:
            queries = get_queries_by_category(category)
            if not queries:
                print(f"No test queries found for category: {category}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Evaluating category: {category}")
            print(f"{'='*60}")
            
            results = run_retrieval_speed_evaluation(
                test_queries=queries,
                top_k=args.top_k,
                category=category,
                run_name=f"{args.run_name}-{category}",
                tags={
                    "category": category,
                    "evaluation_scope": "per_category",
                }
            )
            all_results[category] = results
        
        print(f"\n{'='*60}")
        print("Summary of all categories:")
        print(f"{'='*60}")
        for category, results in all_results.items():
            print(f"\n{category}:")
            print(f"  Mean E2E Latency: {results['mean_e2e_latency_ms']:.2f} ms")
            print(f"  Mean Embedding Latency: {results['mean_embedding_latency_ms']:.2f} ms")
            print(f"  Mean Search Latency: {results['mean_search_latency_ms']:.2f} ms")
    else:
        # Run evaluation on all queries or specific category
        if args.category:
            queries = get_queries_by_category(args.category)
            if not queries:
                print(f"No test queries found for category: {args.category}")
                sys.exit(1)
        else:
            queries = get_all_queries()
        
        print(f"Evaluating {len(queries)} queries...")
        results = run_retrieval_speed_evaluation(
            test_queries=queries,
            top_k=args.top_k,
            category=args.category,
            run_name=args.run_name,
            tags={
                "category": args.category or "all",
                "evaluation_scope": "full_dataset",
            }
        )


if __name__ == "__main__":
    main()
