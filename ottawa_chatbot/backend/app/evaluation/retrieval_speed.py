"""
Retrieval Speed Evaluation Module

Measures and logs retrieval performance metrics:
- Embedding latency: Time to generate embeddings for queries
- Vector search latency: Time to search vector database
- Total retrieval time: End-to-end retrieval latency

All metrics are logged to MLflow for tracking and analysis.
"""

import time
import json
import statistics
from typing import Dict, List, Any, Optional
import os

from dotenv import load_dotenv
from openai import OpenAI
import mlflow
import mlflow.pytorch

load_dotenv()

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")


def setup_mlflow(experiment_name: str = "retrieval-speed-evaluation"):
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)


def measure_embedding_latency(
    queries: List[str],
    model: str = EMBED_MODEL
) -> Dict[str, Any]:
    """
    Measure latency for embedding generation.
    
    Args:
        queries: List of query strings to embed
        model: Embedding model to use
    
    Returns:
        Dictionary containing:
        - latencies: List of individual latencies (ms)
        - mean_latency_ms: Average latency
        - min_latency_ms: Minimum latency
        - max_latency_ms: Maximum latency
        - total_latency_ms: Total time for all queries
    """
    client = OpenAI()
    latencies = []
    
    for query in queries:
        start = time.time()
        client.embeddings.create(
            model=model,
            input=query
        )
        elapsed = (time.time() - start) * 1000  # Convert to milliseconds
        latencies.append(elapsed)
    
    return {
        "latencies": latencies,
        "mean_latency_ms": statistics.mean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "stdev_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "total_latency_ms": sum(latencies),
        "num_queries": len(queries),
    }


def measure_vector_search_latency(
    query_embeddings: List[List[float]],
    top_k: int = 3,
    category: Optional[str] = None
) -> Dict[str, Any]:
    """
    Measure latency for vector search operations.
    
    Args:
        query_embeddings: List of embedding vectors to search
        top_k: Number of results to return
        category: Optional category filter
    
    Returns:
        Dictionary containing:
        - latencies: List of individual search latencies (ms)
        - mean_latency_ms: Average search latency
        - min_latency_ms: Minimum search latency
        - max_latency_ms: Maximum search latency
        - total_latency_ms: Total time for all searches
    """
    from backend.vector_store_sqlite import query_similar
    
    latencies = []
    
    for embedding in query_embeddings:
        start = time.time()
        query_similar(
            query_embedding=embedding,
            top_k=top_k,
            category=category,
        )
        elapsed = (time.time() - start) * 1000  # Convert to milliseconds
        latencies.append(elapsed)
    
    return {
        "latencies": latencies,
        "mean_latency_ms": statistics.mean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "stdev_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "total_latency_ms": sum(latencies),
        "num_searches": len(query_embeddings),
        "top_k": top_k,
    }


def measure_end_to_end_retrieval_latency(
    queries: List[str],
    top_k: int = 3,
    category: Optional[str] = None,
    embedding_model: str = EMBED_MODEL
) -> Dict[str, Any]:
    """
    Measure end-to-end retrieval latency including both embedding and search.
    
    Args:
        queries: List of query strings
        top_k: Number of results to return
        category: Optional category filter
        embedding_model: Embedding model to use
    
    Returns:
        Dictionary containing:
        - e2e_latencies: List of end-to-end latencies (ms)
        - embedding_latencies: List of embedding latencies (ms)
        - search_latencies: List of search latencies (ms)
        - mean_e2e_latency_ms: Average end-to-end latency
        - mean_embedding_latency_ms: Average embedding latency
        - mean_search_latency_ms: Average search latency
        - And other statistics...
    """
    client = OpenAI()
    e2e_latencies = []
    embedding_latencies = []
    search_latencies = []
    
    for query in queries:
        # Measure embedding
        start_emb = time.time()
        embedding_response = client.embeddings.create(
            model=embedding_model,
            input=query
        )
        emb_latency = (time.time() - start_emb) * 1000
        embedding_latencies.append(emb_latency)
        
        # Measure search
        query_embedding = embedding_response.data[0].embedding
        
        from backend.vector_store_sqlite import query_similar
        start_search = time.time()
        query_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            category=category,
        )
        search_latency = (time.time() - start_search) * 1000
        search_latencies.append(search_latency)
        
        e2e_latency = emb_latency + search_latency
        e2e_latencies.append(e2e_latency)
    
    return {
        "e2e_latencies": e2e_latencies,
        "embedding_latencies": embedding_latencies,
        "search_latencies": search_latencies,
        "mean_e2e_latency_ms": statistics.mean(e2e_latencies),
        "median_e2e_latency_ms": statistics.median(e2e_latencies),
        "min_e2e_latency_ms": min(e2e_latencies),
        "max_e2e_latency_ms": max(e2e_latencies),
        "stdev_e2e_latency_ms": statistics.stdev(e2e_latencies) if len(e2e_latencies) > 1 else 0,
        "mean_embedding_latency_ms": statistics.mean(embedding_latencies),
        "mean_search_latency_ms": statistics.mean(search_latencies),
        "total_e2e_latency_ms": sum(e2e_latencies),
        "num_queries": len(queries),
        "top_k": top_k,
    }


def run_retrieval_speed_evaluation(
    test_queries: List[str],
    top_k: int = 3,
    category: Optional[str] = None,
    embedding_model: str = EMBED_MODEL,
    run_name: str = "retrieval-speed-eval",
    tags: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Run full retrieval speed evaluation and log metrics to MLflow.
    
    Args:
        test_queries: List of test queries
        top_k: Number of results to return
        category: Optional category filter
        embedding_model: Embedding model to use
        run_name: MLflow run name
        tags: Optional tags for MLflow run
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    setup_mlflow()
    
    with mlflow.start_run(run_name=run_name):
        # Set tags
        if tags:
            mlflow.set_tags(tags)
        
        # Log parameters
        mlflow.log_param("num_queries", len(test_queries))
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("embedding_model", embedding_model)
        if category:
            mlflow.log_param("category", category)
        
        # Run evaluations
        print(f"Running retrieval speed evaluation on {len(test_queries)} queries...")
        
        # End-to-end evaluation
        e2e_results = measure_end_to_end_retrieval_latency(
            test_queries,
            top_k=top_k,
            category=category,
            embedding_model=embedding_model,
        )
        
        # Log E2E metrics
        mlflow.log_metric("e2e_mean_latency_ms", e2e_results["mean_e2e_latency_ms"])
        mlflow.log_metric("e2e_median_latency_ms", e2e_results["median_e2e_latency_ms"])
        mlflow.log_metric("e2e_min_latency_ms", e2e_results["min_e2e_latency_ms"])
        mlflow.log_metric("e2e_max_latency_ms", e2e_results["max_e2e_latency_ms"])
        mlflow.log_metric("e2e_stdev_latency_ms", e2e_results["stdev_e2e_latency_ms"])
        mlflow.log_metric("e2e_total_latency_ms", e2e_results["total_e2e_latency_ms"])
        
        # Log embedding metrics
        mlflow.log_metric("embedding_mean_latency_ms", e2e_results["mean_embedding_latency_ms"])
        mlflow.log_metric("embedding_total_latency_ms", sum(e2e_results["embedding_latencies"]))
        
        # Log search metrics
        mlflow.log_metric("search_mean_latency_ms", e2e_results["mean_search_latency_ms"])
        mlflow.log_metric("search_total_latency_ms", sum(e2e_results["search_latencies"]))
        
        # Log detailed latency lists as JSON artifacts
        latency_details = {
            "e2e_latencies_ms": e2e_results["e2e_latencies"],
            "embedding_latencies_ms": e2e_results["embedding_latencies"],
            "search_latencies_ms": e2e_results["search_latencies"],
            "test_queries": test_queries,
        }
        
        with open("/tmp/latency_details.json", "w") as f:
            json.dump(latency_details, f, indent=2)
        
        mlflow.log_artifact("/tmp/latency_details.json", artifact_path="latencies")
        
        print(f"\n=== Retrieval Speed Evaluation Results ===")
        print(f"Queries evaluated: {e2e_results['num_queries']}")
        print(f"\nEnd-to-End Latency:")
        print(f"  Mean: {e2e_results['mean_e2e_latency_ms']:.2f} ms")
        print(f"  Median: {e2e_results['median_e2e_latency_ms']:.2f} ms")
        print(f"  Min: {e2e_results['min_e2e_latency_ms']:.2f} ms")
        print(f"  Max: {e2e_results['max_e2e_latency_ms']:.2f} ms")
        print(f"  StDev: {e2e_results['stdev_e2e_latency_ms']:.2f} ms")
        
        print(f"\nEmbedding Latency:")
        print(f"  Mean: {e2e_results['mean_embedding_latency_ms']:.2f} ms")
        
        print(f"\nVector Search Latency:")
        print(f"  Mean: {e2e_results['mean_search_latency_ms']:.2f} ms")
        
        print(f"\nMLflow run logged at: {mlflow.get_tracking_uri()}")
        
        return e2e_results


if __name__ == "__main__":
    # Example usage with test questions
    from backend.app.evaluation.test_questions import load_test_questions
    
    try:
        test_questions_data = load_test_questions()
        test_queries = [q.get("question", "") for q in test_questions_data]
    except:
        # Fallback test queries if loading fails
        test_queries = [
            "What is the process to apply for healthcare in Ottawa?",
            "How can I find housing in Ottawa?",
            "What job opportunities are available?",
            "Where can I study or get educational support?",
            "What transportation options exist in Ottawa?",
        ]
    
    results = run_retrieval_speed_evaluation(
        test_queries=test_queries,
        top_k=3,
        run_name="retrieval-speed-baseline",
        tags={"environment": "local", "version": "1.0"}
    )
