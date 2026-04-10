# Retrieval Speed Evaluation Guide

## Overview

The Retrieval Speed Evaluation module measures and tracks the performance of the RAG (Retrieval-Augmented Generation) pipeline's retrieval components. It provides detailed latency metrics for:

- **Embedding Latency**: Time required to generate embeddings for queries
- **Vector Search Latency**: Time required to search the vector database
- **End-to-End (E2E) Latency**: Total time from query to retrieved results

All metrics are automatically logged to MLflow for tracking and comparison.

## Quick Start

### Run Full Dataset Evaluation

```bash
cd ottawa_chatbot
python -m scripts.run_retrieval_speed_eval
```

### Run Category-Specific Evaluation

```bash
python -m scripts.run_retrieval_speed_eval --category housing
python -m scripts.run_retrieval_speed_eval --category healthcare
```

### Run Per-Category Evaluation

```bash
python -m scripts.run_retrieval_speed_eval --all-categories
```

### Custom Parameters

```bash
python -m scripts.run_retrieval_speed_eval --top-k 5 --category jobs
```

## Programmatic Usage

### Basic Usage

```python
from backend.app.evaluation.retrieval_speed import run_retrieval_speed_evaluation

test_queries = [
    "How do I find housing in Ottawa?",
    "What healthcare services are available?",
]

results = run_retrieval_speed_evaluation(
    test_queries=test_queries,
    top_k=3,
    run_name="custom-retrieval-eval"
)

print(f"Mean E2E latency: {results['mean_e2e_latency_ms']:.2f} ms")
```

### With Category Filter

```python
results = run_retrieval_speed_evaluation(
    test_queries=test_queries,
    top_k=3,
    category="housing",
    run_name="housing-retrieval-eval",
    tags={"category": "housing", "version": "1.0"}
)
```

### Loading Test Questions

```python
from backend.app.evaluation.test_questions_loader import (
    get_all_queries,
    get_queries_by_category,
)

# Get all test queries
all_queries = get_all_queries()

# Get queries for specific category
housing_queries = get_queries_by_category("housing")
```

## Metrics Explained

### End-to-End Latency (E2E)
Total time from receiving a query to retrieving relevant documents. This includes:
- Time to generate query embedding
- Time to search vector database
- Time to collect and format results

**What it measures**: Overall retrieval system performance

### Embedding Latency
Time required by the embedding model (OpenAI text-embedding-3-small) to generate a vector representation of the query.

**What it measures**: LLM API performance and query encoding overhead

### Vector Search Latency
Time required to search the SQLite vector database and find the top-k most similar documents.

**What it measures**: Database query efficiency and vector similarity computation

## MLflow Integration

All metrics are automatically logged to MLflow. View results:

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Default tracking URI: `./mlruns`

### Logged Metrics

For each run, the following metrics are logged:

```
e2e_mean_latency_ms              - Mean end-to-end latency
e2e_median_latency_ms            - Median end-to-end latency
e2e_min_latency_ms               - Minimum end-to-end latency
e2e_max_latency_ms               - Maximum end-to-end latency
e2e_stdev_latency_ms             - Standard deviation of E2E latency
e2e_total_latency_ms             - Total time for all queries

embedding_mean_latency_ms        - Mean embedding latency
embedding_total_latency_ms       - Total embedding time

search_mean_latency_ms           - Mean vector search latency
search_total_latency_ms          - Total search time
```

### Logged Parameters

- `num_queries` - Number of queries evaluated
- `top_k` - Number of results retrieved
- `embedding_model` - Embedding model used
- `category` - Category filter (if specified)

### Logged Artifacts

Detailed latency breakdowns are saved as JSON artifacts:
- Individual latencies for each query
- Breakdown of embedding vs. search time
- Query text and category information

## Performance Benchmarks

### Expected Performance (baseline)

With 6 test queries and text-embedding-3-small model:
- **Embedding latency**: 300-600 ms per query
- **Vector search latency**: 10-50 ms per query
- **E2E latency**: 350-650 ms per query

*Note: These are approximate values and depend on:
- Network latency to OpenAI API
- SQLite database size
- Hardware specifications
- System load*

## Troubleshooting

### Issue: "No such file or directory: 'test_questions.json'"

**Solution**: Ensure you're running from the `ottawa_chatbot` directory:
```bash
cd ottawa_chatbot
python -m scripts.run_retrieval_speed_eval
```

### Issue: OpenAI API errors

**Solution**: Ensure environment variables are set:
```bash
export OPENAI_API_KEY=your_key_here
export EMBEDDING_MODEL=text-embedding-3-small
```

### Issue: MLflow not recording metrics

**Solution**: Check MLflow tracking URI:
```python
import mlflow
print(mlflow.get_tracking_uri())
```

## Integration with CI/CD

Add to your CI/CD pipeline:

```yaml
- name: Run Retrieval Speed Evaluation
  run: |
    cd ottawa_chatbot
    python -m scripts.run_retrieval_speed_eval --all-categories
```

Compare metrics across runs:
```yaml
- name: Compare with baseline
  run: python scripts/compare_retrieval_metrics.py --baseline main
```

## Advanced: Custom Evaluation

Measure specific components:

```python
from backend.app.evaluation.retrieval_speed import (
    measure_embedding_latency,
    measure_vector_search_latency,
)
from openai import OpenAI

client = OpenAI()

# Get embeddings
embeddings = [
    client.embeddings.create(
        model="text-embedding-3-small",
        input=q
    ).data[0].embedding
    for q in test_queries
]

# Measure embedding time
emb_metrics = measure_embedding_latency(test_queries)
print(f"Mean embedding latency: {emb_metrics['mean_latency_ms']:.2f} ms")

# Measure search time
search_metrics = measure_vector_search_latency(embeddings, top_k=3)
print(f"Mean search latency: {search_metrics['mean_latency_ms']:.2f} ms")
```

## See Also

- [Evaluator Module](evaluator.py) - Main evaluation orchestrator
- [Test Questions Loader](test_questions_loader.py) - Load evaluation data
- [Accuracy Report](accuracy_report.py) - Generate evaluation reports
