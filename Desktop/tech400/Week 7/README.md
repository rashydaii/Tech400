# CISI Information Retrieval Evaluation

## Project Overview

This project evaluates two Information Retrieval models, TF-IDF and BM25, using the CISI dataset.

## Evaluation Metrics

- Precision@10
- Recall@10
- F1@10
- Mean Average Precision (MAP)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@10)

## Requirements

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## How to Run

1. Extract the CISI dataset.
2. Place the dataset files in the project directory.
3. Update the dataset path in the Python script if necessary.
4. Run the evaluation script.
5. Review the generated CSV files and performance chart.

## Outputs

- `model_comparison.csv`: Average evaluation scores by model.
- `per_query_results.csv`: Evaluation results for individual queries.
- `model_performance.png`: Performance comparison chart.

## Academic Context

This project was developed for TECH 400: Introduction to Information Retrieval as an evaluation of the CISI retrieval project.

The numerical results should be reproduced by running the code against the dataset.
