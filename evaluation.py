
from pathlib import Path
from collections import Counter, defaultdict
import math
import random
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ==========================================================
# 1. PATHS AND SETTINGS
# ==========================================================

DATA_DIR = Path(
    r"C:\Users\Acer\Desktop\tech400\Week 7\cisi_dataset"
)

DOC_FILE = DATA_DIR / "CISI.ALL"
QUERY_FILE = DATA_DIR / "CISI.QRY"
REL_FILE = DATA_DIR / "CISI.REL"

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 10
RANDOM_SEED = 42
BOOTSTRAP_ITERATIONS = 10000
PERMUTATION_ITERATIONS = 20000

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

sns.set_theme(style="whitegrid")

# ==========================================================
# 2. LOAD CISI FILES
# ==========================================================

def parse_cisi(path):
    """Parse CISI records using .I, .T, .W, .B and .A fields."""
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Check DATA_DIR and ensure the CISI files exist."
        )

    records = {}
    current_id = None
    current_field = None
    fields = {}

    def save_record():
        if current_id is not None:
            text = " ".join(
                " ".join(fields.get(key, []))
                for key in ("T", "W", "B", "A")
            ).strip()
            records[str(current_id)] = text

    with open(path, "r", encoding="latin-1", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.rstrip()
            match = re.match(r"^\.(\w+)\s*(.*)$", line)

            if match:
                marker, value = match.groups()

                if marker == "I":
                    save_record()
                    current_id = (
                        value.strip().split()[0]
                        if value.strip() else None
                    )
                    fields = {}
                    current_field = None

                elif marker in {"T", "W", "B", "A"}:
                    current_field = marker
                    fields.setdefault(marker, [])
                    if value.strip():
                        fields[marker].append(value.strip())

                else:
                    current_field = None

            elif current_field is not None:
                fields.setdefault(current_field, []).append(line.strip())

    save_record()
    return records


def load_relevance(path):
    """
    Read query-document pairs from CISI.REL.

    This treats every valid listed pair as relevant. Confirm this assumption
    against the particular CISI.REL version supplied with your dataset.
    """
    if not path.exists():
        raise FileNotFoundError(f"Relevance file not found: {path}")

    relevance = defaultdict(set)

    with open(path, "r", encoding="latin-1", errors="ignore") as file:
        for line in file:
            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                qid = str(int(float(parts[0])))
                did = str(int(float(parts[1])))
                relevance[qid].add(did)
            except ValueError:
                continue

    return dict(relevance)


# ==========================================================
# 3. TEXT PREPROCESSING
# ==========================================================

stemmer = PorterStemmer()
STOP_WORDS = set(ENGLISH_STOP_WORDS)


def preprocess(text):
    """Lowercase, remove stop words, and apply Porter stemming."""
    tokens = re.findall(r"[a-zA-Z]+", str(text).lower())
    return [
        stemmer.stem(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]


# ==========================================================
# 4. RETRIEVAL MODELS
# ==========================================================

def build_bm25(tokenized_docs, k1=1.2, b=0.75):
    """Create a BM25 scoring function using preprocessed tokens."""
    n_docs = len(tokenized_docs)
    lengths = np.array([len(doc) for doc in tokenized_docs], dtype=float)
    avgdl = max(float(lengths.mean()), 1.0)

    term_counts = [Counter(doc) for doc in tokenized_docs]
    document_frequency = Counter()

    for counts in term_counts:
        document_frequency.update(counts.keys())

    def score(query, query_weighted=False, k3=8.0):
        query_counts = Counter(query)
        scores = np.zeros(n_docs, dtype=float)

        for term, query_frequency in query_counts.items():
            df = document_frequency.get(term, 0)
            if df == 0:
                continue

            idf = math.log(
                1.0 + (n_docs - df + 0.5) / (df + 0.5)
            )

            for i, counts in enumerate(term_counts):
                tf = counts.get(term, 0)
                if tf == 0:
                    continue

                denominator = tf + k1 * (
                    1.0 - b + b * lengths[i] / avgdl
                )

                contribution = (
                    idf * tf * (k1 + 1.0) / denominator
                )

                if query_weighted:
                    contribution *= (
                        (k3 + 1.0) * query_frequency
                        / (k3 + query_frequency)
                    )

                scores[i] += contribution

        return scores

    return score


def build_jm_scorer(tokenized_docs, smoothing=0.7):
    """Create a Jelinek-Mercer query-likelihood scorer."""
    document_lengths = np.array(
        [max(len(doc), 1) for doc in tokenized_docs]
    )

    document_counts = [Counter(doc) for doc in tokenized_docs]
    collection_counts = Counter()

    for doc in tokenized_docs:
        collection_counts.update(doc)

    collection_length = max(sum(collection_counts.values()), 1)

    def score(query):
        scores = np.zeros(len(tokenized_docs), dtype=float)

        for term in query:
            p_collection = (
                collection_counts.get(term, 0) / collection_length
            )

            if p_collection == 0:
                continue

            for i, counts in enumerate(document_counts):
                p_document = (
                    counts.get(term, 0) / document_lengths[i]
                )

                probability = (
                    smoothing * p_document
                    + (1.0 - smoothing) * p_collection
                )

                scores[i] += math.log(max(probability, 1e-300))

        return scores

    return score


def ranked_ids(scores, document_ids):
    """Convert scores to document IDs in descending score order."""
    order = np.argsort(-scores, kind="stable")
    return [document_ids[i] for i in order]


# ==========================================================
# 5. EVALUATION METRICS
# ==========================================================

def precision_at_k(ranked, relevant, k):
    return (
        sum(doc in relevant for doc in ranked[:k]) / k
        if k > 0 else 0.0
    )


def recall_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    return sum(doc in relevant for doc in ranked[:k]) / len(relevant)


def average_precision(ranked, relevant):
    if not relevant:
        return 0.0

    hits = 0
    total = 0.0

    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / rank

    return total / len(relevant)


def reciprocal_rank(ranked, relevant):
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def r_precision(ranked, relevant):
    if not relevant:
        return 0.0

    r = len(relevant)
    return sum(doc in relevant for doc in ranked[:r]) / r


def f1_at_k(ranked, relevant, k):
    p = precision_at_k(ranked, relevant, k)
    r = recall_at_k(ranked, relevant, k)
    return 2 * p * r / (p + r) if p + r else 0.0


def success_at_k(ranked, relevant, k):
    return float(any(doc in relevant for doc in ranked[:k]))


def dcg_at_k(ranked, relevant, k):
    return sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], start=1)
        if doc_id in relevant
    )


def ndcg_at_k(ranked, relevant, k):
    ideal_hits = min(len(relevant), k)

    if ideal_hits == 0:
        return 0.0

    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg_at_k(ranked, relevant, k) / ideal_dcg


def evaluate_ranking(ranked, relevant):
    return {
        "P@1": precision_at_k(ranked, relevant, 1),
        "P@5": precision_at_k(ranked, relevant, 5),
        "P@10": precision_at_k(ranked, relevant, 10),
        "P@20": precision_at_k(ranked, relevant, 20),
        "R@10": recall_at_k(ranked, relevant, 10),
        "R@100": recall_at_k(ranked, relevant, 100),
        "F1@10": f1_at_k(ranked, relevant, 10),
        "Success@10": success_at_k(ranked, relevant, 10),
        "MAP": average_precision(ranked, relevant),
        "MRR": reciprocal_rank(ranked, relevant),
        "R-precision": r_precision(ranked, relevant),
        "nDCG@5": ndcg_at_k(ranked, relevant, 5),
        "nDCG@10": ndcg_at_k(ranked, relevant, 10),
        "nDCG@20": ndcg_at_k(ranked, relevant, 20),
    }


# ==========================================================
# 6. STATISTICAL ANALYSIS
# ==========================================================

def bootstrap_ci(values, iterations=BOOTSTRAP_ITERATIONS):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(RANDOM_SEED)
    samples = rng.choice(
        values, size=(iterations, len(values)), replace=True
    )
    means = samples.mean(axis=1)

    return tuple(np.percentile(means, [2.5, 97.5]))


def paired_permutation_p(a, b, iterations=PERMUTATION_ITERATIONS):
    """Two-sided paired sign-flip permutation test."""
    differences = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    differences = differences[np.isfinite(differences)]

    if len(differences) == 0:
        return np.nan

    observed = abs(differences.mean())
    rng = np.random.default_rng(RANDOM_SEED)
    exceed = 0
    batch_size = 1000

    for start in range(0, iterations, batch_size):
        batch = min(batch_size, iterations - start)
        signs = rng.choice(
            [-1, 1], size=(batch, len(differences))
        )
        simulated = np.abs((signs * differences).mean(axis=1))
        exceed += int(np.sum(simulated >= observed))

    return (exceed + 1) / (iterations + 1)


def holm_adjust(p_values):
    """Apply Holm-Bonferroni correction to p-values."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)

    valid = np.where(np.isfinite(p_values))[0]
    order = valid[np.argsort(p_values[valid])]
    running_max = 0.0
    m = len(order)

    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[index])
        running_max = max(running_max, value)
        adjusted[index] = running_max

    return adjusted


# ==========================================================
# 7. EXTRA FIGURES: 5, 6, 8, 9, 10 AND 11
# ==========================================================

def create_extra_figures(
    documents, queries, relevance, rankings,
    tokenized_docs, tokenized_queries, output_dir
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    document_ids = list(documents)
    document_tokens = [
        tokenized_docs[doc_id] for doc_id in document_ids
    ]

    # Keep relevance IDs that exist in the loaded collection.
    valid_relevance = {
        qid: {
            did for did in relevant
            if did in documents
        }
        for qid, relevant in relevance.items()
        if qid in queries
    }
    valid_relevance = {
        qid: rel for qid, rel in valid_relevance.items() if rel
    }

    # ------------------------------------------------------
    # FIGURE 5: Oracle ranking and metric ceilings
    # ------------------------------------------------------
    oracle_p10 = []
    oracle_ndcg10 = []
    actual_ndcg10 = []

    reference_model = "BM25-weighted"

    for qid, relevant in valid_relevance.items():
        ideal_ranking = (
            list(relevant)
            + [doc_id for doc_id in document_ids if doc_id not in relevant]
        )

        oracle_p10.append(precision_at_k(ideal_ranking, relevant, 10))
        oracle_ndcg10.append(ndcg_at_k(ideal_ranking, relevant, 10))

        actual_ranking = rankings[reference_model].get(qid, [])
        actual_ndcg10.append(
            ndcg_at_k(actual_ranking, relevant, 10)
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [
        "Ideal P@10",
        "Ideal nDCG@10",
        "BM25 nDCG@10",
    ]
    values = [
        np.mean(oracle_p10),
        np.mean(oracle_ndcg10),
        np.mean(actual_ndcg10),
    ]

    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean score")
    ax.set_title("Figure 5. Oracle Ranking and Metric Ceilings")
    fig.tight_layout()
    fig.savefig(output_dir / "figure5_oracle_ceilings.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------
    # FIGURE 6: Statistical evaluation workflow
    # ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.axis("off")

    workflow = [
        ("Per-query\nmetric scores", 0.10),
        ("Bootstrap confidence\nintervals", 0.30),
        ("Paired permutation\ntests", 0.50),
        ("Holm-Bonferroni\ncorrection", 0.70),
        ("Interpret and\nreport results", 0.90),
    ]

    for label, x in workflow:
        ax.text(
            x, 0.52, label,
            ha="center", va="center", fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="white",
                edgecolor="black"
            ),
            transform=ax.transAxes
        )

    for i in range(len(workflow) - 1):
        ax.annotate(
            "",
            xy=(workflow[i + 1][1] - 0.075, 0.52),
            xytext=(workflow[i][1] + 0.075, 0.52),
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", lw=1.3)
        )

    ax.set_title("Figure 6. Statistical Evaluation Workflow")
    fig.tight_layout()
    fig.savefig(output_dir / "figure6_statistical_workflow.png", dpi=300)
    plt.close(fig)

    # Calculate per-query AP for all models.
    rows = []

    for qid, relevant in valid_relevance.items():
        row = {
            "Query_ID": qid,
            "Query_Length": len(tokenized_queries[qid])
        }

        for model_name in rankings:
            ranked = rankings[model_name].get(qid, [])
            row[model_name] = average_precision(ranked, relevant)

        rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        output_dir / "extra_figure_query_data.csv", index=False
    )

    # ------------------------------------------------------
    # FIGURE 8: Query length and retrieval difference
    # ------------------------------------------------------
    comparison["MAP_Difference"] = (
        comparison["BM25-weighted"] - comparison["TF-IDF"]
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        comparison["Query_Length"],
        comparison["MAP_Difference"],
        alpha=0.7
    )
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Query length (preprocessed terms)")
    ax.set_ylabel("BM25 MAP - TF-IDF MAP")
    ax.set_title(
        "Figure 8. Query Length and Retrieval Performance Differences"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "figure8_query_length_difference.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------
    # FIGURE 9: BM25 and TF-IDF on short and long queries
    # ------------------------------------------------------
    median_length = comparison["Query_Length"].median()

    comparison["Query_Group"] = np.where(
        comparison["Query_Length"] <= median_length,
        "Short queries",
        "Long queries"
    )

    grouped = comparison.groupby("Query_Group")[
        ["TF-IDF", "BM25-weighted"]
    ].mean().reindex(["Short queries", "Long queries"])

    ax = grouped.plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Mean Average Precision")
    ax.set_xlabel("")
    ax.set_title(
        "Figure 9. BM25 and TF-IDF Across Short and Long Queries"
    )
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Retrieval model")
    plt.tight_layout()
    plt.savefig(output_dir / "figure9_short_long_queries.png", dpi=300)
    plt.close()

    grouped.to_csv(output_dir / "figure9_short_long_summary.csv")

    # ------------------------------------------------------
    # FIGURE 10: BM25 parameter heat map
    # ------------------------------------------------------
    k1_values = [0.6, 0.9, 1.2, 1.5, 1.8, 2.0]
    b_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    heatmap = np.full(
        (len(b_values), len(k1_values)), np.nan
    )

    for i, b_value in enumerate(b_values):
        for j, k1_value in enumerate(k1_values):
            scorer = build_bm25(
                document_tokens,
                k1=k1_value,
                b=b_value
            )

            aps = []

            for qid, relevant in valid_relevance.items():
                scores = scorer(tokenized_queries[qid])
                ranked = ranked_ids(scores, document_ids)
                aps.append(average_precision(ranked, relevant))

            heatmap[i, j] = np.mean(aps) if aps else np.nan

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        heatmap,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=k1_values,
        yticklabels=b_values,
        ax=ax
    )

    ax.set_xlabel("BM25 k1")
    ax.set_ylabel("BM25 b")
    ax.set_title(
        "Figure 10. Effect of BM25 Parameter Combinations on MAP"
    )

    fig.tight_layout()
    fig.savefig(
        output_dir / "figure10_bm25_parameter_heatmap.png",
        dpi=300
    )
    plt.close(fig)

    pd.DataFrame(
        heatmap,
        index=b_values,
        columns=k1_values
    ).to_csv(output_dir / "figure10_bm25_parameter_results.csv")

    # ------------------------------------------------------
    # FIGURE 11: Jelinek-Mercer smoothing parameter
    # ------------------------------------------------------
    lambdas = np.arange(0.1, 1.0, 0.1)
    jm_map = []

    for lam in lambdas:
        scorer = build_jm_scorer(document_tokens, smoothing=lam)
        aps = []

        for qid, relevant in valid_relevance.items():
            scores = scorer(tokenized_queries[qid])
            ranked = ranked_ids(scores, document_ids)
            aps.append(average_precision(ranked, relevant))

        jm_map.append(np.mean(aps) if aps else np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lambdas, jm_map, marker="o")
    ax.set_xlabel("Jelinek-Mercer smoothing parameter (λ)")
    ax.set_ylabel("Mean Average Precision (MAP)")
    ax.set_xticks(lambdas)
    ax.set_title(
        "Figure 11. Effect of JM Smoothing on Retrieval Performance"
    )

    fig.tight_layout()
    fig.savefig(output_dir / "figure11_jm_smoothing.png", dpi=300)
    plt.close(fig)

    pd.DataFrame({
        "lambda": lambdas,
        "MAP": jm_map
    }).to_csv(
        output_dir / "figure11_jm_smoothing_results.csv",
        index=False
    )

    print("\nAdditional figures generated:")
    for name in [
        "figure5_oracle_ceilings.png",
        "figure6_statistical_workflow.png",
        "figure8_query_length_difference.png",
        "figure9_short_long_queries.png",
        "figure10_bm25_parameter_heatmap.png",
        "figure11_jm_smoothing.png",
    ]:
        print(" -", name)


# ==========================================================
# 8. MAIN PROGRAM
# ==========================================================

def main():
    print("Loading CISI dataset...")

    documents = parse_cisi(DOC_FILE)
    queries = parse_cisi(QUERY_FILE)
    relevance = load_relevance(REL_FILE)

    # Keep only query and document IDs present in the dataset.
    documents = {str(k): v for k, v in documents.items()}
    queries = {str(k): v for k, v in queries.items()}

    relevance = {
        str(qid): {
            str(did) for did in dids if str(did) in documents
        }
        for qid, dids in relevance.items()
        if str(qid) in queries
    }

    relevance = {
        qid: rel for qid, rel in relevance.items() if rel
    }

    print("Documents loaded:", len(documents))
    print("Queries loaded:", len(queries))
    print("Queries with usable relevance judgments:", len(relevance))
    print(
        "Relevant query-document pairs:",
        sum(len(rel) for rel in relevance.values())
    )

    if not documents or not queries or not relevance:
        raise ValueError(
            "No usable CISI data was loaded. Check the file paths, "
            "CISI format, and query/document IDs."
        )

    document_ids = list(documents)
    raw_documents = [documents[doc_id] for doc_id in document_ids]

    tokenized_docs = {
        doc_id: preprocess(documents[doc_id])
        for doc_id in document_ids
    }

    tokenized_queries = {
        qid: preprocess(text)
        for qid, text in queries.items()
    }

    tokenized_document_list = [
        tokenized_docs[doc_id] for doc_id in document_ids
    ]

    # TF-IDF model.
    vectorizer = TfidfVectorizer(
        tokenizer=preprocess,
        token_pattern=None,
        lowercase=False
    )

    tfidf_matrix = vectorizer.fit_transform(raw_documents)

    # BM25 models.
    bm25_scorer = build_bm25(tokenized_document_list)
    bm25_weighted_scorer = build_bm25(tokenized_document_list)

    # Language models.
    jm01_scorer = build_jm_scorer(
        tokenized_document_list, smoothing=0.1
    )
    jm07_scorer = build_jm_scorer(
        tokenized_document_list, smoothing=0.7
    )

    model_names = [
        "Random",
        "TF-IDF",
        "BM25-set",
        "BM25-weighted",
        "LM-JM-0.1",
        "LM-JM-0.7"
    ]

    rankings = {name: {} for name in model_names}
    per_query_rows = []

    rng = np.random.default_rng(RANDOM_SEED)

    print("\nEvaluating retrieval models...")

    for qid, relevant in relevance.items():
        query_tokens = tokenized_queries[qid]

        query_vector = vectorizer.transform([queries[qid]])
        tfidf_scores = cosine_similarity(
            query_vector, tfidf_matrix
        ).ravel()

        score_map = {
            "Random": rng.random(len(document_ids)),
            "TF-IDF": tfidf_scores,
            "BM25-set": bm25_scorer(
                list(dict.fromkeys(query_tokens))
            ),
            "BM25-weighted": bm25_weighted_scorer(
                query_tokens, query_weighted=True
            ),
            "LM-JM-0.1": jm01_scorer(query_tokens),
            "LM-JM-0.7": jm07_scorer(query_tokens),
        }

        for model_name, scores in score_map.items():
            ranked = ranked_ids(scores, document_ids)
            rankings[model_name][qid] = ranked

            row = {
                "query_id": qid,
                "query_length": len(query_tokens),
                "model": model_name
            }

            row.update(evaluate_ranking(ranked, relevant))
            per_query_rows.append(row)

    detail = pd.DataFrame(per_query_rows)

    detail.to_csv(
        OUTPUT_DIR / "per_query_results.csv",
        index=False
    )

    metric_columns = [
        "P@1", "P@5", "P@10", "P@20",
        "R@10", "R@100", "F1@10", "Success@10",
        "MAP", "MRR", "R-precision",
        "nDCG@5", "nDCG@10", "nDCG@20"
    ]

    summary = detail.groupby("model")[metric_columns].mean()
    summary = summary.reindex(model_names)

    summary.to_csv(OUTPUT_DIR / "overall_results.csv")

    print("\nOVERALL RESULTS")
    print(summary.round(4).to_string())

    # ------------------------------------------------------
    # Bootstrap confidence intervals
    # ------------------------------------------------------
    ci_rows = []

    for model_name in model_names:
        values = detail.loc[
            detail["model"] == model_name, "MAP"
        ].values

        low, high = bootstrap_ci(values)

        ci_rows.append({
            "model": model_name,
            "MAP": np.mean(values),
            "CI_lower": low,
            "CI_upper": high
        })

    pd.DataFrame(ci_rows).to_csv(
        OUTPUT_DIR / "bootstrap_confidence_intervals.csv",
        index=False
    )

    # ------------------------------------------------------
    # Paired permutation tests and Holm correction
    # ------------------------------------------------------
    test_rows = []
    reference_model = "BM25-weighted"

    for metric in ["MAP", "P@10", "nDCG@10"]:
        wide = detail.pivot(
            index="query_id",
            columns="model",
            values=metric
        ).dropna()

        pending = []

        for model_name in model_names:
            if model_name == reference_model:
                continue

            p_value = paired_permutation_p(
                wide[reference_model].values,
                wide[model_name].values
            )

            pending.append({
                "metric": metric,
                "reference": reference_model,
                "comparison": model_name,
                "mean_difference": (
                    wide[reference_model] - wide[model_name]
                ).mean(),
                "p_raw": p_value
            })

        adjusted = holm_adjust([
            row["p_raw"] for row in pending
        ])

        for row, adjusted_p in zip(pending, adjusted):
            row["p_holm"] = adjusted_p
            test_rows.append(row)

    pd.DataFrame(test_rows).to_csv(
        OUTPUT_DIR / "statistical_tests.csv",
        index=False
    )

    # ------------------------------------------------------
    # Additional query-length analysis
    # ------------------------------------------------------
    query_analysis = detail.pivot(
        index="query_id",
        columns="model",
        values=["query_length", "MAP", "nDCG@10"]
    )

    qlen = query_analysis["query_length"].iloc[:, 0]
    bm25_map = query_analysis["MAP"]["BM25-weighted"]
    tfidf_map = query_analysis["MAP"]["TF-IDF"]

    length_df = pd.DataFrame({
        "query_length": qlen,
        "BM25_MAP": bm25_map,
        "TFIDF_MAP": tfidf_map,
    })

    length_df["MAP_difference"] = (
        length_df["BM25_MAP"] - length_df["TFIDF_MAP"]
    )

    length_df.to_csv(
        OUTPUT_DIR / "query_length_analysis.csv"
    )

    if len(length_df) >= 3:
        rho, p_value = spearmanr(
            length_df["query_length"],
            length_df["MAP_difference"]
        )

        print(
            "\nQuery length vs BM25-TF-IDF MAP difference:",
            f"Spearman rho={rho:.4f}, p={p_value:.4f}"
        )

    # ------------------------------------------------------
    # Core comparison figures
    # ------------------------------------------------------
    ax = summary["MAP"].plot(
        kind="bar", figsize=(9, 5)
    )
    ax.set_title("Mean Average Precision by Retrieval Model")
    ax.set_ylabel("MAP")
    ax.set_xlabel("Retrieval model")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "map_comparison.png", dpi=300)
    plt.close()

    ax = summary[
        ["P@10", "R@10", "F1@10", "Success@10"]
    ].plot(kind="bar", figsize=(10, 5))

    ax.set_title("Precision, Recall, F1 and Success at 10")
    ax.set_ylabel("Mean score")
    ax.set_xlabel("Retrieval model")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "metric_comparison.png", dpi=300)
    plt.close()

    ax = summary[
        ["MRR", "R-precision", "nDCG@10"]
    ].plot(kind="bar", figsize=(10, 5))

    ax.set_title("Ranking Quality Comparison")
    ax.set_ylabel("Mean score")
    ax.set_xlabel("Retrieval model")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ranking_metrics.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        length_df["query_length"],
        length_df["MAP_difference"],
        alpha=0.7
    )
    ax.axhline(0, linestyle="--")
    ax.set_title("Query Length vs BM25-TF-IDF MAP Difference")
    ax.set_xlabel("Query length (processed terms)")
    ax.set_ylabel("BM25 MAP - TF-IDF MAP")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "query_length_plot.png", dpi=300)
    plt.close()

    # ------------------------------------------------------
    # Generate Figures 5, 6, 8, 9, 10 and 11
    # ------------------------------------------------------
    create_extra_figures(
        documents=documents,
        queries=queries,
        relevance=relevance,
        rankings=rankings,
        tokenized_docs=tokenized_docs,
        tokenized_queries=tokenized_queries,
        output_dir=OUTPUT_DIR
    )

    print("\nFinished successfully.")
    print("Results folder:", OUTPUT_DIR)
    print("CSV tables and PNG figures have been saved.")


if __name__ == "__main__":
    main()
