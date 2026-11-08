import numpy as np
from typing import Dict, List, Optional


def compute_recall(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10
) -> float:
    n_queries = predicted.shape[0]
    k = min(k, predicted.shape[1], ground_truth.shape[1])

    recalls = []
    for i in range(n_queries):
        pred_set = set(predicted[i, :k])
        true_set = set(ground_truth[i, :k])
        recall = len(pred_set & true_set) / k
        recalls.append(recall)

    return np.mean(recalls) * 100


def compute_precision(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10
) -> float:
    return compute_recall(predicted, ground_truth, k)


def compute_mrr(
    predicted: np.ndarray,
    ground_truth: np.ndarray
) -> float:
    n_queries = predicted.shape[0]

    reciprocal_ranks = []
    for i in range(n_queries):
        hits = np.where(predicted[i] == ground_truth[i, 0])[0]
        reciprocal_ranks.append(1.0 / (hits[0] + 1) if len(hits) else 0.0)

    return np.mean(reciprocal_ranks)


def compute_ndcg(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10
) -> float:
    n_queries = predicted.shape[0]
    k = min(k, predicted.shape[1], ground_truth.shape[1])

    ndcg_scores = []
    for i in range(n_queries):
        true_set = set(ground_truth[i, :k])

        dcg = 0.0
        for j, pred_id in enumerate(predicted[i, :k]):
            if pred_id in true_set:
                dcg += 1.0 / np.log2(j + 2)

        idcg = sum(1.0 / np.log2(j + 2) for j in range(k))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores)


def compute_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    k_values: List[int] = [1, 5, 10, 20, 50, 100],
    latencies: Optional[np.ndarray] = None
) -> Dict:
    metrics = {}

    for k in k_values:
        if k <= predicted.shape[1] and k <= ground_truth.shape[1]:
            metrics[f'recall@{k}'] = compute_recall(predicted, ground_truth, k)

    metrics['mrr'] = compute_mrr(predicted, ground_truth)

    if 10 <= predicted.shape[1]:
        metrics['ndcg@10'] = compute_ndcg(predicted, ground_truth, 10)

    if latencies is not None:
        latencies_ms = latencies * 1000
        metrics['latency_mean_ms'] = np.mean(latencies_ms)
        metrics['latency_median_ms'] = np.median(latencies_ms)
        metrics['latency_p95_ms'] = np.percentile(latencies_ms, 95)
        metrics['latency_p99_ms'] = np.percentile(latencies_ms, 99)
        metrics['throughput_qps'] = 1000 / np.mean(latencies_ms)

    return metrics


def compute_distance_stats(distances: np.ndarray) -> Dict:
    return {
        'mean': float(np.mean(distances)),
        'std': float(np.std(distances)),
        'min': float(np.min(distances)),
        'max': float(np.max(distances)),
        'median': float(np.median(distances))
    }


class BenchmarkResult:

    def __init__(
        self,
        name: str,
        metrics: Dict,
        build_time: float,
        memory_mb: float
    ):
        self.name = name
        self.metrics = metrics
        self.build_time = build_time
        self.memory_mb = memory_mb

    def __repr__(self) -> str:
        recall = self.metrics.get('recall@10', 0)
        latency = self.metrics.get('latency_mean_ms', 0)
        return f"BenchmarkResult({self.name}: recall@10={recall:.1f}%, latency={latency:.3f}ms)"

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'metrics': self.metrics,
            'build_time': self.build_time,
            'memory_mb': self.memory_mb
        }

    def compare_to(self, baseline: 'BenchmarkResult') -> Dict:
        comparison = {}

        for key in self.metrics:
            if key.startswith('recall@'):
                baseline_val = baseline.metrics.get(key, 0)
                this_val = self.metrics.get(key, 0)
                comparison[f'{key}_diff'] = this_val - baseline_val

        baseline_lat = baseline.metrics.get('latency_mean_ms', 1)
        this_lat = self.metrics.get('latency_mean_ms', 1)
        comparison['speedup'] = baseline_lat / this_lat

        comparison['memory_ratio'] = self.memory_mb / baseline.memory_mb

        return comparison
