"""
Evaluation Metrics for ZGQ v8
==============================

Comprehensive evaluation metrics for ANNS including:
- Recall@k
- Precision@k
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_recall(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10
) -> float:
    """
    Compute recall@k.
    
    Recall@k measures the proportion of true nearest neighbors
    that appear in the top-k predictions.
    
    Args:
        predicted: Predicted neighbor IDs of shape (n_queries, k_pred)
        ground_truth: Ground truth IDs of shape (n_queries, k_gt)
        k: Number of neighbors to consider
        
    Returns:
        Recall@k as percentage (0-100)
    """
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
    """
    Compute precision@k.
    
    Precision@k measures the proportion of predictions
    that are true nearest neighbors.
    
    Args:
        predicted: Predicted neighbor IDs
        ground_truth: Ground truth IDs
        k: Number of neighbors
        
    Returns:
        Precision@k as percentage
    """
    # For exact k-NN, precision@k = recall@k
    return compute_recall(predicted, ground_truth, k)


def compute_mrr(
    predicted: np.ndarray,
    ground_truth: np.ndarray
) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    MRR measures the average reciprocal rank of the first
    correct result.
    
    Args:
        predicted: Predicted neighbor IDs
        ground_truth: Ground truth IDs (at least first neighbor)
        
    Returns:
        MRR score (0-1)
    """
    n_queries = predicted.shape[0]
    
    reciprocal_ranks = []
    for i in range(n_queries):
        true_neighbor = ground_truth[i, 0]  # First true neighbor
        try:
            rank = np.where(predicted[i] == true_neighbor)[0][0] + 1
            reciprocal_ranks.append(1.0 / rank)
        except IndexError:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks)


def compute_ndcg(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain.
    
    NDCG accounts for the position of correct results,
    with higher weights for earlier positions.
    
    Args:
        predicted: Predicted neighbor IDs
        ground_truth: Ground truth IDs
        k: Number of neighbors
        
    Returns:
        NDCG@k score (0-1)
    """
    n_queries = predicted.shape[0]
    k = min(k, predicted.shape[1], ground_truth.shape[1])
    
    ndcg_scores = []
    for i in range(n_queries):
        true_set = set(ground_truth[i, :k])
        
        # DCG
        dcg = 0.0
        for j, pred_id in enumerate(predicted[i, :k]):
            if pred_id in true_set:
                dcg += 1.0 / np.log2(j + 2)  # +2 because rank starts at 1
        
        # Ideal DCG (all correct in order)
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
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predicted: Predicted neighbor IDs
        ground_truth: Ground truth IDs
        k_values: List of k values for recall
        latencies: Query latencies in seconds (optional)
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Recall at various k
    for k in k_values:
        if k <= predicted.shape[1] and k <= ground_truth.shape[1]:
            metrics[f'recall@{k}'] = compute_recall(predicted, ground_truth, k)
    
    # MRR
    metrics['mrr'] = compute_mrr(predicted, ground_truth)
    
    # NDCG@10
    if 10 <= predicted.shape[1]:
        metrics['ndcg@10'] = compute_ndcg(predicted, ground_truth, 10)
    
    # Latency metrics
    if latencies is not None:
        latencies_ms = latencies * 1000  # Convert to ms
        metrics['latency_mean_ms'] = np.mean(latencies_ms)
        metrics['latency_median_ms'] = np.median(latencies_ms)
        metrics['latency_p95_ms'] = np.percentile(latencies_ms, 95)
        metrics['latency_p99_ms'] = np.percentile(latencies_ms, 99)
        metrics['throughput_qps'] = 1000 / np.mean(latencies_ms)
    
    return metrics


def compute_distance_stats(distances: np.ndarray) -> Dict:
    """
    Compute statistics on distance values.
    
    Useful for debugging and understanding search quality.
    
    Args:
        distances: Distance values
        
    Returns:
        Statistics dictionary
    """
    return {
        'mean': float(np.mean(distances)),
        'std': float(np.std(distances)),
        'min': float(np.min(distances)),
        'max': float(np.max(distances)),
        'median': float(np.median(distances))
    }


class BenchmarkResult:
    """Container for benchmark results with comparison utilities."""
    
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
        """Compare this result to a baseline."""
        comparison = {}
        
        # Recall comparison
        for key in self.metrics:
            if key.startswith('recall@'):
                baseline_val = baseline.metrics.get(key, 0)
                this_val = self.metrics.get(key, 0)
                comparison[f'{key}_diff'] = this_val - baseline_val
        
        # Latency comparison (speedup)
        baseline_lat = baseline.metrics.get('latency_mean_ms', 1)
        this_lat = self.metrics.get('latency_mean_ms', 1)
        comparison['speedup'] = baseline_lat / this_lat
        
        # Memory comparison
        comparison['memory_ratio'] = self.memory_mb / baseline.memory_mb
        
        return comparison
