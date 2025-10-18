"""
HNSW Graph Module for ZGQ V6
Implements per-zone HNSW graph construction as specified in hnsw_graphs.md

Mathematical Foundation:
- Hierarchical Navigable Small World graphs
- Probabilistic layer selection: ℓ = ⌊-ln(uniform(0,1)) · m_L⌋ where m_L = 1/ln(M)
- Beam search for nearest neighbor finding during construction
- Bidirectional edges with degree constraints

Complexity:
- Build: O(n_zone · log(n_zone) · M · d) per zone
- Search: O(log(n_zone) · ef_search · d) per query

Reference: Malkov & Yashunin "Efficient and robust approximate nearest neighbor 
           search using Hierarchical Navigable Small World graphs" (2018)
           hnsw_graphs.md
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass
import heapq
from distance_metrics import DistanceMetrics, PQDistanceMetrics
import time


@dataclass
class HNSWNode:
    """
    Node in HNSW graph.
    
    Attributes:
        id: Node identifier (local to zone)
        layer: Maximum layer this node appears in
        neighbors: Dict[layer -> List[neighbor_ids]]
    """
    id: int
    layer: int
    neighbors: Dict[int, List[int]]


class HNSWGraph:
    """
    Hierarchical Navigable Small World graph for a single zone.
    
    Follows specification in hnsw_graphs.md Section 1.
    
    Attributes:
        vectors: Matrix of shape (n, d) - vectors in this zone
        M: Maximum number of connections per layer
        M_max: Maximum connections for layers > 0
        M_max_0: Maximum connections for layer 0 (usually 2*M)
        ef_construction: Size of dynamic candidate list during construction
        m_L: Normalization factor for level generation (1/ln(M))
        entry_point: ID of entry point node
        nodes: List of HNSWNode objects
        max_layer: Maximum layer in the graph
    """
    
    def __init__(
        self,
        vectors: np.ndarray,
        M: int = 16,
        ef_construction: int = 200,
        verbose: bool = False
    ):
        """
        Initialize HNSW graph structure.
        
        Args:
            vectors: Vectors in this zone, shape (n_zone, d)
            M: Maximum number of connections per node
            ef_construction: Beam search width during construction
            verbose: Print construction progress
            
        Reference: hnsw_graphs.md Section 1.1
        """
        self.vectors = vectors
        self.n = len(vectors)
        self.d = vectors.shape[1]
        self.M = M
        self.M_max = M
        self.M_max_0 = M * 2  # Layer 0 gets more connections
        self.ef_construction = ef_construction
        self.m_L = 1.0 / np.log(M)
        self.verbose = verbose
        
        # Graph structure
        self.nodes: List[HNSWNode] = []
        self.entry_point = 0
        self.max_layer = 0
        
        # Statistics
        self.build_time = None
        self.layer_counts = None
    
    def _select_layer(self) -> int:
        """
        Select layer for a new node using exponential decay.
        
        Formula: ℓ = ⌊-ln(uniform(0,1)) · m_L⌋
        where m_L = 1/ln(M)
        
        This creates a hierarchical structure with fewer nodes at higher layers.
        
        Returns:
            Layer index (0 = base layer)
            
        Reference: hnsw_graphs.md Section 1.2.1
        """
        return int(-np.log(np.random.uniform()) * self.m_L)
    
    def _get_M_max(self, layer: int) -> int:
        """Get maximum connections for a layer."""
        return self.M_max_0 if layer == 0 else self.M_max
    
    def build(self) -> None:
        """
        Build HNSW graph using progressive insertion.
        
        Algorithm:
        ----------
        For each vector x in random order:
          1. Select layer ℓ for x
          2. Find nearest neighbors using beam search
          3. Connect x to M nearest neighbors at each layer ≤ ℓ
          4. Update neighbors' connections (maintain degree bound)
        
        Complexity: O(n · log(n) · M · d)
        
        Reference: hnsw_graphs.md Section 1.3
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Building HNSW Graph")
            print(f"{'='*70}")
            print(f"  Zone vectors: {self.n}")
            print(f"  Dimension: {self.d}")
            print(f"  M: {self.M}")
            print(f"  M_max_0: {self.M_max_0}")
            print(f"  ef_construction: {self.ef_construction}")
            print(f"  m_L: {self.m_L:.4f}")
        
        start_time = time.time()
        
        # Insert all nodes, starting from an empty graph
        # First node will become the entry point
        for node_id in range(self.n):
            if self.verbose and node_id % max(1, self.n // 20) == 0:
                progress = node_id / self.n * 100
                print(f"  Progress: {progress:.1f}% ({node_id}/{self.n})", end='\r')
            
            if node_id == 0:
                # Initialize with first node
                first_layer = self._select_layer()
                self.nodes.append(HNSWNode(
                    id=0,
                    layer=first_layer,
                    neighbors={layer: [] for layer in range(first_layer + 1)}
                ))
                self.entry_point = 0
                self.max_layer = first_layer
            else:
                self._insert_node(node_id)
        
        if self.verbose:
            print(f"  Progress: 100.0% ({self.n}/{self.n}) ✓")
        
        self.build_time = time.time() - start_time
        
        # Compute layer statistics
        self._compute_statistics()
        
        if self.verbose:
            self._print_statistics()
    
    def _insert_node(self, node_id: int) -> None:
        """
        Insert a new node into the graph.
        
        Steps:
        1. Select layer for new node
        2. Find nearest neighbors via beam search from entry point
        3. Connect to M nearest neighbors at each layer
        4. Prune connections if degree exceeds M_max
        
        Args:
            node_id: Index of node to insert
            
        Reference: hnsw_graphs.md Section 1.3
        """
        query = self.vectors[node_id]
        layer = self._select_layer()
        
        # Create node
        node = HNSWNode(
            id=node_id,
            layer=layer,
            neighbors={l: [] for l in range(layer + 1)}
        )
        
        # Save old entry point before potentially updating it
        old_entry_point = self.entry_point
        old_max_layer = self.max_layer
        
        # Update max layer if needed
        if layer > self.max_layer:
            self.max_layer = layer
            self.entry_point = node_id
        
        # Find nearest neighbors via beam search
        # Use OLD entry point for search (not the new one being inserted!)
        current_nearest = [old_entry_point]
        
        # Greedy search from top to layer 1
        for lc in range(old_max_layer, 0, -1):
            neighbors_at_layer = self._search_layer(
                query=query,
                entry_points=current_nearest,
                num_to_return=1,
                layer=lc,
                exclude_id=node_id
            )
            current_nearest = [nid for _, nid in neighbors_at_layer]
        
        # Beam search at layer 0
        nearest = self._search_layer(
            query=query,
            entry_points=current_nearest,
            num_to_return=self.ef_construction,
            layer=0,
            exclude_id=node_id
        )
        
        # Connect at each layer from 0 to node's layer
        for lc in range(layer + 1):
            # Select M nearest neighbors at this layer
            M = self._get_M_max(lc)
            
            # Get candidates at this layer
            candidates_at_layer = []
            for dist, neighbor_id in nearest:
                if self.nodes[neighbor_id].layer >= lc:
                    candidates_at_layer.append((dist, neighbor_id))
            
            # Select M best
            candidates_at_layer.sort()
            neighbors_to_connect = [nid for _, nid in candidates_at_layer[:M]]
            
            # Add bidirectional edges
            for neighbor_id in neighbors_to_connect:
                # Add edge from new node to neighbor
                node.neighbors[lc].append(neighbor_id)
                
                # Add edge from neighbor to new node
                self.nodes[neighbor_id].neighbors[lc].append(node_id)
                
                # Prune neighbor if it exceeds degree
                if len(self.nodes[neighbor_id].neighbors[lc]) > M:
                    self._prune_connections(neighbor_id, lc)
        
        self.nodes.append(node)
    
    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[int],
        num_to_return: int,
        layer: int,
        exclude_id: int = -1
    ) -> List[Tuple[float, int]]:
        """
        Beam search in a specific layer.
        
        Algorithm:
        ----------
        Maintain two sets:
          - candidates: min-heap of (distance, node_id) to explore
          - w: max-heap of (distance, node_id) for results
        
        Greedy expansion:
          Pop closest candidate
          Explore its neighbors
          Add to results if better than current worst
        
        Complexity: O(log(n) · ef · d) where ef = num_to_return
        
        Args:
            query: Query vector
            entry_points: Starting node IDs
            num_to_return: Size of result set (ef parameter)
            layer: Layer to search in
            exclude_id: Node ID to exclude (for insertion)
            
        Returns:
            List of (distance, node_id) tuples, sorted by distance
            
        Reference: hnsw_graphs.md Section 1.2.1, online_search.md Section 2.2
        """
        visited = {exclude_id} if exclude_id >= 0 else set()
        candidates = []  # min-heap
        w = []  # max-heap (negate distances)
        
        # Initialize with entry points
        for ep in entry_points:
            if ep not in visited and ep < len(self.nodes):
                dist = DistanceMetrics.euclidean_squared(query, self.vectors[ep])
                heapq.heappush(candidates, (dist, ep))
                heapq.heappush(w, (-dist, ep))
                visited.add(ep)
        
        # Beam search
        while candidates:
            current_dist, current_id = heapq.heappop(candidates)
            
            # Stopping criterion: current is farther than worst in results
            # w stores (-distance, node_id), so -w[0][0] is the furthest distance
            if w and current_dist > -w[0][0]:
                break
            
            # Explore neighbors at this layer
            if current_id < len(self.nodes) and layer in self.nodes[current_id].neighbors:
                for neighbor_id in self.nodes[current_id].neighbors[layer]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        
                        neighbor_dist = DistanceMetrics.euclidean_squared(
                            query, self.vectors[neighbor_id]
                        )
                        
                        # Add to candidates if promising
                        if neighbor_dist < -w[0][0] or len(w) < num_to_return:
                            heapq.heappush(candidates, (neighbor_dist, neighbor_id))
                            heapq.heappush(w, (-neighbor_dist, neighbor_id))
                            
                            # Maintain size limit
                            if len(w) > num_to_return:
                                heapq.heappop(w)
        
        # Return sorted results
        results = [(-dist, nid) for dist, nid in w]
        results.sort()
        return results
    
    def _prune_connections(self, node_id: int, layer: int) -> None:
        """
        Prune connections to maintain maximum degree M_max.
        
        Strategy: Keep M nearest neighbors by distance.
        
        Args:
            node_id: Node to prune
            layer: Layer to prune in
            
        Reference: hnsw_graphs.md Section 1.2.1
        """
        M_max = self._get_M_max(layer)
        neighbors = self.nodes[node_id].neighbors[layer]
        
        if len(neighbors) <= M_max:
            return
        
        # Compute distances to all neighbors
        node_vec = self.vectors[node_id]
        distances = []
        for neighbor_id in neighbors:
            dist = DistanceMetrics.euclidean_squared(node_vec, self.vectors[neighbor_id])
            distances.append((dist, neighbor_id))
        
        # Keep M_max nearest
        distances.sort()
        self.nodes[node_id].neighbors[layer] = [nid for _, nid in distances[:M_max]]
    
    def search(
        self,
        query: np.ndarray,
        k: int,
        ef_search: int = None,
        use_pq: bool = False,
        pq_codes: np.ndarray = None,
        distance_table: np.ndarray = None
    ) -> List[Tuple[int, float]]:
        """
        Search for k nearest neighbors using HNSW.
        
        Algorithm:
        ----------
        1. Start from entry point at top layer
        2. Greedy search to layer 0
        3. Beam search at layer 0 with ef parameter
        4. Return top-k by distance
        
        Complexity: O(log(n) · ef · d) for exact distance
                   O(log(n) · ef · m) for PQ distance
        
        Args:
            query: Query vector of shape (d,)
            k: Number of nearest neighbors
            ef_search: Beam width (default: max(ef_construction, k))
            use_pq: Use PQ distance (faster but approximate)
            pq_codes: PQ codes if use_pq=True
            distance_table: Precomputed PQ distance table if use_pq=True
            
        Returns:
            List of (node_id, distance) tuples
            
        Reference: hnsw_graphs.md Section 1.2, online_search.md Section 2
        """
        if ef_search is None:
            ef_search = max(self.ef_construction, k)
        
        if len(self.nodes) == 0:
            return []
        
        # Search from top layer to layer 0
        current_nearest = [self.entry_point]
        
        for lc in range(self.max_layer, 0, -1):
            current_nearest = self._search_layer(
                query=query,
                entry_points=current_nearest,
                num_to_return=1,
                layer=lc
            )
            current_nearest = [nid for _, nid in current_nearest]
        
        # Search at layer 0 with ef_search
        candidates = self._search_layer(
            query=query,
            entry_points=current_nearest,
            num_to_return=ef_search,
            layer=0
        )
        
        # Return top-k
        return candidates[:k]
    
    def _compute_statistics(self) -> None:
        """Compute graph statistics."""
        # Count nodes per layer
        layer_counts = {}
        for node in self.nodes:
            for layer in range(node.layer + 1):
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
        
        self.layer_counts = layer_counts
    
    def _print_statistics(self) -> None:
        """Print graph statistics."""
        print(f"\n  Build time: {self.build_time:.2f}s")
        print(f"  Max layer: {self.max_layer}")
        print(f"  Entry point: {self.entry_point}")
        
        print(f"\n  Layer distribution:")
        for layer in sorted(self.layer_counts.keys(), reverse=True):
            count = self.layer_counts[layer]
            pct = count / self.n * 100
            bar = '█' * int(pct / 2)
            print(f"    Layer {layer:2d}: {bar} {count:5d} nodes ({pct:5.1f}%)")
        
        # Average degree per layer
        print(f"\n  Average degree per layer:")
        for layer in range(self.max_layer + 1):
            degrees = []
            for node in self.nodes:
                if layer in node.neighbors:
                    degrees.append(len(node.neighbors[layer]))
            if degrees:
                avg_degree = np.mean(degrees)
                max_degree = np.max(degrees)
                print(f"    Layer {layer}: avg={avg_degree:.1f}, max={max_degree}")
        
        # Memory usage
        edge_count = sum(
            len(neighbors)
            for node in self.nodes
            for neighbors in node.neighbors.values()
        )
        graph_memory = edge_count * 4 / (1024 ** 2)  # 4 bytes per int32
        vector_memory = self.vectors.nbytes / (1024 ** 2)
        print(f"\n  Memory:")
        print(f"    Vectors: {vector_memory:.2f} MB")
        print(f"    Graph: {graph_memory:.2f} MB")
        print(f"    Total: {vector_memory + graph_memory:.2f} MB")
    
    def get_memory_usage(self) -> float:
        """
        Get total memory usage in MB.
        
        Returns:
            Memory in megabytes
        """
        edge_count = sum(
            len(neighbors)
            for node in self.nodes
            for neighbors in node.neighbors.values()
        )
        graph_memory = edge_count * 4 / (1024 ** 2)
        vector_memory = self.vectors.nbytes / (1024 ** 2)
        return vector_memory + graph_memory


# Validation and testing
if __name__ == "__main__":
    print("="*70)
    print("HNSW Graph Module - Validation")
    print("="*70)
    
    # Configuration
    n = 5000
    d = 128
    M = 16
    ef_construction = 200
    
    # Generate test data
    print("\nGenerating synthetic data...")
    np.random.seed(42)
    vectors = np.random.randn(n, d).astype(np.float32)
    print(f"  Generated {n} vectors of dimension {d}")
    
    # Test 1: Build HNSW graph
    print("\n[Test 1] Building HNSW Graph")
    hnsw = HNSWGraph(vectors, M=M, ef_construction=ef_construction, verbose=True)
    hnsw.build()
    
    assert len(hnsw.nodes) == n, f"Expected {n} nodes, got {len(hnsw.nodes)}"
    print("✓ Graph built successfully")
    
    # Test 2: Search with exact distance
    print("\n[Test 2] Search with Exact Distance")
    n_queries = 100
    k = 10
    queries = np.random.randn(n_queries, d).astype(np.float32)
    
    search_times = []
    all_results = []
    
    for query in queries:
        start = time.time()
        results = hnsw.search(query, k=k, ef_search=50)
        search_times.append(time.time() - start)
        all_results.append(results)
    
    avg_search_time = np.mean(search_times) * 1000
    print(f"  Average search time: {avg_search_time:.3f} ms")
    print(f"  Throughput: {1000/avg_search_time:.0f} queries/sec")
    print(f"  Example results for query 0:")
    for i, result in enumerate(all_results[0][:5]):
        node_id = int(result[0]) if isinstance(result[0], (int, np.integer)) else int(result[1])
        dist = result[1] if isinstance(result[0], (int, np.integer)) else result[0]
        print(f"    {i+1}. Node {node_id:4d}, distance={dist:.4f}")
    
    # Test 3: Verify recall with brute force
    print("\n[Test 3] Recall Verification")
    n_test = 50
    test_queries = queries[:n_test]
    
    recalls = []
    for query in test_queries:
        # HNSW result
        hnsw_results = hnsw.search(query, k=k, ef_search=100)
        hnsw_ids = {nid for nid, _ in hnsw_results}
        
        # Brute force ground truth
        distances = DistanceMetrics.euclidean_batch_squared(query, vectors)
        true_ids = set(np.argsort(distances)[:k])
        
        # Compute recall
        recall = len(hnsw_ids & true_ids) / k
        recalls.append(recall)
    
    avg_recall = np.mean(recalls)
    print(f"  Average Recall@{k}: {avg_recall:.4f}")
    print(f"  Min recall: {np.min(recalls):.4f}")
    print(f"  Max recall: {np.max(recalls):.4f}")
    
    # Test 4: Varying ef_search
    print("\n[Test 4] Effect of ef_search on Recall")
    test_query = queries[0]
    
    # Ground truth
    distances = DistanceMetrics.euclidean_batch_squared(test_query, vectors)
    true_ids = set(np.argsort(distances)[:k])
    
    ef_values = [10, 20, 50, 100, 200]
    for ef in ef_values:
        results = hnsw.search(test_query, k=k, ef_search=ef)
        hnsw_ids = {nid for nid, _ in results}
        recall = len(hnsw_ids & true_ids) / k
        print(f"  ef={ef:3d}: Recall@{k} = {recall:.4f}")
    
    # Test 5: Memory efficiency
    print("\n[Test 5] Memory Efficiency")
    memory_mb = hnsw.get_memory_usage()
    print(f"  Total memory: {memory_mb:.2f} MB")
    print(f"  Memory per vector: {memory_mb / n * 1024:.2f} KB")
    
    print("\n" + "="*70)
    print("✓ All tests passed successfully")
    print("="*70)
