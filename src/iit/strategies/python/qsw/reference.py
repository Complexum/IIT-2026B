"""
Stoer-Wagner global minimum cut.

This is a direct port of the algorithm as described in:
  M. Stoer and F. Wagner, "A Simple Min-Cut Algorithm",
  Journal of the ACM 44(4), 585-591, 1997.

It's the same algorithm igraph uses internally (igraph_mincut, for
undirected graphs) and the same one Boost Graph Library implements
in boost/graph/stoer_wagner_min_cut.hpp. Deterministic, always correct
in a single run (unlike Karger/Karger-Stein).

Representation: dense adjacency matrix (weights[i][j] = edge weight
between i and j, 0 if none). This matches how the original paper and
most textbook implementations describe it. Complexity: O(V^3).

Note on scale: O(V^3) means this is about the NUMBER OF VERTICES, not
edges. For a bipartite graph with, say, 5,000 nodes total (even if it
has millions of edges among them), this runs fine. If you have millions
of *vertices*, this dense-matrix approach breaks down (memory alone:
V^2 floats) and you'd want igraph/Boost's sparse, heap-based version
instead, which is the same algorithm with a smarter data structure for
the "maximum adjacency search" step (priority queue instead of a full
V-length scan), bringing it to O(VE + V^2 log V).
"""

import math
from typing import List, Tuple


def stoer_wagner(weights: List[List[float]]) -> Tuple[float, List[int]]:
    """
    weights: V x V symmetric adjacency matrix, weights[i][j] = edge weight
             (0 if no edge). Diagonal should be 0.
    Returns: (min_cut_value, one_side_of_the_partition)
    """
    n = len(weights)
    # Work on a mutable copy; "merged" tracks which original vertices
    # are absorbed into each remaining supernode.
    w = [row[:] for row in weights]
    vertices = list(range(n))
    merged = {i: [i] for i in range(n)}

    best_cut = math.inf
    best_partition: List[int] = []

    while len(vertices) > 1:
        # --- one "minimum cut phase": maximum adjacency search ---
        A = [vertices[0]]
        in_A = {vertices[0]}
        weights_to_A = {v: w[vertices[0]][v] for v in vertices if v != vertices[0]}

        while len(A) < len(vertices):
            # pick the vertex most tightly connected to the current set A
            next_v = max(weights_to_A, key=weights_to_A.get)
            A.append(next_v)
            in_A.add(next_v)
            del weights_to_A[next_v]
            for v in weights_to_A:
                weights_to_A[v] += w[next_v][v]

        # cut-of-the-phase: weight separating the last vertex added
        # from everything else added before it
        s, t = A[-2], A[-1]
        cut_of_phase = sum(w[t][v] for v in vertices if v != t)

        if cut_of_phase < best_cut:
            best_cut = cut_of_phase
            best_partition = list(merged[t])

        # merge t into s (Stoer-Wagner's key step)
        for v in vertices:
            if v != s and v != t:
                w[s][v] += w[t][v]
                w[v][s] += w[v][t]
        merged[s].extend(merged[t])
        vertices.remove(t)

    return best_cut, best_partition


def edges_to_matrix(n: int, edges: List[Tuple[int, int]], weight: float = 1.0):
    """Helper: build a dense weighted adjacency matrix from an edge list."""
    w = [[0.0] * n for _ in range(n)]
    for u, v in edges:
        w[u][v] += weight
        w[v][u] += weight
    return w


if __name__ == "__main__":
    # Same sanity check as before: two triangles joined by one bridge edge
    edges = [
        (0, 1),
        (1, 2),
        (2, 0),  # triangle A
        (3, 4),
        (4, 5),
        (5, 3),  # triangle B
        (2, 3),  # bridge
    ]
    w = edges_to_matrix(6, edges)
    value, partition = stoer_wagner(w)
    print("min cut value (expected 1):", value)
    print("one side of the partition:", sorted(partition))
