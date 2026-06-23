//! Fuzz harness for the minimum-spanning-tree traits. For a random small graph
//! (up to 6 nodes), it checks each of `Kruskal`, `Prim`, and `Boruvka` against
//! an independent brute-force minimum over every spanning forest, and that each
//! result is a valid acyclic forest with the right edge and component counts.

use geometric_traits::{
    impls::{ValuedCSR2D, WeightedForest},
    naive_structs::GenericEdgesBuilder,
    traits::{
        EdgesBuilder,
        algorithms::minimum_spanning_tree::{Boruvka, Kruskal, Prim},
    },
};
use honggfuzz::fuzz;

type WeightedMatrix = ValuedCSR2D<usize, usize, usize, f64>;

const MAX_NODES: usize = 6;

/// Simple union-find with no ranking, enough for tiny graphs.
struct DisjointSet {
    parent: Vec<usize>,
}

impl DisjointSet {
    fn new(size: usize) -> Self {
        Self { parent: (0..size).collect() }
    }

    fn find(&mut self, node: usize) -> usize {
        let mut root = node;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut current = node;
        while current != root {
            let next = self.parent[current];
            self.parent[current] = root;
            current = next;
        }
        root
    }

    fn union(&mut self, left: usize, right: usize) -> bool {
        let (left_root, right_root) = (self.find(left), self.find(right));
        if left_root == right_root {
            return false;
        }
        self.parent[left_root] = right_root;
        true
    }

    fn components(&mut self, size: usize) -> usize {
        let mut roots = std::collections::HashSet::new();
        for node in 0..size {
            let root = self.find(node);
            roots.insert(root);
        }
        roots.len()
    }
}

fn build_matrix(node_count: usize, edges: &[(usize, usize, u32)]) -> WeightedMatrix {
    let mut directed = Vec::with_capacity(edges.len() * 2);
    for &(source, destination, weight) in edges {
        directed.push((source, destination, f64::from(weight)));
        directed.push((destination, source, f64::from(weight)));
    }
    directed.sort_unstable_by(|(ls, ld, _), (rs, rd, _)| (ls, ld).cmp(&(rs, rd)));
    GenericEdgesBuilder::<_, WeightedMatrix>::default()
        .expected_number_of_edges(directed.len())
        .expected_shape((node_count, node_count))
        .edges(directed.into_iter())
        .build()
        .unwrap()
}

/// Brute-force minimum total weight over every spanning forest of the graph.
fn brute_force_weight(node_count: usize, edges: &[(usize, usize, u32)], components: usize) -> u64 {
    let target = node_count - components;
    let mut best = u64::MAX;
    for subset in 0u32..(1u32 << edges.len()) {
        if subset.count_ones() as usize != target {
            continue;
        }
        let mut disjoint = DisjointSet::new(node_count);
        let mut acyclic = true;
        let mut weight = 0u64;
        for (index, &(source, destination, edge_weight)) in edges.iter().enumerate() {
            if subset & (1 << index) == 0 {
                continue;
            }
            if !disjoint.union(source, destination) {
                acyclic = false;
                break;
            }
            weight += u64::from(edge_weight);
        }
        if acyclic && disjoint.components(node_count) == components {
            best = best.min(weight);
        }
    }
    best
}

/// Validates structural invariants and returns the integer total weight.
fn validate_forest(
    node_count: usize,
    edges: &[(usize, usize, u32)],
    components: usize,
    forest: &WeightedForest<usize, f64>,
) -> u64 {
    assert_eq!(forest.number_of_nodes(), node_count);
    assert_eq!(forest.number_of_components(), components);
    assert_eq!(forest.len(), node_count - components);

    let available: std::collections::HashSet<(usize, usize)> = edges
        .iter()
        .map(|&(s, d, _)| if s < d { (s, d) } else { (d, s) })
        .collect();

    let mut disjoint = DisjointSet::new(node_count);
    let mut weight = 0u64;
    for (source, destination, edge_weight) in forest.edge_iter() {
        assert!(source < destination);
        let key = if source < destination { (source, destination) } else { (destination, source) };
        assert!(available.contains(&key));
        assert!(disjoint.union(source, destination), "forest has a cycle");
        // Weights round-tripped through f64 from small integers are exact.
        weight += edge_weight as u64;
    }
    assert_eq!(weight as f64, forest.total_weight());
    weight
}

fn main() {
    loop {
        fuzz!(|data: &[u8]| {
            if data.is_empty() {
                return;
            }
            let node_count = (data[0] as usize % MAX_NODES) + 1;
            let slots = node_count * (node_count - 1) / 2;
            if data.len() < 1 + slots {
                return;
            }

            let mut edges = Vec::new();
            let mut offset = 1;
            for source in 0..node_count {
                for destination in (source + 1)..node_count {
                    let byte = data[offset];
                    offset += 1;
                    if byte & 1 != 0 {
                        let weight = u32::from((byte >> 1) % 6) + 1;
                        edges.push((source, destination, weight));
                    }
                }
            }

            let mut full = DisjointSet::new(node_count);
            for &(source, destination, _) in &edges {
                full.union(source, destination);
            }
            let components = full.components(node_count);

            let matrix = build_matrix(node_count, &edges);
            let kruskal = matrix.minimum_spanning_tree_kruskal().unwrap();
            let prim = matrix.minimum_spanning_tree_prim().unwrap();
            let boruvka = matrix.minimum_spanning_tree_boruvka().unwrap();

            let expected = brute_force_weight(node_count, &edges, components);
            for forest in [&kruskal, &prim, &boruvka] {
                let weight = validate_forest(node_count, &edges, components, forest);
                assert_eq!(weight, expected, "result is not a minimum spanning forest");
            }
        });
    }
}
