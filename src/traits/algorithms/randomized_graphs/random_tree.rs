//! Generator for random labeled trees via Prüfer sequences.
#![cfg(feature = "alloc")]

use alloc::{collections::BTreeSet, vec::Vec};

use super::{XorShift64, builder_utils::build_symmetric};
use crate::impls::{CSR2D, SymmetricCSR2D};

/// Generates a random labeled tree on `n` vertices.
///
/// The generator is deterministic for a fixed `(seed, n)` pair. For `n >= 2`,
/// it samples a uniformly random labeled tree by generating a random Prüfer
/// sequence of length `n - 2` and decoding it.
#[must_use]
pub fn random_tree_graph(seed: u64, n: usize) -> SymmetricCSR2D<CSR2D<usize, usize, usize>> {
    if n <= 1 {
        return build_symmetric(n, Vec::new());
    }
    if n == 2 {
        return build_symmetric(n, vec![(0, 1)]);
    }

    let mut rng = XorShift64::from(XorShift64::normalize_seed(seed));
    let mut prufer = Vec::with_capacity(n - 2);
    for _ in 0..(n - 2) {
        prufer.push((rng.next().unwrap() as usize) % n);
    }

    let mut degrees = vec![1usize; n];
    for &vertex in &prufer {
        degrees[vertex] += 1;
    }

    let mut leaves: BTreeSet<usize> = degrees
        .iter()
        .enumerate()
        .filter_map(|(vertex, &degree)| (degree == 1).then_some(vertex))
        .collect();
    let mut edges = Vec::with_capacity(n - 1);

    for vertex in prufer {
        let leaf = *leaves.first().expect("a valid Prüfer decoding step must always have a leaf");
        leaves.remove(&leaf);

        let edge = if leaf < vertex { (leaf, vertex) } else { (vertex, leaf) };
        edges.push(edge);

        degrees[leaf] -= 1;
        degrees[vertex] -= 1;
        if degrees[vertex] == 1 {
            leaves.insert(vertex);
        }
    }

    let mut remaining = leaves.into_iter();
    let first = remaining.next().expect("Prüfer decoding must leave two vertices");
    let second = remaining.next().expect("Prüfer decoding must leave two vertices");
    edges.push(if first < second { (first, second) } else { (second, first) });

    edges.sort_unstable();
    build_symmetric(n, edges)
}

#[cfg(test)]
mod tests {
    use super::random_tree_graph;
    use crate::{
        naive_structs::UndiGraph,
        traits::{
            SizedSparseMatrix, SquareMatrix, VocabularyBuilder,
            algorithms::tree_detection::TreeDetection,
        },
    };

    fn wrap_undi(
        graph: crate::impls::SymmetricCSR2D<crate::impls::CSR2D<usize, usize, usize>>,
    ) -> UndiGraph<usize> {
        let order = graph.order();
        let nodes = crate::naive_structs::GenericVocabularyBuilder::default()
            .expected_number_of_symbols(order)
            .symbols((0..order).enumerate())
            .build()
            .unwrap();
        UndiGraph::from((nodes, graph))
    }

    #[test]
    fn test_random_tree_graph_handles_small_orders() {
        assert_eq!(random_tree_graph(7, 0).order(), 0);
        assert_eq!(random_tree_graph(7, 0).number_of_defined_values(), 0);

        assert_eq!(random_tree_graph(7, 1).order(), 1);
        assert_eq!(random_tree_graph(7, 1).number_of_defined_values(), 0);

        let two = random_tree_graph(7, 2);
        assert_eq!(two.order(), 2);
        assert_eq!(two.number_of_defined_values(), 2);
        assert!(wrap_undi(two).is_tree());
    }

    #[test]
    fn test_random_tree_graph_is_a_tree_for_non_empty_orders() {
        for n in 1..12 {
            let graph = random_tree_graph(13, n);
            assert_eq!(graph.number_of_defined_values(), 2 * (n - 1));
            assert!(wrap_undi(graph).is_tree(), "n={n} should produce a tree");
        }
    }

    #[test]
    fn test_random_tree_graph_is_deterministic_for_same_seed() {
        let left = random_tree_graph(12345, 16);
        let right = random_tree_graph(12345, 16);

        assert_eq!(left, right);
    }
}
