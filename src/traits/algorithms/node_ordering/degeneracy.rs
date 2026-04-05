use alloc::vec::Vec;

use num_traits::AsPrimitive;

use super::{NodeScorer, NodeSorter};
use crate::traits::UndirectedMonopartiteMonoplexGraph;

const NO_NODE: usize = usize::MAX;

struct DegeneracyDecomposition<NodeId> {
    smallest_last_order: Vec<NodeId>,
}

#[inline]
fn bucket_insert_head(
    node_index: usize,
    degree: usize,
    bucket_heads: &mut [usize],
    next_in_bucket: &mut [usize],
    prev_in_bucket: &mut [usize],
) {
    let previous_head = bucket_heads[degree];
    bucket_heads[degree] = node_index;
    next_in_bucket[node_index] = previous_head;
    prev_in_bucket[node_index] = NO_NODE;
    if previous_head != NO_NODE {
        prev_in_bucket[previous_head] = node_index;
    }
}

#[inline]
fn bucket_remove(
    node_index: usize,
    degree: usize,
    bucket_heads: &mut [usize],
    next_in_bucket: &mut [usize],
    prev_in_bucket: &mut [usize],
) {
    let previous = prev_in_bucket[node_index];
    let next = next_in_bucket[node_index];

    if previous == NO_NODE {
        bucket_heads[degree] = next;
    } else {
        next_in_bucket[previous] = next;
    }

    if next != NO_NODE {
        prev_in_bucket[next] = previous;
    }

    next_in_bucket[node_index] = NO_NODE;
    prev_in_bucket[node_index] = NO_NODE;
}

fn degeneracy_decomposition<G>(graph: &G) -> DegeneracyDecomposition<G::NodeId>
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    let n = graph.number_of_nodes().as_();
    if n == 0 {
        return DegeneracyDecomposition { smallest_last_order: Vec::new() };
    }

    let nodes: Vec<G::NodeId> = graph.node_ids().collect();
    debug_assert_eq!(nodes.len(), n);
    debug_assert!(nodes.iter().enumerate().all(|(i, node)| (*node).as_() == i));

    let mut degrees: Vec<usize> = nodes.iter().map(|&node| graph.degree(node).as_()).collect();
    let max_degree = degrees.iter().copied().max().unwrap_or(0);
    let mut bucket_heads = vec![NO_NODE; max_degree + 1];
    let mut next_in_bucket = vec![NO_NODE; n];
    let mut prev_in_bucket = vec![NO_NODE; n];
    let mut removed = vec![false; n];

    for node_index in (0..n).rev() {
        bucket_insert_head(
            node_index,
            degrees[node_index],
            &mut bucket_heads,
            &mut next_in_bucket,
            &mut prev_in_bucket,
        );
    }

    let mut min_degree = degrees.iter().copied().min().unwrap_or(0);
    let mut removal_order = Vec::with_capacity(n);

    for _ in 0..n {
        while min_degree < bucket_heads.len() && bucket_heads[min_degree] == NO_NODE {
            min_degree += 1;
        }
        debug_assert!(min_degree < bucket_heads.len());

        let node_index = bucket_heads[min_degree];
        debug_assert_ne!(node_index, NO_NODE);
        bucket_remove(
            node_index,
            min_degree,
            &mut bucket_heads,
            &mut next_in_bucket,
            &mut prev_in_bucket,
        );

        removed[node_index] = true;
        let node = nodes[node_index];
        removal_order.push(node);

        for neighbor in graph.neighbors(node) {
            let neighbor_index = neighbor.as_();
            if removed[neighbor_index] {
                continue;
            }

            let previous_degree = degrees[neighbor_index];
            bucket_remove(
                neighbor_index,
                previous_degree,
                &mut bucket_heads,
                &mut next_in_bucket,
                &mut prev_in_bucket,
            );

            let new_degree = previous_degree.saturating_sub(1);
            degrees[neighbor_index] = new_degree;
            bucket_insert_head(
                neighbor_index,
                new_degree,
                &mut bucket_heads,
                &mut next_in_bucket,
                &mut prev_in_bucket,
            );

            if new_degree < min_degree {
                min_degree = new_degree;
            }
        }
    }

    removal_order.reverse();
    DegeneracyDecomposition { smallest_last_order: removal_order }
}

fn core_numbers<G>(graph: &G) -> Vec<usize>
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    let n = graph.number_of_nodes().as_();
    if n == 0 {
        return Vec::new();
    }

    let nodes: Vec<G::NodeId> = graph.node_ids().collect();
    debug_assert_eq!(nodes.len(), n);
    debug_assert!(nodes.iter().enumerate().all(|(i, node)| (*node).as_() == i));

    let mut core_numbers: Vec<usize> = nodes.iter().map(|&node| graph.degree(node).as_()).collect();
    let max_degree = core_numbers.iter().copied().max().unwrap_or(0);
    let mut bins = vec![0usize; max_degree + 1];
    for &degree in &core_numbers {
        bins[degree] += 1;
    }

    let mut start = 0usize;
    for bin in &mut bins {
        let count = *bin;
        *bin = start;
        start += count;
    }

    let mut positions = vec![0usize; n];
    let mut ordering = vec![0usize; n];

    for node_index in 0..n {
        let degree = core_numbers[node_index];
        let position = bins[degree];
        positions[node_index] = position;
        ordering[position] = node_index;
        bins[degree] += 1;
    }

    for degree in (1..=max_degree).rev() {
        bins[degree] = bins[degree - 1];
    }
    bins[0] = 0;

    for position in 0..n {
        let node_index = ordering[position];
        let node = nodes[node_index];

        for neighbor in graph.neighbors(node) {
            let neighbor_index = neighbor.as_();
            if core_numbers[neighbor_index] <= core_numbers[node_index] {
                continue;
            }

            let neighbor_degree = core_numbers[neighbor_index];
            let neighbor_position = positions[neighbor_index];
            let first_in_bin_position = bins[neighbor_degree];
            let first_in_bin_node = ordering[first_in_bin_position];

            if neighbor_index != first_in_bin_node {
                ordering[neighbor_position] = first_in_bin_node;
                ordering[first_in_bin_position] = neighbor_index;
                positions[neighbor_index] = first_in_bin_position;
                positions[first_in_bin_node] = neighbor_position;
            }

            bins[neighbor_degree] += 1;
            core_numbers[neighbor_index] -= 1;
        }
    }

    core_numbers
}

/// Degeneracy (smallest-last) ordering.
///
/// This is the linear-time bucket-queue smallest-last algorithm of Matula and
/// Beck. It returns the final ordering obtained by reversing the minimum-degree
/// removal sequence, so denser-core vertices appear first. Exact tie order
/// within the same degree bucket is not part of the contract; the guarantee is
/// that the returned order is a valid smallest-last ordering.
///
/// References:
/// - Matula, D. W., & Beck, L. L. (1983). Smallest-last ordering and clustering
///   and graph coloring algorithms. *Journal of the ACM*, 30(3), 417-427. DOI:
///   `10.1145/2402.322385`
/// - Batagelj, V., & Zaversnik, M. (2003). An O(m) algorithm for cores
///   decomposition of networks. [arXiv:cs/0310049](https://arxiv.org/abs/cs/0310049)
#[derive(Clone, Copy, Debug, Default)]
pub struct DegeneracySorter;

impl<G> NodeSorter<G> for DegeneracySorter
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    fn sort_nodes(&self, graph: &G) -> Vec<G::NodeId> {
        degeneracy_decomposition(graph).smallest_last_order
    }
}

/// Core-number scorer.
///
/// The score of a node is its k-core number. This is computed from the same
/// linear bucket-queue decomposition used by [`DegeneracySorter`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CoreNumberScorer;

impl<G> NodeScorer<G> for CoreNumberScorer
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    type Score = usize;

    fn score_nodes(&self, graph: &G) -> Vec<Self::Score> {
        core_numbers(graph)
    }
}
