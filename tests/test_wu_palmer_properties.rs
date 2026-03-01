//! Property-style regression tests for the `WuPalmer` trait.
#![cfg(feature = "std")]

use std::collections::HashSet;

use geometric_traits::{
    impls::{CSR2D, SortedVec, SquareCSR2D},
    prelude::{DiEdgesBuilder, DiGraph, GenericVocabularyBuilder},
    traits::{EdgesBuilder, ScalarSimilarity, VocabularyBuilder, WuPalmer},
};

fn build_digraph(number_of_nodes: usize, edge_list: Vec<(usize, usize)>) -> DiGraph<usize> {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(number_of_nodes)
        .symbols((0..number_of_nodes).enumerate())
        .build()
        .unwrap();
    let edges: SquareCSR2D<CSR2D<usize, usize, usize>> = DiEdgesBuilder::default()
        .expected_number_of_edges(edge_list.len())
        .expected_shape(nodes.len())
        .edges(edge_list.into_iter())
        .build()
        .unwrap();
    DiGraph::from((nodes, edges))
}

fn assert_approx_eq(actual: f64, expected: f64) {
    let eps = 1e-12;
    assert!(
        (actual - expected).abs() <= eps,
        "expected {expected:.15}, got {actual:.15} (diff {:.15})",
        (actual - expected).abs()
    );
}

fn xorshift64_next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn randomized_dag_edges(seed: u64, number_of_nodes: usize) -> Vec<(usize, usize)> {
    if number_of_nodes <= 1 {
        return Vec::new();
    }
    let mut state = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
    let max_number_of_edges = number_of_nodes * (number_of_nodes - 1) / 2;
    let edges_modulus = u64::try_from(max_number_of_edges + 1).unwrap();
    let number_of_edges = usize::try_from(xorshift64_next(&mut state) % edges_modulus).unwrap();
    let number_of_nodes_u64 = u64::try_from(number_of_nodes).unwrap();

    let mut edges = HashSet::with_capacity(number_of_edges);
    while edges.len() < number_of_edges {
        let first = usize::try_from(xorshift64_next(&mut state) % number_of_nodes_u64).unwrap();
        let second = usize::try_from(xorshift64_next(&mut state) % number_of_nodes_u64).unwrap();
        if first == second {
            continue;
        }
        let (src, dst) = if first < second { (first, second) } else { (second, first) };
        edges.insert((src, dst));
    }

    let mut sorted_edges: Vec<(usize, usize)> = edges.into_iter().collect();
    sorted_edges.sort_unstable();
    sorted_edges
}

fn randomized_componentized_dag(
    seed: u64,
    component_sizes: &[usize],
) -> (usize, Vec<(usize, usize)>, Vec<usize>) {
    let mut state = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
    let number_of_nodes: usize = component_sizes.iter().sum();
    let mut component_of = vec![0usize; number_of_nodes];
    let mut all_edges = Vec::new();

    let mut offset = 0usize;
    for (component_id, &size) in component_sizes.iter().enumerate() {
        for component_slot in component_of.iter_mut().skip(offset).take(size) {
            *component_slot = component_id;
        }

        if size > 1 {
            let max_number_of_edges = size * (size - 1) / 2;
            let edges_modulus = u64::try_from(max_number_of_edges + 1).unwrap();
            let number_of_edges =
                usize::try_from(xorshift64_next(&mut state) % edges_modulus).unwrap();
            let size_u64 = u64::try_from(size).unwrap();
            let mut local_edges = HashSet::with_capacity(number_of_edges);

            while local_edges.len() < number_of_edges {
                let first = usize::try_from(xorshift64_next(&mut state) % size_u64).unwrap();
                let second = usize::try_from(xorshift64_next(&mut state) % size_u64).unwrap();
                if first == second {
                    continue;
                }
                let (src_local, dst_local) =
                    if first < second { (first, second) } else { (second, first) };
                local_edges.insert((offset + src_local, offset + dst_local));
            }

            all_edges.extend(local_edges);
        }

        offset += size;
    }

    all_edges.sort_unstable();
    (number_of_nodes, all_edges, component_of)
}

#[test]
fn test_wu_palmer_randomized_dag_invariants_fixed_seeds() {
    let seeds: [u64; 12] = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233];

    for &seed in &seeds {
        for number_of_nodes in 2..=12usize {
            let edges = randomized_dag_edges(
                seed.wrapping_add(number_of_nodes as u64 * 4099),
                number_of_nodes,
            );
            let graph = build_digraph(number_of_nodes, edges);
            let wu_palmer = graph.wu_palmer().expect("randomized_dag should always be acyclic");

            for src in 0..number_of_nodes {
                let self_similarity = wu_palmer.similarity(&src, &src);
                assert_approx_eq(self_similarity, 1.0);

                for dst in 0..number_of_nodes {
                    let similarity = wu_palmer.similarity(&src, &dst);
                    let symmetric_similarity = wu_palmer.similarity(&dst, &src);

                    assert!(
                        (-1e-12..=1.0 + 1e-12).contains(&similarity),
                        "similarity out of bounds for ({src}, {dst}): {similarity}"
                    );
                    assert!(
                        (similarity - symmetric_similarity).abs() <= 1e-12,
                        "symmetry violation for ({src}, {dst}): {similarity} vs {symmetric_similarity}"
                    );
                }
            }
        }
    }
}

#[test]
fn test_wu_palmer_root_query_monotonicity_on_chains() {
    // For chain 0 -> 1 -> ... -> n-1 with canonical Wu-Palmer and depth(root)=1:
    // sim(0, k) = 2 / (k + 2), therefore it strictly decreases with k.
    for number_of_nodes in 3..=10usize {
        let mut edges = Vec::with_capacity(number_of_nodes - 1);
        for src in 0..(number_of_nodes - 1) {
            edges.push((src, src + 1));
        }
        let graph = build_digraph(number_of_nodes, edges);
        let wu_palmer = graph.wu_palmer().unwrap();

        for node in 1..number_of_nodes {
            let expected = 2.0 / (f64::from(u32::try_from(node).unwrap()) + 2.0);
            assert_approx_eq(wu_palmer.similarity(&0, &node), expected);
        }

        for near in 1..(number_of_nodes - 1) {
            for far in (near + 1)..number_of_nodes {
                assert!(
                    wu_palmer.similarity(&0, &near) > wu_palmer.similarity(&0, &far),
                    "expected sim(0,{near}) > sim(0,{far}) on chain with {number_of_nodes} nodes"
                );
            }
        }
    }
}

#[test]
fn test_wu_palmer_chain_anchor_monotonicity_all_prefixes() {
    // For chain 0 -> 1 -> ... -> n-1 and any anchor a < b:
    // sim(a, b) = 2*depth(a) / (depth(a) + depth(b)).
    for number_of_nodes in 4..=12usize {
        let mut edges = Vec::with_capacity(number_of_nodes - 1);
        for src in 0..(number_of_nodes - 1) {
            edges.push((src, src + 1));
        }
        let graph = build_digraph(number_of_nodes, edges);
        let wu_palmer = graph.wu_palmer().unwrap();

        for anchor in 0..(number_of_nodes - 1) {
            let depth_anchor = f64::from(u32::try_from(anchor + 1).unwrap());

            for dst in (anchor + 1)..number_of_nodes {
                let depth_dst = f64::from(u32::try_from(dst + 1).unwrap());
                let expected = (2.0 * depth_anchor) / (depth_anchor + depth_dst);
                assert_approx_eq(wu_palmer.similarity(&anchor, &dst), expected);
            }
        }

        for anchor in 0..(number_of_nodes - 2) {
            for near in (anchor + 1)..(number_of_nodes - 1) {
                for far in (near + 1)..number_of_nodes {
                    assert!(
                        wu_palmer.similarity(&anchor, &near) > wu_palmer.similarity(&anchor, &far),
                        "expected sim({anchor},{near}) > sim({anchor},{far}) on chain with {number_of_nodes} nodes"
                    );
                }
            }
        }
    }
}

#[test]
fn test_wu_palmer_invariants_on_fixed_dense_dag_corpus() {
    let cases: Vec<(usize, Vec<(usize, usize)>)> = vec![
        (4, vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
        (6, vec![(0, 2), (0, 3), (1, 3), (1, 4), (2, 5), (3, 5), (4, 5)]),
        (7, vec![(0, 2), (0, 3), (1, 3), (1, 4), (2, 5), (3, 5), (3, 6), (4, 6), (5, 6)]),
    ];

    for (number_of_nodes, edge_list) in cases {
        let graph = build_digraph(number_of_nodes, edge_list);
        let wu_palmer = graph.wu_palmer().unwrap();

        for src in 0..number_of_nodes {
            assert_approx_eq(wu_palmer.similarity(&src, &src), 1.0);
            for dst in 0..number_of_nodes {
                let similarity = wu_palmer.similarity(&src, &dst);
                let symmetric_similarity = wu_palmer.similarity(&dst, &src);
                assert!(
                    (-1e-12..=1.0 + 1e-12).contains(&similarity),
                    "similarity out of bounds for ({src}, {dst}) in dense corpus case: {similarity}"
                );
                assert!(
                    (similarity - symmetric_similarity).abs() <= 1e-12,
                    "symmetry violation for ({src}, {dst}): {similarity} vs {symmetric_similarity}"
                );
            }
        }
    }
}

#[test]
fn test_wu_palmer_component_isolation_property() {
    let seeds: [u64; 4] = [7, 31, 127, 511];
    let component_layouts: [&[usize]; 4] = [&[3, 2, 1], &[4, 3], &[2, 2, 2], &[5, 1, 3]];

    for &seed in &seeds {
        for (layout_index, component_sizes) in component_layouts.iter().enumerate() {
            let layout_index_u64 = u64::try_from(layout_index).unwrap();
            let (number_of_nodes, edge_list, component_of) = randomized_componentized_dag(
                seed.wrapping_add(layout_index_u64.wrapping_mul(10_007)),
                component_sizes,
            );
            let graph = build_digraph(number_of_nodes, edge_list);
            let wu_palmer = graph.wu_palmer().unwrap();

            for left in 0..number_of_nodes {
                for right in 0..number_of_nodes {
                    if component_of[left] != component_of[right] {
                        assert_approx_eq(wu_palmer.similarity(&left, &right), 0.0);
                    }
                }
            }
        }
    }
}
