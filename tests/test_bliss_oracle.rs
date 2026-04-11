//! Optional live-oracle smoke tests against a local `bliss` executable.
//!
//! Build vendored bliss first, for example:
//!
//! ```bash
//! cmake -S papers/software/bliss-0.77 -B /tmp/bliss-build
//! cmake --build /tmp/bliss-build -j2
//! cargo test test_bliss_oracle -- --ignored
//! ```
//!
//! Or point `GEOMETRIC_TRAITS_BLISS_BIN` at another `bliss` executable.
#![cfg(feature = "std")]
#![allow(clippy::pedantic)]
#![allow(clippy::identity_op, clippy::manual_repeat_n)]

use std::{collections::BTreeSet, env, fmt::Write as _, process::Command};

use indicatif::{ParallelProgressIterator, ProgressBar, ProgressDrawTarget, ProgressStyle};
#[path = "support/bliss_oracle.rs"]
mod bliss_oracle;
#[path = "support/canon_bench_fixture.rs"]
#[allow(dead_code)]
mod canon_bench_fixture;

use bliss_oracle::{
    canonicalize_labeled_simple_graph, encode_labeled_simple_graph_as_dimacs, locate_bliss_binary,
    run_bliss_on_dimacs_file_with_options,
};
use canon_bench_fixture::{benchmark_cases, scaling_cases};
use geometric_traits::{
    impls::{SortedVec, SymmetricCSR2D, ValuedCSR2D},
    naive_structs::GenericGraph,
    prelude::*,
    traits::{
        CanonSplittingHeuristic, CanonicalLabelingOptions, Edges, MonoplexGraph,
        SparseValuedMatrix2D, VocabularyBuilder, canonical_label_labeled_simple_graph,
        canonical_label_labeled_simple_graph_with_options,
    },
};
use rand::{Rng, SeedableRng, rngs::SmallRng, seq::SliceRandom};
use rayon::prelude::*;

type LabeledUndirectedEdges = SymmetricCSR2D<ValuedCSR2D<usize, usize, usize, u8>>;
type LabeledUndirectedGraph = GenericGraph<SortedVec<usize>, LabeledUndirectedEdges>;

fn build_bidirectional_labeled_graph(
    number_of_nodes: usize,
    edges: &[(usize, usize, u8)],
) -> LabeledUndirectedGraph {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(number_of_nodes)
        .symbols((0..number_of_nodes).enumerate())
        .build()
        .unwrap();
    let mut upper_edges: Vec<(usize, usize, u8)> = edges
        .iter()
        .map(|&(source, destination, label)| {
            if source <= destination {
                (source, destination, label)
            } else {
                (destination, source, label)
            }
        })
        .collect();
    upper_edges.sort_unstable_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
    upper_edges.dedup();
    let edges: LabeledUndirectedEdges =
        SymmetricCSR2D::from_sorted_upper_triangular_entries(number_of_nodes, upper_edges).unwrap();

    GenericGraph::from((nodes, edges))
}

fn certificate_from_order(
    graph: &LabeledUndirectedGraph,
    vertex_labels: &[u8],
    order: &[usize],
) -> (Vec<u8>, Vec<Option<u8>>) {
    let matrix = Edges::matrix(graph.edges());
    let ordered_vertex_labels =
        order.iter().map(|&vertex| vertex_labels[vertex]).collect::<Vec<_>>();
    let mut upper_triangle_edge_labels = Vec::new();

    for left in 0..order.len() {
        for right in (left + 1)..order.len() {
            upper_triangle_edge_labels.push(matrix.sparse_value_at(order[left], order[right]));
        }
    }

    (ordered_vertex_labels, upper_triangle_edge_labels)
}

fn expanded_order_from_labeling(canonical_labeling: &[usize]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..canonical_labeling.len()).collect();
    order.sort_unstable_by_key(|&vertex| canonical_labeling[vertex]);
    order
}

#[derive(Clone, Copy)]
struct NamedOracleCase {
    name: &'static str,
    vertex_labels: &'static [u8],
    edges: &'static [(usize, usize, u8)],
}

#[derive(Clone)]
struct OwnedOracleCase {
    name: String,
    vertex_labels: Vec<u8>,
    edges: Vec<(usize, usize, u8)>,
}

fn compare_rust_and_bliss_case(case: NamedOracleCase) -> Result<(), String> {
    let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), case.edges);
    let matrix = Edges::matrix(graph.edges());
    let rust_result = canonical_label_labeled_simple_graph(
        &graph,
        |node| case.vertex_labels[node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
    );
    let bliss_result = canonicalize_labeled_simple_graph(case.vertex_labels, case.edges)
        .map_err(|error| format!("{}: bliss oracle run failed: {error}", case.name))?;

    let rust_certificate = certificate_from_order(&graph, case.vertex_labels, &rust_result.order);
    let bliss_certificate =
        certificate_from_order(&graph, case.vertex_labels, &bliss_result.original_canonical_order);

    if rust_certificate != bliss_certificate {
        return Err(format!(
            "{}: rust={:?} bliss={:?}",
            case.name, rust_certificate, bliss_certificate
        ));
    }

    Ok(())
}

fn compare_rust_and_bliss_graph(
    name: &str,
    graph: &LabeledUndirectedGraph,
    vertex_labels: &[u8],
    edges: &[(usize, usize, u8)],
) -> Result<(), String> {
    let matrix = Edges::matrix(graph.edges());
    let rust_result = canonical_label_labeled_simple_graph(
        graph,
        |node| vertex_labels[node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
    );
    let bliss_result = canonicalize_labeled_simple_graph(vertex_labels, edges)
        .map_err(|error| format!("{name}: bliss oracle run failed: {error}"))?;

    let rust_certificate = certificate_from_order(graph, vertex_labels, &rust_result.order);
    let bliss_certificate =
        certificate_from_order(graph, vertex_labels, &bliss_result.original_canonical_order);

    if rust_certificate != bliss_certificate {
        return Err(format!("{name}: rust={:?} bliss={:?}", rust_certificate, bliss_certificate));
    }

    Ok(())
}

fn compare_rust_and_bliss_owned_case(case: &OwnedOracleCase) -> Result<(), String> {
    let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
    compare_rust_and_bliss_graph(&case.name, &graph, &case.vertex_labels, &case.edges)
}

fn compare_rust_and_bliss_stats_graph(
    name: &str,
    graph: &LabeledUndirectedGraph,
    vertex_labels: &[u8],
    edges: &[(usize, usize, u8)],
) -> Result<(), String> {
    let matrix = Edges::matrix(graph.edges());
    let rust_result = canonical_label_labeled_simple_graph(
        graph,
        |node| vertex_labels[node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
    );
    let bliss_result = canonicalize_labeled_simple_graph(vertex_labels, edges)
        .map_err(|error| format!("{name}: bliss oracle run failed: {error}"))?;

    let bliss_nodes = bliss_result
        .stats
        .nodes
        .ok_or_else(|| format!("{name}: bliss did not report node count"))?;
    let bliss_leaf_nodes = bliss_result
        .stats
        .leaf_nodes
        .ok_or_else(|| format!("{name}: bliss did not report leaf-node count"))?;

    let rust_stats = rust_result.stats;
    if rust_stats.search_nodes != bliss_nodes || rust_stats.leaf_nodes != bliss_leaf_nodes {
        return Err(format!(
            "{name}: rust nodes/leaves={}/{} bliss nodes/leaves={}/{}",
            rust_stats.search_nodes, rust_stats.leaf_nodes, bliss_nodes, bliss_leaf_nodes
        ));
    }

    Ok(())
}

fn compare_rust_and_bliss_stats_owned_case(case: &OwnedOracleCase) -> Result<(), String> {
    let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
    compare_rust_and_bliss_stats_graph(&case.name, &graph, &case.vertex_labels, &case.edges)
}

fn rust_stats_for_owned_case(
    case: &OwnedOracleCase,
) -> geometric_traits::traits::CanonicalSearchStats {
    let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
    let matrix = Edges::matrix(graph.edges());
    canonical_label_labeled_simple_graph(
        &graph,
        |node| case.vertex_labels[node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
    )
    .stats
}

fn rust_stats_for_owned_case_with_heuristic(
    case: &OwnedOracleCase,
    splitting_heuristic: CanonSplittingHeuristic,
) -> geometric_traits::traits::CanonicalSearchStats {
    let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
    let matrix = Edges::matrix(graph.edges());
    canonical_label_labeled_simple_graph_with_options(
        &graph,
        |node| case.vertex_labels[node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
        CanonicalLabelingOptions { splitting_heuristic },
    )
    .stats
}

fn rust_certificate_for_case_with_heuristic(
    case: NamedOracleCase,
    splitting_heuristic: CanonSplittingHeuristic,
) -> (Vec<u8>, Vec<Option<u8>>) {
    let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), case.edges);
    let matrix = Edges::matrix(graph.edges());
    let result = canonical_label_labeled_simple_graph_with_options(
        &graph,
        |node| case.vertex_labels[node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
        CanonicalLabelingOptions { splitting_heuristic },
    );
    certificate_from_order(&graph, case.vertex_labels, &result.order)
}

fn uniform_cycle_case(name: &str, number_of_nodes: usize) -> OwnedOracleCase {
    assert!(number_of_nodes >= 4);
    let mut edges = (0..number_of_nodes)
        .map(|index| (index, (index + 1) % number_of_nodes, 1_u8))
        .collect::<Vec<_>>();
    edges.sort_unstable();
    OwnedOracleCase { name: name.to_owned(), vertex_labels: vec![0_u8; number_of_nodes], edges }
}

fn prism_case(name: &str, cycle_size: usize) -> OwnedOracleCase {
    assert!(cycle_size >= 3);
    let number_of_nodes = cycle_size * 2;
    let mut edges = Vec::with_capacity(cycle_size * 3);
    for ring in 0..2 {
        let offset = ring * cycle_size;
        for index in 0..cycle_size {
            edges.push((offset + index, offset + ((index + 1) % cycle_size), 1_u8));
        }
    }
    for index in 0..cycle_size {
        edges.push((index, cycle_size + index, 2_u8));
    }
    OwnedOracleCase { name: name.to_owned(), vertex_labels: vec![0_u8; number_of_nodes], edges }
}

fn crown_case(name: &str, side_size: usize) -> OwnedOracleCase {
    assert!(side_size >= 3);
    let number_of_nodes = side_size * 2;
    let vertex_labels =
        (0..number_of_nodes).map(|index| if index < side_size { 0_u8 } else { 1_u8 }).collect();
    let mut edges = Vec::new();
    for left in 0..side_size {
        for right in 0..side_size {
            if left == right {
                continue;
            }
            edges.push((left, side_size + right, 1_u8));
        }
    }
    OwnedOracleCase { name: name.to_owned(), vertex_labels, edges }
}

fn matched_bipartite_shift_case(name: &str, side_size: usize) -> OwnedOracleCase {
    assert!(side_size >= 3);
    let number_of_nodes = side_size * 2;
    let vertex_labels =
        (0..number_of_nodes).map(|index| if index < side_size { 0_u8 } else { 1_u8 }).collect();
    let mut edges = Vec::with_capacity(side_size * 2);
    for left in 0..side_size {
        edges.push((left, side_size + left, 1_u8));
        edges.push((left, side_size + ((left + 1) % side_size), 1_u8));
    }
    OwnedOracleCase { name: name.to_owned(), vertex_labels, edges }
}

fn dimension_colored_cube_q3_case() -> OwnedOracleCase {
    let mut edges = Vec::new();
    for vertex in 0..8 {
        for bit in 0..3 {
            let neighbour = vertex ^ (1 << bit);
            if vertex < neighbour {
                edges.push((vertex, neighbour, (bit + 1) as u8));
            }
        }
    }
    OwnedOracleCase {
        name: "dimension_colored_cube_q3".to_owned(),
        vertex_labels: vec![0_u8; 8],
        edges,
    }
}

fn unlabeled_case_from_edges(
    name: &str,
    number_of_nodes: usize,
    edges: Vec<(usize, usize)>,
) -> OwnedOracleCase {
    OwnedOracleCase {
        name: name.to_owned(),
        vertex_labels: vec![0_u8; number_of_nodes],
        edges: edges.into_iter().map(|(left, right)| (left, right, 0_u8)).collect(),
    }
}

fn complement_case(name: &str, case: &OwnedOracleCase) -> OwnedOracleCase {
    let number_of_nodes = case.vertex_labels.len();
    let mut present = BTreeSet::new();
    for &(left, right, _) in &case.edges {
        let pair = if left < right { (left, right) } else { (right, left) };
        present.insert(pair);
    }

    let mut edges = Vec::new();
    for left in 0..number_of_nodes {
        for right in (left + 1)..number_of_nodes {
            if !present.contains(&(left, right)) {
                edges.push((left, right, 0_u8));
            }
        }
    }

    OwnedOracleCase { name: name.to_owned(), vertex_labels: case.vertex_labels.clone(), edges }
}

fn triangular_graph_case(name: &str, n: usize) -> OwnedOracleCase {
    assert!(n >= 4);
    let vertices =
        (0..n).flat_map(|left| ((left + 1)..n).map(move |right| (left, right))).collect::<Vec<_>>();
    let mut edges = Vec::new();
    for (left_index, &(a, b)) in vertices.iter().enumerate() {
        for (right_index, &(other_a, other_b)) in vertices.iter().enumerate().skip(left_index + 1) {
            if a == other_a || a == other_b || b == other_a || b == other_b {
                edges.push((left_index, right_index, 0_u8));
            }
        }
    }

    OwnedOracleCase { name: name.to_owned(), vertex_labels: vec![0_u8; vertices.len()], edges }
}

fn rook_graph_case(name: &str, rows: usize, cols: usize) -> OwnedOracleCase {
    let number_of_nodes = rows * cols;
    let mut edges = Vec::new();
    for left_row in 0..rows {
        for left_col in 0..cols {
            let left = left_row * cols + left_col;
            for right_row in 0..rows {
                for right_col in 0..cols {
                    let right = right_row * cols + right_col;
                    if left < right && (left_row == right_row || left_col == right_col) {
                        edges.push((left, right, 0_u8));
                    }
                }
            }
        }
    }

    OwnedOracleCase { name: name.to_owned(), vertex_labels: vec![0_u8; number_of_nodes], edges }
}

fn complete_multipartite_case(name: &str, parts: &[usize]) -> OwnedOracleCase {
    let mut offsets = Vec::with_capacity(parts.len());
    let mut total = 0;
    for &part in parts {
        offsets.push(total);
        total += part;
    }

    let mut edges = Vec::new();
    for left_part in 0..parts.len() {
        for right_part in (left_part + 1)..parts.len() {
            for left in 0..parts[left_part] {
                for right in 0..parts[right_part] {
                    edges.push((offsets[left_part] + left, offsets[right_part] + right, 0_u8));
                }
            }
        }
    }

    OwnedOracleCase { name: name.to_owned(), vertex_labels: vec![0_u8; total], edges }
}

fn mobius_ladder_case(name: &str, cycle_half_size: usize) -> OwnedOracleCase {
    assert!(cycle_half_size >= 3);
    let number_of_nodes = cycle_half_size * 2;
    let mut edges = Vec::with_capacity(number_of_nodes + cycle_half_size);
    for vertex in 0..number_of_nodes {
        edges.push((vertex, (vertex + 1) % number_of_nodes, 0_u8));
    }
    for vertex in 0..cycle_half_size {
        edges.push((vertex, vertex + cycle_half_size, 0_u8));
    }
    edges.sort_unstable();
    edges.dedup();

    OwnedOracleCase { name: name.to_owned(), vertex_labels: vec![0_u8; number_of_nodes], edges }
}

fn family_search_stat_mismatch(case: &OwnedOracleCase) -> Result<(), String> {
    let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
    let matrix = Edges::matrix(graph.edges());
    let rust_result = canonical_label_labeled_simple_graph(
        &graph,
        |node| case.vertex_labels[node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
    );
    let bliss_result = canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges)
        .map_err(|error| format!("{}: bliss oracle run failed: {error}", case.name))?;

    let bliss_nodes = bliss_result
        .stats
        .nodes
        .ok_or_else(|| format!("{}: bliss did not report node count", case.name))?;
    let bliss_leaf_nodes = bliss_result
        .stats
        .leaf_nodes
        .ok_or_else(|| format!("{}: bliss did not report leaf-node count", case.name))?;

    let rust_stats = rust_result.stats;
    if rust_stats.search_nodes != bliss_nodes || rust_stats.leaf_nodes != bliss_leaf_nodes {
        return Err(format!(
            "{}\nvertex_labels={:?}\nedges={:?}\nrust_stats={:?}\nbliss_stats={{search_nodes: {bliss_nodes}, leaf_nodes: {bliss_leaf_nodes}}}",
            case.name, case.vertex_labels, case.edges, rust_stats
        ));
    }

    Ok(())
}

fn named_symmetric_family_cases() -> Vec<OwnedOracleCase> {
    let petersen = unlabeled_case_from_edges(
        "petersen",
        10,
        vec![
            (0, 1),
            (0, 4),
            (0, 5),
            (1, 2),
            (1, 6),
            (2, 3),
            (2, 7),
            (3, 4),
            (3, 8),
            (4, 9),
            (5, 7),
            (5, 8),
            (6, 8),
            (6, 9),
            (7, 9),
        ],
    );
    let prism_n3 = prism_case("triangular_prism_n3", 3);
    let prism_n4 = prism_case("square_prism_n4", 4);
    let mobius_n4 = mobius_ladder_case("mobius_ladder_n4", 4);
    let mobius_n5 = mobius_ladder_case("mobius_ladder_n5", 5);
    let q4 = unlabeled_case_from_edges(
        "hypercube_q4",
        16,
        (0..16)
            .flat_map(|vertex| {
                (0..4).filter_map(move |bit| {
                    let neighbour = vertex ^ (1 << bit);
                    (vertex < neighbour).then_some((vertex, neighbour))
                })
            })
            .collect(),
    );
    let q5 = unlabeled_case_from_edges(
        "hypercube_q5",
        32,
        (0..32)
            .flat_map(|vertex| {
                (0..5).filter_map(move |bit| {
                    let neighbour = vertex ^ (1 << bit);
                    (vertex < neighbour).then_some((vertex, neighbour))
                })
            })
            .collect(),
    );
    let rook_3x3 = rook_graph_case("rook_3x3", 3, 3);
    let rook_4x4 = rook_graph_case("rook_4x4", 4, 4);
    let triangular_5 = triangular_graph_case("triangular_graph_t5", 5);
    let triangular_6 = triangular_graph_case("triangular_graph_t6", 6);
    let friendship_3 = unlabeled_case_from_edges(
        "friendship_f3",
        7,
        vec![(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4), (0, 5), (0, 6), (5, 6)],
    );
    let friendship_4 = unlabeled_case_from_edges(
        "friendship_f4",
        9,
        vec![
            (0, 1),
            (0, 2),
            (1, 2),
            (0, 3),
            (0, 4),
            (3, 4),
            (0, 5),
            (0, 6),
            (5, 6),
            (0, 7),
            (0, 8),
            (7, 8),
        ],
    );
    let windmill_3_4 = unlabeled_case_from_edges(
        "windmill_wd_3_4",
        10,
        vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (4, 5),
            (4, 6),
            (5, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (7, 8),
            (7, 9),
            (8, 9),
        ],
    );
    let multipartite_222 = complete_multipartite_case("complete_multipartite_2_2_2", &[2, 2, 2]);
    let multipartite_333 = complete_multipartite_case("complete_multipartite_3_3_3", &[3, 3, 3]);
    let multipartite_2222 =
        complete_multipartite_case("complete_multipartite_2_2_2_2", &[2, 2, 2, 2]);
    let prism_n5 = prism_case("triangular_prism_n5", 5);

    let cases = vec![
        petersen.clone(),
        complement_case("petersen_complement", &petersen),
        prism_n3.clone(),
        complement_case("triangular_prism_n3_complement", &prism_n3),
        prism_n4.clone(),
        complement_case("square_prism_n4_complement", &prism_n4),
        mobius_n4.clone(),
        complement_case("mobius_ladder_n4_complement", &mobius_n4),
        mobius_n5.clone(),
        complement_case("mobius_ladder_n5_complement", &mobius_n5),
        q4.clone(),
        complement_case("hypercube_q4_complement", &q4),
        q5,
        rook_3x3.clone(),
        complement_case("rook_3x3_complement", &rook_3x3),
        rook_4x4.clone(),
        complement_case("rook_4x4_complement", &rook_4x4),
        triangular_5.clone(),
        complement_case("triangular_graph_t5_complement", &triangular_5),
        triangular_6.clone(),
        complement_case("triangular_graph_t6_complement", &triangular_6),
        friendship_3.clone(),
        complement_case("friendship_f3_complement", &friendship_3),
        friendship_4.clone(),
        complement_case("friendship_f4_complement", &friendship_4),
        windmill_3_4.clone(),
        complement_case("windmill_wd_3_4_complement", &windmill_3_4),
        multipartite_222.clone(),
        complement_case("complete_multipartite_2_2_2_complement", &multipartite_222),
        multipartite_333.clone(),
        complement_case("complete_multipartite_3_3_3_complement", &multipartite_333),
        multipartite_2222.clone(),
        complement_case("complete_multipartite_2_2_2_2_complement", &multipartite_2222),
        prism_n5.clone(),
        complement_case("triangular_prism_n5_complement", &prism_n5),
    ];

    cases
}

fn repeated_component_case<FV, FE>(
    name: String,
    copies: usize,
    component_vertex_labels: &[u8],
    component_edges: &[(usize, usize, u8)],
    mut vertex_label_transform: FV,
    mut edge_label_transform: FE,
) -> OwnedOracleCase
where
    FV: FnMut(usize, usize, u8) -> u8,
    FE: FnMut(usize, usize, usize, u8) -> u8,
{
    let component_vertex_count = component_vertex_labels.len();
    let mut vertex_labels = Vec::with_capacity(copies * component_vertex_count);
    let mut edges = Vec::with_capacity(copies * component_edges.len());

    for copy in 0..copies {
        let offset = copy * component_vertex_count;
        for (local_vertex, &label) in component_vertex_labels.iter().enumerate() {
            vertex_labels.push(vertex_label_transform(copy, local_vertex, label));
        }
        for &(left, right, label) in component_edges {
            edges.push((
                offset + left,
                offset + right,
                edge_label_transform(copy, left, right, label),
            ));
        }
    }

    OwnedOracleCase { name, vertex_labels, edges }
}

fn path_component_case(path_vertices: usize) -> (Vec<u8>, Vec<(usize, usize, u8)>) {
    assert!(path_vertices >= 2);
    let vertex_labels = vec![0_u8; path_vertices];
    let mut edges = Vec::with_capacity(path_vertices - 1);
    for index in 0..(path_vertices - 1) {
        edges.push((index, index + 1, 0_u8));
    }
    (vertex_labels, edges)
}

fn cycle_component_case(cycle_vertices: usize) -> (Vec<u8>, Vec<(usize, usize, u8)>) {
    assert!(cycle_vertices >= 3);
    let vertex_labels = vec![0_u8; cycle_vertices];
    let mut edges = Vec::with_capacity(cycle_vertices);
    for index in 0..cycle_vertices {
        edges.push((index, (index + 1) % cycle_vertices, 0_u8));
    }
    (vertex_labels, edges)
}

fn ladder_component_case(rungs: usize) -> (Vec<u8>, Vec<(usize, usize, u8)>) {
    assert!(rungs >= 2);
    let vertex_labels = vec![0_u8; rungs * 2];
    let mut edges = Vec::with_capacity(rungs * 3 - 2);
    for index in 0..(rungs - 1) {
        edges.push((index, index + 1, 0_u8));
        edges.push((rungs + index, rungs + index + 1, 0_u8));
    }
    for index in 0..rungs {
        edges.push((index, rungs + index, 0_u8));
    }
    (vertex_labels, edges)
}

fn star_component_case(leaves: usize) -> (Vec<u8>, Vec<(usize, usize, u8)>) {
    assert!(leaves >= 2);
    let vertex_labels = vec![0_u8; leaves + 1];
    let mut edges = Vec::with_capacity(leaves);
    for leaf in 1..=leaves {
        edges.push((0, leaf, 0_u8));
    }
    (vertex_labels, edges)
}

fn complete_bipartite_component_case(
    left_size: usize,
    right_size: usize,
) -> (Vec<u8>, Vec<(usize, usize, u8)>) {
    assert!(left_size >= 1);
    assert!(right_size >= 1);
    let vertex_labels = vec![0_u8; left_size + right_size];
    let mut edges = Vec::with_capacity(left_size * right_size);
    for left in 0..left_size {
        for right in 0..right_size {
            edges.push((left, left_size + right, 0_u8));
        }
    }
    (vertex_labels, edges)
}

fn repeated_component_family_cases() -> Vec<OwnedOracleCase> {
    let mut cases = Vec::new();

    let copies = [2_usize, 3, 4, 5];
    let path_sizes = [2_usize, 3, 4, 5, 6];
    let cycle_sizes = [3_usize, 4, 5, 6];
    let ladder_rungs = [2_usize, 3, 4, 5];
    let star_leaves = [2_usize, 3, 4, 5, 6];
    let bipartite_sizes = [(1_usize, 2_usize), (1, 3), (2, 2), (2, 3), (2, 4), (3, 3)];

    for &copies in &copies {
        for &path_vertices in &path_sizes {
            let (component_vertex_labels, component_edges) = path_component_case(path_vertices);
            let base_name = format!("repeated_path_k{copies}_p{path_vertices}");
            cases.push(repeated_component_case(
                base_name.clone(),
                copies,
                &component_vertex_labels,
                &component_edges,
                |_, _, label| label,
                |_, _, _, label| label,
            ));
            cases.push(repeated_component_case(
                format!("{base_name}_component_colored"),
                copies,
                &component_vertex_labels,
                &component_edges,
                |copy, _, _| copy as u8,
                |_, _, _, label| label,
            ));
            cases.push(repeated_component_case(
                format!("{base_name}_root_perturbed"),
                copies,
                &component_vertex_labels,
                &component_edges,
                |_, local_vertex, _| if local_vertex == 0 { 1 } else { 0 },
                |_, _, _, label| label,
            ));
            cases.push(repeated_component_case(
                format!("{base_name}_first_edge_perturbed"),
                copies,
                &component_vertex_labels,
                &component_edges,
                |_, _, label| label,
                |copy, left, right, label| {
                    if copy == 0 && left == 0 && right == 1 { 1 } else { label }
                },
            ));
        }

        for &cycle_vertices in &cycle_sizes {
            let (component_vertex_labels, component_edges) = cycle_component_case(cycle_vertices);
            let base_name = format!("repeated_cycle_k{copies}_c{cycle_vertices}");
            cases.push(repeated_component_case(
                base_name.clone(),
                copies,
                &component_vertex_labels,
                &component_edges,
                |_, _, label| label,
                |_, _, _, label| label,
            ));
            cases.push(repeated_component_case(
                format!("{base_name}_component_colored"),
                copies,
                &component_vertex_labels,
                &component_edges,
                |copy, _, _| copy as u8,
                |_, _, _, label| label,
            ));
            cases.push(repeated_component_case(
                format!("{base_name}_alternating_labels"),
                copies,
                &component_vertex_labels,
                &component_edges,
                |copy, local_vertex, _| ((copy + local_vertex) % 2) as u8,
                |_, _, _, label| label,
            ));
            cases.push(repeated_component_case(
                format!("{base_name}_first_edge_perturbed"),
                copies,
                &component_vertex_labels,
                &component_edges,
                |_, _, label| label,
                |copy, left, right, label| {
                    if copy == 0 && left == 0 && right == 1 { 1 } else { label }
                },
            ));
        }

        for &rungs in &ladder_rungs {
            let (component_vertex_labels, component_edges) = ladder_component_case(rungs);
            let base_name = format!("repeated_ladder_k{copies}_r{rungs}");
            cases.push(repeated_component_case(
                base_name.clone(),
                copies,
                &component_vertex_labels,
                &component_edges,
                |_, _, label| label,
                |_, _, _, label| label,
            ));
            cases.push(repeated_component_case(
                format!("{base_name}_component_colored"),
                copies,
                &component_vertex_labels,
                &component_edges,
                |copy, _, _| copy as u8,
                |_, _, _, label| label,
            ));
            cases.push(repeated_component_case(
                format!("{base_name}_first_edge_perturbed"),
                copies,
                &component_vertex_labels,
                &component_edges,
                |_, _, label| label,
                |copy, left, right, label| {
                    if copy == 0 && left == 0 && right == 1 { 1 } else { label }
                },
            ));
        }

        for &leaves in &star_leaves {
            let (component_vertex_labels, component_edges) = star_component_case(leaves);
            let base_name = format!("repeated_star_k{copies}_s{leaves}");
            cases.push(repeated_component_case(
                base_name.clone(),
                copies,
                &component_vertex_labels,
                &component_edges,
                |_, _, label| label,
                |_, _, _, label| label,
            ));
            cases.push(repeated_component_case(
                format!("{base_name}_component_colored"),
                copies,
                &component_vertex_labels,
                &component_edges,
                |copy, _, _| copy as u8,
                |_, _, _, label| label,
            ));
            cases.push(repeated_component_case(
                format!("{base_name}_hub_perturbed"),
                copies,
                &component_vertex_labels,
                &component_edges,
                |_, local_vertex, _| if local_vertex == 0 { 1 } else { 0 },
                |_, _, _, label| label,
            ));
        }

        for &(left_size, right_size) in &bipartite_sizes {
            let (component_vertex_labels, component_edges) =
                complete_bipartite_component_case(left_size, right_size);
            let base_name =
                format!("repeated_complete_bipartite_k{copies}_{left_size}x{right_size}");
            cases.push(repeated_component_case(
                base_name.clone(),
                copies,
                &component_vertex_labels,
                &component_edges,
                |_, _, label| label,
                |_, _, _, label| label,
            ));
            cases.push(repeated_component_case(
                format!("{base_name}_component_colored"),
                copies,
                &component_vertex_labels,
                &component_edges,
                |copy, _, _| copy as u8,
                |_, _, _, label| label,
            ));
            cases.push(repeated_component_case(
                format!("{base_name}_one_label_perturbed"),
                copies,
                &component_vertex_labels,
                &component_edges,
                |_, local_vertex, _| if local_vertex == 0 { 1 } else { 0 },
                |_, _, _, label| label,
            ));
            cases.push(repeated_component_case(
                format!("{base_name}_first_edge_perturbed"),
                copies,
                &component_vertex_labels,
                &component_edges,
                |_, _, label| label,
                |copy, left, right, label| {
                    if copy == 0 && left == 0 && right == 0 + left_size { 1 } else { label }
                },
            ));
        }
    }

    cases
}

fn describe_case_mismatch(case: &OwnedOracleCase, error: &str) -> String {
    let mut description = String::new();
    let _ = writeln!(&mut description, "name={}", case.name);
    let _ = writeln!(&mut description, "vertex_labels={:?}", case.vertex_labels);
    let _ = writeln!(&mut description, "edges={:?}", case.edges);
    let _ = write!(&mut description, "error={error}");
    description
}

fn random_case_count() -> usize {
    env::var("GEOMETRIC_TRAITS_BLISS_RANDOM_CASES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(256)
}

fn random_case_base_seed() -> u64 {
    env::var("GEOMETRIC_TRAITS_BLISS_RANDOM_BASE_SEED")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0xB115_5100_0000_0001)
}

fn random_case_parallelism() -> usize {
    env::var("GEOMETRIC_TRAITS_BLISS_RANDOM_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&threads| threads > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism().map(std::num::NonZeroUsize::get).unwrap_or(1)
        })
}

fn random_window_progress_enabled() -> bool {
    env::var("GEOMETRIC_TRAITS_BLISS_PROGRESS")
        .ok()
        .map(|value| !matches!(value.as_str(), "0" | "false" | "False" | "FALSE"))
        .unwrap_or(true)
}

fn random_window_progress_bar(case_count: usize, label: &str) -> ProgressBar {
    let progress = ProgressBar::new(case_count as u64);
    if random_window_progress_enabled() {
        let style = ProgressStyle::with_template(
            "{spinner:.green} {msg:>14} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos:>7}/{len:7} ({per_sec}, eta {eta_precise})",
        )
        .expect("progress template should be valid")
        .progress_chars("##-");
        progress.set_style(style);
        progress.set_message(label.to_owned());
    } else {
        progress.set_draw_target(ProgressDrawTarget::hidden());
    }
    progress
}

fn random_labeled_simple_case(seed: u64) -> OwnedOracleCase {
    let mut rng = SmallRng::seed_from_u64(seed);
    let number_of_nodes = rng.gen_range(2..=12);
    let vertex_palette = rng.gen_range(1_u8..=4);
    let edge_palette = rng.gen_range(1_u8..=4);
    let vertex_labels =
        (0..number_of_nodes).map(|_| rng.gen_range(0_u8..vertex_palette)).collect::<Vec<_>>();
    let connected = rng.gen_bool(0.6);
    let max_edges = number_of_nodes * (number_of_nodes - 1) / 2;
    let min_edges = if connected { number_of_nodes - 1 } else { 0 };
    let target_edges = rng.gen_range(min_edges..=max_edges);

    let mut seen = BTreeSet::new();
    let mut edges = Vec::with_capacity(target_edges);
    if connected {
        for node in 1..number_of_nodes {
            let parent = rng.gen_range(0..node);
            let (left, right) = if parent < node { (parent, node) } else { (node, parent) };
            seen.insert((left, right));
            edges.push((left, right, rng.gen_range(0_u8..edge_palette)));
        }
    }

    let mut remaining_pairs = Vec::with_capacity(max_edges.saturating_sub(edges.len()));
    for left in 0..number_of_nodes {
        for right in (left + 1)..number_of_nodes {
            if seen.contains(&(left, right)) {
                continue;
            }
            remaining_pairs.push((left, right));
        }
    }
    remaining_pairs.shuffle(&mut rng);

    for (left, right) in remaining_pairs.into_iter().take(target_edges.saturating_sub(edges.len()))
    {
        edges.push((left, right, rng.gen_range(0_u8..edge_palette)));
    }

    edges.sort_unstable_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));

    OwnedOracleCase {
        name: format!("random_seed_{seed}_n{number_of_nodes}_m{}", edges.len()),
        vertex_labels,
        edges,
    }
}

fn ternary_edge_mask_case(name: &str, number_of_nodes: usize, mask: u64) -> OwnedOracleCase {
    let mut edges = Vec::new();
    let mut digit_stream = mask;
    for left in 0..number_of_nodes {
        for right in (left + 1)..number_of_nodes {
            match digit_stream % 3 {
                0 => {}
                1 => edges.push((left, right, 0_u8)),
                2 => edges.push((left, right, 1_u8)),
                _ => unreachable!(),
            }
            digit_stream /= 3;
        }
    }

    OwnedOracleCase { name: name.to_owned(), vertex_labels: vec![0_u8; number_of_nodes], edges }
}

fn binary_vertex_mask_and_ternary_edge_mask_case(
    name: &str,
    number_of_nodes: usize,
    vertex_mask: u64,
    edge_mask: u64,
) -> OwnedOracleCase {
    let vertex_labels =
        (0..number_of_nodes).map(|index| ((vertex_mask >> index) & 1) as u8).collect::<Vec<_>>();
    let mut edges = Vec::new();
    let mut digit_stream = edge_mask;
    for left in 0..number_of_nodes {
        for right in (left + 1)..number_of_nodes {
            match digit_stream % 3 {
                0 => {}
                1 => edges.push((left, right, 0_u8)),
                2 => edges.push((left, right, 1_u8)),
                _ => unreachable!(),
            }
            digit_stream /= 3;
        }
    }

    OwnedOracleCase { name: name.to_owned(), vertex_labels, edges }
}

fn describe_owned_case(seed: u64, case: &OwnedOracleCase, error: &str) -> String {
    let mut description = String::new();
    let _ = writeln!(&mut description, "seed={seed} name={}", case.name);
    let _ = writeln!(&mut description, "vertex_labels={:?}", case.vertex_labels);
    let _ = writeln!(&mut description, "edges={:?}", case.edges);
    let _ = write!(&mut description, "error={error}");
    description
}

fn collect_random_window_mismatches(
    base_seed: u64,
    case_count: usize,
    parallelism: usize,
) -> Vec<String> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallelism)
        .build()
        .expect("rayon pool for random bliss differential testing should build");

    let progress = random_window_progress_bar(case_count, "bliss parity");
    let mismatches = pool.install(|| {
        (0..case_count)
            .into_par_iter()
            .progress_with(progress.clone())
            .filter_map(|offset| {
                let seed = base_seed.wrapping_add(offset as u64);
                let case = random_labeled_simple_case(seed);
                compare_rust_and_bliss_owned_case(&case)
                    .err()
                    .map(|error| describe_owned_case(seed, &case, &error))
            })
            .collect::<Vec<_>>()
    });
    progress.finish_and_clear();
    mismatches
}

fn collect_random_window_stat_mismatches(
    base_seed: u64,
    case_count: usize,
    parallelism: usize,
) -> Vec<String> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallelism)
        .build()
        .expect("rayon pool for random bliss search-stat differential testing should build");

    let progress = random_window_progress_bar(case_count, "bliss stats");
    let mismatches = pool.install(|| {
        (0..case_count)
            .into_par_iter()
            .progress_with(progress.clone())
            .filter_map(|offset| {
                let seed = base_seed.wrapping_add(offset as u64);
                let case = random_labeled_simple_case(seed);
                compare_rust_and_bliss_stats_owned_case(&case)
                    .err()
                    .map(|error| describe_owned_case(seed, &case, &error))
            })
            .collect::<Vec<_>>()
    });
    progress.finish_and_clear();
    mismatches
}

fn discovered_random_gap_cases() -> Vec<OwnedOracleCase> {
    vec![
        OwnedOracleCase {
            name: "random_seed_12760194179666018358_n9_m34".to_owned(),
            vertex_labels: vec![3, 3, 0, 2, 0, 1, 3, 0, 3],
            edges: vec![
                (0, 1, 1),
                (0, 2, 0),
                (0, 3, 1),
                (0, 4, 1),
                (0, 5, 0),
                (0, 6, 1),
                (0, 7, 1),
                (0, 8, 0),
                (1, 2, 1),
                (1, 3, 0),
                (1, 4, 1),
                (1, 6, 0),
                (1, 7, 0),
                (1, 8, 0),
                (2, 3, 0),
                (2, 4, 1),
                (2, 5, 0),
                (2, 7, 0),
                (2, 8, 0),
                (3, 4, 0),
                (3, 5, 1),
                (3, 6, 1),
                (3, 7, 1),
                (3, 8, 1),
                (4, 5, 1),
                (4, 6, 1),
                (4, 7, 0),
                (4, 8, 1),
                (5, 6, 0),
                (5, 7, 0),
                (5, 8, 1),
                (6, 7, 0),
                (6, 8, 1),
                (7, 8, 1),
            ],
        },
        OwnedOracleCase {
            name: "random_seed_12760194179666018420_n12_m44".to_owned(),
            vertex_labels: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            edges: vec![
                (0, 2, 1),
                (0, 3, 0),
                (0, 4, 0),
                (0, 5, 1),
                (0, 6, 0),
                (0, 7, 0),
                (0, 8, 0),
                (0, 9, 0),
                (0, 10, 1),
                (0, 11, 1),
                (1, 2, 1),
                (1, 3, 0),
                (1, 5, 0),
                (1, 6, 1),
                (1, 8, 0),
                (1, 9, 0),
                (1, 10, 0),
                (2, 3, 0),
                (2, 4, 1),
                (2, 6, 0),
                (2, 8, 1),
                (2, 9, 0),
                (2, 11, 0),
                (3, 5, 1),
                (3, 6, 1),
                (3, 7, 1),
                (3, 8, 1),
                (3, 9, 0),
                (4, 5, 1),
                (4, 7, 1),
                (4, 8, 0),
                (4, 10, 1),
                (5, 6, 0),
                (5, 7, 1),
                (5, 11, 1),
                (6, 8, 1),
                (6, 10, 1),
                (6, 11, 0),
                (7, 8, 0),
                (7, 9, 1),
                (7, 10, 0),
                (7, 11, 0),
                (8, 10, 1),
                (10, 11, 0),
            ],
        },
        OwnedOracleCase {
            name: "random_seed_424242424242424574_n10_m45".to_owned(),
            vertex_labels: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            edges: vec![
                (0, 1, 0),
                (0, 2, 3),
                (0, 3, 3),
                (0, 4, 3),
                (0, 5, 1),
                (0, 6, 2),
                (0, 7, 3),
                (0, 8, 3),
                (0, 9, 0),
                (1, 2, 0),
                (1, 3, 2),
                (1, 4, 2),
                (1, 5, 0),
                (1, 6, 3),
                (1, 7, 2),
                (1, 8, 1),
                (1, 9, 0),
                (2, 3, 1),
                (2, 4, 3),
                (2, 5, 0),
                (2, 6, 0),
                (2, 7, 3),
                (2, 8, 0),
                (2, 9, 3),
                (3, 4, 0),
                (3, 5, 2),
                (3, 6, 3),
                (3, 7, 0),
                (3, 8, 1),
                (3, 9, 2),
                (4, 5, 2),
                (4, 6, 0),
                (4, 7, 3),
                (4, 8, 1),
                (4, 9, 2),
                (5, 6, 2),
                (5, 7, 1),
                (5, 8, 3),
                (5, 9, 3),
                (6, 7, 1),
                (6, 8, 1),
                (6, 9, 3),
                (7, 8, 3),
                (7, 9, 0),
                (8, 9, 0),
            ],
        },
    ]
}

fn aggressive_random_gap_seeds() -> [u64; 7] {
    [
        424242424242424431,
        424242424242424687,
        424242424242424702,
        424242424242424821,
        424242424242425041,
        424242424242425158,
        424242424242425189,
    ]
}

fn representative_random_stat_gap_cases() -> Vec<OwnedOracleCase> {
    vec![
        OwnedOracleCase {
            name: "random_seed_12760194179666018312_n5_m1".to_owned(),
            vertex_labels: vec![1_u8, 0, 0, 1, 2],
            edges: vec![(0, 3, 2)],
        },
        OwnedOracleCase {
            name: "random_seed_12760194179666018313_n6_m7".to_owned(),
            vertex_labels: vec![0_u8, 0, 0, 0, 0, 0],
            edges: vec![
                (0, 1, 0),
                (0, 2, 0),
                (1, 2, 0),
                (1, 3, 0),
                (1, 4, 0),
                (1, 5, 0),
                (4, 5, 0),
            ],
        },
        OwnedOracleCase {
            name: "random_seed_12760194179666018467_n6_m0".to_owned(),
            vertex_labels: vec![2_u8, 1, 1, 0, 2, 1],
            edges: vec![],
        },
        OwnedOracleCase {
            name: "random_seed_12760194179666018549_n5_m2".to_owned(),
            vertex_labels: vec![0_u8, 0, 0, 0, 0],
            edges: vec![(1, 4, 0), (3, 4, 0)],
        },
    ]
}

fn heavy_sparse_random_stat_gap_cases() -> Vec<OwnedOracleCase> {
    vec![
        OwnedOracleCase {
            name: "random_seed_738654835591397_n10_m6".to_owned(),
            vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            edges: vec![(0, 6, 0), (1, 7, 0), (4, 6, 0), (5, 7, 0), (6, 8, 0), (7, 9, 0)],
        },
        OwnedOracleCase {
            name: "random_seed_777777777777789045_n7_m3".to_owned(),
            vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0],
            edges: vec![(0, 1, 0), (3, 5, 0), (4, 6, 0)],
        },
        OwnedOracleCase {
            name: "random_seed_777777777777794946_n12_m3".to_owned(),
            vertex_labels: vec![1_u8, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1],
            edges: vec![(0, 10, 0), (1, 11, 0), (5, 9, 0)],
        },
        OwnedOracleCase {
            name: "random_seed_777777777777811691_n11_m3".to_owned(),
            vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            edges: vec![(0, 10, 1), (1, 3, 1), (6, 8, 1)],
        },
        OwnedOracleCase {
            name: "random_seed_777777777777829310_n12_m3".to_owned(),
            vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            edges: vec![(2, 6, 0), (3, 4, 0), (5, 11, 0)],
        },
        OwnedOracleCase {
            name: "random_seed_777777777777830031_n9_m3".to_owned(),
            vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0],
            edges: vec![(1, 3, 0), (4, 7, 0), (5, 8, 0)],
        },
        OwnedOracleCase {
            name: "random_seed_777777777777834966_n11_m5".to_owned(),
            vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            edges: vec![(0, 2, 0), (3, 4, 0), (6, 8, 0), (7, 10, 0), (9, 10, 0)],
        },
        OwnedOracleCase {
            name: "random_seed_777777777777835244_n8_m3".to_owned(),
            vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0],
            edges: vec![(1, 7, 0), (2, 5, 0), (3, 4, 0)],
        },
        OwnedOracleCase {
            name: "random_seed_777777777777841385_n8_m3".to_owned(),
            vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0],
            edges: vec![(0, 4, 0), (3, 6, 0), (5, 7, 0)],
        },
        OwnedOracleCase {
            name: "random_seed_1000000000000001877_n10_m3".to_owned(),
            vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            edges: vec![(1, 4, 0), (2, 8, 0), (3, 9, 0)],
        },
    ]
}

fn heavy_dense_random_stat_gap_cases() -> Vec<OwnedOracleCase> {
    vec![
        OwnedOracleCase {
            name: "random_seed_333333333333457088_n9_m36".to_owned(),
            vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0],
            edges: vec![
                (0, 1, 1),
                (0, 2, 1),
                (0, 3, 1),
                (0, 4, 0),
                (0, 5, 1),
                (0, 6, 1),
                (0, 7, 1),
                (0, 8, 0),
                (1, 2, 1),
                (1, 3, 1),
                (1, 4, 1),
                (1, 5, 0),
                (1, 6, 1),
                (1, 7, 0),
                (1, 8, 1),
                (2, 3, 1),
                (2, 4, 0),
                (2, 5, 0),
                (2, 6, 1),
                (2, 7, 1),
                (2, 8, 1),
                (3, 4, 0),
                (3, 5, 1),
                (3, 6, 1),
                (3, 7, 0),
                (3, 8, 1),
                (4, 5, 1),
                (4, 6, 1),
                (4, 7, 1),
                (4, 8, 0),
                (5, 6, 1),
                (5, 7, 0),
                (5, 8, 1),
                (6, 7, 0),
                (6, 8, 0),
                (7, 8, 1),
            ],
        },
        OwnedOracleCase {
            name: "random_seed_333333333333355939_n10_m26".to_owned(),
            vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            edges: vec![
                (0, 1, 0),
                (0, 3, 0),
                (0, 5, 0),
                (0, 7, 0),
                (0, 9, 0),
                (1, 2, 0),
                (1, 3, 0),
                (1, 7, 0),
                (1, 9, 0),
                (2, 3, 0),
                (2, 4, 0),
                (2, 5, 0),
                (2, 6, 0),
                (2, 7, 0),
                (3, 5, 0),
                (3, 6, 0),
                (3, 7, 0),
                (3, 8, 0),
                (4, 6, 0),
                (4, 8, 0),
                (5, 7, 0),
                (5, 8, 0),
                (5, 9, 0),
                (6, 7, 0),
                (6, 8, 0),
                (7, 8, 0),
            ],
        },
        OwnedOracleCase {
            name: "random_seed_238646355093013_n8_m21".to_owned(),
            vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0],
            edges: vec![
                (0, 1, 0),
                (0, 4, 0),
                (0, 5, 0),
                (0, 6, 0),
                (0, 7, 0),
                (1, 2, 0),
                (1, 3, 0),
                (1, 4, 0),
                (1, 6, 0),
                (2, 3, 0),
                (2, 4, 0),
                (2, 5, 0),
                (2, 7, 0),
                (3, 4, 0),
                (3, 5, 0),
                (3, 7, 0),
                (4, 5, 0),
                (4, 6, 0),
                (4, 7, 0),
                (5, 6, 0),
                (6, 7, 0),
            ],
        },
        OwnedOracleCase {
            name: "random_seed_238646355258545_n7_m21".to_owned(),
            vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0],
            edges: vec![
                (0, 1, 0),
                (0, 2, 1),
                (0, 3, 0),
                (0, 4, 1),
                (0, 5, 0),
                (0, 6, 0),
                (1, 2, 0),
                (1, 3, 0),
                (1, 4, 0),
                (1, 5, 1),
                (1, 6, 1),
                (2, 3, 1),
                (2, 4, 0),
                (2, 5, 0),
                (2, 6, 0),
                (3, 4, 1),
                (3, 5, 0),
                (3, 6, 0),
                (4, 5, 0),
                (4, 6, 0),
                (5, 6, 1),
            ],
        },
        OwnedOracleCase {
            name: "random_seed_777777777777782620_n8_m15".to_owned(),
            vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0],
            edges: vec![
                (0, 2, 0),
                (0, 4, 0),
                (0, 6, 0),
                (0, 7, 0),
                (1, 2, 0),
                (1, 4, 0),
                (1, 5, 0),
                (1, 6, 0),
                (2, 3, 0),
                (2, 7, 0),
                (3, 5, 0),
                (3, 6, 0),
                (4, 7, 0),
                (5, 6, 0),
                (5, 7, 0),
            ],
        },
        OwnedOracleCase {
            name: "random_seed_777777777777822712_n9_m29".to_owned(),
            vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0],
            edges: vec![
                (0, 1, 0),
                (0, 2, 0),
                (0, 4, 0),
                (0, 6, 0),
                (0, 7, 0),
                (0, 8, 0),
                (1, 3, 0),
                (1, 4, 0),
                (1, 5, 0),
                (1, 6, 0),
                (1, 7, 0),
                (2, 3, 0),
                (2, 4, 0),
                (2, 5, 0),
                (2, 7, 0),
                (2, 8, 0),
                (3, 4, 0),
                (3, 6, 0),
                (3, 7, 0),
                (3, 8, 0),
                (4, 5, 0),
                (4, 6, 0),
                (4, 7, 0),
                (4, 8, 0),
                (5, 6, 0),
                (5, 7, 0),
                (5, 8, 0),
                (6, 7, 0),
                (7, 8, 0),
            ],
        },
    ]
}

fn complete_graph_edges(number_of_nodes: usize, label: u8) -> Vec<(usize, usize, u8)> {
    let mut edges = Vec::with_capacity(number_of_nodes * number_of_nodes.saturating_sub(1) / 2);
    for left in 0..number_of_nodes {
        for right in (left + 1)..number_of_nodes {
            edges.push((left, right, label));
        }
    }
    edges
}

fn cycle_edges(number_of_nodes: usize, label: u8) -> Vec<(usize, usize, u8)> {
    let mut edges = (0..number_of_nodes)
        .map(|index| (index, (index + 1) % number_of_nodes, label))
        .collect::<Vec<_>>();
    edges.sort_unstable_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
    edges
}

fn complete_bipartite_edges(
    left_size: usize,
    right_size: usize,
    label: u8,
) -> Vec<(usize, usize, u8)> {
    let mut edges = Vec::with_capacity(left_size * right_size);
    for left in 0..left_size {
        for right in 0..right_size {
            edges.push((left, left_size + right, label));
        }
    }
    edges
}

fn repeated_component_edges(component_sizes: &[usize], label: u8) -> Vec<(usize, usize, u8)> {
    let mut edges = Vec::new();
    let mut offset = 0;
    for &component_size in component_sizes {
        for left in 0..component_size {
            for right in (left + 1)..component_size {
                edges.push((offset + left, offset + right, label));
            }
        }
        offset += component_size;
    }
    edges
}

fn complete_minus_edges(
    number_of_nodes: usize,
    missing: &[(usize, usize)],
    label: u8,
) -> Vec<(usize, usize, u8)> {
    let mut edges = complete_graph_edges(number_of_nodes, label);
    edges.retain(|(left, right, _)| {
        !missing.iter().any(|&(missing_left, missing_right)| {
            (*left == missing_left && *right == missing_right)
                || (*left == missing_right && *right == missing_left)
        })
    });
    edges
}

fn label_perturbation_sweep_cases() -> Vec<OwnedOracleCase> {
    let mut cases = Vec::new();

    for &number_of_nodes in &[6_usize, 8] {
        let base_edges = cycle_edges(number_of_nodes, 0);
        cases.push(OwnedOracleCase {
            name: format!("cycle_n{number_of_nodes}_uniform"),
            vertex_labels: vec![0_u8; number_of_nodes],
            edges: base_edges.clone(),
        });

        let orbit_labels = (0..number_of_nodes).map(|index| (index % 3) as u8).collect::<Vec<_>>();
        cases.push(OwnedOracleCase {
            name: format!("cycle_n{number_of_nodes}_orbit_012"),
            vertex_labels: orbit_labels,
            edges: base_edges.clone(),
        });

        if number_of_nodes % 2 == 0 {
            let alternating_labels =
                (0..number_of_nodes).map(|index| (index % 2) as u8).collect::<Vec<_>>();
            cases.push(OwnedOracleCase {
                name: format!("cycle_n{number_of_nodes}_orbit_01"),
                vertex_labels: alternating_labels,
                edges: base_edges.clone(),
            });
        }

        for vertex in 0..number_of_nodes {
            let mut vertex_labels = vec![0_u8; number_of_nodes];
            vertex_labels[vertex] = 1;
            cases.push(OwnedOracleCase {
                name: format!("cycle_n{number_of_nodes}_vertex_{vertex}_label_1"),
                vertex_labels,
                edges: base_edges.clone(),
            });
        }

        for edge_index in 0..number_of_nodes {
            let mut edges = base_edges.clone();
            edges[edge_index].2 = 1;
            cases.push(OwnedOracleCase {
                name: format!("cycle_n{number_of_nodes}_edge_{edge_index}_label_1"),
                vertex_labels: vec![0_u8; number_of_nodes],
                edges,
            });
        }

        let mut mixed_edges = base_edges.clone();
        mixed_edges[0].2 = 1;
        let midpoint = number_of_nodes / 2;
        mixed_edges[midpoint].2 = 2;
        cases.push(OwnedOracleCase {
            name: format!("cycle_n{number_of_nodes}_mixed_edge_labels_12"),
            vertex_labels: vec![0_u8; number_of_nodes],
            edges: mixed_edges,
        });
    }

    for &(left_size, right_size) in &[(3_usize, 3_usize), (4, 4)] {
        let number_of_nodes = left_size + right_size;
        let base_edges = complete_bipartite_edges(left_size, right_size, 0);
        let partition_labels = (0..number_of_nodes)
            .map(|index| if index < left_size { 0_u8 } else { 1_u8 })
            .collect::<Vec<_>>();
        let swapped_partition_labels = (0..number_of_nodes)
            .map(|index| if index < left_size { 1_u8 } else { 0_u8 })
            .collect::<Vec<_>>();

        cases.push(OwnedOracleCase {
            name: format!("complete_bipartite_{left_size}x{right_size}_partition_01"),
            vertex_labels: partition_labels.clone(),
            edges: base_edges.clone(),
        });
        cases.push(OwnedOracleCase {
            name: format!("complete_bipartite_{left_size}x{right_size}_partition_10"),
            vertex_labels: swapped_partition_labels,
            edges: base_edges.clone(),
        });

        for vertex in 0..number_of_nodes {
            let mut vertex_labels = partition_labels.clone();
            vertex_labels[vertex] ^= 1;
            cases.push(OwnedOracleCase {
                name: format!("complete_bipartite_{left_size}x{right_size}_vertex_{vertex}_flip"),
                vertex_labels,
                edges: base_edges.clone(),
            });
        }

        for edge_index in 0..base_edges.len() {
            let mut edges = base_edges.clone();
            edges[edge_index].2 = 1;
            cases.push(OwnedOracleCase {
                name: format!(
                    "complete_bipartite_{left_size}x{right_size}_edge_{edge_index}_label_1"
                ),
                vertex_labels: partition_labels.clone(),
                edges,
            });
        }

        let mut mixed_edges = base_edges.clone();
        mixed_edges[0].2 = 1;
        let midpoint = base_edges.len() / 2;
        mixed_edges[midpoint].2 = 2;
        cases.push(OwnedOracleCase {
            name: format!("complete_bipartite_{left_size}x{right_size}_mixed_edge_labels_12"),
            vertex_labels: partition_labels,
            edges: mixed_edges,
        });
    }

    for &number_of_nodes in &[7_usize, 8] {
        let base_edges = complete_minus_edges(number_of_nodes, &[(0, 1)], 0);
        cases.push(OwnedOracleCase {
            name: format!("near_complete_n{number_of_nodes}_uniform"),
            vertex_labels: vec![0_u8; number_of_nodes],
            edges: base_edges.clone(),
        });

        for vertex in 0..number_of_nodes {
            let mut vertex_labels = vec![0_u8; number_of_nodes];
            vertex_labels[vertex] = 1;
            cases.push(OwnedOracleCase {
                name: format!("near_complete_n{number_of_nodes}_vertex_{vertex}_label_1"),
                vertex_labels,
                edges: base_edges.clone(),
            });
        }

        for edge_index in 0..base_edges.len() {
            let mut edges = base_edges.clone();
            edges[edge_index].2 = 1;
            cases.push(OwnedOracleCase {
                name: format!("near_complete_n{number_of_nodes}_edge_{edge_index}_label_1"),
                vertex_labels: vec![0_u8; number_of_nodes],
                edges,
            });
        }

        let mut mixed_edges = base_edges.clone();
        mixed_edges[0].2 = 1;
        let midpoint = base_edges.len() / 2;
        mixed_edges[midpoint].2 = 2;
        cases.push(OwnedOracleCase {
            name: format!("near_complete_n{number_of_nodes}_mixed_edge_labels_12"),
            vertex_labels: vec![0_u8; number_of_nodes],
            edges: mixed_edges,
        });
    }

    for &number_of_nodes in &[8_usize, 9] {
        let base_edges = complete_graph_edges(number_of_nodes, 0);
        cases.push(OwnedOracleCase {
            name: format!("complete_uniform_n{number_of_nodes}"),
            vertex_labels: vec![0_u8; number_of_nodes],
            edges: base_edges.clone(),
        });

        let orbit_labels = (0..number_of_nodes).map(|index| (index % 3) as u8).collect::<Vec<_>>();
        cases.push(OwnedOracleCase {
            name: format!("complete_uniform_n{number_of_nodes}_orbit_012"),
            vertex_labels: orbit_labels,
            edges: base_edges.clone(),
        });

        for vertex in 0..number_of_nodes {
            let mut vertex_labels = vec![0_u8; number_of_nodes];
            vertex_labels[vertex] = 1;
            cases.push(OwnedOracleCase {
                name: format!("complete_uniform_n{number_of_nodes}_vertex_{vertex}_label_1"),
                vertex_labels,
                edges: base_edges.clone(),
            });
        }

        for edge_index in 0..base_edges.len() {
            let mut edges = base_edges.clone();
            edges[edge_index].2 = 1;
            cases.push(OwnedOracleCase {
                name: format!("complete_uniform_n{number_of_nodes}_edge_{edge_index}_label_1"),
                vertex_labels: vec![0_u8; number_of_nodes],
                edges,
            });
        }

        let mut mixed_edges = base_edges.clone();
        mixed_edges[0].2 = 1;
        mixed_edges[base_edges.len() / 2].2 = 2;
        cases.push(OwnedOracleCase {
            name: format!("complete_uniform_n{number_of_nodes}_mixed_edge_labels_12"),
            vertex_labels: vec![0_u8; number_of_nodes],
            edges: mixed_edges,
        });
    }

    for &(name, component_sizes) in &[
        ("two_triangles", &[3_usize, 3_usize][..]),
        ("two_squares", &[4_usize, 4_usize][..]),
        ("three_disjoint_edges", &[2_usize, 2, 2][..]),
    ] {
        let number_of_nodes = component_sizes.iter().sum();
        let base_edges = repeated_component_edges(component_sizes, 0);
        cases.push(OwnedOracleCase {
            name: format!("{name}_uniform"),
            vertex_labels: vec![0_u8; number_of_nodes],
            edges: base_edges.clone(),
        });

        let mut component_labels = Vec::with_capacity(number_of_nodes);
        for (component_index, &component_size) in component_sizes.iter().enumerate() {
            component_labels.extend(std::iter::repeat(component_index as u8).take(component_size));
        }
        cases.push(OwnedOracleCase {
            name: format!("{name}_component_labels"),
            vertex_labels: component_labels.clone(),
            edges: base_edges.clone(),
        });

        for vertex in 0..number_of_nodes {
            let mut vertex_labels = vec![0_u8; number_of_nodes];
            vertex_labels[vertex] = 1;
            cases.push(OwnedOracleCase {
                name: format!("{name}_vertex_{vertex}_label_1"),
                vertex_labels,
                edges: base_edges.clone(),
            });
        }

        for edge_index in 0..base_edges.len() {
            let mut edges = base_edges.clone();
            edges[edge_index].2 = 1;
            cases.push(OwnedOracleCase {
                name: format!("{name}_edge_{edge_index}_label_1"),
                vertex_labels: vec![0_u8; number_of_nodes],
                edges,
            });
        }

        let mut mixed_edges = base_edges.clone();
        if !mixed_edges.is_empty() {
            let middle_index = mixed_edges.len() / 2;
            mixed_edges[0].2 = 1;
            mixed_edges[middle_index].2 = 2;
        }
        cases.push(OwnedOracleCase {
            name: format!("{name}_mixed_edge_labels_12"),
            vertex_labels: vec![0_u8; number_of_nodes],
            edges: mixed_edges,
        });
    }

    cases
}

fn compare_rust_and_bliss_stats_owned_case_verbose(case: &OwnedOracleCase) -> Result<(), String> {
    let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
    let matrix = Edges::matrix(graph.edges());
    let rust_result = canonical_label_labeled_simple_graph(
        &graph,
        |node| case.vertex_labels[node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
    );
    let bliss_result = canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges)
        .map_err(|error| format!("{}: bliss oracle run failed: {error}", case.name))?;

    let bliss_nodes = bliss_result
        .stats
        .nodes
        .ok_or_else(|| format!("{}: bliss did not report node count", case.name))?;
    let bliss_leaf_nodes = bliss_result
        .stats
        .leaf_nodes
        .ok_or_else(|| format!("{}: bliss did not report leaf-node count", case.name))?;

    let rust_stats = rust_result.stats;
    let rust_nodes = rust_stats.search_nodes;
    let rust_leaf_nodes = rust_stats.leaf_nodes;
    if rust_nodes != bliss_nodes || rust_leaf_nodes != bliss_leaf_nodes {
        return Err(format!(
            "{}: vertex_labels={:?} edges={:?} rust_stats={rust_stats:?} bliss_stats={:?} rust_nodes/leaves={}/{} bliss_nodes/leaves={}/{}",
            case.name,
            case.vertex_labels,
            case.edges,
            bliss_result.stats,
            rust_nodes,
            rust_leaf_nodes,
            bliss_nodes,
            bliss_leaf_nodes
        ));
    }

    Ok(())
}

#[test]
#[ignore = "requires a local bliss executable"]
fn test_bliss_oracle_is_relabeling_invariant_for_labeled_simple_graphs() {
    let bliss = locate_bliss_binary().expect("missing bliss executable");

    let first = canonicalize_labeled_simple_graph(
        &[10_u8, 10, 20, 20],
        &[(0, 1, 7_u8), (1, 2, 9), (2, 3, 7), (3, 0, 9)],
    )
    .expect("first bliss oracle run should succeed");
    let second = canonicalize_labeled_simple_graph(
        &[10_u8, 20, 10, 20],
        &[(0, 2, 7_u8), (0, 3, 9), (1, 3, 7), (1, 2, 9)],
    )
    .expect("second bliss oracle run should succeed");

    assert!(bliss.is_file(), "expected located bliss binary {} to exist", bliss.display());
    assert_eq!(first.canonical_dimacs, second.canonical_dimacs);
    assert_eq!(first.stats.nodes, second.stats.nodes);
    assert_eq!(first.stats.leaf_nodes, second.stats.leaf_nodes);
    assert_eq!(first.stats.generators, second.stats.generators);
}

#[test]
#[ignore = "requires a local bliss executable"]
fn test_bliss_oracle_returns_expanded_labeling_and_stats() {
    let result = canonicalize_labeled_simple_graph(
        &[1_u8, 1, 2, 2],
        &[(0, 1, 3_u8), (1, 2, 4), (2, 3, 3), (3, 0, 4)],
    )
    .expect("bliss oracle run should succeed");

    assert_eq!(result.expanded_canonical_labeling.len(), 8);
    assert_eq!(result.original_canonical_order.len(), 4);
    assert!(result.original_canonical_order.iter().all(|&vertex| vertex < 4));
    assert_eq!(result.stats.nodes, Some(3));
    assert_eq!(result.stats.leaf_nodes, Some(3));
    assert_eq!(result.stats.bad_nodes, Some(0));
    assert_eq!(result.stats.canrep_updates, Some(1));
    assert_eq!(result.stats.generators, Some(1));
    assert_eq!(result.stats.max_level, Some(1));
    assert_eq!(result.stats.group_size.as_deref(), Some("2"));
    assert!(result.stdout.contains("Canonical labeling:"));
}

#[test]
#[ignore = "requires a local bliss executable"]
fn test_rust_canonizer_matches_bliss_on_easy_labeled_corpus() {
    let cases = [
        NamedOracleCase {
            name: "branched_path_5",
            vertex_labels: &[1_u8, 1, 1, 2, 2],
            edges: &[(0, 1, 3), (1, 2, 3), (1, 3, 4), (1, 4, 4)],
        },
        NamedOracleCase {
            name: "complete_bipartite_4x4",
            vertex_labels: &[0_u8, 0, 0, 0, 1, 1, 1, 1],
            edges: &[
                (0, 4, 1),
                (0, 5, 1),
                (0, 6, 1),
                (0, 7, 1),
                (1, 4, 1),
                (1, 5, 1),
                (1, 6, 1),
                (1, 7, 1),
                (2, 4, 1),
                (2, 5, 1),
                (2, 6, 1),
                (2, 7, 1),
                (3, 4, 1),
                (3, 5, 1),
                (3, 6, 1),
                (3, 7, 1),
            ],
        },
        NamedOracleCase {
            name: "matched_bipartite_8",
            vertex_labels: &[0_u8, 0, 0, 0, 1, 1, 1, 1],
            edges: &[
                (0, 4, 1),
                (0, 5, 1),
                (1, 4, 1),
                (1, 6, 1),
                (2, 5, 1),
                (2, 7, 1),
                (3, 6, 1),
                (3, 7, 1),
            ],
        },
    ];

    for case in cases {
        compare_rust_and_bliss_case(case).unwrap();
    }
}

#[test]
#[ignore = "diagnostic oracle sweep for current bliss divergences"]
fn test_rust_canonizer_reports_current_bliss_divergences_on_hard_corpus() {
    let cases = [
        NamedOracleCase {
            name: "alternating_cycle_4",
            vertex_labels: &[10_u8, 10, 20, 20],
            edges: &[(0, 1, 7), (1, 2, 9), (2, 3, 7), (3, 0, 9)],
        },
        NamedOracleCase {
            name: "matched_bipartite_8",
            vertex_labels: &[0_u8, 0, 0, 0, 1, 1, 1, 1],
            edges: &[
                (0, 4, 1),
                (0, 5, 1),
                (1, 4, 1),
                (1, 6, 1),
                (2, 5, 1),
                (2, 7, 1),
                (3, 6, 1),
                (3, 7, 1),
            ],
        },
        NamedOracleCase {
            name: "decorated_cycle_8",
            vertex_labels: &[5_u8, 5, 6, 6, 1, 1, 1, 1],
            edges: &[
                (0, 1, 1),
                (1, 2, 2),
                (2, 3, 1),
                (3, 0, 2),
                (0, 4, 3),
                (1, 5, 3),
                (2, 6, 3),
                (3, 7, 3),
            ],
        },
        NamedOracleCase {
            name: "crown_8",
            vertex_labels: &[0_u8, 0, 0, 0, 1, 1, 1, 1],
            edges: &[
                (0, 5, 1),
                (0, 6, 1),
                (0, 7, 1),
                (1, 4, 1),
                (1, 6, 1),
                (1, 7, 1),
                (2, 4, 1),
                (2, 5, 1),
                (2, 7, 1),
                (3, 4, 1),
                (3, 5, 1),
                (3, 6, 1),
            ],
        },
    ];

    let mut mismatches = Vec::new();
    for case in cases {
        if let Err(error) = compare_rust_and_bliss_case(case) {
            mismatches.push(error);
        }
    }

    assert_eq!(mismatches, Vec::<String>::new(),);
}

#[test]
#[ignore = "diagnostic oracle sweep over the benchmark corpus; currently expected clean"]
fn test_rust_canonizer_tracks_current_bliss_divergence_on_benchmark_corpus() {
    let mut mismatches = benchmark_cases()
        .into_iter()
        .filter_map(|case| {
            compare_rust_and_bliss_graph(&case.name, &case.graph, &case.vertex_labels, &case.edges)
                .err()
                .map(|_| case.name)
        })
        .collect::<Vec<_>>();
    mismatches.sort();

    assert_eq!(mismatches, Vec::<String>::new());
}

#[test]
#[ignore = "diagnostic oracle sweep over the scaling corpus; currently expected clean"]
fn test_rust_canonizer_tracks_current_bliss_divergence_on_scaling_corpus() {
    let mut mismatches = scaling_cases()
        .into_iter()
        .filter_map(|case| {
            compare_rust_and_bliss_graph(&case.name, &case.graph, &case.vertex_labels, &case.edges)
                .err()
                .map(|_| case.name)
        })
        .collect::<Vec<_>>();
    mismatches.sort();

    assert_eq!(mismatches, Vec::<String>::new());
}

#[test]
#[ignore = "diagnostic oracle sweep over extended symmetric generic families; currently expected clean"]
fn test_rust_canonizer_tracks_current_bliss_divergence_on_extended_symmetric_corpus() {
    let cases = vec![
        uniform_cycle_case("uniform_cycle_8", 8),
        prism_case("prism_6", 6),
        crown_case("crown_10", 5),
        matched_bipartite_shift_case("matched_bipartite_shift_10", 5),
        dimension_colored_cube_q3_case(),
    ];

    let mut mismatches = cases
        .into_iter()
        .filter_map(|case| {
            let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
            compare_rust_and_bliss_graph(&case.name, &graph, &case.vertex_labels, &case.edges)
                .err()
                .map(|_| case.name)
        })
        .collect::<Vec<_>>();
    mismatches.sort();

    assert_eq!(mismatches, Vec::<String>::new());
}

#[test]
#[ignore = "local bliss parity check on the hard-gap corpus"]
fn test_rust_canonizer_eventually_matches_bliss_on_hard_gap_corpus() {
    let cases = [NamedOracleCase {
        name: "matched_bipartite_8",
        vertex_labels: &[0_u8, 0, 0, 0, 1, 1, 1, 1],
        edges: &[
            (0, 4, 1),
            (0, 5, 1),
            (1, 4, 1),
            (1, 6, 1),
            (2, 5, 1),
            (2, 7, 1),
            (3, 6, 1),
            (3, 7, 1),
        ],
    }];

    let mismatches = cases
        .into_iter()
        .filter_map(|case| compare_rust_and_bliss_case(case).err())
        .collect::<Vec<_>>();
    assert!(
        mismatches.is_empty(),
        "strict bliss parity still fails on hard-gap corpus: {mismatches:?}"
    );
}

#[test]
#[ignore = "local bliss parity check on the benchmark corpus"]
fn test_rust_canonizer_eventually_matches_bliss_on_benchmark_gap_corpus() {
    let mismatches = benchmark_cases()
        .into_iter()
        .filter_map(|case| {
            compare_rust_and_bliss_graph(&case.name, &case.graph, &case.vertex_labels, &case.edges)
                .err()
                .map(|_| case.name)
        })
        .collect::<Vec<_>>();

    assert!(
        mismatches.is_empty(),
        "strict bliss parity still fails on benchmark corpus: {mismatches:?}"
    );
}

#[test]
#[ignore = "local bliss parity check on the scaling corpus"]
fn test_rust_canonizer_eventually_matches_bliss_on_scaling_gap_corpus() {
    let mismatches = scaling_cases()
        .into_iter()
        .filter_map(|case| {
            compare_rust_and_bliss_graph(&case.name, &case.graph, &case.vertex_labels, &case.edges)
                .err()
                .map(|_| case.name)
        })
        .collect::<Vec<_>>();

    assert!(
        mismatches.is_empty(),
        "strict bliss parity still fails on scaling corpus: {mismatches:?}"
    );
}

#[test]
#[ignore = "local bliss parity check on the extended symmetric corpus"]
fn test_rust_canonizer_eventually_matches_bliss_on_extended_symmetric_corpus() {
    let cases = vec![
        uniform_cycle_case("uniform_cycle_8", 8),
        prism_case("prism_6", 6),
        crown_case("crown_10", 5),
        matched_bipartite_shift_case("matched_bipartite_shift_10", 5),
        dimension_colored_cube_q3_case(),
    ];

    let mismatches = cases
        .into_iter()
        .filter_map(|case| {
            let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
            compare_rust_and_bliss_graph(&case.name, &graph, &case.vertex_labels, &case.edges)
                .err()
                .map(|_| case.name)
        })
        .collect::<Vec<_>>();

    assert!(
        mismatches.is_empty(),
        "strict bliss parity still fails on extended symmetric corpus: {mismatches:?}"
    );
}

#[test]
#[ignore = "local bliss parity check on discovered random gap cases"]
fn test_rust_canonizer_eventually_matches_bliss_on_discovered_random_gap_corpus() {
    let mismatches = discovered_random_gap_cases()
        .into_iter()
        .filter_map(|case| compare_rust_and_bliss_owned_case(&case).err().map(|_| case.name))
        .collect::<Vec<_>>();

    assert!(
        mismatches.is_empty(),
        "strict bliss parity still fails on discovered random gap cases: {mismatches:?}"
    );
}

#[test]
#[ignore = "local bliss parity check on aggressive discovered random-gap seeds"]
fn test_rust_canonizer_eventually_matches_bliss_on_aggressive_random_gap_seed_corpus() {
    let mismatches = aggressive_random_gap_seeds()
        .into_iter()
        .filter_map(|seed| {
            let case = random_labeled_simple_case(seed);
            compare_rust_and_bliss_owned_case(&case).err().map(|_| case.name)
        })
        .collect::<Vec<_>>();

    assert!(
        mismatches.is_empty(),
        "strict bliss parity still fails on aggressive discovered random-gap seeds: {mismatches:?}"
    );
}

#[test]
#[ignore = "diagnostic oracle sweep for current bliss search-stat divergences on the benchmark corpus"]
fn test_rust_canonizer_tracks_current_bliss_search_stat_divergence_on_benchmark_corpus() {
    let mut mismatches = benchmark_cases()
        .into_iter()
        .filter_map(|case| {
            compare_rust_and_bliss_stats_graph(
                &case.name,
                &case.graph,
                &case.vertex_labels,
                &case.edges,
            )
            .err()
        })
        .collect::<Vec<_>>();
    mismatches.sort();

    assert_eq!(mismatches, Vec::<String>::new());
}

#[test]
#[ignore = "diagnostic oracle sweep for current bliss search-stat divergences on the scaling corpus"]
fn test_rust_canonizer_tracks_current_bliss_search_stat_divergence_on_scaling_corpus() {
    let mut mismatches = scaling_cases()
        .into_iter()
        .filter_map(|case| {
            compare_rust_and_bliss_stats_graph(
                &case.name,
                &case.graph,
                &case.vertex_labels,
                &case.edges,
            )
            .err()
        })
        .collect::<Vec<_>>();
    mismatches.sort();

    assert_eq!(mismatches, Vec::<String>::new());
}

#[test]
#[ignore = "strict local bliss search-stat parity on discovered random gap cases"]
fn test_rust_search_stats_eventually_match_bliss_on_discovered_random_gap_corpus() {
    let mismatches = discovered_random_gap_cases()
        .into_iter()
        .filter_map(|case| compare_rust_and_bliss_stats_owned_case(&case).err())
        .collect::<Vec<_>>();

    assert!(
        mismatches.is_empty(),
        "strict bliss search-stat parity still fails on discovered random gap cases: {mismatches:?}"
    );
}

#[test]
#[ignore = "strict local bliss search-stat parity on the smallest empty symmetric case"]
fn test_rust_search_stats_eventually_match_bliss_on_empty_uniform_n3() {
    let case = OwnedOracleCase {
        name: "empty_uniform_n3".to_owned(),
        vertex_labels: vec![0_u8, 0, 0],
        edges: vec![],
    };

    compare_rust_and_bliss_stats_owned_case(&case).unwrap();
}

#[test]
#[ignore = "strict local bliss search-stat parity on the next uniform edgeless symmetric case"]
fn test_rust_search_stats_eventually_match_bliss_on_empty_uniform_n4() {
    let case = OwnedOracleCase {
        name: "empty_uniform_n4".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0],
        edges: vec![],
    };

    compare_rust_and_bliss_stats_owned_case(&case).unwrap();
}

#[test]
#[ignore = "strict local bliss search-stat parity on a larger uniform edgeless symmetric case"]
fn test_rust_search_stats_eventually_match_bliss_on_empty_uniform_n6() {
    let case = OwnedOracleCase {
        name: "empty_uniform_n6".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0],
        edges: vec![],
    };

    compare_rust_and_bliss_stats_owned_case(&case).unwrap();
}

#[test]
#[ignore = "strict local bliss search-stat parity on a larger edgeless colored case"]
fn test_rust_search_stats_eventually_match_bliss_on_empty_colored_n10() {
    let case = OwnedOracleCase {
        name: "empty_colored_n10".to_owned(),
        vertex_labels: vec![1_u8, 0, 2, 1, 3, 2, 0, 1, 2, 0],
        edges: vec![],
    };

    compare_rust_and_bliss_stats_owned_case(&case).unwrap();
}

#[test]
#[ignore = "strict local bliss search-stat parity on representative random-gap cases"]
fn test_rust_search_stats_eventually_match_bliss_on_representative_random_gap_corpus() {
    let mismatches = representative_random_stat_gap_cases()
        .into_iter()
        .filter_map(|case| compare_rust_and_bliss_stats_owned_case(&case).err())
        .collect::<Vec<_>>();

    assert!(
        mismatches.is_empty(),
        "strict bliss search-stat parity still fails on representative random-gap cases: {mismatches:?}"
    );
}

#[test]
#[ignore = "strict local bliss search-stat parity on heavy sparse/disconnected random-gap cases"]
fn test_rust_search_stats_eventually_match_bliss_on_heavy_sparse_random_gap_corpus() {
    let mismatches = heavy_sparse_random_stat_gap_cases()
        .into_iter()
        .filter_map(|case| compare_rust_and_bliss_stats_owned_case(&case).err())
        .collect::<Vec<_>>();

    assert!(
        mismatches.is_empty(),
        "strict bliss search-stat parity still fails on heavy sparse/disconnected random-gap cases: {mismatches:?}"
    );
}

#[test]
#[ignore = "strict local bliss search-stat parity on heavy dense symmetric random-gap cases"]
fn test_rust_search_stats_eventually_match_bliss_on_heavy_dense_random_gap_corpus() {
    let mismatches = heavy_dense_random_stat_gap_cases()
        .into_iter()
        .filter_map(|case| compare_rust_and_bliss_stats_owned_case(&case).err())
        .collect::<Vec<_>>();

    assert!(
        mismatches.is_empty(),
        "strict bliss search-stat parity still fails on heavy dense symmetric random-gap cases: {mismatches:?}"
    );
}

#[test]
#[ignore = "temporary sweep for strict bliss search-stat gaps in repeated disconnected-component families"]
fn test_probe_repeated_disconnected_component_families_bliss_search_stats() {
    let cases = repeated_component_family_cases();
    let mut mismatches = Vec::new();

    for case in cases {
        if let Err(error) = compare_rust_and_bliss_stats_owned_case(&case) {
            mismatches.push(describe_case_mismatch(&case, &error));
        }
    }

    assert!(
        mismatches.is_empty(),
        "repeated disconnected-component family search-stat mismatches: {mismatches:?}"
    );
}

#[test]
#[ignore = "strict local bliss search-stat parity on a smaller edgeless colored component-recursion case"]
fn test_rust_search_stats_eventually_match_bliss_on_empty_colored_n6_component_case() {
    let case = OwnedOracleCase {
        name: "empty_colored_n6_component_case".to_owned(),
        vertex_labels: vec![2_u8, 1, 1, 0, 2, 1],
        edges: vec![],
    };

    compare_rust_and_bliss_stats_owned_case(&case).unwrap();
}

#[test]
#[ignore = "strict local bliss search-stat parity on a slightly larger edgeless colored case"]
fn test_rust_search_stats_eventually_match_bliss_on_empty_colored_n7() {
    let case = OwnedOracleCase {
        name: "empty_colored_n7".to_owned(),
        vertex_labels: vec![0_u8, 1, 1, 0, 1, 1, 0],
        edges: vec![],
    };

    compare_rust_and_bliss_stats_owned_case(&case).unwrap();
}

#[test]
#[ignore = "strict local bliss search-stat parity on a sparse uniform matching-like case"]
fn test_rust_search_stats_eventually_match_bliss_on_three_disjoint_edges_n10() {
    let case = OwnedOracleCase {
        name: "three_disjoint_edges_n10".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        edges: vec![(0, 3, 0), (1, 4, 0), (2, 5, 0)],
    };

    compare_rust_and_bliss_stats_owned_case(&case).unwrap();
}

#[test]
#[ignore = "strict local bliss search-stat parity on two disjoint uniform stars with isolates"]
fn test_rust_search_stats_eventually_match_bliss_on_two_disjoint_stars_n10() {
    let case = OwnedOracleCase {
        name: "two_disjoint_stars_n10".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        edges: vec![(0, 6, 0), (1, 7, 0), (4, 6, 0), (5, 7, 0), (6, 8, 0), (7, 9, 0)],
    };

    compare_rust_and_bliss_stats_owned_case(&case).unwrap();
}

#[test]
#[ignore = "strict local bliss search-stat parity on a dense uniform 9-vertex case"]
fn test_rust_search_stats_eventually_match_bliss_on_dense_uniform_n9_m27() {
    let case = OwnedOracleCase {
        name: "dense_uniform_n9_m27".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0],
        edges: vec![
            (0, 1, 0),
            (0, 2, 0),
            (0, 3, 0),
            (0, 4, 0),
            (0, 5, 0),
            (0, 6, 0),
            (0, 7, 0),
            (0, 8, 0),
            (1, 2, 0),
            (1, 3, 0),
            (1, 4, 0),
            (1, 5, 0),
            (1, 6, 0),
            (1, 8, 0),
            (2, 3, 0),
            (2, 6, 0),
            (2, 7, 0),
            (2, 8, 0),
            (3, 6, 0),
            (3, 7, 0),
            (3, 8, 0),
            (4, 6, 0),
            (4, 7, 0),
            (5, 6, 0),
            (5, 7, 0),
            (6, 8, 0),
            (7, 8, 0),
        ],
    };

    compare_rust_and_bliss_stats_owned_case(&case).unwrap();
}

#[test]
#[ignore = "strict local bliss search-stat parity on a dense uniform 8-vertex case from the heavy miner"]
fn test_rust_search_stats_eventually_match_bliss_on_dense_uniform_n8_m21() {
    let case = OwnedOracleCase {
        name: "dense_uniform_n8_m21".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0],
        edges: vec![
            (0, 1, 0),
            (0, 4, 0),
            (0, 5, 0),
            (0, 6, 0),
            (0, 7, 0),
            (1, 2, 0),
            (1, 3, 0),
            (1, 4, 0),
            (1, 6, 0),
            (2, 3, 0),
            (2, 4, 0),
            (2, 5, 0),
            (2, 7, 0),
            (3, 4, 0),
            (3, 5, 0),
            (3, 7, 0),
            (4, 5, 0),
            (4, 6, 0),
            (4, 7, 0),
            (5, 6, 0),
            (6, 7, 0),
        ],
    };

    compare_rust_and_bliss_stats_owned_case(&case).unwrap();
}

#[test]
#[ignore = "strict local bliss search-stat parity on a complete labeled 7-vertex case from the heavy miner"]
fn test_rust_search_stats_eventually_match_bliss_on_complete_labeled_n7_m21() {
    let case = OwnedOracleCase {
        name: "complete_labeled_n7_m21".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0],
        edges: vec![
            (0, 1, 0),
            (0, 2, 1),
            (0, 3, 0),
            (0, 4, 1),
            (0, 5, 0),
            (0, 6, 0),
            (1, 2, 0),
            (1, 3, 0),
            (1, 4, 0),
            (1, 5, 1),
            (1, 6, 1),
            (2, 3, 1),
            (2, 4, 0),
            (2, 5, 0),
            (2, 6, 0),
            (3, 4, 1),
            (3, 5, 0),
            (3, 6, 0),
            (4, 5, 0),
            (4, 6, 0),
            (5, 6, 1),
        ],
    };

    compare_rust_and_bliss_stats_owned_case(&case).unwrap();
}

#[test]
#[ignore = "strict local bliss search-stat parity on a near-complete 7-vertex case"]
fn test_rust_search_stats_eventually_match_bliss_on_near_complete_n7() {
    let case = OwnedOracleCase {
        name: "near_complete_n7".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0],
        edges: vec![
            (0, 1, 0),
            (0, 2, 0),
            (0, 3, 0),
            (0, 4, 0),
            (0, 5, 0),
            (0, 6, 0),
            (1, 2, 0),
            (1, 3, 0),
            (1, 4, 0),
            (1, 5, 0),
            (1, 6, 0),
            (2, 4, 0),
            (2, 5, 0),
            (2, 6, 0),
            (3, 4, 0),
            (3, 5, 0),
            (3, 6, 0),
            (4, 6, 0),
            (5, 6, 0),
        ],
    };

    compare_rust_and_bliss_stats_owned_case(&case).unwrap();
}

#[test]
#[ignore = "temporary diagnostic sweep over named symmetric families"]
fn test_rust_search_stats_named_symmetric_families() {
    let mismatches = named_symmetric_family_cases()
        .into_iter()
        .filter_map(|case| family_search_stat_mismatch(&case).err())
        .collect::<Vec<_>>();

    assert!(
        mismatches.is_empty(),
        "strict bliss search-stat parity still fails on named symmetric families:\n{}",
        mismatches.join("\n\n")
    );
}

#[test]
#[ignore = "requires a local bliss executable; broader random search-stat differential sweep"]
fn test_rust_canonizer_tracks_current_bliss_search_stat_divergence_on_parallel_random_labeled_graphs()
 {
    let mismatches = collect_random_window_stat_mismatches(
        random_case_base_seed(),
        random_case_count(),
        random_case_parallelism(),
    );

    assert_eq!(mismatches, Vec::<String>::new());
}

#[test]
#[ignore = "diagnostic oracle sweep over label perturbations on symmetric topologies"]
fn test_rust_canonizer_tracks_current_bliss_search_stat_divergence_on_label_perturbation_sweep() {
    let mut mismatches = label_perturbation_sweep_cases()
        .into_iter()
        .filter_map(|case| compare_rust_and_bliss_stats_owned_case_verbose(&case).err())
        .collect::<Vec<_>>();
    mismatches.sort();

    assert_eq!(mismatches, Vec::<String>::new());
}

#[test]
#[ignore = "requires a local bliss executable; broader random search-stat differential sweep over fixed seed windows"]
fn test_rust_canonizer_matches_bliss_search_stats_on_parallel_aggressive_seed_windows() {
    let windows = [
        (424242424242424242_u64, 1024_usize, 8_usize),
        (135791357913579_u64, 1024_usize, 8_usize),
        (98765432123456789_u64, 256_usize, 8_usize),
    ];

    for (base_seed, case_count, parallelism) in windows {
        let mismatches = collect_random_window_stat_mismatches(base_seed, case_count, parallelism);
        assert!(
            mismatches.is_empty(),
            "strict bliss search-stat parity failed for seed window base_seed={base_seed} case_count={case_count} parallelism={parallelism}: {mismatches:?}"
        );
    }
}

#[test]
#[ignore = "diagnostic probe for bliss option sensitivity on the smallest empty symmetric case"]
fn test_probe_empty_uniform_n3_bliss_option_sensitivity() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "empty_uniform_n3".to_owned(),
        vertex_labels: vec![0_u8, 0, 0],
        edges: vec![],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir()
        .join(format!("geometric-traits-bliss-empty-uniform-n3-probe-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    for &(failure_recording, component_recursion) in
        &[(false, false), (false, true), (true, false), (true, true)]
    {
        let canonical_path = temp_dir.join(format!(
            "out-fr{}-cr{}.dimacs",
            usize::from(failure_recording),
            usize::from(component_recursion)
        ));
        let result = run_bliss_on_dimacs_file_with_options(
            &bliss,
            &input_path,
            &canonical_path,
            encoded.expanded_vertex_count,
            encoded.original_vertex_count,
            failure_recording,
            component_recursion,
        )
        .unwrap();
        eprintln!(
            "fr={} cr={}: nodes={:?} leaves={:?} generators={:?} order={:?}",
            failure_recording,
            component_recursion,
            result.stats.nodes,
            result.stats.leaf_nodes,
            result.stats.generators,
            result.original_canonical_order
        );
    }
}

#[test]
#[ignore = "diagnostic probe for bliss trace on the next uniform edgeless symmetric case"]
fn test_probe_empty_uniform_n4_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "empty_uniform_n4".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0],
        edges: vec![],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir()
        .join(format!("geometric-traits-bliss-empty-uniform-n4-probe-{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir).unwrap();
    let input_path = temp_dir.join("case.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output =
        Command::new(&bliss).env("BLISS_TRACE", "1").arg("-can").arg(&input_path).output().unwrap();

    eprintln!("bliss_stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("bliss_stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success());
}

#[test]
#[ignore = "diagnostic probe for rust trace on the next uniform edgeless symmetric case; run alone"]
fn test_probe_empty_uniform_n4_rust_trace() {
    // SAFETY: test-only probe that should be run in isolation.
    unsafe {
        std::env::set_var("GEOMETRIC_TRAITS_CANON_TRACE", "1");
    }

    let case = OwnedOracleCase {
        name: "empty_uniform_n4".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0],
        edges: vec![],
    };
    let stats = rust_stats_for_owned_case(&case);
    let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
    let matrix = Edges::matrix(graph.edges());
    let result = canonical_label_labeled_simple_graph(
        &graph,
        |node| case.vertex_labels[node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
    );

    eprintln!("rust_stats={stats:?}");
    eprintln!("rust_order={:?}", result.order);

    // SAFETY: test-only probe that should be run in isolation.
    unsafe {
        std::env::remove_var("GEOMETRIC_TRAITS_CANON_TRACE");
    }
}

#[test]
#[ignore = "diagnostic probe for bliss trace on a slightly larger edgeless colored case"]
fn test_probe_empty_colored_n7_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "empty_colored_n7".to_owned(),
        vertex_labels: vec![0_u8, 1, 1, 0, 1, 1, 0],
        edges: vec![],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir()
        .join(format!("geometric-traits-bliss-empty-colored-n7-probe-{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir).unwrap();
    let input_path = temp_dir.join("case.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output =
        Command::new(&bliss).env("BLISS_TRACE", "1").arg("-can").arg(&input_path).output().unwrap();

    eprintln!("bliss_stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("bliss_stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success());
}

#[test]
#[ignore = "diagnostic probe for rust trace on a slightly larger edgeless colored case; run alone"]
fn test_probe_empty_colored_n7_rust_trace() {
    // SAFETY: test-only probe that should be run in isolation.
    unsafe {
        std::env::set_var("GEOMETRIC_TRAITS_CANON_TRACE", "1");
    }

    let case = OwnedOracleCase {
        name: "empty_colored_n7".to_owned(),
        vertex_labels: vec![0_u8, 1, 1, 0, 1, 1, 0],
        edges: vec![],
    };
    let stats = rust_stats_for_owned_case(&case);
    let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
    let matrix = Edges::matrix(graph.edges());
    let result = canonical_label_labeled_simple_graph(
        &graph,
        |node| case.vertex_labels[node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
    );

    eprintln!("rust_stats={stats:?}");
    eprintln!("rust_order={:?}", result.order);

    // SAFETY: test-only probe that should be run in isolation.
    unsafe {
        std::env::remove_var("GEOMETRIC_TRAITS_CANON_TRACE");
    }
}

#[test]
#[ignore = "diagnostic probe for bliss trace on a sparse uniform matching-like case"]
fn test_probe_three_disjoint_edges_n10_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "three_disjoint_edges_n10".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        edges: vec![(0, 3, 0), (1, 4, 0), (2, 5, 0)],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-three-disjoint-edges-n10-probe-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&temp_dir).unwrap();
    let input_path = temp_dir.join("case.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output =
        Command::new(&bliss).env("BLISS_TRACE", "1").arg("-can").arg(&input_path).output().unwrap();

    eprintln!("bliss_stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("bliss_stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success());
}

#[test]
#[ignore = "diagnostic probe for bliss trace on two disjoint uniform stars with isolates"]
fn test_probe_two_disjoint_stars_n10_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "two_disjoint_stars_n10".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        edges: vec![(0, 6, 0), (1, 7, 0), (4, 6, 0), (5, 7, 0), (6, 8, 0), (7, 9, 0)],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-two-disjoint-stars-n10-probe-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&temp_dir).unwrap();
    let input_path = temp_dir.join("case.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output =
        Command::new(&bliss).env("BLISS_TRACE", "1").arg("-can").arg(&input_path).output().unwrap();

    eprintln!("bliss_stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("bliss_stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success());
}

#[test]
#[ignore = "diagnostic probe for rust trace on a sparse uniform matching-like case; run alone"]
fn test_probe_three_disjoint_edges_n10_rust_trace() {
    // SAFETY: test-only probe that should be run in isolation.
    unsafe {
        std::env::set_var("GEOMETRIC_TRAITS_CANON_TRACE", "1");
    }

    let case = OwnedOracleCase {
        name: "three_disjoint_edges_n10".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        edges: vec![(0, 3, 0), (1, 4, 0), (2, 5, 0)],
    };
    let stats = rust_stats_for_owned_case(&case);
    let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
    let matrix = Edges::matrix(graph.edges());
    let result = canonical_label_labeled_simple_graph(
        &graph,
        |node| case.vertex_labels[node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
    );

    eprintln!("rust_stats={stats:?}");
    eprintln!("rust_order={:?}", result.order);

    // SAFETY: test-only probe that should be run in isolation.
    unsafe {
        std::env::remove_var("GEOMETRIC_TRAITS_CANON_TRACE");
    }
}

#[test]
#[ignore = "diagnostic probe for rust trace on two disjoint uniform stars with isolates; run alone"]
fn test_probe_two_disjoint_stars_n10_rust_trace() {
    // SAFETY: test-only probe that should be run in isolation.
    unsafe {
        std::env::set_var("GEOMETRIC_TRAITS_CANON_TRACE", "1");
    }

    let case = OwnedOracleCase {
        name: "two_disjoint_stars_n10".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        edges: vec![(0, 6, 0), (1, 7, 0), (4, 6, 0), (5, 7, 0), (6, 8, 0), (7, 9, 0)],
    };
    let stats = rust_stats_for_owned_case(&case);
    let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
    let matrix = Edges::matrix(graph.edges());
    let result = canonical_label_labeled_simple_graph(
        &graph,
        |node| case.vertex_labels[node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
    );

    eprintln!("rust_stats={stats:?}");
    eprintln!("rust_order={:?}", result.order);

    // SAFETY: test-only probe that should be run in isolation.
    unsafe {
        std::env::remove_var("GEOMETRIC_TRAITS_CANON_TRACE");
    }
}

#[test]
#[ignore = "diagnostic probe for bliss trace on a dense uniform 9-vertex case"]
fn test_probe_dense_uniform_n9_m27_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "dense_uniform_n9_m27".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0],
        edges: vec![
            (0, 1, 0),
            (0, 2, 0),
            (0, 3, 0),
            (0, 4, 0),
            (0, 5, 0),
            (0, 6, 0),
            (0, 7, 0),
            (0, 8, 0),
            (1, 2, 0),
            (1, 3, 0),
            (1, 4, 0),
            (1, 5, 0),
            (1, 6, 0),
            (1, 8, 0),
            (2, 3, 0),
            (2, 6, 0),
            (2, 7, 0),
            (2, 8, 0),
            (3, 6, 0),
            (3, 7, 0),
            (3, 8, 0),
            (4, 6, 0),
            (4, 7, 0),
            (5, 6, 0),
            (5, 7, 0),
            (6, 8, 0),
            (7, 8, 0),
        ],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir()
        .join(format!("geometric-traits-bliss-dense-uniform-n9-m27-probe-{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir).unwrap();
    let input_path = temp_dir.join("case.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output =
        Command::new(&bliss).env("BLISS_TRACE", "1").arg("-can").arg(&input_path).output().unwrap();

    eprintln!("bliss_stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("bliss_stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success());
}

#[test]
#[ignore = "diagnostic probe for bliss trace on the dense uniform 8-vertex heavy-miner case"]
fn test_probe_dense_uniform_n8_m21_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "dense_uniform_n8_m21".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0],
        edges: vec![
            (0, 1, 0),
            (0, 4, 0),
            (0, 5, 0),
            (0, 6, 0),
            (0, 7, 0),
            (1, 2, 0),
            (1, 3, 0),
            (1, 4, 0),
            (1, 6, 0),
            (2, 3, 0),
            (2, 4, 0),
            (2, 5, 0),
            (2, 7, 0),
            (3, 4, 0),
            (3, 5, 0),
            (3, 7, 0),
            (4, 5, 0),
            (4, 6, 0),
            (4, 7, 0),
            (5, 6, 0),
            (6, 7, 0),
        ],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir()
        .join(format!("geometric-traits-bliss-dense-uniform-n8-m21-probe-{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir).unwrap();
    let input_path = temp_dir.join("case.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output =
        Command::new(&bliss).env("BLISS_TRACE", "1").arg("-can").arg(&input_path).output().unwrap();

    eprintln!("bliss_stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("bliss_stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success());
}

#[test]
#[ignore = "diagnostic probe for bliss trace on the complete labeled 7-vertex heavy-miner case"]
fn test_probe_complete_labeled_n7_m21_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "complete_labeled_n7_m21".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0],
        edges: vec![
            (0, 1, 0),
            (0, 2, 1),
            (0, 3, 0),
            (0, 4, 1),
            (0, 5, 0),
            (0, 6, 0),
            (1, 2, 0),
            (1, 3, 0),
            (1, 4, 0),
            (1, 5, 1),
            (1, 6, 1),
            (2, 3, 1),
            (2, 4, 0),
            (2, 5, 0),
            (2, 6, 0),
            (3, 4, 1),
            (3, 5, 0),
            (3, 6, 0),
            (4, 5, 0),
            (4, 6, 0),
            (5, 6, 1),
        ],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-complete-labeled-n7-m21-probe-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&temp_dir).unwrap();
    let input_path = temp_dir.join("case.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output =
        Command::new(&bliss).env("BLISS_TRACE", "1").arg("-can").arg(&input_path).output().unwrap();

    eprintln!("bliss_stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("bliss_stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success());
}

#[test]
#[ignore = "diagnostic probe for rust trace on the dense uniform 8-vertex heavy-miner case; run alone"]
fn test_probe_dense_uniform_n8_m21_rust_trace() {
    // SAFETY: test-only probe that should be run in isolation.
    unsafe {
        std::env::set_var("GEOMETRIC_TRAITS_CANON_TRACE", "1");
    }

    let case = OwnedOracleCase {
        name: "dense_uniform_n8_m21".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0],
        edges: vec![
            (0, 1, 0),
            (0, 4, 0),
            (0, 5, 0),
            (0, 6, 0),
            (0, 7, 0),
            (1, 2, 0),
            (1, 3, 0),
            (1, 4, 0),
            (1, 6, 0),
            (2, 3, 0),
            (2, 4, 0),
            (2, 5, 0),
            (2, 7, 0),
            (3, 4, 0),
            (3, 5, 0),
            (3, 7, 0),
            (4, 5, 0),
            (4, 6, 0),
            (4, 7, 0),
            (5, 6, 0),
            (6, 7, 0),
        ],
    };
    let stats = rust_stats_for_owned_case(&case);
    let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
    let matrix = Edges::matrix(graph.edges());
    let result = canonical_label_labeled_simple_graph(
        &graph,
        |node| case.vertex_labels[node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
    );

    eprintln!("rust_stats={stats:?}");
    eprintln!("rust_order={:?}", result.order);

    // SAFETY: test-only probe that should be run in isolation.
    unsafe {
        std::env::remove_var("GEOMETRIC_TRAITS_CANON_TRACE");
    }
}

#[test]
#[ignore = "diagnostic probe for rust trace on the complete labeled 7-vertex heavy-miner case; run alone"]
fn test_probe_complete_labeled_n7_m21_rust_trace() {
    // SAFETY: test-only probe that should be run in isolation.
    unsafe {
        std::env::set_var("GEOMETRIC_TRAITS_CANON_TRACE", "1");
    }

    let case = OwnedOracleCase {
        name: "complete_labeled_n7_m21".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0],
        edges: vec![
            (0, 1, 0),
            (0, 2, 1),
            (0, 3, 0),
            (0, 4, 1),
            (0, 5, 0),
            (0, 6, 0),
            (1, 2, 0),
            (1, 3, 0),
            (1, 4, 0),
            (1, 5, 1),
            (1, 6, 1),
            (2, 3, 1),
            (2, 4, 0),
            (2, 5, 0),
            (2, 6, 0),
            (3, 4, 1),
            (3, 5, 0),
            (3, 6, 0),
            (4, 5, 0),
            (4, 6, 0),
            (5, 6, 1),
        ],
    };
    let stats = rust_stats_for_owned_case(&case);
    let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
    let matrix = Edges::matrix(graph.edges());
    let result = canonical_label_labeled_simple_graph(
        &graph,
        |node| case.vertex_labels[node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
    );

    eprintln!("rust_stats={stats:?}");
    eprintln!("rust_order={:?}", result.order);

    // SAFETY: test-only probe that should be run in isolation.
    unsafe {
        std::env::remove_var("GEOMETRIC_TRAITS_CANON_TRACE");
    }
}

#[test]
#[ignore = "diagnostic probe for bliss option sensitivity on a dense uniform 9-vertex case"]
fn test_probe_dense_uniform_n9_m27_bliss_option_sensitivity() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "dense_uniform_n9_m27".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0],
        edges: vec![
            (0, 1, 0),
            (0, 2, 0),
            (0, 3, 0),
            (0, 4, 0),
            (0, 5, 0),
            (0, 6, 0),
            (0, 7, 0),
            (0, 8, 0),
            (1, 2, 0),
            (1, 3, 0),
            (1, 4, 0),
            (1, 5, 0),
            (1, 6, 0),
            (1, 8, 0),
            (2, 3, 0),
            (2, 6, 0),
            (2, 7, 0),
            (2, 8, 0),
            (3, 6, 0),
            (3, 7, 0),
            (3, 8, 0),
            (4, 6, 0),
            (4, 7, 0),
            (5, 6, 0),
            (5, 7, 0),
            (6, 8, 0),
            (7, 8, 0),
        ],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-dense-uniform-n9-m27-option-probe-{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    for &(failure_recording, component_recursion) in
        &[(false, false), (false, true), (true, false), (true, true)]
    {
        let canonical_path = temp_dir.join(format!(
            "out-fr{}-cr{}.dimacs",
            usize::from(failure_recording),
            usize::from(component_recursion)
        ));
        let result = run_bliss_on_dimacs_file_with_options(
            &bliss,
            &input_path,
            &canonical_path,
            encoded.expanded_vertex_count,
            encoded.original_vertex_count,
            failure_recording,
            component_recursion,
        )
        .unwrap();
        eprintln!(
            "fr={} cr={}: nodes={:?} leaves={:?} generators={:?} order={:?}",
            failure_recording,
            component_recursion,
            result.stats.nodes,
            result.stats.leaf_nodes,
            result.stats.generators,
            result.original_canonical_order
        );
    }
}

#[test]
#[ignore = "diagnostic probe for rust stats across heuristics on a dense uniform 9-vertex case"]
fn test_probe_dense_uniform_n9_m27_across_heuristics() {
    let case = OwnedOracleCase {
        name: "dense_uniform_n9_m27".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0],
        edges: vec![
            (0, 1, 0),
            (0, 2, 0),
            (0, 3, 0),
            (0, 4, 0),
            (0, 5, 0),
            (0, 6, 0),
            (0, 7, 0),
            (0, 8, 0),
            (1, 2, 0),
            (1, 3, 0),
            (1, 4, 0),
            (1, 5, 0),
            (1, 6, 0),
            (1, 8, 0),
            (2, 3, 0),
            (2, 6, 0),
            (2, 7, 0),
            (2, 8, 0),
            (3, 6, 0),
            (3, 7, 0),
            (3, 8, 0),
            (4, 6, 0),
            (4, 7, 0),
            (5, 6, 0),
            (5, 7, 0),
            (6, 8, 0),
            (7, 8, 0),
        ],
    };
    let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
    let bliss = canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap();
    eprintln!("bliss_stats={:?}", bliss.stats);
    for heuristic in [
        CanonSplittingHeuristic::First,
        CanonSplittingHeuristic::FirstSmallest,
        CanonSplittingHeuristic::FirstLargest,
        CanonSplittingHeuristic::FirstMaxNeighbours,
        CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
        CanonSplittingHeuristic::FirstLargestMaxNeighbours,
    ] {
        let matrix = Edges::matrix(graph.edges());
        let stats = canonical_label_labeled_simple_graph_with_options(
            &graph,
            |node| case.vertex_labels[node],
            |left, right| matrix.sparse_value_at(left, right).unwrap(),
            CanonicalLabelingOptions { splitting_heuristic: heuristic },
        )
        .stats;
        eprintln!("{heuristic:?}: {stats:?}");
    }
}

#[test]
#[ignore = "diagnostic probe for rust trace on a dense uniform 9-vertex case; run alone"]
fn test_probe_dense_uniform_n9_m27_rust_trace() {
    // SAFETY: test-only probe that should be run in isolation.
    unsafe {
        std::env::set_var("GEOMETRIC_TRAITS_CANON_TRACE", "1");
    }

    let case = OwnedOracleCase {
        name: "dense_uniform_n9_m27".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0],
        edges: vec![
            (0, 1, 0),
            (0, 2, 0),
            (0, 3, 0),
            (0, 4, 0),
            (0, 5, 0),
            (0, 6, 0),
            (0, 7, 0),
            (0, 8, 0),
            (1, 2, 0),
            (1, 3, 0),
            (1, 4, 0),
            (1, 5, 0),
            (1, 6, 0),
            (1, 8, 0),
            (2, 3, 0),
            (2, 6, 0),
            (2, 7, 0),
            (2, 8, 0),
            (3, 6, 0),
            (3, 7, 0),
            (3, 8, 0),
            (4, 6, 0),
            (4, 7, 0),
            (5, 6, 0),
            (5, 7, 0),
            (6, 8, 0),
            (7, 8, 0),
        ],
    };
    let stats = rust_stats_for_owned_case(&case);
    let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
    let matrix = Edges::matrix(graph.edges());
    let result = canonical_label_labeled_simple_graph(
        &graph,
        |node| case.vertex_labels[node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
    );

    eprintln!("rust_stats={stats:?}");
    eprintln!("rust_order={:?}", result.order);

    // SAFETY: test-only probe that should be run in isolation.
    unsafe {
        std::env::remove_var("GEOMETRIC_TRAITS_CANON_TRACE");
    }
}

#[test]
#[ignore = "diagnostic probe for bliss trace on a near-complete 7-vertex case"]
fn test_probe_near_complete_n7_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "near_complete_n7".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0],
        edges: vec![
            (0, 1, 0),
            (0, 2, 0),
            (0, 3, 0),
            (0, 4, 0),
            (0, 5, 0),
            (0, 6, 0),
            (1, 2, 0),
            (1, 3, 0),
            (1, 4, 0),
            (1, 5, 0),
            (1, 6, 0),
            (2, 4, 0),
            (2, 5, 0),
            (2, 6, 0),
            (3, 4, 0),
            (3, 5, 0),
            (3, 6, 0),
            (4, 6, 0),
            (5, 6, 0),
        ],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir()
        .join(format!("geometric-traits-bliss-near-complete-n7-probe-{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir).unwrap();
    let input_path = temp_dir.join("case.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output =
        Command::new(&bliss).env("BLISS_TRACE", "1").arg("-can").arg(&input_path).output().unwrap();

    eprintln!("bliss_stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("bliss_stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success());
}

#[test]
#[ignore = "diagnostic probe for rust trace on a near-complete 7-vertex case; run alone"]
fn test_probe_near_complete_n7_rust_trace() {
    // SAFETY: test-only probe that should be run in isolation.
    unsafe {
        std::env::set_var("GEOMETRIC_TRAITS_CANON_TRACE", "1");
    }

    let case = OwnedOracleCase {
        name: "near_complete_n7".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0],
        edges: vec![
            (0, 1, 0),
            (0, 2, 0),
            (0, 3, 0),
            (0, 4, 0),
            (0, 5, 0),
            (0, 6, 0),
            (1, 2, 0),
            (1, 3, 0),
            (1, 4, 0),
            (1, 5, 0),
            (1, 6, 0),
            (2, 4, 0),
            (2, 5, 0),
            (2, 6, 0),
            (3, 4, 0),
            (3, 5, 0),
            (3, 6, 0),
            (4, 6, 0),
            (5, 6, 0),
        ],
    };
    let stats = rust_stats_for_owned_case(&case);
    let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
    let matrix = Edges::matrix(graph.edges());
    let result = canonical_label_labeled_simple_graph(
        &graph,
        |node| case.vertex_labels[node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
    );

    eprintln!("rust_stats={stats:?}");
    eprintln!("rust_order={:?}", result.order);

    // SAFETY: test-only probe that should be run in isolation.
    unsafe {
        std::env::remove_var("GEOMETRIC_TRAITS_CANON_TRACE");
    }
}

#[test]
#[ignore = "diagnostic probe for rust trace on complete_bipartite_4x4; run alone"]
fn test_probe_complete_bipartite_4x4_rust_trace() {
    // SAFETY: test-only probe that should be run in isolation.
    unsafe {
        std::env::set_var("GEOMETRIC_TRAITS_CANON_TRACE", "1");
    }

    let case = OwnedOracleCase {
        name: "complete_bipartite_4x4".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 1, 1, 1, 1],
        edges: vec![
            (0, 4, 1),
            (0, 5, 1),
            (0, 6, 1),
            (0, 7, 1),
            (1, 4, 1),
            (1, 5, 1),
            (1, 6, 1),
            (1, 7, 1),
            (2, 4, 1),
            (2, 5, 1),
            (2, 6, 1),
            (2, 7, 1),
            (3, 4, 1),
            (3, 5, 1),
            (3, 6, 1),
            (3, 7, 1),
        ],
    };
    let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
    let matrix = Edges::matrix(graph.edges());
    let result = canonical_label_labeled_simple_graph(
        &graph,
        |node| case.vertex_labels[node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
    );

    eprintln!("rust_stats={:?}", result.stats);
    eprintln!("rust_order={:?}", result.order);

    // SAFETY: test-only probe that should be run in isolation.
    unsafe {
        std::env::remove_var("GEOMETRIC_TRAITS_CANON_TRACE");
    }
}

#[test]
#[ignore = "diagnostic probe for bliss option sensitivity on a larger edgeless colored case"]
fn test_probe_empty_colored_n10_bliss_option_sensitivity() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "empty_colored_n10".to_owned(),
        vertex_labels: vec![1_u8, 0, 2, 1, 3, 2, 0, 1, 2, 0],
        edges: vec![],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir()
        .join(format!("geometric-traits-bliss-empty-colored-n10-probe-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    for &(failure_recording, component_recursion) in
        &[(false, false), (false, true), (true, false), (true, true)]
    {
        let canonical_path = temp_dir.join(format!(
            "out-fr{}-cr{}.dimacs",
            usize::from(failure_recording),
            usize::from(component_recursion)
        ));
        let result = run_bliss_on_dimacs_file_with_options(
            &bliss,
            &input_path,
            &canonical_path,
            encoded.expanded_vertex_count,
            encoded.original_vertex_count,
            failure_recording,
            component_recursion,
        )
        .unwrap();
        eprintln!(
            "fr={} cr={}: nodes={:?} leaves={:?} generators={:?} order={:?}",
            failure_recording,
            component_recursion,
            result.stats.nodes,
            result.stats.leaf_nodes,
            result.stats.generators,
            result.original_canonical_order
        );
    }
}

#[test]
#[ignore = "diagnostic probe for bliss option sensitivity on the smaller edgeless colored component-recursion case"]
fn test_probe_empty_colored_n6_component_case_bliss_option_sensitivity() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "empty_colored_n6_component_case".to_owned(),
        vertex_labels: vec![2_u8, 1, 1, 0, 2, 1],
        edges: vec![],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-empty-colored-n6-component-probe-{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    for &(failure_recording, component_recursion) in
        &[(false, false), (false, true), (true, false), (true, true)]
    {
        let canonical_path = temp_dir.join(format!(
            "out-fr{}-cr{}.dimacs",
            usize::from(failure_recording),
            usize::from(component_recursion)
        ));
        let result = run_bliss_on_dimacs_file_with_options(
            &bliss,
            &input_path,
            &canonical_path,
            encoded.expanded_vertex_count,
            encoded.original_vertex_count,
            failure_recording,
            component_recursion,
        )
        .unwrap();
        eprintln!(
            "fr={} cr={}: nodes={:?} leaves={:?} generators={:?} order={:?}",
            failure_recording,
            component_recursion,
            result.stats.nodes,
            result.stats.leaf_nodes,
            result.stats.generators,
            result.original_canonical_order
        );
    }
}

#[test]
#[ignore = "diagnostic probe for the bliss trace on the smaller edgeless colored component-recursion case"]
fn test_probe_empty_colored_n6_component_case_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "empty_colored_n6_component_case".to_owned(),
        vertex_labels: vec![2_u8, 1, 1, 0, 2, 1],
        edges: vec![],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir()
        .join(format!("geometric-traits-bliss-empty-colored-n6-trace-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output = Command::new(&bliss)
        .env("BLISS_TRACE", "1")
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg("-fr=y")
        .arg("-cr=y")
        .arg(format!("-ocan={}", canonical_path.display()))
        .arg(&input_path)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "bliss trace probe failed: status={} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    eprintln!("bliss_stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("bliss_stderr:\n{}", String::from_utf8_lossy(&output.stderr));
}

#[test]
#[ignore = "diagnostic probe for the bliss trace on the larger edgeless colored component-recursion case"]
fn test_probe_empty_colored_n10_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "empty_colored_n10".to_owned(),
        vertex_labels: vec![1_u8, 0, 2, 1, 3, 2, 0, 1, 2, 0],
        edges: vec![],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir()
        .join(format!("geometric-traits-bliss-empty-colored-n10-trace-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output = Command::new(&bliss)
        .env("BLISS_TRACE", "1")
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg("-fr=y")
        .arg("-cr=y")
        .arg(format!("-ocan={}", canonical_path.display()))
        .arg(&input_path)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "bliss trace probe failed: status={} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    eprintln!("bliss_stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("bliss_stderr:\n{}", String::from_utf8_lossy(&output.stderr));
}

#[test]
#[ignore = "diagnostic probe for the rust trace on the larger edgeless colored component-recursion case; run this one alone"]
fn test_probe_empty_colored_n10_rust_trace() {
    let case = OwnedOracleCase {
        name: "empty_colored_n10".to_owned(),
        vertex_labels: vec![1_u8, 0, 2, 1, 3, 2, 0, 1, 2, 0],
        edges: vec![],
    };
    let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
    let matrix = Edges::matrix(graph.edges());
    let previous = std::env::var_os("GEOMETRIC_TRAITS_CANON_TRACE");
    unsafe {
        std::env::set_var("GEOMETRIC_TRAITS_CANON_TRACE", "1");
    }
    let result = canonical_label_labeled_simple_graph(
        &graph,
        |node| case.vertex_labels[node],
        |left, right| matrix.sparse_value_at(left, right).unwrap(),
    );
    match previous {
        Some(value) => unsafe {
            std::env::set_var("GEOMETRIC_TRAITS_CANON_TRACE", value);
        },
        None => unsafe {
            std::env::remove_var("GEOMETRIC_TRAITS_CANON_TRACE");
        },
    }
    eprintln!("rust_stats={:?}", result.stats);
    eprintln!("rust_order={:?}", result.order);
}

#[test]
#[ignore = "diagnostic probe for the smallest remaining hard case"]
fn test_probe_matched_bipartite_8_across_heuristics() {
    let case = NamedOracleCase {
        name: "matched_bipartite_8",
        vertex_labels: &[0_u8, 0, 0, 0, 1, 1, 1, 1],
        edges: &[
            (0, 4, 1),
            (0, 5, 1),
            (1, 4, 1),
            (1, 6, 1),
            (2, 5, 1),
            (2, 7, 1),
            (3, 6, 1),
            (3, 7, 1),
        ],
    };
    let heuristics = [
        CanonSplittingHeuristic::First,
        CanonSplittingHeuristic::FirstSmallest,
        CanonSplittingHeuristic::FirstLargest,
        CanonSplittingHeuristic::FirstMaxNeighbours,
        CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
        CanonSplittingHeuristic::FirstLargestMaxNeighbours,
    ];
    let bliss = canonicalize_labeled_simple_graph(case.vertex_labels, case.edges).unwrap();
    let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), case.edges);
    let bliss_certificate =
        certificate_from_order(&graph, case.vertex_labels, &bliss.original_canonical_order);

    for heuristic in heuristics {
        let rust_certificate = rust_certificate_for_case_with_heuristic(case, heuristic);
        eprintln!("{heuristic:?}: rust={rust_certificate:?} bliss={bliss_certificate:?}");
    }
}

#[test]
#[ignore = "diagnostic probe for bliss option sensitivity on the smallest hard case"]
fn test_probe_matched_bipartite_8_bliss_option_sensitivity() {
    let case = NamedOracleCase {
        name: "matched_bipartite_8",
        vertex_labels: &[0_u8, 0, 0, 0, 1, 1, 1, 1],
        edges: &[
            (0, 4, 1),
            (0, 5, 1),
            (1, 4, 1),
            (1, 6, 1),
            (2, 5, 1),
            (2, 7, 1),
            (3, 6, 1),
            (3, 7, 1),
        ],
    };
    let bliss = locate_bliss_binary().unwrap();
    let encoded = encode_labeled_simple_graph_as_dimacs(case.vertex_labels, case.edges).unwrap();
    let temp_dir =
        std::env::temp_dir().join(format!("geometric-traits-bliss-probe-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    for &(failure_recording, component_recursion) in
        &[(false, false), (false, true), (true, false), (true, true)]
    {
        let canonical_path = temp_dir.join(format!(
            "out-fr{}-cr{}.dimacs",
            usize::from(failure_recording),
            usize::from(component_recursion)
        ));
        let result = run_bliss_on_dimacs_file_with_options(
            &bliss,
            &input_path,
            &canonical_path,
            encoded.expanded_vertex_count,
            encoded.original_vertex_count,
            failure_recording,
            component_recursion,
        )
        .unwrap();
        let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), case.edges);
        let certificate =
            certificate_from_order(&graph, case.vertex_labels, &result.original_canonical_order);
        eprintln!(
            "fr={} cr={}: order={:?} cert={:?}",
            failure_recording, component_recursion, result.original_canonical_order, certificate
        );
    }
}

#[test]
#[ignore = "diagnostic probe for the expanded bliss labeling on the hard case"]
fn test_probe_matched_bipartite_8_expanded_bliss_labeling() {
    let case = NamedOracleCase {
        name: "matched_bipartite_8",
        vertex_labels: &[0_u8, 0, 0, 0, 1, 1, 1, 1],
        edges: &[
            (0, 4, 1),
            (0, 5, 1),
            (1, 4, 1),
            (1, 6, 1),
            (2, 5, 1),
            (2, 7, 1),
            (3, 6, 1),
            (3, 7, 1),
        ],
    };
    let result = canonicalize_labeled_simple_graph(case.vertex_labels, case.edges).unwrap();
    eprintln!(
        "expanded labeling={:?} original order={:?}",
        result.expanded_canonical_labeling, result.original_canonical_order
    );
}

#[test]
#[ignore = "diagnostic probe for the remaining uniform-cycle oracle gap"]
fn test_probe_uniform_cycle_8_across_heuristics() {
    let case = uniform_cycle_case("uniform_cycle_8", 8);
    let heuristics = [
        CanonSplittingHeuristic::First,
        CanonSplittingHeuristic::FirstSmallest,
        CanonSplittingHeuristic::FirstLargest,
        CanonSplittingHeuristic::FirstMaxNeighbours,
        CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
        CanonSplittingHeuristic::FirstLargestMaxNeighbours,
    ];
    let bliss = canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap();
    let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
    let bliss_certificate =
        certificate_from_order(&graph, &case.vertex_labels, &bliss.original_canonical_order);

    for heuristic in heuristics {
        let rust_result = canonical_label_labeled_simple_graph_with_options(
            &graph,
            |node| case.vertex_labels[node],
            |left, right| Edges::matrix(graph.edges()).sparse_value_at(left, right).unwrap(),
            CanonicalLabelingOptions { splitting_heuristic: heuristic },
        );
        let rust_certificate =
            certificate_from_order(&graph, &case.vertex_labels, &rust_result.order);
        eprintln!(
            "{heuristic:?}: rust={rust_certificate:?} order={:?} bliss={bliss_certificate:?} bliss_order={:?}",
            rust_result.order, bliss.original_canonical_order
        );
    }
}

#[test]
#[ignore = "diagnostic probe for bliss option sensitivity on discovered random gap cases"]
fn test_probe_discovered_random_gap_option_sensitivity() {
    let bliss = locate_bliss_binary().unwrap();
    for case in discovered_random_gap_cases() {
        let encoded =
            encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
        let temp_dir = std::env::temp_dir().join(format!(
            "geometric-traits-bliss-random-gap-probe-{}-{}",
            std::process::id(),
            case.name
        ));
        let _ = std::fs::create_dir_all(&temp_dir);
        let input_path = temp_dir.join("input.dimacs");
        std::fs::write(&input_path, &encoded.dimacs).unwrap();

        eprintln!("case {}", case.name);
        for &(failure_recording, component_recursion) in
            &[(false, false), (false, true), (true, false), (true, true)]
        {
            let canonical_path = temp_dir.join(format!(
                "out-fr{}-cr{}.dimacs",
                usize::from(failure_recording),
                usize::from(component_recursion)
            ));
            let result = run_bliss_on_dimacs_file_with_options(
                &bliss,
                &input_path,
                &canonical_path,
                encoded.expanded_vertex_count,
                encoded.original_vertex_count,
                failure_recording,
                component_recursion,
            )
            .unwrap();
            let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
            let certificate = certificate_from_order(
                &graph,
                &case.vertex_labels,
                &result.original_canonical_order,
            );
            eprintln!(
                "  fr={} cr={}: order={:?} cert={:?}",
                failure_recording,
                component_recursion,
                result.original_canonical_order,
                certificate
            );
        }
    }
}

#[test]
#[ignore = "diagnostic probe for heuristic sensitivity on discovered random gap cases"]
fn test_probe_discovered_random_gap_across_heuristics() {
    let heuristics = [
        CanonSplittingHeuristic::First,
        CanonSplittingHeuristic::FirstSmallest,
        CanonSplittingHeuristic::FirstLargest,
        CanonSplittingHeuristic::FirstMaxNeighbours,
        CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
        CanonSplittingHeuristic::FirstLargestMaxNeighbours,
    ];

    for case in discovered_random_gap_cases() {
        let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
        let matrix = Edges::matrix(graph.edges());
        let bliss = canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap();
        let bliss_certificate =
            certificate_from_order(&graph, &case.vertex_labels, &bliss.original_canonical_order);
        let bliss_expanded_order = expanded_order_from_labeling(&bliss.expanded_canonical_labeling);

        eprintln!("case {}", case.name);
        eprintln!(
            "  bliss: order={:?} expanded_order={:?} cert={:?}",
            bliss.original_canonical_order, bliss_expanded_order, bliss_certificate
        );

        for heuristic in heuristics {
            let result = canonical_label_labeled_simple_graph_with_options(
                &graph,
                |node| case.vertex_labels[node],
                |left, right| matrix.sparse_value_at(left, right).unwrap(),
                CanonicalLabelingOptions { splitting_heuristic: heuristic },
            );
            let certificate = certificate_from_order(&graph, &case.vertex_labels, &result.order);
            eprintln!(
                "  {heuristic:?}: order={:?} cert={:?} stats={:?}",
                result.order, certificate, result.stats
            );
        }
    }
}

#[test]
#[ignore = "diagnostic probe for heuristic sensitivity on aggressive discovered random-gap seeds"]
fn test_probe_aggressive_random_gap_seeds_across_heuristics() {
    let heuristics = [
        CanonSplittingHeuristic::First,
        CanonSplittingHeuristic::FirstSmallest,
        CanonSplittingHeuristic::FirstLargest,
        CanonSplittingHeuristic::FirstMaxNeighbours,
        CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
        CanonSplittingHeuristic::FirstLargestMaxNeighbours,
    ];

    for seed in aggressive_random_gap_seeds() {
        let case = random_labeled_simple_case(seed);
        let graph = build_bidirectional_labeled_graph(case.vertex_labels.len(), &case.edges);
        let matrix = Edges::matrix(graph.edges());
        let bliss = canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap();
        let bliss_certificate =
            certificate_from_order(&graph, &case.vertex_labels, &bliss.original_canonical_order);
        let bliss_expanded_order = expanded_order_from_labeling(&bliss.expanded_canonical_labeling);

        eprintln!("seed {} case {}", seed, case.name);
        eprintln!(
            "  bliss: order={:?} expanded_order={:?} cert={:?}",
            bliss.original_canonical_order, bliss_expanded_order, bliss_certificate
        );

        for heuristic in heuristics {
            let result = canonical_label_labeled_simple_graph_with_options(
                &graph,
                |node| case.vertex_labels[node],
                |left, right| matrix.sparse_value_at(left, right).unwrap(),
                CanonicalLabelingOptions { splitting_heuristic: heuristic },
            );
            let certificate = certificate_from_order(&graph, &case.vertex_labels, &result.order);
            eprintln!(
                "  {heuristic:?}: order={:?} cert={:?} stats={:?}",
                result.order, certificate, result.stats
            );
        }
    }
}

#[test]
#[ignore = "diagnostic probe for bliss trace on a discovered random gap case"]
fn test_probe_discovered_random_gap_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = discovered_random_gap_cases().into_iter().next().unwrap();
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-trace-probe-{}-{}",
        std::process::id(),
        case.name
    ));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output = Command::new(bliss)
        .env("BLISS_TRACE", "1")
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg("-fr=y")
        .arg("-cr=y")
        .arg(format!("-ocan={}", canonical_path.display()))
        .arg(&input_path)
        .output()
        .unwrap();

    eprintln!("stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "bliss trace run failed");
}

#[test]
#[ignore = "diagnostic probe for bliss trace on the latest discovered random gap case"]
fn test_probe_latest_discovered_random_gap_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = discovered_random_gap_cases().into_iter().last().unwrap();
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-trace-probe-latest-{}-{}",
        std::process::id(),
        case.name
    ));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output = Command::new(bliss)
        .env("BLISS_TRACE", "1")
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg("-fr=y")
        .arg("-cr=y")
        .arg(format!("-ocan={}", canonical_path.display()))
        .arg(&input_path)
        .output()
        .unwrap();

    eprintln!("stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "bliss trace run failed");
}

#[test]
#[ignore = "diagnostic probe for bliss trace on the first aggressive random gap seed"]
fn test_probe_first_aggressive_random_gap_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = random_labeled_simple_case(aggressive_random_gap_seeds()[0]);
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-trace-probe-aggressive-{}-{}",
        std::process::id(),
        case.name
    ));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output = Command::new(bliss)
        .env("BLISS_TRACE", "1")
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg("-fr=y")
        .arg("-cr=y")
        .arg(format!("-ocan={}", canonical_path.display()))
        .arg(&input_path)
        .output()
        .unwrap();

    eprintln!("stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "bliss trace run failed");
}

#[test]
#[ignore = "diagnostic probe for bliss trace on the branched_path_5 search-stat gap"]
fn test_probe_branched_path_5_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "branched_path_5".to_owned(),
        vertex_labels: vec![1_u8, 1, 1, 2, 2],
        edges: vec![(0, 1, 3), (1, 2, 3), (1, 3, 4), (1, 4, 4)],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-trace-probe-branched-path-{}-{}",
        std::process::id(),
        case.name
    ));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output = Command::new(bliss)
        .env("BLISS_TRACE", "1")
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg("-fr=y")
        .arg("-cr=y")
        .arg(format!("-ocan={}", canonical_path.display()))
        .arg(&input_path)
        .output()
        .unwrap();

    eprintln!("stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "bliss trace run failed");
}

#[test]
#[ignore = "diagnostic probe for bliss trace on the branched_path_5 search-stat gap without component recursion"]
fn test_probe_branched_path_5_bliss_trace_without_component_recursion() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "branched_path_5".to_owned(),
        vertex_labels: vec![1_u8, 1, 1, 2, 2],
        edges: vec![(0, 1, 3), (1, 2, 3), (1, 3, 4), (1, 4, 4)],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-trace-probe-branched-path-no-cr-{}-{}",
        std::process::id(),
        case.name
    ));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output = Command::new(bliss)
        .env("BLISS_TRACE", "1")
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg("-fr=n")
        .arg("-cr=n")
        .arg(format!("-ocan={}", canonical_path.display()))
        .arg(&input_path)
        .output()
        .unwrap();

    eprintln!("stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "bliss trace run failed");
}

#[test]
#[ignore = "diagnostic probe for bliss trace on the complete_bipartite_4x4 search-stat gap"]
fn test_probe_complete_bipartite_4x4_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "complete_bipartite_4x4".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 1, 1, 1, 1],
        edges: vec![
            (0, 4, 1),
            (0, 5, 1),
            (0, 6, 1),
            (0, 7, 1),
            (1, 4, 1),
            (1, 5, 1),
            (1, 6, 1),
            (1, 7, 1),
            (2, 4, 1),
            (2, 5, 1),
            (2, 6, 1),
            (2, 7, 1),
            (3, 4, 1),
            (3, 5, 1),
            (3, 6, 1),
            (3, 7, 1),
        ],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-trace-probe-k44-{}-{}",
        std::process::id(),
        case.name
    ));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output = Command::new(bliss)
        .env("BLISS_TRACE", "1")
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg("-fr=y")
        .arg("-cr=y")
        .arg(format!("-ocan={}", canonical_path.display()))
        .arg(&input_path)
        .output()
        .unwrap();

    eprintln!("stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "bliss trace run failed");
}

#[test]
#[ignore = "diagnostic probe for search-stat divergence against bliss on representative cases"]
fn test_probe_representative_stats_against_bliss() {
    let cases = vec![
        OwnedOracleCase {
            name: "branched_path_5".to_owned(),
            vertex_labels: vec![1_u8, 1, 1, 2, 2],
            edges: vec![(0, 1, 3), (1, 2, 3), (1, 3, 4), (1, 4, 4)],
        },
        OwnedOracleCase {
            name: "complete_bipartite_4x4".to_owned(),
            vertex_labels: vec![0_u8, 0, 0, 0, 1, 1, 1, 1],
            edges: vec![
                (0, 4, 1),
                (0, 5, 1),
                (0, 6, 1),
                (0, 7, 1),
                (1, 4, 1),
                (1, 5, 1),
                (1, 6, 1),
                (1, 7, 1),
                (2, 4, 1),
                (2, 5, 1),
                (2, 6, 1),
                (2, 7, 1),
                (3, 4, 1),
                (3, 5, 1),
                (3, 6, 1),
                (3, 7, 1),
            ],
        },
        uniform_cycle_case("uniform_cycle_8", 8),
        prism_case("prism_6", 6),
        dimension_colored_cube_q3_case(),
    ];

    for case in cases {
        let rust_stats = rust_stats_for_owned_case(&case);
        let bliss_stats =
            canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap().stats;
        eprintln!("case {}", case.name);
        eprintln!("  rust_stats={rust_stats:?}");
        eprintln!("  bliss_stats={bliss_stats:?}");
    }
}

#[test]
#[ignore = "diagnostic probe for heuristic sensitivity of search stats against bliss"]
fn test_probe_representative_stats_across_heuristics() {
    let cases = vec![
        OwnedOracleCase {
            name: "branched_path_5".to_owned(),
            vertex_labels: vec![1_u8, 1, 1, 2, 2],
            edges: vec![(0, 1, 3), (1, 2, 3), (1, 3, 4), (1, 4, 4)],
        },
        OwnedOracleCase {
            name: "complete_bipartite_4x4".to_owned(),
            vertex_labels: vec![0_u8, 0, 0, 0, 1, 1, 1, 1],
            edges: vec![
                (0, 4, 1),
                (0, 5, 1),
                (0, 6, 1),
                (0, 7, 1),
                (1, 4, 1),
                (1, 5, 1),
                (1, 6, 1),
                (1, 7, 1),
                (2, 4, 1),
                (2, 5, 1),
                (2, 6, 1),
                (2, 7, 1),
                (3, 4, 1),
                (3, 5, 1),
                (3, 6, 1),
                (3, 7, 1),
            ],
        },
    ];
    let heuristics = [
        CanonSplittingHeuristic::First,
        CanonSplittingHeuristic::FirstSmallest,
        CanonSplittingHeuristic::FirstLargest,
        CanonSplittingHeuristic::FirstMaxNeighbours,
        CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
        CanonSplittingHeuristic::FirstLargestMaxNeighbours,
    ];

    for case in cases {
        let bliss_stats =
            canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap().stats;
        eprintln!("case {}", case.name);
        eprintln!("  bliss_stats={bliss_stats:?}");
        for heuristic in heuristics {
            let rust_stats = rust_stats_for_owned_case_with_heuristic(&case, heuristic);
            eprintln!("  {heuristic:?}: {rust_stats:?}");
        }
    }
}

#[test]
#[ignore = "diagnostic probe for representative bliss stats across failure-recording/component-recursion options"]
fn test_probe_representative_stats_across_bliss_options() {
    let bliss = locate_bliss_binary().unwrap();
    let cases = vec![
        OwnedOracleCase {
            name: "branched_path_5".to_owned(),
            vertex_labels: vec![1_u8, 1, 1, 2, 2],
            edges: vec![(0, 1, 3), (1, 2, 3), (1, 3, 4), (1, 4, 4)],
        },
        OwnedOracleCase {
            name: "complete_bipartite_4x4".to_owned(),
            vertex_labels: vec![0_u8, 0, 0, 0, 1, 1, 1, 1],
            edges: vec![
                (0, 4, 1),
                (0, 5, 1),
                (0, 6, 1),
                (0, 7, 1),
                (1, 4, 1),
                (1, 5, 1),
                (1, 6, 1),
                (1, 7, 1),
                (2, 4, 1),
                (2, 5, 1),
                (2, 6, 1),
                (2, 7, 1),
                (3, 4, 1),
                (3, 5, 1),
                (3, 6, 1),
                (3, 7, 1),
            ],
        },
    ];

    for case in cases {
        let encoded =
            encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
        let temp_dir = std::env::temp_dir().join(format!(
            "geometric-traits-bliss-options-probe-{}-{}",
            std::process::id(),
            case.name
        ));
        let _ = std::fs::create_dir_all(&temp_dir);
        let input_path = temp_dir.join("input.dimacs");
        std::fs::write(&input_path, &encoded.dimacs).unwrap();

        eprintln!("case {}", case.name);
        eprintln!("  rust_stats={:?}", rust_stats_for_owned_case(&case));
        for &(failure_recording, component_recursion) in
            &[(false, false), (false, true), (true, false), (true, true)]
        {
            let canonical_path = temp_dir.join(format!(
                "out-fr{}-cr{}.dimacs",
                usize::from(failure_recording),
                usize::from(component_recursion)
            ));
            let result = run_bliss_on_dimacs_file_with_options(
                &bliss,
                &input_path,
                &canonical_path,
                encoded.expanded_vertex_count,
                encoded.original_vertex_count,
                failure_recording,
                component_recursion,
            )
            .unwrap();
            eprintln!(
                "  bliss fr={} cr={}: {:?}",
                failure_recording, component_recursion, result.stats
            );
        }
    }
}

#[test]
#[ignore = "strict bliss search-stat fidelity target on a small asymmetric-with-symmetry case"]
fn test_rust_search_stats_eventually_match_bliss_on_branched_path_5() {
    let case = OwnedOracleCase {
        name: "branched_path_5".to_owned(),
        vertex_labels: vec![1_u8, 1, 1, 2, 2],
        edges: vec![(0, 1, 3), (1, 2, 3), (1, 3, 4), (1, 4, 4)],
    };

    let rust_stats = rust_stats_for_owned_case(&case);
    let bliss_stats =
        canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap().stats;

    assert_eq!(
        (rust_stats.search_nodes, rust_stats.leaf_nodes),
        (
            bliss_stats.nodes.expect("bliss nodes should be present"),
            bliss_stats.leaf_nodes.expect("bliss leaf_nodes should be present"),
        ),
        "search-stat mismatch on {}: rust={rust_stats:?} bliss={bliss_stats:?}",
        case.name
    );
}

#[test]
#[ignore = "strict bliss search-stat fidelity target on a highly symmetric case"]
fn test_rust_search_stats_eventually_match_bliss_on_complete_bipartite_4x4() {
    let case = OwnedOracleCase {
        name: "complete_bipartite_4x4".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 1, 1, 1, 1],
        edges: vec![
            (0, 4, 1),
            (0, 5, 1),
            (0, 6, 1),
            (0, 7, 1),
            (1, 4, 1),
            (1, 5, 1),
            (1, 6, 1),
            (1, 7, 1),
            (2, 4, 1),
            (2, 5, 1),
            (2, 6, 1),
            (2, 7, 1),
            (3, 4, 1),
            (3, 5, 1),
            (3, 6, 1),
            (3, 7, 1),
        ],
    };

    let rust_stats = rust_stats_for_owned_case(&case);
    let bliss_stats =
        canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap().stats;

    assert_eq!(
        (rust_stats.search_nodes, rust_stats.leaf_nodes),
        (
            bliss_stats.nodes.expect("bliss nodes should be present"),
            bliss_stats.leaf_nodes.expect("bliss leaf_nodes should be present"),
        ),
        "search-stat mismatch on {}: rust={rust_stats:?} bliss={bliss_stats:?}",
        case.name
    );
}

#[test]
#[ignore = "strict bliss search-stat fidelity target on a dense symmetric 10-vertex case"]
fn test_rust_search_stats_eventually_match_bliss_on_dense_uniform_n10_m26() {
    let case = OwnedOracleCase {
        name: "random_seed_333333333333355939_n10_m26".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        edges: vec![
            (0, 1, 0),
            (0, 3, 0),
            (0, 5, 0),
            (0, 7, 0),
            (0, 9, 0),
            (1, 2, 0),
            (1, 3, 0),
            (1, 7, 0),
            (1, 9, 0),
            (2, 3, 0),
            (2, 4, 0),
            (2, 5, 0),
            (2, 6, 0),
            (2, 7, 0),
            (3, 5, 0),
            (3, 6, 0),
            (3, 7, 0),
            (3, 8, 0),
            (4, 6, 0),
            (4, 8, 0),
            (5, 7, 0),
            (5, 8, 0),
            (5, 9, 0),
            (6, 7, 0),
            (6, 8, 0),
            (7, 8, 0),
        ],
    };

    let rust_stats = rust_stats_for_owned_case(&case);
    let bliss_stats =
        canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap().stats;

    assert_eq!(
        (rust_stats.search_nodes, rust_stats.leaf_nodes),
        (
            bliss_stats.nodes.expect("bliss nodes should be present"),
            bliss_stats.leaf_nodes.expect("bliss leaf_nodes should be present"),
        ),
        "search-stat mismatch on {}: rust={rust_stats:?} bliss={bliss_stats:?}",
        case.name
    );
}

#[test]
#[ignore = "strict bliss search-stat fidelity target on a dense labeled 9-vertex case"]
fn test_rust_search_stats_eventually_match_bliss_on_dense_labeled_n9_m36() {
    let case = OwnedOracleCase {
        name: "random_seed_333333333333457088_n9_m36".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0],
        edges: vec![
            (0, 1, 1),
            (0, 2, 1),
            (0, 3, 1),
            (0, 4, 0),
            (0, 5, 1),
            (0, 6, 1),
            (0, 7, 1),
            (0, 8, 0),
            (1, 2, 1),
            (1, 3, 1),
            (1, 4, 1),
            (1, 5, 0),
            (1, 6, 1),
            (1, 7, 0),
            (1, 8, 1),
            (2, 3, 1),
            (2, 4, 0),
            (2, 5, 0),
            (2, 6, 1),
            (2, 7, 1),
            (2, 8, 1),
            (3, 4, 0),
            (3, 5, 1),
            (3, 6, 1),
            (3, 7, 0),
            (3, 8, 1),
            (4, 5, 1),
            (4, 6, 1),
            (4, 7, 1),
            (4, 8, 0),
            (5, 6, 1),
            (5, 7, 0),
            (5, 8, 1),
            (6, 7, 0),
            (6, 8, 0),
            (7, 8, 1),
        ],
    };

    let rust_stats = rust_stats_for_owned_case(&case);
    let bliss_stats =
        canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap().stats;

    assert_eq!(
        (rust_stats.search_nodes, rust_stats.leaf_nodes),
        (
            bliss_stats.nodes.expect("bliss nodes should be present"),
            bliss_stats.leaf_nodes.expect("bliss leaf_nodes should be present"),
        ),
        "search-stat mismatch on {}: rust={rust_stats:?} bliss={bliss_stats:?}",
        case.name
    );
}

#[test]
#[ignore = "strict bliss search-stat fidelity target on a sparse disconnected uniform case"]
fn test_rust_search_stats_eventually_match_bliss_on_sparse_uniform_n9_m5() {
    let case = OwnedOracleCase {
        name: "sparse_uniform_n9_m5".to_owned(),
        vertex_labels: vec![0_u8; 9],
        edges: vec![(0, 2, 0), (1, 4, 0), (2, 5, 0), (3, 7, 0), (6, 7, 0)],
    };

    let rust_stats = rust_stats_for_owned_case(&case);
    let bliss_stats =
        canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap().stats;

    assert_eq!(
        (rust_stats.search_nodes, rust_stats.leaf_nodes),
        (
            bliss_stats.nodes.expect("bliss nodes should be present"),
            bliss_stats.leaf_nodes.expect("bliss leaf_nodes should be present"),
        ),
        "search-stat mismatch on {}: rust={rust_stats:?} bliss={bliss_stats:?}",
        case.name
    );
}

#[test]
#[ignore = "strict bliss search-stat fidelity target on a disconnected uniform 10-vertex case"]
fn test_rust_search_stats_eventually_match_bliss_on_disconnected_uniform_n10_m8() {
    let case = OwnedOracleCase {
        name: "disconnected_uniform_n10_m8".to_owned(),
        vertex_labels: vec![0_u8; 10],
        edges: vec![
            (0, 4, 0),
            (0, 9, 0),
            (1, 5, 0),
            (1, 8, 0),
            (3, 4, 0),
            (3, 9, 0),
            (5, 6, 0),
            (7, 8, 0),
        ],
    };

    let rust_stats = rust_stats_for_owned_case(&case);
    let bliss_stats =
        canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap().stats;

    assert_eq!(
        (rust_stats.search_nodes, rust_stats.leaf_nodes),
        (
            bliss_stats.nodes.expect("bliss nodes should be present"),
            bliss_stats.leaf_nodes.expect("bliss leaf_nodes should be present"),
        ),
        "search-stat mismatch on {}: rust={rust_stats:?} bliss={bliss_stats:?}",
        case.name
    );
}

#[test]
#[ignore = "diagnostic probe for heuristic sensitivity on sparse_uniform_n9_m5"]
fn test_probe_sparse_uniform_n9_m5_across_heuristics() {
    let case = OwnedOracleCase {
        name: "sparse_uniform_n9_m5".to_owned(),
        vertex_labels: vec![0_u8; 9],
        edges: vec![(0, 2, 0), (1, 4, 0), (2, 5, 0), (3, 7, 0), (6, 7, 0)],
    };
    let heuristics = [
        CanonSplittingHeuristic::First,
        CanonSplittingHeuristic::FirstSmallest,
        CanonSplittingHeuristic::FirstLargest,
        CanonSplittingHeuristic::FirstMaxNeighbours,
        CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
        CanonSplittingHeuristic::FirstLargestMaxNeighbours,
    ];

    let bliss_stats =
        canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap().stats;
    eprintln!("bliss_stats={bliss_stats:?}");
    for heuristic in heuristics {
        let rust_stats = rust_stats_for_owned_case_with_heuristic(&case, heuristic);
        eprintln!("{heuristic:?}: {rust_stats:?}");
    }
}

#[test]
#[ignore = "diagnostic probe for heuristic sensitivity on disconnected_uniform_n10_m8"]
fn test_probe_disconnected_uniform_n10_m8_across_heuristics() {
    let case = OwnedOracleCase {
        name: "disconnected_uniform_n10_m8".to_owned(),
        vertex_labels: vec![0_u8; 10],
        edges: vec![
            (0, 4, 0),
            (0, 9, 0),
            (1, 5, 0),
            (1, 8, 0),
            (3, 4, 0),
            (3, 9, 0),
            (5, 6, 0),
            (7, 8, 0),
        ],
    };
    let heuristics = [
        CanonSplittingHeuristic::First,
        CanonSplittingHeuristic::FirstSmallest,
        CanonSplittingHeuristic::FirstLargest,
        CanonSplittingHeuristic::FirstMaxNeighbours,
        CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
        CanonSplittingHeuristic::FirstLargestMaxNeighbours,
    ];

    let bliss_stats =
        canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap().stats;
    eprintln!("bliss_stats={bliss_stats:?}");
    for heuristic in heuristics {
        let rust_stats = rust_stats_for_owned_case_with_heuristic(&case, heuristic);
        eprintln!("{heuristic:?}: {rust_stats:?}");
    }
}

#[test]
#[ignore = "diagnostic probe for bliss trace on disconnected_uniform_n10_m8"]
fn test_probe_disconnected_uniform_n10_m8_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "disconnected_uniform_n10_m8".to_owned(),
        vertex_labels: vec![0_u8; 10],
        edges: vec![
            (0, 4, 0),
            (0, 9, 0),
            (1, 5, 0),
            (1, 8, 0),
            (3, 4, 0),
            (3, 9, 0),
            (5, 6, 0),
            (7, 8, 0),
        ],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-trace-probe-{}-{}",
        std::process::id(),
        case.name
    ));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output = Command::new(bliss)
        .env("BLISS_TRACE", "1")
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg("-fr=y")
        .arg("-cr=y")
        .arg(format!("-ocan={}", canonical_path.display()))
        .arg(&input_path)
        .output()
        .unwrap();

    eprintln!("stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "bliss trace run failed");
}

#[test]
#[ignore = "diagnostic probe for bliss option sensitivity on sparse_uniform_n9_m5"]
fn test_probe_sparse_uniform_n9_m5_bliss_option_sensitivity() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "sparse_uniform_n9_m5".to_owned(),
        vertex_labels: vec![0_u8; 9],
        edges: vec![(0, 2, 0), (1, 4, 0), (2, 5, 0), (3, 7, 0), (6, 7, 0)],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-options-probe-{}-{}",
        std::process::id(),
        case.name
    ));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    eprintln!("rust_stats={:?}", rust_stats_for_owned_case(&case));
    for &(failure_recording, component_recursion) in
        &[(false, false), (false, true), (true, false), (true, true)]
    {
        let canonical_path = temp_dir.join(format!(
            "out-fr{}-cr{}.dimacs",
            usize::from(failure_recording),
            usize::from(component_recursion)
        ));
        let result = run_bliss_on_dimacs_file_with_options(
            &bliss,
            &input_path,
            &canonical_path,
            encoded.expanded_vertex_count,
            encoded.original_vertex_count,
            failure_recording,
            component_recursion,
        )
        .unwrap();
        eprintln!("bliss fr={} cr={}: {:?}", failure_recording, component_recursion, result.stats);
    }
}

#[test]
#[ignore = "diagnostic probe for bliss trace on sparse_uniform_n9_m5"]
fn test_probe_sparse_uniform_n9_m5_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "sparse_uniform_n9_m5".to_owned(),
        vertex_labels: vec![0_u8; 9],
        edges: vec![(0, 2, 0), (1, 4, 0), (2, 5, 0), (3, 7, 0), (6, 7, 0)],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-trace-probe-{}-{}",
        std::process::id(),
        case.name
    ));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output = Command::new(bliss)
        .env("BLISS_TRACE", "1")
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg("-fr=y")
        .arg("-cr=y")
        .arg(format!("-ocan={}", canonical_path.display()))
        .arg(&input_path)
        .output()
        .unwrap();

    eprintln!("stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "bliss trace run failed");
}

#[test]
#[ignore = "diagnostic probe for bliss heuristic sensitivity on sparse_uniform_n9_m5"]
fn test_probe_sparse_uniform_n9_m5_bliss_across_heuristics() {
    fn parse_stat(stdout: &str, prefix: &str) -> Option<usize> {
        stdout.lines().find_map(|line| {
            let suffix = line.strip_prefix(prefix)?.trim();
            suffix.parse::<usize>().ok()
        })
    }

    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "sparse_uniform_n9_m5".to_owned(),
        vertex_labels: vec![0_u8; 9],
        edges: vec![(0, 2, 0), (1, 4, 0), (2, 5, 0), (3, 7, 0), (6, 7, 0)],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-heuristic-probe-{}-{}",
        std::process::id(),
        case.name
    ));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    for sh in ["f", "fs", "fl", "fm", "fsm", "flm"] {
        let canonical_path = temp_dir.join(format!("out-{sh}.dimacs"));
        let output = Command::new(&bliss)
            .arg("-can")
            .arg("-v=1")
            .arg(format!("-sh={sh}"))
            .arg("-fr=y")
            .arg("-cr=y")
            .arg(format!("-ocan={}", canonical_path.display()))
            .arg(&input_path)
            .output()
            .unwrap();
        assert!(output.status.success(), "bliss heuristic probe failed for sh={sh}");
        let stdout = String::from_utf8(output.stdout).unwrap();
        eprintln!(
            "bliss sh={sh}: nodes={:?} leaves={:?}",
            parse_stat(&stdout, "Nodes:"),
            parse_stat(&stdout, "Leaf nodes:")
        );
    }
}

#[test]
#[ignore = "diagnostic probe for heuristic sensitivity on representative heavy sparse random-gap case"]
fn test_probe_heavy_sparse_n7_m3_across_heuristics() {
    let case = OwnedOracleCase {
        name: "random_seed_777777777777789045_n7_m3".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0],
        edges: vec![(0, 1, 0), (3, 5, 0), (4, 6, 0)],
    };
    let heuristics = [
        CanonSplittingHeuristic::First,
        CanonSplittingHeuristic::FirstSmallest,
        CanonSplittingHeuristic::FirstLargest,
        CanonSplittingHeuristic::FirstMaxNeighbours,
        CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
        CanonSplittingHeuristic::FirstLargestMaxNeighbours,
    ];

    let bliss_stats =
        canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap().stats;
    eprintln!("bliss_stats={bliss_stats:?}");
    for heuristic in heuristics {
        let rust_stats = rust_stats_for_owned_case_with_heuristic(&case, heuristic);
        eprintln!("{heuristic:?}: {rust_stats:?}");
    }
}

#[test]
#[ignore = "diagnostic probe for bliss trace on representative heavy sparse random-gap case"]
fn test_probe_heavy_sparse_n7_m3_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "random_seed_777777777777789045_n7_m3".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0],
        edges: vec![(0, 1, 0), (3, 5, 0), (4, 6, 0)],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-trace-probe-{}-{}",
        std::process::id(),
        case.name
    ));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output = Command::new(bliss)
        .env("BLISS_TRACE", "1")
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg("-fr=y")
        .arg("-cr=y")
        .arg(format!("-ocan={}", canonical_path.display()))
        .arg(&input_path)
        .output()
        .unwrap();

    eprintln!("stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "bliss trace run failed");
}

#[test]
#[ignore = "diagnostic probe for heuristic sensitivity on representative heavy dense random-gap case"]
fn test_probe_heavy_dense_n9_m29_across_heuristics() {
    let case = OwnedOracleCase {
        name: "random_seed_777777777777822712_n9_m29".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0],
        edges: vec![
            (0, 1, 0),
            (0, 2, 0),
            (0, 4, 0),
            (0, 6, 0),
            (0, 7, 0),
            (0, 8, 0),
            (1, 3, 0),
            (1, 4, 0),
            (1, 5, 0),
            (1, 6, 0),
            (1, 7, 0),
            (2, 3, 0),
            (2, 4, 0),
            (2, 5, 0),
            (2, 7, 0),
            (2, 8, 0),
            (3, 4, 0),
            (3, 6, 0),
            (3, 7, 0),
            (3, 8, 0),
            (4, 5, 0),
            (4, 6, 0),
            (4, 7, 0),
            (4, 8, 0),
            (5, 6, 0),
            (5, 7, 0),
            (5, 8, 0),
            (6, 7, 0),
            (7, 8, 0),
        ],
    };
    let heuristics = [
        CanonSplittingHeuristic::First,
        CanonSplittingHeuristic::FirstSmallest,
        CanonSplittingHeuristic::FirstLargest,
        CanonSplittingHeuristic::FirstMaxNeighbours,
        CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
        CanonSplittingHeuristic::FirstLargestMaxNeighbours,
    ];

    let bliss_stats =
        canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap().stats;
    eprintln!("bliss_stats={bliss_stats:?}");
    for heuristic in heuristics {
        let rust_stats = rust_stats_for_owned_case_with_heuristic(&case, heuristic);
        eprintln!("{heuristic:?}: {rust_stats:?}");
    }
}

#[test]
#[ignore = "diagnostic probe for heuristic sensitivity on dense_uniform_n10_m26"]
fn test_probe_dense_uniform_n10_m26_across_heuristics() {
    let case = OwnedOracleCase {
        name: "random_seed_333333333333355939_n10_m26".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        edges: vec![
            (0, 1, 0),
            (0, 3, 0),
            (0, 5, 0),
            (0, 7, 0),
            (0, 9, 0),
            (1, 2, 0),
            (1, 3, 0),
            (1, 7, 0),
            (1, 9, 0),
            (2, 3, 0),
            (2, 4, 0),
            (2, 5, 0),
            (2, 6, 0),
            (2, 7, 0),
            (3, 5, 0),
            (3, 6, 0),
            (3, 7, 0),
            (3, 8, 0),
            (4, 6, 0),
            (4, 8, 0),
            (5, 7, 0),
            (5, 8, 0),
            (5, 9, 0),
            (6, 7, 0),
            (6, 8, 0),
            (7, 8, 0),
        ],
    };
    let heuristics = [
        CanonSplittingHeuristic::First,
        CanonSplittingHeuristic::FirstSmallest,
        CanonSplittingHeuristic::FirstLargest,
        CanonSplittingHeuristic::FirstMaxNeighbours,
        CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
        CanonSplittingHeuristic::FirstLargestMaxNeighbours,
    ];

    let bliss_stats =
        canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap().stats;
    eprintln!("bliss_stats={bliss_stats:?}");
    for heuristic in heuristics {
        let rust_stats = rust_stats_for_owned_case_with_heuristic(&case, heuristic);
        eprintln!("{heuristic:?}: {rust_stats:?}");
    }
}

#[test]
#[ignore = "diagnostic probe for heuristic sensitivity on dense_labeled_n9_m36"]
fn test_probe_dense_labeled_n9_m36_across_heuristics() {
    let case = OwnedOracleCase {
        name: "random_seed_333333333333457088_n9_m36".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0],
        edges: vec![
            (0, 1, 1),
            (0, 2, 1),
            (0, 3, 1),
            (0, 4, 0),
            (0, 5, 1),
            (0, 6, 1),
            (0, 7, 1),
            (0, 8, 0),
            (1, 2, 1),
            (1, 3, 1),
            (1, 4, 1),
            (1, 5, 0),
            (1, 6, 1),
            (1, 7, 0),
            (1, 8, 1),
            (2, 3, 1),
            (2, 4, 0),
            (2, 5, 0),
            (2, 6, 1),
            (2, 7, 1),
            (2, 8, 1),
            (3, 4, 0),
            (3, 5, 1),
            (3, 6, 1),
            (3, 7, 0),
            (3, 8, 1),
            (4, 5, 1),
            (4, 6, 1),
            (4, 7, 1),
            (4, 8, 0),
            (5, 6, 1),
            (5, 7, 0),
            (5, 8, 1),
            (6, 7, 0),
            (6, 8, 0),
            (7, 8, 1),
        ],
    };
    let heuristics = [
        CanonSplittingHeuristic::First,
        CanonSplittingHeuristic::FirstSmallest,
        CanonSplittingHeuristic::FirstLargest,
        CanonSplittingHeuristic::FirstMaxNeighbours,
        CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
        CanonSplittingHeuristic::FirstLargestMaxNeighbours,
    ];

    let bliss_stats =
        canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap().stats;
    eprintln!("bliss_stats={bliss_stats:?}");
    for heuristic in heuristics {
        let rust_stats = rust_stats_for_owned_case_with_heuristic(&case, heuristic);
        eprintln!("{heuristic:?}: {rust_stats:?}");
    }
}

#[test]
#[ignore = "diagnostic probe for bliss trace on dense_labeled_n9_m36"]
fn test_probe_dense_labeled_n9_m36_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "random_seed_333333333333457088_n9_m36".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0],
        edges: vec![
            (0, 1, 1),
            (0, 2, 1),
            (0, 3, 1),
            (0, 4, 0),
            (0, 5, 1),
            (0, 6, 1),
            (0, 7, 1),
            (0, 8, 0),
            (1, 2, 1),
            (1, 3, 1),
            (1, 4, 1),
            (1, 5, 0),
            (1, 6, 1),
            (1, 7, 0),
            (1, 8, 1),
            (2, 3, 1),
            (2, 4, 0),
            (2, 5, 0),
            (2, 6, 1),
            (2, 7, 1),
            (2, 8, 1),
            (3, 4, 0),
            (3, 5, 1),
            (3, 6, 1),
            (3, 7, 0),
            (3, 8, 1),
            (4, 5, 1),
            (4, 6, 1),
            (4, 7, 1),
            (4, 8, 0),
            (5, 6, 1),
            (5, 7, 0),
            (5, 8, 1),
            (6, 7, 0),
            (6, 8, 0),
            (7, 8, 1),
        ],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-trace-probe-{}-{}",
        std::process::id(),
        case.name
    ));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output = Command::new(bliss)
        .env("BLISS_TRACE", "1")
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg("-fr=y")
        .arg("-cr=y")
        .arg(format!("-ocan={}", canonical_path.display()))
        .arg(&input_path)
        .output()
        .unwrap();

    eprintln!("stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "bliss trace run failed");
}

#[test]
#[ignore = "diagnostic probe for bliss trace on dense_uniform_n10_m26"]
fn test_probe_dense_uniform_n10_m26_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "random_seed_333333333333355939_n10_m26".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        edges: vec![
            (0, 1, 0),
            (0, 3, 0),
            (0, 5, 0),
            (0, 7, 0),
            (0, 9, 0),
            (1, 2, 0),
            (1, 3, 0),
            (1, 7, 0),
            (1, 9, 0),
            (2, 3, 0),
            (2, 4, 0),
            (2, 5, 0),
            (2, 6, 0),
            (2, 7, 0),
            (3, 5, 0),
            (3, 6, 0),
            (3, 7, 0),
            (3, 8, 0),
            (4, 6, 0),
            (4, 8, 0),
            (5, 7, 0),
            (5, 8, 0),
            (5, 9, 0),
            (6, 7, 0),
            (6, 8, 0),
            (7, 8, 0),
        ],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-trace-probe-{}-{}",
        std::process::id(),
        case.name
    ));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output = Command::new(bliss)
        .env("BLISS_TRACE", "1")
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg("-fr=y")
        .arg("-cr=y")
        .arg(format!("-ocan={}", canonical_path.display()))
        .arg(&input_path)
        .output()
        .unwrap();

    eprintln!("stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "bliss trace run failed");
}

#[test]
#[ignore = "diagnostic probe for bliss trace on representative heavy dense random-gap case"]
fn test_probe_heavy_dense_n9_m29_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "random_seed_777777777777822712_n9_m29".to_owned(),
        vertex_labels: vec![0_u8, 0, 0, 0, 0, 0, 0, 0, 0],
        edges: vec![
            (0, 1, 0),
            (0, 2, 0),
            (0, 4, 0),
            (0, 6, 0),
            (0, 7, 0),
            (0, 8, 0),
            (1, 3, 0),
            (1, 4, 0),
            (1, 5, 0),
            (1, 6, 0),
            (1, 7, 0),
            (2, 3, 0),
            (2, 4, 0),
            (2, 5, 0),
            (2, 7, 0),
            (2, 8, 0),
            (3, 4, 0),
            (3, 6, 0),
            (3, 7, 0),
            (3, 8, 0),
            (4, 5, 0),
            (4, 6, 0),
            (4, 7, 0),
            (4, 8, 0),
            (5, 6, 0),
            (5, 7, 0),
            (5, 8, 0),
            (6, 7, 0),
            (7, 8, 0),
        ],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-trace-probe-{}-{}",
        std::process::id(),
        case.name
    ));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output = Command::new(bliss)
        .env("BLISS_TRACE", "1")
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg("-fr=y")
        .arg("-cr=y")
        .arg(format!("-ocan={}", canonical_path.display()))
        .arg(&input_path)
        .output()
        .unwrap();

    eprintln!("stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "bliss trace run failed");
}

#[test]
#[ignore = "parallel differential stress test against a local bliss oracle"]
fn test_rust_canonizer_matches_bliss_on_parallel_random_labeled_graphs() {
    let bliss = locate_bliss_binary().expect("missing bliss executable");
    assert!(bliss.is_file(), "expected located bliss binary {} to exist", bliss.display());

    let case_count = random_case_count();
    let base_seed = random_case_base_seed();
    let parallelism = random_case_parallelism();
    let mismatches = collect_random_window_mismatches(base_seed, case_count, parallelism);

    assert!(
        mismatches.is_empty(),
        "found {} random bliss mismatches out of {} cases (threads={}, base_seed={}):\n\n{}",
        mismatches.len(),
        case_count,
        parallelism,
        base_seed,
        mismatches.join("\n\n")
    );
}

#[test]
#[ignore = "parallel differential soak test across multiple deterministic seed windows against a local bliss oracle"]
fn test_rust_canonizer_matches_bliss_on_parallel_random_seed_windows() {
    let bliss = locate_bliss_binary().expect("missing bliss executable");
    assert!(bliss.is_file(), "expected located bliss binary {} to exist", bliss.display());

    let parallelism = 4;
    let windows =
        [(0xB115_5100_0000_0001_u64, 256_usize), (12_760_194_179_666_019_333_u64, 256_usize)];
    let window_count = 2_usize;

    let mismatches = windows
        .into_iter()
        .flat_map(|(base_seed, case_count)| {
            collect_random_window_mismatches(base_seed, case_count, parallelism).into_iter().map(
                move |mismatch| {
                    format!("window base_seed={base_seed} case_count={case_count}\n{mismatch}")
                },
            )
        })
        .collect::<Vec<_>>();

    assert!(
        mismatches.is_empty(),
        "found {} random bliss mismatches across {} deterministic windows (threads={}):\n\n{}",
        mismatches.len(),
        window_count,
        parallelism,
        mismatches.join("\n\n")
    );
}

#[test]
#[ignore = "aggressive parallel differential soak test across larger deterministic seed windows against a local bliss oracle"]
fn test_rust_canonizer_matches_bliss_on_parallel_aggressive_seed_windows() {
    let bliss = locate_bliss_binary().expect("missing bliss executable");
    assert!(bliss.is_file(), "expected located bliss binary {} to exist", bliss.display());

    let windows = [
        (424_242_424_242_424_242_u64, 1024_usize, 8_usize),
        (135_791_357_913_579_u64, 1024_usize, 8_usize),
        (222_222_222_222_222_222_u64, 2048_usize, 8_usize),
        (333_333_333_333_333_333_u64, 2048_usize, 8_usize),
        (98_765_432_123_456_789_u64, 256_usize, 8_usize),
    ];
    let window_count = windows.as_slice().len();

    let mismatches = windows
        .into_iter()
        .flat_map(|(base_seed, case_count, parallelism)| {
            collect_random_window_mismatches(base_seed, case_count, parallelism)
                .into_iter()
                .map(move |mismatch| {
                    format!(
                        "window base_seed={base_seed} case_count={case_count} threads={parallelism}\n{mismatch}"
                    )
                })
        })
        .collect::<Vec<_>>();

    assert!(
        mismatches.is_empty(),
        "found {} random bliss mismatches across {} aggressive deterministic windows:\n\n{}",
        mismatches.len(),
        window_count,
        mismatches.join("\n\n")
    );
}

#[test]
#[ignore = "exhaustive oracle check over all ternary edge masks on 5 unlabeled vertices"]
fn test_rust_canonizer_matches_bliss_on_exhaustive_ternary_edge_masks_n5() {
    let bliss = locate_bliss_binary().expect("missing bliss executable");
    assert!(bliss.is_file(), "expected located bliss binary {} to exist", bliss.display());

    let number_of_nodes = 5_usize;
    let edge_count = number_of_nodes * (number_of_nodes - 1) / 2;
    let case_count = 3_u64.pow(edge_count as u32);
    let parallelism = 8_usize;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallelism)
        .build()
        .expect("rayon pool for exhaustive bliss differential testing should build");

    let mismatches = pool.install(|| {
        (0..case_count)
            .into_par_iter()
            .filter_map(|mask| {
                let case = ternary_edge_mask_case(
                    &format!("ternary_edge_mask_n5_{mask}"),
                    number_of_nodes,
                    mask,
                );
                compare_rust_and_bliss_owned_case(&case)
                    .err()
                    .map(|error| describe_owned_case(mask, &case, &error))
            })
            .collect::<Vec<_>>()
    });

    assert!(
        mismatches.is_empty(),
        "found {} bliss mismatches across {} exhaustive ternary edge masks on n=5 (threads={}):\n\n{}",
        mismatches.len(),
        case_count,
        parallelism,
        mismatches.join("\n\n")
    );
}

#[test]
#[ignore = "exhaustive oracle check over binary vertex labels and ternary edge masks on 4 vertices"]
fn test_rust_canonizer_matches_bliss_on_exhaustive_binary_vertex_and_ternary_edge_masks_n4() {
    let bliss = locate_bliss_binary().expect("missing bliss executable");
    assert!(bliss.is_file(), "expected located bliss binary {} to exist", bliss.display());

    let number_of_nodes = 4_usize;
    let edge_count = number_of_nodes * (number_of_nodes - 1) / 2;
    let vertex_mask_count = 1_u64 << number_of_nodes;
    let edge_mask_count = 3_u64.pow(edge_count as u32);
    let case_count = vertex_mask_count * edge_mask_count;
    let parallelism = 8_usize;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallelism)
        .build()
        .expect("rayon pool for exhaustive bliss differential testing should build");

    let mismatches = pool.install(|| {
        (0..case_count)
            .into_par_iter()
            .filter_map(|index| {
                let vertex_mask = index / edge_mask_count;
                let edge_mask = index % edge_mask_count;
                let case = binary_vertex_mask_and_ternary_edge_mask_case(
                    &format!("binary_vertex_ternary_edge_mask_n4_v{vertex_mask}_e{edge_mask}"),
                    number_of_nodes,
                    vertex_mask,
                    edge_mask,
                );
                compare_rust_and_bliss_owned_case(&case)
                    .err()
                    .map(|error| describe_owned_case(index, &case, &error))
            })
            .collect::<Vec<_>>()
    });

    assert!(
        mismatches.is_empty(),
        "found {} bliss mismatches across {} exhaustive mixed-label cases on n=4 (threads={}):\n\n{}",
        mismatches.len(),
        case_count,
        parallelism,
        mismatches.join("\n\n")
    );
}

#[test]
#[ignore = "strict bliss stat-fidelity regression for a fuzz-found dense connected graph"]
fn test_rust_search_stats_eventually_match_bliss_on_fuzz_dense_connected_n14_m19() {
    let case = OwnedOracleCase {
        name: "fuzz_dense_connected_n14_m19".to_owned(),
        vertex_labels: vec![0_u8; 14],
        edges: vec![
            (0, 1, 0),
            (1, 2, 0),
            (1, 3, 0),
            (1, 4, 0),
            (2, 6, 0),
            (2, 7, 0),
            (2, 8, 0),
            (3, 11, 0),
            (3, 12, 0),
            (3, 13, 0),
            (5, 9, 0),
            (5, 10, 0),
            (5, 11, 0),
            (7, 11, 0),
            (7, 12, 0),
            (7, 13, 0),
            (11, 12, 0),
            (11, 13, 0),
            (12, 13, 0),
        ],
    };
    compare_rust_and_bliss_stats_owned_case(&case).unwrap();
}

#[test]
#[ignore = "strict bliss stat-fidelity regression for a fuzz-found dense labeled 24-vertex graph"]
fn test_rust_search_stats_eventually_match_bliss_on_fuzz_dense_labeled_n24_m276() {
    let case = OwnedOracleCase {
        name: "fuzz_dense_labeled_n24_m276".to_owned(),
        vertex_labels: vec![0_u8; 24],
        edges: vec![
            (0, 1, 3),
            (0, 2, 3),
            (0, 3, 3),
            (0, 4, 3),
            (0, 5, 3),
            (0, 6, 0),
            (0, 7, 3),
            (0, 8, 3),
            (0, 9, 3),
            (0, 10, 3),
            (0, 11, 3),
            (0, 12, 0),
            (0, 13, 3),
            (0, 14, 3),
            (0, 15, 3),
            (0, 16, 3),
            (0, 17, 3),
            (0, 18, 0),
            (0, 19, 3),
            (0, 20, 3),
            (0, 21, 3),
            (0, 22, 3),
            (0, 23, 3),
            (1, 2, 0),
            (1, 3, 3),
            (1, 4, 3),
            (1, 5, 3),
            (1, 6, 3),
            (1, 7, 3),
            (1, 8, 0),
            (1, 9, 3),
            (1, 10, 3),
            (1, 11, 3),
            (1, 12, 3),
            (1, 13, 3),
            (1, 14, 0),
            (1, 15, 3),
            (1, 16, 3),
            (1, 17, 3),
            (1, 18, 3),
            (1, 19, 3),
            (1, 20, 0),
            (1, 21, 3),
            (1, 22, 3),
            (1, 23, 3),
            (2, 3, 3),
            (2, 4, 3),
            (2, 5, 0),
            (2, 6, 3),
            (2, 7, 3),
            (2, 8, 3),
            (2, 9, 3),
            (2, 10, 3),
            (2, 11, 0),
            (2, 12, 3),
            (2, 13, 3),
            (2, 14, 3),
            (2, 15, 3),
            (2, 16, 3),
            (2, 17, 0),
            (2, 18, 3),
            (2, 19, 3),
            (2, 20, 3),
            (2, 21, 3),
            (2, 22, 3),
            (2, 23, 0),
            (3, 4, 3),
            (3, 5, 3),
            (3, 6, 3),
            (3, 7, 3),
            (3, 8, 3),
            (3, 9, 0),
            (3, 10, 3),
            (3, 11, 3),
            (3, 12, 3),
            (3, 13, 3),
            (3, 14, 3),
            (3, 15, 0),
            (3, 16, 3),
            (3, 17, 3),
            (3, 18, 3),
            (3, 19, 3),
            (3, 20, 3),
            (3, 21, 0),
            (3, 22, 3),
            (3, 23, 3),
            (4, 5, 3),
            (4, 6, 3),
            (4, 7, 3),
            (4, 8, 0),
            (4, 9, 3),
            (4, 10, 3),
            (4, 11, 3),
            (4, 12, 3),
            (4, 13, 3),
            (4, 14, 0),
            (4, 15, 3),
            (4, 16, 3),
            (4, 17, 3),
            (4, 18, 3),
            (4, 19, 3),
            (4, 20, 0),
            (4, 21, 3),
            (4, 22, 3),
            (4, 23, 3),
            (5, 6, 3),
            (5, 7, 3),
            (5, 8, 0),
            (5, 9, 3),
            (5, 10, 3),
            (5, 11, 3),
            (5, 12, 3),
            (5, 13, 3),
            (5, 14, 0),
            (5, 15, 3),
            (5, 16, 3),
            (5, 17, 3),
            (5, 18, 3),
            (5, 19, 3),
            (5, 20, 0),
            (5, 21, 3),
            (5, 22, 3),
            (5, 23, 3),
            (6, 7, 3),
            (6, 8, 3),
            (6, 9, 0),
            (6, 10, 3),
            (6, 11, 3),
            (6, 12, 3),
            (6, 13, 3),
            (6, 14, 3),
            (6, 15, 0),
            (6, 16, 3),
            (6, 17, 3),
            (6, 18, 3),
            (6, 19, 3),
            (6, 20, 3),
            (6, 21, 0),
            (6, 22, 3),
            (6, 23, 3),
            (7, 8, 3),
            (7, 9, 3),
            (7, 10, 3),
            (7, 11, 0),
            (7, 12, 3),
            (7, 13, 3),
            (7, 14, 3),
            (7, 15, 3),
            (7, 16, 3),
            (7, 17, 0),
            (7, 18, 3),
            (7, 19, 3),
            (7, 20, 3),
            (7, 21, 3),
            (7, 22, 3),
            (7, 23, 0),
            (8, 9, 3),
            (8, 10, 3),
            (8, 11, 3),
            (8, 12, 3),
            (8, 13, 3),
            (8, 14, 0),
            (8, 15, 3),
            (8, 16, 3),
            (8, 17, 3),
            (8, 18, 3),
            (8, 19, 3),
            (8, 20, 0),
            (8, 21, 3),
            (8, 22, 3),
            (8, 23, 3),
            (9, 10, 3),
            (9, 11, 3),
            (9, 12, 0),
            (9, 13, 3),
            (9, 14, 3),
            (9, 15, 3),
            (9, 16, 3),
            (9, 17, 3),
            (9, 18, 0),
            (9, 19, 3),
            (9, 20, 3),
            (9, 21, 3),
            (9, 22, 3),
            (9, 23, 3),
            (10, 11, 0),
            (10, 12, 3),
            (10, 13, 3),
            (10, 14, 3),
            (10, 15, 3),
            (10, 16, 3),
            (10, 17, 0),
            (10, 18, 3),
            (10, 19, 3),
            (10, 20, 3),
            (10, 21, 3),
            (10, 22, 3),
            (10, 23, 0),
            (11, 12, 3),
            (11, 13, 3),
            (11, 14, 3),
            (11, 15, 3),
            (11, 16, 3),
            (11, 17, 0),
            (11, 18, 3),
            (11, 19, 3),
            (11, 20, 3),
            (11, 21, 3),
            (11, 22, 3),
            (11, 23, 0),
            (12, 13, 3),
            (12, 14, 3),
            (12, 15, 3),
            (12, 16, 3),
            (12, 17, 3),
            (12, 18, 0),
            (12, 19, 3),
            (12, 20, 3),
            (12, 21, 3),
            (12, 22, 3),
            (12, 23, 3),
            (13, 14, 0),
            (13, 15, 3),
            (13, 16, 3),
            (13, 17, 3),
            (13, 18, 3),
            (13, 19, 3),
            (13, 20, 0),
            (13, 21, 3),
            (13, 22, 3),
            (13, 23, 3),
            (14, 15, 3),
            (14, 16, 3),
            (14, 17, 0),
            (14, 18, 3),
            (14, 19, 3),
            (14, 20, 3),
            (14, 21, 3),
            (14, 22, 3),
            (14, 23, 0),
            (15, 16, 3),
            (15, 17, 3),
            (15, 18, 3),
            (15, 19, 3),
            (15, 20, 3),
            (15, 21, 0),
            (15, 22, 3),
            (15, 23, 3),
            (16, 17, 3),
            (16, 18, 3),
            (16, 19, 3),
            (16, 20, 0),
            (16, 21, 3),
            (16, 22, 3),
            (16, 23, 3),
            (17, 18, 3),
            (17, 19, 3),
            (17, 20, 0),
            (17, 21, 3),
            (17, 22, 3),
            (17, 23, 3),
            (18, 19, 3),
            (18, 20, 3),
            (18, 21, 0),
            (18, 22, 3),
            (18, 23, 3),
            (19, 20, 3),
            (19, 21, 3),
            (19, 22, 3),
            (19, 23, 0),
            (20, 21, 3),
            (20, 22, 3),
            (20, 23, 3),
            (21, 22, 3),
            (21, 23, 3),
            (22, 23, 0),
        ],
    };
    compare_rust_and_bliss_stats_owned_case(&case).unwrap();
}

#[test]
#[ignore = "strict bliss stat-fidelity regression for a fuzz-found sparse structured 24-vertex graph"]
fn test_rust_search_stats_eventually_match_bliss_on_fuzz_sparse_structured_n24_m46() {
    let case = OwnedOracleCase {
        name: "fuzz_sparse_structured_n24_m46".to_owned(),
        vertex_labels: vec![0_u8; 24],
        edges: vec![
            (0, 6, 0),
            (0, 12, 0),
            (0, 18, 0),
            (1, 2, 0),
            (1, 8, 0),
            (1, 14, 0),
            (1, 20, 0),
            (2, 5, 0),
            (2, 11, 0),
            (2, 17, 0),
            (2, 23, 0),
            (3, 9, 0),
            (3, 15, 0),
            (3, 21, 0),
            (4, 8, 0),
            (4, 14, 0),
            (4, 20, 0),
            (5, 8, 0),
            (5, 14, 0),
            (5, 20, 0),
            (6, 9, 0),
            (6, 15, 0),
            (6, 21, 0),
            (7, 11, 0),
            (7, 17, 0),
            (7, 23, 0),
            (8, 14, 0),
            (8, 20, 0),
            (9, 12, 0),
            (9, 18, 0),
            (10, 11, 0),
            (10, 17, 0),
            (10, 23, 0),
            (11, 17, 0),
            (11, 23, 0),
            (12, 18, 0),
            (13, 14, 0),
            (13, 20, 0),
            (14, 17, 0),
            (14, 23, 0),
            (15, 21, 0),
            (16, 20, 0),
            (17, 20, 0),
            (18, 21, 0),
            (19, 23, 0),
            (22, 23, 0),
        ],
    };
    compare_rust_and_bliss_stats_owned_case(&case).unwrap();
}

#[test]
#[ignore = "strict bliss stat-fidelity regression for a fuzz-found labeled sparse structured 25-vertex graph"]
fn test_rust_search_stats_eventually_match_bliss_on_fuzz_labeled_sparse_n25_m50() {
    let case = OwnedOracleCase {
        name: "fuzz_labeled_sparse_n25_m50".to_owned(),
        vertex_labels: vec![
            0_u8, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
        ],
        edges: vec![
            (0, 3, 0),
            (0, 9, 0),
            (0, 15, 0),
            (0, 21, 0),
            (1, 4, 0),
            (1, 10, 0),
            (1, 16, 0),
            (1, 22, 0),
            (2, 6, 0),
            (2, 12, 0),
            (2, 18, 0),
            (2, 24, 0),
            (3, 9, 0),
            (3, 15, 0),
            (3, 21, 0),
            (4, 7, 0),
            (4, 13, 0),
            (4, 19, 0),
            (5, 6, 0),
            (5, 12, 0),
            (5, 18, 0),
            (5, 24, 0),
            (6, 12, 0),
            (6, 18, 0),
            (6, 24, 0),
            (7, 13, 0),
            (7, 19, 0),
            (8, 9, 0),
            (8, 15, 0),
            (8, 21, 0),
            (9, 12, 0),
            (9, 18, 0),
            (9, 24, 0),
            (10, 16, 0),
            (10, 22, 0),
            (11, 15, 0),
            (11, 21, 0),
            (12, 15, 0),
            (12, 21, 0),
            (13, 16, 0),
            (13, 22, 0),
            (14, 18, 0),
            (14, 24, 0),
            (15, 21, 0),
            (16, 19, 0),
            (17, 18, 0),
            (17, 24, 0),
            (18, 24, 0),
            (20, 21, 0),
            (21, 24, 0),
        ],
    };
    compare_rust_and_bliss_stats_owned_case(&case).unwrap();
}

#[test]
#[ignore = "strict bliss stat-fidelity regression for a fuzz-found sparse structured 25-vertex graph"]
fn test_rust_search_stats_eventually_match_bliss_on_fuzz_sparse_structured_n25_m13() {
    let case = OwnedOracleCase {
        name: "fuzz_sparse_structured_n25_m13".to_owned(),
        vertex_labels: vec![0_u8; 25],
        edges: vec![
            (1, 19, 0),
            (1, 20, 0),
            (3, 19, 0),
            (3, 20, 0),
            (5, 23, 0),
            (5, 24, 0),
            (8, 15, 0),
            (8, 16, 0),
            (11, 16, 0),
            (11, 17, 0),
            (15, 17, 0),
            (15, 18, 0),
            (23, 24, 0),
        ],
    };
    compare_rust_and_bliss_stats_owned_case(&case).unwrap();
}

#[test]
#[ignore = "diagnostic probe for heuristic sensitivity on fuzz_sparse_structured_n25_m13"]
fn test_probe_fuzz_sparse_structured_n25_m13_across_heuristics() {
    let case = OwnedOracleCase {
        name: "fuzz_sparse_structured_n25_m13".to_owned(),
        vertex_labels: vec![0_u8; 25],
        edges: vec![
            (1, 19, 0),
            (1, 20, 0),
            (3, 19, 0),
            (3, 20, 0),
            (5, 23, 0),
            (5, 24, 0),
            (8, 15, 0),
            (8, 16, 0),
            (11, 16, 0),
            (11, 17, 0),
            (15, 17, 0),
            (15, 18, 0),
            (23, 24, 0),
        ],
    };
    let heuristics = [
        CanonSplittingHeuristic::First,
        CanonSplittingHeuristic::FirstSmallest,
        CanonSplittingHeuristic::FirstLargest,
        CanonSplittingHeuristic::FirstMaxNeighbours,
        CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
        CanonSplittingHeuristic::FirstLargestMaxNeighbours,
    ];

    let bliss_stats =
        canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap().stats;
    eprintln!("bliss_stats={bliss_stats:?}");
    for heuristic in heuristics {
        let rust_stats = rust_stats_for_owned_case_with_heuristic(&case, heuristic);
        eprintln!("{heuristic:?}: {rust_stats:?}");
    }
}

#[test]
#[ignore = "diagnostic probe for bliss option sensitivity on fuzz_sparse_structured_n25_m13"]
fn test_probe_fuzz_sparse_structured_n25_m13_bliss_option_sensitivity() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "fuzz_sparse_structured_n25_m13".to_owned(),
        vertex_labels: vec![0_u8; 25],
        edges: vec![
            (1, 19, 0),
            (1, 20, 0),
            (3, 19, 0),
            (3, 20, 0),
            (5, 23, 0),
            (5, 24, 0),
            (8, 15, 0),
            (8, 16, 0),
            (11, 16, 0),
            (11, 17, 0),
            (15, 17, 0),
            (15, 18, 0),
            (23, 24, 0),
        ],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!("gt-bliss-probe-{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir).unwrap();
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();
    for failure_recording in [false, true] {
        for component_recursion in [false, true] {
            let result = run_bliss_on_dimacs_file_with_options(
                &bliss,
                &input_path,
                &canonical_path,
                encoded.expanded_vertex_count,
                encoded.original_vertex_count,
                failure_recording,
                component_recursion,
            )
            .unwrap();
            eprintln!(
                "fr={failure_recording} cr={component_recursion}: nodes={:?} leaves={:?} generators={:?}",
                result.stats.nodes, result.stats.leaf_nodes, result.stats.generators
            );
        }
    }
    let _ = std::fs::remove_dir_all(temp_dir);
}

#[test]
#[ignore = "diagnostic probe for bliss trace on fuzz_sparse_structured_n25_m13"]
fn test_probe_fuzz_sparse_structured_n25_m13_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "fuzz_sparse_structured_n25_m13".to_owned(),
        vertex_labels: vec![0_u8; 25],
        edges: vec![
            (1, 19, 0),
            (1, 20, 0),
            (3, 19, 0),
            (3, 20, 0),
            (5, 23, 0),
            (5, 24, 0),
            (8, 15, 0),
            (8, 16, 0),
            (11, 16, 0),
            (11, 17, 0),
            (15, 17, 0),
            (15, 18, 0),
            (23, 24, 0),
        ],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-trace-probe-{}-{}",
        std::process::id(),
        case.name
    ));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output = Command::new(bliss)
        .env("BLISS_TRACE", "1")
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg("-fr=y")
        .arg("-cr=y")
        .arg(format!("-ocan={}", canonical_path.display()))
        .arg(&input_path)
        .output()
        .unwrap();

    eprintln!("stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "bliss trace run failed");
}

#[test]
#[ignore = "diagnostic probe for heuristic sensitivity on fuzz_labeled_sparse_n25_m50"]
fn test_probe_fuzz_labeled_sparse_n25_m50_across_heuristics() {
    let case = OwnedOracleCase {
        name: "fuzz_labeled_sparse_n25_m50".to_owned(),
        vertex_labels: vec![
            0_u8, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
        ],
        edges: vec![
            (0, 3, 0),
            (0, 9, 0),
            (0, 15, 0),
            (0, 21, 0),
            (1, 4, 0),
            (1, 10, 0),
            (1, 16, 0),
            (1, 22, 0),
            (2, 6, 0),
            (2, 12, 0),
            (2, 18, 0),
            (2, 24, 0),
            (3, 9, 0),
            (3, 15, 0),
            (3, 21, 0),
            (4, 7, 0),
            (4, 13, 0),
            (4, 19, 0),
            (5, 6, 0),
            (5, 12, 0),
            (5, 18, 0),
            (5, 24, 0),
            (6, 12, 0),
            (6, 18, 0),
            (6, 24, 0),
            (7, 13, 0),
            (7, 19, 0),
            (8, 9, 0),
            (8, 15, 0),
            (8, 21, 0),
            (9, 12, 0),
            (9, 18, 0),
            (9, 24, 0),
            (10, 16, 0),
            (10, 22, 0),
            (11, 15, 0),
            (11, 21, 0),
            (12, 15, 0),
            (12, 21, 0),
            (13, 16, 0),
            (13, 22, 0),
            (14, 18, 0),
            (14, 24, 0),
            (15, 21, 0),
            (16, 19, 0),
            (17, 18, 0),
            (17, 24, 0),
            (18, 24, 0),
            (20, 21, 0),
            (21, 24, 0),
        ],
    };
    let heuristics = [
        CanonSplittingHeuristic::First,
        CanonSplittingHeuristic::FirstSmallest,
        CanonSplittingHeuristic::FirstLargest,
        CanonSplittingHeuristic::FirstMaxNeighbours,
        CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
        CanonSplittingHeuristic::FirstLargestMaxNeighbours,
    ];

    let bliss_stats =
        canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap().stats;
    eprintln!("bliss_stats={bliss_stats:?}");
    for heuristic in heuristics {
        let rust_stats = rust_stats_for_owned_case_with_heuristic(&case, heuristic);
        eprintln!("{heuristic:?}: {rust_stats:?}");
    }
}

#[test]
#[ignore = "diagnostic probe for bliss option sensitivity on fuzz_labeled_sparse_n25_m50"]
fn test_probe_fuzz_labeled_sparse_n25_m50_bliss_option_sensitivity() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "fuzz_labeled_sparse_n25_m50".to_owned(),
        vertex_labels: vec![
            0_u8, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
        ],
        edges: vec![
            (0, 3, 0),
            (0, 9, 0),
            (0, 15, 0),
            (0, 21, 0),
            (1, 4, 0),
            (1, 10, 0),
            (1, 16, 0),
            (1, 22, 0),
            (2, 6, 0),
            (2, 12, 0),
            (2, 18, 0),
            (2, 24, 0),
            (3, 9, 0),
            (3, 15, 0),
            (3, 21, 0),
            (4, 7, 0),
            (4, 13, 0),
            (4, 19, 0),
            (5, 6, 0),
            (5, 12, 0),
            (5, 18, 0),
            (5, 24, 0),
            (6, 12, 0),
            (6, 18, 0),
            (6, 24, 0),
            (7, 13, 0),
            (7, 19, 0),
            (8, 9, 0),
            (8, 15, 0),
            (8, 21, 0),
            (9, 12, 0),
            (9, 18, 0),
            (9, 24, 0),
            (10, 16, 0),
            (10, 22, 0),
            (11, 15, 0),
            (11, 21, 0),
            (12, 15, 0),
            (12, 21, 0),
            (13, 16, 0),
            (13, 22, 0),
            (14, 18, 0),
            (14, 24, 0),
            (15, 21, 0),
            (16, 19, 0),
            (17, 18, 0),
            (17, 24, 0),
            (18, 24, 0),
            (20, 21, 0),
            (21, 24, 0),
        ],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!("gt-bliss-probe-{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir).unwrap();
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();
    for failure_recording in [false, true] {
        for component_recursion in [false, true] {
            let result = run_bliss_on_dimacs_file_with_options(
                &bliss,
                &input_path,
                &canonical_path,
                encoded.expanded_vertex_count,
                encoded.original_vertex_count,
                failure_recording,
                component_recursion,
            )
            .unwrap();
            eprintln!(
                "fr={failure_recording} cr={component_recursion}: nodes={:?} leaves={:?} bad={:?} generators={:?}",
                result.stats.nodes,
                result.stats.leaf_nodes,
                result.stats.bad_nodes,
                result.stats.generators
            );
        }
    }
    let _ = std::fs::remove_dir_all(temp_dir);
}

#[test]
#[ignore = "diagnostic probe for bliss trace on fuzz_labeled_sparse_n25_m50"]
fn test_probe_fuzz_labeled_sparse_n25_m50_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "fuzz_labeled_sparse_n25_m50".to_owned(),
        vertex_labels: vec![
            0_u8, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
        ],
        edges: vec![
            (0, 3, 0),
            (0, 9, 0),
            (0, 15, 0),
            (0, 21, 0),
            (1, 4, 0),
            (1, 10, 0),
            (1, 16, 0),
            (1, 22, 0),
            (2, 6, 0),
            (2, 12, 0),
            (2, 18, 0),
            (2, 24, 0),
            (3, 9, 0),
            (3, 15, 0),
            (3, 21, 0),
            (4, 7, 0),
            (4, 13, 0),
            (4, 19, 0),
            (5, 6, 0),
            (5, 12, 0),
            (5, 18, 0),
            (5, 24, 0),
            (6, 12, 0),
            (6, 18, 0),
            (6, 24, 0),
            (7, 13, 0),
            (7, 19, 0),
            (8, 9, 0),
            (8, 15, 0),
            (8, 21, 0),
            (9, 12, 0),
            (9, 18, 0),
            (9, 24, 0),
            (10, 16, 0),
            (10, 22, 0),
            (11, 15, 0),
            (11, 21, 0),
            (12, 15, 0),
            (12, 21, 0),
            (13, 16, 0),
            (13, 22, 0),
            (14, 18, 0),
            (14, 24, 0),
            (15, 21, 0),
            (16, 19, 0),
            (17, 18, 0),
            (17, 24, 0),
            (18, 24, 0),
            (20, 21, 0),
            (21, 24, 0),
        ],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!("gt-bliss-trace-{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir).unwrap();
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();
    let output = Command::new(&bliss)
        .env("BLISS_TRACE", "1")
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg("-fr=y")
        .arg("-cr=y")
        .arg(format!("-ocan={}", canonical_path.display()))
        .arg(&input_path)
        .output()
        .unwrap();
    eprintln!("stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    let _ = std::fs::remove_dir_all(temp_dir);
}

#[test]
#[ignore = "triage latest fuzz-found case before reducing it further"]
fn test_triage_fuzz_case_n25_m50() {
    let case = OwnedOracleCase {
        name: "fuzz_case_n25_m50".to_owned(),
        vertex_labels: vec![0_u8; 25],
        edges: vec![
            (0, 6, 0),
            (0, 12, 0),
            (0, 18, 0),
            (0, 24, 0),
            (1, 7, 0),
            (1, 13, 0),
            (1, 19, 0),
            (2, 3, 0),
            (2, 9, 0),
            (2, 15, 0),
            (2, 21, 0),
            (3, 6, 0),
            (3, 12, 0),
            (3, 18, 0),
            (3, 24, 0),
            (4, 10, 0),
            (4, 16, 0),
            (4, 22, 0),
            (5, 9, 0),
            (5, 15, 0),
            (5, 21, 0),
            (6, 9, 0),
            (6, 15, 0),
            (6, 21, 0),
            (7, 10, 0),
            (7, 16, 0),
            (7, 22, 0),
            (8, 12, 0),
            (8, 18, 0),
            (8, 24, 0),
            (9, 15, 0),
            (9, 21, 0),
            (10, 13, 0),
            (10, 19, 0),
            (11, 12, 0),
            (11, 18, 0),
            (11, 24, 0),
            (12, 18, 0),
            (12, 24, 0),
            (13, 19, 0),
            (14, 15, 0),
            (14, 21, 0),
            (15, 18, 0),
            (15, 24, 0),
            (16, 22, 0),
            (17, 21, 0),
            (18, 21, 0),
            (19, 22, 0),
            (20, 24, 0),
            (23, 24, 0),
        ],
    };
    compare_rust_and_bliss_owned_case(&case).unwrap();
    compare_rust_and_bliss_stats_owned_case(&case).unwrap();
}

#[test]
#[ignore = "diagnostic probe for heuristic sensitivity on fuzz_case_n25_m50"]
fn test_probe_fuzz_case_n25_m50_across_heuristics() {
    let case = OwnedOracleCase {
        name: "fuzz_case_n25_m50".to_owned(),
        vertex_labels: vec![0_u8; 25],
        edges: vec![
            (0, 6, 0),
            (0, 12, 0),
            (0, 18, 0),
            (0, 24, 0),
            (1, 7, 0),
            (1, 13, 0),
            (1, 19, 0),
            (2, 3, 0),
            (2, 9, 0),
            (2, 15, 0),
            (2, 21, 0),
            (3, 6, 0),
            (3, 12, 0),
            (3, 18, 0),
            (3, 24, 0),
            (4, 10, 0),
            (4, 16, 0),
            (4, 22, 0),
            (5, 9, 0),
            (5, 15, 0),
            (5, 21, 0),
            (6, 9, 0),
            (6, 15, 0),
            (6, 21, 0),
            (7, 10, 0),
            (7, 16, 0),
            (7, 22, 0),
            (8, 12, 0),
            (8, 18, 0),
            (8, 24, 0),
            (9, 15, 0),
            (9, 21, 0),
            (10, 13, 0),
            (10, 19, 0),
            (11, 12, 0),
            (11, 18, 0),
            (11, 24, 0),
            (12, 18, 0),
            (12, 24, 0),
            (13, 19, 0),
            (14, 15, 0),
            (14, 21, 0),
            (15, 18, 0),
            (15, 24, 0),
            (16, 22, 0),
            (17, 21, 0),
            (18, 21, 0),
            (19, 22, 0),
            (20, 24, 0),
            (23, 24, 0),
        ],
    };
    let heuristics = [
        CanonSplittingHeuristic::First,
        CanonSplittingHeuristic::FirstSmallest,
        CanonSplittingHeuristic::FirstLargest,
        CanonSplittingHeuristic::FirstMaxNeighbours,
        CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
        CanonSplittingHeuristic::FirstLargestMaxNeighbours,
    ];

    let bliss_stats =
        canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap().stats;
    eprintln!("bliss_stats={bliss_stats:?}");
    for heuristic in heuristics {
        let rust_stats = rust_stats_for_owned_case_with_heuristic(&case, heuristic);
        eprintln!("{heuristic:?}: {rust_stats:?}");
    }
}

#[test]
#[ignore = "diagnostic probe for heuristic sensitivity on fuzz_sparse_structured_n24_m46"]
fn test_probe_fuzz_sparse_structured_n24_m46_across_heuristics() {
    let case = OwnedOracleCase {
        name: "fuzz_sparse_structured_n24_m46".to_owned(),
        vertex_labels: vec![0_u8; 24],
        edges: vec![
            (0, 6, 0),
            (0, 12, 0),
            (0, 18, 0),
            (1, 2, 0),
            (1, 8, 0),
            (1, 14, 0),
            (1, 20, 0),
            (2, 5, 0),
            (2, 11, 0),
            (2, 17, 0),
            (2, 23, 0),
            (3, 9, 0),
            (3, 15, 0),
            (3, 21, 0),
            (4, 8, 0),
            (4, 14, 0),
            (4, 20, 0),
            (5, 8, 0),
            (5, 14, 0),
            (5, 20, 0),
            (6, 9, 0),
            (6, 15, 0),
            (6, 21, 0),
            (7, 11, 0),
            (7, 17, 0),
            (7, 23, 0),
            (8, 14, 0),
            (8, 20, 0),
            (9, 12, 0),
            (9, 18, 0),
            (10, 11, 0),
            (10, 17, 0),
            (10, 23, 0),
            (11, 17, 0),
            (11, 23, 0),
            (12, 18, 0),
            (13, 14, 0),
            (13, 20, 0),
            (14, 17, 0),
            (14, 23, 0),
            (15, 21, 0),
            (16, 20, 0),
            (17, 20, 0),
            (18, 21, 0),
            (19, 23, 0),
            (22, 23, 0),
        ],
    };
    let heuristics = [
        CanonSplittingHeuristic::First,
        CanonSplittingHeuristic::FirstSmallest,
        CanonSplittingHeuristic::FirstLargest,
        CanonSplittingHeuristic::FirstMaxNeighbours,
        CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
        CanonSplittingHeuristic::FirstLargestMaxNeighbours,
    ];

    let bliss_stats =
        canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap().stats;
    eprintln!("bliss_stats={bliss_stats:?}");
    for heuristic in heuristics {
        let rust_stats = rust_stats_for_owned_case_with_heuristic(&case, heuristic);
        eprintln!("{heuristic:?}: {rust_stats:?}");
    }
}

#[test]
#[ignore = "diagnostic probe for bliss option sensitivity on fuzz_sparse_structured_n24_m46"]
fn test_probe_fuzz_sparse_structured_n24_m46_bliss_option_sensitivity() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "fuzz_sparse_structured_n24_m46".to_owned(),
        vertex_labels: vec![0_u8; 24],
        edges: vec![
            (0, 6, 0),
            (0, 12, 0),
            (0, 18, 0),
            (1, 2, 0),
            (1, 8, 0),
            (1, 14, 0),
            (1, 20, 0),
            (2, 5, 0),
            (2, 11, 0),
            (2, 17, 0),
            (2, 23, 0),
            (3, 9, 0),
            (3, 15, 0),
            (3, 21, 0),
            (4, 8, 0),
            (4, 14, 0),
            (4, 20, 0),
            (5, 8, 0),
            (5, 14, 0),
            (5, 20, 0),
            (6, 9, 0),
            (6, 15, 0),
            (6, 21, 0),
            (7, 11, 0),
            (7, 17, 0),
            (7, 23, 0),
            (8, 14, 0),
            (8, 20, 0),
            (9, 12, 0),
            (9, 18, 0),
            (10, 11, 0),
            (10, 17, 0),
            (10, 23, 0),
            (11, 17, 0),
            (11, 23, 0),
            (12, 18, 0),
            (13, 14, 0),
            (13, 20, 0),
            (14, 17, 0),
            (14, 23, 0),
            (15, 21, 0),
            (16, 20, 0),
            (17, 20, 0),
            (18, 21, 0),
            (19, 23, 0),
            (22, 23, 0),
        ],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!("gt-bliss-probe-{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir).unwrap();
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();
    for failure_recording in [false, true] {
        for component_recursion in [false, true] {
            let result = run_bliss_on_dimacs_file_with_options(
                &bliss,
                &input_path,
                &canonical_path,
                encoded.expanded_vertex_count,
                encoded.original_vertex_count,
                failure_recording,
                component_recursion,
            )
            .unwrap();
            eprintln!(
                "fr={failure_recording} cr={component_recursion}: nodes={:?} leaves={:?} generators={:?}",
                result.stats.nodes, result.stats.leaf_nodes, result.stats.generators
            );
        }
    }
    let _ = std::fs::remove_dir_all(temp_dir);
}

#[test]
#[ignore = "diagnostic probe for bliss trace on fuzz_sparse_structured_n24_m46"]
fn test_probe_fuzz_sparse_structured_n24_m46_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "fuzz_sparse_structured_n24_m46".to_owned(),
        vertex_labels: vec![0_u8; 24],
        edges: vec![
            (0, 6, 0),
            (0, 12, 0),
            (0, 18, 0),
            (1, 2, 0),
            (1, 8, 0),
            (1, 14, 0),
            (1, 20, 0),
            (2, 5, 0),
            (2, 11, 0),
            (2, 17, 0),
            (2, 23, 0),
            (3, 9, 0),
            (3, 15, 0),
            (3, 21, 0),
            (4, 8, 0),
            (4, 14, 0),
            (4, 20, 0),
            (5, 8, 0),
            (5, 14, 0),
            (5, 20, 0),
            (6, 9, 0),
            (6, 15, 0),
            (6, 21, 0),
            (7, 11, 0),
            (7, 17, 0),
            (7, 23, 0),
            (8, 14, 0),
            (8, 20, 0),
            (9, 12, 0),
            (9, 18, 0),
            (10, 11, 0),
            (10, 17, 0),
            (10, 23, 0),
            (11, 17, 0),
            (11, 23, 0),
            (12, 18, 0),
            (13, 14, 0),
            (13, 20, 0),
            (14, 17, 0),
            (14, 23, 0),
            (15, 21, 0),
            (16, 20, 0),
            (17, 20, 0),
            (18, 21, 0),
            (19, 23, 0),
            (22, 23, 0),
        ],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!("gt-bliss-trace-{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir).unwrap();
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();
    let output = Command::new(&bliss)
        .env("BLISS_TRACE", "1")
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg("-fr=y")
        .arg("-cr=y")
        .arg(format!("-ocan={}", canonical_path.display()))
        .arg(&input_path)
        .output()
        .unwrap();
    eprintln!("stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    let _ = std::fs::remove_dir_all(temp_dir);
}

#[test]
#[ignore = "diagnostic probe for bliss trace on fuzz_case_n25_m50"]
fn test_probe_fuzz_case_n25_m50_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "fuzz_case_n25_m50".to_owned(),
        vertex_labels: vec![0_u8; 25],
        edges: vec![
            (0, 6, 0),
            (0, 12, 0),
            (0, 18, 0),
            (0, 24, 0),
            (1, 7, 0),
            (1, 13, 0),
            (1, 19, 0),
            (2, 3, 0),
            (2, 9, 0),
            (2, 15, 0),
            (2, 21, 0),
            (3, 6, 0),
            (3, 12, 0),
            (3, 18, 0),
            (3, 24, 0),
            (4, 10, 0),
            (4, 16, 0),
            (4, 22, 0),
            (5, 9, 0),
            (5, 15, 0),
            (5, 21, 0),
            (6, 9, 0),
            (6, 15, 0),
            (6, 21, 0),
            (7, 10, 0),
            (7, 16, 0),
            (7, 22, 0),
            (8, 12, 0),
            (8, 18, 0),
            (8, 24, 0),
            (9, 15, 0),
            (9, 21, 0),
            (10, 13, 0),
            (10, 19, 0),
            (11, 12, 0),
            (11, 18, 0),
            (11, 24, 0),
            (12, 18, 0),
            (12, 24, 0),
            (13, 19, 0),
            (14, 15, 0),
            (14, 21, 0),
            (15, 18, 0),
            (15, 24, 0),
            (16, 22, 0),
            (17, 21, 0),
            (18, 21, 0),
            (19, 22, 0),
            (20, 24, 0),
            (23, 24, 0),
        ],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-trace-probe-{}-{}",
        std::process::id(),
        case.name
    ));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output = Command::new(bliss)
        .env("BLISS_TRACE", "1")
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg("-fr=y")
        .arg("-cr=y")
        .arg(format!("-ocan={}", canonical_path.display()))
        .arg(&input_path)
        .output()
        .unwrap();
    eprintln!("{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("{}", String::from_utf8_lossy(&output.stderr));
    let _ = std::fs::remove_dir_all(temp_dir);
}

#[test]
#[ignore = "diagnostic probe for heuristic sensitivity on fuzz_dense_connected_n14_m19"]
fn test_probe_fuzz_dense_connected_n14_m19_across_heuristics() {
    let case = OwnedOracleCase {
        name: "fuzz_dense_connected_n14_m19".to_owned(),
        vertex_labels: vec![0_u8; 14],
        edges: vec![
            (0, 1, 0),
            (1, 2, 0),
            (1, 3, 0),
            (1, 4, 0),
            (2, 6, 0),
            (2, 7, 0),
            (2, 8, 0),
            (3, 11, 0),
            (3, 12, 0),
            (3, 13, 0),
            (5, 9, 0),
            (5, 10, 0),
            (5, 11, 0),
            (7, 11, 0),
            (7, 12, 0),
            (7, 13, 0),
            (11, 12, 0),
            (11, 13, 0),
            (12, 13, 0),
        ],
    };
    let heuristics = [
        CanonSplittingHeuristic::First,
        CanonSplittingHeuristic::FirstSmallest,
        CanonSplittingHeuristic::FirstLargest,
        CanonSplittingHeuristic::FirstMaxNeighbours,
        CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
        CanonSplittingHeuristic::FirstLargestMaxNeighbours,
    ];

    let bliss_stats =
        canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap().stats;
    eprintln!("bliss_stats={bliss_stats:?}");
    for heuristic in heuristics {
        let rust_stats = rust_stats_for_owned_case_with_heuristic(&case, heuristic);
        eprintln!("{heuristic:?}: {rust_stats:?}");
    }
}

#[test]
#[ignore = "diagnostic probe for heuristic sensitivity on fuzz_dense_labeled_n24_m276"]
fn test_probe_fuzz_dense_labeled_n24_m276_across_heuristics() {
    let case = OwnedOracleCase {
        name: "fuzz_dense_labeled_n24_m276".to_owned(),
        vertex_labels: vec![0_u8; 24],
        edges: vec![
            (0, 1, 3),
            (0, 2, 3),
            (0, 3, 3),
            (0, 4, 3),
            (0, 5, 3),
            (0, 6, 0),
            (0, 7, 3),
            (0, 8, 3),
            (0, 9, 3),
            (0, 10, 3),
            (0, 11, 3),
            (0, 12, 0),
            (0, 13, 3),
            (0, 14, 3),
            (0, 15, 3),
            (0, 16, 3),
            (0, 17, 3),
            (0, 18, 0),
            (0, 19, 3),
            (0, 20, 3),
            (0, 21, 3),
            (0, 22, 3),
            (0, 23, 3),
            (1, 2, 0),
            (1, 3, 3),
            (1, 4, 3),
            (1, 5, 3),
            (1, 6, 3),
            (1, 7, 3),
            (1, 8, 0),
            (1, 9, 3),
            (1, 10, 3),
            (1, 11, 3),
            (1, 12, 3),
            (1, 13, 3),
            (1, 14, 0),
            (1, 15, 3),
            (1, 16, 3),
            (1, 17, 3),
            (1, 18, 3),
            (1, 19, 3),
            (1, 20, 0),
            (1, 21, 3),
            (1, 22, 3),
            (1, 23, 3),
            (2, 3, 3),
            (2, 4, 3),
            (2, 5, 0),
            (2, 6, 3),
            (2, 7, 3),
            (2, 8, 3),
            (2, 9, 3),
            (2, 10, 3),
            (2, 11, 0),
            (2, 12, 3),
            (2, 13, 3),
            (2, 14, 3),
            (2, 15, 3),
            (2, 16, 3),
            (2, 17, 0),
            (2, 18, 3),
            (2, 19, 3),
            (2, 20, 3),
            (2, 21, 3),
            (2, 22, 3),
            (2, 23, 0),
            (3, 4, 3),
            (3, 5, 3),
            (3, 6, 3),
            (3, 7, 3),
            (3, 8, 3),
            (3, 9, 0),
            (3, 10, 3),
            (3, 11, 3),
            (3, 12, 3),
            (3, 13, 3),
            (3, 14, 3),
            (3, 15, 0),
            (3, 16, 3),
            (3, 17, 3),
            (3, 18, 3),
            (3, 19, 3),
            (3, 20, 3),
            (3, 21, 0),
            (3, 22, 3),
            (3, 23, 3),
            (4, 5, 3),
            (4, 6, 3),
            (4, 7, 3),
            (4, 8, 0),
            (4, 9, 3),
            (4, 10, 3),
            (4, 11, 3),
            (4, 12, 3),
            (4, 13, 3),
            (4, 14, 0),
            (4, 15, 3),
            (4, 16, 3),
            (4, 17, 3),
            (4, 18, 3),
            (4, 19, 3),
            (4, 20, 0),
            (4, 21, 3),
            (4, 22, 3),
            (4, 23, 3),
            (5, 6, 3),
            (5, 7, 3),
            (5, 8, 0),
            (5, 9, 3),
            (5, 10, 3),
            (5, 11, 3),
            (5, 12, 3),
            (5, 13, 3),
            (5, 14, 0),
            (5, 15, 3),
            (5, 16, 3),
            (5, 17, 3),
            (5, 18, 3),
            (5, 19, 3),
            (5, 20, 0),
            (5, 21, 3),
            (5, 22, 3),
            (5, 23, 3),
            (6, 7, 3),
            (6, 8, 3),
            (6, 9, 0),
            (6, 10, 3),
            (6, 11, 3),
            (6, 12, 3),
            (6, 13, 3),
            (6, 14, 3),
            (6, 15, 0),
            (6, 16, 3),
            (6, 17, 3),
            (6, 18, 3),
            (6, 19, 3),
            (6, 20, 3),
            (6, 21, 0),
            (6, 22, 3),
            (6, 23, 3),
            (7, 8, 3),
            (7, 9, 3),
            (7, 10, 3),
            (7, 11, 0),
            (7, 12, 3),
            (7, 13, 3),
            (7, 14, 3),
            (7, 15, 3),
            (7, 16, 3),
            (7, 17, 0),
            (7, 18, 3),
            (7, 19, 3),
            (7, 20, 3),
            (7, 21, 3),
            (7, 22, 3),
            (7, 23, 0),
            (8, 9, 3),
            (8, 10, 3),
            (8, 11, 3),
            (8, 12, 3),
            (8, 13, 3),
            (8, 14, 0),
            (8, 15, 3),
            (8, 16, 3),
            (8, 17, 3),
            (8, 18, 3),
            (8, 19, 3),
            (8, 20, 0),
            (8, 21, 3),
            (8, 22, 3),
            (8, 23, 3),
            (9, 10, 3),
            (9, 11, 3),
            (9, 12, 0),
            (9, 13, 3),
            (9, 14, 3),
            (9, 15, 3),
            (9, 16, 3),
            (9, 17, 3),
            (9, 18, 0),
            (9, 19, 3),
            (9, 20, 3),
            (9, 21, 3),
            (9, 22, 3),
            (9, 23, 3),
            (10, 11, 0),
            (10, 12, 3),
            (10, 13, 3),
            (10, 14, 3),
            (10, 15, 3),
            (10, 16, 3),
            (10, 17, 0),
            (10, 18, 3),
            (10, 19, 3),
            (10, 20, 3),
            (10, 21, 3),
            (10, 22, 3),
            (10, 23, 0),
            (11, 12, 3),
            (11, 13, 3),
            (11, 14, 3),
            (11, 15, 3),
            (11, 16, 3),
            (11, 17, 0),
            (11, 18, 3),
            (11, 19, 3),
            (11, 20, 3),
            (11, 21, 3),
            (11, 22, 3),
            (11, 23, 0),
            (12, 13, 3),
            (12, 14, 3),
            (12, 15, 3),
            (12, 16, 3),
            (12, 17, 3),
            (12, 18, 0),
            (12, 19, 3),
            (12, 20, 3),
            (12, 21, 3),
            (12, 22, 3),
            (12, 23, 3),
            (13, 14, 0),
            (13, 15, 3),
            (13, 16, 3),
            (13, 17, 3),
            (13, 18, 3),
            (13, 19, 3),
            (13, 20, 0),
            (13, 21, 3),
            (13, 22, 3),
            (13, 23, 3),
            (14, 15, 3),
            (14, 16, 3),
            (14, 17, 0),
            (14, 18, 3),
            (14, 19, 3),
            (14, 20, 3),
            (14, 21, 3),
            (14, 22, 3),
            (14, 23, 0),
            (15, 16, 3),
            (15, 17, 3),
            (15, 18, 3),
            (15, 19, 3),
            (15, 20, 3),
            (15, 21, 0),
            (15, 22, 3),
            (15, 23, 3),
            (16, 17, 3),
            (16, 18, 3),
            (16, 19, 3),
            (16, 20, 0),
            (16, 21, 3),
            (16, 22, 3),
            (16, 23, 3),
            (17, 18, 3),
            (17, 19, 3),
            (17, 20, 0),
            (17, 21, 3),
            (17, 22, 3),
            (17, 23, 3),
            (18, 19, 3),
            (18, 20, 3),
            (18, 21, 0),
            (18, 22, 3),
            (18, 23, 3),
            (19, 20, 3),
            (19, 21, 3),
            (19, 22, 3),
            (19, 23, 0),
            (20, 21, 3),
            (20, 22, 3),
            (20, 23, 3),
            (21, 22, 3),
            (21, 23, 3),
            (22, 23, 0),
        ],
    };
    let heuristics = [
        CanonSplittingHeuristic::First,
        CanonSplittingHeuristic::FirstSmallest,
        CanonSplittingHeuristic::FirstLargest,
        CanonSplittingHeuristic::FirstMaxNeighbours,
        CanonSplittingHeuristic::FirstSmallestMaxNeighbours,
        CanonSplittingHeuristic::FirstLargestMaxNeighbours,
    ];

    let bliss_stats =
        canonicalize_labeled_simple_graph(&case.vertex_labels, &case.edges).unwrap().stats;
    eprintln!("bliss_stats={bliss_stats:?}");
    for heuristic in heuristics {
        let rust_stats = rust_stats_for_owned_case_with_heuristic(&case, heuristic);
        eprintln!("{heuristic:?}: {rust_stats:?}");
    }
}

#[test]
#[ignore = "diagnostic probe for bliss trace on fuzz_dense_labeled_n24_m276"]
fn test_probe_fuzz_dense_labeled_n24_m276_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "fuzz_dense_labeled_n24_m276".to_owned(),
        vertex_labels: vec![0_u8; 24],
        edges: vec![
            (0, 1, 3),
            (0, 2, 3),
            (0, 3, 3),
            (0, 4, 3),
            (0, 5, 3),
            (0, 6, 0),
            (0, 7, 3),
            (0, 8, 3),
            (0, 9, 3),
            (0, 10, 3),
            (0, 11, 3),
            (0, 12, 0),
            (0, 13, 3),
            (0, 14, 3),
            (0, 15, 3),
            (0, 16, 3),
            (0, 17, 3),
            (0, 18, 0),
            (0, 19, 3),
            (0, 20, 3),
            (0, 21, 3),
            (0, 22, 3),
            (0, 23, 3),
            (1, 2, 0),
            (1, 3, 3),
            (1, 4, 3),
            (1, 5, 3),
            (1, 6, 3),
            (1, 7, 3),
            (1, 8, 0),
            (1, 9, 3),
            (1, 10, 3),
            (1, 11, 3),
            (1, 12, 3),
            (1, 13, 3),
            (1, 14, 0),
            (1, 15, 3),
            (1, 16, 3),
            (1, 17, 3),
            (1, 18, 3),
            (1, 19, 3),
            (1, 20, 0),
            (1, 21, 3),
            (1, 22, 3),
            (1, 23, 3),
            (2, 3, 3),
            (2, 4, 3),
            (2, 5, 0),
            (2, 6, 3),
            (2, 7, 3),
            (2, 8, 3),
            (2, 9, 3),
            (2, 10, 3),
            (2, 11, 0),
            (2, 12, 3),
            (2, 13, 3),
            (2, 14, 3),
            (2, 15, 3),
            (2, 16, 3),
            (2, 17, 0),
            (2, 18, 3),
            (2, 19, 3),
            (2, 20, 3),
            (2, 21, 3),
            (2, 22, 3),
            (2, 23, 0),
            (3, 4, 3),
            (3, 5, 3),
            (3, 6, 3),
            (3, 7, 3),
            (3, 8, 3),
            (3, 9, 0),
            (3, 10, 3),
            (3, 11, 3),
            (3, 12, 3),
            (3, 13, 3),
            (3, 14, 3),
            (3, 15, 0),
            (3, 16, 3),
            (3, 17, 3),
            (3, 18, 3),
            (3, 19, 3),
            (3, 20, 3),
            (3, 21, 0),
            (3, 22, 3),
            (3, 23, 3),
            (4, 5, 3),
            (4, 6, 3),
            (4, 7, 3),
            (4, 8, 0),
            (4, 9, 3),
            (4, 10, 3),
            (4, 11, 3),
            (4, 12, 3),
            (4, 13, 3),
            (4, 14, 0),
            (4, 15, 3),
            (4, 16, 3),
            (4, 17, 3),
            (4, 18, 3),
            (4, 19, 3),
            (4, 20, 0),
            (4, 21, 3),
            (4, 22, 3),
            (4, 23, 3),
            (5, 6, 3),
            (5, 7, 3),
            (5, 8, 0),
            (5, 9, 3),
            (5, 10, 3),
            (5, 11, 3),
            (5, 12, 3),
            (5, 13, 3),
            (5, 14, 0),
            (5, 15, 3),
            (5, 16, 3),
            (5, 17, 3),
            (5, 18, 3),
            (5, 19, 3),
            (5, 20, 0),
            (5, 21, 3),
            (5, 22, 3),
            (5, 23, 3),
            (6, 7, 3),
            (6, 8, 3),
            (6, 9, 0),
            (6, 10, 3),
            (6, 11, 3),
            (6, 12, 3),
            (6, 13, 3),
            (6, 14, 3),
            (6, 15, 0),
            (6, 16, 3),
            (6, 17, 3),
            (6, 18, 3),
            (6, 19, 3),
            (6, 20, 3),
            (6, 21, 0),
            (6, 22, 3),
            (6, 23, 3),
            (7, 8, 3),
            (7, 9, 3),
            (7, 10, 3),
            (7, 11, 0),
            (7, 12, 3),
            (7, 13, 3),
            (7, 14, 3),
            (7, 15, 3),
            (7, 16, 3),
            (7, 17, 0),
            (7, 18, 3),
            (7, 19, 3),
            (7, 20, 3),
            (7, 21, 3),
            (7, 22, 3),
            (7, 23, 0),
            (8, 9, 3),
            (8, 10, 3),
            (8, 11, 3),
            (8, 12, 3),
            (8, 13, 3),
            (8, 14, 0),
            (8, 15, 3),
            (8, 16, 3),
            (8, 17, 3),
            (8, 18, 3),
            (8, 19, 3),
            (8, 20, 0),
            (8, 21, 3),
            (8, 22, 3),
            (8, 23, 3),
            (9, 10, 3),
            (9, 11, 3),
            (9, 12, 0),
            (9, 13, 3),
            (9, 14, 3),
            (9, 15, 3),
            (9, 16, 3),
            (9, 17, 3),
            (9, 18, 0),
            (9, 19, 3),
            (9, 20, 3),
            (9, 21, 3),
            (9, 22, 3),
            (9, 23, 3),
            (10, 11, 0),
            (10, 12, 3),
            (10, 13, 3),
            (10, 14, 3),
            (10, 15, 3),
            (10, 16, 3),
            (10, 17, 0),
            (10, 18, 3),
            (10, 19, 3),
            (10, 20, 3),
            (10, 21, 3),
            (10, 22, 3),
            (10, 23, 0),
            (11, 12, 3),
            (11, 13, 3),
            (11, 14, 3),
            (11, 15, 3),
            (11, 16, 3),
            (11, 17, 0),
            (11, 18, 3),
            (11, 19, 3),
            (11, 20, 3),
            (11, 21, 3),
            (11, 22, 3),
            (11, 23, 0),
            (12, 13, 3),
            (12, 14, 3),
            (12, 15, 3),
            (12, 16, 3),
            (12, 17, 3),
            (12, 18, 0),
            (12, 19, 3),
            (12, 20, 3),
            (12, 21, 3),
            (12, 22, 3),
            (12, 23, 3),
            (13, 14, 0),
            (13, 15, 3),
            (13, 16, 3),
            (13, 17, 3),
            (13, 18, 3),
            (13, 19, 3),
            (13, 20, 0),
            (13, 21, 3),
            (13, 22, 3),
            (13, 23, 3),
            (14, 15, 3),
            (14, 16, 3),
            (14, 17, 0),
            (14, 18, 3),
            (14, 19, 3),
            (14, 20, 3),
            (14, 21, 3),
            (14, 22, 3),
            (14, 23, 0),
            (15, 16, 3),
            (15, 17, 3),
            (15, 18, 3),
            (15, 19, 3),
            (15, 20, 3),
            (15, 21, 0),
            (15, 22, 3),
            (15, 23, 3),
            (16, 17, 3),
            (16, 18, 3),
            (16, 19, 3),
            (16, 20, 0),
            (16, 21, 3),
            (16, 22, 3),
            (16, 23, 3),
            (17, 18, 3),
            (17, 19, 3),
            (17, 20, 0),
            (17, 21, 3),
            (17, 22, 3),
            (17, 23, 3),
            (18, 19, 3),
            (18, 20, 3),
            (18, 21, 0),
            (18, 22, 3),
            (18, 23, 3),
            (19, 20, 3),
            (19, 21, 3),
            (19, 22, 3),
            (19, 23, 0),
            (20, 21, 3),
            (20, 22, 3),
            (20, 23, 3),
            (21, 22, 3),
            (21, 23, 3),
            (22, 23, 0),
        ],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-trace-probe-{}-{}",
        std::process::id(),
        case.name
    ));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output = Command::new(bliss)
        .env("BLISS_TRACE", "1")
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg("-fr=y")
        .arg("-cr=y")
        .arg(format!("-ocan={}", canonical_path.display()))
        .arg(&input_path)
        .output()
        .unwrap();

    eprintln!(
        "stdout:
{}",
        String::from_utf8_lossy(&output.stdout)
    );
    eprintln!(
        "stderr:
{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.status.success(), "bliss trace run failed");
}

#[test]
#[ignore = "diagnostic probe for bliss trace on fuzz_dense_connected_n14_m19"]
fn test_probe_fuzz_dense_connected_n14_m19_bliss_trace() {
    let bliss = locate_bliss_binary().unwrap();
    let case = OwnedOracleCase {
        name: "fuzz_dense_connected_n14_m19".to_owned(),
        vertex_labels: vec![0_u8; 14],
        edges: vec![
            (0, 1, 0),
            (1, 2, 0),
            (1, 3, 0),
            (1, 4, 0),
            (2, 6, 0),
            (2, 7, 0),
            (2, 8, 0),
            (3, 11, 0),
            (3, 12, 0),
            (3, 13, 0),
            (5, 9, 0),
            (5, 10, 0),
            (5, 11, 0),
            (7, 11, 0),
            (7, 12, 0),
            (7, 13, 0),
            (11, 12, 0),
            (11, 13, 0),
            (12, 13, 0),
        ],
    };
    let encoded = encode_labeled_simple_graph_as_dimacs(&case.vertex_labels, &case.edges).unwrap();
    let temp_dir = std::env::temp_dir().join(format!(
        "geometric-traits-bliss-trace-probe-{}-{}",
        std::process::id(),
        case.name
    ));
    let _ = std::fs::create_dir_all(&temp_dir);
    let input_path = temp_dir.join("input.dimacs");
    let canonical_path = temp_dir.join("canonical.dimacs");
    std::fs::write(&input_path, &encoded.dimacs).unwrap();

    let output = Command::new(bliss)
        .env("BLISS_TRACE", "1")
        .arg("-can")
        .arg("-v=1")
        .arg("-sh=fsm")
        .arg("-fr=y")
        .arg("-cr=y")
        .arg(format!("-ocan={}", canonical_path.display()))
        .arg(&input_path)
        .output()
        .unwrap();

    eprintln!("stdout:\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "bliss trace run failed");
}
