//! Tests for labeled equitable refinement used by canonization.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::{CSR2D, SortedVec, SymmetricCSR2D, ValuedCSR2D},
    naive_structs::GenericGraph,
    prelude::*,
    traits::{
        Edges, EdgesBuilder, MonoplexGraph, SparseValuedMatrix2D, VocabularyBuilder,
        WeisfeilerLehmanColoring, refine_partition_to_labeled_equitable,
    },
};

type LabeledUndirectedEdges = SymmetricCSR2D<ValuedCSR2D<usize, usize, usize, u8>>;
type LabeledUndirectedGraph = GenericGraph<SortedVec<usize>, LabeledUndirectedEdges>;

fn build_undirected_graph(number_of_nodes: usize, edges: &[(usize, usize)]) -> UndiGraph<usize> {
    let nodes: SortedVec<usize> = GenericVocabularyBuilder::default()
        .expected_number_of_symbols(number_of_nodes)
        .symbols((0..number_of_nodes).enumerate())
        .build()
        .unwrap();
    let mut canonical_edges: Vec<(usize, usize)> = edges
        .iter()
        .map(
            |&(source, destination)| {
                if source <= destination { (source, destination) } else { (destination, source) }
            },
        )
        .collect();
    canonical_edges.sort_unstable();
    canonical_edges.dedup();
    let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_number_of_edges(canonical_edges.len())
        .expected_shape(number_of_nodes)
        .edges(canonical_edges.into_iter())
        .build()
        .unwrap();

    UndiGraph::from((nodes, edges))
}

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

fn partition_classes(partition: &BacktrackableOrderedPartition) -> Vec<Vec<usize>> {
    let mut classes = partition
        .cells()
        .map(|cell| {
            let mut elements = cell.elements().to_vec();
            elements.sort_unstable();
            elements
        })
        .collect::<Vec<_>>();
    classes.sort_unstable();
    classes
}

fn color_classes(colors: &[usize]) -> Vec<Vec<usize>> {
    let mut grouped = std::collections::BTreeMap::<usize, Vec<usize>>::new();
    for (vertex, &color) in colors.iter().enumerate() {
        grouped.entry(color).or_default().push(vertex);
    }
    let mut classes = grouped.into_values().collect::<Vec<_>>();
    classes.sort_unstable();
    classes
}

#[test]
fn test_equitable_refinement_matches_unlabeled_path_symmetry() {
    let graph = build_undirected_graph(3, &[(0, 1), (1, 2)]);
    let mut partition = BacktrackableOrderedPartition::new(3);

    let changed = refine_partition_to_labeled_equitable(&graph, &mut partition, |_, _| ());

    assert!(changed);
    assert_eq!(partition_classes(&partition), vec![vec![0, 2], vec![1]]);
    assert!(!refine_partition_to_labeled_equitable(&graph, &mut partition, |_, _| ()));
}

#[test]
fn test_equitable_refinement_with_edge_labels_distinguishes_terminal_vertices() {
    let graph = build_bidirectional_labeled_graph(3, &[(0, 1, 1), (1, 2, 2)]);
    let matrix = Edges::matrix(graph.edges());
    let mut partition = BacktrackableOrderedPartition::new(3);

    let changed =
        refine_partition_to_labeled_equitable(&graph, &mut partition, |source, destination| {
            matrix.sparse_value_at(source, destination).unwrap()
        });

    assert!(changed);
    assert!(partition.is_discrete());
    assert_eq!(partition_classes(&partition), vec![vec![0], vec![1], vec![2]]);
}

#[test]
fn test_equitable_refinement_matches_wl_from_seed_and_edge_labels() {
    let graph = build_bidirectional_labeled_graph(
        6,
        &[(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 4, 2), (4, 5, 1)],
    );
    let seed_colors = [3_u8, 1, 1, 1, 1, 3];
    let matrix = Edges::matrix(graph.edges());
    let mut partition = BacktrackableOrderedPartition::new(6);
    let root = partition.cell_of(0);
    let _ = partition.split_cell_by_key(root, |vertex| seed_colors[vertex]);

    let changed =
        refine_partition_to_labeled_equitable(&graph, &mut partition, |source, destination| {
            matrix.sparse_value_at(source, destination).unwrap()
        });
    let wl_colors = graph
        .wl_coloring_with_seed_and_edge_colors(&seed_colors, |source, destination| {
            matrix.sparse_value_at(source, destination).unwrap()
        });

    assert!(changed);
    assert_eq!(partition_classes(&partition), color_classes(&wl_colors));
}
