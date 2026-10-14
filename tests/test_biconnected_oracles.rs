//! Internal consistency checks for the Hopcroft-Tarjan oracle corpora.
#![cfg(feature = "std")]

#[path = "support/biconnected_fixture.rs"]
mod biconnected_fixture;

use std::collections::BTreeSet;

use biconnected_fixture::{
    BiconnectedFixtureCase, build_undigraph, load_fixture_suite, normalize_edge, semantic_cases,
};
use geometric_traits::{
    prelude::connected_components::ConnectedComponentsResult,
    traits::{ConnectedComponents, MonopartiteGraph},
};

const EXHAUSTIVE_FIXTURE_NAME: &str = "biconnected_components_order5_exhaustive.json.gz";

fn canonicalize_edge_list(mut edges: Vec<[usize; 2]>) -> Vec<[usize; 2]> {
    for edge in &mut edges {
        *edge = normalize_edge(*edge);
    }
    edges.sort_unstable();
    edges.dedup();
    edges
}

fn canonicalize_vertex_list(mut vertices: Vec<usize>) -> Vec<usize> {
    vertices.sort_unstable();
    vertices.dedup();
    vertices
}

fn canonicalize_vertex_components(components: &[Vec<usize>]) -> BTreeSet<Vec<usize>> {
    components.iter().cloned().map(canonicalize_vertex_list).collect()
}

fn vertex_block_from_edges(component: &[[usize; 2]]) -> Vec<usize> {
    canonicalize_vertex_list(component.iter().flat_map(|edge| [edge[0], edge[1]]).collect())
}

fn derived_articulation_points(node_count: usize, vertex_components: &[Vec<usize>]) -> Vec<usize> {
    let mut memberships = vec![0usize; node_count];
    for component in vertex_components {
        for &vertex in component {
            memberships[vertex] += 1;
        }
    }
    memberships
        .into_iter()
        .enumerate()
        .filter_map(|(vertex, count)| (count > 1).then_some(vertex))
        .collect()
}

fn derived_missing_vertices(node_count: usize, vertex_components: &[Vec<usize>]) -> Vec<usize> {
    let mut present = vec![false; node_count];
    for component in vertex_components {
        for &vertex in component {
            present[vertex] = true;
        }
    }
    present
        .into_iter()
        .enumerate()
        .filter_map(|(vertex, is_present)| (!is_present).then_some(vertex))
        .collect()
}

fn derived_cyclic_indices(edge_components: &[Vec<[usize; 2]>]) -> Vec<usize> {
    edge_components
        .iter()
        .enumerate()
        .filter_map(|(index, component)| {
            let vertex_count = vertex_block_from_edges(component).len();
            (component.len() >= vertex_count && !component.is_empty()).then_some(index)
        })
        .collect()
}

fn expected_is_biconnected(case: &BiconnectedFixtureCase) -> bool {
    case.node_count >= 2
        && case.connected_components.len() == 1
        && case.articulation_points.is_empty()
}

#[allow(clippy::too_many_lines)]
fn assert_case(case: &BiconnectedFixtureCase) {
    assert!(!case.notes.is_empty(), "fixture case {} must explain its purpose", case.name);

    assert_eq!(
        case.edges,
        canonicalize_edge_list(case.edges.clone()),
        "fixture case {} must store canonical graph edges",
        case.name
    );

    for edge in &case.edges {
        assert!(
            edge[0] < case.node_count && edge[1] < case.node_count,
            "fixture case {} references an out-of-range vertex in edge {:?}",
            case.name,
            edge
        );
        assert_ne!(
            edge[0], edge[1],
            "fixture case {} should stay in the simple-graph regime for now",
            case.name
        );
    }

    let full_edge_set: BTreeSet<[usize; 2]> = case.edges.iter().copied().collect();
    let mut union_of_block_edges = BTreeSet::new();
    for component in &case.edge_biconnected_components {
        assert!(
            !component.is_empty(),
            "fixture case {} should not contain empty edge components",
            case.name
        );
        assert_eq!(
            component,
            &canonicalize_edge_list(component.clone()),
            "fixture case {} must store canonical block edges",
            case.name
        );
        for &edge in component {
            assert!(
                full_edge_set.contains(&edge),
                "fixture case {} block edge {:?} is not present in the graph",
                case.name,
                edge
            );
            assert!(
                union_of_block_edges.insert(edge),
                "fixture case {} block edge {:?} appears in more than one block",
                case.name,
                edge
            );
        }
    }
    assert_eq!(
        union_of_block_edges, full_edge_set,
        "fixture case {} must partition every graph edge into exactly one block",
        case.name
    );

    let derived_vertex_components: Vec<Vec<usize>> = case
        .edge_biconnected_components
        .iter()
        .map(|component| vertex_block_from_edges(component))
        .collect();
    assert_eq!(
        case.vertex_biconnected_components, derived_vertex_components,
        "fixture case {} stores vertex blocks that do not match the edge partition",
        case.name
    );

    let derived_bridges: Vec<[usize; 2]> = case
        .edge_biconnected_components
        .iter()
        .filter_map(|component| (component.len() == 1).then_some(component[0]))
        .collect();
    assert_eq!(
        case.bridges, derived_bridges,
        "fixture case {} stores bridges that do not match the one-edge blocks",
        case.name
    );

    let articulation_points =
        derived_articulation_points(case.node_count, &case.vertex_biconnected_components);
    assert_eq!(
        case.articulation_points, articulation_points,
        "fixture case {} stores articulation points inconsistent with block membership",
        case.name
    );

    let missing_vertices =
        derived_missing_vertices(case.node_count, &case.vertex_biconnected_components);
    assert_eq!(
        case.vertices_without_biconnected_component, missing_vertices,
        "fixture case {} stores the wrong omitted-vertex list",
        case.name
    );

    let cyclic_indices = derived_cyclic_indices(&case.edge_biconnected_components);
    assert_eq!(
        case.cyclic_biconnected_component_indices, cyclic_indices,
        "fixture case {} stores the wrong cyclic block indices",
        case.name
    );

    assert_eq!(
        case.is_biconnected,
        expected_is_biconnected(case),
        "fixture case {} stores the wrong graph-level biconnected flag",
        case.name
    );

    let graph = build_undigraph(case);
    assert_eq!(graph.number_of_nodes(), case.node_count, "case {}", case.name);

    let actual_connected_components = {
        let connected_components: ConnectedComponentsResult<'_, _, usize> =
            graph.connected_components().unwrap();
        (0..connected_components.number_of_components())
            .map(|component_id| {
                let mut vertices: Vec<usize> =
                    connected_components.node_ids_of_component(component_id).collect();
                vertices.sort_unstable();
                vertices
            })
            .collect::<BTreeSet<_>>()
    };
    let expected_connected_components = canonicalize_vertex_components(&case.connected_components);
    assert_eq!(
        actual_connected_components, expected_connected_components,
        "fixture case {} stores connected components inconsistent with the graph itself",
        case.name
    );

    let covered_vertices: Vec<usize> =
        case.connected_components.iter().flatten().copied().collect();
    assert_eq!(
        canonicalize_vertex_list(covered_vertices),
        (0..case.node_count).collect::<Vec<_>>(),
        "fixture case {} connected components must cover every vertex exactly once",
        case.name
    );
}

#[test]
fn test_biconnected_exhaustive_fixture_suite_header() {
    let suite = load_fixture_suite(EXHAUSTIVE_FIXTURE_NAME);
    assert_eq!(suite.schema_version, 1);
    assert_eq!(suite.algorithm, "hopcroft_tarjan_biconnected_components");
    assert_eq!(suite.graph_kind, "undirected_simple");
    assert_eq!(suite.generator, "exact_definition_exhaustive_labeled_graphs_order_5");
    assert_eq!(suite.primary_oracle, "edge_biconnected_components");
    assert!(suite.dyad_is_biconnected_component);
    assert!(!suite.isolated_vertices_form_biconnected_components);
    assert_eq!(suite.component_ordering, "lexicographic_by_smallest_normalized_edge");
    assert_eq!(suite.cases.len(), 1 << 10);

    let family_counts: BTreeSet<(&str, usize)> = [
        ("edge_count_0", 1),
        ("edge_count_1", 10),
        ("edge_count_2", 45),
        ("edge_count_3", 120),
        ("edge_count_4", 210),
        ("edge_count_5", 252),
        ("edge_count_6", 210),
        ("edge_count_7", 120),
        ("edge_count_8", 45),
        ("edge_count_9", 10),
        ("edge_count_10", 1),
    ]
    .into_iter()
    .collect();
    let actual_family_counts: BTreeSet<(&str, usize)> = suite
        .cases
        .iter()
        .fold(std::collections::BTreeMap::new(), |mut counts, case| {
            *counts.entry(case.family.as_str()).or_insert(0usize) += 1;
            counts
        })
        .into_iter()
        .collect();
    assert_eq!(actual_family_counts, family_counts);
    assert!(suite.cases.iter().all(|case| case.node_count == 5));
}

fn assert_cases_internal_consistency(cases: &[BiconnectedFixtureCase]) {
    for case in cases {
        assert_case(case);
    }
}

#[test]
fn test_biconnected_semantic_cases_internal_consistency() {
    let cases = semantic_cases();
    assert_eq!(cases.len(), 10);
    assert_cases_internal_consistency(&cases);
}

#[test]
fn test_biconnected_exhaustive_fixture_suite_internal_consistency() {
    let suite = load_fixture_suite(EXHAUSTIVE_FIXTURE_NAME);
    assert_cases_internal_consistency(&suite.cases);
}
