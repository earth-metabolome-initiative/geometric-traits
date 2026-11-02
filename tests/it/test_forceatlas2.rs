//! Integration tests for the ForceAtlas2 layout: input validation, error
//! variants, initialization determinism and degenerate inputs.
#![cfg(feature = "std")]

use geometric_traits::{
    impls::ValuedCSR2D,
    prelude::*,
    traits::{ForceAtlas2Config, ForceAtlas2Error, ForceAtlas2Result},
};

type WeightedMatrix = ValuedCSR2D<usize, usize, usize, f64>;

/// Builds a symmetric weighted adjacency matrix from undirected edges.
fn build_undirected_weighted_graph(
    node_count: usize,
    undirected_edges: Vec<(usize, usize, f64)>,
) -> WeightedMatrix {
    let mut directed_edges = Vec::with_capacity(undirected_edges.len() * 2);
    for (source, destination, weight) in undirected_edges {
        directed_edges.push((source, destination, weight));
        if source != destination {
            directed_edges.push((destination, source, weight));
        }
    }
    directed_edges.sort_unstable_by(|(s1, d1, _), (s2, d2, _)| (s1, d1).cmp(&(s2, d2)));

    GenericEdgesBuilder::<_, WeightedMatrix>::default()
        .expected_number_of_edges(directed_edges.len())
        .expected_shape((node_count, node_count))
        .edges(directed_edges.into_iter())
        .build()
        .unwrap()
}

/// Builds a directed (possibly asymmetric) weighted adjacency matrix.
fn build_directed_weighted_graph(
    node_count: usize,
    mut directed_edges: Vec<(usize, usize, f64)>,
) -> WeightedMatrix {
    directed_edges.sort_unstable_by(|(s1, d1, _), (s2, d2, _)| (s1, d1).cmp(&(s2, d2)));
    GenericEdgesBuilder::<_, WeightedMatrix>::default()
        .expected_number_of_edges(directed_edges.len())
        .expected_shape((node_count, node_count))
        .edges(directed_edges.into_iter())
        .build()
        .unwrap()
}

fn triangle() -> WeightedMatrix {
    build_undirected_weighted_graph(3, vec![(0, 1, 1.0), (0, 2, 2.0), (1, 2, 1.0)])
}

// ============================================================================
// Configuration validation
// ============================================================================

#[test]
fn test_invalid_scaling_ratio() {
    let graph = triangle();
    for scaling_ratio in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let config = ForceAtlas2Config { scaling_ratio, ..Default::default() };
        assert_eq!(graph.force_atlas2(&config), Err(ForceAtlas2Error::InvalidScalingRatio));
    }
}

#[test]
fn test_invalid_gravity() {
    let graph = triangle();
    for gravity in [-1.0, f64::NAN, f64::INFINITY] {
        let config = ForceAtlas2Config { gravity, ..Default::default() };
        assert_eq!(graph.force_atlas2(&config), Err(ForceAtlas2Error::InvalidGravity));
    }
}

#[test]
fn test_zero_gravity_is_valid() {
    let graph = triangle();
    let config = ForceAtlas2Config { gravity: 0.0, ..Default::default() };
    assert!(graph.force_atlas2(&config).is_ok());
}

#[test]
fn test_invalid_jitter_tolerance() {
    let graph = triangle();
    for jitter_tolerance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let config = ForceAtlas2Config { jitter_tolerance, ..Default::default() };
        assert_eq!(graph.force_atlas2(&config), Err(ForceAtlas2Error::InvalidJitterTolerance));
    }
}

#[test]
fn test_invalid_edge_weight_influence() {
    let graph = triangle();
    for edge_weight_influence in [-1.0, f64::NAN, f64::INFINITY] {
        let config = ForceAtlas2Config { edge_weight_influence, ..Default::default() };
        assert_eq!(graph.force_atlas2(&config), Err(ForceAtlas2Error::InvalidEdgeWeightInfluence));
    }
}

#[test]
fn test_zero_edge_weight_influence_is_valid() {
    let graph = triangle();
    let config = ForceAtlas2Config { edge_weight_influence: 0.0, ..Default::default() };
    assert!(graph.force_atlas2(&config).is_ok());
}

// ============================================================================
// Matrix validation
// ============================================================================

#[test]
fn test_empty_graph() {
    let graph: WeightedMatrix = GenericEdgesBuilder::<_, WeightedMatrix>::default()
        .expected_number_of_edges(0)
        .expected_shape((0, 0))
        .edges(core::iter::empty())
        .build()
        .unwrap();
    assert_eq!(
        graph.force_atlas2(&ForceAtlas2Config::default()),
        Err(ForceAtlas2Error::EmptyGraph)
    );
}

#[test]
fn test_non_square_matrix() {
    let graph: WeightedMatrix = GenericEdgesBuilder::<_, WeightedMatrix>::default()
        .expected_number_of_edges(1)
        .expected_shape((2, 3))
        .edges(vec![(0, 1, 1.0)].into_iter())
        .build()
        .unwrap();
    assert_eq!(
        graph.force_atlas2(&ForceAtlas2Config::default()),
        Err(ForceAtlas2Error::NonSquareMatrix { rows: 2, columns: 3 })
    );
}

#[test]
fn test_asymmetric_matrix() {
    let graph = build_directed_weighted_graph(2, vec![(0, 1, 1.0)]);
    assert_eq!(
        graph.force_atlas2(&ForceAtlas2Config::default()),
        Err(ForceAtlas2Error::NonSymmetricEdge { source_id: 0, destination_id: 1 })
    );
}

#[test]
fn test_asymmetric_weights() {
    // Both directions exist but with different weights.
    let graph = build_directed_weighted_graph(2, vec![(0, 1, 1.0), (1, 0, 2.0)]);
    assert!(matches!(
        graph.force_atlas2(&ForceAtlas2Config::default()),
        Err(ForceAtlas2Error::NonSymmetricEdge { .. })
    ));
}

#[test]
fn test_non_finite_weight() {
    let graph = build_undirected_weighted_graph(2, vec![(0, 1, f64::NAN)]);
    assert!(matches!(
        graph.force_atlas2(&ForceAtlas2Config::default()),
        Err(ForceAtlas2Error::NonFiniteWeight { .. })
    ));
}

#[test]
fn test_negative_weight() {
    let graph = build_undirected_weighted_graph(2, vec![(0, 1, -1.0)]);
    assert!(matches!(
        graph.force_atlas2(&ForceAtlas2Config::default()),
        Err(ForceAtlas2Error::NegativeWeight { .. })
    ));
}

#[test]
fn test_zero_weight_is_valid() {
    let graph = build_undirected_weighted_graph(2, vec![(0, 1, 0.0)]);
    assert!(graph.force_atlas2(&ForceAtlas2Config::default()).is_ok());
}

// ============================================================================
// Initial positions
// ============================================================================

#[test]
fn test_initial_positions_length_mismatch() {
    let graph = triangle();
    let config = ForceAtlas2Config {
        initial_positions: Some(vec![[0.0, 0.0], [1.0, 0.0]]),
        ..Default::default()
    };
    assert_eq!(
        graph.force_atlas2(&config),
        Err(ForceAtlas2Error::InitialPositionsLengthMismatch { expected: 3, actual: 2 })
    );
}

#[test]
fn test_non_finite_initial_position() {
    let graph = triangle();
    let config = ForceAtlas2Config {
        initial_positions: Some(vec![[0.0, 0.0], [f64::NAN, 0.0], [1.0, 1.0]]),
        ..Default::default()
    };
    assert_eq!(
        graph.force_atlas2(&config),
        Err(ForceAtlas2Error::NonFiniteInitialPosition { index: 1 })
    );
}

#[test]
fn test_initial_positions_passthrough_with_zero_iterations() {
    let graph = triangle();
    let positions = vec![[0.5, -0.25], [1.0, 2.0], [-3.0, 0.125]];
    let config = ForceAtlas2Config {
        iterations: 0,
        initial_positions: Some(positions.clone()),
        ..Default::default()
    };
    let result = graph.force_atlas2(&config).unwrap();
    assert_eq!(result.iterations_run(), 0);
    for (index, expected) in positions.iter().enumerate() {
        assert_eq!(result.point(index), expected.as_slice());
    }
}

// ============================================================================
// Initialization determinism
// ============================================================================

#[test]
fn test_same_seed_same_layout() {
    let graph = triangle();
    let config = ForceAtlas2Config::default();
    let first = graph.force_atlas2(&config).unwrap();
    let second = graph.force_atlas2(&config).unwrap();
    assert_eq!(first, second);
}

#[test]
fn test_different_seed_different_layout() {
    let graph = triangle();
    let first = graph.force_atlas2(&ForceAtlas2Config { seed: 1, ..Default::default() }).unwrap();
    let second = graph.force_atlas2(&ForceAtlas2Config { seed: 2, ..Default::default() }).unwrap();
    assert_ne!(first.coordinates_flat(), second.coordinates_flat());
}

#[test]
fn test_random_initialization_is_finite_and_bounded() {
    let graph =
        build_undirected_weighted_graph(10, (0..9).map(|i| (i, i + 1, 1.0)).collect::<Vec<_>>());
    let config = ForceAtlas2Config { iterations: 0, ..Default::default() };
    let result = graph.force_atlas2(&config).unwrap();
    assert_eq!(result.num_points(), 10);
    for value in result.coordinates_flat() {
        assert!(value.is_finite());
        assert!((-0.5..0.5).contains(value));
    }
}

// ============================================================================
// Degenerate graphs
// ============================================================================

#[test]
fn test_single_node() {
    let graph = build_undirected_weighted_graph(1, vec![]);
    let result = graph.force_atlas2(&ForceAtlas2Config::default()).unwrap();
    assert_eq!(result.num_points(), 1);
    assert_eq!(result.dimensions(), 2);
    assert!(result.coordinates_flat().iter().copied().all(f64::is_finite));
}

#[test]
fn test_isolated_nodes_only() {
    // Five nodes, no edges at all.
    let graph = build_undirected_weighted_graph(5, vec![]);
    let result = graph.force_atlas2(&ForceAtlas2Config::default()).unwrap();
    assert_eq!(result.num_points(), 5);
    assert!(result.coordinates_flat().iter().copied().all(f64::is_finite));
}

// ============================================================================
// Core iteration end-to-end
// ============================================================================

fn distance(a: &[f64], b: &[f64]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Two connected nodes with gravity disabled settle at the distance where
/// attraction (e * d) balances repulsion (kr * m1 * m2 / d), i.e. d =
/// sqrt(kr * m1 * m2 / e) = sqrt(2 * 2 * 2 / 1) = 2 * sqrt(2).
#[test]
fn test_two_node_equilibrium() {
    let graph = build_undirected_weighted_graph(2, vec![(0, 1, 1.0)]);
    let config = ForceAtlas2Config {
        iterations: 500,
        gravity: 0.0,
        initial_positions: Some(vec![[0.0, 0.0], [1.0, 0.0]]),
        ..Default::default()
    };
    let result = graph.force_atlas2(&config).unwrap();
    let d = distance(result.point(0), result.point(1));
    let expected = 8.0_f64.sqrt();
    assert!(
        (d - expected).abs() / expected < 0.05,
        "expected equilibrium distance {expected}, found {d}"
    );
}

/// All forces are equivariant under rotations about the origin, so rotating
/// the initial positions rotates the final layout identically.
#[test]
fn test_rotation_equivariance() {
    let graph = build_undirected_weighted_graph(
        4,
        vec![(0, 1, 1.0), (1, 2, 2.0), (2, 3, 1.0), (3, 0, 0.5)],
    );
    let initial = vec![[0.1, 0.2], [-0.3, 0.4], [0.5, -0.1], [-0.2, -0.4]];
    let angle = 30.0_f64.to_radians();
    let (sin, cos) = angle.sin_cos();
    let rotate = |p: &[f64]| [p[0] * cos - p[1] * sin, p[0] * sin + p[1] * cos];
    let rotated: Vec<[f64; 2]> = initial.iter().map(|p| rotate(p.as_slice())).collect();

    let base_config = ForceAtlas2Config {
        iterations: 10,
        initial_positions: Some(initial),
        ..Default::default()
    };
    let rotated_config =
        ForceAtlas2Config { initial_positions: Some(rotated), ..base_config.clone() };

    let plain = graph.force_atlas2(&base_config).unwrap();
    let turned = graph.force_atlas2(&rotated_config).unwrap();

    for i in 0..4 {
        let expected = rotate(plain.point(i));
        assert!((turned.point(i)[0] - expected[0]).abs() < 1e-7);
        assert!((turned.point(i)[1] - expected[1]).abs() < 1e-7);
    }
}

/// Relabeling the nodes relabels the layout: the algorithm does not depend
/// on node identities (up to floating-point summation order).
#[test]
fn test_permutation_invariance() {
    // sigma maps old labels to new labels: 0 -> 2, 1 -> 0, 2 -> 1.
    let sigma = [2, 0, 1];
    let edges = vec![(0, 1, 1.0), (1, 2, 2.0), (0, 2, 0.5)];
    let permuted_edges: Vec<(usize, usize, f64)> =
        edges.iter().map(|&(u, v, w)| (sigma[u], sigma[v], w)).collect();

    let initial = vec![[0.1, 0.2], [-0.3, 0.4], [0.5, -0.1]];
    let mut permuted_initial = vec![[0.0; 2]; 3];
    for (old, &new) in sigma.iter().enumerate() {
        permuted_initial[new] = initial[old];
    }

    let graph = build_undirected_weighted_graph(3, edges);
    let permuted_graph = build_undirected_weighted_graph(3, permuted_edges);

    let config =
        ForceAtlas2Config { iterations: 5, initial_positions: Some(initial), ..Default::default() };
    let permuted_config =
        ForceAtlas2Config { initial_positions: Some(permuted_initial), ..config.clone() };

    let plain = graph.force_atlas2(&config).unwrap();
    let permuted = permuted_graph.force_atlas2(&permuted_config).unwrap();

    for (old, &new) in sigma.iter().enumerate() {
        assert!((plain.point(old)[0] - permuted.point(new)[0]).abs() < 1e-9);
        assert!((plain.point(old)[1] - permuted.point(new)[1]).abs() < 1e-9);
    }
}

/// Fully coincident initial positions stay finite (the zero-distance guards
/// suppress repulsion and attraction until gravity separates the masses).
#[test]
fn test_coincident_initial_positions_stay_finite() {
    let graph = build_undirected_weighted_graph(4, vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]);
    let config = ForceAtlas2Config {
        iterations: 50,
        initial_positions: Some(vec![[1.0, 1.0]; 4]),
        ..Default::default()
    };
    let result = graph.force_atlas2(&config).unwrap();
    assert!(result.coordinates_flat().iter().copied().all(f64::is_finite));
}

/// A path graph spreads out: the layout has nonzero extent, finite
/// coordinates and reports its run statistics.
#[test]
fn test_path_graph_layout_sanity() {
    let graph =
        build_undirected_weighted_graph(10, (0..9).map(|i| (i, i + 1, 1.0)).collect::<Vec<_>>());
    let config = ForceAtlas2Config { iterations: 100, ..Default::default() };
    let result = graph.force_atlas2(&config).unwrap();

    assert_eq!(result.iterations_run(), 100);
    assert!(result.coordinates_flat().iter().copied().all(f64::is_finite));
    assert!(result.final_swinging().is_finite());
    assert!(result.final_traction().is_finite());
    assert!(result.final_swinging() >= 0.0);
    assert!(result.final_traction() >= 0.0);

    // The repulsion must have spread the nodes apart.
    let max_pairwise = (0..10)
        .flat_map(|i| (0..i).map(move |j| (i, j)))
        .map(|(i, j)| distance(result.point(i), result.point(j)))
        .fold(0.0_f64, f64::max);
    assert!(max_pairwise > 1.0, "layout did not spread: max distance {max_pairwise}");
}

/// The star center stays surrounded: its hub ends near the barycenter of
/// the leaves.
#[test]
fn test_star_graph_center_is_central() {
    let graph =
        build_undirected_weighted_graph(7, (1..7).map(|leaf| (0, leaf, 1.0)).collect::<Vec<_>>());
    let config = ForceAtlas2Config { iterations: 200, ..Default::default() };
    let result = graph.force_atlas2(&config).unwrap();

    let center = result.point(0);
    let mut leaf_barycenter = [0.0_f64; 2];
    for leaf in 1..7 {
        leaf_barycenter[0] += result.point(leaf)[0] / 6.0;
        leaf_barycenter[1] += result.point(leaf)[1] / 6.0;
    }
    let center_offset = distance(center, &leaf_barycenter);
    let mean_leaf_distance: f64 =
        (1..7).map(|leaf| distance(center, result.point(leaf))).sum::<f64>() / 6.0;
    assert!(
        center_offset < mean_leaf_distance,
        "hub offset {center_offset} vs leaf distance {mean_leaf_distance}"
    );
}

// ============================================================================
// Feature modes (LinLog, dissuade hubs, edge weight influence)
// ============================================================================

/// With delta = 0 the weights are ignored entirely: a wildly weighted graph
/// lays out bit-identically to the same graph with unit weights.
#[test]
fn test_delta_zero_ignores_weights() {
    let edges = vec![(0, 1, 5.0), (1, 2, 0.3), (0, 2, 17.0), (2, 3, 0.001)];
    let unit_edges: Vec<(usize, usize, f64)> = edges.iter().map(|&(u, v, _)| (u, v, 1.0)).collect();
    let weighted = build_undirected_weighted_graph(4, edges);
    let unit = build_undirected_weighted_graph(4, unit_edges);

    let config =
        ForceAtlas2Config { iterations: 50, edge_weight_influence: 0.0, ..Default::default() };
    let a = weighted.force_atlas2(&config).unwrap();
    let b = unit.force_atlas2(&config).unwrap();
    assert_eq!(a.coordinates_flat(), b.coordinates_flat());
}

/// Unit weights are a fixed point of the delta exponent: delta = 1 and
/// delta = 2 agree bit-identically on an all-ones graph.
#[test]
fn test_delta_irrelevant_on_unit_weights() {
    let graph = build_undirected_weighted_graph(4, vec![(0, 1, 1.0), (1, 2, 1.0), (0, 3, 1.0)]);
    let linear =
        ForceAtlas2Config { iterations: 50, edge_weight_influence: 1.0, ..Default::default() };
    let squared = ForceAtlas2Config { edge_weight_influence: 2.0, ..linear.clone() };
    let a = graph.force_atlas2(&linear).unwrap();
    let b = graph.force_atlas2(&squared).unwrap();
    assert_eq!(a.coordinates_flat(), b.coordinates_flat());
}

/// Two nodes with weight 2 at delta = 2 attract with e = 4, settling at
/// d = sqrt(kr * m1 * m2 / e) = sqrt(2 * 2 * 2 / 4) = sqrt(2).
#[test]
fn test_delta_two_equilibrium() {
    let graph = build_undirected_weighted_graph(2, vec![(0, 1, 2.0)]);
    let config = ForceAtlas2Config {
        iterations: 500,
        gravity: 0.0,
        edge_weight_influence: 2.0,
        initial_positions: Some(vec![[0.0, 0.0], [1.0, 0.0]]),
        ..Default::default()
    };
    let result = graph.force_atlas2(&config).unwrap();
    let d = distance(result.point(0), result.point(1));
    let expected = 2.0_f64.sqrt();
    assert!(
        (d - expected).abs() / expected < 0.05,
        "expected equilibrium distance {expected}, found {d}"
    );
}

/// A zero-weight edge exerts no attraction: the pair drifts much farther
/// apart than a unit-weight pair (only gravity holds it together).
#[test]
fn test_zero_weight_edge_behaves_as_no_attraction() {
    let zero = build_undirected_weighted_graph(2, vec![(0, 1, 0.0)]);
    let unit = build_undirected_weighted_graph(2, vec![(0, 1, 1.0)]);
    let config = ForceAtlas2Config {
        iterations: 500,
        initial_positions: Some(vec![[-0.5, 0.0], [0.5, 0.0]]),
        ..Default::default()
    };
    let zero_result = zero.force_atlas2(&config).unwrap();
    let unit_result = unit.force_atlas2(&config).unwrap();
    let zero_distance = distance(zero_result.point(0), zero_result.point(1));
    let unit_distance = distance(unit_result.point(0), unit_result.point(1));
    assert!(
        zero_distance > 1.5 * unit_distance,
        "zero-weight pair at {zero_distance}, unit pair at {unit_distance}"
    );
}

/// On a regular graph all masses are equal, so the mean-mass compensation
/// cancels the mass division exactly: dissuade hubs is bit-identical to the
/// plain mode.
#[test]
fn test_dissuade_hubs_neutral_on_regular_graph() {
    // A 6-cycle: every node has degree 2, mass 3.
    let edges: Vec<(usize, usize, f64)> = (0..6).map(|i| (i, (i + 1) % 6, 1.0)).collect();
    let graph = build_undirected_weighted_graph(6, edges);
    let plain = ForceAtlas2Config { iterations: 50, ..Default::default() };
    let dissuaded = ForceAtlas2Config { dissuade_hubs: true, ..plain.clone() };
    let a = graph.force_atlas2(&plain).unwrap();
    let b = graph.force_atlas2(&dissuaded).unwrap();
    assert_eq!(a.coordinates_flat(), b.coordinates_flat());
}

/// On a star the masses differ, so dissuade hubs changes the layout.
#[test]
fn test_dissuade_hubs_changes_star_layout() {
    let graph =
        build_undirected_weighted_graph(7, (1..7).map(|leaf| (0, leaf, 1.0)).collect::<Vec<_>>());
    let plain = ForceAtlas2Config { iterations: 50, ..Default::default() };
    let dissuaded = ForceAtlas2Config { dissuade_hubs: true, ..plain.clone() };
    let a = graph.force_atlas2(&plain).unwrap();
    let b = graph.force_atlas2(&dissuaded).unwrap();
    assert_ne!(a.coordinates_flat(), b.coordinates_flat());
}

/// LinLog equilibrium for two connected nodes satisfies
/// d * ln(1 + d) = kr * m1 * m2 = 8.
#[test]
fn test_lin_log_two_node_equilibrium() {
    let graph = build_undirected_weighted_graph(2, vec![(0, 1, 1.0)]);
    let config = ForceAtlas2Config {
        iterations: 500,
        gravity: 0.0,
        lin_log: true,
        initial_positions: Some(vec![[0.0, 0.0], [1.0, 0.0]]),
        ..Default::default()
    };
    let result = graph.force_atlas2(&config).unwrap();
    let d = distance(result.point(0), result.point(1));
    let balance = d * (1.0 + d).ln();
    assert!(
        (balance - 8.0).abs() / 8.0 < 0.05,
        "expected d * ln(1 + d) = 8 at equilibrium, found {balance} (d = {d})"
    );
}

/// All modes composed (LinLog + dissuade hubs + delta = 2) run NaN-free and
/// deterministically on a weighted wheel graph.
#[test]
#[allow(clippy::cast_precision_loss)]
fn test_mode_composition_smoke() {
    // Wheel: hub 0 connected to a 12-cycle, deterministic varied weights.
    let mut edges = Vec::new();
    for i in 1..13_usize {
        let next = if i == 12 { 1 } else { i + 1 };
        edges.push((0, i, 0.5 + ((i * 3) % 7) as f64 * 0.5));
        edges.push((i, next, 0.5 + ((i * 5 + next) % 11) as f64 * 0.25));
    }
    let graph = build_undirected_weighted_graph(13, edges);
    let config = ForceAtlas2Config {
        iterations: 100,
        lin_log: true,
        dissuade_hubs: true,
        edge_weight_influence: 2.0,
        ..Default::default()
    };
    let first = graph.force_atlas2(&config).unwrap();
    let second = graph.force_atlas2(&config).unwrap();
    assert!(first.coordinates_flat().iter().copied().all(f64::is_finite));
    assert_eq!(first, second);
}

/// Strong gravity compacts the layout: the mean distance from the origin
/// shrinks compared to normal gravity.
#[test]
fn test_strong_gravity_compacts_layout() {
    let graph =
        build_undirected_weighted_graph(10, (0..9).map(|i| (i, i + 1, 1.0)).collect::<Vec<_>>());
    let normal = ForceAtlas2Config { iterations: 200, ..Default::default() };
    let strong = ForceAtlas2Config { strong_gravity: true, ..normal.clone() };
    let mean_radius = |result: &ForceAtlas2Result| {
        (0..10).map(|i| distance(result.point(i), &[0.0, 0.0])).sum::<f64>() / 10.0
    };
    let normal_result = graph.force_atlas2(&normal).unwrap();
    let strong_result = graph.force_atlas2(&strong).unwrap();
    assert!(
        mean_radius(&strong_result) < mean_radius(&normal_result),
        "strong gravity should compact the layout"
    );
}

// ============================================================================
// Barnes-Hut
// ============================================================================

#[test]
fn test_invalid_barnes_hut_theta() {
    let graph = triangle();
    for barnes_hut_theta in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let config = ForceAtlas2Config { barnes_hut_theta, ..Default::default() };
        assert_eq!(graph.force_atlas2(&config), Err(ForceAtlas2Error::InvalidBarnesHutTheta));
    }
}

/// Builds a ring of `n` nodes with deterministic long-range chords, a
/// simple connected graph with non-trivial structure for BH testing.
fn ring_with_chords(n: usize) -> WeightedMatrix {
    let mut edges: Vec<(usize, usize, f64)> = (0..n).map(|i| (i, (i + 1) % n, 1.0)).collect();
    for i in (0..n).step_by(5) {
        let j = (i + 97) % n;
        edges.push((i.min(j), i.max(j), 1.0));
    }
    edges.sort_unstable_by_key(|edge| (edge.0, edge.1));
    edges.dedup_by_key(|&mut (u, v, _)| (u, v));
    build_undirected_weighted_graph(n, edges)
}

/// With a tiny theta the quadtree opens every region down to the leaves,
/// so Barnes-Hut equals the exact backend up to floating-point summation
/// order.
#[test]
fn test_barnes_hut_tiny_theta_matches_exact() {
    let graph = ring_with_chords(50);
    let exact_config = ForceAtlas2Config { iterations: 25, ..Default::default() };
    let bh_config =
        ForceAtlas2Config { barnes_hut: true, barnes_hut_theta: 1e-12, ..exact_config.clone() };
    let exact = graph.force_atlas2(&exact_config).unwrap();
    let bh = graph.force_atlas2(&bh_config).unwrap();
    for node in 0..50 {
        let d = distance(exact.point(node), bh.point(node));
        assert!(d < 1e-6, "node {node} diverged by {d}");
    }
}

/// At the default theta (1.2, coarse by design) the approximation perturbs
/// positions but preserves the layout quality: Noack normalized edge length
/// stays within 15 percent of the exact backend (observed gap ~10 percent,
/// with the approximated layout actually scoring slightly better). The
/// numeric fidelity of the quadtree itself is covered bit-exactly by the
/// fa2 oracle corpus.
#[test]
fn test_barnes_hut_default_theta_preserves_quality() {
    let n = 200;
    let graph = ring_with_chords(n);
    let exact_config = ForceAtlas2Config { iterations: 100, ..Default::default() };
    let bh_config = ForceAtlas2Config { barnes_hut: true, ..exact_config.clone() };
    let exact = graph.force_atlas2(&exact_config).unwrap();
    let bh = graph.force_atlas2(&bh_config).unwrap();

    let quality = |result: &ForceAtlas2Result| {
        let mut mean_edge = 0.0;
        let mut edge_count = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            mean_edge += distance(result.point(i), result.point(j));
            edge_count += 1.0;
        }
        mean_edge /= edge_count;
        let mut mean_pairwise = 0.0;
        let mut pair_count = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                mean_pairwise += distance(result.point(i), result.point(j));
                pair_count += 1.0;
            }
        }
        mean_pairwise /= pair_count;
        mean_edge / mean_pairwise
    };

    let exact_quality = quality(&exact);
    let bh_quality = quality(&bh);
    let gap = (exact_quality - bh_quality).abs() / exact_quality;
    assert!(gap < 0.15, "quality gap {gap} (exact {exact_quality}, bh {bh_quality})");
}

/// Degenerate initial positions (1000 nodes piled on four points) must not
/// panic, overflow the stack or produce non-finite coordinates with
/// Barnes-Hut enabled.
#[test]
fn test_barnes_hut_degenerate_positions_robustness() {
    let n = 1000;
    let graph = ring_with_chords(n);
    let clusters = [[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]];
    let positions: Vec<[f64; 2]> = (0..n).map(|i| clusters[i % 4]).collect();
    let config = ForceAtlas2Config {
        iterations: 50,
        barnes_hut: true,
        initial_positions: Some(positions),
        ..Default::default()
    };
    let result = graph.force_atlas2(&config).unwrap();
    assert!(result.coordinates_flat().iter().copied().all(f64::is_finite));
}

/// Barnes-Hut with the default theta on a graph below Gephi's automatic
/// threshold still produces a sane, deterministic layout.
#[test]
fn test_barnes_hut_deterministic() {
    let graph = ring_with_chords(100);
    let config = ForceAtlas2Config { iterations: 50, barnes_hut: true, ..Default::default() };
    let first = graph.force_atlas2(&config).unwrap();
    let second = graph.force_atlas2(&config).unwrap();
    assert_eq!(first, second);
}

// ============================================================================
// Anti-collision (prevent overlap / adjustSizes)
// ============================================================================

#[test]
fn test_node_sizes_length_mismatch() {
    let graph = triangle();
    let config = ForceAtlas2Config { node_sizes: Some(vec![1.0, 1.0]), ..Default::default() };
    assert_eq!(
        graph.force_atlas2(&config),
        Err(ForceAtlas2Error::NodeSizesLengthMismatch { expected: 3, actual: 2 })
    );
}

#[test]
fn test_invalid_node_size() {
    let graph = triangle();
    for bad in [f64::NAN, f64::INFINITY, -1.0] {
        let config =
            ForceAtlas2Config { node_sizes: Some(vec![1.0, bad, 1.0]), ..Default::default() };
        assert_eq!(
            graph.force_atlas2(&config),
            Err(ForceAtlas2Error::InvalidNodeSize { index: 1 })
        );
    }
}

/// Counts node pairs whose disks overlap (center distance below the size
/// sum).
fn overlapping_pairs(result: &ForceAtlas2Result, sizes: &[f64]) -> usize {
    let n = sizes.len();
    let mut count = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            if distance(result.point(i), result.point(j)) < sizes[i] + sizes[j] {
                count += 1;
            }
        }
    }
    count
}

/// Starting from a tight cluster, the anti-collision mode pushes nearly
/// every pair apart: the number of overlapping pairs collapses.
#[test]
fn test_prevent_overlap_resolves_cluster() {
    let n = 30;
    let graph =
        build_undirected_weighted_graph(n, (0..n - 1).map(|i| (i, i + 1, 1.0)).collect::<Vec<_>>());
    let sizes = vec![0.5; n];
    // Tight grid of positions well inside each other's radii.
    #[allow(clippy::cast_precision_loss)]
    let positions: Vec<[f64; 2]> =
        (0..n).map(|i| [(i % 6) as f64 * 0.1, (i / 6) as f64 * 0.1]).collect();

    let initially_overlapping = {
        let config = ForceAtlas2Config {
            iterations: 0,
            initial_positions: Some(positions.clone()),
            ..Default::default()
        };
        overlapping_pairs(&graph.force_atlas2(&config).unwrap(), &sizes)
    };
    assert!(initially_overlapping > 100, "the start must be heavily overlapped");

    let config = ForceAtlas2Config {
        iterations: 500,
        node_sizes: Some(sizes.clone()),
        initial_positions: Some(positions),
        ..Default::default()
    };
    let result = graph.force_atlas2(&config).unwrap();
    assert!(result.coordinates_flat().iter().copied().all(f64::is_finite));
    let remaining = overlapping_pairs(&result, &sizes);
    assert!(
        remaining * 10 < initially_overlapping,
        "overlaps did not collapse: {initially_overlapping} -> {remaining}"
    );
}

/// Zero sizes keep the anti-collision machinery (gating, 0.1 speed,
/// displacement cap) but degenerate the border distance to the center
/// distance, the layout stays finite and spread.
#[test]
fn test_prevent_overlap_zero_sizes_is_sane() {
    let graph =
        build_undirected_weighted_graph(5, (0..4).map(|i| (i, i + 1, 1.0)).collect::<Vec<_>>());
    let config =
        ForceAtlas2Config { iterations: 200, node_sizes: Some(vec![0.0; 5]), ..Default::default() };
    let result = graph.force_atlas2(&config).unwrap();
    assert!(result.coordinates_flat().iter().copied().all(f64::is_finite));
    assert!(distance(result.point(0), result.point(4)) > 0.1);
}

/// Huge sizes mean permanent overlap and a constant repulsion kick, the
/// displacement cap keeps every step bounded and the layout finite.
#[test]
fn test_prevent_overlap_huge_sizes_stay_finite() {
    let graph = triangle();
    let config =
        ForceAtlas2Config { iterations: 100, node_sizes: Some(vec![1e6; 3]), ..Default::default() };
    let result = graph.force_atlas2(&config).unwrap();
    assert!(result.coordinates_flat().iter().copied().all(f64::is_finite));
}

/// A single node in anti-collision mode has zero net force, exercising the
/// df == 0 guard of the capped position update: it must not move (and in
/// particular must not become NaN as in the Java source).
#[test]
fn test_prevent_overlap_single_node_stays_put() {
    let graph = build_undirected_weighted_graph(1, vec![]);
    let position = [0.25, -0.75];
    let config = ForceAtlas2Config {
        iterations: 50,
        gravity: 0.0,
        node_sizes: Some(vec![1.0]),
        initial_positions: Some(vec![position]),
        ..Default::default()
    };
    let result = graph.force_atlas2(&config).unwrap();
    assert_eq!(result.point(0), position.as_slice());
}

/// Anti-collision composes with Barnes-Hut (sizes at the exact leaf level,
/// plain approximation for regions): finite and deterministic.
#[test]
fn test_prevent_overlap_with_barnes_hut() {
    let n = 200;
    let graph = ring_with_chords(n);
    let config = ForceAtlas2Config {
        iterations: 100,
        barnes_hut: true,
        node_sizes: Some(vec![0.5; n]),
        ..Default::default()
    };
    let first = graph.force_atlas2(&config).unwrap();
    let second = graph.force_atlas2(&config).unwrap();
    assert!(first.coordinates_flat().iter().copied().all(f64::is_finite));
    assert_eq!(first, second);
}

// ============================================================================
// Fuzzer-found regressions
// ============================================================================

/// Fuzzer regression: with one weight near f64::MAX, a naive symmetry
/// tolerance (`scale * 16.0 * EPSILON` evaluated left to right) overflows
/// to infinity and accepts ANY weight pair as symmetric. The wildly
/// asymmetric pair below must be rejected.
#[test]
fn test_regression_huge_weight_asymmetry_is_rejected() {
    let graph = build_directed_weighted_graph(
        96,
        vec![(59, 95, 2.527457358493282e307), (95, 59, 1.656712100688973e-24)],
    );
    assert!(matches!(
        graph.force_atlas2(&ForceAtlas2Config::default()),
        Err(ForceAtlas2Error::NonSymmetricEdge { .. })
    ));
}

/// Fuzzer regression: a symmetric weight near f64::MAX overflows the
/// attraction force (and squares to infinity at delta = 2). Coordinates
/// and the reported run statistics must stay finite regardless.
#[test]
fn test_regression_huge_weight_stays_finite() {
    let graph = build_undirected_weighted_graph(3, vec![(0, 1, 2.5e307), (1, 2, 1.0)]);
    for edge_weight_influence in [1.0, 2.0] {
        let config =
            ForceAtlas2Config { iterations: 10, edge_weight_influence, ..Default::default() };
        let result = graph.force_atlas2(&config).unwrap();
        assert!(result.coordinates_flat().iter().copied().all(f64::is_finite));
        assert!(result.final_swinging().is_finite());
        assert!(result.final_traction().is_finite());
        assert!(result.final_swinging() >= 0.0);
        assert!(result.final_traction() >= 0.0);
    }
}

/// The anti-collision displacement cap bounds every node to 10 units per
/// iteration, even under the huge overlap kick.
#[test]
fn test_prevent_overlap_displacement_capped_at_ten() {
    // Two heavily overlapping nodes 50 units apart: the overlap kick is
    // 100 * kr * m1 * m2 = 800 on the center vector, a force of 40000,
    // whose damped displacement (about 14) exceeds the cap.
    let graph = build_undirected_weighted_graph(2, vec![(0, 1, 1.0)]);
    let initial = [[0.0, 0.0], [50.0, 0.0]];
    let config = ForceAtlas2Config {
        iterations: 1,
        node_sizes: Some(vec![100.0, 100.0]),
        initial_positions: Some(initial.to_vec()),
        ..Default::default()
    };
    let result = graph.force_atlas2(&config).unwrap();
    for (node, start) in initial.iter().enumerate() {
        let moved = distance(result.point(node), start);
        assert!((moved - 10.0).abs() < 1e-9, "node {node} moved {moved}, expected the 10 cap");
    }
}

/// All anti-collision attraction variants compose with LinLog and dissuade
/// hubs: finite and deterministic.
#[test]
fn test_prevent_overlap_mode_composition() {
    let graph = build_undirected_weighted_graph(
        7,
        #[allow(clippy::cast_precision_loss)]
        (1..7).map(|leaf| (0, leaf, 1.0 + leaf as f64 * 0.25)).collect::<Vec<_>>(),
    );
    for (lin_log, dissuade_hubs) in [(true, false), (false, true), (true, true)] {
        let config = ForceAtlas2Config {
            iterations: 100,
            lin_log,
            dissuade_hubs,
            node_sizes: Some(vec![0.5; 7]),
            ..Default::default()
        };
        let first = graph.force_atlas2(&config).unwrap();
        let second = graph.force_atlas2(&config).unwrap();
        assert!(first.coordinates_flat().iter().copied().all(f64::is_finite));
        assert_eq!(first, second);
    }
}

/// A fractional delta exercises the generic pow path: the equilibrium of a
/// two-node system with weight 4 and delta = 0.5 satisfies
/// d = sqrt(kr * m1 * m2 / 4^0.5) = sqrt(8 / 2) = 2.
#[test]
fn test_delta_fractional_equilibrium() {
    let graph = build_undirected_weighted_graph(2, vec![(0, 1, 4.0)]);
    let config = ForceAtlas2Config {
        iterations: 500,
        gravity: 0.0,
        edge_weight_influence: 0.5,
        initial_positions: Some(vec![[0.0, 0.0], [1.0, 0.0]]),
        ..Default::default()
    };
    let result = graph.force_atlas2(&config).unwrap();
    let d = distance(result.point(0), result.point(1));
    assert!((d - 2.0).abs() / 2.0 < 0.05, "expected equilibrium distance 2, found {d}");
}

/// The callback fires once per iteration, in order, with `done` from 1 to the
/// total and a constant total.
#[test]
fn test_progress_callback_reports_every_iteration() {
    let graph = build_undirected_weighted_graph(4, vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]);
    let config = ForceAtlas2Config { iterations: 17, ..Default::default() };

    let mut observed: Vec<(usize, usize)> = Vec::new();
    graph
        .force_atlas2_with_progress(&config, &mut |done, total| observed.push((done, total)))
        .unwrap();

    assert_eq!(observed, (1..=17).map(|done| (done, 17)).collect::<Vec<_>>());
}

/// Observing the loop does not perturb the result.
#[test]
fn test_progress_variant_matches_plain_result() {
    let graph = build_undirected_weighted_graph(
        6,
        vec![(0, 1, 1.0), (1, 2, 2.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 3.0), (5, 0, 1.0)],
    );
    let config = ForceAtlas2Config { iterations: 60, ..Default::default() };

    let plain = graph.force_atlas2(&config).unwrap();
    let with_progress = graph.force_atlas2_with_progress(&config, &mut |_, _| {}).unwrap();

    assert_eq!(plain, with_progress);
}
