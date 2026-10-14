//! Cross-validation of the ForceAtlas2 layout against two independent
//! oracles: the Python `fa2` package (v1.1.2, a Gephi-faithful port
//! computing in pure f64) and the canonical Java implementation itself
//! (gephi-toolkit 0.10.1, single-threaded, exact repulsion).
//!
//! The fixtures are one-shot recordings of those oracles (the generator
//! scripts are not part of the repository, the embedded `generator` and
//! `oracle` metadata strings name them for provenance). The `fa2` corpus
//! holds position snapshots after 1, 5, 25 and 100 iterations for 78 cases
//! spanning graph shapes, sizes (2 to 200 nodes), weights and all mode
//! combinations including Barnes-Hut. The toolkit anchor set holds 14
//! cases, including the four adjustSizes cases.
//!
//! # Oracle quirks baked into the fixtures
//!
//! - fa2's strong gravity multiplies by the scaling ratio where Gephi nets out
//!   to `mass * gravity`, so its strong-gravity cases were recorded with
//!   `gravity / scaling` passed to fa2 while the stored settings carry the real
//!   value.
//! - fa2's adjustSizes position update deviates from Gephi master (it caps the
//!   factor at `10 / size` instead of the 0.1 speed factor plus the
//!   displacement cap), so adjustSizes ground truth comes exclusively from the
//!   toolkit anchor set.
//! - fa2 derives edges from matrix nonzeros, so the corpus avoids zero-weight
//!   edges and self-loops.
//!
//! # Tolerance protocol
//!
//! The `fa2` oracle computes in pure f64 with identical operation order, so
//! early iterations must agree almost bit-exactly. Later iterations
//! accumulate floating-point divergence through the chaotic dynamics (tiny
//! ULP differences in `ln`/`pow` grow exponentially), so the bounds widen
//! with the iteration count. Errors are measured per node as the Euclidean
//! distance to the oracle position, normalized by the oracle layout scale
//! (the root mean square node distance from the layout barycenter).
#![cfg(feature = "std")]

#[path = "support/fixture_io.rs"]
mod fixture_io;

use std::collections::HashMap;

use fixture_io::load_fixture_json;
use geometric_traits::{
    impls::ValuedCSR2D,
    prelude::*,
    traits::{ForceAtlas2Config, ForceAtlas2Result},
};
use serde::Deserialize;

type WeightedMatrix = ValuedCSR2D<usize, usize, usize, f64>;

#[derive(Debug, Deserialize)]
struct Fixture {
    snapshot_iterations: Vec<usize>,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    n: usize,
    edges: Vec<(usize, usize, f64)>,
    settings: Settings,
    initial_positions: Vec<[f64; 2]>,
    /// Node radii for the anti-collision (adjustSizes) cases.
    #[serde(default)]
    node_sizes: Option<Vec<f64>>,
    snapshots: HashMap<String, Vec<[f64; 2]>>,
}

#[derive(Debug, Deserialize)]
struct Settings {
    scaling_ratio: f64,
    gravity: f64,
    strong_gravity: bool,
    jitter_tolerance: f64,
    edge_weight_influence: f64,
    lin_log: bool,
    dissuade_hubs: bool,
    barnes_hut: bool,
    barnes_hut_theta: f64,
    /// Only present in the Gephi toolkit fixtures.
    #[serde(default)]
    adjust_sizes: bool,
}

/// Maximum allowed normalized per-node error per snapshot iteration.
///
/// Calibrated against the observed drift: the corpus maximum is below
/// 1e-13 at iteration 1, 1e-10 at iteration 5 and 4e-5 at iteration 25.
/// At iteration 100 a handful of larger or stiffer cases (200 nodes,
/// scaling 10, LinLog combinations) drift up to ~0.17 of the layout scale
/// through pure floating-point divergence, those fall back to the quality
/// comparison below.
fn tolerance(iterations: usize) -> f64 {
    match iterations {
        1 => 1e-12,
        5 => 1e-9,
        25 => 1e-4,
        _ => 2e-2,
    }
}

/// Noack's normalized edge length (paper eq 15): mean edge length divided
/// by mean pairwise node distance. Lower is better. Robust to the node
/// displacements caused by chaotic floating-point divergence, which is why
/// it backs up the positional comparison at iteration 100 (see the module
/// docs).
fn normalized_edge_length(positions: &[[f64; 2]], edges: &[(usize, usize, f64)]) -> f64 {
    let distance =
        |a: [f64; 2], b: [f64; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    #[allow(clippy::cast_precision_loss)]
    let mean_edge =
        edges.iter().map(|&(u, v, _)| distance(positions[u], positions[v])).sum::<f64>()
            / edges.len() as f64;
    let n = positions.len();
    let mut total = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            total += distance(positions[i], positions[j]);
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let mean_pairwise = total / (n * (n - 1) / 2) as f64;
    mean_edge / mean_pairwise
}

fn build_matrix(case: &Case) -> WeightedMatrix {
    let mut directed_edges = Vec::with_capacity(case.edges.len() * 2);
    for &(source, destination, weight) in &case.edges {
        directed_edges.push((source, destination, weight));
        directed_edges.push((destination, source, weight));
    }
    directed_edges.sort_unstable_by(|(s1, d1, _), (s2, d2, _)| (s1, d1).cmp(&(s2, d2)));
    GenericEdgesBuilder::<_, WeightedMatrix>::default()
        .expected_number_of_edges(directed_edges.len())
        .expected_shape((case.n, case.n))
        .edges(directed_edges.into_iter())
        .build()
        .unwrap()
}

fn config_for(case: &Case, iterations: usize) -> ForceAtlas2Config {
    ForceAtlas2Config {
        iterations,
        scaling_ratio: case.settings.scaling_ratio,
        gravity: case.settings.gravity,
        strong_gravity: case.settings.strong_gravity,
        jitter_tolerance: case.settings.jitter_tolerance,
        edge_weight_influence: case.settings.edge_weight_influence,
        lin_log: case.settings.lin_log,
        dissuade_hubs: case.settings.dissuade_hubs,
        barnes_hut: case.settings.barnes_hut,
        barnes_hut_theta: case.settings.barnes_hut_theta,
        node_sizes: if case.settings.adjust_sizes { case.node_sizes.clone() } else { None },
        seed: 0,
        initial_positions: Some(case.initial_positions.clone()),
    }
}

/// Root mean square distance of the oracle positions from their
/// barycenter, used to normalize the error.
fn layout_scale(positions: &[[f64; 2]]) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let n = positions.len() as f64;
    let mean_x = positions.iter().map(|p| p[0]).sum::<f64>() / n;
    let mean_y = positions.iter().map(|p| p[1]).sum::<f64>() / n;
    let mean_square =
        positions.iter().map(|p| (p[0] - mean_x).powi(2) + (p[1] - mean_y).powi(2)).sum::<f64>()
            / n;
    mean_square.sqrt().max(1e-9)
}

fn max_normalized_error(result: &ForceAtlas2Result, oracle: &[[f64; 2]]) -> f64 {
    let scale = layout_scale(oracle);
    oracle
        .iter()
        .enumerate()
        .map(|(node, expected)| {
            let actual = result.point(node);
            ((actual[0] - expected[0]).powi(2) + (actual[1] - expected[1]).powi(2)).sqrt() / scale
        })
        .fold(0.0_f64, f64::max)
}

/// Tolerances for the Gephi toolkit anchor set.
///
/// Gephi stores coordinates as `f32` and truncates most intermediate
/// distances to `f32`, so each iteration injects roughly 1e-7 relative
/// noise that then amplifies. Iteration 1 confirms the semantics at the
/// `f32` noise floor (observed corpus maximum 1.1e-7), deeper snapshots
/// widen with the observed drift (4.5e-6 at 5, 3.3e-5 at 25) and
/// iteration 100 also falls back to the quality comparison.
fn gephi_tolerance(iterations: usize) -> f64 {
    match iterations {
        1 => 1e-6,
        5 => 1e-4,
        25 => 1e-3,
        _ => 5e-1,
    }
}

/// Validates a corpus, returning human-readable failure descriptions.
///
/// `quality_fallback_from` is the snapshot iteration from which a
/// positional miss may fall back to the quality comparison. The fa2 corpus
/// computes in f64 like we do, so only iteration 100 qualifies. The Gephi
/// corpus carries f32 truncation noise that the discontinuous
/// anti-collision branches amplify within a few iterations, so there the
/// fallback opens at iteration 5 (iteration 1 stays strictly positional in
/// both corpora, it is the semantic gate).
fn check_corpus(
    fixture: &Fixture,
    tolerance_of: fn(usize) -> f64,
    quality_fallback_from: usize,
) -> Vec<String> {
    let mut failures = Vec::new();
    for case in &fixture.cases {
        let matrix = build_matrix(case);
        for &iterations in &fixture.snapshot_iterations {
            let oracle = &case.snapshots[&iterations.to_string()];
            let result = matrix.force_atlas2(&config_for(case, iterations)).unwrap();
            let error = max_normalized_error(&result, oracle);
            if error > tolerance_of(iterations) {
                // Chaotic-divergence fallback, only legitimate at the
                // deepest snapshot: the layouts must still be equally good
                // by Noack's quality measure.
                let ours: Vec<[f64; 2]> = (0..case.n)
                    .map(|node| [result.point(node)[0], result.point(node)[1]])
                    .collect();
                let our_quality = normalized_edge_length(&ours, &case.edges);
                let oracle_quality = normalized_edge_length(oracle, &case.edges);
                let quality_gap = (our_quality - oracle_quality).abs() / oracle_quality;
                if iterations < quality_fallback_from || quality_gap > 0.03 {
                    failures.push(format!(
                        "{name} @ {iterations} iterations: error {error:.3e} > {tol:.0e}, \
                         quality gap {quality_gap:.3e}",
                        name = case.name,
                        tol = tolerance_of(iterations),
                    ));
                }
            }
        }
    }
    failures
}

#[test]
fn test_fa2_ground_truth_corpus() {
    let fixture: Fixture = load_fixture_json("forceatlas2_fa2_ground_truth.json.gz");
    assert!(!fixture.cases.is_empty());
    let failures = check_corpus(&fixture, tolerance, 100);
    assert!(
        failures.is_empty(),
        "{} of {} case-snapshots exceeded tolerance:\n{}",
        failures.len(),
        fixture.cases.len() * fixture.snapshot_iterations.len(),
        failures.join("\n")
    );
}

/// Anchor set generated by the canonical Java implementation itself
/// (gephi-toolkit 0.10.1, single-threaded, exact repulsion).
#[test]
fn test_gephi_ground_truth_anchor() {
    let fixture: Fixture = load_fixture_json("forceatlas2_gephi_ground_truth.json.gz");
    assert_eq!(fixture.cases.len(), 14);
    let failures = check_corpus(&fixture, gephi_tolerance, 5);
    assert!(
        failures.is_empty(),
        "{} of {} case-snapshots exceeded tolerance:\n{}",
        failures.len(),
        fixture.cases.len() * fixture.snapshot_iterations.len(),
        failures.join("\n")
    );
}
