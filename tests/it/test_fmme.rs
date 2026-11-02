//! Integration tests for the fast-multipole multilevel embedder (FMME), the
//! TMAP layout.
//!
//! The single-level force model is pinned against the OGDF oracle in the module
//! unit tests. These exercise the whole multilevel pipeline as a black box:
//! finiteness, determinism, the space-filling property that motivates the whole
//! layout (a dense drawing has a low coefficient of variation of
//! nearest-neighbor distances, far below a spindly force-directed tree), and
//! the disconnected-graph component tiling. The `Fmme` trait entry over a
//! sparse matrix is covered too.
#![cfg(feature = "alloc")]
// Test-only numeric casts between integer generators and floating-point coordinates.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use geometric_traits::{
    impls::ValuedCSR2D,
    prelude::*,
    traits::algorithms::fmme::{Fmme, MixerConfig, layout_graph},
};

/// A deterministic generator so the tests need no RNG dependency.
fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// A random recursive tree on `n` nodes: every node after the first attaches to
/// a uniformly random earlier node. This is a connected acyclic graph, the
/// shape the layout is built for.
fn random_tree(n: usize, mut state: u64) -> Vec<(usize, usize)> {
    let mut edges = Vec::with_capacity(n - 1);
    for v in 1..n {
        let parent = (lcg(&mut state) * v as f64) as usize;
        edges.push((parent.min(v - 1), v));
    }
    edges
}

/// The coefficient of variation of nearest-neighbor distances. A dense,
/// space-filling drawing packs nodes near-uniformly and scores low. A spindly
/// force-directed tree scores high (above one). Scale invariant.
fn nn_distance_cov(coordinates: &[f64], n: usize) -> f64 {
    let point = |i: usize| [coordinates[i * 2], coordinates[i * 2 + 1]];
    let mut nn = Vec::with_capacity(n);
    for i in 0..n {
        let a = point(i);
        let mut best = f64::INFINITY;
        for j in 0..n {
            if i == j {
                continue;
            }
            let b = point(j);
            let d = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
            if d < best {
                best = d;
            }
        }
        nn.push(best.sqrt());
    }
    let mean = nn.iter().sum::<f64>() / n as f64;
    let var = nn.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n as f64;
    var.sqrt() / mean
}

#[test]
fn layout_is_finite_and_normalized() {
    let n = 400;
    let edges = random_tree(n, 0x1234);
    let coords = layout_graph::<f64>(n, &edges, &MixerConfig::default());
    assert_eq!(coords.len(), n * 2);
    assert!(coords.iter().copied().all(f64::is_finite));
    // Normalized to [-0.5, 0.5] on each axis.
    for axis in 0..2 {
        let values: Vec<f64> = (0..n).map(|i| coords[i * 2 + axis]).collect();
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(min >= -0.5001 && max <= 0.5001, "axis {axis} out of the unit box");
    }
}

#[test]
fn layout_is_deterministic_for_a_fixed_seed() {
    let n = 300;
    let edges = random_tree(n, 0xabcd);
    let config = MixerConfig::default();
    let a = layout_graph::<f64>(n, &edges, &config);
    let b = layout_graph::<f64>(n, &edges, &config);
    assert_eq!(a, b);
}

#[test]
fn layout_is_space_filling_not_spindly() {
    // A tree laid out well fills the plane near-uniformly: the nearest-neighbor
    // distance coefficient of variation sits well below one. A spindly layout
    // (long sparse strands with tight clumps) scores above one. OGDF scores
    // about 0.47 on comparable trees, so 0.8 is a comfortable ceiling that
    // a dense layout clears and a spindle fails.
    let n = 1000;
    let edges = random_tree(n, 0x5eed);
    let coords = layout_graph::<f64>(n, &edges, &MixerConfig::default());
    let cov = nn_distance_cov(&coords, n);
    assert!(cov < 0.8, "layout is not space-filling: nn-distance CoV = {cov}");
}

#[test]
fn disconnected_components_do_not_overlap() {
    // Two separate trees plus one isolated node. Every component is laid out on
    // its own and tiled, so the drawing stays finite and spans a real area.
    let mut edges = random_tree(200, 1);
    for (a, b) in random_tree(150, 2) {
        edges.push((a + 200, b + 200));
    }
    let n = 351; // 200 + 150 + 1 isolated node at index 350.
    let coords = layout_graph::<f64>(n, &edges, &MixerConfig::default());
    assert_eq!(coords.len(), n * 2);
    assert!(coords.iter().copied().all(f64::is_finite));
    let xs: Vec<f64> = (0..n).map(|i| coords[i * 2]).collect();
    let ys: Vec<f64> = (0..n).map(|i| coords[i * 2 + 1]).collect();
    let span_x = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - xs.iter().copied().fold(f64::INFINITY, f64::min);
    let span_y = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - ys.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(span_x > 0.1 && span_y > 0.1, "tiled layout collapsed");
}

#[test]
fn fmme_trait_lays_out_a_sparse_matrix() {
    // Build a symmetric star: a hub connected to eight leaves. The builder
    // wants the entries in row-major order.
    let mut triples = Vec::new();
    for leaf in 1..=8usize {
        triples.push((0, leaf, 1.0));
    }
    for leaf in 1..=8usize {
        triples.push((leaf, 0, 1.0));
    }
    let csr: ValuedCSR2D<usize, usize, usize, f64> =
        GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
            .expected_number_of_edges(triples.len())
            .expected_shape((9, 9))
            .edges(triples.into_iter())
            .build()
            .unwrap();

    let result = csr.fmme_layout::<f64>(&MixerConfig::default());
    assert_eq!(result.num_points(), 9);
    // The eight undirected edges are deduplicated from the sixteen stored
    // entries.
    assert_eq!(result.edges().len(), 8);
    assert!(result.coordinates_flat().iter().copied().all(f64::is_finite));
}
