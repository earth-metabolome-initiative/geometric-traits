//! Testing utilities for constructing type instances from raw bytes and
//! replaying fuzz corpus or crash files, and shared invariant-checking
//! functions used by both fuzz targets and regression tests.
//!
//! This module is available when the `arbitrary` feature is enabled. It
//! provides helpers used by both fuzz targets and regression tests, so
//! crash files produced by fuzzing can be directly replayed as unit tests.

use alloc::{
    collections::BTreeMap,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt::Debug;

use arbitrary::{Arbitrary, Unstructured};
use bitvec::vec::BitVec;
use num_traits::AsPrimitive;

use crate::{
    impls::{CSR2D, SymmetricCSR2D, ValuedCSR2D, VecMatrix2D},
    prelude::*,
    traits::{
        DenseValuedMatrix2D, EdgesBuilder, SparseMatrix, SparseMatrix2D, SparseSquareMatrix,
        SparseValuedMatrix, SparseValuedMatrix2D, VocabularyBuilder,
        algorithms::randomized_graphs::{
            barabasi_albert, chung_lu, erdos_renyi_gnm, erdos_renyi_gnp, random_geometric_graph,
            watts_strogatz,
        },
    },
};

// ============================================================================
// Deserialization helpers
// ============================================================================

/// Construct a value of type `T` from raw bytes using the [`Arbitrary`] trait.
///
/// Returns `None` if the bytes are insufficient or do not produce a valid
/// instance.
#[must_use]
#[inline]
pub fn from_bytes<T: for<'a> Arbitrary<'a>>(bytes: &[u8]) -> Option<T> {
    let mut u = Unstructured::new(bytes);
    T::arbitrary(&mut u).ok()
}

/// Load all files from a directory and construct instances of `T` from each
/// file's raw bytes.
///
/// Files that fail to produce valid instances are silently skipped.
/// Returns an empty vector if the directory does not exist or is unreadable.
#[must_use]
#[inline]
pub fn replay_dir<T: for<'a> Arbitrary<'a>>(dir: &std::path::Path) -> Vec<T> {
    let mut results = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return results;
    };
    for entry in entries.flatten() {
        if entry.path().is_file() {
            if let Ok(bytes) = std::fs::read(entry.path()) {
                if let Some(instance) = from_bytes::<T>(&bytes) {
                    results.push(instance);
                }
            }
        }
    }
    results
}

// ============================================================================
// CSR2D invariants (from fuzz/fuzz_targets/csr2d.rs)
// ============================================================================

/// Check that a sparse matrix has sorted, unique columns per row and sorted,
/// unique global coordinates.
///
/// # Panics
///
/// Panics if any row has unsorted or duplicate column indices, or if the
/// global sparse coordinates are unsorted or contain duplicates.
#[inline]
pub fn check_sparse_matrix_invariants<M>(csr: &M)
where
    M: SparseMatrix2D,
    M::ColumnIndex: Ord + Clone + Debug,
    M::RowIndex: Debug,
    M::Coordinates: Ord + Clone + Debug + PartialEq,
{
    for row_index in csr.row_indices() {
        let column_indices: Vec<M::ColumnIndex> = csr.sparse_row(row_index).collect();
        let mut sorted_column_indices = column_indices.clone();
        sorted_column_indices.sort_unstable();
        assert_eq!(column_indices, sorted_column_indices, "The row {row_index:?} is not sorted");
        sorted_column_indices.dedup();
        assert_eq!(column_indices, sorted_column_indices, "The row {row_index:?} has duplicates");
    }

    let sparse_coordinates: Vec<M::Coordinates> = SparseMatrix::sparse_coordinates(csr).collect();
    let mut clone_sparse_coordinates = sparse_coordinates.clone();
    clone_sparse_coordinates.sort_unstable();
    assert_eq!(
        sparse_coordinates, clone_sparse_coordinates,
        "The sparse coordinates are not sorted"
    );
    clone_sparse_coordinates.dedup();
    assert_eq!(
        sparse_coordinates, clone_sparse_coordinates,
        "The sparse coordinates have duplicates"
    );
}

// ============================================================================
// ValuedCSR2D invariants (from fuzz/fuzz_targets/valued_csr2d.rs)
// ============================================================================

/// Check that each row of a valued sparse matrix has the same number of
/// column indices and values.
///
/// # Panics
///
/// Panics if any row has a different count of column indices vs. values.
#[inline]
pub fn check_valued_matrix_invariants<M>(csr: &M)
where
    M: SparseValuedMatrix2D,
    M::RowIndex: Debug,
{
    for row_index in csr.row_indices() {
        let column_indices: Vec<M::ColumnIndex> = csr.sparse_row(row_index).collect();
        let column_values: Vec<M::Value> = csr.sparse_row_values(row_index).collect();
        assert_eq!(
            column_indices.len(),
            column_values.len(),
            "The row {row_index:?} has different lengths for column indices and values"
        );
    }
}

// ============================================================================
// Blossom V fuzz input and invariants
// ============================================================================

/// Arbitrary valid-input family for Blossom V fuzzing.
///
/// The generated graph always has an even number of vertices. Edges are
/// interpreted as undirected; self-loops and out-of-range endpoints are
/// discarded during normalization.
#[derive(Clone, Debug)]
pub struct FuzzBlossomVCase {
    /// Even number of vertices.
    pub order: u8,
    /// Raw undirected weighted edges before normalization.
    pub edges: Vec<(u8, u8, i32)>,
}

impl<'a> Arbitrary<'a> for FuzzBlossomVCase {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let pair_count: u8 = u.int_in_range(0..=16)?;
        let order = pair_count.saturating_mul(2);
        let max_edges = usize::from(order) * usize::from(order.saturating_sub(1)) / 2;
        let edge_count: usize = u.int_in_range(0..=max_edges.min(192))?;
        let mut edges = Vec::with_capacity(edge_count);

        for _ in 0..edge_count {
            let a = if order == 0 { 0 } else { u.int_in_range(0..=order - 1)? };
            let b = if order == 0 { 0 } else { u.int_in_range(0..=order - 1)? };
            let weight = i32::from(i16::arbitrary(u)?);
            edges.push((a, b, weight));
        }

        Ok(Self { order, edges })
    }
}

/// Structured topology family for Blossom V fuzzing.
///
/// This complements [`FuzzBlossomVCase`] without changing its byte encoding, so
/// the saved crash corpus remains replayable.
#[derive(Clone, Debug)]
pub struct FuzzStructuredBlossomVCase {
    /// Even number of vertices.
    pub order: u8,
    /// Topology family selector.
    pub family: u8,
    /// Weight regime selector.
    pub weight_mode: u8,
    /// Whether to overlay a guaranteed perfect-matching backbone.
    pub ensure_perfect_support: bool,
    /// Seed driving the topology and weight generation.
    pub seed: u64,
}

impl<'a> Arbitrary<'a> for FuzzStructuredBlossomVCase {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        // Keep structured cases smaller than the raw edge-bag target so honggfuzz
        // spends its budget on diverse families rather than very large dense graphs.
        let pair_count: u8 = u.int_in_range(1..=10)?;
        let order = pair_count.saturating_mul(2);
        Ok(Self {
            order,
            family: u.int_in_range(0..=13)?,
            weight_mode: u.int_in_range(0..=4)?,
            ensure_perfect_support: u.int_in_range(0..=3)? != 0,
            seed: u64::arbitrary(u)?,
        })
    }
}

#[derive(Clone, Copy)]
struct StructuredFuzzRng(u64);

impl StructuredFuzzRng {
    #[inline]
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9e37_79b9_7f4a_7c15))
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    #[inline]
    fn next_usize(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            0
        } else {
            let upper_u64 = u64::try_from(upper).expect("usize upper bound fits into u64");
            let raw = self.next_u64() % upper_u64;
            usize::try_from(raw).expect("modulo upper bound always fits into usize")
        }
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        let raw = u32::try_from(self.next_u64() >> 32).expect("upper 32 bits fit in u32");
        f64::from(raw) / f64::from(u32::MAX)
    }

    #[inline]
    fn next_i16(&mut self) -> i16 {
        let bytes = self.next_u64().to_ne_bytes();
        i16::from_ne_bytes([bytes[0], bytes[1]])
    }
}

/// Weighted CSR type used by the Blossom V fuzz helpers.
pub type FuzzBlossomVCsr = ValuedCSR2D<usize, usize, usize, i32>;
type FuzzBlossomVSupportGraph = SymmetricCSR2D<CSR2D<usize, usize, usize>>;

#[must_use]
#[inline]
fn normalize_blossom_v_edges(case: &FuzzBlossomVCase) -> Vec<(usize, usize, i32)> {
    let order = usize::from(case.order);
    let mut by_pair: BTreeMap<(usize, usize), i32> = BTreeMap::new();

    for &(a, b, weight) in &case.edges {
        let u = usize::from(a);
        let v = usize::from(b);
        if u >= order || v >= order || u == v {
            continue;
        }
        let pair = if u < v { (u, v) } else { (v, u) };
        by_pair.insert(pair, weight);
    }

    by_pair.into_iter().map(|((u, v), w)| (u, v, w)).collect()
}

#[doc(hidden)]
#[must_use]
#[inline]
pub fn build_blossom_v_graph(
    case: &FuzzBlossomVCase,
) -> (FuzzBlossomVCsr, Vec<(usize, usize, i32)>) {
    let order = usize::from(case.order);
    let edges = normalize_blossom_v_edges(case);
    let mut csr: FuzzBlossomVCsr =
        SparseMatrixMut::with_sparse_shaped_capacity((order, order), edges.len() * 2);
    let mut directed_edges = Vec::with_capacity(edges.len() * 2);
    for &(u, v, weight) in &edges {
        directed_edges.push((u, v, weight));
        directed_edges.push((v, u, weight));
    }
    directed_edges.sort_unstable_by_key(|&(u, v, _)| (u, v));

    for (u, v, weight) in directed_edges {
        MatrixMut::add(&mut csr, (u, v, weight)).expect("insert Blossom V edge");
    }

    (csr, edges)
}

#[must_use]
fn structured_support_graph(case: &FuzzStructuredBlossomVCase) -> FuzzBlossomVSupportGraph {
    let family = case.family % 14;
    let n_cap = match family {
        // Dense or clique-heavy families benefit most from a tighter cap.
        3 | 5 | 11 | 13 => 16usize,
        _ => 20usize,
    };
    let n = usize::from(case.order).min(n_cap).max(2);
    let seed = case.seed;

    match family {
        0..=3 => {
            let p = match family {
                0 => 0.22,
                1 => 0.38,
                2 => 0.58,
                _ => 0.85,
            };
            erdos_renyi_gnp(seed, n, p)
        }
        4 => {
            let max_edges = n * (n.saturating_sub(1)) / 2;
            let m = (n * 3 / 2).clamp(1, max_edges.max(1));
            erdos_renyi_gnm(seed, n, m)
        }
        5 => {
            let max_edges = n * (n.saturating_sub(1)) / 2;
            let m = (n * n / 4).clamp(1, max_edges.max(1));
            erdos_renyi_gnm(seed, n, m)
        }
        6 => {
            let m0 = n.clamp(2, 6).min(n.saturating_sub(1)).max(1);
            barabasi_albert(seed, n.max(2), m0)
        }
        7 => {
            let mut rng = StructuredFuzzRng::new(seed ^ 0x51ed_d15c_7a11_ce5d);
            let weights = (0..n).map(|_| 1.0 + 4.0 * rng.next_f64()).collect::<Vec<_>>();
            chung_lu(seed, &weights)
        }
        8 => {
            let radius =
                0.18 + 0.62 * StructuredFuzzRng::new(seed ^ 0x74ad_0f9c_0b13_5eed).next_f64();
            random_geometric_graph(seed, n, radius)
        }
        9 => {
            let max_even_k = n.saturating_sub(2).max(2);
            let mut k = (2 + 2 * StructuredFuzzRng::new(seed ^ 0x0ddc_0ffe_e15e_d123)
                .next_usize(3))
            .min(max_even_k);
            if k % 2 == 1 {
                k -= 1;
            }
            let beta =
                0.05 + 0.55 * StructuredFuzzRng::new(seed ^ 0x1234_5678_9abc_def0).next_f64();
            watts_strogatz(seed, n.max(k + 1), k.max(2), beta)
        }
        10 => blossom_triangle_chain_support_graph(n),
        11 => blossom_barbell_support_graph(n),
        12 => overlapping_odd_cycles_support_graph(n),
        _ => blossom_ladder_support_graph(n),
    }
}

fn blossom_triangle_chain_support_graph(order: usize) -> FuzzBlossomVSupportGraph {
    let mut edges = Vec::new();
    let mut triangle_starts = Vec::new();
    let triangle_vertices = order / 3 * 3;

    for start in (0..triangle_vertices).step_by(3) {
        triangle_starts.push(start);
        edges.push((start, start + 1));
        edges.push((start + 1, start + 2));
        edges.push((start, start + 2));
    }

    for window in triangle_starts.windows(2) {
        let a = window[0];
        let b = window[1];
        edges.push((a + 2, b));
    }

    for u in triangle_vertices..order {
        if u > 0 {
            edges.push((u - 1, u));
        }
    }
    edges.sort_unstable();

    UndiEdgesBuilder::default()
        .expected_shape(order)
        .expected_number_of_edges(edges.len())
        .edges(edges)
        .build()
        .expect("build triangle-chain support graph")
}

fn blossom_barbell_support_graph(order: usize) -> FuzzBlossomVSupportGraph {
    let mut edges = Vec::new();
    let left = order / 2;
    let right_start = left;

    for u in 0..left {
        for v in (u + 1)..left {
            edges.push((u, v));
        }
    }
    for u in right_start..order {
        for v in (u + 1)..order {
            edges.push((u, v));
        }
    }
    if order >= 2 {
        edges.push((left.saturating_sub(1), right_start.min(order - 1)));
    }
    edges.sort_unstable();

    UndiEdgesBuilder::default()
        .expected_shape(order)
        .expected_number_of_edges(edges.len())
        .edges(edges)
        .build()
        .expect("build barbell support graph")
}

fn overlapping_odd_cycles_support_graph(order: usize) -> FuzzBlossomVSupportGraph {
    let n = order.max(6);
    // Two triangles sharing a vertex, plus a tail/ring to propagate trees.
    let mut edges = vec![(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4)];

    for u in 4..n.saturating_sub(1) {
        edges.push((u, u + 1));
    }
    if n >= 8 {
        edges.push((1, 5));
        edges.push((3, 6));
        edges.push((0, 7.min(n - 1)));
    }

    edges.sort_unstable();
    edges.dedup();

    UndiEdgesBuilder::default()
        .expected_shape(n)
        .expected_number_of_edges(edges.len())
        .edges(edges)
        .build()
        .expect("build overlapping odd cycles support graph")
}

fn blossom_ladder_support_graph(order: usize) -> FuzzBlossomVSupportGraph {
    let n = order.max(6);
    let mut edges = Vec::new();
    let half = n / 2;

    for i in 0..half.saturating_sub(1) {
        edges.push((i, i + 1));
        edges.push((half + i, half + i + 1));
        edges.push((i, half + i));
    }
    if half > 0 {
        edges.push((half - 1, n - 1));
    }
    // Add crossed rungs to encourage alternating structures and odd cycles.
    for i in 0..half.saturating_sub(1) {
        edges.push((i, half + i + 1));
    }

    edges.sort_unstable();
    edges.dedup();

    UndiEdgesBuilder::default()
        .expected_shape(n)
        .expected_number_of_edges(edges.len())
        .edges(edges)
        .build()
        .expect("build blossom ladder support graph")
}

fn structured_weight(weight_mode: u8, rng: &mut StructuredFuzzRng, is_backbone: bool) -> i32 {
    match weight_mode % 5 {
        0 => i32::from(rng.next_i16()),
        1 => {
            const TIES: [i32; 9] = [-3, -2, -1, 0, 0, 0, 1, 2, 3];
            TIES[rng.next_usize(TIES.as_slice().len())]
        }
        2 => {
            if rng.next_usize(5) == 0 {
                i32::from(rng.next_i16())
            } else {
                0
            }
        }
        3 => {
            if is_backbone {
                -(8000
                    + i32::try_from(rng.next_usize(24000))
                        .expect("bounded backbone weight offset fits in i32"))
            } else {
                i32::try_from(rng.next_usize(2001)).expect("bounded weight offset fits in i32")
                    - 1000
            }
        }
        _ => {
            let base =
                i32::try_from(rng.next_usize(33)).expect("bounded base weight fits in i32") - 16;
            if rng.next_usize(4) == 0 { base * 2048 } else { base }
        }
    }
}

fn overlay_perfect_matching_backbone(
    edges: &mut BTreeMap<(usize, usize), i32>,
    order: usize,
    weight_mode: u8,
    rng: &mut StructuredFuzzRng,
) {
    let mut vertices = (0..order).collect::<Vec<_>>();
    for i in (1..vertices.len()).rev() {
        let j = rng.next_usize(i + 1);
        vertices.swap(i, j);
    }
    for pair in vertices.chunks_exact(2) {
        let u = pair[0].min(pair[1]);
        let v = pair[0].max(pair[1]);
        edges.entry((u, v)).or_insert_with(|| structured_weight(weight_mode, rng, true));
    }
}

#[must_use]
fn build_structured_blossom_v_case(case: &FuzzStructuredBlossomVCase) -> FuzzBlossomVCase {
    let order = usize::from(case.order);
    let support = structured_support_graph(case);
    let mut rng = StructuredFuzzRng::new(case.seed ^ 0xa02b_dbf7_bb3c_0a7d);
    let mut by_pair = BTreeMap::new();

    for u in support.row_indices() {
        for v in support.sparse_row(u) {
            if v > u {
                by_pair.insert((u, v), structured_weight(case.weight_mode, &mut rng, false));
            }
        }
    }

    if case.ensure_perfect_support {
        overlay_perfect_matching_backbone(&mut by_pair, order, case.weight_mode, &mut rng);
    }

    let edges = by_pair
        .into_iter()
        .map(|((u, v), w)| {
            (
                u8::try_from(u).expect("structured fuzz case vertex id fits in u8"),
                u8::try_from(v).expect("structured fuzz case vertex id fits in u8"),
                w,
            )
        })
        .collect();

    FuzzBlossomVCase { order: case.order, edges }
}

#[must_use]
#[inline]
fn blossom_v_matching_cost(edges: &[(usize, usize, i32)], matching: &[(usize, usize)]) -> i64 {
    matching
        .iter()
        .map(|&(u, v)| {
            edges.iter().find(|&&(a, b, _)| (a == u && b == v) || (a == v && b == u)).map_or_else(
                || panic!("matched edge ({u}, {v}) not found in graph"),
                |&(_, _, w)| i64::from(w),
            )
        })
        .sum()
}

#[inline]
fn validate_blossom_v_matching(
    order: usize,
    edges: &[(usize, usize, i32)],
    matching: &[(usize, usize)],
) {
    assert_eq!(
        matching.len(),
        order / 2,
        "matching cardinality {} does not cover graph of order {order}",
        matching.len()
    );
    let mut used = vec![false; order];

    for &(u, v) in matching {
        assert!(u < v, "matching pair must satisfy u < v, got ({u}, {v})");
        assert!(u < order && v < order, "matching pair ({u}, {v}) is out of bounds for n={order}");
        assert!(!used[u], "vertex {u} used twice");
        assert!(!used[v], "vertex {v} used twice");
        assert!(
            edges.iter().any(|&(a, b, _)| (a == u && b == v) || (a == v && b == u)),
            "matching includes non-edge ({u}, {v})"
        );
        used[u] = true;
        used[v] = true;
    }
}

#[must_use]
fn brute_force_blossom_v_cost(order: usize, edges: &[(usize, usize, i32)]) -> Option<i64> {
    #[derive(Clone, Copy)]
    enum BruteForceMemoEntry {
        Uncomputed,
        Computed(Option<i64>),
    }

    fn solve(
        mask: usize,
        weights: &[Vec<Option<i64>>],
        memo: &mut [BruteForceMemoEntry],
    ) -> Option<i64> {
        if mask == 0 {
            return Some(0);
        }
        if let BruteForceMemoEntry::Computed(cached) = memo[mask] {
            return cached;
        }

        let i = mask.trailing_zeros() as usize;
        let rest = mask & !(1usize << i);
        let mut best = None;
        let mut candidates = rest;

        while candidates != 0 {
            let j_bit = candidates & candidates.wrapping_neg();
            let j = j_bit.trailing_zeros() as usize;
            candidates &= candidates - 1;

            let Some(weight) = weights[i][j] else {
                continue;
            };
            let submask = rest & !(1usize << j);
            if let Some(subcost) = solve(submask, weights, memo) {
                let total = subcost + weight;
                best = Some(best.map_or(total, |current: i64| current.min(total)));
            }
        }

        memo[mask] = BruteForceMemoEntry::Computed(best);
        best
    }

    if order == 0 {
        return Some(0);
    }

    let mut weights = vec![vec![None; order]; order];
    for &(u, v, weight) in edges {
        weights[u][v] = Some(i64::from(weight));
        weights[v][u] = Some(i64::from(weight));
    }

    let mut memo = vec![BruteForceMemoEntry::Uncomputed; 1usize << order];
    solve((1usize << order) - 1, &weights, &mut memo)
}

#[must_use]
fn blossom_v_support_has_perfect_matching(order: usize, edges: &[(usize, usize, i32)]) -> bool {
    let support = UndiEdgesBuilder::default()
        .expected_shape(order)
        .expected_number_of_edges(edges.len())
        .edges(edges.iter().map(|&(u, v, _)| (u, v)))
        .build()
        .expect("build Blossom V support graph");
    support.blossom().len() == order / 2
}

/// Run Blossom V on a valid arbitrary undirected weighted graph and check that
/// it either returns a valid perfect matching or correctly reports that none
/// exists.
///
/// For graphs up to 12 vertices, this cross-checks the returned cost or
/// infeasibility against a brute-force oracle. For larger graphs, it checks
/// structural validity and whether the unweighted support graph admits a
/// perfect matching.
#[inline]
pub fn check_blossom_v_invariants(case: &FuzzBlossomVCase) {
    check_blossom_v_invariants_with_bruteforce_limit(case, 12);
}

/// Fuzz-oriented Blossom V invariant checker.
///
/// This keeps the raw crash-replay-compatible byte encoding unchanged, but it
/// avoids spending honggfuzz's 1-second per-input budget on oversized dense raw
/// edge bags that are not useful for crash discovery.
#[inline]
pub fn check_blossom_v_invariants_fuzz(case: &FuzzBlossomVCase) {
    let order = usize::from(case.order);
    let normalized_edge_count = normalize_blossom_v_edges(case).len();

    if order > 20 || normalized_edge_count > 96 {
        return;
    }

    check_blossom_v_invariants_with_bruteforce_limit(case, 10);
}

#[inline]
fn check_blossom_v_invariants_with_bruteforce_limit(
    case: &FuzzBlossomVCase,
    brute_force_limit: usize,
) {
    let order = usize::from(case.order);
    let (graph, edges) = build_blossom_v_graph(case);
    let brute_force_cost =
        (order <= brute_force_limit).then(|| brute_force_blossom_v_cost(order, &edges));
    let support_has_perfect_matching =
        (order > brute_force_limit).then(|| blossom_v_support_has_perfect_matching(order, &edges));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| graph.blossom_v()));

    match result {
        Ok(Ok(matching)) => {
            validate_blossom_v_matching(order, &edges, &matching);
            if let Some(brute_force_cost) = brute_force_cost {
                let actual = blossom_v_matching_cost(&edges, &matching);
                let Some(optimum) = brute_force_cost else {
                    panic!(
                        "Blossom V returned a perfect matching for graph {case:?}, but brute force found none"
                    );
                };
                assert_eq!(
                    actual, optimum,
                    "Blossom V returned non-optimal cost {actual} for graph {case:?}; expected {optimum}"
                );
            }
        }
        Ok(Err(crate::traits::algorithms::BlossomVError::NoPerfectMatching)) => {
            if let Some(brute_force_cost) = brute_force_cost {
                assert!(
                    brute_force_cost.is_none(),
                    "Blossom V reported no perfect matching for graph {case:?}, but brute force found cost {brute_force_cost:?}"
                );
            } else {
                assert!(
                    !support_has_perfect_matching
                        .expect("support-feasibility result should be present"),
                    "Blossom V reported no perfect matching for graph {case:?}, but the support graph has one"
                );
            }
        }
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "<non-string panic payload>".to_string()
            };
            panic!("Blossom V panicked for graph {case:?}: {msg}");
        }
    }
}

/// Run Blossom V invariants on a structured seeded graph family.
#[inline]
pub fn check_structured_blossom_v_invariants(case: &FuzzStructuredBlossomVCase) {
    let derived = build_structured_blossom_v_case(case);
    check_blossom_v_invariants_with_bruteforce_limit(&derived, 10);
}

// ============================================================================
// VF2 fuzz input and invariants
// ============================================================================

/// Arbitrary small-graph family for VF2 fuzzing.
///
/// The generated graphs stay small enough for an exact brute-force oracle, and
/// they can toggle directedness, simple equality labels, and the three VF2
/// match modes.
#[derive(Clone, Debug)]
pub struct FuzzVf2Case {
    /// Whether the graphs are directed.
    pub directed: bool,
    /// Whether node and edge labels participate via equality matching.
    pub use_labels: bool,
    /// Match-mode selector: `0 = isomorphism`,
    /// `1 = induced subgraph isomorphism`,
    /// `2 = non-induced subgraph isomorphism`.
    pub mode_selector: u8,
    /// Number of query nodes.
    pub query_node_count: u8,
    /// Number of target nodes.
    pub target_node_count: u8,
    /// Query node labels.
    pub query_node_labels: Vec<u8>,
    /// Target node labels.
    pub target_node_labels: Vec<u8>,
    /// Raw query edges `(src, dst, label)` before normalization.
    pub query_edges: Vec<(u8, u8, u8)>,
    /// Raw target edges `(src, dst, label)` before normalization.
    pub target_edges: Vec<(u8, u8, u8)>,
}

impl<'a> Arbitrary<'a> for FuzzVf2Case {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let directed = bool::arbitrary(u)?;
        let use_labels = bool::arbitrary(u)?;
        let mode_selector = u.int_in_range(0..=2)?;
        let query_node_count = u.int_in_range(0..=4)?;
        let target_node_count = u.int_in_range(0..=5)?;
        let query_order = usize::from(query_node_count);
        let target_order = usize::from(target_node_count);
        let query_edge_count =
            u.int_in_range(0..=vf2_max_unique_edges(query_order, directed).min(24))?;
        let target_edge_count =
            u.int_in_range(0..=vf2_max_unique_edges(target_order, directed).min(24))?;
        let mut query_node_labels = Vec::with_capacity(query_order);
        let mut target_node_labels = Vec::with_capacity(target_order);
        let mut query_edges = Vec::with_capacity(query_edge_count);
        let mut target_edges = Vec::with_capacity(target_edge_count);

        for _ in 0..query_order {
            query_node_labels.push(u.int_in_range(0..=3)?);
        }
        for _ in 0..target_order {
            target_node_labels.push(u.int_in_range(0..=3)?);
        }
        for _ in 0..query_edge_count {
            let src =
                if query_node_count == 0 { 0 } else { u.int_in_range(0..=query_node_count - 1)? };
            let dst =
                if query_node_count == 0 { 0 } else { u.int_in_range(0..=query_node_count - 1)? };
            let label = u.int_in_range(0..=3)?;
            query_edges.push((src, dst, label));
        }
        for _ in 0..target_edge_count {
            let src =
                if target_node_count == 0 { 0 } else { u.int_in_range(0..=target_node_count - 1)? };
            let dst =
                if target_node_count == 0 { 0 } else { u.int_in_range(0..=target_node_count - 1)? };
            let label = u.int_in_range(0..=3)?;
            target_edges.push((src, dst, label));
        }

        Ok(Self {
            directed,
            use_labels,
            mode_selector,
            query_node_count,
            target_node_count,
            query_node_labels,
            target_node_labels,
            query_edges,
            target_edges,
        })
    }
}

#[derive(Clone, Debug)]
struct NormalizedVf2Graph {
    node_count: usize,
    node_labels: Vec<u8>,
    edge_labels: BTreeMap<(usize, usize), u8>,
}

impl NormalizedVf2Graph {
    #[inline]
    fn edge_key(src: usize, dst: usize, directed: bool) -> (usize, usize) {
        if directed || src == dst { (src, dst) } else { (src.min(dst), src.max(dst)) }
    }

    #[inline]
    fn has_edge(&self, src: usize, dst: usize, directed: bool) -> bool {
        self.edge_labels.contains_key(&Self::edge_key(src, dst, directed))
    }

    #[inline]
    fn edge_label(&self, src: usize, dst: usize, directed: bool) -> Option<u8> {
        self.edge_labels.get(&Self::edge_key(src, dst, directed)).copied()
    }

    #[inline]
    fn sorted_edges(&self) -> Vec<(usize, usize)> {
        self.edge_labels.keys().copied().collect()
    }
}

#[must_use]
#[inline]
fn vf2_max_unique_edges(node_count: usize, directed: bool) -> usize {
    if directed {
        node_count.saturating_mul(node_count)
    } else {
        node_count.saturating_mul(node_count.saturating_add(1)) / 2
    }
}

#[must_use]
#[inline]
fn vf2_mode_from_selector(selector: u8) -> Vf2Mode {
    match selector % 3 {
        0 => Vf2Mode::Isomorphism,
        1 => Vf2Mode::InducedSubgraphIsomorphism,
        2 => Vf2Mode::SubgraphIsomorphism,
        _ => unreachable!("selector modulo 3 is always in range"),
    }
}

#[must_use]
fn normalize_vf2_graph(
    node_count: u8,
    node_labels: &[u8],
    raw_edges: &[(u8, u8, u8)],
    directed: bool,
) -> NormalizedVf2Graph {
    let node_count = usize::from(node_count);
    let mut edge_labels = BTreeMap::new();

    for &(src, dst, label) in raw_edges {
        let src = usize::from(src);
        let dst = usize::from(dst);
        if src >= node_count || dst >= node_count {
            continue;
        }
        edge_labels.insert(NormalizedVf2Graph::edge_key(src, dst, directed), label);
    }

    NormalizedVf2Graph { node_count, node_labels: node_labels[..node_count].to_vec(), edge_labels }
}

#[must_use]
fn build_vf2_nodes(node_count: usize) -> SortedVec<usize> {
    GenericVocabularyBuilder::default()
        .expected_number_of_symbols(node_count)
        .symbols((0..node_count).enumerate())
        .build()
        .expect("build dense VF2 node vocabulary")
}

#[must_use]
fn build_vf2_undigraph(graph: &NormalizedVf2Graph) -> UndiGraph<usize> {
    let nodes = build_vf2_nodes(graph.node_count);
    let edges: SymmetricCSR2D<CSR2D<usize, usize, usize>> = UndiEdgesBuilder::default()
        .expected_number_of_edges(graph.edge_labels.len())
        .expected_shape(graph.node_count)
        .edges(graph.sorted_edges().into_iter())
        .build()
        .expect("build undirected VF2 fuzz graph");
    UndiGraph::from((nodes, edges))
}

#[must_use]
fn build_vf2_digraph(graph: &NormalizedVf2Graph) -> DiGraph<usize> {
    let nodes = build_vf2_nodes(graph.node_count);
    let edges: SquareCSR2D<CSR2D<usize, usize, usize>> = DiEdgesBuilder::default()
        .expected_number_of_edges(graph.edge_labels.len())
        .expected_shape(graph.node_count)
        .edges(graph.sorted_edges().into_iter())
        .build()
        .expect("build directed VF2 fuzz graph");
    DiGraph::from((nodes, edges))
}

#[must_use]
fn vf2_mapping_is_valid(
    query: &NormalizedVf2Graph,
    target: &NormalizedVf2Graph,
    mapping: &[usize],
    mode: Vf2Mode,
    directed: bool,
    use_labels: bool,
) -> bool {
    if query.node_count != mapping.len() {
        return false;
    }
    if query.node_count > target.node_count {
        return false;
    }
    if mode == Vf2Mode::Isomorphism && query.node_count != target.node_count {
        return false;
    }

    if use_labels
        && query.node_labels.iter().enumerate().any(|(query_node, &query_label)| {
            query_label != target.node_labels[mapping[query_node]]
        })
    {
        return false;
    }

    if directed {
        for query_src in 0..query.node_count {
            for query_dst in 0..query.node_count {
                let target_src = mapping[query_src];
                let target_dst = mapping[query_dst];
                let query_has_edge = query.has_edge(query_src, query_dst, true);
                let target_has_edge = target.has_edge(target_src, target_dst, true);

                if query_has_edge {
                    if !target_has_edge {
                        return false;
                    }
                    if use_labels
                        && query.edge_label(query_src, query_dst, true)
                            != target.edge_label(target_src, target_dst, true)
                    {
                        return false;
                    }
                }
                if matches!(mode, Vf2Mode::Isomorphism | Vf2Mode::InducedSubgraphIsomorphism)
                    && query_has_edge != target_has_edge
                {
                    return false;
                }
            }
        }
    } else {
        for query_src in 0..query.node_count {
            for query_dst in query_src..query.node_count {
                let target_src = mapping[query_src];
                let target_dst = mapping[query_dst];
                let query_has_edge = query.has_edge(query_src, query_dst, false);
                let target_has_edge = target.has_edge(target_src, target_dst, false);

                if query_has_edge {
                    if !target_has_edge {
                        return false;
                    }
                    if use_labels
                        && query.edge_label(query_src, query_dst, false)
                            != target.edge_label(target_src, target_dst, false)
                    {
                        return false;
                    }
                }
                if matches!(mode, Vf2Mode::Isomorphism | Vf2Mode::InducedSubgraphIsomorphism)
                    && query_has_edge != target_has_edge
                {
                    return false;
                }
            }
        }
    }

    true
}

#[must_use]
fn brute_force_vf2_matches(
    query: &NormalizedVf2Graph,
    target: &NormalizedVf2Graph,
    mode: Vf2Mode,
    directed: bool,
    use_labels: bool,
) -> Vec<Vec<[usize; 2]>> {
    struct BruteForceVf2Oracle<'a> {
        query: &'a NormalizedVf2Graph,
        target: &'a NormalizedVf2Graph,
        mode: Vf2Mode,
        directed: bool,
        use_labels: bool,
    }

    impl BruteForceVf2Oracle<'_> {
        fn dfs(
            &self,
            query_node: usize,
            used_targets: &mut [bool],
            current_mapping: &mut [usize],
            matches: &mut Vec<Vec<[usize; 2]>>,
        ) {
            if query_node == self.query.node_count {
                if vf2_mapping_is_valid(
                    self.query,
                    self.target,
                    current_mapping,
                    self.mode,
                    self.directed,
                    self.use_labels,
                ) {
                    matches.push(
                        current_mapping
                            .iter()
                            .enumerate()
                            .map(|(query_node, &target_node)| [query_node, target_node])
                            .collect(),
                    );
                }
                return;
            }

            for target_node in 0..self.target.node_count {
                if used_targets[target_node] {
                    continue;
                }
                current_mapping[query_node] = target_node;
                used_targets[target_node] = true;
                self.dfs(query_node + 1, used_targets, current_mapping, matches);
                used_targets[target_node] = false;
            }
        }

        fn enumerate(&self) -> Vec<Vec<[usize; 2]>> {
            let mut matches = Vec::new();
            let mut used_targets = vec![false; self.target.node_count];
            let mut current_mapping = vec![0; self.query.node_count];
            self.dfs(0, &mut used_targets, &mut current_mapping, &mut matches);
            matches.sort_unstable();
            matches
        }
    }

    BruteForceVf2Oracle { query, target, mode, directed, use_labels }.enumerate()
}

#[must_use]
#[inline]
fn canonicalize_vf2_mapping(pairs: &[(usize, usize)]) -> Vec<[usize; 2]> {
    let mut canonical = pairs
        .iter()
        .map(|&(query_node, target_node)| [query_node, target_node])
        .collect::<Vec<_>>();
    canonical.sort_unstable();
    canonical
}

/// Check that VF2 matches the brute-force oracle on a small arbitrary graph
/// pair.
///
/// This validates `has_match()`, `first_match()`, and exhaustive
/// `for_each_match(...)` enumeration for all three match modes, both directed
/// and undirected graphs, and optional equality-based node/edge labels.
///
/// # Panics
///
/// Panics if the current VF2 implementation disagrees with the brute-force
/// oracle or violates the expected search-surface invariants.
pub fn check_vf2_invariants(case: &FuzzVf2Case) {
    let query = normalize_vf2_graph(
        case.query_node_count,
        &case.query_node_labels,
        &case.query_edges,
        case.directed,
    );
    let target = normalize_vf2_graph(
        case.target_node_count,
        &case.target_node_labels,
        &case.target_edges,
        case.directed,
    );
    let mode = vf2_mode_from_selector(case.mode_selector);
    let expected_matches =
        brute_force_vf2_matches(&query, &target, mode, case.directed, case.use_labels);
    if case.directed {
        check_directed_vf2_invariants(case, &query, &target, mode, &expected_matches);
    } else {
        check_undirected_vf2_invariants(case, &query, &target, mode, &expected_matches);
    }
}

fn assert_vf2_search_results(
    case: &FuzzVf2Case,
    direction: &str,
    expected_matches: &[Vec<[usize; 2]>],
    has_match: bool,
    first_match: Option<Vf2Match<usize, usize>>,
    exhausted: bool,
    mut actual_matches: Vec<Vec<[usize; 2]>>,
) {
    let expected_has_match = !expected_matches.is_empty();

    assert_eq!(
        has_match, expected_has_match,
        "VF2 {direction} has_match disagreement for case {case:?}"
    );

    match first_match {
        Some(mapping) => {
            assert!(
                expected_matches.contains(&canonicalize_vf2_mapping(mapping.pairs())),
                "VF2 {direction} first_match returned an embedding outside the oracle set for case {case:?}"
            );
        }
        None => {
            assert!(
                !expected_has_match,
                "VF2 {direction} first_match returned None despite an oracle match for case {case:?}"
            );
        }
    }

    assert!(exhausted, "VF2 {direction} enumeration should exhaust the search for case {case:?}");
    actual_matches.sort_unstable();
    assert_eq!(
        actual_matches, expected_matches,
        "VF2 {direction} embeddings disagreed with the brute-force oracle for case {case:?}"
    );
}

fn check_directed_vf2_invariants(
    case: &FuzzVf2Case,
    query: &NormalizedVf2Graph,
    target: &NormalizedVf2Graph,
    mode: Vf2Mode,
    expected_matches: &[Vec<[usize; 2]>],
) {
    let query_graph = build_vf2_digraph(query);
    let target_graph = build_vf2_digraph(target);

    if case.use_labels {
        let has_match = query_graph
            .vf2(&target_graph)
            .with_mode(mode)
            .with_node_match(|query_node, target_node| {
                query.node_labels[query_node] == target.node_labels[target_node]
            })
            .with_edge_match(|query_src, query_dst, target_src, target_dst| {
                query.edge_label(query_src, query_dst, true)
                    == target.edge_label(target_src, target_dst, true)
            })
            .has_match();
        let first_match = query_graph
            .vf2(&target_graph)
            .with_mode(mode)
            .with_node_match(|query_node, target_node| {
                query.node_labels[query_node] == target.node_labels[target_node]
            })
            .with_edge_match(|query_src, query_dst, target_src, target_dst| {
                query.edge_label(query_src, query_dst, true)
                    == target.edge_label(target_src, target_dst, true)
            })
            .first_match();
        let mut actual_matches = Vec::new();
        let exhausted = query_graph
            .vf2(&target_graph)
            .with_mode(mode)
            .with_node_match(|query_node, target_node| {
                query.node_labels[query_node] == target.node_labels[target_node]
            })
            .with_edge_match(|query_src, query_dst, target_src, target_dst| {
                query.edge_label(query_src, query_dst, true)
                    == target.edge_label(target_src, target_dst, true)
            })
            .for_each_match(|mapping| {
                actual_matches.push(canonicalize_vf2_mapping(mapping.pairs()));
                true
            });
        assert_vf2_search_results(
            case,
            "directed",
            expected_matches,
            has_match,
            first_match,
            exhausted,
            actual_matches,
        );
    } else {
        let has_match = query_graph.vf2(&target_graph).with_mode(mode).has_match();
        let first_match = query_graph.vf2(&target_graph).with_mode(mode).first_match();
        let mut actual_matches = Vec::new();
        let exhausted = query_graph.vf2(&target_graph).with_mode(mode).for_each_match(|mapping| {
            actual_matches.push(canonicalize_vf2_mapping(mapping.pairs()));
            true
        });
        assert_vf2_search_results(
            case,
            "directed",
            expected_matches,
            has_match,
            first_match,
            exhausted,
            actual_matches,
        );
    }
}

fn check_undirected_vf2_invariants(
    case: &FuzzVf2Case,
    query: &NormalizedVf2Graph,
    target: &NormalizedVf2Graph,
    mode: Vf2Mode,
    expected_matches: &[Vec<[usize; 2]>],
) {
    let query_graph = build_vf2_undigraph(query);
    let target_graph = build_vf2_undigraph(target);

    if case.use_labels {
        let has_match = query_graph
            .vf2(&target_graph)
            .with_mode(mode)
            .with_node_match(|query_node, target_node| {
                query.node_labels[query_node] == target.node_labels[target_node]
            })
            .with_edge_match(|query_src, query_dst, target_src, target_dst| {
                query.edge_label(query_src, query_dst, false)
                    == target.edge_label(target_src, target_dst, false)
            })
            .has_match();
        let first_match = query_graph
            .vf2(&target_graph)
            .with_mode(mode)
            .with_node_match(|query_node, target_node| {
                query.node_labels[query_node] == target.node_labels[target_node]
            })
            .with_edge_match(|query_src, query_dst, target_src, target_dst| {
                query.edge_label(query_src, query_dst, false)
                    == target.edge_label(target_src, target_dst, false)
            })
            .first_match();
        let mut actual_matches = Vec::new();
        let exhausted = query_graph
            .vf2(&target_graph)
            .with_mode(mode)
            .with_node_match(|query_node, target_node| {
                query.node_labels[query_node] == target.node_labels[target_node]
            })
            .with_edge_match(|query_src, query_dst, target_src, target_dst| {
                query.edge_label(query_src, query_dst, false)
                    == target.edge_label(target_src, target_dst, false)
            })
            .for_each_match(|mapping| {
                actual_matches.push(canonicalize_vf2_mapping(mapping.pairs()));
                true
            });
        assert_vf2_search_results(
            case,
            "undirected",
            expected_matches,
            has_match,
            first_match,
            exhausted,
            actual_matches,
        );
    } else {
        let has_match = query_graph.vf2(&target_graph).with_mode(mode).has_match();
        let first_match = query_graph.vf2(&target_graph).with_mode(mode).first_match();
        let mut actual_matches = Vec::new();
        let exhausted = query_graph.vf2(&target_graph).with_mode(mode).for_each_match(|mapping| {
            actual_matches.push(canonicalize_vf2_mapping(mapping.pairs()));
            true
        });
        assert_vf2_search_results(
            case,
            "undirected",
            expected_matches,
            has_match,
            first_match,
            exhausted,
            actual_matches,
        );
    }
}

/// Fuzz-oriented VF2 invariant checker.
///
/// The arbitrary case family is already size-bounded, so the fuzz harness can
/// use the exact same oracle as the regression tests.
#[inline]
pub fn check_vf2_invariants_fuzz(case: &FuzzVf2Case) {
    check_vf2_invariants(case);
}

// ============================================================================
// GenericMatrix2DWithPaddedDiagonal invariants
// (from fuzz/fuzz_targets/generic_matrix2d_with_padded_diagonal.rs)
// ============================================================================

/// Type alias for the padded diagonal type used by fuzz targets.
pub type FuzzPaddedDiag =
    GenericMatrix2DWithPaddedDiagonal<ValuedCSR2D<u16, u8, u8, f64>, fn(u8) -> f64>;

/// Check invariants of a [`GenericMatrix2DWithPaddedDiagonal`]: the matrix
/// must be square, every row must contain its diagonal element, column/value
/// counts must match, and imputation flags must be consistent with the
/// underlying matrix.
///
/// # Panics
///
/// Panics if any invariant is violated.
#[inline]
pub fn check_padded_diagonal_invariants(padded_csr: &FuzzPaddedDiag) {
    assert_eq!(
        padded_csr.number_of_rows(),
        padded_csr.number_of_columns(),
        "The number of rows and columns should be equal",
    );

    for row_index in padded_csr.row_indices() {
        // Check that the diagonal of the row is imputed.
        let mut sparse_column_indices = padded_csr.sparse_row(row_index);
        sparse_column_indices.find(|column_index| *column_index == row_index).expect(
            "The diagonal of the row should always be imputed but was not found in the sparse row",
        );

        // Check that the number of sparse column indices and values are equal.
        let number_of_sparse_column_indices = padded_csr.sparse_row(row_index).count();
        let number_of_sparse_column_values = padded_csr.sparse_row_values(row_index).count();

        assert_eq!(
            number_of_sparse_column_indices, number_of_sparse_column_values,
            "The number of sparse column indices and values should be equal"
        );

        // Check that the `is_diagonal_imputed` method works as expected.
        let underlying_matrix = padded_csr.matrix();
        let has_diagonal = if row_index < underlying_matrix.number_of_rows() {
            underlying_matrix.sparse_row(row_index).any(|column_index| column_index == row_index)
        } else {
            false
        };
        let is_diagonal_imputed = padded_csr.is_diagonal_imputed(row_index);
        assert_eq!(
            has_diagonal, !is_diagonal_imputed,
            "The inner diagonal state was `{has_diagonal}` but the `is_diagonal_imputed` method returned `{is_diagonal_imputed}`"
        );

        // Check that the number of elements is consistent.
        let expected_number_of_elements = if row_index < underlying_matrix.number_of_rows() {
            let number_of_inner_sparse_column_indices =
                underlying_matrix.sparse_row(row_index).count();
            if has_diagonal {
                number_of_inner_sparse_column_indices
            } else {
                number_of_inner_sparse_column_indices + 1
            }
        } else {
            1
        };

        assert_eq!(
            number_of_sparse_column_indices, expected_number_of_elements,
            "The number of elements in the padded sparse row should be equal to the number of \
             elements in the inner sparse row plus the diagonal element if it has been imputed"
        );
    }
}

// ============================================================================
// PaddedMatrix2D invariants (from fuzz/fuzz_targets/padded_matrix2d.rs)
// ============================================================================

/// Check invariants of a [`PaddedMatrix2D`]: all values from the underlying
/// CSR matrix must appear in the padded matrix, and imputed values must use
/// the padding value.
///
/// # Panics
///
/// Panics if any invariant is violated.
#[inline]
pub fn check_padded_matrix2d_invariants(csr: &ValuedCSR2D<u16, u8, u8, u8>) {
    let Ok(padded_matrix) = PaddedMatrix2D::new(csr, |_| 1) else {
        return;
    };
    let padded_number_of_rows = padded_matrix.number_of_rows();
    let padded_number_of_columns = padded_matrix.number_of_columns();
    let csr_number_of_rows = csr.number_of_rows();
    let csr_number_of_columns = csr.number_of_columns();
    let mut last_tuple = None;

    for row_index in csr.row_indices() {
        let csr_column_values: Vec<(u8, u8)> =
            csr.sparse_row(row_index).zip(csr.sparse_row_values(row_index)).collect();
        let padded_column_values: Vec<(u8, u8)> =
            padded_matrix.column_indices().zip(padded_matrix.row_values(row_index)).collect();

        for &(column_index, value) in &csr_column_values {
            assert!(
                padded_column_values.contains(&(column_index, value)),
                "The padded matrix does not contain the value {value} (last tuple was \
                 {last_tuple:?}) at column index {column_index}/{padded_number_of_columns} \
                 ({csr_number_of_columns}) for row index {row_index}/{padded_number_of_rows} \
                 ({csr_number_of_rows}). {csr:?}"
            );
            last_tuple = Some((column_index, value));
        }

        for (column_index, value) in padded_column_values {
            if padded_matrix.is_imputed((row_index, column_index)) {
                assert_eq!(value, 1);
            } else {
                assert!(
                    csr_column_values.contains(&(column_index, value)),
                    "The csr matrix does not contain the value {value} at column index \
                     {column_index} for row index {row_index}"
                );
            }
        }
    }
}

// ============================================================================
// Gabow 1976 invariants (from fuzz/fuzz_targets/gabow_1976.rs)
// ============================================================================

/// Check structural validity and exact-size agreement for Gabow's 1976
/// maximum matching implementation against the existing blossom solver.
///
/// # Panics
///
/// Panics if Gabow 1976 returns an invalid matching, violates maximality, or
/// disagrees with `blossom()` on matching size.
#[inline]
pub fn check_gabow_1976_invariants<M>(csr: &M)
where
    M: SparseSquareMatrix + Blossom + Gabow1976,
    M::Index: Debug,
{
    let n = csr.order().as_();
    let gabow_matching = csr.gabow_1976();
    let blossom_matching = csr.blossom();

    assert_eq!(
        gabow_matching.len(),
        blossom_matching.len(),
        "Gabow1976 and Blossom disagree on matching size (n={n})"
    );
    assert!(gabow_matching.len() <= n / 2);

    let mut matched = vec![false; n];
    for &(u, v) in &gabow_matching {
        let ui = u.as_();
        let vi = v.as_();
        assert!(u < v);
        assert!(!matched[ui], "vertex {u:?} matched twice");
        assert!(!matched[vi], "vertex {v:?} matched twice");
        matched[ui] = true;
        matched[vi] = true;
        assert!(csr.has_entry(u, v));
    }

    for u in csr.row_indices() {
        if matched[u.as_()] {
            continue;
        }
        for w in csr.sparse_row(u) {
            assert!(w == u || matched[w.as_()], "edge ({u:?}, {w:?}) has both endpoints unmatched");
        }
    }
}

// ============================================================================
// Kahn ordering (from fuzz/fuzz_targets/kahn.rs)
// ============================================================================

/// Check that a Kahn topological ordering is valid: for every edge (u, v),
/// the position of u must be <= the position of v. Also verifies that
/// the ordering can produce a valid upper triangular matrix.
///
/// Does nothing if the matrix has more than `max_size` rows (to avoid slow
/// tests) or if the matrix contains a cycle.
///
/// # Panics
///
/// Panics if the ordering violates the topological invariant.
#[inline]
pub fn check_kahn_ordering(matrix: &SquareCSR2D<CSR2D<u16, u8, u8>>, max_size: usize) {
    let number_of_rows: usize = matrix.number_of_rows().as_();
    let number_of_columns: usize = matrix.number_of_columns().as_();
    if number_of_rows > max_size || number_of_columns > max_size {
        return;
    }

    let Ok(ordering) = matrix.kahn() else {
        return;
    };

    matrix.row_indices().for_each(|row_id| {
        let row_index = usize::from(row_id);
        let resorted_row_id = ordering[row_index];
        matrix.sparse_row(row_id).for_each(|successor_id| {
            let successor_index = usize::from(successor_id);
            let resorted_successor_id = ordering[successor_index];
            assert!(
                resorted_row_id <= resorted_successor_id,
                "The ordering {ordering:?} is not valid: {resorted_row_id} ({row_id}) > \
                 {resorted_successor_id} ({successor_id})",
            );
        });
    });

    // If the ordering is valid, it must be possible to construct an
    // upper triangular matrix from the ordering.
    let mut coordinates: Vec<(u8, u8)> = SparseMatrix::sparse_coordinates(matrix)
        .map(|(i, j)| (ordering[usize::from(i)], ordering[usize::from(j)]))
        .collect();
    coordinates.sort_unstable();

    let _triangular: UpperTriangularCSR2D<CSR2D<u16, u8, u8>> =
        UpperTriangularCSR2D::from_entries(coordinates).expect("The ordering should be valid");
}

// ============================================================================
// Similarity invariants (from fuzz/fuzz_targets/wu_palmer.rs, lin.rs)
// ============================================================================

/// Check that a [`ScalarSimilarity`] implementation satisfies basic
/// invariants: self-similarity > 0.99, symmetry, and bounds [0, 1].
///
/// Only the first `max_outer` source nodes are tested (all destinations
/// are always tested).
///
/// # Panics
///
/// Panics if any invariant is violated.
#[inline]
pub fn check_similarity_invariants<S, N>(similarity: &S, node_ids: &[N], max_outer: usize)
where
    S: ScalarSimilarity<N, N, Similarity = f64>,
    N: Copy + Eq + Debug,
{
    for &src in node_ids.iter().take(max_outer) {
        for &dst in node_ids {
            let sim = similarity.similarity(&src, &dst);
            if src == dst {
                assert!(sim > 0.99, "Expected self-similarity of {src:?} > 0.99, got {sim}");
            } else {
                let symmetric_similarity = similarity.similarity(&dst, &src);
                assert!(
                    (symmetric_similarity - sim).abs() < f64::EPSILON,
                    "Expected sim({src:?}, {dst:?}) == sim({dst:?}, {src:?}) got \
                     {sim}!={symmetric_similarity}"
                );
            }
            assert!(sim <= 1.0, "Expected sim({src:?},{dst:?}) = {sim} <= 1");
            assert!(sim >= 0.0, "Expected sim({src:?},{dst:?}) = {sim} >= 0");
        }
    }
}

// ============================================================================
// LAP assignment validation (from fuzz/fuzz_targets/lap.rs)
// ============================================================================

/// Validate that a LAP assignment is valid: no duplicate rows or columns,
/// all edges exist, and indices are within bounds.
///
/// # Panics
///
/// Panics if the assignment is invalid.
#[inline]
pub fn validate_lap_assignment(
    csr: &ValuedCSR2D<u16, u8, u8, f64>,
    assignment: &[(u8, u8)],
    label: &str,
) {
    let number_of_rows: usize = csr.number_of_rows().as_();
    let number_of_columns: usize = csr.number_of_columns().as_();
    let mut seen_rows = vec![false; number_of_rows];
    let mut seen_columns = vec![false; number_of_columns];

    for &(row, column) in assignment {
        let row_index: usize = row.as_();
        let column_index: usize = column.as_();

        assert!(
            row_index < number_of_rows,
            "{label}: row index out of bounds ({row_index} >= {number_of_rows})"
        );
        assert!(
            column_index < number_of_columns,
            "{label}: column index out of bounds ({column_index} >= {number_of_columns})"
        );
        assert!(
            csr.has_entry(row, column),
            "{label}: assignment includes non-existing edge ({row}, {column})"
        );
        assert!(!seen_rows[row_index], "{label}: duplicate row in assignment ({row})");
        assert!(!seen_columns[column_index], "{label}: duplicate column in assignment ({column})");

        seen_rows[row_index] = true;
        seen_columns[column_index] = true;
    }
}

/// Returns `true` when edge weights span a numerically stable range,
/// avoiding extreme floating-point regimes.
#[must_use]
#[inline]
pub fn lap_values_are_numerically_stable(csr: &ValuedCSR2D<u16, u8, u8, f64>) -> bool {
    let mut minimum_value = f64::INFINITY;
    let mut maximum_value = 0.0_f64;

    for row in csr.row_indices() {
        for value in csr.sparse_row_values(row) {
            if value > 0.0 && value.is_finite() {
                minimum_value = minimum_value.min(value);
                maximum_value = maximum_value.max(value);
            }
        }
    }

    if minimum_value.is_infinite() {
        return true;
    }

    minimum_value >= f64::MIN_POSITIVE
        && maximum_value <= 1e150
        && (maximum_value / minimum_value) <= 1e12
}

/// Compute the total cost of an assignment.
#[must_use]
#[inline]
pub fn lap_assignment_cost(csr: &ValuedCSR2D<u16, u8, u8, f64>, assignment: &[(u8, u8)]) -> f64 {
    assignment
        .iter()
        .map(|&(row, column)| {
            csr.sparse_row(row)
                .zip(csr.sparse_row_values(row))
                .find_map(|(c, v)| (c == column).then_some(v))
                .unwrap_or_else(|| {
                    panic!("Assignment includes non-existing edge ({row}, {column})")
                })
        })
        .sum()
}

/// Check full LAP sparse-wrapper invariants: both `jaqaman` and
/// `sparse_lapjv` should agree on results when the weight range is
/// numerically stable.
///
/// # Panics
///
/// Panics if the wrappers disagree when they should agree.
#[inline]
pub fn check_lap_sparse_wrapper_invariants(csr: &ValuedCSR2D<u16, u8, u8, f64>) {
    let numerically_stable = lap_values_are_numerically_stable(csr);
    let maximum_value = csr.max_sparse_value().unwrap_or(1000.0);
    // Use multiplicative scaling so that η/2 = 1.05 × max > max at any
    // magnitude.  The old additive formula (max + 1.0) * 2.0 failed when
    // max + 1.0 == max in floating point (values above ~1e16).
    let padding_value = maximum_value * 2.1;
    let maximal_cost = padding_value * 2.0;

    if !padding_value.is_finite()
        || !maximal_cost.is_finite()
        || padding_value <= 0.0
        || maximal_cost <= padding_value
    {
        return;
    }

    let jaqaman_result = csr.jaqaman(padding_value, maximal_cost);
    let sparse_lapjv = csr.sparse_lapjv(padding_value, maximal_cost);
    let sparse_hungarian = csr.sparse_hungarian(padding_value, maximal_cost);

    match (&jaqaman_result, &sparse_lapjv) {
        (Ok(jaqaman_assignment), Ok(lapjv_assignment)) => {
            validate_lap_assignment(csr, jaqaman_assignment, "Jaqaman");
            validate_lap_assignment(csr, lapjv_assignment, "SparseLAPJV");

            if numerically_stable {
                assert_eq!(
                    jaqaman_assignment.len(),
                    lapjv_assignment.len(),
                    "Jaqaman/SparseLAPJV cardinality mismatch: {csr:?}"
                );

                let jaqaman_cost = lap_assignment_cost(csr, jaqaman_assignment);
                let lapjv_cost = lap_assignment_cost(csr, lapjv_assignment);
                let denom = jaqaman_cost.abs().max(lapjv_cost.abs()).max(1e-30);
                assert!(
                    (jaqaman_cost - lapjv_cost).abs() / denom < 1e-9,
                    "Jaqaman/SparseLAPJV objective mismatch ({jaqaman_cost} vs \
                     {lapjv_cost}): {csr:?}"
                );
                if let Ok(hopcroft_karp_assignment) = csr.hopcroft_karp() {
                    assert_eq!(
                        jaqaman_assignment.len(),
                        hopcroft_karp_assignment.len(),
                        "Jaqaman/Hopcroft-Karp cardinality mismatch: {csr:?}"
                    );
                }
            }
        }
        (Err(jaqaman_error), Err(lapjv_error)) => {
            if numerically_stable {
                assert_eq!(
                    jaqaman_error, lapjv_error,
                    "Sparse wrapper error mismatch: Jaqaman={jaqaman_error:?} \
                     SparseLAPJV={lapjv_error:?} matrix={csr:?}"
                );
            }
        }
        (Ok(jaqaman_assignment), Err(lapjv_error)) => {
            validate_lap_assignment(csr, jaqaman_assignment, "Jaqaman");
            assert!(
                !numerically_stable,
                "Sparse wrapper mismatch: Jaqaman returned assignment of len {} \
                 but SparseLAPJV failed with {lapjv_error:?}: {csr:?}",
                jaqaman_assignment.len(),
            );
        }
        (Err(jaqaman_error), Ok(lapjv_assignment)) => {
            validate_lap_assignment(csr, lapjv_assignment, "SparseLAPJV");
            assert!(
                !numerically_stable,
                "Sparse wrapper mismatch: Jaqaman failed with {jaqaman_error:?} \
                 but SparseLAPJV returned assignment of len {}: {csr:?}",
                lapjv_assignment.len(),
            );
        }
    }

    // Cross-validate SparseHungarian against the other solvers.
    if let Ok(hungarian_assignment) = &sparse_hungarian {
        validate_lap_assignment(csr, hungarian_assignment, "SparseHungarian");

        if numerically_stable {
            if let Ok(jaqaman_assignment) = &jaqaman_result {
                assert_eq!(
                    hungarian_assignment.len(),
                    jaqaman_assignment.len(),
                    "SparseHungarian/Jaqaman cardinality mismatch: {csr:?}"
                );

                let hungarian_cost = lap_assignment_cost(csr, hungarian_assignment);
                let jaqaman_cost = lap_assignment_cost(csr, jaqaman_assignment);
                let denom = hungarian_cost.abs().max(jaqaman_cost.abs()).max(1e-30);
                assert!(
                    (hungarian_cost - jaqaman_cost).abs() / denom < 1e-9,
                    "SparseHungarian/Jaqaman objective mismatch ({hungarian_cost} vs \
                     {jaqaman_cost}): {csr:?}"
                );
            }
        }
    } else if numerically_stable && jaqaman_result.is_ok() {
        panic!(
            "SparseHungarian failed but Jaqaman succeeded on numerically stable matrix: {csr:?}"
        );
    }
}

/// Check full LAP invariants on square matrices: core `lapmod` should agree
/// with the sparse wrappers.
///
/// # Panics
///
/// Panics if results are inconsistent.
#[inline]
#[allow(clippy::too_many_lines)]
pub fn check_lap_square_invariants(csr: &ValuedCSR2D<u16, u8, u8, f64>) {
    let number_of_rows: usize = csr.number_of_rows().as_();
    let number_of_columns: usize = csr.number_of_columns().as_();
    if number_of_rows != number_of_columns {
        return;
    }

    let maximum_value = csr.max_sparse_value().unwrap_or(1000.0);
    let max_cost = maximum_value * 2.1;
    if !max_cost.is_finite() || max_cost <= 0.0 {
        return;
    }

    let Ok(lapmod_assignment) = csr.lapmod(max_cost) else {
        return;
    };
    validate_lap_assignment(csr, &lapmod_assignment, "LAPMOD");

    let numerically_stable = lap_values_are_numerically_stable(csr);
    if numerically_stable {
        if let Ok(hopcroft_karp_assignment) = csr.hopcroft_karp() {
            assert_eq!(
                lapmod_assignment.len(),
                hopcroft_karp_assignment.len(),
                "LAPMOD/Hopcroft-Karp cardinality mismatch: {csr:?}"
            );
        }
    }

    let padding_value = maximum_value * 4.2;
    let maximal_cost = padding_value * 2.0;
    if !padding_value.is_finite() || !maximal_cost.is_finite() || maximal_cost <= padding_value {
        return;
    }

    let jaqaman_assignment = csr.jaqaman(padding_value, maximal_cost);
    let sparse_lapjv_assignment = csr.sparse_lapjv(padding_value, maximal_cost);
    let sparse_hungarian_assignment = csr.sparse_hungarian(padding_value, maximal_cost);

    if !numerically_stable {
        if let Ok(assignment) = &jaqaman_assignment {
            validate_lap_assignment(csr, assignment, "Jaqaman");
        }
        if let Ok(assignment) = &sparse_lapjv_assignment {
            validate_lap_assignment(csr, assignment, "SparseLAPJV");
        }
        if let Ok(assignment) = &sparse_hungarian_assignment {
            validate_lap_assignment(csr, assignment, "SparseHungarian");
        }
        return;
    }

    let jaqaman_assignment = jaqaman_assignment.unwrap_or_else(|error| {
        panic!(
            "Jaqaman failed on square matrix that LAPMOD solved with error \
             {error:?}: {csr:?}"
        )
    });
    let sparse_lapjv_assignment = sparse_lapjv_assignment.unwrap_or_else(|error| {
        panic!(
            "SparseLAPJV failed on square matrix that LAPMOD solved with error \
             {error:?}: {csr:?}"
        )
    });
    let sparse_hungarian_assignment = sparse_hungarian_assignment.unwrap_or_else(|error| {
        panic!(
            "SparseHungarian failed on square matrix that LAPMOD solved with error \
             {error:?}: {csr:?}"
        )
    });

    validate_lap_assignment(csr, &jaqaman_assignment, "Jaqaman");
    validate_lap_assignment(csr, &sparse_lapjv_assignment, "SparseLAPJV");
    validate_lap_assignment(csr, &sparse_hungarian_assignment, "SparseHungarian");

    assert_eq!(
        lapmod_assignment.len(),
        jaqaman_assignment.len(),
        "LAPMOD/Jaqaman cardinality mismatch: {csr:?}"
    );
    assert_eq!(
        lapmod_assignment.len(),
        sparse_lapjv_assignment.len(),
        "LAPMOD/SparseLAPJV cardinality mismatch: {csr:?}"
    );
    assert_eq!(
        lapmod_assignment.len(),
        sparse_hungarian_assignment.len(),
        "LAPMOD/SparseHungarian cardinality mismatch: {csr:?}"
    );

    let lapmod_cost = lap_assignment_cost(csr, &lapmod_assignment);
    let jaqaman_cost = lap_assignment_cost(csr, &jaqaman_assignment);
    let sparse_lapjv_cost = lap_assignment_cost(csr, &sparse_lapjv_assignment);
    let sparse_hungarian_cost = lap_assignment_cost(csr, &sparse_hungarian_assignment);

    let denom1 = lapmod_cost.abs().max(jaqaman_cost.abs()).max(1e-30);
    assert!(
        (lapmod_cost - jaqaman_cost).abs() / denom1 < 1e-9,
        "LAPMOD/Jaqaman objective mismatch ({lapmod_cost} vs \
         {jaqaman_cost}): {csr:?}",
    );
    let denom2 = lapmod_cost.abs().max(sparse_lapjv_cost.abs()).max(1e-30);
    assert!(
        (lapmod_cost - sparse_lapjv_cost).abs() / denom2 < 1e-9,
        "LAPMOD/SparseLAPJV objective mismatch ({lapmod_cost} vs \
         {sparse_lapjv_cost}): {csr:?}",
    );
    let denom3 = lapmod_cost.abs().max(sparse_hungarian_cost.abs()).max(1e-30);
    assert!(
        (lapmod_cost - sparse_hungarian_cost).abs() / denom3 < 1e-9,
        "LAPMOD/SparseHungarian objective mismatch ({lapmod_cost} vs \
         {sparse_hungarian_cost}): {csr:?}",
    );
}

// ============================================================================
// Louvain/Leiden invariants (from fuzz/fuzz_targets/{louvain,leiden}.rs)
// ============================================================================

/// Returns `true` when edge weights are in a numerically stable range for
/// modularity comparisons.
#[must_use]
#[inline]
pub fn louvain_weights_are_numerically_stable(csr: &ValuedCSR2D<u16, u8, u8, f64>) -> bool {
    let mut min_val = f64::INFINITY;
    let mut max_val = 0.0_f64;

    for row in csr.row_indices() {
        for val in csr.sparse_row_values(row) {
            if val > 0.0 && val.is_finite() && val.is_normal() {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }
    }

    if min_val.is_infinite() {
        return true;
    }

    min_val >= f64::MIN_POSITIVE && max_val <= 1e150 && (max_val / min_val) <= 1e12
}

fn symmetrized_positive_graph(
    csr: &ValuedCSR2D<u16, u8, u8, f64>,
) -> Option<ValuedCSR2D<u8, u8, u8, f64>> {
    let rows: usize = csr.number_of_rows().as_();
    let cols: usize = csr.number_of_columns().as_();
    if rows != cols || rows == 0 || rows > u8::MAX as usize {
        return None;
    }

    let Ok(n) = u8::try_from(rows) else {
        return None;
    };

    // Extract upper-triangle edges with finite positive weights, then mirror.
    let mut edges: Vec<(u8, u8, f64)> = Vec::new();
    for row in csr.row_indices() {
        let r: usize = row.as_();
        if r >= rows {
            continue;
        }
        for (col, val) in csr.sparse_row(row).zip(csr.sparse_row_values(row)) {
            let c: usize = col.as_();
            if r <= c && val.is_finite() && val.is_normal() && val > 0.0 {
                let Ok(r8) = u8::try_from(r) else {
                    continue;
                };
                let Ok(c8) = u8::try_from(c) else {
                    continue;
                };
                edges.push((r8, c8, val));
                if r8 != c8 {
                    edges.push((c8, r8, val));
                }
            }
        }
    }

    if edges.is_empty() {
        return None;
    }

    edges.sort_unstable_by(|(r1, c1, _), (r2, c2, _)| (r1, c1).cmp(&(r2, c2)));
    edges.dedup_by(|(r1, c1, _), (r2, c2, _)| (*r1, *c1) == (*r2, *c2));

    let Ok(edge_count) = u8::try_from(edges.len()) else {
        return None;
    };

    GenericEdgesBuilder::default()
        .expected_number_of_edges(edge_count)
        .expected_shape((n, n))
        .edges(edges.into_iter())
        .build()
        .ok()
}

fn partition_communities_are_connected(
    csr: &ValuedCSR2D<u8, u8, u8, f64>,
    partition: &[usize],
) -> bool {
    let node_count: usize = csr.number_of_rows().as_();
    if node_count == 0 || partition.len() != node_count {
        return false;
    }

    let number_of_communities =
        partition.iter().copied().max().map_or(0usize, |max| max.saturating_add(1));
    let mut nodes_per_community: Vec<Vec<usize>> = vec![Vec::new(); number_of_communities];
    for (node, community) in partition.iter().copied().enumerate() {
        nodes_per_community[community].push(node);
    }

    let mut queue: Vec<usize> = Vec::new();
    let mut is_member = vec![false; node_count];
    let mut visited = vec![false; node_count];

    for nodes in nodes_per_community {
        if nodes.len() <= 1 {
            continue;
        }

        for node in &nodes {
            is_member[*node] = true;
        }

        queue.clear();
        let start = nodes[0];
        queue.push(start);
        visited[start] = true;

        let mut visited_count = 0usize;
        while let Some(node) = queue.pop() {
            visited_count += 1;

            let Ok(row) = u8::try_from(node) else {
                return false;
            };
            for destination in csr.sparse_row(row) {
                let destination: usize = destination.as_();
                if destination < node_count && is_member[destination] && !visited[destination] {
                    visited[destination] = true;
                    queue.push(destination);
                }
            }
        }

        if visited_count != nodes.len() {
            return false;
        }

        for node in &nodes {
            is_member[*node] = false;
            visited[*node] = false;
        }
    }

    true
}

/// Check Louvain invariants on arbitrary input (should never panic) and,
/// when possible, on a symmetrized version of the matrix (partition length,
/// modularity bounds, determinism).
///
/// # Panics
///
/// Panics if Louvain fails on valid symmetric input or produces invalid
/// results.
#[inline]
pub fn check_louvain_invariants(csr: &ValuedCSR2D<u16, u8, u8, f64>) {
    // Louvain must never panic on arbitrary input.
    let _: Result<LouvainResult<usize>, _> = csr.louvain(&LouvainConfig::default());

    // Skip symmetric invariant checking for extreme weight ranges.
    if !louvain_weights_are_numerically_stable(csr) {
        return;
    }

    let Some(sym_csr) = symmetrized_positive_graph(csr) else {
        return;
    };

    let config = LouvainConfig::default();
    let result = Louvain::<usize>::louvain(&sym_csr, &config)
        .expect("Louvain must not fail on a valid symmetric graph");

    let n: usize = sym_csr.number_of_rows().as_();
    assert_eq!(result.final_partition().len(), n, "partition length must equal node count");
    let modularity = result.final_modularity();
    assert!(
        (-0.5 - 1e-9..=1.0 + 1e-9).contains(&modularity),
        "modularity {modularity} out of [-0.5, 1.0] (with FP tolerance)"
    );

    // Determinism check.
    let result2 = Louvain::<usize>::louvain(&sym_csr, &config).unwrap();
    assert_eq!(
        result.final_partition(),
        result2.final_partition(),
        "Louvain must be deterministic for the same seed"
    );
    assert!(
        (result.final_modularity() - result2.final_modularity()).abs() <= 1.0e-12,
        "modularity must be deterministic"
    );
}

/// Check Leiden invariants on arbitrary input (should never panic) and,
/// when possible, on a symmetrized version of the matrix (partition length,
/// modularity bounds, determinism, community connectedness).
///
/// # Panics
///
/// Panics if Leiden fails on valid symmetric input or produces invalid
/// results.
#[inline]
pub fn check_leiden_invariants(csr: &ValuedCSR2D<u16, u8, u8, f64>) {
    // Leiden must never panic on arbitrary input.
    let _: Result<LeidenResult<usize>, _> = csr.leiden(&LeidenConfig::default());

    // Skip symmetric invariant checking for extreme weight ranges.
    if !louvain_weights_are_numerically_stable(csr) {
        return;
    }

    let Some(sym_csr) = symmetrized_positive_graph(csr) else {
        return;
    };

    let config = LeidenConfig::default();
    let result = Leiden::<usize>::leiden(&sym_csr, &config)
        .expect("Leiden must not fail on a valid symmetric graph");

    let n: usize = sym_csr.number_of_rows().as_();
    let final_partition = result.final_partition();
    assert_eq!(final_partition.len(), n, "partition length must equal node count");
    let modularity = result.final_modularity();
    assert!(
        (-0.5 - 1e-9..=1.0 + 1e-9).contains(&modularity),
        "modularity {modularity} out of [-0.5, 1.0] (with FP tolerance)"
    );
    assert!(
        partition_communities_are_connected(&sym_csr, final_partition),
        "Leiden communities must induce connected subgraphs"
    );

    // Determinism check.
    let result2 = Leiden::<usize>::leiden(&sym_csr, &config).unwrap();
    assert_eq!(
        result.final_partition(),
        result2.final_partition(),
        "Leiden must be deterministic for the same seed"
    );
    assert!(
        (result.final_modularity() - result2.final_modularity()).abs() <= 1.0e-12,
        "modularity must be deterministic"
    );
}

// ============================================================================
// Jacobi eigenvalue decomposition invariants (from fuzz/fuzz_targets/jacobi.rs)
// ============================================================================

/// Check Jacobi eigenvalue decomposition invariants on arbitrary input.
///
/// Wraps the sparse CSR in a [`PaddedMatrix2D`] (padding with 0.0) and, when
/// the resulting matrix is symmetric, square, finite, and small enough
/// (n ≤ 32), verifies:
/// - eigenvalues are sorted descending and all finite
/// - eigenvectors are orthonormal (VᵀV ≈ I)
/// - reconstruction (A ≈ VΛVᵀ)
/// - determinism (same input → same output)
///
/// # Panics
///
/// Panics if any invariant is violated.
#[inline]
pub fn check_jacobi_invariants(csr: &ValuedCSR2D<u16, u8, u8, f64>) {
    let Ok(padded) = PaddedMatrix2D::new(csr, |_| 0.0) else {
        return;
    };
    let rows: usize = padded.number_of_rows().as_();
    let cols: usize = padded.number_of_columns().as_();

    // Must not panic on any input.
    let result = padded.jacobi(&JacobiConfig::default());

    if rows != cols || rows == 0 || rows > 32 {
        return;
    }

    // Read dense values and check for finiteness / symmetry.
    let n = rows;
    let mut a_flat = Vec::with_capacity(n * n);
    let mut all_finite = true;
    for row_idx in padded.row_indices() {
        for val in padded.row_values(row_idx) {
            if !val.is_finite() {
                all_finite = false;
            }
            a_flat.push(val);
        }
    }
    if !all_finite {
        return;
    }

    let mut is_symmetric = true;
    for i in 0..n {
        for j in (i + 1)..n {
            let scale = a_flat[i * n + j].abs().max(a_flat[j * n + i].abs()).max(1.0);
            if (a_flat[i * n + j] - a_flat[j * n + i]).abs() > 16.0 * f64::EPSILON * scale {
                is_symmetric = false;
            }
        }
    }
    if !is_symmetric {
        return;
    }

    // Skip detailed numerical invariants for extreme value ranges.
    // Jacobi rotations square values internally; if max_abs ≈ 1e155,
    // then max_abs² ≈ 1e310 which is near f64::MAX ≈ 1.8e308.
    let max_abs = a_flat.iter().copied().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    if max_abs > 1e150 {
        return;
    }

    let result = result.expect("Jacobi should succeed on a finite symmetric square matrix");

    // Eigenvalues sorted descending.
    for w in result.eigenvalues().windows(2) {
        assert!(w[0] >= w[1], "eigenvalues not sorted descending: {:?}", result.eigenvalues());
    }

    // All eigenvalues finite.
    for &ev in result.eigenvalues() {
        assert!(ev.is_finite(), "non-finite eigenvalue: {ev}");
    }

    // Orthonormality: VᵀV ≈ I.
    for k in 0..n {
        for l in 0..n {
            let dot: f64 =
                (0..n).map(|i| result.eigenvector(k)[i] * result.eigenvector(l)[i]).sum();
            let expected = if k == l { 1.0 } else { 0.0 };
            assert!((dot - expected).abs() < 1e-6, "VᵀV[{k},{l}] = {dot}, expected {expected}");
        }
    }

    // Reconstruction: A ≈ VΛVᵀ.
    for i in 0..n {
        for j in 0..n {
            let mut reconstructed = 0.0;
            for k in 0..n {
                reconstructed +=
                    result.eigenvalues()[k] * result.eigenvector(k)[i] * result.eigenvector(k)[j];
            }
            let expected = a_flat[i * n + j];
            let scale = expected.abs().max(1.0);
            assert!(
                (reconstructed - expected).abs() < 1e-6 * scale,
                "Reconstruction failed at ({i}, {j}): expected {expected}, got {reconstructed}"
            );
        }
    }

    // Determinism.
    let result2 = padded.jacobi(&JacobiConfig::default()).unwrap();
    assert_eq!(
        result.eigenvalues(),
        result2.eigenvalues(),
        "Jacobi must be deterministic for the same input"
    );
}

// ============================================================================
// Classical MDS invariants (from fuzz/fuzz_targets/mds.rs)
// ============================================================================

/// Check classical MDS invariants on arbitrary input.
///
/// Wraps the sparse CSR in a [`PaddedMatrix2D`] (padding with 0.0) and, when
/// the resulting matrix forms a valid distance matrix (square, finite,
/// non-negative, zero diagonal, symmetric, and small enough: n ≤ 32), verifies:
/// - coordinates are all finite
/// - eigenvalues are all finite and sorted descending
/// - stress is finite and non-negative
/// - determinism (same input → same output)
///
/// # Panics
///
/// Panics if any invariant is violated.
#[inline]
pub fn check_mds_invariants(csr: &ValuedCSR2D<u16, u8, u8, f64>) {
    let Ok(padded) = PaddedMatrix2D::new(csr, |_| 0.0) else {
        return;
    };
    let rows: usize = padded.number_of_rows().as_();
    let cols: usize = padded.number_of_columns().as_();

    // Must not panic on any input.
    let config = MdsConfig::default();
    let result = padded.classical_mds(&config);

    if rows != cols || rows <= 1 || rows > 32 {
        return;
    }

    // Read dense values and check for valid distance matrix properties.
    let n = rows;
    let mut d_flat = Vec::with_capacity(n * n);
    let mut all_valid = true;
    for row_idx in padded.row_indices() {
        for val in padded.row_values(row_idx) {
            if !val.is_finite() || val < 0.0 {
                all_valid = false;
            }
            d_flat.push(val);
        }
    }
    if !all_valid {
        return;
    }

    // Check diagonal is zero.
    for i in 0..n {
        if d_flat[i * n + i] != 0.0 {
            return;
        }
    }

    // Check symmetry.
    for i in 0..n {
        for j in (i + 1)..n {
            let scale = d_flat[i * n + j].abs().max(d_flat[j * n + i].abs()).max(1.0);
            if (d_flat[i * n + j] - d_flat[j * n + i]).abs() > 16.0 * f64::EPSILON * scale {
                return;
            }
        }
    }

    // Skip detailed numerical invariants for extreme value ranges.
    let max_abs = d_flat.iter().copied().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    if max_abs > 1e150 {
        return;
    }

    let result = result.expect("MDS should succeed on a valid distance matrix");

    // Coordinates are all finite.
    for &c in result.coordinates_flat() {
        assert!(c.is_finite(), "non-finite coordinate: {c}");
    }

    // Eigenvalues are finite.
    for &ev in result.eigenvalues() {
        assert!(ev.is_finite(), "non-finite eigenvalue: {ev}");
    }

    // Eigenvalues sorted descending.
    for w in result.eigenvalues().windows(2) {
        assert!(w[0] >= w[1], "eigenvalues not sorted descending: {:?}", result.eigenvalues());
    }

    // Stress is finite and non-negative.
    assert!(result.stress().is_finite(), "non-finite stress: {}", result.stress());
    assert!(result.stress() >= 0.0, "negative stress: {}", result.stress());

    // Determinism.
    let result2 = padded.classical_mds(&config).unwrap();
    assert_eq!(
        result.coordinates_flat(),
        result2.coordinates_flat(),
        "MDS must be deterministic for the same input"
    );
    assert_eq!(
        result.eigenvalues(),
        result2.eigenvalues(),
        "MDS eigenvalues must be deterministic"
    );
    assert!(
        (result.stress() - result2.stress()).abs() <= f64::EPSILON,
        "MDS stress must be deterministic"
    );
}

// ============================================================================
// Dense GTH invariants (from fuzz/fuzz_targets/gth.rs)
// ============================================================================

/// Lower bound used when projecting arbitrary floating-point payloads into a
/// numerically tame dense Markov matrix for GTH fuzzing.
const GTH_FUZZ_MIN_WEIGHT: f64 = 1.0e-3;

/// Upper bound used when projecting arbitrary floating-point payloads into a
/// numerically tame dense Markov matrix for GTH fuzzing.
const GTH_FUZZ_MAX_WEIGHT: f64 = 1.0;

fn projected_row_stochastic_dense_matrix(matrix: &VecMatrix2D<f64>) -> Option<VecMatrix2D<f64>> {
    let rows = matrix.number_of_rows();
    let columns = matrix.number_of_columns();
    if rows != columns || rows == 0 || rows > 32 {
        return None;
    }

    let mut data = Vec::with_capacity(rows * columns);
    for row in 0..rows {
        let mut row_values = Vec::with_capacity(columns);
        let mut row_sum = 0.0;
        for value in matrix.row_values(row) {
            let projected = if value.is_finite() {
                value.abs().clamp(GTH_FUZZ_MIN_WEIGHT, GTH_FUZZ_MAX_WEIGHT)
            } else {
                0.0
            };
            row_sum += projected;
            row_values.push(projected);
        }

        if !row_sum.is_finite() || row_sum <= 0.0 {
            row_values.fill(0.0);
            row_values[row] = 1.0;
        } else {
            for value in &mut row_values {
                *value /= row_sum;
            }
        }

        data.extend(row_values);
    }

    Some(VecMatrix2D::new(rows, columns, data))
}

fn dense_gth_residual_l1(matrix: &VecMatrix2D<f64>, stationary: &[f64]) -> f64 {
    let n = matrix.number_of_rows();
    let mut total = 0.0;
    for column in 0..n {
        let mut projected = 0.0;
        for (row, value) in stationary.iter().enumerate().take(n) {
            projected += *value * matrix.value((row, column));
        }
        total += (projected - stationary[column]).abs();
    }
    total
}

/// Check dense GTH invariants on arbitrary input.
///
/// The raw input matrix is passed to `gth()` first to ensure the public API
/// never panics on malformed dense inputs. When the matrix is square, bounded
/// in size, and can be projected into a finite nonnegative row-stochastic
/// matrix with bounded positive weights, the helper verifies:
/// - `gth()` succeeds on the projected matrix
/// - the stationary vector length matches the matrix order
/// - all stationary entries are finite and nonnegative
/// - the stationary entries sum to one
/// - the residual `||πP - π||₁` is small
/// - determinism for identical input
///
/// # Panics
///
/// Panics if any invariant is violated on the projected row-stochastic matrix.
#[inline]
pub fn check_gth_invariants(matrix: &VecMatrix2D<f64>) {
    let config = GthConfig::default();

    // The public API must never panic, even on malformed input.
    let _ = matrix.gth(&config);

    let Some(projected) = projected_row_stochastic_dense_matrix(matrix) else {
        return;
    };

    let result =
        projected.gth(&config).expect("GTH should succeed on a projected row-stochastic matrix");
    let stationary = result.stationary();
    let order = projected.number_of_rows();

    assert_eq!(result.order(), order, "stationary result order mismatch");
    assert_eq!(stationary.len(), order, "stationary vector length mismatch");

    let sum: f64 = stationary.iter().sum();
    assert!((sum - 1.0).abs() <= 1.0e-10, "stationary distribution must sum to one, got {sum}");
    for &value in stationary {
        assert!(value.is_finite(), "stationary distribution contains non-finite values");
        assert!(value >= 0.0, "stationary distribution contains a negative value: {value}");
    }

    let residual = dense_gth_residual_l1(&projected, stationary);
    assert!(residual <= 1.0e-10, "stationary distribution residual too large: {residual}");

    let second =
        projected.gth(&config).expect("GTH should remain deterministic on repeated evaluation");
    for (index, (left, right)) in stationary.iter().zip(second.stationary()).enumerate() {
        assert!(
            (left - right).abs() <= 1.0e-12,
            "GTH must be deterministic at index {index}: {left} vs {right}",
        );
    }
}

// ============================================================================
// Floyd-Warshall invariants (from fuzz/fuzz_targets/floyd_warshall.rs)
// ============================================================================

fn bellman_ford_all_pairs(
    order: usize,
    edges: &[(usize, usize, f64)],
) -> Result<Vec<Option<f64>>, usize> {
    let mut all_pairs = vec![None; order * order];

    for source_id in 0..order {
        let mut distances = vec![f64::INFINITY; order];
        distances[source_id] = 0.0;

        for _ in 0..order.saturating_sub(1) {
            let mut updated = false;
            for &(from, to, weight) in edges {
                if !distances[from].is_finite() {
                    continue;
                }
                let candidate = distances[from] + weight;
                if candidate < distances[to] {
                    distances[to] = candidate;
                    updated = true;
                }
            }
            if !updated {
                break;
            }
        }

        for &(from, to, weight) in edges {
            if distances[from].is_finite() && distances[from] + weight < distances[to] {
                return Err(to);
            }
        }

        for (destination_id, distance) in distances.into_iter().enumerate() {
            if distance.is_finite() {
                all_pairs[source_id * order + destination_id] = Some(distance);
            }
        }
    }

    Ok(all_pairs)
}

/// Check Floyd-Warshall invariants on arbitrary weighted sparse input.
///
/// For arbitrary matrices this helper verifies that the algorithm never
/// panics. For square matrices with finite weights, order ≤ 24, and a
/// moderate dynamic range, it cross-checks the result against a slower
/// Bellman-Ford reference implementation, validates zero diagonal and triangle
/// inequality, and checks determinism.
///
/// # Panics
///
/// Panics if any checked invariant is violated.
#[inline]
#[allow(clippy::too_many_lines)]
pub fn check_floyd_warshall_invariants(csr: &ValuedCSR2D<u16, u8, u8, f64>) {
    let result = csr.floyd_warshall();
    let rows: usize = csr.number_of_rows().as_();
    let columns: usize = csr.number_of_columns().as_();

    if rows != columns {
        assert!(
            matches!(
                result,
                Err(FloydWarshallError::NonSquareMatrix {
                    rows: found_rows,
                    columns: found_columns,
                }) if found_rows == rows && found_columns == columns
            ),
            "non-square matrix should return NonSquareMatrix, got {result:?}"
        );
        return;
    }

    if rows == 0 {
        let distances = result.expect("Floyd-Warshall should succeed on an empty matrix");
        assert_eq!(distances.shape(), vec![0, 0]);
        return;
    }

    let mut edges = Vec::new();
    let mut max_abs = 0.0_f64;
    for row_id in csr.row_indices() {
        let source_id = row_id.as_();
        for (column_id, weight) in csr.sparse_row(row_id).zip(csr.sparse_row_values(row_id)) {
            let destination_id = column_id.as_();
            if !weight.is_finite() {
                assert!(
                    matches!(
                        result,
                        Err(FloydWarshallError::NonFiniteWeight {
                            source_id: found_source,
                            destination_id: found_destination,
                        }) if found_source == source_id && found_destination == destination_id
                    ),
                    "non-finite input weight should return NonFiniteWeight, got {result:?}"
                );
                return;
            }
            max_abs = max_abs.max(weight.abs());
            edges.push((source_id, destination_id, weight));
        }
    }

    if rows > 24 || max_abs > 1e150 {
        return;
    }

    let reference = bellman_ford_all_pairs(rows, &edges);
    match reference {
        Err(_) => {
            assert!(
                matches!(result, Err(FloydWarshallError::NegativeCycle { .. })),
                "negative cycle should return NegativeCycle, got {result:?}"
            );
        }
        Ok(expected) => {
            let distances = result.expect(
                "Floyd-Warshall should succeed on finite square matrices without negative cycles",
            );

            assert_eq!(distances.shape(), vec![rows, rows]);
            for node_id in 0..rows {
                assert_eq!(
                    distances.value((node_id, node_id)),
                    Some(0.0),
                    "distance from a node to itself must be zero in absence of negative cycles"
                );
            }

            for source_id in 0..rows {
                for destination_id in 0..rows {
                    let actual = distances.value((source_id, destination_id));
                    let expected = expected[source_id * rows + destination_id];
                    match (actual, expected) {
                        (None, None) => {}
                        (Some(actual), Some(expected)) => {
                            let tolerance = expected.abs().max(1.0) * 1e-9;
                            assert!(
                                (actual - expected).abs() <= tolerance,
                                "distance mismatch at ({source_id}, {destination_id}): expected {expected}, got {actual}"
                            );
                        }
                        _ => {
                            panic!(
                                "reachability mismatch at ({source_id}, {destination_id}): expected {expected:?}, got {actual:?}"
                            );
                        }
                    }
                }
            }

            for source_id in 0..rows {
                for pivot_id in 0..rows {
                    let Some(source_to_pivot) = distances.value((source_id, pivot_id)) else {
                        continue;
                    };
                    for destination_id in 0..rows {
                        let Some(pivot_to_destination) =
                            distances.value((pivot_id, destination_id))
                        else {
                            continue;
                        };
                        let Some(source_to_destination) =
                            distances.value((source_id, destination_id))
                        else {
                            continue;
                        };
                        let bound = source_to_pivot + pivot_to_destination;
                        let tolerance =
                            bound.abs().max(source_to_destination.abs()).max(1.0) * 1e-9;
                        assert!(
                            source_to_destination <= bound + tolerance,
                            "triangle inequality violated for ({source_id}, {pivot_id}, {destination_id}): {source_to_destination} > {bound}"
                        );
                    }
                }
            }

            let distances2 =
                csr.floyd_warshall().expect("Floyd-Warshall should be deterministic on replay");
            assert_eq!(distances, distances2, "Floyd-Warshall must be deterministic");
        }
    }
}

/// Check PairwiseBFS invariants on arbitrary unweighted square sparse input.
///
/// This helper verifies that repeated BFS returns a square distance matrix with
/// zero diagonal, is deterministic, and matches Floyd-Warshall exactly when
/// the same graph is interpreted as having implicit unit weights.
///
/// Large matrices are still exercised for PairwiseBFS itself, but the
/// cross-check against Floyd-Warshall is capped to keep fuzzing throughput
/// reasonable.
///
/// # Panics
///
/// Panics if any checked invariant is violated.
#[inline]
pub fn check_pairwise_bfs_matches_unit_floyd_warshall(csr: &SquareCSR2D<CSR2D<u16, u8, u8>>) {
    let distances = csr.pairwise_bfs();
    let order = csr.order().as_();

    assert_eq!(distances.shape(), vec![order, order]);
    for node_id in 0..order {
        assert_eq!(
            distances.value((node_id, node_id)),
            Some(0),
            "distance from a node to itself must be zero"
        );
    }

    let distances2 = csr.pairwise_bfs();
    assert_eq!(distances, distances2, "PairwiseBFS must be deterministic");

    if order > 64 {
        return;
    }

    let floyd_warshall = GenericImplicitValuedMatrix2D::new(csr.clone(), |_| 1usize)
        .floyd_warshall()
        .expect("unit-weight Floyd-Warshall should succeed on square unweighted matrices");
    assert_eq!(
        distances, floyd_warshall,
        "PairwiseBFS must match Floyd-Warshall with implicit unit weights"
    );
}

/// Check PairwiseDijkstra invariants on arbitrary weighted sparse input.
///
/// This helper verifies that repeated Dijkstra returns a square distance matrix
/// with zero diagonal, is deterministic, and matches Floyd-Warshall exactly on
/// finite square matrices with non-negative weights. Large matrices and
/// extreme magnitudes are still exercised for PairwiseDijkstra itself, but the
/// cross-check against Floyd-Warshall is capped to keep fuzzing throughput
/// reasonable.
///
/// # Panics
///
/// Panics if any checked invariant is violated.
#[inline]
#[allow(clippy::too_many_lines)]
pub fn check_pairwise_dijkstra_matches_floyd_warshall(csr: &ValuedCSR2D<u16, u8, u8, f64>) {
    let rows: usize = csr.number_of_rows().as_();
    let columns: usize = csr.number_of_columns().as_();

    if rows != columns {
        let result = csr.pairwise_dijkstra();
        assert!(
            matches!(
                result,
                Err(PairwiseDijkstraError::NonSquareMatrix {
                    rows: found_rows,
                    columns: found_columns,
                }) if found_rows == rows && found_columns == columns
            ),
            "non-square matrix should return NonSquareMatrix, got {result:?}"
        );
        return;
    }

    if rows == 0 {
        let result = csr.pairwise_dijkstra();
        let distances = result.expect("PairwiseDijkstra should succeed on an empty matrix");
        assert_eq!(distances.shape(), vec![0, 0]);
        return;
    }

    let mut max_abs = 0.0_f64;
    let mut non_finite_edges = Vec::new();
    let mut negative_edges = Vec::new();
    for row_id in csr.row_indices() {
        let source_id = row_id.as_();
        for (column_id, weight) in csr.sparse_row(row_id).zip(csr.sparse_row_values(row_id)) {
            let destination_id = column_id.as_();
            if !weight.is_finite() {
                non_finite_edges.push((source_id, destination_id));
                continue;
            }
            if weight < 0.0 {
                negative_edges.push((source_id, destination_id));
                continue;
            }
            max_abs = max_abs.max(weight.abs());
        }
    }

    if !non_finite_edges.is_empty() || !negative_edges.is_empty() {
        let result = csr.pairwise_dijkstra();
        match result {
            Err(PairwiseDijkstraError::NonFiniteWeight { source_id, destination_id }) => {
                assert!(
                    non_finite_edges.contains(&(source_id, destination_id)),
                    "expected a non-finite edge in {non_finite_edges:?}, got ({source_id}, {destination_id})"
                );
            }
            Err(PairwiseDijkstraError::NegativeWeight { source_id, destination_id }) => {
                assert!(
                    negative_edges.contains(&(source_id, destination_id)),
                    "expected a negative edge in {negative_edges:?}, got ({source_id}, {destination_id})"
                );
            }
            _ => {
                panic!(
                    "invalid weighted input should return NonFiniteWeight or NegativeWeight, got {result:?}"
                );
            }
        }
        return;
    }

    if rows > 32 || max_abs > 1e150 {
        return;
    }

    let result = csr.pairwise_dijkstra();
    let distances = result.expect(
        "PairwiseDijkstra should succeed on finite square matrices without negative weights",
    );
    let floyd_warshall = csr
        .floyd_warshall()
        .expect("Floyd-Warshall should succeed on the same non-negative weighted matrix");

    assert_eq!(distances.shape(), vec![rows, rows]);
    for node_id in 0..rows {
        assert_eq!(
            distances.value((node_id, node_id)),
            Some(0.0),
            "distance from a node to itself must be zero"
        );
    }

    for source_id in 0..rows {
        for destination_id in 0..rows {
            let actual = distances.value((source_id, destination_id));
            let expected = floyd_warshall.value((source_id, destination_id));
            match (actual, expected) {
                (None, None) => {}
                (Some(actual), Some(expected)) => {
                    let tolerance = expected.abs().max(1.0) * 1e-9;
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "distance mismatch at ({source_id}, {destination_id}): expected {expected}, got {actual}"
                    );
                }
                _ => {
                    panic!(
                        "reachability mismatch at ({source_id}, {destination_id}): expected {expected:?}, got {actual:?}"
                    );
                }
            }
        }
    }

    let distances2 =
        csr.pairwise_dijkstra().expect("PairwiseDijkstra should be deterministic on replay");
    assert_eq!(distances, distances2, "PairwiseDijkstra must be deterministic");
}

// ============================================================================
// Line-graph invariants
// ============================================================================

/// Check that the line-graph algorithms satisfy structural invariants on an
/// arbitrary directed graph wrapped in [`GenericGraph`].
///
/// Invariants checked:
/// - **Undirected `line_graph`**: `|E(L(G))| == sum_v C(deg(v), 2)` where
///   `deg(v)` counts edges with `src < dst` incident to `v`. Every edge in L(G)
///   corresponds to two original edges sharing a common endpoint.
/// - **Directed `directed_line_graph`**: `|E(L(G))| == sum_v in_deg(v) *
///   out_deg(v)`. Every edge `(i, j)` in L(G) satisfies `head(original_edge_i)
///   == tail(original_edge_j)`.
/// - **Edge map length** equals the line-graph vertex count.
/// - **Determinism**: repeated calls produce identical results.
///
/// Graphs larger than `max_nodes` are silently skipped to keep fuzzing fast.
///
/// # Panics
///
/// Panics if any invariant is violated.
#[inline]
pub fn check_line_graph_invariants(
    graph: &GenericGraph<u8, SquareCSR2D<CSR2D<u16, u8, u8>>>,
    max_nodes: usize,
) {
    let n: usize = graph.number_of_nodes().into();
    if n > max_nodes {
        return;
    }

    // ── Undirected line graph ──────────────────────────────────────────
    let lg = graph.line_graph();
    let em = lg.edge_map();

    // edge_map length == number of vertices in L(G)
    assert_eq!(lg.number_of_vertices(), em.len());

    // Collect undirected edges (src < dst) and compute degrees.
    let undi_edges: Vec<(u8, u8)> =
        SparseMatrix::sparse_coordinates(graph.edges()).filter(|&(s, d)| s < d).collect();
    assert_eq!(lg.number_of_vertices(), undi_edges.len());

    // Degree per vertex for undirected view.
    let mut deg = vec![0usize; n];
    for &(s, d) in &undi_edges {
        deg[usize::from(s)] += 1;
        deg[usize::from(d)] += 1;
    }
    let expected_undi_lg_edges: usize = deg.iter().map(|&d| d * d.saturating_sub(1) / 2).sum();
    let actual_undi_lg_edges = Edges::number_of_edges(lg.graph()) / 2;
    assert_eq!(
        actual_undi_lg_edges, expected_undi_lg_edges,
        "undirected |E(L(G))| mismatch: got {actual_undi_lg_edges}, expected {expected_undi_lg_edges}"
    );

    // Every edge in L(G) must correspond to original edges sharing an endpoint.
    for (i, j) in Edges::sparse_coordinates(lg.graph()) {
        if i < j {
            let (a1, a2) = em[i];
            let (b1, b2) = em[j];
            assert!(
                a1 == b1 || a1 == b2 || a2 == b1 || a2 == b2,
                "undirected L(G) edge ({i},{j}): originals ({a1},{a2}),({b1},{b2}) share no endpoint"
            );
        }
    }

    // Determinism.
    let lg2 = graph.line_graph();
    assert_eq!(lg.edge_map(), lg2.edge_map(), "undirected line_graph not deterministic");

    // ── Directed line graph ────────────────────────────────────────────
    let dlg = graph.directed_line_graph();
    let dem = dlg.edge_map();

    assert_eq!(dlg.number_of_vertices(), dem.len());
    let total_edges: usize = SparseMatrix::sparse_coordinates(graph.edges()).count();
    assert_eq!(dlg.number_of_vertices(), total_edges);

    // |E(L(G))| == sum_v in_deg(v) * out_deg(v).
    let mut in_deg = vec![0usize; n];
    let mut out_deg = vec![0usize; n];
    for (s, d) in SparseMatrix::sparse_coordinates(graph.edges()) {
        out_deg[usize::from(s)] += 1;
        in_deg[usize::from(d)] += 1;
    }
    let expected_di_lg_edges: usize = (0..n).map(|v| in_deg[v] * out_deg[v]).sum();
    let actual_di_lg_edges: usize = Edges::number_of_edges(dlg.graph());
    assert_eq!(
        actual_di_lg_edges, expected_di_lg_edges,
        "directed |E(L(G))| mismatch: got {actual_di_lg_edges}, expected {expected_di_lg_edges}"
    );

    // Every edge (i,j) in directed L(G): head of original edge i == tail of
    // original edge j.
    for (i, j) in Edges::sparse_coordinates(dlg.graph()) {
        let (_src_i, dst_i) = dem[i];
        let (src_j, _dst_j) = dem[j];
        assert_eq!(
            dst_i, src_j,
            "directed L(G) edge ({i},{j}): head of edge {i} is {dst_i}, tail of edge {j} is {src_j}"
        );
    }

    // Determinism.
    let dlg2 = graph.directed_line_graph();
    assert_eq!(dlg.edge_map(), dlg2.edge_map(), "directed_line_graph not deterministic");
}

// ============================================================================
// BitSquareMatrix invariants
// ============================================================================

/// Comprehensive invariant checks for [`BitSquareMatrix`].
///
/// Validates sparse matrix contracts, transpose roundtrip, `ExactSizeIterator`
/// guarantees, `neighbor_intersection_count`, `row_and_count`, constructor
/// paths, and iterator consistency.
///
/// # Arguments
///
/// * `m` – the matrix to check
/// * `mask_bytes` – arbitrary bytes used to build a `BitVec` mask for
///   `row_and_count` validation
///
/// # Panics
///
/// Panics if any invariant is violated.
#[allow(clippy::too_many_lines)]
pub fn check_bit_square_matrix_invariants(m: &BitSquareMatrix, mask_bytes: &[u8]) {
    let order = m.order();
    let cap = order.min(16);

    // ── Basic sparse matrix invariants ───────────────────────────────────
    check_sparse_matrix_invariants(m);

    // ── Edge count matches sum of row counts ─────────────────────────────
    let actual: usize = (0..order).map(|r| m.sparse_row(r).count()).sum();
    assert_eq!(m.number_of_defined_values(), actual);

    // ── has_entry consistent with sparse_row ─────────────────────────────
    for r in 0..cap {
        let row_cols: Vec<usize> = m.sparse_row(r).collect();
        for c in 0..order {
            assert_eq!(m.has_entry(r, c), row_cols.contains(&c));
        }
    }

    // ── Forward + reverse coordinate iteration ───────────────────────────
    let fwd: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(m).collect();
    let mut rev: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(m).rev().collect();
    rev.reverse();
    assert_eq!(fwd, rev);

    // ── Diagonal count consistency ───────────────────────────────────────
    let diag_count = (0..order).filter(|&i| m.has_entry(i, i)).count();
    assert_eq!(m.number_of_defined_diagonal_values(), diag_count);

    // ── Row sizes consistency ────────────────────────────────────────────
    let sizes: Vec<usize> = m.sparse_row_sizes().collect();
    assert_eq!(sizes.len(), order);
    for (r, &sz) in sizes.iter().enumerate() {
        assert_eq!(sz, m.number_of_defined_values_in_row(r));
    }

    // ── sparse_rows() yields row indices repeated per entry ──────────────
    let sparse_rows: Vec<usize> = m.sparse_rows().collect();
    assert_eq!(sparse_rows.len(), actual);
    let mut expected_rows = Vec::new();
    for r in 0..order {
        let count = m.sparse_row(r).count();
        for _ in 0..count {
            expected_rows.push(r);
        }
    }
    assert_eq!(sparse_rows, expected_rows);

    // ── ExactSizeIterator contracts ──────────────────────────────────────
    let coords = SparseMatrix::sparse_coordinates(m);
    assert_eq!(coords.len(), actual);

    let row_sizes = m.sparse_row_sizes();
    assert_eq!(row_sizes.len(), order);

    let cols = m.sparse_columns();
    assert_eq!(cols.len(), actual);

    // ── last_sparse_coordinates ──────────────────────────────────────────
    let all_coords: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(m).collect();
    if let Some(last) = m.last_sparse_coordinates() {
        assert_eq!(all_coords.last().copied(), Some(last));
    } else {
        assert!(all_coords.is_empty());
    }

    // ── sparse_columns consistency ───────────────────────────────────────
    let from_coords: Vec<usize> = SparseMatrix::sparse_coordinates(m).map(|(_, c)| c).collect();
    let from_iter: Vec<usize> = m.sparse_columns().collect();
    assert_eq!(from_coords, from_iter);

    // ── Transpose roundtrip and semantics ────────────────────────────────
    let t = m.transpose();
    assert_eq!(t.order(), order);
    assert_eq!(t.number_of_defined_values(), m.number_of_defined_values());
    assert_eq!(t.transpose(), *m, "transpose is not involutory");
    for &(r, c) in &all_coords {
        assert!(t.has_entry(c, r), "transpose missing ({c},{r}) for original ({r},{c})");
    }

    // ── neighbor_intersection_count cross-validation (small matrices) ────
    for i in 0..cap {
        for j in i..cap {
            let expected: usize = m.sparse_row(i).filter(|&col| m.has_entry(j, col)).count();
            assert_eq!(
                m.neighbor_intersection_count(i, j),
                expected,
                "neighbor_intersection_count({i},{j}) mismatch"
            );
            // Symmetry: AND + popcount is commutative
            assert_eq!(
                m.neighbor_intersection_count(i, j),
                m.neighbor_intersection_count(j, i),
                "neighbor_intersection_count not symmetric for ({i},{j})"
            );
        }
    }

    // ── row_and_count cross-validation ───────────────────────────────────
    let mask = if order > 0 {
        let mut bv = BitVec::repeat(false, order);
        for (idx, &byte) in mask_bytes.iter().enumerate() {
            if idx >= order {
                break;
            }
            bv.set(idx, byte & 1 != 0);
        }
        bv
    } else {
        BitVec::new()
    };
    for r in 0..cap {
        let expected: usize = m.sparse_row(r).filter(|&col| col < mask.len() && mask[col]).count();
        assert_eq!(m.row_and_count(r, &mask), expected, "row_and_count({r}) mismatch");
    }

    // ── row_bitslice consistency ─────────────────────────────────────────
    for r in 0..cap {
        let bits = m.row_bitslice(r);
        assert_eq!(bits.len(), order);
        for c in 0..order {
            assert_eq!(bits[c], m.has_entry(r, c));
        }
    }

    // ── from_edges constructor roundtrip ─────────────────────────────────
    let edges: Vec<(usize, usize)> = SparseMatrix::sparse_coordinates(m).collect();
    let rebuilt = BitSquareMatrix::from_edges(order, edges.iter().copied());
    assert_eq!(rebuilt, *m, "from_edges roundtrip mismatch");

    // ── from_symmetric_edges constructor ─────────────────────────────────
    // Collect undirected edges (src <= dst) from original, build symmetric, verify
    let sym_edges: Vec<(usize, usize)> = edges.iter().filter(|&&(r, c)| r <= c).copied().collect();
    let sym = BitSquareMatrix::from_symmetric_edges(order, sym_edges.iter().copied());
    // Every original edge that has its mirror should appear in sym
    for &(r, c) in &sym_edges {
        assert!(sym.has_entry(r, c));
        assert!(sym.has_entry(c, r));
    }
}

#[cfg(all(test, feature = "arbitrary", feature = "std"))]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        process,
        time::{SystemTime, UNIX_EPOCH},
    };

    use arbitrary::Arbitrary;

    use super::*;
    use crate::traits::{MatrixMut, ScalarSimilarity};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    struct NeedsThreeBytes([u8; 3]);

    impl<'a> Arbitrary<'a> for NeedsThreeBytes {
        fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
            let bytes = u.bytes(3)?;
            Ok(Self([bytes[0], bytes[1], bytes[2]]))
        }
    }

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(label: &str) -> Self {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock should be after epoch")
                .as_nanos();
            let pid = process::id();
            let path = std::env::temp_dir().join(format!("geometric_traits_{label}_{pid}_{now}"));
            fs::create_dir_all(&path).expect("failed to create temp directory");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn sample_sparse_csr() -> CSR2D<u16, u8, u8> {
        let mut csr: CSR2D<u16, u8, u8> = CSR2D::with_sparse_shaped_capacity((3, 3), 4);
        MatrixMut::add(&mut csr, (0, 0)).expect("insert (0,0)");
        MatrixMut::add(&mut csr, (0, 2)).expect("insert (0,2)");
        MatrixMut::add(&mut csr, (1, 1)).expect("insert (1,1)");
        MatrixMut::add(&mut csr, (2, 2)).expect("insert (2,2)");
        csr
    }

    fn sample_valued_csr_f64() -> ValuedCSR2D<u16, u8, u8, f64> {
        let mut csr: ValuedCSR2D<u16, u8, u8, f64> =
            ValuedCSR2D::with_sparse_shaped_capacity((2, 2), 4);
        MatrixMut::add(&mut csr, (0, 0, 1.0)).expect("insert (0,0)");
        MatrixMut::add(&mut csr, (0, 1, 2.0)).expect("insert (0,1)");
        MatrixMut::add(&mut csr, (1, 0, 2.0)).expect("insert (1,0)");
        MatrixMut::add(&mut csr, (1, 1, 1.0)).expect("insert (1,1)");
        csr
    }

    fn sample_valued_csr_u8() -> ValuedCSR2D<u16, u8, u8, u8> {
        let mut csr: ValuedCSR2D<u16, u8, u8, u8> =
            ValuedCSR2D::with_sparse_shaped_capacity((2, 2), 3);
        MatrixMut::add(&mut csr, (0, 0, 7)).expect("insert (0,0)");
        MatrixMut::add(&mut csr, (0, 1, 3)).expect("insert (0,1)");
        MatrixMut::add(&mut csr, (1, 1, 9)).expect("insert (1,1)");
        csr
    }

    fn sample_dense_matrix_f64() -> VecMatrix2D<f64> {
        VecMatrix2D::new(3, 3, vec![0.2, 0.3, 0.5, 0.6, 0.1, 0.3, 0.25, 0.25, 0.5])
    }

    fn build_valued_csr_f64(
        shape: (u8, u8),
        edges: &[(u8, u8, f64)],
    ) -> ValuedCSR2D<u16, u8, u8, f64> {
        let mut csr: ValuedCSR2D<u16, u8, u8, f64> =
            ValuedCSR2D::with_sparse_shaped_capacity(shape, u16::try_from(edges.len()).unwrap());
        for &(row, column, value) in edges {
            MatrixMut::add(&mut csr, (row, column, value))
                .expect("insert weighted edge into test matrix");
        }
        csr
    }

    #[test]
    fn test_from_bytes_success_and_failure() {
        assert_eq!(from_bytes::<u8>(&[42]), Some(42));
        assert_eq!(from_bytes::<NeedsThreeBytes>(&[1, 2, 3]), Some(NeedsThreeBytes([1, 2, 3])));
        assert!(from_bytes::<NeedsThreeBytes>(&[7, 8]).is_none());
    }

    #[test]
    fn test_replay_dir_skips_invalid_files() {
        let dir = TempDir::new("replay_dir");
        fs::write(dir.path().join("valid.bin"), [1u8, 2u8, 3u8]).expect("write valid file");
        fs::write(dir.path().join("invalid.bin"), [7u8, 8u8]).expect("write invalid file");
        fs::create_dir_all(dir.path().join("nested")).expect("create nested directory");

        let mut decoded: Vec<NeedsThreeBytes> = replay_dir(dir.path());
        decoded.sort_unstable();
        assert_eq!(decoded, vec![NeedsThreeBytes([1, 2, 3])]);
    }

    #[test]
    fn test_replay_dir_missing_directory_returns_empty() {
        let dir = TempDir::new("replay_missing");
        let missing = dir.path().join("does_not_exist");
        let decoded: Vec<u8> = replay_dir(&missing);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_check_sparse_matrix_invariants_on_valid_matrix() {
        let csr = sample_sparse_csr();
        check_sparse_matrix_invariants(&csr);
    }

    #[test]
    fn test_check_valued_matrix_invariants_on_valid_matrix() {
        let csr = sample_valued_csr_f64();
        check_valued_matrix_invariants(&csr);
    }

    #[test]
    fn test_check_padded_diagonal_invariants_on_valid_matrix() {
        fn one(_: u8) -> f64 {
            1.0
        }

        let mut base: ValuedCSR2D<u16, u8, u8, f64> =
            ValuedCSR2D::with_sparse_shaped_capacity((2, 2), 3);
        MatrixMut::add(&mut base, (0, 1, 4.0)).expect("insert (0,1)");
        MatrixMut::add(&mut base, (1, 0, 4.0)).expect("insert (1,0)");
        MatrixMut::add(&mut base, (1, 1, 2.0)).expect("insert (1,1)");

        let padded = GenericMatrix2DWithPaddedDiagonal::new(base, one as fn(u8) -> f64)
            .expect("padded diagonal construction");
        check_padded_diagonal_invariants(&padded);
    }

    #[test]
    fn test_check_padded_matrix2d_invariants_on_valid_matrix() {
        let csr = sample_valued_csr_u8();
        check_padded_matrix2d_invariants(&csr);
    }

    #[test]
    fn test_check_kahn_ordering_on_simple_dag() {
        let mut matrix: SquareCSR2D<CSR2D<u16, u8, u8>> =
            SquareCSR2D::with_sparse_shaped_capacity(3, 2);
        matrix.extend(vec![(0, 1), (1, 2)]).expect("extend matrix");
        check_kahn_ordering(&matrix, 10);
    }

    struct IdentitySimilarity;

    impl ScalarSimilarity<u8, u8> for IdentitySimilarity {
        type Similarity = f64;

        fn similarity(&self, left: &u8, right: &u8) -> Self::Similarity {
            if left == right { 1.0 } else { 0.5 }
        }
    }

    struct BadSimilarity;

    impl ScalarSimilarity<u8, u8> for BadSimilarity {
        type Similarity = f64;

        fn similarity(&self, left: &u8, right: &u8) -> Self::Similarity {
            if left == right { 0.0 } else { 0.4 }
        }
    }

    #[test]
    fn test_check_similarity_invariants_passes_for_valid_similarity() {
        let nodes = [0u8, 1u8, 2u8];
        check_similarity_invariants(&IdentitySimilarity, &nodes, 3);
    }

    #[test]
    #[should_panic(expected = "self-similarity")]
    fn test_check_similarity_invariants_panics_for_invalid_similarity() {
        let nodes = [0u8, 1u8];
        check_similarity_invariants(&BadSimilarity, &nodes, 2);
    }

    #[test]
    fn test_validate_lap_assignment_accepts_valid_assignment() {
        let csr = sample_valued_csr_f64();
        validate_lap_assignment(&csr, &[(0, 0), (1, 1)], "valid");
    }

    #[test]
    #[should_panic(expected = "duplicate row")]
    fn test_validate_lap_assignment_panics_on_duplicate_row() {
        let csr = sample_valued_csr_f64();
        validate_lap_assignment(&csr, &[(0, 0), (0, 1)], "duplicate_row");
    }

    #[test]
    #[should_panic(expected = "non-existing edge")]
    fn test_validate_lap_assignment_panics_on_missing_edge() {
        let mut csr: ValuedCSR2D<u16, u8, u8, f64> =
            ValuedCSR2D::with_sparse_shaped_capacity((2, 2), 2);
        MatrixMut::add(&mut csr, (0, 0, 1.0)).expect("insert (0,0)");
        MatrixMut::add(&mut csr, (1, 1, 2.0)).expect("insert (1,1)");
        validate_lap_assignment(&csr, &[(0, 1)], "missing");
    }

    #[test]
    fn test_lap_values_are_numerically_stable_true() {
        let csr = sample_valued_csr_f64();
        assert!(lap_values_are_numerically_stable(&csr));
    }

    #[test]
    fn test_lap_values_are_numerically_stable_false_for_large_ratio() {
        let mut csr: ValuedCSR2D<u16, u8, u8, f64> =
            ValuedCSR2D::with_sparse_shaped_capacity((2, 2), 2);
        MatrixMut::add(&mut csr, (0, 0, 1.0e-10)).expect("insert small value");
        MatrixMut::add(&mut csr, (1, 1, 1.0e10)).expect("insert large value");
        assert!(!lap_values_are_numerically_stable(&csr));
    }

    #[test]
    fn test_lap_assignment_cost_returns_expected_cost() {
        let csr = sample_valued_csr_f64();
        let cost = lap_assignment_cost(&csr, &[(0, 0), (1, 1)]);
        assert!((cost - 2.0).abs() <= f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "non-existing edge")]
    fn test_lap_assignment_cost_panics_for_missing_edge() {
        let mut csr: ValuedCSR2D<u16, u8, u8, f64> =
            ValuedCSR2D::with_sparse_shaped_capacity((2, 2), 2);
        MatrixMut::add(&mut csr, (0, 0, 1.0)).expect("insert (0,0)");
        MatrixMut::add(&mut csr, (1, 1, 2.0)).expect("insert (1,1)");
        let _ = lap_assignment_cost(&csr, &[(0, 1)]);
    }

    #[test]
    fn test_check_lap_sparse_wrapper_invariants_smoke() {
        let csr = sample_valued_csr_f64();
        check_lap_sparse_wrapper_invariants(&csr);
    }

    #[test]
    fn test_check_lap_square_invariants_smoke() {
        let csr = sample_valued_csr_f64();
        check_lap_square_invariants(&csr);
    }

    #[test]
    fn test_louvain_weights_are_numerically_stable_true() {
        let csr = sample_valued_csr_f64();
        assert!(louvain_weights_are_numerically_stable(&csr));
    }

    #[test]
    fn test_louvain_weights_are_numerically_stable_false_for_large_ratio() {
        let mut csr: ValuedCSR2D<u16, u8, u8, f64> =
            ValuedCSR2D::with_sparse_shaped_capacity((2, 2), 2);
        MatrixMut::add(&mut csr, (0, 0, 1.0e-20)).expect("insert small value");
        MatrixMut::add(&mut csr, (1, 1, 1.0e20)).expect("insert large value");
        assert!(!louvain_weights_are_numerically_stable(&csr));
    }

    #[test]
    fn test_check_louvain_invariants_smoke() {
        let csr = sample_valued_csr_f64();
        check_louvain_invariants(&csr);
    }

    #[test]
    fn test_check_leiden_invariants_smoke() {
        let csr = sample_valued_csr_f64();
        check_leiden_invariants(&csr);
    }

    #[test]
    fn test_check_jacobi_invariants_smoke() {
        let csr = sample_valued_csr_f64();
        check_jacobi_invariants(&csr);
    }

    #[test]
    fn test_check_mds_invariants_smoke() {
        let csr = sample_valued_csr_f64();
        check_mds_invariants(&csr);
    }

    #[test]
    fn test_check_gth_invariants_smoke() {
        let matrix = sample_dense_matrix_f64();
        check_gth_invariants(&matrix);
    }

    #[test]
    fn test_check_mds_invariants_on_valid_distance_matrix() {
        let csr = build_valued_csr_f64(
            (3, 3),
            &[
                (0, 0, 0.0),
                (0, 1, 1.0),
                (0, 2, 1.0),
                (1, 0, 1.0),
                (1, 1, 0.0),
                (1, 2, 1.0),
                (2, 0, 1.0),
                (2, 1, 1.0),
                (2, 2, 0.0),
            ],
        );
        check_mds_invariants(&csr);
    }

    #[test]
    fn test_check_floyd_warshall_invariants_smoke() {
        let csr = sample_valued_csr_f64();
        check_floyd_warshall_invariants(&csr);
    }

    #[test]
    fn test_check_pairwise_bfs_matches_unit_floyd_warshall_smoke() {
        let mut matrix: SquareCSR2D<CSR2D<u16, u8, u8>> =
            SquareCSR2D::with_sparse_shaped_capacity(3, 2);
        matrix.extend(vec![(0, 1), (1, 2)]).expect("extend matrix");
        check_pairwise_bfs_matches_unit_floyd_warshall(&matrix);
    }

    #[test]
    fn test_check_pairwise_dijkstra_matches_floyd_warshall_smoke() {
        let csr = sample_valued_csr_f64();
        check_pairwise_dijkstra_matches_floyd_warshall(&csr);
    }

    #[test]
    fn test_check_bit_square_matrix_invariants_smoke() {
        let matrix = BitSquareMatrix::from_edges(4, [(0, 0), (0, 1), (1, 2), (2, 1), (2, 3)]);
        check_bit_square_matrix_invariants(&matrix, &[0b1011, 0b0101]);
    }

    #[test]
    fn test_check_bit_square_matrix_invariants_on_empty_matrix() {
        let matrix = BitSquareMatrix::new(0);
        check_bit_square_matrix_invariants(&matrix, &[]);
    }

    #[test]
    fn test_build_blossom_v_graph_normalizes_and_symmetrizes_edges() {
        let case = FuzzBlossomVCase {
            order: 4,
            edges: vec![(0, 0, 99), (4, 1, 13), (1, 0, 7), (0, 1, -3), (2, 3, 5), (3, 2, 9)],
        };

        let (graph, edges) = build_blossom_v_graph(&case);

        assert_eq!(edges, vec![(0, 1, -3), (2, 3, 9)]);
        assert_eq!(graph.number_of_rows(), 4);
        assert_eq!(graph.number_of_columns(), 4);
        assert_eq!(graph.number_of_defined_values(), 4);
        assert_eq!(graph.sparse_row(0).collect::<Vec<_>>(), vec![1]);
        assert_eq!(graph.sparse_row(1).collect::<Vec<_>>(), vec![0]);
        assert_eq!(graph.sparse_row(2).collect::<Vec<_>>(), vec![3]);
        assert_eq!(graph.sparse_row(3).collect::<Vec<_>>(), vec![2]);
    }

    #[test]
    fn test_fuzz_blossom_v_case_arbitrary_produces_even_order() {
        let mut u = arbitrary::Unstructured::new(&[0x7f; 256]);
        let case = FuzzBlossomVCase::arbitrary(&mut u).expect("arbitrary Blossom V case");

        assert_eq!(case.order % 2, 0);
        for &(a, b, _) in &case.edges {
            assert!(a < case.order);
            assert!(b < case.order);
        }
    }

    #[test]
    fn test_structured_blossom_v_case_arbitrary_ranges() {
        let mut u = arbitrary::Unstructured::new(&[0x55; 128]);
        let case = FuzzStructuredBlossomVCase::arbitrary(&mut u)
            .expect("arbitrary structured Blossom V case");

        assert_eq!(case.order % 2, 0);
        assert!((1..=20).contains(&case.order));
        assert!(case.family <= 13);
        assert!(case.weight_mode <= 4);
    }

    #[test]
    fn test_structured_fuzz_rng_small_upper_bounds() {
        let mut rng = StructuredFuzzRng::new(0x1234_5678);
        assert_eq!(rng.next_usize(0), 0);
        assert_eq!(rng.next_usize(1), 0);
        assert!((0.0..=1.0).contains(&rng.next_f64()));
        let _ = rng.next_i16();
    }

    #[test]
    fn test_build_blossom_v_graph_empty_case() {
        let case = FuzzBlossomVCase { order: 0, edges: vec![] };
        let (graph, edges) = build_blossom_v_graph(&case);
        assert!(edges.is_empty());
        assert_eq!(graph.number_of_rows(), 0);
        assert_eq!(graph.number_of_columns(), 0);
        assert_eq!(graph.number_of_defined_values(), 0);
    }

    #[test]
    fn test_structured_support_graph_covers_all_families() {
        for family in 0..=13 {
            let case = FuzzStructuredBlossomVCase {
                order: 8,
                family,
                weight_mode: 0,
                ensure_perfect_support: false,
                seed: 0xdecafbad_u64 + u64::from(family),
            };
            let support = structured_support_graph(&case);

            assert_eq!(support.number_of_rows(), 8);
            assert_eq!(support.number_of_columns(), 8);
            for row in support.row_indices() {
                for column in support.sparse_row(row) {
                    assert_ne!(row, column, "family {family} produced a self-loop at {row}");
                }
            }
        }
    }

    #[test]
    fn test_build_structured_blossom_v_case_backbone_covers_all_weight_modes() {
        for weight_mode in 0..=4 {
            let case = FuzzStructuredBlossomVCase {
                order: 8,
                family: 10,
                weight_mode,
                ensure_perfect_support: true,
                seed: 0x1234_5678_9abc_def0,
            };
            let derived = build_structured_blossom_v_case(&case);
            let normalized = normalize_blossom_v_edges(&derived);

            assert_eq!(derived.order, case.order);
            assert!(
                blossom_v_support_has_perfect_matching(usize::from(derived.order), &normalized),
                "weight_mode={weight_mode} should preserve a perfect matching backbone"
            );
        }
    }

    #[test]
    fn test_check_blossom_v_invariants_fuzz_returns_early_for_large_inputs() {
        let oversized_order = FuzzBlossomVCase { order: 22, edges: vec![(0, 1, -7)] };
        check_blossom_v_invariants_fuzz(&oversized_order);

        let mut dense_edges = Vec::new();
        'outer: for u in 0..20u8 {
            for v in (u + 1)..20u8 {
                dense_edges.push((u, v, i32::from(u) - i32::from(v)));
                if dense_edges.len() == 97 {
                    break 'outer;
                }
            }
        }
        let oversized_dense = FuzzBlossomVCase { order: 20, edges: dense_edges };
        check_blossom_v_invariants_fuzz(&oversized_dense);
    }

    #[test]
    fn test_blossom_v_matching_cost_panics_on_missing_edge() {
        let result = std::panic::catch_unwind(|| blossom_v_matching_cost(&[(0, 1, 3)], &[(0, 2)]));
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_blossom_v_matching_rejects_wrong_cardinality() {
        let result = std::panic::catch_unwind(|| {
            validate_blossom_v_matching(4, &[(0, 1, 3), (2, 3, 4)], &[(0, 1)]);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_blossom_v_matching_rejects_out_of_bounds_vertices() {
        let result = std::panic::catch_unwind(|| {
            validate_blossom_v_matching(4, &[(0, 1, 3), (1, 2, 4)], &[(0, 4), (1, 2)]);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_blossom_v_matching_rejects_duplicate_vertices() {
        let result = std::panic::catch_unwind(|| {
            validate_blossom_v_matching(4, &[(0, 1, 3), (0, 2, 4)], &[(0, 1), (0, 2)]);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_blossom_v_matching_rejects_non_edges() {
        let result = std::panic::catch_unwind(|| {
            validate_blossom_v_matching(4, &[(0, 1, 3), (2, 3, 4)], &[(0, 2), (1, 3)]);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_check_blossom_v_invariants_with_bruteforce_limit_accepts_small_optimal_case() {
        let case = FuzzBlossomVCase {
            order: 4,
            edges: vec![(0, 1, 1), (2, 3, 1), (0, 2, 10), (1, 3, 10)],
        };
        check_blossom_v_invariants_with_bruteforce_limit(&case, 12);
    }

    #[test]
    fn test_check_blossom_v_invariants_with_bruteforce_limit_handles_small_infeasible_case() {
        let case = FuzzBlossomVCase { order: 4, edges: vec![(0, 1, 1)] };
        check_blossom_v_invariants_with_bruteforce_limit(&case, 12);
    }

    #[test]
    fn test_check_blossom_v_invariants_with_bruteforce_limit_handles_large_infeasible_case() {
        let case = FuzzBlossomVCase { order: 14, edges: vec![(0, 1, 1), (2, 3, 1)] };
        check_blossom_v_invariants_with_bruteforce_limit(&case, 10);
    }

    #[test]
    fn test_check_blossom_v_invariants_with_bruteforce_limit_wraps_panics() {
        let case = FuzzBlossomVCase { order: 3, edges: vec![(0, 1, 1)] };
        let result = std::panic::catch_unwind(|| {
            check_blossom_v_invariants_with_bruteforce_limit(&case, 12);
        });

        let panic = result.expect_err("odd-order case should panic");
        let msg = if let Some(s) = panic.downcast_ref::<String>() {
            s.clone()
        } else if let Some(s) = panic.downcast_ref::<&str>() {
            (*s).to_string()
        } else {
            String::new()
        };
        assert!(msg.contains("Blossom V panicked"));
    }

    #[test]
    fn test_check_structured_blossom_v_invariants_smoke() {
        let case = FuzzStructuredBlossomVCase {
            order: 8,
            family: 13,
            weight_mode: 3,
            ensure_perfect_support: true,
            seed: 0x3141_5926_5358_9793,
        };
        check_structured_blossom_v_invariants(&case);
    }

    mod coverage_submodule {
        use super::*;

        struct AsymmetricSimilarity;

        impl ScalarSimilarity<u8, u8> for AsymmetricSimilarity {
            type Similarity = f64;

            fn similarity(&self, left: &u8, right: &u8) -> Self::Similarity {
                match (*left, *right) {
                    (0, 1) => 0.2,
                    (1, 0) => 0.8,
                    _ if left == right => 1.0,
                    _ => 0.5,
                }
            }
        }

        #[test]
        fn test_check_kahn_ordering_returns_for_size_guard() {
            let mut matrix: SquareCSR2D<CSR2D<u16, u8, u8>> =
                SquareCSR2D::with_sparse_shaped_capacity(3, 2);
            matrix.extend(vec![(0, 1), (1, 2)]).expect("extend matrix");
            check_kahn_ordering(&matrix, 2);
        }

        #[test]
        #[should_panic(expected = "sim(0, 1)")]
        fn test_check_similarity_invariants_panics_for_asymmetry() {
            check_similarity_invariants(&AsymmetricSimilarity, &[0u8, 1u8], 2);
        }

        #[test]
        fn test_asymmetric_similarity_default_arm() {
            let similarity = AsymmetricSimilarity;
            assert!((similarity.similarity(&2, &3) - 0.5).abs() <= f64::EPSILON);
        }

        #[test]
        #[should_panic(expected = "row index out of bounds")]
        fn test_validate_lap_assignment_panics_on_row_out_of_bounds() {
            let csr = sample_valued_csr_f64();
            validate_lap_assignment(&csr, &[(2, 0)], "row_oob");
        }

        #[test]
        #[should_panic(expected = "column index out of bounds")]
        fn test_validate_lap_assignment_panics_on_column_out_of_bounds() {
            let csr = sample_valued_csr_f64();
            validate_lap_assignment(&csr, &[(0, 2)], "column_oob");
        }

        #[test]
        fn test_check_louvain_invariants_returns_for_unstable_weights() {
            let mut csr: ValuedCSR2D<u16, u8, u8, f64> =
                ValuedCSR2D::with_sparse_shaped_capacity((2, 2), 2);
            MatrixMut::add(&mut csr, (0, 0, 1.0e-20)).expect("insert tiny value");
            MatrixMut::add(&mut csr, (1, 1, 1.0e20)).expect("insert huge value");
            check_louvain_invariants(&csr);
        }

        #[test]
        fn test_check_leiden_invariants_returns_for_unstable_weights() {
            let mut csr: ValuedCSR2D<u16, u8, u8, f64> =
                ValuedCSR2D::with_sparse_shaped_capacity((2, 2), 2);
            MatrixMut::add(&mut csr, (0, 0, 1.0e-20)).expect("insert tiny value");
            MatrixMut::add(&mut csr, (1, 1, 1.0e20)).expect("insert huge value");
            check_leiden_invariants(&csr);
        }

        #[test]
        fn test_check_floyd_warshall_invariants_non_square_error_path() {
            let mut csr: ValuedCSR2D<u16, u8, u8, f64> =
                ValuedCSR2D::with_sparse_shaped_capacity((2, 3), 2);
            MatrixMut::add(&mut csr, (0, 1, 1.0)).expect("insert edge");
            MatrixMut::add(&mut csr, (1, 2, 2.0)).expect("insert edge");
            check_floyd_warshall_invariants(&csr);
        }

        #[test]
        fn test_check_pairwise_dijkstra_matches_floyd_warshall_non_square_error_path() {
            let mut csr: ValuedCSR2D<u16, u8, u8, f64> =
                ValuedCSR2D::with_sparse_shaped_capacity((2, 3), 2);
            MatrixMut::add(&mut csr, (0, 1, 1.0)).expect("insert edge");
            MatrixMut::add(&mut csr, (1, 2, 2.0)).expect("insert edge");
            check_pairwise_dijkstra_matches_floyd_warshall(&csr);
        }

        #[test]
        fn test_check_mds_invariants_returns_for_negative_distance() {
            let csr = build_valued_csr_f64(
                (2, 2),
                &[(0, 0, 0.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 0.0)],
            );
            check_mds_invariants(&csr);
        }

        #[test]
        fn test_check_mds_invariants_returns_for_non_symmetric_matrix() {
            let csr =
                build_valued_csr_f64((2, 2), &[(0, 0, 0.0), (0, 1, 1.0), (1, 0, 2.0), (1, 1, 0.0)]);
            check_mds_invariants(&csr);
        }

        #[test]
        fn test_check_gth_invariants_handles_non_finite_and_zero_rows() {
            let matrix = VecMatrix2D::new(
                3,
                3,
                vec![f64::NAN, f64::INFINITY, -f64::INFINITY, 0.0, 0.0, 0.0, -3.0, 1.0, 2.0],
            );
            check_gth_invariants(&matrix);
        }

        #[test]
        fn test_check_gth_invariants_returns_for_non_square_matrix() {
            let matrix = VecMatrix2D::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            check_gth_invariants(&matrix);
        }

        #[test]
        fn test_check_floyd_warshall_invariants_non_finite_weight_path() {
            let csr = build_valued_csr_f64(
                (2, 2),
                &[(0, 0, 0.0), (0, 1, f64::NAN), (1, 0, 1.0), (1, 1, 0.0)],
            );
            check_floyd_warshall_invariants(&csr);
        }

        #[test]
        fn test_check_floyd_warshall_invariants_negative_cycle_path() {
            let csr = build_valued_csr_f64((2, 2), &[(0, 1, -1.0), (1, 0, -1.0)]);
            check_floyd_warshall_invariants(&csr);
        }

        #[test]
        fn test_check_pairwise_dijkstra_matches_floyd_warshall_non_finite_weight_path() {
            let csr = build_valued_csr_f64(
                (2, 2),
                &[(0, 0, 0.0), (0, 1, f64::NAN), (1, 0, 1.0), (1, 1, 0.0)],
            );
            check_pairwise_dijkstra_matches_floyd_warshall(&csr);
        }

        #[test]
        fn test_check_pairwise_dijkstra_matches_floyd_warshall_negative_weight_path() {
            let csr = build_valued_csr_f64((2, 2), &[(0, 0, 0.0), (0, 1, -1.0), (1, 1, 0.0)]);
            check_pairwise_dijkstra_matches_floyd_warshall(&csr);
        }

        #[test]
        fn test_check_line_graph_invariants_smoke() {
            let mut matrix: SquareCSR2D<CSR2D<u16, u8, u8>> =
                SquareCSR2D::with_sparse_shaped_capacity(4, 4);
            matrix.extend(vec![(0, 1), (0, 2), (1, 2), (2, 3)]).expect("extend matrix");
            let graph: GenericGraph<u8, _> = GenericGraph::from((4u8, matrix));
            check_line_graph_invariants(&graph, 32);
        }

        #[test]
        fn test_check_line_graph_invariants_returns_for_size_guard() {
            let matrix: SquareCSR2D<CSR2D<u16, u8, u8>> =
                SquareCSR2D::with_sparse_shaped_capacity(4, 0);
            let graph: GenericGraph<u8, _> = GenericGraph::from((4u8, matrix));
            check_line_graph_invariants(&graph, 2);
        }

        #[test]
        fn test_check_louvain_invariants_returns_when_symmetrized_edge_count_overflows_u8() {
            let mut csr: ValuedCSR2D<u16, u8, u8, f64> =
                ValuedCSR2D::with_sparse_shaped_capacity((17, 17), 17 * 17);

            for row in 0u8..17 {
                for column in 0u8..17 {
                    MatrixMut::add(&mut csr, (row, column, 1.0)).expect("insert dense edge");
                }
            }

            check_louvain_invariants(&csr);
        }

        #[test]
        fn test_check_blossom_v_invariants_smoke() {
            let case = FuzzBlossomVCase { order: 2, edges: vec![(0, 1, -7)] };
            check_blossom_v_invariants(&case);
        }

        #[test]
        fn test_check_blossom_v_invariants_empty_smoke() {
            let case = FuzzBlossomVCase { order: 0, edges: vec![] };
            check_blossom_v_invariants(&case);
        }
    }
}
