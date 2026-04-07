use alloc::vec::Vec;

use num_traits::{AsPrimitive, cast};
use thiserror::Error;

use super::NodeSorter;
use crate::traits::{MonopartiteEdges, UndirectedMonopartiteMonoplexGraph};

/// LAW-compatible default gamma ladder for layered label propagation.
///
/// This is the fixed multiresolution schedule used by the reference
/// implementation: powers of one half from `1.0` down to `1 / 1024`, followed
/// by `0.0`.
pub const LAYERED_LABEL_PROPAGATION_DEFAULT_GAMMAS: [f64; 12] = [
    1.0,
    1.0 / 2.0,
    1.0 / 4.0,
    1.0 / 8.0,
    1.0 / 16.0,
    1.0 / 32.0,
    1.0 / 64.0,
    1.0 / 128.0,
    1.0 / 256.0,
    1.0 / 512.0,
    1.0 / 1024.0,
    0.0,
];

const DEFAULT_MAX_UPDATES: usize = 100;
const DEFAULT_SEED: u64 = 42;
const GAIN_THRESHOLD: f64 = 0.001;
const SHUFFLE_GRANULARITY: usize = 100_000;
const SPLITMIX_PHI: u64 = 0x9E37_79B9_7F4A_7C15;

/// Small LAW-compatible RNG shim.
///
/// The reference implementation uses a xorshiro128+-family generator together
/// with Java/fastutil bounded draws and shuffles. We keep that behavior local
/// here so the LLP port stays deterministic against the LAW oracles.
#[derive(Clone, Debug, PartialEq, Eq)]
struct LawRng {
    s0: u64,
    s1: u64,
}

impl LawRng {
    #[inline]
    fn new(seed: u64) -> Self {
        // LAW hashes the user seed before expanding it into the xoroshiro state.
        let mut splitmix_state = murmur_hash3(seed);
        Self { s0: splitmix64_next(&mut splitmix_state), s1: splitmix64_next(&mut splitmix_state) }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let s0 = self.s0;
        let mut s1 = self.s1;
        let result = s0.wrapping_add(s1);
        s1 ^= s0;
        self.s0 = s0.rotate_left(24) ^ s1 ^ (s1 << 16);
        self.s1 = s1.rotate_left(37);
        result
    }

    #[inline]
    fn next_usize_bound(&mut self, bound: usize) -> usize {
        let bound_u64 = u64::try_from(bound).expect("usize bounds must fit into u64 for LAW RNG");
        usize::try_from(self.next_u64_bound(bound_u64))
            .expect("LAW RNG draws bounded by usize must fit back into usize")
    }

    #[inline]
    fn next_u64_bound(&mut self, bound: u64) -> u64 {
        assert!(bound > 0, "LAW RNG requires a positive bound");

        // This mirrors Java's bounded `nextLong(n)` behavior, including the
        // fact that `bound == 1` still advances the RNG state.
        let mut t = self.next_u64();
        let n_minus_1 = bound - 1;
        if (bound & n_minus_1) == 0 {
            return t.checked_shr(n_minus_1.leading_zeros()).unwrap_or(t) & n_minus_1;
        }

        let mut u = t >> 1;
        t = u % bound;
        while u.wrapping_add(n_minus_1).wrapping_sub(t) >> 63 != 0 {
            u = self.next_u64() >> 1;
            t = u % bound;
        }
        t
    }

    #[inline]
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        // Reverse Fisher-Yates matches fastutil's array shuffles.
        for index in (0..slice.len()).rev() {
            let other = self.next_usize_bound(index + 1);
            slice.swap(index, other);
        }
    }
}

#[inline]
fn murmur_hash3(mut value: u64) -> u64 {
    value ^= value >> 33;
    value = value.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    value ^= value >> 33;
    value = value.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    value ^ (value >> 33)
}

#[inline]
fn stafford_mix13(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[inline]
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(SPLITMIX_PHI);
    stafford_mix13(*state)
}

#[derive(Clone, Debug, PartialEq)]
struct GammaRun {
    labels: Vec<usize>,
    gap_cost: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RunConfig<'a> {
    gamma: f64,
    max_updates: usize,
    seed: u64,
    start_rank: Option<&'a [usize]>,
    start_order: Option<&'a [usize]>,
    exact: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct LabelPropagationState {
    labels: Vec<usize>,
    volumes: Vec<usize>,
    update_list: Vec<usize>,
    can_change: Vec<bool>,
    objective: f64,
    modified: usize,
}

impl LabelPropagationState {
    #[inline]
    fn new(number_of_nodes: usize) -> Self {
        Self {
            labels: (0..number_of_nodes).collect(),
            volumes: vec![1; number_of_nodes],
            update_list: (0..number_of_nodes).collect(),
            can_change: vec![true; number_of_nodes],
            objective: 0.0,
            modified: 0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        for (index, label) in self.labels.iter_mut().enumerate() {
            *label = index;
        }
        self.volumes.fill(1);
        self.can_change.fill(true);
        self.objective = 0.0;
        self.modified = 0;
        for (index, node) in self.update_list.iter_mut().enumerate() {
            *node = index;
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct LabelPropagationScratch {
    counter: OpenHashCounter,
    majorities: Vec<usize>,
    permuted_neighbors: Vec<usize>,
}

/// Layered label propagation node-ordering algorithm.
///
/// This sorter implements the multiresolution ordering procedure from the LAW
/// `LayeredLabelPropagation` reference implementation. For each `gamma`, it
/// runs label propagation from singleton labels, scores the resulting labeling
/// by gap cost, and then composes the labelings into one final ordering.
///
/// The graph must be undirected and loopless. Symmetry is enforced by the
/// graph trait bound; self-loops are rejected at runtime.
#[derive(Clone, Debug, PartialEq)]
pub struct LayeredLabelPropagationSorter {
    gammas: Vec<f64>,
    max_updates: usize,
    seed: u64,
    start_order: Option<Vec<usize>>,
    exact: bool,
}

/// Errors returned while constructing a
/// [`LayeredLabelPropagationSorter`].
#[derive(Clone, Debug, Error, PartialEq)]
pub enum LayeredLabelPropagationError {
    /// The gamma ladder must contain at least one value.
    #[error("LayeredLabelPropagationSorter requires at least one gamma value")]
    EmptyGammas,
    /// Every gamma must be finite and non-negative.
    #[error("LayeredLabelPropagationSorter requires finite, non-negative gamma values")]
    InvalidGamma,
    /// The maximum number of label-propagation updates must be positive.
    #[error("LayeredLabelPropagationSorter requires max_updates to be greater than zero")]
    InvalidMaxUpdates,
}

impl LayeredLabelPropagationSorter {
    /// Creates a sorter with the provided parameters.
    ///
    /// `start_order` uses the crate's node-order semantics: the item at
    /// position `i` is the original node id that should appear at new position
    /// `i`. When `exact` is `true` and a `start_order` is provided, the result
    /// matches running LLP on the graph reordered by `start_order`.
    ///
    /// # Examples
    /// ```
    /// # #[cfg(feature = "alloc")] {
    /// use geometric_traits::{
    ///     impls::{CSR2D, SymmetricCSR2D},
    ///     naive_structs::GenericGraph,
    ///     prelude::*,
    ///     traits::{
    ///         EdgesBuilder,
    ///         algorithms::{LayeredLabelPropagationSorter, NodeSorter},
    ///     },
    /// };
    ///
    /// type UndirectedGraph = SymmetricCSR2D<CSR2D<usize, usize, usize>>;
    /// type UndirectedVecGraph = GenericGraph<Vec<usize>, UndirectedGraph>;
    ///
    /// let matrix: UndirectedGraph = UndiEdgesBuilder::default()
    ///     .expected_number_of_edges(4)
    ///     .expected_shape(5)
    ///     .edges([(0, 1), (1, 2), (2, 3), (3, 4)].into_iter())
    ///     .build()
    ///     .unwrap();
    /// let graph: UndirectedVecGraph = GenericGraph::from(((0..5).collect::<Vec<_>>(), matrix));
    ///
    /// let sorter =
    ///     LayeredLabelPropagationSorter::new(vec![1.0, 0.5, 0.0], 10, 7, None, false).unwrap();
    /// let mut order = sorter.sort_nodes(&graph);
    /// order.sort_unstable();
    ///
    /// assert_eq!(order, vec![0, 1, 2, 3, 4]);
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `gammas` is empty;
    /// - at least one gamma is not finite or is negative;
    /// - `max_updates == 0`.
    #[inline]
    pub fn new(
        gammas: Vec<f64>,
        max_updates: usize,
        seed: u64,
        start_order: Option<Vec<usize>>,
        exact: bool,
    ) -> Result<Self, LayeredLabelPropagationError> {
        validate_config(&gammas, max_updates)?;
        Ok(Self { gammas, max_updates, seed, start_order, exact })
    }
}

impl Default for LayeredLabelPropagationSorter {
    #[inline]
    fn default() -> Self {
        Self::new(
            LAYERED_LABEL_PROPAGATION_DEFAULT_GAMMAS.to_vec(),
            DEFAULT_MAX_UPDATES,
            DEFAULT_SEED,
            None,
            false,
        )
        .expect("LAW default LLP configuration must be valid")
    }
}

impl<G> NodeSorter<G> for LayeredLabelPropagationSorter
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    fn sort_nodes(&self, graph: &G) -> Vec<G::NodeId> {
        assert!(
            !graph.edges().has_self_loops(),
            "LayeredLabelPropagationSorter requires a loopless graph"
        );

        let node_ids: Vec<G::NodeId> = graph.node_ids().collect();
        let node_count = node_ids.len();
        let start_rank = validate_and_invert_start_order(self.start_order.as_deref(), node_count);
        let start_order = start_rank.as_ref().map(|rank| invert_order(rank));
        let mut shuffle_rng = LawRng::new(self.seed);
        let mut state = LabelPropagationState::new(node_count);
        let mut scratch = LabelPropagationScratch::default();
        let mut gamma_runs = Vec::with_capacity(self.gammas.len());
        for &gamma in &self.gammas {
            let config = RunConfig {
                gamma,
                max_updates: self.max_updates,
                seed: self.seed,
                start_rank: start_rank.as_deref(),
                start_order: start_order.as_deref(),
                exact: self.exact,
            };
            gamma_runs.push(run_gamma(
                graph,
                &node_ids,
                &mut state,
                &mut scratch,
                &mut shuffle_rng,
                config,
            ));
        }

        compose_final_order(&gamma_runs, start_rank.as_deref())
            .into_iter()
            .map(|node| node_ids[node])
            .collect()
    }
}

#[inline]
fn validate_config(gammas: &[f64], max_updates: usize) -> Result<(), LayeredLabelPropagationError> {
    if gammas.is_empty() {
        return Err(LayeredLabelPropagationError::EmptyGammas);
    }
    if max_updates == 0 {
        return Err(LayeredLabelPropagationError::InvalidMaxUpdates);
    }
    for &gamma in gammas {
        if !gamma.is_finite() || gamma < 0.0 {
            return Err(LayeredLabelPropagationError::InvalidGamma);
        }
    }
    Ok(())
}

#[inline]
fn validate_and_invert_start_order(
    start_order: Option<&[usize]>,
    node_count: usize,
) -> Option<Vec<usize>> {
    let start_order = start_order?;
    assert_eq!(
        start_order.len(),
        node_count,
        "LayeredLabelPropagationSorter start_order must contain exactly one entry per node"
    );

    let mut seen = vec![false; node_count];
    let mut start_rank = vec![0usize; node_count];
    for (position, &node) in start_order.iter().enumerate() {
        assert!(
            node < node_count,
            "LayeredLabelPropagationSorter start_order contains out-of-range node {node}"
        );
        assert!(
            !seen[node],
            "LayeredLabelPropagationSorter start_order contains duplicate node {node}"
        );
        seen[node] = true;
        start_rank[node] = position;
    }
    Some(start_rank)
}

#[inline]
fn run_gamma<G>(
    graph: &G,
    node_ids: &[G::NodeId],
    state: &mut LabelPropagationState,
    scratch: &mut LabelPropagationScratch,
    shuffle_rng: &mut LawRng,
    config: RunConfig<'_>,
) -> GammaRun
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    state.reset();

    for _update in 0..config.max_updates {
        let previous_objective = state.objective;
        update_labels(graph, node_ids, state, scratch, shuffle_rng, config);
        // LAW stops once a pass makes no moves or fails to improve the
        // objective by more than the fixed gain threshold.
        let gain = 1.0 - (previous_objective / state.objective);
        if state.modified == 0
            || !matches!(gain.partial_cmp(&GAIN_THRESHOLD), Some(core::cmp::Ordering::Greater))
        {
            break;
        }
    }

    let order = order_for_gap_cost(&state.labels, config.start_rank);
    let rank = invert_order(&order);
    GammaRun {
        labels: state.labels.clone(),
        gap_cost: gap_cost(graph, node_ids, &rank, &mut scratch.permuted_neighbors),
    }
}

#[inline]
fn update_labels<G>(
    graph: &G,
    node_ids: &[G::NodeId],
    state: &mut LabelPropagationState,
    scratch: &mut LabelPropagationScratch,
    shuffle_rng: &mut LawRng,
    config: RunConfig<'_>,
) where
    G: UndirectedMonopartiteMonoplexGraph,
{
    state.modified = 0;
    if config.exact {
        // The exact mode replays the inherited order directly instead of using
        // the shuffled schedule, matching LAW's "exact" constructor.
        if let Some(start_order) = config.start_order {
            state.update_list.clone_from_slice(start_order);
        } else {
            for (index, node) in state.update_list.iter_mut().enumerate() {
                *node = index;
            }
        }
    }

    // LAW shuffles the visit schedule in large chunks rather than building a
    // fresh random permutation for every pass.
    for block_start in (0..state.update_list.len()).step_by(SHUFFLE_GRANULARITY) {
        let block_end = (block_start + SHUFFLE_GRANULARITY).min(state.update_list.len());
        shuffle_rng.shuffle(&mut state.update_list[block_start..block_end]);
    }

    let mut objective = state.objective;
    // In the single-thread adaptation we rebuild the tie-break RNG from the
    // same seed for each propagation pass, as LAW does for each update round.
    let mut tie_break_rng = LawRng::new(config.seed);

    for index in 0..state.update_list.len() {
        let node = state.update_list[index];
        if !state.can_change[node] {
            continue;
        }
        state.can_change[node] = false;

        let node_id = node_ids[node];
        let degree: usize = graph.degree(node_id).as_();
        if degree == 0 {
            continue;
        }

        let current_label = state.labels[node];
        state.volumes[current_label] = state.volumes[current_label].saturating_sub(1);

        scratch.counter.clear(degree);
        for neighbor in graph.neighbors(node_id) {
            scratch.counter.incr(state.labels[neighbor.as_()]);
        }
        // Staying in the current label must remain a legal choice even when no
        // neighbor currently exposes it.
        if !scratch.counter.contains_key(current_label) {
            scratch.counter.add_zero_count(current_label);
        }

        scratch.majorities.clear();
        let mut best_value = f64::NEG_INFINITY;
        let mut current_value = 0.0;

        for (label, frequency) in scratch.counter.entries() {
            let frequency_f64 = usize_to_f64(frequency);
            let other_volume = usize_to_f64(state.volumes[label] + 1 - frequency);
            let value = frequency_f64 - config.gamma * other_volume;
            if value > best_value {
                scratch.majorities.clear();
                best_value = value;
                scratch.majorities.push(label);
            } else if value.total_cmp(&best_value).is_eq() {
                scratch.majorities.push(label);
            }

            if label == current_label {
                current_value = value;
            }
        }

        if config.exact {
            if let Some(start_rank) = config.start_rank {
                scratch.majorities.sort_unstable_by_key(|&label| start_rank[label]);
            } else {
                scratch.majorities.sort_unstable();
            }
        }

        let next_label =
            scratch.majorities[tie_break_rng.next_usize_bound(scratch.majorities.len())];
        if next_label != current_label {
            state.modified += 1;
            for neighbor in graph.neighbors(node_id) {
                state.can_change[neighbor.as_()] = true;
            }
        }

        state.labels[node] = next_label;
        state.volumes[next_label] += 1;
        objective += best_value - current_value;
    }

    state.objective = objective;
}

#[inline]
fn compose_final_order(gamma_runs: &[GammaRun], start_rank: Option<&[usize]>) -> Vec<usize> {
    debug_assert!(!gamma_runs.is_empty());

    let mut ordered_gamma_indices: Vec<usize> = (0..gamma_runs.len()).collect();
    ordered_gamma_indices.sort_unstable_by(|left, right| {
        gamma_runs[*right].gap_cost.total_cmp(&gamma_runs[*left].gap_cost)
    });
    let best_gamma = *ordered_gamma_indices.last().expect("gamma_runs must not be empty");

    // LAW starts from the best gap-cost labeling and then repeatedly refines it
    // with each gamma pass, re-aligning against the best labeling after every
    // composition step.
    let mut composed_labels = gamma_runs[best_gamma].labels.clone();
    if let Some(start_rank) = start_rank {
        for label in &mut composed_labels {
            *label = start_rank[*label];
        }
    }

    let mut support = vec![0usize; composed_labels.len()];
    for gamma_index in ordered_gamma_indices {
        combine_labels(
            &mut composed_labels,
            &gamma_runs[gamma_index].labels,
            start_rank,
            &mut support,
        );
        combine_labels(
            &mut composed_labels,
            &gamma_runs[best_gamma].labels,
            start_rank,
            &mut support,
        );
    }

    order_from_labels(&composed_labels, start_rank)
}

#[inline]
fn order_from_labels(labels: &[usize], start_rank: Option<&[usize]>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..labels.len()).collect();
    if let Some(start_rank) = start_rank {
        order.sort_unstable_by(|left, right| {
            labels[*left]
                .cmp(&labels[*right])
                .then_with(|| start_rank[*left].cmp(&start_rank[*right]))
        });
    } else {
        order.sort_unstable_by(|left, right| {
            labels[*left].cmp(&labels[*right]).then_with(|| left.cmp(right))
        });
    }
    order
}

#[inline]
fn order_for_gap_cost(labels: &[usize], start_rank: Option<&[usize]>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..labels.len()).collect();
    if let Some(start_rank) = start_rank {
        order.sort_unstable_by(|left, right| {
            start_rank[labels[*left]]
                .cmp(&start_rank[labels[*right]])
                .then_with(|| start_rank[*left].cmp(&start_rank[*right]))
        });
    } else {
        order.sort_unstable_by(|left, right| {
            labels[*left].cmp(&labels[*right]).then_with(|| left.cmp(right))
        });
    }
    order
}

#[inline]
fn invert_order(order: &[usize]) -> Vec<usize> {
    let mut rank = vec![0usize; order.len()];
    for (position, &node) in order.iter().enumerate() {
        rank[node] = position;
    }
    rank
}

#[inline]
fn gap_cost<G>(
    graph: &G,
    node_ids: &[G::NodeId],
    rank: &[usize],
    permuted_neighbors: &mut Vec<usize>,
) -> f64
where
    G: UndirectedMonopartiteMonoplexGraph,
{
    let mut total = 0.0;

    for (node, &node_id) in node_ids.iter().enumerate() {
        if graph.degree(node_id).as_() == 0 {
            continue;
        }
        permuted_neighbors.clear();
        permuted_neighbors.extend(graph.neighbors(node_id).map(|neighbor| rank[neighbor.as_()]));
        permuted_neighbors.sort_unstable();

        // LAW evaluates locality by walking neighbors in rank order and
        // summing ceil(log2(distance)) between consecutive positions.
        let mut previous = rank[node];
        for &neighbor_rank in permuted_neighbors.iter() {
            total += f64::from(ceil_log2_usize(previous.abs_diff(neighbor_rank)));
            previous = neighbor_rank;
        }
    }

    total
}

#[inline]
fn ceil_log2_usize(value: usize) -> u32 {
    debug_assert!(value > 0, "gap cost requires a strictly positive node distance");
    usize::BITS - (value - 1).leading_zeros()
}

#[inline]
fn combine_labels(
    labels: &mut [usize],
    major: &[usize],
    start_rank: Option<&[usize]>,
    support: &mut [usize],
) -> usize {
    let number_of_nodes = labels.len();
    if number_of_nodes == 0 {
        return 0;
    }
    assert_eq!(number_of_nodes, major.len(), "labels and major labelings must have equal length");
    assert_eq!(
        number_of_nodes,
        support.len(),
        "support buffer must be sized like the input labels"
    );

    for (index, slot) in support.iter_mut().enumerate() {
        *slot = index;
    }

    // Sorting by the major labeling first, then by the inherited minor labels,
    // is what makes each refinement keep blocks contiguous without inventing a
    // brand-new order inside a block.
    if let Some(start_rank) = start_rank {
        support.sort_unstable_by(|left, right| {
            labels[major[*left]]
                .cmp(&labels[major[*right]])
                .then_with(|| start_rank[major[*left]].cmp(&start_rank[major[*right]]))
                .then_with(|| labels[*left].cmp(&labels[*right]))
                .then_with(|| left.cmp(right))
        });
    } else {
        support.sort_unstable_by(|left, right| {
            labels[major[*left]]
                .cmp(&labels[major[*right]])
                .then_with(|| major[*left].cmp(&major[*right]))
                .then_with(|| labels[*left].cmp(&labels[*right]))
                .then_with(|| left.cmp(right))
        });
    }

    let first = support[0];
    let mut current_minor = labels[first];
    let mut current_major = major[first];
    let mut current_label = 0usize;
    labels[first] = current_label;

    for &node in support.iter().skip(1) {
        let minor = labels[node];
        if major[node] != current_major || minor != current_minor {
            current_major = major[node];
            current_minor = minor;
            current_label += 1;
        }
        labels[node] = current_label;
    }

    current_label + 1
}

/// Reusable neighborhood frequency table specialized for one LLP update pass.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct OpenHashCounter {
    keys: Vec<usize>,
    counts: Vec<usize>,
    used_positions: Vec<usize>,
    mask: usize,
}

impl OpenHashCounter {
    #[inline]
    fn clear(&mut self, size_hint: usize) {
        let target_capacity = (size_hint.max(1) * 2).next_power_of_two();
        if self.counts.len() < target_capacity {
            self.keys = vec![0usize; target_capacity];
            self.counts = vec![0usize; target_capacity];
            self.used_positions = Vec::with_capacity(size_hint.max(1));
            self.mask = target_capacity - 1;
            return;
        }

        for &position in &self.used_positions {
            self.counts[position] = 0;
        }
        self.used_positions.clear();
    }

    #[inline]
    fn incr(&mut self, key: usize) {
        let mut position = hash_key(key) & self.mask;
        while self.counts[position] != 0 && self.keys[position] != key {
            position = (position + 1) & self.mask;
        }
        if self.counts[position] == 0 {
            self.keys[position] = key;
            self.used_positions.push(position);
        }
        self.counts[position] += 1;
    }

    #[inline]
    fn contains_key(&self, key: usize) -> bool {
        let mut position = hash_key(key) & self.mask;
        while self.counts[position] != 0 && self.keys[position] != key {
            position = (position + 1) & self.mask;
        }
        self.counts[position] != 0
    }

    #[inline]
    fn add_zero_count(&mut self, key: usize) {
        let mut position = hash_key(key) & self.mask;
        while self.counts[position] != 0 && self.keys[position] != key {
            position = (position + 1) & self.mask;
        }
        if self.counts[position] == 0 {
            self.keys[position] = key;
            self.used_positions.push(position);
        }
    }

    #[inline]
    fn entries(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.used_positions.iter().map(|&position| (self.keys[position], self.counts[position]))
    }
}

#[inline]
const fn hash_key(key: usize) -> usize {
    key.wrapping_mul(2_056_437_379usize)
}

#[inline]
fn usize_to_f64(value: usize) -> f64 {
    cast::<usize, f64>(value).expect("graph sizes and label counts must fit into f64")
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{
        DEFAULT_MAX_UPDATES, DEFAULT_SEED, LAYERED_LABEL_PROPAGATION_DEFAULT_GAMMAS,
        LabelPropagationScratch, LabelPropagationState, LawRng, LayeredLabelPropagationError,
        LayeredLabelPropagationSorter, RunConfig, combine_labels, invert_order, order_for_gap_cost,
        run_gamma,
    };
    use crate::{
        impls::{CSR2D, SymmetricCSR2D},
        naive_structs::GenericGraph,
        prelude::UndiEdgesBuilder,
        traits::{EdgesBuilder, MonopartiteGraph},
    };

    type UndirectedGraph = SymmetricCSR2D<CSR2D<usize, usize, usize>>;
    type UndirectedVecGraph = GenericGraph<Vec<usize>, UndirectedGraph>;

    fn build_branching_tree() -> UndirectedVecGraph {
        let matrix: UndirectedGraph = UndiEdgesBuilder::default()
            .expected_number_of_edges(5)
            .expected_shape(6)
            .edges([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)].into_iter())
            .build()
            .unwrap();
        GenericGraph::from(((0..6).collect::<Vec<_>>(), matrix))
    }

    #[test]
    fn test_llp_new_matches_default_sorter_for_law_defaults() {
        assert_eq!(
            LayeredLabelPropagationSorter::new(
                LAYERED_LABEL_PROPAGATION_DEFAULT_GAMMAS.to_vec(),
                DEFAULT_MAX_UPDATES,
                DEFAULT_SEED,
                None,
                false,
            )
            .unwrap(),
            LayeredLabelPropagationSorter::default()
        );
    }

    #[test]
    fn test_llp_default_matches_law_reference() {
        let sorter = LayeredLabelPropagationSorter::default();
        assert_eq!(sorter.gammas, LAYERED_LABEL_PROPAGATION_DEFAULT_GAMMAS.to_vec());
        assert_eq!(sorter.max_updates, DEFAULT_MAX_UPDATES);
        assert_eq!(sorter.seed, DEFAULT_SEED);
        assert!(!sorter.exact);
        assert!(sorter.start_order.is_none());
    }

    #[test]
    fn test_llp_new_rejects_invalid_configuration() {
        assert_eq!(
            LayeredLabelPropagationSorter::new(Vec::new(), 1, DEFAULT_SEED, None, false),
            Err(LayeredLabelPropagationError::EmptyGammas)
        );
        assert_eq!(
            LayeredLabelPropagationSorter::new(vec![-1.0], 1, DEFAULT_SEED, None, false),
            Err(LayeredLabelPropagationError::InvalidGamma)
        );
        assert_eq!(
            LayeredLabelPropagationSorter::new(vec![1.0], 0, DEFAULT_SEED, None, false),
            Err(LayeredLabelPropagationError::InvalidMaxUpdates)
        );
    }

    #[test]
    fn test_combine_labels_keeps_refined_blocks_contiguous() {
        let mut labels = vec![0usize, 0, 1, 1];
        let major = vec![0usize, 1, 0, 1];
        let mut support = vec![0usize; labels.len()];

        let number_of_labels = combine_labels(&mut labels, &major, None, &mut support);

        assert_eq!(number_of_labels, 4);
        let mut sorted_labels = labels.clone();
        sorted_labels.sort_unstable();
        assert_eq!(sorted_labels, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_gap_cost_order_uses_start_order_on_cluster_labels() {
        let labels = vec![1usize, 2, 0];
        let start_order = vec![2usize, 0, 1];
        let start_rank = invert_order(&start_order);

        assert_eq!(order_for_gap_cost(&labels, Some(&start_rank)), vec![1, 2, 0]);
    }

    #[test]
    fn test_law_rng_matches_java_reference_for_seed_7() {
        let mut rng = LawRng::new(7);
        assert_eq!(rng.next_u64(), 1_328_206_959_410_720_230u64);
        assert_eq!(rng.next_u64(), 5_935_505_810_707_603_556u64);
        assert_eq!(rng.next_u64(), 0xF503_0ADC_F95F_A43Fu64);
        assert_eq!(rng.next_u64(), 7_146_999_580_052_588_380u64);

        let mut bounded = LawRng::new(7);
        let draws: Vec<usize> = (0..8).map(|_| bounded.next_usize_bound(5)).collect();
        assert_eq!(draws, vec![0, 3, 4, 0, 2, 3, 0, 2]);

        let mut bounded = LawRng::new(7);
        let draws: Vec<usize> = (0..8).map(|_| bounded.next_usize_bound(3)).collect();
        assert_eq!(draws, vec![2, 1, 2, 2, 2, 0, 0, 2]);

        let mut bounded = LawRng::new(7);
        let draws: Vec<usize> = (0..8).map(|_| bounded.next_usize_bound(2)).collect();
        assert_eq!(draws, vec![0, 0, 1, 0, 0, 1, 1, 0]);

        let mut mixed = LawRng::new(7);
        assert_eq!(mixed.next_usize_bound(3), 2);
        assert_eq!(mixed.next_usize_bound(2), 0);
        assert_eq!(mixed.next_usize_bound(2), 1);
        assert_eq!(mixed.next_usize_bound(3), 2);

        let mut shuffled = [0usize, 1, 2, 3, 4, 5];
        let mut shuffle_rng = LawRng::new(7);
        shuffle_rng.shuffle(&mut shuffled);
        assert_eq!(shuffled, [1, 0, 2, 4, 3, 5]);
    }

    #[test]
    fn test_run_gamma_matches_law_branching_tree_reference() {
        let graph = build_branching_tree();
        let node_ids: Vec<_> = graph.node_ids().collect();
        let mut state = LabelPropagationState::new(node_ids.len());
        let mut scratch = LabelPropagationScratch::default();
        let mut shuffle_rng = LawRng::new(7);

        let gamma_run = run_gamma(
            &graph,
            &node_ids,
            &mut state,
            &mut scratch,
            &mut shuffle_rng,
            RunConfig {
                gamma: 1.0,
                max_updates: 100,
                seed: 7,
                start_rank: None,
                start_order: None,
                exact: false,
            },
        );

        assert_eq!(gamma_run.labels, vec![5, 4, 5, 4, 4, 5]);
    }

    #[test]
    fn test_run_gamma_first_update_matches_law_branching_tree_reference() {
        let graph = build_branching_tree();
        let node_ids: Vec<_> = graph.node_ids().collect();
        let mut state = LabelPropagationState::new(node_ids.len());
        let mut scratch = LabelPropagationScratch::default();
        let mut shuffle_rng = LawRng::new(7);

        let gamma_run = run_gamma(
            &graph,
            &node_ids,
            &mut state,
            &mut scratch,
            &mut shuffle_rng,
            RunConfig {
                gamma: 1.0,
                max_updates: 1,
                seed: 7,
                start_rank: None,
                start_order: None,
                exact: false,
            },
        );

        assert_eq!(gamma_run.labels, vec![2, 4, 5, 4, 4, 5]);
    }
}
