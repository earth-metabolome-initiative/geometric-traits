//! Vertex match inference from edge matches in the MCES pipeline.
//!
//! Given a maximum clique in the modular product of two line graphs (i.e.,
//! a set of matched edge pairs), this module infers which original graph
//! vertices correspond to each other.
//!
//! The algorithm follows the RASCAL approach (Raymond, Gardiner, Willett 2002)
//! with three phases:
//!
//! 1. **Shared endpoints** — adjacent matched edges share a vertex in each
//!    graph; those shared vertices are unambiguously matched.
//! 2. **Propagation** — for each matched edge where one endpoint is already
//!    resolved, the other is forced.
//! 3. **Disambiguation** — for isolated edges (neither endpoint resolved), a
//!    caller-provided closure decides the mapping.

use alloc::{
    collections::btree_map::{BTreeMap, Entry},
    vec::Vec,
};

/// Returns the shared endpoint of two edges, if they share exactly one.
///
/// Given edges `(a, b)` and `(c, d)`, returns `Some(v)` where `v` is the
/// single vertex common to both edges.
///
/// Returns `None` if the edges are disjoint (no shared endpoint) or share
/// both endpoints (same edge, possibly reversed).
///
/// # Examples
///
/// ```
/// use geometric_traits::prelude::shared_endpoint;
///
/// assert_eq!(shared_endpoint((1_u32, 2), (2, 3)), Some(2));
/// assert_eq!(shared_endpoint((1_u32, 2), (3, 1)), Some(1));
/// assert_eq!(shared_endpoint::<u32>((1, 2), (3, 4)), None);
/// assert_eq!(shared_endpoint::<u32>((1, 2), (1, 2)), None);
/// ```
#[inline]
#[must_use]
pub fn shared_endpoint<N: Eq + Copy>(edge1: (N, N), edge2: (N, N)) -> Option<N> {
    let (a, b) = edge1;
    let (c, d) = edge2;
    match (a == c, a == d, b == c, b == d) {
        (true, false, false, false) | (false, true, false, false) => Some(a),
        (false, false, true, false) | (false, false, false, true) => Some(b),
        _ => None,
    }
}

/// Inserts a vertex match into the map, asserting consistency on conflict.
#[inline]
fn try_insert<N1, N2>(map: &mut BTreeMap<N1, N2>, n1: N1, n2: N2)
where
    N1: Eq + Copy + Ord + core::fmt::Debug,
    N2: Eq + Copy + Ord + core::fmt::Debug,
{
    match map.entry(n1) {
        Entry::Vacant(e) => {
            e.insert(n2);
        }
        Entry::Occupied(e) => {
            debug_assert_eq!(
                *e.get(),
                n2,
                "conflicting vertex match: {n1:?} already mapped to {:?}, cannot map to {n2:?}",
                e.get()
            );
        }
    }
}

/// Infers vertex matches from edge matches in the MCES pipeline.
///
/// Given a maximum clique in the modular product of two line graphs, infers
/// which vertices of the original graphs are matched. Each clique member
/// represents a matched edge pair; this function resolves the implied vertex
/// correspondence through three phases:
///
/// 1. **Shared endpoints**: for all pairs of matched edges that share an
///    endpoint in both graphs, the shared vertices are matched.
/// 2. **Propagation**: for each matched edge where one endpoint is already
///    resolved, the other is forced.
/// 3. **Disambiguation**: for isolated edges (neither endpoint resolved), the
///    caller-provided closure decides the mapping.
///
/// # Parameters
///
/// - `clique` — indices into `vertex_pairs` (the maximum clique result).
/// - `vertex_pairs` — from [`ModularProductResult::vertex_pairs()`], maps
///   product vertex → `(lg1_vertex, lg2_vertex)`.
/// - `edge_map_first` — from [`LineGraphResult::edge_map()`] of the first
///   graph, maps line graph vertex → original edge `(src, dst)`.
/// - `edge_map_second` — same for the second graph.
/// - `disambiguate` — called for isolated edges as `disambiguate(a, b, c, d)`
///   where `(a, b)` is the G1 edge and `(c, d)` is the G2 edge. Returns `true`
///   for mapping `a↔c, b↔d`; `false` for `a↔d, b↔c`.
///
/// # Returns
///
/// A sorted `Vec<(N1, N2)>` of matched vertex pairs, deduplicated.
///
/// # Complexity
///
/// O(|clique|²) time, O(V) space.
///
/// [`ModularProductResult::vertex_pairs()`]: crate::traits::algorithms::modular_product::ModularProductResult::vertex_pairs
/// [`LineGraphResult::edge_map()`]: crate::traits::algorithms::line_graph::LineGraphResult::edge_map
///
/// # Examples
///
/// ```
/// use geometric_traits::prelude::infer_vertex_matches;
///
/// // Path graph 0-1-2 matched to path graph 10-11-12.
/// // Line graph vertices: 0=(0,1), 1=(1,2) and 0=(10,11), 1=(11,12)
/// // Clique matches both edges: product vertices [(0,0), (1,1)]
/// let clique = [0, 1];
/// let vertex_pairs = [(0_usize, 0_usize), (1, 1)];
/// let edge_map1 = [(0_u32, 1_u32), (1, 2)];
/// let edge_map2 = [(10_u32, 11_u32), (11, 12)];
///
/// let matches =
///     infer_vertex_matches(&clique, &vertex_pairs, &edge_map1, &edge_map2, |_, _, _, _| true);
/// // Shared endpoint: 1↔11 (Phase 1)
/// // Propagation: 0↔10 and 2↔12 (Phase 2)
/// assert_eq!(matches, vec![(0, 10), (1, 11), (2, 12)]);
/// ```
#[must_use]
pub fn infer_vertex_matches<N1, N2, F>(
    clique: &[usize],
    vertex_pairs: &[(usize, usize)],
    edge_map_first: &[(N1, N1)],
    edge_map_second: &[(N2, N2)],
    mut disambiguate: F,
) -> Vec<(N1, N2)>
where
    N1: Eq + Copy + Ord + core::fmt::Debug,
    N2: Eq + Copy + Ord + core::fmt::Debug,
    F: FnMut(N1, N1, N2, N2) -> bool,
{
    if clique.is_empty() {
        return Vec::new();
    }

    // Decode: map clique indices to original edge pairs.
    let matched_edges: Vec<((N1, N1), (N2, N2))> = clique
        .iter()
        .map(|&k| {
            let (lg1, lg2) = vertex_pairs[k];
            (edge_map_first[lg1], edge_map_second[lg2])
        })
        .collect();

    let mut vertex_map: BTreeMap<N1, N2> = BTreeMap::new();

    // Phase 1: shared endpoints between adjacent matched edges.
    for (i, &(e1_i, e2_i)) in matched_edges.iter().enumerate() {
        for &(e1_j, e2_j) in &matched_edges[i + 1..] {
            if let (Some(v1), Some(v2)) = (shared_endpoint(e1_i, e1_j), shared_endpoint(e2_i, e2_j))
            {
                try_insert(&mut vertex_map, v1, v2);
            }
        }
    }

    // Phase 2: propagation — resolve leaf endpoints.
    for &((a, b), (c, d)) in &matched_edges {
        let a_match = vertex_map.get(&a).copied();
        let b_match = vertex_map.get(&b).copied();
        match (a_match, b_match) {
            (Some(_), Some(_)) | (None, None) => {}
            (Some(target), None) => {
                let b_target = if target == c { d } else { c };
                try_insert(&mut vertex_map, b, b_target);
            }
            (None, Some(target)) => {
                let a_target = if target == d { c } else { d };
                try_insert(&mut vertex_map, a, a_target);
            }
        }
    }

    // Phase 3: disambiguation for isolated edges.
    for &((a, b), (c, d)) in &matched_edges {
        if !vertex_map.contains_key(&a) && !vertex_map.contains_key(&b) {
            if disambiguate(a, b, c, d) {
                try_insert(&mut vertex_map, a, c);
                try_insert(&mut vertex_map, b, d);
            } else {
                try_insert(&mut vertex_map, a, d);
                try_insert(&mut vertex_map, b, c);
            }
        }
    }

    vertex_map.into_iter().collect()
}
