//! Coarsening hierarchy and barycenter prolongation for the multilevel mixer.
//!
//! Ports OGDF `LocalBiconnectedMerger` (coarsening) and `BarycenterPlacer`
//! (prolongation) for TMAP's inputs, which are always spanning forests. Two
//! specializations follow: the bundled OGDF compiles out cut detection, so
//! `canMerge` reduces to `min(degree(parent), degree(child)) <= 2`, and all
//! edge weights are 1.0, so the placer is a plain barycenter of the placed
//! neighbors plus a uniform jitter in `[-1, 1]` (the layout's entropy source,
//! since TMAP disables the FME randomize step).

use alloc::{collections::BTreeSet, vec, vec::Vec};

use num_traits::Float;

use super::super::randomized_graphs::XorShift64;

/// The number of levels below which OGDF `buildOneLevel` stops coarsening.
const MIN_LEVEL_NODES: usize = 3;

/// A graph at one coarsening level: node count and undirected edges over
/// relabeled indices `0..n`.
#[derive(Debug, Clone)]
pub struct LevelGraph {
    /// Node count at this level.
    pub n: usize,
    /// Undirected edges as `(a, b)` with `a < b`, relabeled to `0..n`.
    pub edges: Vec<(usize, usize)>,
}

/// The prolongation map from a coarse level to the next finer level, as
/// produced by coarsening that finer level. Applying it turns a coarse layout
/// into a finer one: each surviving coarse node keeps its position, and each
/// merged child is placed at the barycenter of its already-placed neighbors
/// plus a jitter.
#[derive(Debug, Clone)]
pub struct Prolong {
    /// Node count of the finer level this map produces.
    pub n_fine: usize,
    /// Coarse node `c` maps to fine index `survivor[c]` (its representative).
    survivor: Vec<usize>,
    /// Merged children in prolongation (reverse merge) order, each with the
    /// fine indices of its neighbors at merge time (all already placed when
    /// reached).
    reinsert: Vec<(usize, Vec<usize>)>,
}

impl Prolong {
    /// Turns a coarse layout (`coarse` is a flat `survivor.len() * 2` buffer)
    /// into a finer one (a flat `n_fine * 2` buffer). Survivors inherit their
    /// coarse position, then each child is placed at the barycenter of its
    /// neighbors plus an independent uniform jitter in `[-1, 1]` per axis.
    pub fn apply<F: Float>(&self, coarse: &[F], rng: &mut XorShift64, jitter_scale: F) -> Vec<F> {
        debug_assert_eq!(coarse.len(), self.survivor.len() * 2);
        let mut fine = vec![F::zero(); self.n_fine * 2];
        for (c, &fine_idx) in self.survivor.iter().enumerate() {
            for d in 0..2 {
                fine[fine_idx * 2 + d] = coarse[c * 2 + d];
            }
        }
        for (child, neighbors) in &self.reinsert {
            let mut sum = [F::zero(); 2];
            for &nb in neighbors {
                for d in 0..2 {
                    sum[d] = sum[d] + fine[nb * 2 + d];
                }
            }
            let inv = F::one() / F::from(neighbors.len().max(1)).unwrap();
            for d in 0..2 {
                fine[child * 2 + d] = sum[d] * inv + jitter::<F>(rng) * jitter_scale;
            }
        }
        fine
    }
}

/// A uniform jitter in `[-1, 1]` (OGDF `randomDouble(-1, 1)` in the placer).
fn jitter<F: Float>(rng: &mut XorShift64) -> F {
    // Top 53 bits give a uniform double in [0, 1); map to [-1, 1).
    let bits = rng.next().unwrap() >> 11;
    let unit = F::from(bits).unwrap() / F::from(1u64 << 53).unwrap();
    unit * F::from(2.0).unwrap() - F::one()
}

/// Builds the coarsening hierarchy of a graph on `n` nodes with `edges`.
/// Returns the per-level graphs (index 0 is the input, the last the coarsest)
/// and the prolongation maps (`prolongs[l]` turns a layout of `levels[l + 1]`
/// into one of `levels[l]`). Each level roughly halves the node count via a
/// guarded random matching, until three or fewer nodes remain.
pub fn build_hierarchy(
    n: usize,
    edges: &[(usize, usize)],
    rng: &mut XorShift64,
) -> (Vec<LevelGraph>, Vec<Prolong>) {
    let mut levels = vec![LevelGraph { n, edges: edges.to_vec() }];
    let mut prolongs: Vec<Prolong> = Vec::new();

    loop {
        let current = levels.last().unwrap();
        if current.n <= MIN_LEVEL_NODES {
            break;
        }
        let Some((coarse, prolong)) = coarsen_one_level(current, rng) else {
            break;
        };
        if coarse.n == current.n {
            break;
        }
        levels.push(coarse);
        prolongs.push(prolong);
    }

    (levels, prolongs)
}

/// Coarsens one level. Returns the coarser graph and the map to prolong a
/// coarse layout back to `fine`, or `None` if no contraction was possible.
fn coarsen_one_level(fine: &LevelGraph, rng: &mut XorShift64) -> Option<(LevelGraph, Prolong)> {
    let nf = fine.n;
    let mut adj: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); nf];
    for &(u, v) in &fine.edges {
        if u != v {
            adj[u].insert(v);
            adj[v].insert(u);
        }
    }

    // 1. Random maximal matching over a random edge order, leftovers to `rest`.
    let mut order: Vec<usize> = (0..fine.edges.len()).collect();
    shuffle(&mut order, rng);
    let mut marked = vec![false; nf];
    let mut matching: Vec<(usize, usize)> = Vec::new();
    let mut rest: Vec<(usize, usize)> = Vec::new();
    for &ei in &order {
        let (u, v) = fine.edges[ei];
        if u == v {
            continue;
        }
        if !marked[u] && !marked[v] {
            matching.push((u, v));
            marked[u] = true;
            marked[v] = true;
        } else {
            rest.push((u, v));
        }
    }

    // 2. Edge cover over the leftovers: keep an edge if it still touches an
    //    unmarked node, then mark both endpoints.
    let mut cover_order: Vec<usize> = (0..rest.len()).collect();
    shuffle(&mut cover_order, rng);
    let mut edge_cover: Vec<(usize, usize)> = Vec::new();
    for &ri in &cover_order {
        let (u, v) = rest[ri];
        if !marked[u] || !marked[v] {
            edge_cover.push((u, v));
            marked[u] = true;
            marked[v] = true;
        }
    }

    // 3. Contract, matching first then cover, until the node count is halved.
    let target = nf / 2;
    let mut substitute: Vec<usize> = (0..nf).collect();
    let mut active = vec![true; nf];
    let mut active_count = nf;
    let mut merges: Vec<(usize, Vec<usize>)> = Vec::new();

    for &(u0, v0) in matching.iter().chain(edge_cover.iter()) {
        if active_count <= target {
            break;
        }
        let u = find(&mut substitute, u0);
        let v = find(&mut substitute, v0);
        if u == v || !active[u] || !active[v] {
            continue;
        }
        // Parent is the higher-degree endpoint; the lower-degree one folds in.
        let (parent, child) = if adj[u].len() >= adj[v].len() { (u, v) } else { (v, u) };
        // Forest biconnectivity guard: allow only if one endpoint has degree at
        // most 2 (see the module comment).
        if adj[parent].len() > 2 && adj[child].len() > 2 {
            continue;
        }

        // Record the child's current neighbors (all active) for prolongation.
        let neighbors: Vec<usize> = adj[child].iter().copied().collect();
        merges.push((child, neighbors));

        // Contract: move the child's edges onto the parent, collapsing
        // parallels.
        let child_adj: Vec<usize> = adj[child].iter().copied().collect();
        for w in child_adj {
            adj[w].remove(&child);
            if w != parent {
                adj[parent].insert(w);
                adj[w].insert(parent);
            }
        }
        adj[parent].remove(&child);
        adj[child].clear();
        active[child] = false;
        substitute[child] = parent;
        active_count -= 1;
    }

    if active_count == nf {
        return None;
    }

    // Relabel the survivors to `0..active_count` and build the coarse edge
    // list.
    let mut fine_to_coarse = vec![usize::MAX; nf];
    let mut survivor: Vec<usize> = Vec::with_capacity(active_count);
    for (idx, is_active) in active.iter().enumerate() {
        if *is_active {
            fine_to_coarse[idx] = survivor.len();
            survivor.push(idx);
        }
    }
    let mut coarse_edges: BTreeSet<(usize, usize)> = BTreeSet::new();
    for &fine_idx in &survivor {
        let c = fine_to_coarse[fine_idx];
        for &w in &adj[fine_idx] {
            let cw = fine_to_coarse[w];
            let (a, b) = if c < cw { (c, cw) } else { (cw, c) };
            if a != b {
                coarse_edges.insert((a, b));
            }
        }
    }

    // Prolongation reinserts children in reverse merge order.
    merges.reverse();
    let prolong = Prolong { n_fine: nf, survivor, reinsert: merges };
    let coarse = LevelGraph { n: active_count, edges: coarse_edges.into_iter().collect() };
    Some((coarse, prolong))
}

/// Union-find find with path halving over the substitute array.
fn find(substitute: &mut [usize], mut x: usize) -> usize {
    while substitute[x] != x {
        substitute[x] = substitute[substitute[x]];
        x = substitute[x];
    }
    x
}

/// In-place Fisher-Yates shuffle driven by the generator.
fn shuffle(items: &mut [usize], rng: &mut XorShift64) {
    let n = items.len();
    for i in (1..n).rev() {
        #[allow(clippy::cast_possible_truncation)]
        let j = (rng.next().unwrap() % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}
