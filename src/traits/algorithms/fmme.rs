//! Fast-multipole multilevel embedder (FMME), the layout used by TMAP.
//!
//! Ports the OGDF stack TMAP drives: a `FastMultipoleEmbedder` single-level
//! force layout, wrapped per level by a mean-edge-length scaling pass and
//! driven over a coarsening hierarchy. Disconnected graphs are laid out per
//! component and tiled into rows. 2D only, matching OGDF.
//!
//! # Example
//!
//! ```
//! use geometric_traits::traits::algorithms::fmme::{MixerConfig, layout_graph};
//!
//! // A path of five nodes.
//! let edges = [(0, 1), (1, 2), (2, 3), (3, 4)];
//! let config = MixerConfig::<f64>::default();
//! let coords = layout_graph::<f64>(5, &edges, &config);
//! assert_eq!(coords.len(), 5 * 2);
//! assert!(coords.iter().all(|value| value.is_finite()));
//! ```

// Per-axis loops over `0..2` index flat buffers in lockstep.
#![allow(clippy::needless_range_loop)]

mod fme;
mod mixer;
mod multilevel;

use alloc::{collections::BTreeSet, vec, vec::Vec};

pub use mixer::MixerConfig;
use mixer::multilevel_layout;
use num_traits::{AsPrimitive, Float};

use crate::traits::SparseValuedMatrix2D;

/// Lays out a graph on `n` nodes with undirected `edges` (endpoint indices in
/// `0..n`), returning a flat `n * 2` coordinate buffer normalized so every
/// coordinate lies in `[-0.5, 0.5]`.
///
/// Disconnected graphs are split into connected components, each laid out
/// independently, then the component bounding boxes are tiled into rows so they
/// do not overlap (OGDF `ComponentSplitterLayout` with a `TileToRowsCCPacker`).
///
/// The layout targets forests (TMAP lays out a spanning tree) but accepts any
/// graph. Tiny connected graphs (three or fewer nodes) collapse to a point.
///
/// # Panics
///
/// If any edge endpoint is at least `n`.
#[must_use]
pub fn layout_graph<F>(n: usize, edges: &[(usize, usize)], config: &MixerConfig<F>) -> Vec<F>
where
    F: Float + Send + Sync + core::ops::AddAssign + core::ops::SubAssign,
{
    if n == 0 {
        return Vec::new();
    }

    let components = connected_components(n, edges);
    let mut coordinates = if components.len() == 1 {
        multilevel_layout::<F>(n, edges, config)
    } else {
        layout_components::<F>(n, edges, &components, config)
    };

    normalize_unit_box::<F>(&mut coordinates, n);
    coordinates
}

/// Lays out each connected component and tiles their bounding boxes into rows.
fn layout_components<F>(
    n: usize,
    edges: &[(usize, usize)],
    components: &[Vec<usize>],
    config: &MixerConfig<F>,
) -> Vec<F>
where
    F: Float + Send + Sync + core::ops::AddAssign + core::ops::SubAssign,
{
    struct Placed<F> {
        nodes: Vec<usize>,
        coords: Vec<F>,
        width: F,
        height: F,
    }

    let mut placed: Vec<Placed<F>> = Vec::with_capacity(components.len());
    for nodes in components {
        let local_n = nodes.len();
        let mut index_of = alloc::collections::BTreeMap::new();
        for (local, &global) in nodes.iter().enumerate() {
            index_of.insert(global, local);
        }
        let local_edges: Vec<(usize, usize)> = edges
            .iter()
            .filter(|(a, b)| index_of.contains_key(a) && index_of.contains_key(b))
            .map(|(a, b)| (index_of[a], index_of[b]))
            .collect();
        let mut coords = multilevel_layout::<F>(local_n, &local_edges, config);
        let (mut min0, mut min1, mut max0, mut max1) =
            (F::infinity(), F::infinity(), F::neg_infinity(), F::neg_infinity());
        for i in 0..local_n {
            let x = coords[i * 2];
            let y = coords[i * 2 + 1];
            min0 = min0.min(x);
            max0 = max0.max(x);
            min1 = min1.min(y);
            max1 = max1.max(y);
        }
        for i in 0..local_n {
            coords[i * 2] -= min0;
            coords[i * 2 + 1] -= min1;
        }
        placed.push(Placed {
            nodes: nodes.clone(),
            coords,
            width: (max0 - min0).max(F::zero()),
            height: (max1 - min1).max(F::zero()),
        });
    }

    let total_area = placed.iter().fold(F::zero(), |acc, p| acc + p.width * p.height);
    let target_width = total_area.sqrt().max(F::from(f64::MIN_POSITIVE).unwrap());
    let border = placed.iter().fold(F::zero(), |acc, p| acc + p.width + p.height)
        / F::from(2 * placed.len().max(1)).unwrap()
        * F::from(0.1).unwrap();

    let mut out = vec![F::zero(); n * 2];
    let (mut cursor_x, mut cursor_y, mut row_height) = (F::zero(), F::zero(), F::zero());
    for p in &placed {
        if cursor_x > F::zero() && cursor_x + p.width > target_width {
            cursor_y += row_height + border;
            cursor_x = F::zero();
            row_height = F::zero();
        }
        for (local, &global) in p.nodes.iter().enumerate() {
            for d in 0..2 {
                out[global * 2 + d] = p.coords[local * 2 + d];
            }
            out[global * 2] += cursor_x;
            out[global * 2 + 1] += cursor_y;
        }
        cursor_x += p.width + border;
        row_height = row_height.max(p.height);
    }
    out
}

/// Connected components as lists of node indices, via iterative breadth-first
/// search over the undirected adjacency.
fn connected_components(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(a, b) in edges {
        if a != b {
            adjacency[a].push(b);
            adjacency[b].push(a);
        }
    }
    let mut component_of = vec![usize::MAX; n];
    let mut components: Vec<Vec<usize>> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..n {
        if component_of[start] != usize::MAX {
            continue;
        }
        let id = components.len();
        let mut members = Vec::new();
        component_of[start] = id;
        stack.push(start);
        while let Some(node) = stack.pop() {
            members.push(node);
            for &neighbor in &adjacency[node] {
                if component_of[neighbor] == usize::MAX {
                    component_of[neighbor] = id;
                    stack.push(neighbor);
                }
            }
        }
        components.push(members);
    }
    components
}

/// Recenters and scales `positions` so every coordinate lies in `[-0.5, 0.5]`.
fn normalize_unit_box<F: Float>(positions: &mut [F], n: usize) {
    if n == 0 {
        return;
    }
    for d in 0..2 {
        let mut min = F::infinity();
        let mut max = F::neg_infinity();
        for i in 0..n {
            let value = positions[i * 2 + d];
            min = min.min(value);
            max = max.max(value);
        }
        let range = max - min;
        let inv = if range > F::zero() { F::one() / range } else { F::one() };
        let half = F::from(0.5).unwrap();
        for i in 0..n {
            positions[i * 2 + d] = (positions[i * 2 + d] - min) * inv - half;
        }
    }
}

/// The multilevel layout result: coordinates and the graph edges.
#[derive(Debug, Clone)]
pub struct FmmeResult<F> {
    coordinates: Vec<F>,
    edges: Vec<(usize, usize)>,
    n: usize,
}

impl<F: Float> FmmeResult<F> {
    /// The flat `n * 2` coordinate buffer, normalized to `[-0.5, 0.5]`.
    #[must_use]
    pub fn coordinates_flat(&self) -> &[F] {
        &self.coordinates
    }

    /// The graph edges the layout was computed over.
    #[must_use]
    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }

    /// The node count.
    #[must_use]
    pub fn num_points(&self) -> usize {
        self.n
    }
}

/// Lays out a graph with the FMME, the TMAP layout. Edge weights are ignored
/// (TMAP lays out its unweighted spanning tree). Each stored off-diagonal entry
/// is one undirected edge, deduplicated. Self-loops are ignored.
///
/// # Examples
///
/// ```
/// use geometric_traits::{
///     impls::ValuedCSR2D,
///     prelude::*,
///     traits::algorithms::fmme::{Fmme, MixerConfig},
/// };
///
/// // A symmetric path of four nodes.
/// let edges = vec![(0, 1, 1.0), (1, 0, 1.0), (1, 2, 1.0), (2, 1, 1.0), (2, 3, 1.0), (3, 2, 1.0)];
/// let csr: ValuedCSR2D<usize, usize, usize, f64> =
///     GenericEdgesBuilder::<_, ValuedCSR2D<usize, usize, usize, f64>>::default()
///         .expected_number_of_edges(6)
///         .expected_shape((4, 4))
///         .edges(edges.into_iter())
///         .build()
///         .unwrap();
///
/// let result = csr.fmme_layout::<f64>(&MixerConfig::default());
/// assert_eq!(result.num_points(), 4);
/// assert!(result.coordinates_flat().iter().all(|value| value.is_finite()));
/// ```
pub trait Fmme: SparseValuedMatrix2D + Sized
where
    Self::RowIndex: AsPrimitive<usize>,
    Self::ColumnIndex: AsPrimitive<usize>,
{
    /// Computes a `D`-dimensional FMME layout of the graph. `D` is 2 for the
    /// TMAP tree map. Returns the coordinates and the deduplicated undirected
    /// edge list.
    fn fmme_layout<F>(&self, config: &MixerConfig<F>) -> FmmeResult<F>
    where
        F: Float + Send + Sync + core::ops::AddAssign + core::ops::SubAssign,
    {
        let mut n = self.number_of_rows().as_().max(self.number_of_columns().as_());
        let mut undirected: BTreeSet<(usize, usize)> = BTreeSet::new();
        for row_id in self.row_indices() {
            let source = row_id.as_();
            for column_id in self.sparse_row(row_id) {
                let target = column_id.as_();
                if source != target {
                    let (a, b) = if source < target { (source, target) } else { (target, source) };
                    n = n.max(b + 1);
                    undirected.insert((a, b));
                }
            }
        }
        let edges: Vec<(usize, usize)> = undirected.into_iter().collect();
        let coordinates = layout_graph::<F>(n, &edges, config);
        FmmeResult { coordinates, edges, n }
    }
}

impl<M> Fmme for M
where
    M: SparseValuedMatrix2D + Sized,
    M::RowIndex: AsPrimitive<usize>,
    M::ColumnIndex: AsPrimitive<usize>,
{
}
