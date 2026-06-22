//! Submodule providing the `Diameter` trait and its error type.
//!
//! The implementation uses exact iterative Fringe Upper Bounds (iFUB) for
//! connected undirected unweighted graphs. It combines a deterministic
//! degree-start heuristic with a deterministic 4-sweep fallback before the
//! bottom-up fringe scan from the iFUB paper. In the worst case it may still
//! need a BFS from many nodes, but in practice it usually converges after far
//! fewer traversals than the naive all-sources approach.

use alloc::{collections::VecDeque, vec::Vec};

use num_traits::AsPrimitive;

use crate::traits::{MonopartiteGraph, UndirectedMonopartiteMonoplexGraph};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
/// Error type for exact diameter computation.
pub enum DiameterError {
    /// The graph is disconnected, so the diameter is infinite.
    #[error("Cannot compute the diameter of a disconnected graph.")]
    DisconnectedGraph,
}

impl From<DiameterError>
    for crate::errors::monopartite_graph_error::algorithms::MonopartiteAlgorithmError
{
    #[inline]
    fn from(error: DiameterError) -> Self {
        Self::DiameterError(error)
    }
}

impl<G: MonopartiteGraph> From<DiameterError> for crate::errors::MonopartiteError<G> {
    #[inline]
    fn from(error: DiameterError) -> Self {
        Self::AlgorithmError(error.into())
    }
}

fn bfs_profile<G: UndirectedMonopartiteMonoplexGraph>(
    graph: &G,
    source: G::NodeId,
    distances: &mut [usize],
    parents: &mut [Option<G::NodeId>],
    queue: &mut VecDeque<G::NodeId>,
    order: usize,
) -> Result<(usize, G::NodeId), DiameterError>
where
    G::NodeId: AsPrimitive<usize>,
{
    distances.fill(usize::MAX);
    parents.fill(None);
    queue.clear();

    distances[source.as_()] = 0;
    queue.push_back(source);

    let mut visited = 1;
    let mut eccentricity = 0;
    let mut farthest = source;

    while let Some(node) = queue.pop_front() {
        let node_distance = distances[node.as_()];

        if node_distance > eccentricity
            || (node_distance == eccentricity && node.as_() > farthest.as_())
        {
            eccentricity = node_distance;
            farthest = node;
        }

        for neighbor in graph.neighbors(node) {
            let neighbor_index = neighbor.as_();
            if distances[neighbor_index] != usize::MAX {
                continue;
            }

            distances[neighbor_index] = node_distance + 1;
            parents[neighbor_index] = Some(node);
            visited += 1;
            queue.push_back(neighbor);
        }
    }

    if visited != order {
        return Err(DiameterError::DisconnectedGraph);
    }

    Ok((eccentricity, farthest))
}

#[inline]
fn bfs_eccentricity<G: UndirectedMonopartiteMonoplexGraph>(
    graph: &G,
    source: G::NodeId,
    distances: &mut [usize],
    parents: &mut [Option<G::NodeId>],
    queue: &mut VecDeque<G::NodeId>,
    order: usize,
) -> Result<usize, DiameterError>
where
    G::NodeId: AsPrimitive<usize>,
{
    Ok(bfs_profile(graph, source, distances, parents, queue, order)?.0)
}

fn middle_of_bfs_path<NodeId>(source: NodeId, target: NodeId, parents: &[Option<NodeId>]) -> NodeId
where
    NodeId: Copy + AsPrimitive<usize>,
{
    let mut path = Vec::new();
    let mut current = Some(target);

    while let Some(node) = current {
        path.push(node);
        if node.as_() == source.as_() {
            break;
        }
        current = parents[node.as_()];
    }

    debug_assert_eq!(path.last().map(|node| node.as_()), Some(source.as_()));
    path.reverse();
    path[path.len() / 2]
}

fn select_ifub_start<G: UndirectedMonopartiteMonoplexGraph>(
    graph: &G,
    node_ids: &[G::NodeId],
    degrees: &[usize],
    distances: &mut [usize],
    parents: &mut [Option<G::NodeId>],
    queue: &mut VecDeque<G::NodeId>,
    order: usize,
) -> Result<(G::NodeId, usize), DiameterError>
where
    G::NodeId: AsPrimitive<usize>,
{
    let r1 = node_ids
        .iter()
        .copied()
        .zip(degrees.iter().copied())
        .max_by_key(|&(node, degree)| (degree, node.as_()))
        .map(|(node, _)| node)
        .expect("non-empty graph must contain at least one node");

    let (_, a1) = bfs_profile(graph, r1, distances, parents, queue, order)?;
    let (ecc_a1, b1) = bfs_profile(graph, a1, distances, parents, queue, order)?;
    let r2 = middle_of_bfs_path(a1, b1, parents);

    let (_, a2) = bfs_profile(graph, r2, distances, parents, queue, order)?;
    let (ecc_a2, b2) = bfs_profile(graph, a2, distances, parents, queue, order)?;
    let start = middle_of_bfs_path(a2, b2, parents);

    Ok((start, ecc_a1.max(ecc_a2)))
}

fn ifub_from_profile<G: UndirectedMonopartiteMonoplexGraph>(
    graph: &G,
    node_ids: &[G::NodeId],
    start_eccentricity: usize,
    initial_lower_bound: usize,
    distances: &mut [usize],
    parents: &mut [Option<G::NodeId>],
    queue: &mut VecDeque<G::NodeId>,
) -> Result<usize, DiameterError>
where
    G::NodeId: AsPrimitive<usize>,
{
    let order = distances.len();
    let mut fringe_levels = vec![Vec::new(); start_eccentricity + 1];
    for &node in node_ids {
        fringe_levels[distances[node.as_()]].push(node);
    }

    let mut lower_bound = initial_lower_bound.max(start_eccentricity);
    let mut upper_bound = start_eccentricity.saturating_mul(2);
    let mut level = start_eccentricity;

    while upper_bound > lower_bound && level > 0 {
        let mut fringe_eccentricity = 0;

        for &node in &fringe_levels[level] {
            fringe_eccentricity = fringe_eccentricity
                .max(bfs_eccentricity(graph, node, distances, parents, queue, order)?);
        }

        lower_bound = lower_bound.max(fringe_eccentricity);

        let next_upper_bound = 2 * (level - 1);
        if lower_bound > next_upper_bound {
            return Ok(lower_bound);
        }

        upper_bound = next_upper_bound;
        level -= 1;
    }

    Ok(lower_bound)
}

/// Trait providing exact diameter computation for connected undirected
/// unweighted graphs.
///
/// The diameter is the maximum eccentricity, that is, the maximum shortest-path
/// distance over all pairs of nodes.
///
/// This implementation uses exact iterative Fringe Upper Bounds (iFUB) with an
/// adaptive deterministic start policy: it first tries a highest-degree node
/// and falls back to deterministic 4-sweep when that start appears too
/// peripheral.
pub trait Diameter: UndirectedMonopartiteMonoplexGraph + Sized
where
    Self::NodeId: AsPrimitive<usize>,
{
    /// Returns the exact diameter of the graph.
    ///
    /// Empty and singleton graphs return `0`.
    ///
    /// # Errors
    ///
    /// Returns [`DiameterError::DisconnectedGraph`] when the graph has more
    /// than one connected component.
    ///
    /// # Complexity
    ///
    /// Worst-case O(V * (V + E)) time and O(V) additional space.
    ///
    /// # References
    ///
    /// - P. Crescenzi, R. Grossi, M. Habib, L. Lanzi, A. Marino, *On computing
    ///   the diameter of real-world undirected graphs*, Theoretical Computer
    ///   Science 514 (2013), 84-95.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::{prelude::*, traits::algorithms::randomized_graphs::path_graph};
    ///
    /// let graph = {
    ///     let edges = path_graph(5);
    ///     let nodes = GenericVocabularyBuilder::default()
    ///         .expected_number_of_symbols(edges.order())
    ///         .symbols((0..edges.order()).enumerate())
    ///         .build()
    ///         .unwrap();
    ///     UndiGraph::from((nodes, edges))
    /// };
    ///
    /// assert_eq!(graph.diameter().unwrap(), 4);
    /// ```
    #[inline]
    fn diameter(&self) -> Result<usize, crate::errors::MonopartiteError<Self>> {
        let order = self.number_of_nodes().as_();
        if order <= 1 {
            return Ok(0);
        }

        let node_ids: Vec<Self::NodeId> = self.node_ids().collect();
        let degrees: Vec<usize> = node_ids.iter().map(|&node| self.degree(node).as_()).collect();

        let mut distances = vec![usize::MAX; order];
        let mut parents = vec![None; order];
        let mut queue = VecDeque::with_capacity(order);

        let degree_start = node_ids
            .iter()
            .copied()
            .zip(degrees.iter().copied())
            .max_by_key(|&(node, degree)| (degree, node.as_()))
            .map(|(node, _)| node)
            .expect("non-empty graph must contain at least one node");

        let degree_start_eccentricity =
            bfs_eccentricity(self, degree_start, &mut distances, &mut parents, &mut queue, order)?;

        if degree_start_eccentricity.saturating_mul(2) <= order {
            return Ok(ifub_from_profile(
                self,
                &node_ids,
                degree_start_eccentricity,
                degree_start_eccentricity,
                &mut distances,
                &mut parents,
                &mut queue,
            )?);
        }

        let (start, lower_bound) = select_ifub_start(
            self,
            &node_ids,
            &degrees,
            &mut distances,
            &mut parents,
            &mut queue,
            order,
        )?;

        let start_eccentricity =
            bfs_eccentricity(self, start, &mut distances, &mut parents, &mut queue, order)?;

        Ok(ifub_from_profile(
            self,
            &node_ids,
            start_eccentricity,
            lower_bound.max(degree_start_eccentricity),
            &mut distances,
            &mut parents,
            &mut queue,
        )?)
    }
}

impl<G> Diameter for G
where
    G: UndirectedMonopartiteMonoplexGraph + Sized,
    G::NodeId: AsPrimitive<usize>,
{
}
