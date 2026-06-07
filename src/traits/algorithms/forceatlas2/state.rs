//! Validated internal graph state for the ForceAtlas2 layout.

use alloc::{vec, vec::Vec};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;
use num_traits::{AsPrimitive, ToPrimitive};
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use super::{
    ForceAtlas2Config, ForceAtlas2Error,
    forces::{
        apply_attraction, apply_attraction_anti_collision, apply_gravity, apply_lin_repulsion,
        apply_lin_repulsion_anti_collision,
    },
    quadtree::QuadTree,
    speed::{SpeedState, accumulate_totals, per_node_factor, update_global_speed},
};
use crate::traits::{Finite, SparseValuedMatrix2D, algorithms::modularity::approx_eq};

/// Validated graph and mutable layout state.
///
/// Built once per layout run from the adjacency matrix. The matrix must be
/// square and symmetric with finite, non-negative weights. Self-loops are
/// skipped entirely: they contribute neither mass nor attraction.
#[derive(Debug, Clone)]
pub(super) struct LayoutState {
    /// Undirected edges as `(source, target, weight)` with `source <
    /// target`, sorted lexicographically.
    pub(super) edges: Vec<(usize, usize, f64)>,
    /// Per-node mass, equal to one plus the structural degree (the number
    /// of distinct neighbors, self-loops excluded).
    pub(super) masses: Vec<f64>,
    /// Current node positions.
    pub(super) positions: Vec<[f64; 2]>,
    /// Node sizes (radii) when the anti-collision mode is active.
    pub(super) sizes: Option<Vec<f64>>,
}

impl LayoutState {
    /// Builds the layout state from a symmetric adjacency matrix and a
    /// validated configuration.
    pub(super) fn from_matrix<M>(
        matrix: &M,
        config: &ForceAtlas2Config,
    ) -> Result<Self, ForceAtlas2Error>
    where
        M: SparseValuedMatrix2D,
        M::RowIndex: AsPrimitive<usize>,
        M::ColumnIndex: AsPrimitive<usize>,
        M::Value: ToPrimitive + Finite,
    {
        let rows = matrix.number_of_rows().as_();
        let columns = matrix.number_of_columns().as_();
        if rows != columns {
            return Err(ForceAtlas2Error::NonSquareMatrix { rows, columns });
        }
        if rows == 0 {
            return Err(ForceAtlas2Error::EmptyGraph);
        }

        // Collect the (validated) adjacency, skipping self-loops.
        let mut adjacency: Vec<Vec<(usize, f64)>> = vec![Vec::new(); rows];
        for row_id in matrix.row_indices() {
            let source = row_id.as_();
            for (column_id, weight) in
                matrix.sparse_row(row_id).zip(matrix.sparse_row_values(row_id))
            {
                let destination = column_id.as_();
                if !weight.is_finite() {
                    return Err(ForceAtlas2Error::NonFiniteWeight {
                        source_id: source,
                        destination_id: destination,
                    });
                }
                let weight = weight.to_f64().ok_or(ForceAtlas2Error::UnrepresentableWeight {
                    source_id: source,
                    destination_id: destination,
                })?;
                // Defensive double-check after the conversion. Unreachable
                // for f64 values (the conversion is the identity and the
                // pre-check already ran), kept for exotic value types and
                // parity with the modularity state builder.
                if !weight.is_finite() {
                    return Err(ForceAtlas2Error::NonFiniteWeight {
                        source_id: source,
                        destination_id: destination,
                    });
                }
                if weight < 0.0 {
                    return Err(ForceAtlas2Error::NegativeWeight {
                        source_id: source,
                        destination_id: destination,
                    });
                }
                if source != destination {
                    adjacency[source].push((destination, weight));
                }
            }
        }

        for row in &mut adjacency {
            row.sort_unstable_by_key(|(destination, _)| *destination);
        }

        // Symmetry check: every edge needs a reverse edge of equal weight.
        for (source, neighbors) in adjacency.iter().enumerate() {
            for (destination, weight) in neighbors {
                if !has_matching_edge(&adjacency[*destination], source, *weight) {
                    return Err(ForceAtlas2Error::NonSymmetricEdge {
                        source_id: source,
                        destination_id: *destination,
                    });
                }
            }
        }

        // Mass is one plus the structural degree. The cast is exact for any
        // realistic degree (below 2^53).
        #[allow(clippy::cast_precision_loss)]
        let masses: Vec<f64> = adjacency.iter().map(|row| (row.len() + 1) as f64).collect();

        // Each undirected edge is kept once, with source < target.
        let mut edges = Vec::new();
        for (source, neighbors) in adjacency.iter().enumerate() {
            for (destination, weight) in neighbors {
                if source < *destination {
                    edges.push((source, *destination, *weight));
                }
            }
        }

        let positions = initial_positions(rows, config)?;
        let sizes = validated_sizes(rows, config)?;

        Ok(Self { edges, masses, positions, sizes })
    }

    /// Runs the configured number of layout iterations, mutating the node
    /// positions in place.
    ///
    /// Returns the mass-weighted global `(swinging, traction)` totals of
    /// the last iteration, or zeros when no iterations were run.
    ///
    /// The iteration order mirrors the Gephi reference implementation
    /// (`ForceAtlas2.goAlgo`): force buffer shift, pairwise repulsion,
    /// gravity, per-edge attraction, global totals, global speed update,
    /// position update. Node masses are
    /// degree-based and therefore constant, so the per-iteration mass
    /// recomputation of the Java source is hoisted out.
    pub(super) fn run(&mut self, config: &ForceAtlas2Config) -> (f64, f64) {
        let n = self.masses.len();
        let mut forces = vec![[0.0_f64; 2]; n];
        let mut old_forces = vec![[0.0_f64; 2]; n];
        let mut speed_state = SpeedState::default();
        let mut totals = (0.0, 0.0);

        let effective_weights = effective_weights(&self.edges, config.edge_weight_influence);

        // Mean-mass compensation for the dissuade hubs mode (the
        // `outboundAttCompensation` of the Java source). Masses are
        // constant, so this too is hoisted out of the iteration loop.
        #[allow(clippy::cast_precision_loss)]
        let attraction_coefficient =
            if config.dissuade_hubs { self.masses.iter().sum::<f64>() / n as f64 } else { 1.0 };

        for _ in 0..config.iterations {
            // The previous iteration's forces become the old forces and the
            // accumulators are cleared.
            core::mem::swap(&mut old_forces, &mut forces);
            forces.fill([0.0; 2]);

            self.accumulate_repulsion(config, &mut forces);

            // Gravity toward the origin.
            for ((pos, mass), force) in self.positions.iter().zip(&self.masses).zip(&mut forces) {
                apply_gravity(*pos, *mass, config.gravity, config.strong_gravity, force);
            }

            self.accumulate_attraction(
                config,
                &effective_weights,
                attraction_coefficient,
                &mut forces,
            );

            // Global swinging and traction, then the adaptive speed.
            totals = accumulate_totals(&self.masses, &old_forces, &forces);
            update_global_speed(&mut speed_state, totals.0, totals.1, n, config.jitter_tolerance);

            self.update_positions(speed_state.speed, &old_forces, &forces);
        }

        totals
    }

    /// Accumulates the repulsion forces: either the Barnes-Hut
    /// approximation (one-sided per node against the quadtree, rebuilt
    /// every iteration) or the exact pairwise pass over the strict lower
    /// triangle with both endpoints updated per pair. Node sizes switch
    /// both backends to the anti-collision kernels (in the Barnes-Hut case
    /// only at the exact leaf level, regions ignore sizes like the
    /// reference implementations).
    fn accumulate_repulsion(&self, config: &ForceAtlas2Config, forces: &mut [[f64; 2]]) {
        let n = self.masses.len();
        if config.barnes_hut {
            let tree = QuadTree::build(&self.positions, &self.masses);
            for ((node, pos), (mass, force)) in
                self.positions.iter().enumerate().zip(self.masses.iter().zip(forces))
            {
                tree.apply_repulsion(
                    node,
                    *pos,
                    *mass,
                    config.scaling_ratio,
                    config.barnes_hut_theta,
                    self.sizes.as_deref(),
                    force,
                );
            }
        } else if let Some(sizes) = &self.sizes {
            for i in 1..n {
                let (left, right) = forces.split_at_mut(i);
                let force_i = &mut right[0];
                for j in 0..i {
                    apply_lin_repulsion_anti_collision(
                        self.positions[i],
                        self.positions[j],
                        self.masses[i],
                        self.masses[j],
                        sizes[i],
                        sizes[j],
                        config.scaling_ratio,
                        force_i,
                        &mut left[j],
                    );
                }
            }
        } else {
            for i in 1..n {
                let (left, right) = forces.split_at_mut(i);
                let force_i = &mut right[0];
                for ((pos_j, mass_j), force_j) in
                    self.positions[..i].iter().zip(&self.masses[..i]).zip(left.iter_mut())
                {
                    apply_lin_repulsion(
                        self.positions[i],
                        *pos_j,
                        self.masses[i],
                        *mass_j,
                        config.scaling_ratio,
                        force_i,
                        force_j,
                    );
                }
            }
        }
    }

    /// Accumulates the attraction forces along every undirected edge,
    /// dispatched on the configured mode. The edge source (the smaller
    /// index) is the node whose mass dissuades hubs.
    fn accumulate_attraction(
        &self,
        config: &ForceAtlas2Config,
        effective_weights: &[f64],
        attraction_coefficient: f64,
        forces: &mut [[f64; 2]],
    ) {
        for (&(source, target, _), &e) in self.edges.iter().zip(effective_weights) {
            let (left, right) = forces.split_at_mut(target);
            if let Some(sizes) = &self.sizes {
                apply_attraction_anti_collision(
                    config.lin_log,
                    config.dissuade_hubs,
                    self.positions[source],
                    self.positions[target],
                    e,
                    attraction_coefficient,
                    self.masses[source],
                    sizes[source],
                    sizes[target],
                    &mut left[source],
                    &mut right[0],
                );
            } else {
                apply_attraction(
                    config.lin_log,
                    config.dissuade_hubs,
                    self.positions[source],
                    self.positions[target],
                    e,
                    attraction_coefficient,
                    self.masses[source],
                    &mut left[source],
                    &mut right[0],
                );
            }
        }
    }

    /// Applies the accumulated forces to the positions.
    ///
    /// In anti-collision mode the per-node speed factor carries the extra
    /// `0.1` multiplier and the displacement length is capped at 10 units
    /// (the Java adjustSizes branch). The `df == 0` division and any force
    /// overflow are guarded so coordinates stay finite (the Java source
    /// leaves both unguarded).
    fn update_positions(&mut self, speed: f64, old_forces: &[[f64; 2]], forces: &[[f64; 2]]) {
        let anti_collision = self.sizes.is_some();
        for ((position, mass), (old_force, force)) in
            self.positions.iter_mut().zip(&self.masses).zip(old_forces.iter().zip(forces))
        {
            let diff_x = old_force[0] - force[0];
            let diff_y = old_force[1] - force[1];
            let local_swinging = (diff_x * diff_x + diff_y * diff_y).sqrt();
            let mut factor = per_node_factor(speed, *mass, local_swinging);
            if anti_collision {
                factor *= 0.1;
                let df = (force[0] * force[0] + force[1] * force[1]).sqrt();
                // Added guard (not in the Java source, which divides 0/0
                // here): a zero net force moves nothing.
                if df <= 0.0 {
                    continue;
                }
                factor = (factor * df).min(10.0) / df;
            }
            let new_x = position[0] + force[0] * factor;
            let new_y = position[1] + force[1] * factor;
            // Added guard (not in the Java source): a force overflow must
            // not poison the coordinates with non-finite values, the node
            // does not move this iteration.
            if new_x.is_finite() && new_y.is_finite() {
                *position = [new_x, new_y];
            }
        }
    }
}

/// Computes the effective edge weights `w^delta`, with the exact fast
/// paths of the Java source for `delta == 0` (weights ignored) and
/// `delta == 1` (raw weights, no `pow`).
///
/// The strict float comparisons are intentional: they reproduce the exact
/// dispatch of the reference implementation.
#[allow(clippy::float_cmp)]
fn effective_weights(edges: &[(usize, usize, f64)], delta: f64) -> Vec<f64> {
    if delta == 0.0 {
        vec![1.0; edges.len()]
    } else if delta == 1.0 {
        edges.iter().map(|&(_, _, weight)| weight).collect()
    } else {
        edges.iter().map(|&(_, _, weight)| weight.powf(delta)).collect()
    }
}

/// Validates the optional node sizes: one finite, non-negative size per
/// node.
fn validated_sizes(
    n: usize,
    config: &ForceAtlas2Config,
) -> Result<Option<Vec<f64>>, ForceAtlas2Error> {
    let Some(provided) = &config.node_sizes else {
        return Ok(None);
    };
    if provided.len() != n {
        return Err(ForceAtlas2Error::NodeSizesLengthMismatch {
            expected: n,
            actual: provided.len(),
        });
    }
    for (index, size) in provided.iter().enumerate() {
        if !size.is_finite() || *size < 0.0 {
            return Err(ForceAtlas2Error::InvalidNodeSize { index });
        }
    }
    Ok(Some(provided.clone()))
}

/// Returns one finite position per node, either validated from the
/// configuration or drawn uniformly from `[-0.5, 0.5)` per axis with a
/// seeded RNG.
fn initial_positions(
    n: usize,
    config: &ForceAtlas2Config,
) -> Result<Vec<[f64; 2]>, ForceAtlas2Error> {
    if let Some(provided) = &config.initial_positions {
        if provided.len() != n {
            return Err(ForceAtlas2Error::InitialPositionsLengthMismatch {
                expected: n,
                actual: provided.len(),
            });
        }
        for (index, position) in provided.iter().enumerate() {
            if !position[0].is_finite() || !position[1].is_finite() {
                return Err(ForceAtlas2Error::NonFiniteInitialPosition { index });
            }
        }
        return Ok(provided.clone());
    }

    let mut rng = SmallRng::seed_from_u64(config.seed);
    Ok((0..n).map(|_| [rng.random_range(-0.5..0.5), rng.random_range(-0.5..0.5)]).collect())
}

/// Returns whether `row` contains an edge to `destination` whose weight is
/// approximately `weight` (per [`approx_eq`], whose tolerance is overflow
/// safe, the pitfall found by the ForceAtlas2 fuzzer).
fn has_matching_edge(row: &[(usize, f64)], destination: usize, weight: f64) -> bool {
    row.binary_search_by_key(&destination, |(col, _)| *col)
        .is_ok_and(|idx| approx_eq(weight, row[idx].1))
}
