//! Barnes-Hut quadtree for approximate ForceAtlas2 repulsion.
//!
//! Mirrors the region tree of the reference implementations (Gephi
//! `Region.java`, refined by the Python `fa2` port that this crate is
//! cross-validated against):
//!
//! - regions split at the mass center (not the geometric center), into the four
//!   quadrants ordered by the `(x >= cx) | (y >= cy) << 1` bitmask,
//! - region `size` is twice the largest node distance from the barycenter (a
//!   diameter-like quantity, not a bounding-box side),
//! - the acceptance test is `distance * theta > size`, so a larger theta means
//!   more approximation (inverse of the textbook `s/d < theta` convention,
//!   hence the unusual default of 1.2),
//! - when every node of a region falls into the same quadrant (coincident
//!   points), each node becomes its own single-node subregion instead of
//!   recursing forever,
//! - forces are one-sided and a leaf region never repulses its own node (the
//!   `fa2` refinements of Gephi's traversal, which double-counted leaf pairs).
//!
//! The tree is arena-allocated and traversed with explicit stacks, so
//! pathological inputs cannot overflow the call stack.

use alloc::{vec, vec::Vec};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use super::forces::{apply_lin_repulsion_anti_collision_one_sided, apply_region_repulsion};

/// One region of the quadtree.
#[derive(Debug, Clone)]
struct Region {
    /// Total mass of the contained nodes.
    mass: f64,
    /// Mass-weighted barycenter of the contained nodes.
    center: [f64; 2],
    /// Twice the largest node distance from the barycenter.
    size: f64,
    /// Node index when the region holds exactly one node.
    leaf: Option<usize>,
    /// Arena indices of the non-empty subregions, in quadrant order.
    children: Vec<usize>,
}

/// Arena-allocated Barnes-Hut quadtree. Region 0 is the root.
#[derive(Debug, Clone)]
pub(super) struct QuadTree {
    regions: Vec<Region>,
    /// Node ids per region, used only while building (cleared afterward).
    build_ids: Vec<Vec<usize>>,
}

impl QuadTree {
    /// Builds the tree over all nodes. Requires at least one node.
    pub(super) fn build(positions: &[[f64; 2]], masses: &[f64]) -> Self {
        let mut tree = Self { regions: Vec::new(), build_ids: Vec::new() };
        let ids: Vec<usize> = (0..positions.len()).collect();
        // Work stack of (arena index, contained node ids), explicit to
        // keep the build stack-safe on degenerate geometries.
        let root = tree.push_region(ids, positions, masses);
        let mut pending = vec![root];
        while let Some(index) = pending.pop() {
            let Some((buckets, total)) = tree.split_ids(index, positions) else { continue };
            let mut children = Vec::new();
            for bucket in buckets {
                if bucket.is_empty() {
                    continue;
                }
                if bucket.len() < total {
                    let child = tree.push_region(bucket, positions, masses);
                    pending.push(child);
                    children.push(child);
                } else {
                    // Degenerate guard: every node landed in one quadrant
                    // (coincident points), so each becomes its own leaf.
                    for id in bucket {
                        children.push(tree.push_region(vec![id], positions, masses));
                    }
                }
            }
            tree.regions[index].children = children;
        }
        tree.build_ids.clear();
        tree.build_ids.shrink_to_fit();
        tree
    }

    /// Accumulates the one-sided Barnes-Hut repulsion onto `force` for the
    /// node at `pos` with mass `node_mass`.
    ///
    /// When `sizes` is provided (anti-collision mode), exact leaf-level
    /// interactions use the size-aware kernel while accepted regions keep
    /// the plain center-distance approximation, matching the reference
    /// implementations which ignore sizes in the aggregate.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_repulsion(
        &self,
        node: usize,
        pos: [f64; 2],
        node_mass: f64,
        coefficient: f64,
        theta: f64,
        sizes: Option<&[f64]>,
        force: &mut [f64; 2],
    ) {
        // Depth-first traversal with an explicit stack, children visited
        // in quadrant order to match the reference recursion.
        let mut stack = vec![0_usize];
        while let Some(region_index) = stack.pop() {
            let region = &self.regions[region_index];
            if let Some(leaf) = region.leaf {
                // A leaf never repulses its own node.
                if leaf != node {
                    if let Some(sizes) = sizes {
                        apply_lin_repulsion_anti_collision_one_sided(
                            pos,
                            region.center,
                            node_mass,
                            region.mass,
                            sizes[node],
                            sizes[leaf],
                            coefficient,
                            force,
                        );
                    } else {
                        apply_region_repulsion(
                            pos,
                            region.center,
                            node_mass,
                            region.mass,
                            coefficient,
                            force,
                        );
                    }
                }
                continue;
            }
            let x_dist = pos[0] - region.center[0];
            let y_dist = pos[1] - region.center[1];
            let distance = (x_dist * x_dist + y_dist * y_dist).sqrt();
            if distance * theta > region.size {
                apply_region_repulsion(
                    pos,
                    region.center,
                    node_mass,
                    region.mass,
                    coefficient,
                    force,
                );
            } else {
                stack.extend(region.children.iter().rev());
            }
        }
    }

    /// Splits the ids of a multi-node region into the four quadrant
    /// buckets around its barycenter, returning them together with the
    /// total id count. Returns `None` for leaf regions.
    fn split_ids(
        &mut self,
        index: usize,
        positions: &[[f64; 2]],
    ) -> Option<([Vec<usize>; 4], usize)> {
        if self.regions[index].leaf.is_some() {
            return None;
        }
        let center = self.regions[index].center;
        let ids = core::mem::take(&mut self.build_ids[index]);
        let total = ids.len();
        let mut buckets: [Vec<usize>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for id in ids {
            let mut bucket = 0;
            if positions[id][0] >= center[0] {
                bucket |= 1;
            }
            if positions[id][1] >= center[1] {
                bucket |= 2;
            }
            buckets[bucket].push(id);
        }
        Some((buckets, total))
    }

    /// Computes the mass, barycenter and size of a region and pushes it
    /// into the arena, returning its index.
    fn push_region(&mut self, ids: Vec<usize>, positions: &[[f64; 2]], masses: &[f64]) -> usize {
        let region = if ids.len() == 1 {
            let id = ids[0];
            Region {
                mass: masses[id],
                center: positions[id],
                size: 0.0,
                leaf: Some(id),
                children: Vec::new(),
            }
        } else {
            let mut mass = 0.0;
            let mut weighted = [0.0_f64; 2];
            for &id in &ids {
                mass += masses[id];
                weighted[0] += positions[id][0] * masses[id];
                weighted[1] += positions[id][1] * masses[id];
            }
            // Masses are at least 1, so the division is well defined for
            // any non-empty region.
            let center = [weighted[0] / mass, weighted[1] / mass];
            let mut size = 0.0_f64;
            for &id in &ids {
                let x_dist = positions[id][0] - center[0];
                let y_dist = positions[id][1] - center[1];
                size = size.max(2.0 * (x_dist * x_dist + y_dist * y_dist).sqrt());
            }
            Region { mass, center, size, leaf: None, children: Vec::new() }
        };
        self.regions.push(region);
        self.build_ids.push(ids);
        self.regions.len() - 1
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use alloc::vec::Vec;

    use super::QuadTree;

    /// Root mass is the sum of node masses and the barycenter is
    /// mass-weighted: nodes at (0, 0) and (3, 0) with masses 1 and 3 give
    /// mass 4, center (2.25, 0) and size 2 * 2.25 = 4.5 (the larger of
    /// 2 * 2.25 and 2 * 0.75).
    #[test]
    fn root_mass_center_and_size() {
        let positions = [[0.0, 0.0], [3.0, 0.0]];
        let masses = [1.0, 3.0];
        let tree = QuadTree::build(&positions, &masses);
        let root = &tree.regions[0];
        assert_eq!(root.mass, 4.0);
        assert_eq!(root.center, [2.25, 0.0]);
        assert_eq!(root.size, 4.5);
    }

    /// A single node builds a one-region tree that exerts no force on
    /// itself.
    #[test]
    fn single_node_no_self_force() {
        let positions = [[1.0, 2.0]];
        let masses = [5.0];
        let tree = QuadTree::build(&positions, &masses);
        assert_eq!(tree.regions.len(), 1);
        let mut force = [0.0; 2];
        tree.apply_repulsion(0, positions[0], masses[0], 2.0, 1.2, None, &mut force);
        assert_eq!(force, [0.0, 0.0]);
    }

    /// Fully coincident nodes trigger the single-node-subregion guard
    /// instead of infinite recursion, and exert no force on each other
    /// (zero distance).
    #[test]
    fn coincident_nodes_build_finite_tree() {
        let positions = [[1.0, 1.0]; 5];
        let masses = [1.0; 5];
        let tree = QuadTree::build(&positions, &masses);
        // Root plus five single-node leaves.
        assert_eq!(tree.regions.len(), 6);
        assert_eq!(tree.regions[0].children.len(), 5);
        let mut force = [0.0; 2];
        tree.apply_repulsion(0, positions[0], masses[0], 2.0, 1.2, None, &mut force);
        assert_eq!(force, [0.0, 0.0]);
    }

    /// With a huge theta every query accepts the root region: the force on
    /// an external probe equals the super-node force `c * m * M / d` along
    /// the barycenter direction.
    #[test]
    fn huge_theta_collapses_to_super_node() {
        let positions = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [100.0, 0.0]];
        let masses = [1.0, 1.0, 1.0, 1.0, 2.0];
        let tree = QuadTree::build(&positions, &masses);
        let mut force = [0.0; 2];
        // Probe is node 4 at (100, 0). With theta -> infinity the root is
        // accepted immediately even though the probe is inside it, and the
        // probe's own mass is part of the aggregate (reference behavior).
        tree.apply_repulsion(4, positions[4], masses[4], 2.0, 1e12, None, &mut force);
        let root = &tree.regions[0];
        let x_dist = positions[4][0] - root.center[0];
        let expected = 2.0 * masses[4] * root.mass / (x_dist * x_dist) * x_dist;
        assert!((force[0] - expected).abs() < 1e-12);
        assert_eq!(force[1], 0.0);
    }

    /// With a tiny theta the traversal degenerates to exact pairwise
    /// repulsion: the result matches the brute-force one-sided sum.
    #[test]
    fn tiny_theta_matches_exact_sum() {
        let positions =
            [[0.1, 0.2], [-0.3, 0.4], [0.5, -0.1], [-0.2, -0.4], [0.7, 0.6], [-0.6, 0.1]];
        let masses = [1.0, 2.0, 3.0, 1.5, 2.5, 1.0];
        let tree = QuadTree::build(&positions, &masses);
        for node in 0..positions.len() {
            let mut bh_force = [0.0; 2];
            tree.apply_repulsion(
                node,
                positions[node],
                masses[node],
                2.0,
                1e-12,
                None,
                &mut bh_force,
            );

            let mut exact = [0.0; 2];
            for other in 0..positions.len() {
                if other == node {
                    continue;
                }
                let x_dist = positions[node][0] - positions[other][0];
                let y_dist = positions[node][1] - positions[other][1];
                let squared = x_dist * x_dist + y_dist * y_dist;
                let factor = 2.0 * masses[node] * masses[other] / squared;
                exact[0] += x_dist * factor;
                exact[1] += y_dist * factor;
            }
            assert!((bh_force[0] - exact[0]).abs() < 1e-12, "node {node}");
            assert!((bh_force[1] - exact[1]).abs() < 1e-12, "node {node}");
        }
    }

    /// A coincident cluster mixed with distinct nodes still builds a
    /// finite tree and a tiny theta still reproduces the exact one-sided
    /// sum (coincident pairs exert no force).
    #[test]
    fn partially_coincident_nodes_match_exact_sum() {
        let positions = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [-2.0, 0.5], [0.0, -3.0]];
        let masses = [1.0, 2.0, 3.0, 1.5, 2.5];
        let tree = QuadTree::build(&positions, &masses);
        for node in 0..positions.len() {
            let mut bh_force = [0.0; 2];
            tree.apply_repulsion(
                node,
                positions[node],
                masses[node],
                2.0,
                1e-12,
                None,
                &mut bh_force,
            );

            let mut exact = [0.0; 2];
            for other in 0..positions.len() {
                let x_dist = positions[node][0] - positions[other][0];
                let y_dist = positions[node][1] - positions[other][1];
                let squared = x_dist * x_dist + y_dist * y_dist;
                if other == node || squared <= 0.0 {
                    continue;
                }
                let factor = 2.0 * masses[node] * masses[other] / squared;
                exact[0] += x_dist * factor;
                exact[1] += y_dist * factor;
            }
            assert!((bh_force[0] - exact[0]).abs() < 1e-12, "node {node}");
            assert!((bh_force[1] - exact[1]).abs() < 1e-12, "node {node}");
        }
    }

    /// Leaf regions carry the node mass and position so deep traversals
    /// remain exact.
    #[test]
    fn leaves_carry_node_data() {
        let positions = [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 4.0]];
        let masses = [1.0, 2.0, 3.0, 4.0];
        let tree = QuadTree::build(&positions, &masses);
        let leaves: Vec<_> = tree.regions.iter().filter(|region| region.leaf.is_some()).collect();
        assert_eq!(leaves.len(), 4);
        for leaf in leaves {
            let node = leaf.leaf.unwrap();
            assert_eq!(leaf.mass, masses[node]);
            assert_eq!(leaf.center, positions[node]);
            assert_eq!(leaf.size, 0.0);
        }
    }
}
