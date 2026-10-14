//! Force kernels for the ForceAtlas2 layout.
//!
//! All kernels follow the Gephi convention of multiplying the raw
//! center-to-center difference vector by a scalar factor, so a division by
//! the distance is folded into the factor. Every distance-dividing kernel
//! silently skips coincident points, mirroring the zero-distance guards of
//! the Java `ForceFactory` classes.

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

/// Adds `factor` times the difference vector to `force1` and subtracts it
/// from `force2`.
fn apply_to_pair(
    x_dist: f64,
    y_dist: f64,
    factor: f64,
    force1: &mut [f64; 2],
    force2: &mut [f64; 2],
) {
    force1[0] += x_dist * factor;
    force1[1] += y_dist * factor;
    force2[0] -= x_dist * factor;
    force2[1] -= y_dist * factor;
}

/// Accumulates the degree-weighted linear repulsion between two nodes onto
/// both force accumulators.
///
/// Force magnitude is `coefficient * mass1 * mass2 / d`, pushing the nodes
/// apart. `coefficient` is the scaling ratio `kr`. Coincident nodes exert no
/// force.
pub(super) fn apply_lin_repulsion(
    pos1: [f64; 2],
    pos2: [f64; 2],
    mass1: f64,
    mass2: f64,
    coefficient: f64,
    force1: &mut [f64; 2],
    force2: &mut [f64; 2],
) {
    let x_dist = pos1[0] - pos2[0];
    let y_dist = pos1[1] - pos2[1];
    let squared_distance = x_dist * x_dist + y_dist * y_dist;
    // Zero-distance guard: coincident nodes exert no repulsion.
    if squared_distance > 0.0 {
        // factor = coefficient * m1 * m2 / d^2, applied to the raw
        // difference vector, so the force magnitude is c * m1 * m2 / d.
        let factor = coefficient * mass1 * mass2 / squared_distance;
        apply_to_pair(x_dist, y_dist, factor, force1, force2);
    }
}

/// Accumulates the attraction along one edge onto both force accumulators,
/// in the variant selected by the mode flags. Mirrors the four
/// non-anti-collision classes of the Java `buildAttraction` decision tree.
///
/// In linear mode the factor is `-coefficient * e` (force magnitude
/// `c * e * d`, no distance guard needed because nothing divides by the
/// distance). In LinLog mode it is `-coefficient * e * log(1 + d) / d`
/// (force magnitude `c * e * log(1 + d)`, coincident nodes guarded, which
/// also covers the `log(0)` concern the paper addresses with `log(1 +
/// d)`). Dissuade hubs divides the factor by the source node mass, with
/// `coefficient` carrying the global mean-mass compensation in that mode.
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_attraction(
    lin_log: bool,
    dissuade_hubs: bool,
    pos1: [f64; 2],
    pos2: [f64; 2],
    e: f64,
    coefficient: f64,
    source_mass: f64,
    force1: &mut [f64; 2],
    force2: &mut [f64; 2],
) {
    let x_dist = pos1[0] - pos2[0];
    let y_dist = pos1[1] - pos2[1];
    let mut factor = if lin_log {
        let distance = (x_dist * x_dist + y_dist * y_dist).sqrt();
        // Zero-distance guard: coincident nodes exert no attraction.
        if distance <= 0.0 {
            return;
        }
        -coefficient * e * distance.ln_1p() / distance
    } else {
        -coefficient * e
    };
    if dissuade_hubs {
        factor /= source_mass;
    }
    apply_to_pair(x_dist, y_dist, factor, force1, force2);
}

/// Returns the anti-collision repulsion factor for the given difference
/// vector, or `None` for exactly touching borders.
///
/// The gating distance is border-to-border: `d' = d - size1 - size2`.
/// Separated nodes (`d' > 0`) repulse with `coefficient * m1 * m2 / d'^2`
/// applied to the raw center-to-center vector. Overlapping nodes
/// (`d' < 0`) get the constant overlap kick `100 * coefficient * m1 * m2`
/// (also on the center vector, so the push grows with center separation).
/// All per the Java `linRepulsion_antiCollision` class.
fn anti_collision_repulsion_factor(
    x_dist: f64,
    y_dist: f64,
    mass1: f64,
    mass2: f64,
    size1: f64,
    size2: f64,
    coefficient: f64,
) -> Option<f64> {
    let border_distance = (x_dist * x_dist + y_dist * y_dist).sqrt() - size1 - size2;
    if border_distance > 0.0 {
        Some(coefficient * mass1 * mass2 / border_distance / border_distance)
    } else if border_distance < 0.0 {
        Some(100.0 * coefficient * mass1 * mass2)
    } else {
        None
    }
}

/// Accumulates the anti-collision (prevent overlap) repulsion between two
/// sized nodes onto both force accumulators. See
/// [`anti_collision_repulsion_factor`] for the force law.
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_lin_repulsion_anti_collision(
    pos1: [f64; 2],
    pos2: [f64; 2],
    mass1: f64,
    mass2: f64,
    size1: f64,
    size2: f64,
    coefficient: f64,
    force1: &mut [f64; 2],
    force2: &mut [f64; 2],
) {
    let x_dist = pos1[0] - pos2[0];
    let y_dist = pos1[1] - pos2[1];
    if let Some(factor) =
        anti_collision_repulsion_factor(x_dist, y_dist, mass1, mass2, size1, size2, coefficient)
    {
        apply_to_pair(x_dist, y_dist, factor, force1, force2);
    }
}

/// One-sided variant of [`apply_lin_repulsion_anti_collision`], used at
/// Barnes-Hut leaves when node sizes are active.
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_lin_repulsion_anti_collision_one_sided(
    pos: [f64; 2],
    other_pos: [f64; 2],
    node_mass: f64,
    other_mass: f64,
    node_size: f64,
    other_size: f64,
    coefficient: f64,
    force: &mut [f64; 2],
) {
    let x_dist = pos[0] - other_pos[0];
    let y_dist = pos[1] - other_pos[1];
    if let Some(factor) = anti_collision_repulsion_factor(
        x_dist,
        y_dist,
        node_mass,
        other_mass,
        node_size,
        other_size,
        coefficient,
    ) {
        force[0] += x_dist * factor;
        force[1] += y_dist * factor;
    }
}

/// Accumulates the anti-collision attraction along one edge onto both
/// force accumulators.
///
/// Attraction is gated on the border-to-border distance: overlapping or
/// touching nodes (`d' <= 0`) receive no attraction. Inside the gate the
/// factors match [`apply_attraction`], with the LinLog variant using the
/// border distance (per the Java `*_antiCollision` classes).
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_attraction_anti_collision(
    lin_log: bool,
    dissuade_hubs: bool,
    pos1: [f64; 2],
    pos2: [f64; 2],
    e: f64,
    coefficient: f64,
    source_mass: f64,
    size1: f64,
    size2: f64,
    force1: &mut [f64; 2],
    force2: &mut [f64; 2],
) {
    let x_dist = pos1[0] - pos2[0];
    let y_dist = pos1[1] - pos2[1];
    let border_distance = (x_dist * x_dist + y_dist * y_dist).sqrt() - size1 - size2;
    if border_distance <= 0.0 {
        return;
    }
    let mut factor = if lin_log {
        -coefficient * e * border_distance.ln_1p() / border_distance
    } else {
        -coefficient * e
    };
    if dissuade_hubs {
        factor /= source_mass;
    }
    apply_to_pair(x_dist, y_dist, factor, force1, force2);
}

/// Accumulates the one-sided repulsion of a Barnes-Hut region (a
/// super-node of aggregated mass at the barycenter) onto a node's force
/// accumulator.
///
/// Same formula as [`apply_lin_repulsion`] but only the node moves, the
/// region is an aggregate. Coincident node and barycenter exert no force.
pub(super) fn apply_region_repulsion(
    pos: [f64; 2],
    region_center: [f64; 2],
    node_mass: f64,
    region_mass: f64,
    coefficient: f64,
    force: &mut [f64; 2],
) {
    let x_dist = pos[0] - region_center[0];
    let y_dist = pos[1] - region_center[1];
    let squared_distance = x_dist * x_dist + y_dist * y_dist;
    if squared_distance > 0.0 {
        let factor = coefficient * node_mass * region_mass / squared_distance;
        force[0] += x_dist * factor;
        force[1] += y_dist * factor;
    }
}

/// Accumulates the gravity force toward the origin onto the force
/// accumulator.
///
/// In normal mode the force magnitude is `kg * mass`, independent of the
/// distance. In strong mode it is `kg * mass * d`. Nodes exactly at the
/// origin receive no force.
pub(super) fn apply_gravity(pos: [f64; 2], mass: f64, kg: f64, strong: bool, force: &mut [f64; 2]) {
    let distance = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt();
    // Zero-distance guard: a node at the origin receives no gravity.
    if distance > 0.0 {
        // In Gephi the gravity constant is passed as g / scaling to a force
        // whose coefficient is the scaling ratio, the two cancel. The net
        // factor is mass * kg / d (normal) or mass * kg (strong).
        let factor = if strong { kg * mass } else { kg * mass / distance };
        force[0] -= pos[0] * factor;
        force[1] -= pos[1] * factor;
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::{
        apply_attraction, apply_attraction_anti_collision, apply_gravity, apply_lin_repulsion,
        apply_lin_repulsion_anti_collision, apply_lin_repulsion_anti_collision_one_sided,
    };

    /// Log attraction between (0, 0) and (3, 4): d = 5, e = 2, coefficient
    /// 1.
    ///
    /// factor = -2 * ln(6) / 5, diff = (-3, -4), so node 1 gains
    /// (6 ln(6) / 5, 8 ln(6) / 5). The magnitude is e * ln(1 + d) =
    /// 2 ln(6).
    #[test]
    fn log_attraction_closed_form() {
        let mut f1 = [0.0; 2];
        let mut f2 = [0.0; 2];
        apply_attraction(true, false, [0.0, 0.0], [3.0, 4.0], 2.0, 1.0, 1.0, &mut f1, &mut f2);
        let ln6 = 6.0_f64.ln();
        assert!((f1[0] - 6.0 * ln6 / 5.0).abs() < 1e-12);
        assert!((f1[1] - 8.0 * ln6 / 5.0).abs() < 1e-12);
        assert!((f2[0] + 6.0 * ln6 / 5.0).abs() < 1e-12);
        assert!((f2[1] + 8.0 * ln6 / 5.0).abs() < 1e-12);
    }

    /// Log attraction between coincident nodes is a guarded no-op.
    #[test]
    fn log_attraction_coincident_is_noop() {
        let mut f1 = [1.0, 2.0];
        let mut f2 = [3.0, 4.0];
        apply_attraction(true, false, [7.0, -7.0], [7.0, -7.0], 2.0, 1.0, 1.0, &mut f1, &mut f2);
        assert_eq!(f1, [1.0, 2.0]);
        assert_eq!(f2, [3.0, 4.0]);
    }

    /// Log attraction is antisymmetric.
    #[test]
    fn log_attraction_antisymmetric() {
        let mut f1 = [0.0; 2];
        let mut f2 = [0.0; 2];
        apply_attraction(true, false, [1.0, -2.0], [-0.5, 3.0], 4.0, 2.0, 1.0, &mut f1, &mut f2);
        assert_eq!(f1[0], -f2[0]);
        assert_eq!(f1[1], -f2[1]);
    }

    /// Dissuaded linear attraction divides by the source mass: with mass 4,
    /// compensation coefficient 2 and e = 3, factor = -2 * 3 / 4 = -1.5,
    /// diff = (-3, -4), so node 1 gains (4.5, 6).
    #[test]
    fn lin_attraction_dissuade_closed_form() {
        let mut f1 = [0.0; 2];
        let mut f2 = [0.0; 2];
        apply_attraction(false, true, [1.0, 1.0], [4.0, 5.0], 3.0, 2.0, 4.0, &mut f1, &mut f2);
        assert!((f1[0] - 4.5).abs() < 1e-12);
        assert!((f1[1] - 6.0).abs() < 1e-12);
        assert!((f2[0] + 4.5).abs() < 1e-12);
        assert!((f2[1] + 6.0).abs() < 1e-12);
    }

    /// Halving the source mass doubles the dissuaded attraction.
    #[test]
    fn lin_attraction_dissuade_mass_scaling() {
        let mut heavy = [0.0; 2];
        let mut light = [0.0; 2];
        let mut sink = [0.0; 2];
        apply_attraction(false, true, [0.0, 0.0], [3.0, 4.0], 1.0, 1.0, 4.0, &mut heavy, &mut sink);
        apply_attraction(false, true, [0.0, 0.0], [3.0, 4.0], 1.0, 1.0, 2.0, &mut light, &mut sink);
        assert!((light[0] - 2.0 * heavy[0]).abs() < 1e-12);
        assert!((light[1] - 2.0 * heavy[1]).abs() < 1e-12);
    }

    /// Dissuaded log attraction between (0, 0) and (3, 4): d = 5, e = 2,
    /// coefficient 1, source mass 4.
    ///
    /// factor = -2 * ln(6) / 5 / 4 and diff = (-3, -4), so node 1 gains
    /// (6 ln(6) / 20, 8 ln(6) / 20).
    #[test]
    fn log_attraction_dissuade_closed_form() {
        let mut f1 = [0.0; 2];
        let mut f2 = [0.0; 2];
        apply_attraction(true, true, [0.0, 0.0], [3.0, 4.0], 2.0, 1.0, 4.0, &mut f1, &mut f2);
        let ln6 = 6.0_f64.ln();
        assert!((f1[0] - 6.0 * ln6 / 20.0).abs() < 1e-12);
        assert!((f1[1] - 8.0 * ln6 / 20.0).abs() < 1e-12);
    }

    /// Dissuaded log attraction between coincident nodes is a guarded
    /// no-op.
    #[test]
    fn log_attraction_dissuade_coincident_is_noop() {
        let mut f1 = [1.0, 2.0];
        let mut f2 = [3.0, 4.0];
        apply_attraction(true, true, [7.0, -7.0], [7.0, -7.0], 2.0, 1.0, 4.0, &mut f1, &mut f2);
        assert_eq!(f1, [1.0, 2.0]);
        assert_eq!(f2, [3.0, 4.0]);
    }

    /// Anti-collision repulsion with separated borders: nodes at (0, 0)
    /// and (3, 4) with sizes 1 each have border distance 5 - 2 = 3, so
    /// factor = 2 * 2 * 3 / 9 = 4/3 on the raw center vector (-3, -4).
    #[test]
    fn anti_collision_repulsion_separated() {
        let mut f1 = [0.0; 2];
        let mut f2 = [0.0; 2];
        apply_lin_repulsion_anti_collision(
            [0.0, 0.0],
            [3.0, 4.0],
            2.0,
            3.0,
            1.0,
            1.0,
            2.0,
            &mut f1,
            &mut f2,
        );
        let factor = 2.0 * 2.0 * 3.0 / 9.0;
        assert!((f1[0] - (-3.0 * factor)).abs() < 1e-12);
        assert!((f1[1] - (-4.0 * factor)).abs() < 1e-12);
        assert!((f2[0] - 3.0 * factor).abs() < 1e-12);
        assert!((f2[1] - 4.0 * factor).abs() < 1e-12);
    }

    /// Anti-collision repulsion with overlap: sizes 3 each give border
    /// distance 5 - 6 = -1, so the exact overlap kick factor is
    /// 100 * 2 * 2 * 3 = 1200 on the raw center vector.
    #[test]
    fn anti_collision_repulsion_overlap_kick() {
        let mut f1 = [0.0; 2];
        let mut f2 = [0.0; 2];
        apply_lin_repulsion_anti_collision(
            [0.0, 0.0],
            [3.0, 4.0],
            2.0,
            3.0,
            3.0,
            3.0,
            2.0,
            &mut f1,
            &mut f2,
        );
        assert!((f1[0] - (-3.0 * 1200.0)).abs() < 1e-9);
        assert!((f1[1] - (-4.0 * 1200.0)).abs() < 1e-9);
    }

    /// Exactly touching borders (d' == 0) exert no repulsion.
    #[test]
    fn anti_collision_repulsion_touching_is_noop() {
        let mut f1 = [1.0, 2.0];
        let mut f2 = [3.0, 4.0];
        apply_lin_repulsion_anti_collision(
            [0.0, 0.0],
            [3.0, 4.0],
            2.0,
            3.0,
            2.5,
            2.5,
            2.0,
            &mut f1,
            &mut f2,
        );
        assert_eq!(f1, [1.0, 2.0]);
        assert_eq!(f2, [3.0, 4.0]);
    }

    /// The one-sided variant moves only the queried node.
    #[test]
    fn anti_collision_one_sided_matches_pairwise() {
        let mut pair_f1 = [0.0; 2];
        let mut pair_f2 = [0.0; 2];
        apply_lin_repulsion_anti_collision(
            [0.0, 0.0],
            [3.0, 4.0],
            2.0,
            3.0,
            1.0,
            1.0,
            2.0,
            &mut pair_f1,
            &mut pair_f2,
        );
        let mut single = [0.0; 2];
        apply_lin_repulsion_anti_collision_one_sided(
            [0.0, 0.0],
            [3.0, 4.0],
            2.0,
            3.0,
            1.0,
            1.0,
            2.0,
            &mut single,
        );
        assert_eq!(single, pair_f1);
    }

    /// The one-sided variant also skips exactly touching borders.
    #[test]
    fn anti_collision_one_sided_touching_is_noop() {
        let mut force = [1.0, 2.0];
        apply_lin_repulsion_anti_collision_one_sided(
            [0.0, 0.0],
            [3.0, 4.0],
            2.0,
            3.0,
            2.5,
            2.5,
            2.0,
            &mut force,
        );
        assert_eq!(force, [1.0, 2.0]);
    }

    /// Anti-collision attraction is gated off for overlapping or touching
    /// nodes.
    #[test]
    fn anti_collision_attraction_gated_on_overlap() {
        for (size1, size2) in [(3.0, 3.0), (2.5, 2.5)] {
            let mut f1 = [1.0, 1.0];
            let mut f2 = [2.0, 2.0];
            apply_attraction_anti_collision(
                false,
                false,
                [0.0, 0.0],
                [3.0, 4.0],
                2.0,
                1.0,
                1.0,
                size1,
                size2,
                &mut f1,
                &mut f2,
            );
            assert_eq!(f1, [1.0, 1.0]);
            assert_eq!(f2, [2.0, 2.0]);
        }
    }

    /// Separated anti-collision linear attraction uses the plain factor
    /// `-coefficient * e` on the raw center vector.
    #[test]
    fn anti_collision_attraction_linear_closed_form() {
        let mut f1 = [0.0; 2];
        let mut f2 = [0.0; 2];
        apply_attraction_anti_collision(
            false,
            false,
            [0.0, 0.0],
            [3.0, 4.0],
            2.0,
            1.0,
            1.0,
            1.0,
            1.0,
            &mut f1,
            &mut f2,
        );
        assert!((f1[0] - 6.0).abs() < 1e-12);
        assert!((f1[1] - 8.0).abs() < 1e-12);
    }

    /// Separated anti-collision LinLog attraction uses the BORDER distance
    /// in both the logarithm and the divisor: factor =
    /// -e * ln(1 + 3) / 3 with border distance 3.
    #[test]
    fn anti_collision_attraction_linlog_uses_border_distance() {
        let mut f1 = [0.0; 2];
        let mut f2 = [0.0; 2];
        apply_attraction_anti_collision(
            true,
            false,
            [0.0, 0.0],
            [3.0, 4.0],
            2.0,
            1.0,
            1.0,
            1.0,
            1.0,
            &mut f1,
            &mut f2,
        );
        let factor = 2.0 * 4.0_f64.ln() / 3.0;
        assert!((f1[0] - 3.0 * factor).abs() < 1e-12);
        assert!((f1[1] - 4.0 * factor).abs() < 1e-12);
    }

    /// The dissuade hubs division applies inside the anti-collision gate.
    #[test]
    fn anti_collision_attraction_dissuade_divides_by_mass() {
        let mut plain = [0.0; 2];
        let mut dissuaded = [0.0; 2];
        let mut sink = [0.0; 2];
        apply_attraction_anti_collision(
            false,
            false,
            [0.0, 0.0],
            [3.0, 4.0],
            2.0,
            1.0,
            1.0,
            1.0,
            1.0,
            &mut plain,
            &mut sink,
        );
        apply_attraction_anti_collision(
            false,
            true,
            [0.0, 0.0],
            [3.0, 4.0],
            2.0,
            1.0,
            4.0,
            1.0,
            1.0,
            &mut dissuaded,
            &mut sink,
        );
        assert!((dissuaded[0] - plain[0] / 4.0).abs() < 1e-12);
        assert!((dissuaded[1] - plain[1] / 4.0).abs() < 1e-12);
    }

    /// Repulsion between (0, 0) and (3, 4): d = 5, masses 2 and 3, kr = 2.
    ///
    /// factor = 2 * 2 * 3 / 5^2 = 0.48, diff = (-3, -4), so node 1 gains
    /// (-1.44, -1.92) and node 2 the opposite. The resulting magnitude is
    /// 0.48 * 5 = 2.4 = kr * m1 * m2 / d.
    #[test]
    fn lin_repulsion_closed_form() {
        let mut f1 = [0.0; 2];
        let mut f2 = [0.0; 2];
        apply_lin_repulsion([0.0, 0.0], [3.0, 4.0], 2.0, 3.0, 2.0, &mut f1, &mut f2);
        assert!((f1[0] - (-1.44)).abs() < 1e-12);
        assert!((f1[1] - (-1.92)).abs() < 1e-12);
        assert!((f2[0] - 1.44).abs() < 1e-12);
        assert!((f2[1] - 1.92).abs() < 1e-12);
    }

    /// Repulsion accumulates on top of existing forces.
    #[test]
    fn lin_repulsion_accumulates() {
        let mut f1 = [10.0, 20.0];
        let mut f2 = [0.0; 2];
        apply_lin_repulsion([0.0, 0.0], [3.0, 4.0], 2.0, 3.0, 2.0, &mut f1, &mut f2);
        assert!((f1[0] - (10.0 - 1.44)).abs() < 1e-12);
        assert!((f1[1] - (20.0 - 1.92)).abs() < 1e-12);
    }

    /// Coincident nodes exert no repulsion (zero-distance guard).
    #[test]
    fn lin_repulsion_coincident_is_noop() {
        let mut f1 = [1.0, 2.0];
        let mut f2 = [3.0, 4.0];
        apply_lin_repulsion([7.0, -7.0], [7.0, -7.0], 2.0, 3.0, 2.0, &mut f1, &mut f2);
        assert_eq!(f1, [1.0, 2.0]);
        assert_eq!(f2, [3.0, 4.0]);
    }

    /// Repulsion is antisymmetric: the same force with opposite signs.
    #[test]
    fn lin_repulsion_antisymmetric() {
        let mut f1 = [0.0; 2];
        let mut f2 = [0.0; 2];
        apply_lin_repulsion([1.0, -2.0], [-0.5, 3.0], 4.0, 5.0, 2.0, &mut f1, &mut f2);
        assert_eq!(f1[0], -f2[0]);
        assert_eq!(f1[1], -f2[1]);
    }

    /// Attraction between (1, 1) and (4, 5) with e = 2 and coefficient 1.
    ///
    /// factor = -1 * 2 = -2, diff = (-3, -4), so node 1 gains (6, 8) (pulled
    /// toward node 2) and node 2 gains (-6, -8). The magnitude is e * d =
    /// 10.
    #[test]
    fn lin_attraction_closed_form() {
        let mut f1 = [0.0; 2];
        let mut f2 = [0.0; 2];
        apply_attraction(false, false, [1.0, 1.0], [4.0, 5.0], 2.0, 1.0, 1.0, &mut f1, &mut f2);
        assert!((f1[0] - 6.0).abs() < 1e-12);
        assert!((f1[1] - 8.0).abs() < 1e-12);
        assert!((f2[0] - (-6.0)).abs() < 1e-12);
        assert!((f2[1] - (-8.0)).abs() < 1e-12);
    }

    /// A zero-weight edge exerts no attraction.
    #[test]
    fn lin_attraction_zero_weight_is_noop() {
        let mut f1 = [1.0, 1.0];
        let mut f2 = [2.0, 2.0];
        apply_attraction(false, false, [1.0, 1.0], [4.0, 5.0], 0.0, 1.0, 1.0, &mut f1, &mut f2);
        assert_eq!(f1, [1.0, 1.0]);
        assert_eq!(f2, [2.0, 2.0]);
    }

    /// Attraction between coincident nodes is zero (no NaN).
    #[test]
    fn lin_attraction_coincident_is_noop() {
        let mut f1 = [0.0; 2];
        let mut f2 = [0.0; 2];
        apply_attraction(false, false, [4.0, 5.0], [4.0, 5.0], 2.0, 1.0, 1.0, &mut f1, &mut f2);
        assert_eq!(f1, [0.0, 0.0]);
        assert_eq!(f2, [0.0, 0.0]);
    }

    /// Normal gravity at (3, 4) with mass 2 and kg = 1.5.
    ///
    /// d = 5, factor = 2 * 1.5 / 5 = 0.6, force = -(3, 4) * 0.6 =
    /// (-1.8, -2.4), magnitude mass * kg = 3, independent of d.
    #[test]
    fn gravity_normal_closed_form() {
        let mut f = [0.0; 2];
        apply_gravity([3.0, 4.0], 2.0, 1.5, false, &mut f);
        assert!((f[0] - (-1.8)).abs() < 1e-12);
        assert!((f[1] - (-2.4)).abs() < 1e-12);
    }

    /// Strong gravity at (3, 4) with mass 2 and kg = 1.5.
    ///
    /// factor = 2 * 1.5 = 3, force = -(3, 4) * 3 = (-9, -12), magnitude
    /// mass * kg * d = 15.
    #[test]
    fn gravity_strong_closed_form() {
        let mut f = [0.0; 2];
        apply_gravity([3.0, 4.0], 2.0, 1.5, true, &mut f);
        assert!((f[0] - (-9.0)).abs() < 1e-12);
        assert!((f[1] - (-12.0)).abs() < 1e-12);
    }

    /// A node exactly at the origin receives no gravity (zero-distance
    /// guard).
    #[test]
    fn gravity_at_origin_is_noop() {
        let mut f = [5.0, 6.0];
        apply_gravity([0.0, 0.0], 2.0, 1.5, false, &mut f);
        assert_eq!(f, [5.0, 6.0]);
        apply_gravity([0.0, 0.0], 2.0, 1.5, true, &mut f);
        assert_eq!(f, [5.0, 6.0]);
    }

    /// Zero gravity constant is a no-op in both modes.
    #[test]
    fn gravity_zero_constant_is_noop() {
        let mut f = [0.0; 2];
        apply_gravity([3.0, 4.0], 2.0, 0.0, false, &mut f);
        assert_eq!(f, [0.0, 0.0]);
        apply_gravity([3.0, 4.0], 2.0, 0.0, true, &mut f);
        assert_eq!(f, [0.0, 0.0]);
    }
}
