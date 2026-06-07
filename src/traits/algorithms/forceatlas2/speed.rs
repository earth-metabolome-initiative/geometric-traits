//! Adaptive speed machinery for the ForceAtlas2 layout.
//!
//! Implements the global speed scheme of the current Gephi reference
//! implementation, which replaced the published paper equation (14) with an
//! empirical jitter estimate and the adaptive speed efficiency factor. All
//! constants below are taken verbatim from `ForceAtlas2.goAlgo` in the Java
//! source.

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

/// Mutable global speed state, carried across iterations.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct SpeedState {
    /// Global speed `s(G)`.
    pub(super) speed: f64,
    /// Adaptive speed efficiency factor (not in the paper).
    pub(super) speed_efficiency: f64,
}

impl Default for SpeedState {
    #[inline]
    fn default() -> Self {
        Self { speed: 1.0, speed_efficiency: 1.0 }
    }
}

/// Computes the mass-weighted global swinging and traction totals.
///
/// Per node, swinging is the norm of the difference between the previous
/// and current force and traction is half the norm of their sum. Both are
/// weighted by the node mass. Returns `(swinging, traction)`.
///
/// Added guard (not in the Java source): nodes whose forces overflowed to
/// non-finite values are skipped, so the totals (and through them the
/// reported run statistics) stay finite. Such overflows only occur with
/// astronomically large edge weights, where the layout dynamics are
/// already meaningless.
pub(super) fn accumulate_totals(
    masses: &[f64],
    old_forces: &[[f64; 2]],
    forces: &[[f64; 2]],
) -> (f64, f64) {
    let mut swinging = 0.0;
    let mut traction = 0.0;
    for ((mass, old), new) in masses.iter().zip(old_forces).zip(forces) {
        let diff_x = old[0] - new[0];
        let diff_y = old[1] - new[1];
        let sum_x = old[0] + new[0];
        let sum_y = old[1] + new[1];
        let node_swinging = mass * (diff_x * diff_x + diff_y * diff_y).sqrt();
        let node_traction = mass * 0.5 * (sum_x * sum_x + sum_y * sum_y).sqrt();
        if node_swinging.is_finite() && node_traction.is_finite() {
            swinging += node_swinging;
            traction += node_traction;
        }
    }
    (swinging, traction)
}

/// Updates the global speed and speed efficiency from the iteration totals.
///
/// Mirrors the Gephi master scheme exactly, with one added guard: when both
/// totals are zero (no force changed at all) the state is left untouched,
/// where the Java code would compute `0/0`.
pub(super) fn update_global_speed(
    state: &mut SpeedState,
    swinging: f64,
    traction: f64,
    number_of_nodes: usize,
    jitter_tolerance: f64,
) {
    // Added guard (not in the Java source): nothing moved at all, so there
    // is no signal to adapt on. The Java code would compute 0 / 0 here.
    if swinging <= 0.0 && traction <= 0.0 {
        return;
    }

    // The cast is exact for any realistic node count (below 2^53).
    #[allow(clippy::cast_precision_loss)]
    let n = number_of_nodes as f64;
    let estimated_optimal_jitter_tolerance = 0.05 * n.sqrt();
    let min_jt = estimated_optimal_jitter_tolerance.sqrt();
    let max_jt = 10.0_f64;
    let mut jt = jitter_tolerance
        * min_jt.max(max_jt.min(estimated_optimal_jitter_tolerance * traction / (n * n)));

    let min_speed_efficiency = 0.05;

    // Protection against erratic behavior. The strict comparison matters:
    // on the first iteration the old forces are all zero, making the ratio
    // exactly 2 (halving is exact in IEEE arithmetic), and that must not
    // trigger the erratic branch.
    if swinging / traction > 2.0 {
        if state.speed_efficiency > min_speed_efficiency {
            state.speed_efficiency *= 0.5;
        }
        jt = jt.max(jitter_tolerance);
    }

    let target_speed = jt * state.speed_efficiency * traction / swinging;

    if swinging > jt * traction {
        if state.speed_efficiency > min_speed_efficiency {
            state.speed_efficiency *= 0.7;
        }
    } else if state.speed < 1000.0 {
        state.speed_efficiency *= 1.3;
    }

    // The speed should not rise by more than 50 percent per iteration. The
    // cap also bounds the infinite target produced by zero swinging.
    let max_rise = 0.5;
    state.speed += (target_speed - state.speed).min(max_rise * state.speed);
}

/// Returns the per-node displacement factor.
///
/// `factor = speed / (1 + sqrt(speed * mass * local_swinging))` where
/// `local_swinging` is the unweighted norm of the force change of this node.
/// The mass weighting inside the square root follows current Gephi master.
/// Paper equation (9) instead keeps the swinging unweighted and includes a
/// global 0.1 factor that the Java code applies only in its adjustSizes
/// branch.
pub(super) fn per_node_factor(speed: f64, mass: f64, local_swinging: f64) -> f64 {
    speed / (1.0 + (speed * mass * local_swinging).sqrt())
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::{SpeedState, accumulate_totals, per_node_factor, update_global_speed};

    /// Two nodes with masses 2 and 3.
    ///
    /// Node 0: old (1, 0), new (0, 1): swinging sqrt(2), traction
    /// 0.5 * sqrt(2). Node 1: old (0, 2), new (0, 0): swinging 2, traction
    /// 1. Totals: swinging = 2 sqrt(2) + 6, traction = sqrt(2) + 3.
    #[test]
    fn totals_closed_form() {
        let masses = [2.0, 3.0];
        let old_forces = [[1.0, 0.0], [0.0, 2.0]];
        let forces = [[0.0, 1.0], [0.0, 0.0]];
        let (swinging, traction) = accumulate_totals(&masses, &old_forces, &forces);
        assert!((swinging - (2.0 * core::f64::consts::SQRT_2 + 6.0)).abs() < 1e-12);
        assert!((traction - (core::f64::consts::SQRT_2 + 3.0)).abs() < 1e-12);
    }

    /// Identical old and new forces swing nothing and pull with the full
    /// force norm.
    #[test]
    fn totals_steady_course() {
        let masses = [1.0];
        let old_forces = [[3.0, 4.0]];
        let forces = [[3.0, 4.0]];
        let (swinging, traction) = accumulate_totals(&masses, &old_forces, &forces);
        assert_eq!(swinging, 0.0);
        assert_eq!(traction, 5.0);
    }

    /// A perfect oscillation (new = -old) has zero traction.
    #[test]
    fn totals_perfect_oscillation() {
        let masses = [1.0];
        let old_forces = [[3.0, 4.0]];
        let forces = [[-3.0, -4.0]];
        let (swinging, traction) = accumulate_totals(&masses, &old_forces, &forces);
        assert_eq!(swinging, 10.0);
        assert_eq!(traction, 0.0);
    }

    /// Overflowed forces are skipped so the totals stay finite: only the
    /// second (sane) node contributes.
    #[test]
    fn totals_skip_non_finite_forces() {
        let masses = [1.0, 1.0];
        let old_forces = [[f64::INFINITY, 0.0], [3.0, 4.0]];
        let forces = [[f64::NEG_INFINITY, 0.0], [3.0, 4.0]];
        let (swinging, traction) = accumulate_totals(&masses, &old_forces, &forces);
        assert_eq!(swinging, 0.0);
        assert_eq!(traction, 5.0);
    }

    /// Convergent run: n = 100, traction = 10000, swinging = 1.
    ///
    /// estimated = 0.05 * 10 = 0.5, minJT = sqrt(0.5), jt = max(sqrt(0.5),
    /// min(10, 0.5 * 10000 / 100^2)) = sqrt(0.5). targetSpeed = sqrt(0.5) *
    /// 1 * 10000 / 1, far above the rise cap, so speed grows by exactly 50
    /// percent and the efficiency by 1.3 (growth branch, speed < 1000).
    #[test]
    fn speed_growth_branch_with_max_rise() {
        let mut state = SpeedState::default();
        update_global_speed(&mut state, 1.0, 10000.0, 100, 1.0);
        assert!((state.speed - 1.5).abs() < 1e-12);
        assert!((state.speed_efficiency - 1.3).abs() < 1e-12);
    }

    /// Over-swing without erratic behavior: n = 100, traction = 100,
    /// swinging = 150 (ratio 1.5, between 1 and 2).
    ///
    /// jt = sqrt(0.5) (the minJT floor). swinging > jt * traction, so the
    /// efficiency decays by 0.7 with the OLD efficiency already used in
    /// targetSpeed = jt * 1 * 100 / 150, which is below the current speed,
    /// so the speed drops straight to it.
    #[test]
    fn speed_overswing_branch() {
        let mut state = SpeedState::default();
        update_global_speed(&mut state, 150.0, 100.0, 100, 1.0);
        let jt = 0.5_f64.sqrt();
        assert!((state.speed - jt * 100.0 / 150.0).abs() < 1e-12);
        assert!((state.speed_efficiency - 0.7).abs() < 1e-12);
    }

    /// Erratic behavior: n = 100, traction = 100, swinging = 300 (ratio 3 >
    /// 2).
    ///
    /// The efficiency is first halved (0.5) and jt is raised to the raw
    /// jitter tolerance (1.0). targetSpeed = 1.0 * 0.5 * 100 / 300 = 1/6.
    /// Then the over-swing branch also fires (300 > 1.0 * 100), decaying
    /// the efficiency to 0.35. The speed drops to the target.
    #[test]
    fn speed_erratic_branch() {
        let mut state = SpeedState::default();
        update_global_speed(&mut state, 300.0, 100.0, 100, 1.0);
        assert!((state.speed - 1.0 / 6.0).abs() < 1e-12);
        assert!((state.speed_efficiency - 0.35).abs() < 1e-12);
    }

    /// The jitter tolerance is clamped at maxJT = 10.
    ///
    /// n = 100, traction = swinging = 1e9: estimated * traction / n^2 =
    /// 50000, clamped to 10. Starting from speed 8, targetSpeed = 10 * 1 *
    /// 1 = 10 and the rise cap allows 4, so the speed lands exactly on 10.
    #[test]
    fn speed_jitter_clamped_at_max() {
        let mut state = SpeedState { speed: 8.0, speed_efficiency: 1.0 };
        update_global_speed(&mut state, 1e9, 1e9, 100, 1.0);
        assert!((state.speed - 10.0).abs() < 1e-12);
        assert!((state.speed_efficiency - 1.3).abs() < 1e-12);
    }

    /// The growth branch stops raising the efficiency at speed >= 1000.
    #[test]
    fn speed_growth_gated_above_1000() {
        let mut state = SpeedState { speed: 1000.0, speed_efficiency: 1.0 };
        update_global_speed(&mut state, 1.0, 10000.0, 100, 1.0);
        assert!((state.speed - 1500.0).abs() < 1e-12);
        assert!((state.speed_efficiency - 1.0).abs() < 1e-12);
    }

    /// The efficiency never decays below the 0.05 floor.
    #[test]
    fn speed_efficiency_floor() {
        let mut state = SpeedState { speed: 1.0, speed_efficiency: 0.05 };
        update_global_speed(&mut state, 300.0, 100.0, 100, 1.0);
        // Erratic and over-swing both fire but the floor blocks both
        // decays. jt was raised to 1.0 by the erratic branch.
        assert!((state.speed_efficiency - 0.05).abs() < 1e-12);
        assert!((state.speed - 0.05 / 3.0).abs() < 1e-12);
    }

    /// Zero swinging with nonzero traction rises by the 50 percent cap
    /// (the Java code reaches an infinite target speed).
    #[test]
    fn speed_zero_swinging_rises_by_cap() {
        let mut state = SpeedState::default();
        update_global_speed(&mut state, 0.0, 10.0, 4, 1.0);
        assert!((state.speed - 1.5).abs() < 1e-12);
        assert!((state.speed_efficiency - 1.3).abs() < 1e-12);
    }

    /// Both totals zero: the guarded case leaves the state untouched.
    #[test]
    fn speed_all_zero_is_noop() {
        let mut state = SpeedState { speed: 3.0, speed_efficiency: 0.8 };
        update_global_speed(&mut state, 0.0, 0.0, 4, 1.0);
        assert_eq!(state, SpeedState { speed: 3.0, speed_efficiency: 0.8 });
    }

    /// factor = speed / (1 + sqrt(speed * mass * swinging)): with speed 4,
    /// mass 2 and local swinging 2 the square root is 4 and the factor 0.8.
    #[test]
    fn per_node_factor_closed_form() {
        assert!((per_node_factor(4.0, 2.0, 2.0) - 0.8).abs() < 1e-12);
    }

    /// A node with no swinging moves at the full global speed.
    #[test]
    fn per_node_factor_no_swinging() {
        assert_eq!(per_node_factor(2.5, 7.0, 0.0), 2.5);
    }
}
