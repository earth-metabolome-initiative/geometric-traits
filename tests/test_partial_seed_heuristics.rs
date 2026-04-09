//! Focused timing experiment for root incumbent heuristics on selected 200K
//! cases.
#![cfg(feature = "std")]
#![allow(dead_code)]

#[path = "test_mces_ground_truth/support.rs"]
mod support;

use std::{env, time::Instant};

use support::*;

#[test]
#[ignore = "timing experiment for root incumbent heuristics on selected 200K labeled cases"]
fn test_selected_200k_cases_partial_seed_heuristics() {
    let default_case_names = ["massspecgym_default_26787", "massspecgym_default_134104"];
    let case_names: Vec<String> = env::var("CASE_NAMES")
        .ok()
        .map(|raw| {
            raw.split(',')
                .filter(|item| !item.trim().is_empty())
                .map(|item| item.trim().to_string())
                .collect()
        })
        .unwrap_or_else(|| default_case_names.iter().map(|name| (*name).to_string()).collect());
    let heuristic = env::var("PARTIAL_SEED_HEURISTIC").unwrap_or_else(|_| "current".to_string());
    let cases = load_massspecgym_ground_truth_200000();

    println!("partial_seed_heuristic={heuristic}");
    for case_name in case_names {
        let case = find_case(&cases, &case_name);
        let started = Instant::now();
        let result = run_labeled_case(case);
        let elapsed = started.elapsed().as_secs_f64();
        println!(
            "case={} matched_edges={} expected_edges={} similarity={:.6} expected_similarity={:.6} elapsed_s={:.6}",
            case.name,
            result.matched_edges().len(),
            case.expected_bond_matches,
            result.johnson_similarity(),
            case.expected_similarity,
            elapsed,
        );
        assert_labeled_result_matches_ground_truth(
            case,
            &result,
            "partial seed heuristic experiment",
        );
    }
}
