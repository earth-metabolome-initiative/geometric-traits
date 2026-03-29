use crate::support::*;

#[test]
#[ignore = "focused diagnostic for the remaining 10K allBest mismatch 5585"]
fn print_massspecgym_case_10000_all_best_5585() {
    let default_cases = load_massspecgym_ground_truth_10000();
    let all_best_cases = load_massspecgym_all_best_ground_truth_10000();
    let default_case = find_case(&default_cases, "massspecgym_default_5585");
    let all_best_case = find_case(&all_best_cases, "massspecgym_default_5585");

    let partial = run_labeled_case_with_search_mode(
        all_best_case,
        true,
        false,
        McesSearchMode::PartialEnumeration,
    );
    let all_best =
        run_labeled_case_with_search_mode(all_best_case, true, false, McesSearchMode::AllBest);

    let matches_default = |info: &EagerCliqueInfo<usize>| {
        info.matched_edges().len() == default_case.expected_bond_matches
            && info.vertex_matches().len() == default_case.expected_atom_matches
            && (info_johnson_similarity(default_case, info) - default_case.expected_similarity)
                .abs()
                <= 1e-6
    };
    let matches_all_best = |info: &EagerCliqueInfo<usize>| {
        info.matched_edges().len() == all_best_case.expected_bond_matches
            && info.vertex_matches().len() == all_best_case.expected_atom_matches
            && (info_johnson_similarity(all_best_case, info) - all_best_case.expected_similarity)
                .abs()
                <= 1e-6
    };

    let partial_default_index = partial.all_cliques().iter().position(matches_default);
    let partial_all_best_index = partial.all_cliques().iter().position(matches_all_best);
    let all_best_default_index = all_best.all_cliques().iter().position(matches_default);
    let all_best_all_best_index = all_best.all_cliques().iter().position(matches_all_best);

    println!("case: {}", all_best_case.name);
    println!(
        "default_fixture bonds={} atoms={} similarity={:.6}",
        default_case.expected_bond_matches,
        default_case.expected_atom_matches,
        default_case.expected_similarity,
    );
    println!(
        "all_best_fixture bonds={} atoms={} similarity={:.6}",
        all_best_case.expected_bond_matches,
        all_best_case.expected_atom_matches,
        all_best_case.expected_similarity,
    );
    println!(
        "partial: retained={} top_bonds={} top_atoms={} top_similarity={:.6} default_index={partial_default_index:?} all_best_index={partial_all_best_index:?}",
        partial.all_cliques().len(),
        partial.matched_edges().len(),
        partial.vertex_matches().len(),
        partial.johnson_similarity(),
    );
    println!(
        "all_best: retained={} top_bonds={} top_atoms={} top_similarity={:.6} default_index={all_best_default_index:?} all_best_index={all_best_all_best_index:?}",
        all_best.all_cliques().len(),
        all_best.matched_edges().len(),
        all_best.vertex_matches().len(),
        all_best.johnson_similarity(),
    );

    for (index, info) in all_best.all_cliques().iter().take(12).enumerate() {
        println!(
            "  #{index}: bonds={} atoms={} similarity={:.6} fragments={} largest_fragment_atoms={}",
            info.matched_edges().len(),
            info.vertex_matches().len(),
            info_johnson_similarity(all_best_case, info),
            info.fragment_count(),
            info.largest_fragment_atom_count(),
        );
    }
}

#[test]
#[ignore = "focused scorecard diagnostic for the remaining 10K allBest mismatch 5585"]
fn print_massspecgym_case_10000_all_best_5585_scorecards() {
    let cases = load_massspecgym_all_best_ground_truth_10000();
    let case = find_case(&cases, "massspecgym_default_5585");
    let result = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
    let scorecards = scorecards_for_infos(case, result.all_cliques());

    let expected_index = scorecards.iter().position(|scorecard| {
        scorecard.matched_bonds == case.expected_bond_matches
            && scorecard.matched_atoms == case.expected_atom_matches
            && (geometric_traits::traits::algorithms::johnson_similarity(
                scorecard.matched_bonds,
                scorecard.matched_atoms,
                case.graph1.n_atoms,
                case.graph1.edges.len(),
                case.graph2.n_atoms,
                case.graph2.edges.len(),
            ) - case.expected_similarity)
                .abs()
                <= 1e-6
    });

    let mut approx_rdkit_order: Vec<usize> = (0..scorecards.len()).collect();
    approx_rdkit_order.sort_unstable_by(|&left, &right| {
        approx_rdkit_compare(&scorecards[left], &scorecards[right]).then_with(|| left.cmp(&right))
    });
    let first_missing_order =
        ranked_indices_by_scorecards(&scorecards, first_missing_rdkit_compare);

    println!("case: {}", case.name);
    println!(
        "fixture all_best bonds={} atoms={} similarity={:.6} expected_index={expected_index:?}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity,
    );
    println!("rust_order:");
    for (index, scorecard) in scorecards.iter().take(12).enumerate() {
        println!(
            "  #{index}: bonds={} atoms={} fragments={} largest_fragment={} ring_non_ring={} atom_h={} max_delta={}",
            scorecard.matched_bonds,
            scorecard.matched_atoms,
            scorecard.fragment_count,
            scorecard.largest_fragment_size,
            scorecard.ring_non_ring_bond_score,
            scorecard.atom_h_score,
            scorecard.max_delta_atom_atom_dist,
        );
    }
    println!("first_missing_ranker_order:");
    for &index in first_missing_order.iter().take(12) {
        let scorecard = &scorecards[index];
        println!(
            "  #{index}: bonds={} atoms={} fragments={} largest_fragment={} ring_non_ring={} atom_h={} max_delta={}",
            scorecard.matched_bonds,
            scorecard.matched_atoms,
            scorecard.fragment_count,
            scorecard.largest_fragment_size,
            scorecard.ring_non_ring_bond_score,
            scorecard.atom_h_score,
            scorecard.max_delta_atom_atom_dist,
        );
    }
    println!("approx_rdkit_order:");
    for &index in approx_rdkit_order.iter().take(12) {
        let scorecard = &scorecards[index];
        println!(
            "  #{index}: bonds={} atoms={} fragments={} largest_fragment={} ring_non_ring={} atom_h={} max_delta={}",
            scorecard.matched_bonds,
            scorecard.matched_atoms,
            scorecard.fragment_count,
            scorecard.largest_fragment_size,
            scorecard.ring_non_ring_bond_score,
            scorecard.atom_h_score,
            scorecard.max_delta_atom_atom_dist,
        );
    }
}
