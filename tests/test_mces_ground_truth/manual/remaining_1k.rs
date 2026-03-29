use crate::support::*;

#[test]
#[ignore = "focused diagnostic for the remaining 1K default mismatch 0719"]
fn print_massspecgym_case_1000_0719_partial_vs_all_best() {
    let cases = load_massspecgym_ground_truth_1000();
    let case = find_case(&cases, "massspecgym_default_0719");
    let partial =
        run_labeled_case_with_search_mode(case, true, false, McesSearchMode::PartialEnumeration);
    let partial_with_orientation =
        run_labeled_case_with_search_mode(case, true, true, McesSearchMode::PartialEnumeration);
    let all_best = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);

    let matches_expected = |info: &EagerCliqueInfo<usize>| {
        info.matched_edges().len() == case.expected_bond_matches
            && info.vertex_matches().len() == case.expected_atom_matches
            && (info_johnson_similarity(case, info) - case.expected_similarity).abs() <= 1e-6
    };

    let partial_expected = partial.all_cliques().iter().position(matches_expected);
    let partial_orientation_expected =
        partial_with_orientation.all_cliques().iter().position(matches_expected);
    let all_best_expected = all_best.all_cliques().iter().position(matches_expected);

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!(
        "partial: retained={} top_bonds={} top_atoms={} top_similarity={:.6} expected_index={partial_expected:?}",
        partial.all_cliques().len(),
        partial.matched_edges().len(),
        partial.vertex_matches().len(),
        partial.johnson_similarity(),
    );
    println!(
        "partial+orientation: retained={} top_bonds={} top_atoms={} top_similarity={:.6} expected_index={partial_orientation_expected:?}",
        partial_with_orientation.all_cliques().len(),
        partial_with_orientation.matched_edges().len(),
        partial_with_orientation.vertex_matches().len(),
        partial_with_orientation.johnson_similarity(),
    );
    println!(
        "all_best: retained={} top_bonds={} top_atoms={} top_similarity={:.6} expected_index={all_best_expected:?}",
        all_best.all_cliques().len(),
        all_best.matched_edges().len(),
        all_best.vertex_matches().len(),
        all_best.johnson_similarity(),
    );

    println!("partial_order:");
    for (index, info) in partial.all_cliques().iter().take(8).enumerate() {
        println!(
            "  #{index}: bonds={} atoms={} similarity={:.6} fragments={} largest_fragment_atoms={}",
            info.matched_edges().len(),
            info.vertex_matches().len(),
            info_johnson_similarity(case, info),
            info.fragment_count(),
            info.largest_fragment_atom_count(),
        );
    }

    println!("all_best_order:");
    for (index, info) in all_best.all_cliques().iter().take(8).enumerate() {
        println!(
            "  #{index}: bonds={} atoms={} similarity={:.6} fragments={} largest_fragment_atoms={}",
            info.matched_edges().len(),
            info.vertex_matches().len(),
            info_johnson_similarity(case, info),
            info.fragment_count(),
            info.largest_fragment_atom_count(),
        );
    }
}

#[test]
#[ignore = "focused diagnostic for the remaining 1K default mismatches 0702 and 0911"]
fn print_massspecgym_case_1000_remaining_partial_partition_side_probe() {
    let cases = load_massspecgym_ground_truth_1000();

    for case_name in ["massspecgym_default_0702", "massspecgym_default_0911"] {
        let case = find_case(&cases, case_name);
        let partial = run_labeled_case_with_search_mode(
            case,
            true,
            false,
            McesSearchMode::PartialEnumeration,
        );
        let partial_with_orientation =
            run_labeled_case_with_search_mode(case, true, true, McesSearchMode::PartialEnumeration);
        let all_best =
            run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
        let diagnostics = collect_fixture_order_product_diagnostics(case, true);
        let order = product_order_rdkit_raw_pair_order(case, &diagnostics);

        println!("case: {}", case.name);
        println!(
            "expected bonds={} atoms={} similarity={:.6}",
            case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
        );
        println!(
            "partial: retained={} top_bonds={} top_atoms={} top_similarity={:.6}",
            partial.all_cliques().len(),
            partial.matched_edges().len(),
            partial.vertex_matches().len(),
            partial.johnson_similarity(),
        );
        println!(
            "partial+orientation: retained={} top_bonds={} top_atoms={} top_similarity={:.6}",
            partial_with_orientation.all_cliques().len(),
            partial_with_orientation.matched_edges().len(),
            partial_with_orientation.vertex_matches().len(),
            partial_with_orientation.johnson_similarity(),
        );
        println!(
            "all_best: retained={} top_bonds={} top_atoms={} top_similarity={:.6}",
            all_best.all_cliques().len(),
            all_best.matched_edges().len(),
            all_best.vertex_matches().len(),
            all_best.johnson_similarity(),
        );

        let expected_in_all_best = all_best.all_cliques().iter().position(|info| {
            info.matched_edges().len() == case.expected_bond_matches
                && info.vertex_matches().len() == case.expected_atom_matches
                && (info_johnson_similarity(case, info) - case.expected_similarity).abs() <= 1e-6
        });
        println!("expected_index_in_all_best={expected_in_all_best:?}");

        for side in [
            geometric_traits::traits::algorithms::maximum_clique::PartitionSide::First,
            geometric_traits::traits::algorithms::maximum_clique::PartitionSide::Second,
        ] {
            let infos = permuted_partitioned_infos(
                case,
                &diagnostics,
                &order,
                side,
                McesSearchMode::PartialEnumeration,
            );
            let expected_index = infos.iter().position(|info| {
                info.matched_edges().len() == case.expected_bond_matches
                    && info.vertex_matches().len() == case.expected_atom_matches
                    && (info_johnson_similarity(case, info) - case.expected_similarity).abs()
                        <= 1e-6
            });
            println!(
                "  side={side:?} retained={} top_bonds={} top_atoms={} top_similarity={:.6} expected_index={expected_index:?}",
                infos.len(),
                infos.first().map_or(0, |info| info.matched_edges().len()),
                infos.first().map_or(0, |info| info.vertex_matches().len()),
                infos.first().map(|info| info_johnson_similarity(case, info)).unwrap_or(0.0),
            );
        }
    }
}

#[test]
#[ignore = "focused context-admission diagnostic for the remaining 1K default mismatch 0719"]
fn print_massspecgym_case_1000_0719_context_comparison() {
    let cases = load_massspecgym_ground_truth_1000();
    let case = find_case(&cases, "massspecgym_default_0719");
    let prepared = prepare_labeled_case(case);

    let diagnostics_without =
        collect_prepared_labeled_case_product_diagnostics(case, &prepared, false);
    let result_without =
        run_labeled_case_with_search_mode(case, false, false, McesSearchMode::AllBest);

    let diagnostics_with = collect_prepared_labeled_case_product_diagnostics(case, &prepared, true);
    let result_with = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!(
        "without contexts: product_vertices={} retained={} top_bonds={} top_atoms={} top_similarity={:.6}",
        diagnostics_without.vertex_pairs.len(),
        result_without.all_cliques().len(),
        result_without.matched_edges().len(),
        result_without.vertex_matches().len(),
        result_without.johnson_similarity(),
    );
    println!(
        "with contexts: product_vertices={} retained={} top_bonds={} top_atoms={} top_similarity={:.6}",
        diagnostics_with.vertex_pairs.len(),
        result_with.all_cliques().len(),
        result_with.matched_edges().len(),
        result_with.vertex_matches().len(),
        result_with.johnson_similarity(),
    );

    let removed_pairs: Vec<(usize, usize)> = diagnostics_without
        .vertex_pairs
        .iter()
        .copied()
        .filter(|pair| !diagnostics_with.vertex_pairs.contains(pair))
        .collect();
    println!("removed product vertices: {}", removed_pairs.len());
    for (i, j) in removed_pairs.iter().copied().take(20) {
        println!(
            "  ({i}, {j}) left_label={:?} right_label={:?} left_contexts={:?} right_contexts={:?}",
            diagnostics_without.first_bond_labels[i],
            diagnostics_without.second_bond_labels[j],
            case.graph1.aromatic_ring_contexts[i],
            case.graph2.aromatic_ring_contexts[j],
        );
    }
}

#[test]
#[ignore = "focused diagnostic proving the remaining default mismatches are RDKit raw-pair-order effects"]
fn test_massspecgym_ground_truth_labeled_mces_partial_enumeration_rdkit_raw_pair_order_cases() {
    let cases = load_massspecgym_ground_truth();
    for case_name in ["massspecgym_default_0038", "massspecgym_default_0092"] {
        let case = find_case(&cases, case_name);
        let diagnostics = collect_fixture_order_product_diagnostics(case, true);
        let order = product_order_rdkit_raw_pair_order(case, &diagnostics);
        let partition_side =
            geometric_traits::traits::algorithms::maximum_clique::choose_partition_side(
                &diagnostics.vertex_pairs,
                diagnostics.first_edge_map.len(),
                diagnostics.second_edge_map.len(),
            );
        let infos = permuted_partitioned_infos(
            case,
            &diagnostics,
            &order,
            partition_side,
            McesSearchMode::PartialEnumeration,
        );
        let top = infos.first().expect("expected at least one retained clique");
        assert_eq!(top.matched_edges().len(), case.expected_bond_matches, "{case_name}");
        assert_eq!(top.vertex_matches().len(), case.expected_atom_matches, "{case_name}");
        let similarity = info_johnson_similarity(case, top);
        assert!(
            (similarity - case.expected_similarity).abs() <= 1e-6,
            "{case_name}: expected similarity {:.6}, got {:.6}",
            case.expected_similarity,
            similarity
        );
    }
}

#[test]
#[ignore = "manual diagnostic harness for 1K MassSpecGym PartialEnumeration vs AllBest retention"]
fn print_massspecgym_case_1000_partial_vs_all_best() {
    let case_name =
        std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym 1K case name");
    let limit: usize =
        std::env::var("MCES_COMPARE_LIMIT").ok().and_then(|value| value.parse().ok()).unwrap_or(8);

    let cases = load_massspecgym_ground_truth_1000();
    let case = find_case(&cases, &case_name);
    let partial =
        run_labeled_case_with_search_mode(case, true, false, McesSearchMode::PartialEnumeration);
    let partial_with_orientation =
        run_labeled_case_with_search_mode(case, true, true, McesSearchMode::PartialEnumeration);
    let all_best = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!(
        "partial: retained={} top_bonds={} top_atoms={} top_similarity={:.6}",
        partial.all_cliques().len(),
        partial.matched_edges().len(),
        partial.vertex_matches().len(),
        partial.johnson_similarity(),
    );
    println!(
        "partial+orientation: retained={} top_bonds={} top_atoms={} top_similarity={:.6}",
        partial_with_orientation.all_cliques().len(),
        partial_with_orientation.matched_edges().len(),
        partial_with_orientation.vertex_matches().len(),
        partial_with_orientation.johnson_similarity(),
    );
    println!(
        "all_best: retained={} top_bonds={} top_atoms={} top_similarity={:.6}",
        all_best.all_cliques().len(),
        all_best.matched_edges().len(),
        all_best.vertex_matches().len(),
        all_best.johnson_similarity(),
    );

    println!("partial_order:");
    for (index, info) in partial.all_cliques().iter().take(limit).enumerate() {
        println!(
            "  #{index}: bonds={} atoms={} similarity={:.6} fragments={} largest_fragment_atoms={}",
            info.matched_edges().len(),
            info.vertex_matches().len(),
            info_johnson_similarity(case, info),
            info.fragment_count(),
            info.largest_fragment_atom_count(),
        );
        println!("    vertex_matches={:?}", info.vertex_matches());
    }

    println!("all_best_order:");
    for (index, info) in all_best.all_cliques().iter().take(limit).enumerate() {
        println!(
            "  #{index}: bonds={} atoms={} similarity={:.6} fragments={} largest_fragment_atoms={}",
            info.matched_edges().len(),
            info.vertex_matches().len(),
            info_johnson_similarity(case, info),
            info.fragment_count(),
            info.largest_fragment_atom_count(),
        );
        println!("    vertex_matches={:?}", info.vertex_matches());
    }
}
