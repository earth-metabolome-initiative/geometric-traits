use crate::support::*;

#[test]
#[ignore = "manual diagnostic harness for MassSpecGym atom-count mismatches"]
fn print_massspecgym_case_atom_mismatch_details() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let prepared = prepare_labeled_case(case);
    let diagnostics = collect_prepared_labeled_case_product_diagnostics(case, &prepared, true);
    let result = run_labeled_case(case);
    let selected_edge_indices = selected_clique_edge_indices(&result, &diagnostics);
    let inferred_atoms = inferred_atom_count_from_similarity(
        case,
        result.matched_edges().len(),
        result.johnson_similarity(),
    );

    println!("case: {}", case.name);
    println!("smiles1: {}", case.smiles1);
    println!("smiles2: {}", case.smiles2);
    println!(
        "expected: bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!(
        "rust: bonds={} atoms={} similarity={:.6}",
        result.matched_edges().len(),
        result.vertex_matches().len(),
        result.johnson_similarity(),
    );
    println!("inferred_atoms_from_similarity: {inferred_atoms}");
    println!(
        "atom_delta: {}",
        result.vertex_matches().len() as isize - case.expected_atom_matches as isize
    );
    println!(
        "selected clique members: {}",
        result.all_cliques().first().map(|info| info.clique().len()).unwrap_or(0)
    );
    println!("selected edge indices: {:?}", selected_edge_indices);
    println!("matched edge pairs: {:?}", result.matched_edges());
    println!("vertex matches: {:?}", result.vertex_matches());
}

#[test]
#[ignore = "manual diagnostic harness for MassSpecGym isolated-bond orientation"]
fn print_massspecgym_case_isolated_bond_analysis() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let prepared = prepare_labeled_case(case);
    let diagnostics = collect_prepared_labeled_case_product_diagnostics(case, &prepared, true);
    let result = run_labeled_case(case);
    let selected_edge_indices = selected_clique_edge_indices(&result, &diagnostics);
    let components = edge_components(&selected_edge_indices, &case.graph1.edges);

    println!("case: {}", case.name);
    println!(
        "expected atoms={} rust atoms={}",
        case.expected_atom_matches,
        result.vertex_matches().len()
    );
    println!(
        "component sizes on graph1 matched-edge subgraph: {:?}",
        components.iter().map(Vec::len).collect::<Vec<_>>()
    );

    for component in components.into_iter().filter(|component| component.len() == 1) {
        let (left_edge_index, right_edge_index) = component[0];
        let left_edge = case.graph1.edges[left_edge_index];
        let right_edge = case.graph2.edges[right_edge_index];
        let left_orientation = graph_edge_orientation(&case.graph1, left_edge_index);
        let right_orientation = graph_edge_orientation(&case.graph2, right_edge_index);
        let rdkit_mapping =
            rdkit_preferred_isolated_mapping(case, left_edge_index, right_edge_index);

        println!(
            "isolated bond pair ({left_edge_index}, {right_edge_index}) \
             left_edge={left_edge:?} right_edge={right_edge:?} \
             left_orientation={left_orientation:?} right_orientation={right_orientation:?}"
        );
        println!(
            "  left atom types: {:?} / {:?}, total_hs: {:?} / {:?}",
            case.graph1.atom_types[left_orientation[0]],
            case.graph1.atom_types[left_orientation[1]],
            graph_atom_total_hs(&case.graph1, left_orientation[0]),
            graph_atom_total_hs(&case.graph1, left_orientation[1]),
        );
        println!(
            "  right atom types: {:?} / {:?}, total_hs: {:?} / {:?}",
            case.graph2.atom_types[right_orientation[0]],
            case.graph2.atom_types[right_orientation[1]],
            graph_atom_total_hs(&case.graph2, right_orientation[0]),
            graph_atom_total_hs(&case.graph2, right_orientation[1]),
        );
        println!(
            "  rust default mapping: ({}, {}) ({}, {})",
            left_edge[0], right_edge[0], left_edge[1], right_edge[1]
        );
        println!(
            "  rdkit preferred mapping [{}]: {:?} {:?}",
            rdkit_mapping.2, rdkit_mapping.0, rdkit_mapping.1
        );
    }
}

#[test]
#[ignore = "manual diagnostic harness for MassSpecGym all-best clique summaries"]
fn print_massspecgym_case_all_best_atom_counts() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let limit: usize = std::env::var("MCES_ALL_BEST_LIMIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20);

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let result = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!("all_best_cliques={}", result.all_cliques().len());

    for (index, info) in result.all_cliques().iter().take(limit).enumerate() {
        println!(
            "#{index}: bonds={} atoms={} fragments={} largest_fragment={}",
            info.matched_edges().len(),
            info.vertex_matches().len(),
            info.fragment_count(),
            info.largest_fragment_size(),
        );
        println!("  matched_edges={:?}", info.matched_edges());
        println!("  vertex_matches={:?}", info.vertex_matches());
    }
}

#[test]
#[ignore = "manual diagnostic harness for MassSpecGym AllBest scorecards"]
fn print_massspecgym_case_all_best_scorecards() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let limit: usize = std::env::var("MCES_SCORECARD_LIMIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let result = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
    let first_ring_membership = graph_ring_membership_by_edge(&case.graph1);
    let second_ring_membership = graph_ring_membership_by_edge(&case.graph2);
    let first_distances = graph_distance_matrix(&case.graph1);
    let second_distances = graph_distance_matrix(&case.graph2);

    let scorecards: Vec<CliqueScorecard> = result
        .all_cliques()
        .iter()
        .map(|info| {
            score_clique(
                case,
                info,
                &first_ring_membership,
                &second_ring_membership,
                &first_distances,
                &second_distances,
            )
        })
        .collect();

    let mut approx_rdkit_order: Vec<usize> = (0..scorecards.len()).collect();
    approx_rdkit_order.sort_by(|&left, &right| {
        approx_rdkit_compare(&scorecards[left], &scorecards[right]).then_with(|| left.cmp(&right))
    });

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!("all_best_cliques={}", result.all_cliques().len());

    println!("current_rust_order:");
    for (index, (info, scorecard)) in
        result.all_cliques().iter().zip(scorecards.iter()).take(limit).enumerate()
    {
        println!(
            "#{index}: bonds={} atoms={} fragments={} largest_fragment={} ring_non_ring={} atom_h={} max_delta_dist={}",
            scorecard.matched_bonds,
            scorecard.matched_atoms,
            scorecard.fragment_count,
            scorecard.largest_fragment_size,
            scorecard.ring_non_ring_bond_score,
            scorecard.atom_h_score,
            scorecard.max_delta_atom_atom_dist,
        );
        println!("  matched_edges={:?}", info.matched_edges());
        println!("  vertex_matches={:?}", info.vertex_matches());
    }

    println!("approx_rdkit_order:");
    for (rank, &index) in approx_rdkit_order.iter().take(limit).enumerate() {
        let info = &result.all_cliques()[index];
        let scorecard = &scorecards[index];
        println!(
            "#{rank} [all_best_index={index}]: bonds={} atoms={} fragments={} largest_fragment={} ring_non_ring={} atom_h={} max_delta_dist={}",
            scorecard.matched_bonds,
            scorecard.matched_atoms,
            scorecard.fragment_count,
            scorecard.largest_fragment_size,
            scorecard.ring_non_ring_bond_score,
            scorecard.atom_h_score,
            scorecard.max_delta_atom_atom_dist,
        );
        println!("  matched_edges={:?}", info.matched_edges());
        println!("  vertex_matches={:?}", info.vertex_matches());
    }
}

#[test]
#[ignore = "manual diagnostic harness for MassSpecGym AllBest RDKit-style atom materialization"]
fn print_massspecgym_case_all_best_rdkit_materialization() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let limit: usize = std::env::var("MCES_RDKit_MATERIALIZATION_LIMIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let diagnostics = collect_labeled_case_product_diagnostics(case, true);
    let result = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
    let rdkit_matches: Vec<Vec<(usize, usize)>> = result
        .all_cliques()
        .iter()
        .map(|info| infer_vertex_matches_rdkit_style(case, info.clique(), &diagnostics))
        .collect();

    let first_rust_expected = result
        .all_cliques()
        .iter()
        .position(|info| info.vertex_matches().len() == case.expected_atom_matches);
    let first_rdkit_expected =
        rdkit_matches.iter().position(|matches| matches.len() == case.expected_atom_matches);

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!("all_best_cliques={}", result.all_cliques().len());
    println!("first_rust_expected_index={first_rust_expected:?}");
    println!("first_rdkit_expected_index={first_rdkit_expected:?}");

    for (index, info) in result.all_cliques().iter().take(limit).enumerate() {
        let rdkit_like = &rdkit_matches[index];
        let rust_only: Vec<(usize, usize)> = info
            .vertex_matches()
            .iter()
            .copied()
            .filter(|pair| !rdkit_like.contains(pair))
            .collect();
        let rdkit_only: Vec<(usize, usize)> = rdkit_like
            .iter()
            .copied()
            .filter(|pair| !info.vertex_matches().contains(pair))
            .collect();

        println!(
            "#{index}: bonds={} rust_atoms={} rdkit_atoms={} fragments={} largest_fragment_atoms={}",
            info.matched_edges().len(),
            info.vertex_matches().len(),
            rdkit_like.len(),
            info.fragment_count(),
            info.largest_fragment_atom_count(),
        );
        println!("  clique={:?}", info.clique());
        println!("  matched_edges={:?}", info.matched_edges());
        println!("  rust_vertex_matches={:?}", info.vertex_matches());
        println!("  rdkit_vertex_matches={:?}", rdkit_like);
        if !rust_only.is_empty() || !rdkit_only.is_empty() {
            println!("  rust_only={:?}", rust_only);
            println!("  rdkit_only={:?}", rdkit_only);
        }
    }
}

#[test]
#[ignore = "manual diagnostic harness for MassSpecGym edge-context admission comparison"]
fn print_massspecgym_case_context_comparison() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let corpus_size: usize =
        std::env::var("MCES_CORPUS_SIZE").ok().and_then(|value| value.parse().ok()).unwrap_or(100);
    let removed_limit: usize =
        std::env::var("MCES_REMOVED_LIMIT").ok().and_then(|value| value.parse().ok()).unwrap_or(20);

    let cases = load_massspecgym_ground_truth_by_size(corpus_size);
    let case = find_case(&cases, &case_name);
    let prepared = prepare_labeled_case(case);

    let started_without = Instant::now();
    let result_without =
        run_labeled_case_with_search_mode(case, false, false, McesSearchMode::AllBest);
    let elapsed_without = started_without.elapsed();
    let diagnostics_without =
        collect_prepared_labeled_case_product_diagnostics(case, &prepared, false);

    let started_with = Instant::now();
    let result_with = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
    let elapsed_with = started_with.elapsed();
    let diagnostics_with = collect_prepared_labeled_case_product_diagnostics(case, &prepared, true);

    println!("case: {}", case.name);
    println!("options: {:?}", case.options);
    println!(
        "graph1 non-empty aromatic context rows: {} / {}",
        non_empty_context_row_count(&case.graph1),
        case.graph1.aromatic_ring_contexts.len()
    );
    println!(
        "graph2 non-empty aromatic context rows: {} / {}",
        non_empty_context_row_count(&case.graph2),
        case.graph2.aromatic_ring_contexts.len()
    );
    println!(
        "without contexts: elapsed={elapsed_without:?} product_vertices={} all_cliques={} matched_edges={} atoms={} similarity={:.6}",
        diagnostics_without.vertex_pairs.len(),
        result_without.all_cliques().len(),
        result_without.matched_edges().len(),
        result_without.vertex_matches().len(),
        result_without.johnson_similarity(),
    );
    println!(
        "with contexts: elapsed={elapsed_with:?} product_vertices={} all_cliques={} matched_edges={} atoms={} similarity={:.6}",
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
    for (i, j) in removed_pairs.iter().copied().take(removed_limit) {
        println!(
            "  ({i}, {j}) left_label={:?} right_label={:?} left_contexts={:?} right_contexts={:?}",
            diagnostics_without.first_bond_labels[i],
            diagnostics_without.second_bond_labels[j],
            case.graph1.aromatic_ring_contexts[i],
            case.graph2.aromatic_ring_contexts[j],
        );
    }
    if removed_pairs.len() > removed_limit {
        println!("  ... {} more removed pairs", removed_pairs.len() - removed_limit);
    }
}

#[test]
#[ignore = "manual diagnostic harness for MassSpecGym PartialEnumeration vs AllBest retention"]
fn print_massspecgym_case_partial_vs_all_best() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let limit: usize =
        std::env::var("MCES_COMPARE_LIMIT").ok().and_then(|value| value.parse().ok()).unwrap_or(8);

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let partial =
        run_labeled_case_with_search_mode(case, true, false, McesSearchMode::PartialEnumeration);
    let partial_with_orientation =
        run_labeled_case_with_search_mode(case, true, true, McesSearchMode::PartialEnumeration);
    let all_best = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);

    let partial_expected = partial
        .all_cliques()
        .iter()
        .position(|info| info.vertex_matches().len() == case.expected_atom_matches);
    let partial_orientation_expected = partial_with_orientation
        .all_cliques()
        .iter()
        .position(|info| info.vertex_matches().len() == case.expected_atom_matches);
    let all_best_expected = all_best
        .all_cliques()
        .iter()
        .position(|info| info.vertex_matches().len() == case.expected_atom_matches);

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!(
        "partial: retained={} top_atoms={} top_similarity={:.6} expected_index={partial_expected:?}",
        partial.all_cliques().len(),
        partial.vertex_matches().len(),
        partial.johnson_similarity(),
    );
    println!(
        "partial+orientation: retained={} top_atoms={} top_similarity={:.6} expected_index={partial_orientation_expected:?}",
        partial_with_orientation.all_cliques().len(),
        partial_with_orientation.vertex_matches().len(),
        partial_with_orientation.johnson_similarity(),
    );
    println!(
        "all_best: retained={} top_atoms={} top_similarity={:.6} expected_index={all_best_expected:?}",
        all_best.all_cliques().len(),
        all_best.vertex_matches().len(),
        all_best.johnson_similarity(),
    );

    println!("partial_order:");
    for (index, info) in partial.all_cliques().iter().take(limit).enumerate() {
        println!(
            "  #{index}: bonds={} atoms={} fragments={} largest_fragment_atoms={}",
            info.matched_edges().len(),
            info.vertex_matches().len(),
            info.fragment_count(),
            info.largest_fragment_atom_count(),
        );
    }

    println!("all_best_order:");
    for (index, info) in all_best.all_cliques().iter().take(limit).enumerate() {
        println!(
            "  #{index}: bonds={} atoms={} fragments={} largest_fragment_atoms={}",
            info.matched_edges().len(),
            info.vertex_matches().len(),
            info.fragment_count(),
            info.largest_fragment_atom_count(),
        );
    }
}

#[test]
#[ignore = "manual diagnostic harness for modular-product vertex-order sensitivity"]
fn print_massspecgym_case_product_order_sensitivity() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let partition_side = match std::env::var("MCES_PARTITION_SIDE").ok().as_deref() {
        Some("second") | Some("Second") | Some("SECOND") => {
            geometric_traits::traits::algorithms::maximum_clique::PartitionSide::Second
        }
        _ => geometric_traits::traits::algorithms::maximum_clique::PartitionSide::First,
    };

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let diagnostics = collect_labeled_case_product_diagnostics(case, true);
    let identity_all_best = permuted_partitioned_infos(
        case,
        &diagnostics,
        &product_order_identity(&diagnostics.vertex_pairs),
        partition_side,
        McesSearchMode::AllBest,
    );
    let target_clique = identity_all_best
        .iter()
        .find(|info| info.vertex_matches().len() == case.expected_atom_matches)
        .map(|info| info.clique().to_vec())
        .unwrap_or_default();
    let strategies = [
        ("identity", product_order_identity(&diagnostics.vertex_pairs)),
        ("reverse", product_order_reverse(&diagnostics.vertex_pairs)),
        ("second_then_first", product_order_second_then_first(&diagnostics.vertex_pairs)),
        (
            "reverse_within_first_buckets",
            product_order_reverse_within_first_buckets(&diagnostics.vertex_pairs),
        ),
        ("fixture_edge_indices", product_order_fixture_edge_indices(case, &diagnostics)),
        (
            "fixture_edge_indices_second_then_first",
            product_order_fixture_edge_indices_second_then_first(case, &diagnostics),
        ),
        ("target_last", product_order_target_last(&diagnostics.vertex_pairs, &target_clique)),
    ];

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6} partition_side={partition_side:?}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!("product_vertices={}", diagnostics.vertex_pairs.len());

    for (name, order) in strategies {
        let partial = permuted_partitioned_infos(
            case,
            &diagnostics,
            &order,
            partition_side,
            McesSearchMode::PartialEnumeration,
        );
        let all_best = permuted_partitioned_infos(
            case,
            &diagnostics,
            &order,
            partition_side,
            McesSearchMode::AllBest,
        );
        let partial_expected = partial
            .iter()
            .position(|info| info.vertex_matches().len() == case.expected_atom_matches);
        let all_best_expected = all_best
            .iter()
            .position(|info| info.vertex_matches().len() == case.expected_atom_matches);
        let partial_top_similarity =
            partial.first().map(|info| info_johnson_similarity(case, info)).unwrap_or(0.0);
        let all_best_top_similarity =
            all_best.first().map(|info| info_johnson_similarity(case, info)).unwrap_or(0.0);

        println!(
            "{name}: partial_retained={} partial_top_atoms={} partial_top_similarity={:.6} partial_expected_index={partial_expected:?}",
            partial.len(),
            partial.first().map_or(0, |info| info.vertex_matches().len()),
            partial_top_similarity,
        );
        println!(
            "  {name}: all_best_retained={} all_best_top_atoms={} all_best_top_similarity={:.6} all_best_expected_index={all_best_expected:?}",
            all_best.len(),
            all_best.first().map_or(0, |info| info.vertex_matches().len()),
            all_best_top_similarity,
        );
    }
}

#[test]
#[ignore = "manual diagnostic harness for fixture-order line-graph and product construction"]
fn print_massspecgym_case_fixture_order_line_graph_product() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let corpus_size: usize =
        std::env::var("MCES_CORPUS_SIZE").ok().and_then(|value| value.parse().ok()).unwrap_or(100);
    let cases = load_massspecgym_ground_truth_by_size(corpus_size);
    let case = find_case(&cases, &case_name);

    let row_major = collect_labeled_case_product_diagnostics(case, true);
    let fixture_order = collect_fixture_order_product_diagnostics(case, true);
    let partition_side =
        geometric_traits::traits::algorithms::maximum_clique::choose_partition_side(
            &fixture_order.vertex_pairs,
            fixture_order.first_edge_map.len(),
            fixture_order.second_edge_map.len(),
        );

    let row_major_partial = permuted_partitioned_infos(
        case,
        &row_major,
        &product_order_identity(&row_major.vertex_pairs),
        partition_side,
        McesSearchMode::PartialEnumeration,
    );
    let row_major_all_best = permuted_partitioned_infos(
        case,
        &row_major,
        &product_order_identity(&row_major.vertex_pairs),
        partition_side,
        McesSearchMode::AllBest,
    );
    let fixture_partial = permuted_partitioned_infos(
        case,
        &fixture_order,
        &product_order_identity(&fixture_order.vertex_pairs),
        partition_side,
        McesSearchMode::PartialEnumeration,
    );
    let fixture_all_best = permuted_partitioned_infos(
        case,
        &fixture_order,
        &product_order_identity(&fixture_order.vertex_pairs),
        partition_side,
        McesSearchMode::AllBest,
    );

    let row_major_partial_expected = row_major_partial
        .iter()
        .position(|info| info.vertex_matches().len() == case.expected_atom_matches);
    let row_major_all_best_expected = row_major_all_best
        .iter()
        .position(|info| info.vertex_matches().len() == case.expected_atom_matches);
    let fixture_partial_expected = fixture_partial
        .iter()
        .position(|info| info.vertex_matches().len() == case.expected_atom_matches);
    let fixture_all_best_expected = fixture_all_best
        .iter()
        .position(|info| info.vertex_matches().len() == case.expected_atom_matches);

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!(
        "row_major: product_vertices={} partial_top_atoms={} partial_expected_index={row_major_partial_expected:?} all_best_top_atoms={} all_best_expected_index={row_major_all_best_expected:?}",
        row_major.vertex_pairs.len(),
        row_major_partial.first().map(|info| info.vertex_matches().len()).unwrap_or(0),
        row_major_all_best.first().map(|info| info.vertex_matches().len()).unwrap_or(0),
    );
    let first_edge_order_diff = row_major
        .first_edge_map
        .iter()
        .zip(fixture_order.first_edge_map.iter())
        .filter(|(left, right)| left != right)
        .count();
    let second_edge_order_diff = row_major
        .second_edge_map
        .iter()
        .zip(fixture_order.second_edge_map.iter())
        .filter(|(left, right)| left != right)
        .count();
    println!(
        "fixture_order: product_vertices={} partial_top_atoms={} partial_expected_index={fixture_partial_expected:?} all_best_top_atoms={} all_best_expected_index={fixture_all_best_expected:?} first_edge_order_diff={} second_edge_order_diff={}",
        fixture_order.vertex_pairs.len(),
        fixture_partial.first().map(|info| info.vertex_matches().len()).unwrap_or(0),
        fixture_all_best.first().map(|info| info.vertex_matches().len()).unwrap_or(0),
        first_edge_order_diff,
        second_edge_order_diff,
    );
}

#[test]
#[ignore = "manual diagnostic harness for fixture edge-order canonicalization"]
fn print_massspecgym_case_fixture_order_canonicalization() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let prepared_original = prepare_labeled_case(case);
    let diagnostics_original =
        collect_prepared_labeled_case_product_diagnostics(case, &prepared_original, true);

    let reversed_graph1 = reverse_graph_bond_payload(&case.graph1);
    let reversed_graph2 = reverse_graph_bond_payload(&case.graph2);
    let prepared_reversed =
        prepare_labeled_case_from_graph_data(case, &reversed_graph1, &reversed_graph2);
    let diagnostics_reversed =
        collect_prepared_labeled_case_product_diagnostics(case, &prepared_reversed, true);

    println!("case: {}", case.name);
    println!(
        "graph1_edge_map_equal={} graph2_edge_map_equal={} product_pairs_equal={}",
        diagnostics_original.first_edge_map == diagnostics_reversed.first_edge_map,
        diagnostics_original.second_edge_map == diagnostics_reversed.second_edge_map,
        diagnostics_original.vertex_pairs == diagnostics_reversed.vertex_pairs,
    );
    println!("product_matrix_equal={}", diagnostics_original.matrix == diagnostics_reversed.matrix,);
}

#[test]
#[ignore = "manual diagnostic harness for comparing retained and expected clique members"]
fn print_massspecgym_case_clique_member_differences() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let diagnostics = collect_labeled_case_product_diagnostics(case, true);
    let partial =
        run_labeled_case_with_search_mode(case, true, false, McesSearchMode::PartialEnumeration);
    let all_best = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
    let selected = partial.all_cliques().first().expect("partial result should retain a clique");
    let expected = all_best
        .all_cliques()
        .iter()
        .find(|info| info.vertex_matches().len() == case.expected_atom_matches)
        .expect("AllBest should contain the expected clique");

    let selected_only: Vec<usize> =
        selected.clique().iter().copied().filter(|v| !expected.clique().contains(v)).collect();
    let expected_only: Vec<usize> =
        expected.clique().iter().copied().filter(|v| !selected.clique().contains(v)).collect();

    println!("case: {}", case.name);
    println!("selected_only_vertices={selected_only:?}");
    for vertex in selected_only {
        let (left, right) = diagnostics.vertex_pairs[vertex];
        println!(
            "  selected_only vertex={vertex} pair=({left},{right}) g1_edge={:?} g2_edge={:?}",
            diagnostics.first_edge_map[left], diagnostics.second_edge_map[right],
        );
    }

    println!("expected_only_vertices={expected_only:?}");
    for vertex in expected_only {
        let (left, right) = diagnostics.vertex_pairs[vertex];
        println!(
            "  expected_only vertex={vertex} pair=({left},{right}) g1_edge={:?} g2_edge={:?}",
            diagnostics.first_edge_map[left], diagnostics.second_edge_map[right],
        );
    }
}

#[test]
#[ignore = "manual diagnostic harness for rerunning PartialEnumeration after rejecting the current winner"]
fn print_massspecgym_case_partial_after_rejecting_current_winner() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let diagnostics = collect_labeled_case_product_diagnostics(case, true);
    let partial =
        run_labeled_case_with_search_mode(case, true, false, McesSearchMode::PartialEnumeration);
    let rejected = partial
        .all_cliques()
        .first()
        .expect("partial result should retain a clique")
        .clique()
        .to_vec();
    let (g1_label_indices, g2_label_indices, num_labels) =
        intern_case_bond_labels(&diagnostics.first_bond_labels, &diagnostics.second_bond_labels);
    let partition = PartitionInfo {
        pairs: &diagnostics.vertex_pairs,
        g1_labels: &g1_label_indices,
        g2_labels: &g2_label_indices,
        num_labels,
        partition_side: geometric_traits::traits::algorithms::maximum_clique::PartitionSide::First,
    };
    let rerun = geometric_traits::traits::algorithms::maximum_clique::partial_search(
        &diagnostics.matrix,
        &partition,
        0,
        |clique| {
            clique != rejected
                && !clique_has_delta_y_from_product(
                    clique,
                    &diagnostics.vertex_pairs,
                    &diagnostics.first_edge_map,
                    &diagnostics.second_edge_map,
                    case.graph1.n_atoms,
                    case.graph2.n_atoms,
                )
        },
    );
    let reranked = rank_partitioned_cliques(
        rerun,
        &diagnostics.vertex_pairs,
        &diagnostics.first_edge_map,
        &diagnostics.second_edge_map,
    );

    println!("case: {}", case.name);
    println!("rejected_clique={rejected:?}");
    println!(
        "rerun_retained={} top_atoms={} top_similarity={:.6}",
        reranked.len(),
        reranked.first().map_or(0, |info| info.vertex_matches().len()),
        reranked.first().map(|info| info_johnson_similarity(case, info)).unwrap_or(0.0),
    );
    if let Some(info) = reranked.first() {
        println!("rerun_top_clique={:?}", info.clique());
        println!("rerun_top_vertex_matches={:?}", info.vertex_matches());
    }
}

#[test]
#[ignore = "manual diagnostic harness for rerunning PartialEnumeration after rejecting the retained set"]
fn print_massspecgym_case_partial_after_rejecting_retained_set() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let diagnostics = collect_labeled_case_product_diagnostics(case, true);
    let partial =
        run_labeled_case_with_search_mode(case, true, false, McesSearchMode::PartialEnumeration);
    let rejected: Vec<Vec<usize>> =
        partial.all_cliques().iter().map(|info| info.clique().to_vec()).collect();
    let (g1_label_indices, g2_label_indices, num_labels) =
        intern_case_bond_labels(&diagnostics.first_bond_labels, &diagnostics.second_bond_labels);
    let partition = PartitionInfo {
        pairs: &diagnostics.vertex_pairs,
        g1_labels: &g1_label_indices,
        g2_labels: &g2_label_indices,
        num_labels,
        partition_side: geometric_traits::traits::algorithms::maximum_clique::PartitionSide::First,
    };
    let rerun = geometric_traits::traits::algorithms::maximum_clique::partial_search(
        &diagnostics.matrix,
        &partition,
        0,
        |clique| {
            !rejected.contains(&clique.to_vec())
                && !clique_has_delta_y_from_product(
                    clique,
                    &diagnostics.vertex_pairs,
                    &diagnostics.first_edge_map,
                    &diagnostics.second_edge_map,
                    case.graph1.n_atoms,
                    case.graph2.n_atoms,
                )
        },
    );
    let reranked = rank_partitioned_cliques(
        rerun,
        &diagnostics.vertex_pairs,
        &diagnostics.first_edge_map,
        &diagnostics.second_edge_map,
    );

    println!("case: {}", case.name);
    println!("rejected_retained_count={}", rejected.len());
    println!(
        "rerun_retained={} top_atoms={} top_similarity={:.6}",
        reranked.len(),
        reranked.first().map_or(0, |info| info.vertex_matches().len()),
        reranked.first().map(|info| info_johnson_similarity(case, info)).unwrap_or(0.0),
    );
    if let Some(info) = reranked.first() {
        println!("rerun_top_clique={:?}", info.clique());
        println!("rerun_top_vertex_matches={:?}", info.vertex_matches());
    }
}

#[test]
#[ignore = "manual diagnostic harness for walking successive PartialEnumeration retained horizons"]
fn print_massspecgym_case_partial_horizon_walk() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let max_steps: usize =
        std::env::var("MCES_HORIZON_STEPS").ok().and_then(|value| value.parse().ok()).unwrap_or(8);

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let diagnostics = collect_labeled_case_product_diagnostics(case, true);
    let (g1_label_indices, g2_label_indices, num_labels) =
        intern_case_bond_labels(&diagnostics.first_bond_labels, &diagnostics.second_bond_labels);
    let partition = PartitionInfo {
        pairs: &diagnostics.vertex_pairs,
        g1_labels: &g1_label_indices,
        g2_labels: &g2_label_indices,
        num_labels,
        partition_side: geometric_traits::traits::algorithms::maximum_clique::PartitionSide::First,
    };
    let mut rejected: Vec<Vec<usize>> = Vec::new();

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );

    for step in 0..max_steps {
        let rerun = geometric_traits::traits::algorithms::maximum_clique::partial_search(
            &diagnostics.matrix,
            &partition,
            0,
            |clique| {
                !rejected.contains(&clique.to_vec())
                    && !clique_has_delta_y_from_product(
                        clique,
                        &diagnostics.vertex_pairs,
                        &diagnostics.first_edge_map,
                        &diagnostics.second_edge_map,
                        case.graph1.n_atoms,
                        case.graph2.n_atoms,
                    )
            },
        );
        let reranked = rank_partitioned_cliques(
            rerun,
            &diagnostics.vertex_pairs,
            &diagnostics.first_edge_map,
            &diagnostics.second_edge_map,
        );

        if reranked.is_empty() {
            println!("step={step} retained=0 rejected_total={}", rejected.len());
            break;
        }

        let expected_index = reranked.iter().position(|info| {
            info.matched_edges().len() == case.expected_bond_matches
                && info.vertex_matches().len() == case.expected_atom_matches
                && (info_johnson_similarity(case, info) - case.expected_similarity).abs() < 1e-6
        });
        let top = &reranked[0];
        let top_similarity = info_johnson_similarity(case, top);
        println!(
            "step={step} retained={} rejected_total={} top_bonds={} top_atoms={} top_similarity={:.6} expected_index={expected_index:?}",
            reranked.len(),
            rejected.len(),
            top.matched_edges().len(),
            top.vertex_matches().len(),
            top_similarity,
        );
        if let Some(index) = expected_index {
            println!("expected_found_at_step={step} retained_index={index}");
            println!("expected_clique={:?}", reranked[index].clique());
            break;
        }

        rejected.extend(reranked.into_iter().map(|info| info.clique().to_vec()));
    }
}

#[test]
#[ignore = "manual diagnostic harness for counting distinct result signatures in PartialEnumeration"]
fn print_massspecgym_case_partial_result_signature_counts() {
    use std::collections::BTreeMap;

    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let partial =
        run_labeled_case_with_search_mode(case, true, false, McesSearchMode::PartialEnumeration);
    let mut counts: BTreeMap<Vec<(usize, usize)>, (usize, usize, f64)> = BTreeMap::new();

    for info in partial.all_cliques() {
        let key = info.vertex_matches().to_vec();
        let entry = counts.entry(key.clone()).or_insert((
            0,
            info.matched_edges().len(),
            info_johnson_similarity(case, info),
        ));
        entry.0 += 1;
    }

    println!("case: {}", case.name);
    println!(
        "retained_cliques={} distinct_result_signatures={}",
        partial.all_cliques().len(),
        counts.len(),
    );
    for (index, (vertex_matches, (count, bonds, similarity))) in counts.iter().enumerate() {
        println!(
            "  signature#{index}: cliques={} bonds={} atoms={} similarity={:.6}",
            count,
            bonds,
            vertex_matches.len(),
            similarity,
        );
        if index >= 9 {
            break;
        }
    }
}

#[test]
#[ignore = "manual diagnostic harness for sweeping the initial lower bound in PartialEnumeration"]
fn print_massspecgym_case_partial_lower_bound_sweep() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let corpus_size: usize =
        std::env::var("MCES_CORPUS_SIZE").ok().and_then(|value| value.parse().ok()).unwrap_or(100);
    let max_bound: usize = std::env::var("MCES_MAX_LOWER_BOUND")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| {
            load_massspecgym_ground_truth_by_size(corpus_size)
                .iter()
                .find(|case| case.name == case_name)
                .map(|case| case.expected_bond_matches)
                .unwrap_or(0)
        });

    let cases = load_massspecgym_ground_truth_by_size(corpus_size);
    let case = find_case(&cases, &case_name);
    let diagnostics = collect_labeled_case_product_diagnostics(case, true);
    let (g1_label_indices, g2_label_indices, num_labels) =
        intern_case_bond_labels(&diagnostics.first_bond_labels, &diagnostics.second_bond_labels);
    let partition = PartitionInfo {
        pairs: &diagnostics.vertex_pairs,
        g1_labels: &g1_label_indices,
        g2_labels: &g2_label_indices,
        num_labels,
        partition_side: geometric_traits::traits::algorithms::maximum_clique::PartitionSide::First,
    };

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );

    for lower_bound in 0..=max_bound {
        let rerun = geometric_traits::traits::algorithms::maximum_clique::partial_search(
            &diagnostics.matrix,
            &partition,
            lower_bound,
            |clique| {
                !clique_has_delta_y_from_product(
                    clique,
                    &diagnostics.vertex_pairs,
                    &diagnostics.first_edge_map,
                    &diagnostics.second_edge_map,
                    case.graph1.n_atoms,
                    case.graph2.n_atoms,
                )
            },
        );
        let reranked = rank_partitioned_cliques(
            rerun,
            &diagnostics.vertex_pairs,
            &diagnostics.first_edge_map,
            &diagnostics.second_edge_map,
        );
        let expected_index = reranked.iter().position(|info| {
            info.matched_edges().len() == case.expected_bond_matches
                && info.vertex_matches().len() == case.expected_atom_matches
                && (info_johnson_similarity(case, info) - case.expected_similarity).abs() < 1e-6
        });
        println!(
            "lower_bound={lower_bound} retained={} top_bonds={} top_atoms={} top_similarity={:.6} expected_index={expected_index:?}",
            reranked.len(),
            reranked.first().map_or(0, |info| info.matched_edges().len()),
            reranked.first().map_or(0, |info| info.vertex_matches().len()),
            reranked.first().map(|info| info_johnson_similarity(case, info)).unwrap_or(0.0),
        );
    }
}

#[test]
#[ignore = "manual diagnostic harness for sweeping partition side in PartialEnumeration"]
fn print_massspecgym_case_partial_partition_side_sweep() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");
    let corpus_size: usize =
        std::env::var("MCES_CORPUS_SIZE").ok().and_then(|value| value.parse().ok()).unwrap_or(100);
    let lower_bound: usize = std::env::var("MCES_PARTITION_SIDE_LOWER_BOUND")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);

    let cases = load_massspecgym_ground_truth_by_size(corpus_size);
    let case = find_case(&cases, &case_name);
    let diagnostics = collect_labeled_case_product_diagnostics(case, true);
    let (g1_label_indices, g2_label_indices, num_labels) =
        intern_case_bond_labels(&diagnostics.first_bond_labels, &diagnostics.second_bond_labels);
    let heuristic_side =
        geometric_traits::traits::algorithms::maximum_clique::choose_partition_side(
            &diagnostics.vertex_pairs,
            diagnostics.first_edge_map.len(),
            diagnostics.second_edge_map.len(),
        );

    println!("case: {}", case.name);
    println!(
        "expected bonds={} atoms={} similarity={:.6} lower_bound={lower_bound} heuristic_side={heuristic_side:?}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );

    for side in [
        geometric_traits::traits::algorithms::maximum_clique::PartitionSide::First,
        geometric_traits::traits::algorithms::maximum_clique::PartitionSide::Second,
    ] {
        let partition = PartitionInfo {
            pairs: &diagnostics.vertex_pairs,
            g1_labels: &g1_label_indices,
            g2_labels: &g2_label_indices,
            num_labels,
            partition_side: side,
        };
        let rerun = geometric_traits::traits::algorithms::maximum_clique::partial_search(
            &diagnostics.matrix,
            &partition,
            lower_bound,
            |clique| {
                !clique_has_delta_y_from_product(
                    clique,
                    &diagnostics.vertex_pairs,
                    &diagnostics.first_edge_map,
                    &diagnostics.second_edge_map,
                    case.graph1.n_atoms,
                    case.graph2.n_atoms,
                )
            },
        );
        let reranked = rank_partitioned_cliques(
            rerun,
            &diagnostics.vertex_pairs,
            &diagnostics.first_edge_map,
            &diagnostics.second_edge_map,
        );
        let expected_index = reranked.iter().position(|info| {
            info.matched_edges().len() == case.expected_bond_matches
                && info.vertex_matches().len() == case.expected_atom_matches
                && (info_johnson_similarity(case, info) - case.expected_similarity).abs() < 1e-6
        });
        println!(
            "side={side:?} retained={} top_bonds={} top_atoms={} top_similarity={:.6} expected_index={expected_index:?}",
            reranked.len(),
            reranked.first().map_or(0, |info| info.matched_edges().len()),
            reranked.first().map_or(0, |info| info.vertex_matches().len()),
            reranked.first().map(|info| info_johnson_similarity(case, info)).unwrap_or(0.0),
        );
    }
}

#[test]
#[ignore = "manual diagnostic harness for rerunning a case with the graph order swapped"]
fn print_massspecgym_case_swapped_graph_order() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a MassSpecGym case name");

    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, &case_name);
    let original = run_labeled_case(case);
    let prepared_swapped = prepare_labeled_case_from_graph_data(case, &case.graph2, &case.graph1);
    let swapped = McesBuilder::new(&prepared_swapped.first, &prepared_swapped.second)
        .with_largest_fragment_metric(LargestFragmentMetric::Atoms)
        .compute_labeled();

    println!("case: {}", case.name);
    println!(
        "graph1 atoms={} bonds={} graph2 atoms={} bonds={}",
        case.graph1.n_atoms,
        case.graph1.edges.len(),
        case.graph2.n_atoms,
        case.graph2.edges.len(),
    );
    println!(
        "expected bonds={} atoms={} similarity={:.6}",
        case.expected_bond_matches, case.expected_atom_matches, case.expected_similarity
    );
    println!(
        "original bonds={} atoms={} similarity={:.6}",
        original.matched_edges().len(),
        original.vertex_matches().len(),
        original.johnson_similarity(),
    );
    println!(
        "swapped bonds={} atoms={} similarity={:.6}",
        swapped.matched_edges().len(),
        swapped.vertex_matches().len(),
        swapped.johnson_similarity(),
    );
}

#[test]
#[ignore = "timing harness for manual RDKit/Rust comparisons"]
fn print_labeled_case_timing() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a ground-truth case name");
    let repeats: usize =
        std::env::var("MCES_REPEATS").ok().and_then(|value| value.parse().ok()).unwrap_or(1);
    let use_edge_contexts = std::env::var("MCES_USE_EDGE_CONTEXTS")
        .ok()
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE"))
        .unwrap_or(true);
    let use_partition_orientation_heuristic =
        std::env::var("MCES_USE_PARTITION_ORIENTATION_HEURISTIC")
            .ok()
            .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE"))
            .unwrap_or(false);

    let cases = load_ground_truth();
    let case = find_case(&cases, &case_name);

    for repeat in 0..repeats {
        let started = Instant::now();
        let result = run_labeled_case_with_options(
            case,
            use_edge_contexts,
            use_partition_orientation_heuristic,
        );
        let elapsed = started.elapsed();
        println!(
            "{} repeat {}: edge_contexts={} partition_orientation_heuristic={} elapsed={elapsed:?} matched_edges={} similarity={:.6}",
            case.name,
            repeat + 1,
            use_edge_contexts,
            use_partition_orientation_heuristic,
            result.matched_edges().len(),
            result.johnson_similarity(),
        );
        std::io::stdout().flush().unwrap();
    }
}

#[test]
#[ignore = "timing harness for manual RDKit/Rust comparisons"]
fn print_all_labeled_case_timings() {
    let cases = load_ground_truth();
    let use_edge_contexts = std::env::var("MCES_USE_EDGE_CONTEXTS")
        .ok()
        .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE"))
        .unwrap_or(true);
    let use_partition_orientation_heuristic =
        std::env::var("MCES_USE_PARTITION_ORIENTATION_HEURISTIC")
            .ok()
            .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE"))
            .unwrap_or(false);

    for case in &cases {
        let started = Instant::now();
        let result = run_labeled_case_with_options(
            case,
            use_edge_contexts,
            use_partition_orientation_heuristic,
        );
        let elapsed = started.elapsed();
        println!(
            "{}: edge_contexts={} partition_orientation_heuristic={} elapsed={elapsed:?} matched_edges={} similarity={:.6}",
            case.name,
            use_edge_contexts,
            use_partition_orientation_heuristic,
            result.matched_edges().len(),
            result.johnson_similarity(),
        );
        std::io::stdout().flush().unwrap();
    }
}

#[test]
#[ignore = "manual diagnostic harness for edge-context wiring"]
fn print_labeled_case_context_comparison() {
    let case_name = std::env::var("MCES_CASE").expect("set MCES_CASE to a ground-truth case name");
    let removed_limit: usize =
        std::env::var("MCES_REMOVED_LIMIT").ok().and_then(|value| value.parse().ok()).unwrap_or(20);

    let cases = load_ground_truth();
    let case = find_case(&cases, &case_name);
    let prepared = prepare_labeled_case(case);

    let started_without = Instant::now();
    let result_without = run_labeled_case_with_contexts(case, false);
    let elapsed_without = started_without.elapsed();
    let diagnostics_without =
        collect_prepared_labeled_case_product_diagnostics(case, &prepared, false);

    let started_with = Instant::now();
    let result_with = run_labeled_case_with_contexts(case, true);
    let elapsed_with = started_with.elapsed();
    let diagnostics_with = collect_prepared_labeled_case_product_diagnostics(case, &prepared, true);

    println!("case: {}", case.name);
    println!("options: {:?}", case.options);
    println!(
        "without contexts: elapsed={elapsed_without:?} product_vertices={} matched_edges={} similarity={:.6}",
        diagnostics_without.vertex_pairs.len(),
        result_without.matched_edges().len(),
        result_without.johnson_similarity(),
    );
    println!(
        "with contexts: elapsed={elapsed_with:?} product_vertices={} matched_edges={} similarity={:.6}",
        diagnostics_with.vertex_pairs.len(),
        result_with.matched_edges().len(),
        result_with.johnson_similarity(),
    );

    println!("graph1 non-empty context rows:");
    for (edge_index, contexts) in case.graph1.aromatic_ring_contexts.iter().enumerate() {
        if !contexts.is_empty() {
            println!("  {edge_index}: {:?}", contexts);
        }
    }
    println!("graph2 non-empty context rows:");
    for (edge_index, contexts) in case.graph2.aromatic_ring_contexts.iter().enumerate() {
        if !contexts.is_empty() {
            println!("  {edge_index}: {:?}", contexts);
        }
    }

    let removed_pairs: Vec<(usize, usize)> = diagnostics_without
        .vertex_pairs
        .iter()
        .copied()
        .filter(|pair| !diagnostics_with.vertex_pairs.contains(pair))
        .collect();
    println!("removed product vertices: {}", removed_pairs.len());
    for (i, j) in removed_pairs.iter().copied().take(removed_limit) {
        println!(
            "  ({i}, {j}) left_label={:?} right_label={:?} left_contexts={:?} right_contexts={:?}",
            diagnostics_without.first_bond_labels[i],
            diagnostics_without.second_bond_labels[j],
            case.graph1.aromatic_ring_contexts[i],
            case.graph2.aromatic_ring_contexts[j],
        );
    }
    if removed_pairs.len() > removed_limit {
        println!("  ... {} more removed pairs", removed_pairs.len() - removed_limit);
    }
}
