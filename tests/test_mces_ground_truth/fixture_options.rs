use super::support::*;

#[test]
fn test_ground_truth_fixture_metadata() {
    let cases = load_ground_truth();
    assert!(cases.len() >= 10, "expected at least 10 test cases");

    let very_small = cases.iter().find(|c| c.name == "very_small");
    assert!(very_small.is_some(), "missing 'very_small' case");
    let vs = very_small.unwrap();
    assert_eq!(vs.graph1.n_atoms, 5);
    assert_eq!(vs.graph2.n_atoms, 5);
    assert_eq!(vs.expected_bond_matches, 3);
    assert!((vs.expected_similarity - 0.6049).abs() < 0.001);

    let symmetrical_esters = find_case(&cases, "symmetrical_esters");
    assert_eq!(
        symmetrical_esters.graph1.aromatic_ring_contexts.len(),
        symmetrical_esters.graph1.edges.len()
    );
    assert!(
        symmetrical_esters
            .graph1
            .aromatic_ring_contexts
            .iter()
            .any(|contexts| !contexts.is_empty()),
        "expected aromatic-ring contexts for symmetrical_esters"
    );
}

#[test]
fn test_ground_truth_similarity_threshold_option() {
    let cases = load_ground_truth();
    let case = find_case(&cases, "very_small_similarity_threshold_reject");
    let result = run_labeled_case(case);

    assert_eq!(case_similarity_threshold(case), Some(0.99));
    assert_labeled_result_matches_ground_truth(case, &result, "similarity threshold");
}

#[test]
fn test_ground_truth_ignore_bond_orders_option() {
    let cases = load_ground_truth();
    let case = find_case(&cases, "ignore_bond_orders");
    let result = run_labeled_case(case);

    assert_labeled_result_matches_ground_truth(case, &result, "ignore bond orders");
}

#[test]
fn test_ground_truth_ring_matches_ring_only_option() {
    let cases = load_ground_truth();
    for case_name in ["ring_matches_ring", "single_fragment_1", "single_fragment_2"] {
        let case = find_case(&cases, case_name);
        let result = run_labeled_case(case);

        assert_labeled_result_matches_ground_truth(case, &result, "ring matches ring only");
    }
}

#[test]
fn test_ground_truth_exact_connections_option() {
    let cases = load_ground_truth();
    let case = find_case(&cases, "exact_connections_1");
    let result = run_labeled_case(case);

    assert_labeled_result_matches_ground_truth(case, &result, "exact connections match");
}

#[test]
fn test_ground_truth_atom_aromaticity_respect_option() {
    let cases = load_ground_truth();
    let case = find_case(&cases, "atom_aromaticity_respect");
    let result = run_labeled_case(case);

    assert_labeled_result_matches_ground_truth(case, &result, "atom aromaticity respect");
}

#[test]
fn test_ground_truth_bad_aromatics_contexts_change_pair_admission() {
    let cases = load_ground_truth();
    let case = find_case(&cases, "bad_aromatics_1a");

    let without_contexts = collect_labeled_case_product_diagnostics(case, false);
    let with_contexts = collect_labeled_case_product_diagnostics(case, true);

    assert!(with_contexts.vertex_pairs.len() < without_contexts.vertex_pairs.len());
}

#[test]
fn test_ground_truth_exact_connections_refines_node_labels() {
    let cases = load_ground_truth();
    let case = find_case(&cases, "exact_connections_1");
    let prepared = prepare_labeled_case(case);

    let first_labels: Vec<GroundTruthNodeLabel> =
        prepared.first.nodes().map(|node| node.node_type()).collect();
    let second_labels: Vec<GroundTruthNodeLabel> =
        prepared.second.nodes().map(|node| node.node_type()).collect();

    let find_refined_pair = |labels: &[GroundTruthNodeLabel]| {
        labels.iter().enumerate().find_map(|(left_index, left_label)| {
            labels.iter().enumerate().skip(left_index + 1).find_map(|(right_index, right_label)| {
                if left_label.atom_type == right_label.atom_type
                    && left_label.explicit_degree != right_label.explicit_degree
                {
                    Some((left_index, right_index))
                } else {
                    None
                }
            })
        })
    };

    let (first_left, first_right) = find_refined_pair(&first_labels)
        .expect("expected a same-type/different-degree witness in graph1");
    let (second_left, second_right) = find_refined_pair(&second_labels)
        .expect("expected a same-type/different-degree witness in graph2");

    assert_eq!(first_labels[first_left].atom_type, first_labels[first_right].atom_type);
    assert_ne!(first_labels[first_left].explicit_degree, first_labels[first_right].explicit_degree);
    assert_ne!(first_labels[first_left], first_labels[first_right]);

    assert_eq!(second_labels[second_left].atom_type, second_labels[second_right].atom_type);
    assert_ne!(
        second_labels[second_left].explicit_degree,
        second_labels[second_right].explicit_degree
    );
    assert_ne!(second_labels[second_left], second_labels[second_right]);
}

#[test]
fn test_ground_truth_atom_aromaticity_option_refines_node_labels() {
    let cases = load_ground_truth();
    let ignore_case = find_case(&cases, "atom_aromaticity_ignore");
    let respect_case = find_case(&cases, "atom_aromaticity_respect");
    let prepared_ignore = prepare_labeled_case(ignore_case);
    let prepared_respect = prepare_labeled_case(respect_case);

    let ignore_labels: Vec<GroundTruthNodeLabel> =
        prepared_ignore.first.nodes().map(|node| node.node_type()).collect();
    let respect_labels: Vec<GroundTruthNodeLabel> =
        prepared_respect.first.nodes().map(|node| node.node_type()).collect();

    let witness = ignore_case
        .graph1
        .atom_types
        .iter()
        .enumerate()
        .find_map(|(left_index, left_type)| {
            ignore_case.graph1.atom_types.iter().enumerate().skip(left_index + 1).find_map(
                |(right_index, right_type)| {
                    if left_type == right_type
                        && ignore_case.graph1.atom_is_aromatic[left_index]
                            != ignore_case.graph1.atom_is_aromatic[right_index]
                    {
                        Some((left_index, right_index))
                    } else {
                        None
                    }
                },
            )
        })
        .expect("expected same-type atoms with different aromaticity in atom_aromaticity fixture");

    let (left_index, right_index) = witness;

    assert_eq!(ignore_labels[left_index], ignore_labels[right_index]);
    assert_eq!(ignore_labels[left_index].is_aromatic, None);

    assert_ne!(respect_labels[left_index], respect_labels[right_index]);
    assert_eq!(
        respect_labels[left_index].is_aromatic,
        Some(ignore_case.graph1.atom_is_aromatic[left_index])
    );
    assert_eq!(
        respect_labels[right_index].is_aromatic,
        Some(ignore_case.graph1.atom_is_aromatic[right_index])
    );
}

#[test]
fn test_ground_truth_bad_aromatics_uses_shared_atom_labels() {
    let cases = load_ground_truth();
    let case = find_case(&cases, "bad_aromatics_1a");
    let prepared = prepare_labeled_case(case);
    let diagnostics = collect_prepared_labeled_case_product_diagnostics(case, &prepared, true);
    let result = run_labeled_case(case);

    let first_lg = prepared.first.labeled_line_graph();
    let second_lg = prepared.second.labeled_line_graph();
    let carbonyl_left =
        first_lg.edge_map().iter().position(|&(src, dst)| (src, dst) == (6, 7)).unwrap();
    let carbonyl_right =
        second_lg.edge_map().iter().position(|&(src, dst)| (src, dst) == (6, 7)).unwrap();

    assert_eq!(
        diagnostics.first_bond_labels[carbonyl_left],
        diagnostics.second_bond_labels[carbonyl_right]
    );
    assert!(diagnostics.vertex_pairs.contains(&(carbonyl_left, carbonyl_right)));
    assert_labeled_result_matches_ground_truth(case, &result, "bad aromatics shared labels");
}

#[test]
fn test_ground_truth_ring_matches_ring_only_blocks_mixed_ring_pairs() {
    let cases = load_ground_truth();
    let case = find_case(&cases, "ring_matches_ring");
    let diagnostics = collect_labeled_case_product_diagnostics(case, true);

    let mismatched_pair = diagnostics
        .first_bond_labels
        .iter()
        .enumerate()
        .find_map(|(i, first_label)| {
            diagnostics.second_bond_labels.iter().enumerate().find_map(|(j, second_label)| {
                let same_primary = first_label.0 == second_label.0
                    && first_label.1.bond_order == second_label.1.bond_order
                    && first_label.2 == second_label.2;
                let different_ring_class = first_label.1.in_ring != second_label.1.in_ring;
                if same_primary && different_ring_class { Some((i, j)) } else { None }
            })
        })
        .expect("expected at least one ring-vs-chain bond pair with matching primary label");

    assert!(
        !diagnostics.vertex_pairs.contains(&mismatched_pair),
        "ringMatchesRingOnly should reject mixed ring/chain bond pairs",
    );
}

#[test]
fn test_ground_truth_unlabeled_identical_topology() {
    let cases = load_ground_truth();
    let ibo = cases.iter().find(|c| c.name == "ignore_bond_orders");
    assert!(ibo.is_some());
    let case = ibo.unwrap();

    let g1 = build_unlabeled_graph(case.graph1.n_atoms, &case.graph1.edges);
    let g2 = build_unlabeled_graph(case.graph2.n_atoms, &case.graph2.edges);

    let result = McesBuilder::new(&g1, &g2).compute_unlabeled();
    assert_eq!(result.matched_edges().len(), 3);
    let j = result.johnson_similarity();
    assert!((j - 1.0).abs() < 1e-6, "identical topology should give similarity 1.0, got {j}");
}

#[test]
fn test_ground_truth_delta_y_cases() {
    let cases = load_ground_truth();
    let dy = cases.iter().find(|c| c.name == "delta_y_small").unwrap();

    let g1 = build_unlabeled_graph(dy.graph1.n_atoms, &dy.graph1.edges);
    let g2 = build_unlabeled_graph(dy.graph2.n_atoms, &dy.graph2.edges);

    let without_dy = McesBuilder::new(&g1, &g2).with_delta_y(false).compute_unlabeled();
    let with_dy = McesBuilder::new(&g1, &g2).with_delta_y(true).compute_unlabeled();

    assert!(with_dy.matched_edges().len() <= without_dy.matched_edges().len());
}
