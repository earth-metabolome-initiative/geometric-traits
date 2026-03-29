use super::support::*;

#[test]
fn test_ground_truth_labeled_mces() {
    let cases = load_ground_truth();
    for case in &cases {
        if case.timed_out {
            continue;
        }

        let result = run_labeled_case(case);
        assert_labeled_result_matches_ground_truth(case, &result, "labeled MCES");
    }
}

#[test]
fn test_ground_truth_labeled_mces_with_partition_orientation_heuristic() {
    let cases = load_ground_truth();
    for case in &cases {
        if case.timed_out {
            continue;
        }

        let result = run_labeled_case_with_options(case, true, true);
        assert_labeled_result_matches_ground_truth(
            case,
            &result,
            "labeled MCES with orientation heuristic",
        );
    }
}

#[test]
#[ignore = "large-corpus parity check against MassSpecGym-derived RDKit fixtures"]
fn test_massspecgym_ground_truth_labeled_mces() {
    let cases = load_massspecgym_ground_truth();
    assert_eq!(cases.len(), 100, "expected exactly 100 large-corpus cases");
    assert!(
        cases.iter().all(|case| !case.timed_out),
        "large-corpus fixture must exclude timed-out RDKit pairs",
    );

    let mismatch = first_parallel_mismatch(&cases, run_labeled_case);

    println!(
        "checked {} MassSpecGym default-config cases; mismatches={}",
        cases.len(),
        usize::from(mismatch.is_some())
    );

    assert!(
        mismatch.is_none(),
        "found mismatch in MassSpecGym default-config corpus:\n{}",
        mismatch.unwrap()
    );
}

#[test]
#[ignore = "1K-corpus parity check against fast 1-second MassSpecGym-derived RDKit default fixtures"]
fn test_massspecgym_ground_truth_labeled_mces_1000() {
    let cases = load_massspecgym_ground_truth_1000();
    assert_eq!(cases.len(), 1000, "expected exactly 1000 fast large-corpus cases");
    assert!(
        cases.iter().all(|case| !case.timed_out),
        "fast 1K default fixture must exclude timed-out RDKit pairs",
    );

    let mismatch = first_parallel_mismatch(&cases, run_labeled_case);

    println!(
        "checked {} MassSpecGym default-config 1K fast cases; mismatches={}",
        cases.len(),
        usize::from(mismatch.is_some())
    );

    assert!(
        mismatch.is_none(),
        "found mismatch in MassSpecGym default-config 1K fast corpus:\n{}",
        mismatch.unwrap()
    );
}

#[test]
#[ignore = "10K-corpus parity check against fast 1-second MassSpecGym-derived RDKit default fixtures"]
fn test_massspecgym_ground_truth_labeled_mces_10000() {
    let cases = load_massspecgym_ground_truth_10000();
    assert_eq!(cases.len(), 10000, "expected exactly 10000 fast large-corpus cases");
    assert!(
        cases.iter().all(|case| !case.timed_out),
        "fast 10K default fixture must exclude timed-out RDKit pairs",
    );

    let mismatch = first_parallel_mismatch(&cases, run_labeled_case);

    println!(
        "checked {} MassSpecGym default-config 10K fast cases; mismatches={}",
        cases.len(),
        usize::from(mismatch.is_some())
    );

    assert!(
        mismatch.is_none(),
        "found mismatch in MassSpecGym default-config 10K fast corpus:\n{}",
        mismatch.unwrap()
    );
}

#[test]
#[ignore = "50K-corpus parity check against fast 1-second MassSpecGym-derived RDKit default fixtures"]
fn test_massspecgym_ground_truth_labeled_mces_50000() {
    let cases = load_massspecgym_ground_truth_50000();
    assert_eq!(cases.len(), 50000, "expected exactly 50000 fast large-corpus cases");
    assert!(
        cases.iter().all(|case| !case.timed_out),
        "fast 50K default fixture must exclude timed-out RDKit pairs",
    );

    let mut mismatches: Vec<String> = cases
        .par_iter()
        .filter_map(|case| {
            let result = run_labeled_case(case);
            labeled_result_mismatch(case, &result)
        })
        .collect();
    mismatches.sort();

    println!(
        "checked {} MassSpecGym default-config 50K fast cases; mismatches={}",
        cases.len(),
        mismatches.len()
    );
    for mismatch in mismatches.iter().take(20) {
        println!("{mismatch}");
    }

    assert!(
        mismatches.is_empty(),
        "found {} mismatches in MassSpecGym default-config 50K fast corpus",
        mismatches.len()
    );
}

#[test]
#[ignore = "50K-corpus diagnostic parity check with partition orientation heuristic disabled"]
fn print_massspecgym_ground_truth_labeled_mces_50000_without_partition_orientation_heuristic() {
    let cases = load_massspecgym_ground_truth_50000();
    assert_eq!(cases.len(), 50000, "expected exactly 50000 fast large-corpus cases");
    assert!(
        cases.iter().all(|case| !case.timed_out),
        "fast 50K default fixture must exclude timed-out RDKit pairs",
    );

    let mut mismatches: Vec<String> = cases
        .par_iter()
        .filter_map(|case| {
            let result = run_labeled_case_with_options(case, true, false);
            labeled_result_mismatch(case, &result)
        })
        .collect();
    mismatches.sort();

    println!(
        "checked {} MassSpecGym default-config 50K fast cases without orientation heuristic; mismatches={}",
        cases.len(),
        mismatches.len()
    );
    for mismatch in mismatches.iter().take(20) {
        println!("{mismatch}");
    }
}

#[test]
fn test_massspecgym_ground_truth_labeled_mces_smoke() {
    let cases = load_massspecgym_ground_truth();
    let indices = evenly_spaced_case_indices(cases.len(), 25);
    let mismatch = indices
        .par_iter()
        .try_for_each(|&index| -> Result<(), String> {
            let case = &cases[index];
            let result = run_labeled_case(case);
            match labeled_result_mismatch(case, &result) {
                Some(mismatch) => Err(format!("[{index}] {mismatch}")),
                None => Ok(()),
            }
        })
        .err();

    assert!(
        mismatch.is_none(),
        "found mismatch in MassSpecGym default-config smoke sample:\n{}",
        mismatch.unwrap()
    );
}

#[test]
#[ignore = "large-corpus parity check against MassSpecGym-derived RDKit allBestMCESs=true fixtures"]
fn test_massspecgym_ground_truth_labeled_mces_all_best() {
    let cases = load_massspecgym_all_best_ground_truth();
    assert!(
        cases.iter().all(|case| !case.timed_out),
        "fast allBest fixture must exclude timed-out RDKit pairs",
    );
    assert!(
        !cases.is_empty(),
        "fast allBest fixture must retain at least one non-timeout RDKit pair",
    );

    let mismatch = first_parallel_mismatch(&cases, |case| {
        run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest)
    });

    println!(
        "checked {} MassSpecGym allBestMCESs=true cases in AllBest mode; mismatches={}",
        cases.len(),
        usize::from(mismatch.is_some())
    );

    assert!(
        mismatch.is_none(),
        "found mismatch in MassSpecGym allBestMCESs=true corpus with AllBest:\n{}",
        mismatch.unwrap()
    );
}

#[test]
#[ignore = "1K-corpus parity check against fast 1-second MassSpecGym-derived RDKit allBestMCESs=true fixtures"]
fn test_massspecgym_ground_truth_labeled_mces_all_best_1000() {
    let cases = load_massspecgym_all_best_ground_truth_1000();
    assert!(
        !cases.is_empty(),
        "fast 1K allBest fixture must retain at least one non-timeout RDKit pair",
    );
    assert!(
        cases.iter().all(|case| !case.timed_out),
        "fast 1K allBest fixture must exclude timed-out RDKit pairs",
    );

    let mismatch = first_parallel_mismatch(&cases, |case| {
        run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest)
    });

    println!(
        "checked {} MassSpecGym allBestMCESs=true 1K fast cases in AllBest mode; mismatches={}",
        cases.len(),
        usize::from(mismatch.is_some())
    );

    assert!(
        mismatch.is_none(),
        "found mismatch in MassSpecGym allBestMCESs=true 1K fast corpus with AllBest:\n{}",
        mismatch.unwrap()
    );
}

#[test]
#[ignore = "10K-corpus parity check against fast 1-second MassSpecGym-derived RDKit allBestMCESs=true fixtures"]
fn test_massspecgym_ground_truth_labeled_mces_all_best_10000() {
    let cases = load_massspecgym_all_best_ground_truth_10000();
    assert!(
        !cases.is_empty(),
        "fast 10K allBest fixture must retain at least one non-timeout RDKit pair",
    );
    assert!(
        cases.iter().all(|case| !case.timed_out),
        "fast 10K allBest fixture must exclude timed-out RDKit pairs",
    );

    let mismatch = first_parallel_mismatch(&cases, |case| {
        run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest)
    });

    println!(
        "checked {} MassSpecGym allBestMCESs=true 10K fast cases in AllBest mode; mismatches={}",
        cases.len(),
        usize::from(mismatch.is_some())
    );

    assert!(
        mismatch.is_none(),
        "found mismatch in MassSpecGym allBestMCESs=true 10K fast corpus with AllBest:\n{}",
        mismatch.unwrap()
    );
}

#[test]
fn test_massspecgym_ground_truth_labeled_mces_all_best_smoke() {
    let cases = load_massspecgym_all_best_ground_truth();
    assert!(
        cases.iter().all(|case| !case.timed_out),
        "fast allBest fixture must exclude timed-out RDKit pairs",
    );
    let indices = evenly_spaced_case_indices(cases.len(), 25);
    let mismatch = indices
        .par_iter()
        .try_for_each(|&index| -> Result<(), String> {
            let case = &cases[index];
            let result =
                run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
            match labeled_result_mismatch(case, &result) {
                Some(mismatch) => Err(format!("[{}:{}] {}", index, case.name, mismatch)),
                None => Ok(()),
            }
        })
        .err();

    assert!(
        mismatch.is_none(),
        "found mismatch in MassSpecGym allBestMCESs=true smoke sample:\n{}",
        mismatch.unwrap()
    );
}

#[test]
#[ignore = "focused probe showing where AllBest differs from the RDKit default-path fixture"]
fn test_massspecgym_ground_truth_labeled_mces_all_best_known_differences_from_default_fixture() {
    let mismatch_cases = [
        "massspecgym_default_0006",
        "massspecgym_default_0010",
        "massspecgym_default_0018",
        "massspecgym_default_0029",
        "massspecgym_default_0038",
        "massspecgym_default_0054",
        "massspecgym_default_0086",
        "massspecgym_default_0092",
    ];

    let cases = load_massspecgym_ground_truth();
    let mut mismatches = Vec::new();

    for case_name in mismatch_cases {
        let case = find_case(&cases, case_name);
        let result = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
        println!(
            "{}: bonds={} expected_bonds={} similarity={:.6} expected_similarity={:.6}",
            case.name,
            result.matched_edges().len(),
            case.expected_bond_matches,
            result.johnson_similarity(),
            case.expected_similarity,
        );
        if let Some(mismatch) = labeled_result_mismatch(case, &result) {
            mismatches.push(mismatch);
        }
    }

    println!(
        "checked {} known default-fixture difference cases in AllBest mode; mismatches={}",
        8,
        mismatches.len()
    );

    assert!(
        mismatches.is_empty(),
        "found {} differences from the RDKit default-path fixture among known AllBest cases:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

#[test]
#[ignore = "focused parity check against RDKit allBestMCESs=true for the known holdouts"]
fn test_massspecgym_ground_truth_labeled_mces_all_best_rdkit_holdouts() {
    let cases = load_massspecgym_all_best_holdouts();
    assert_eq!(cases.len(), 3, "expected exactly 3 focused allBest holdout cases");

    for case in &cases {
        let result = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
        assert_labeled_result_matches_ground_truth(case, &result, "AllBest vs RDKit allBestMCESs");
    }
}

#[test]
#[ignore = "focused PartialEnumeration regression for a known MassSpecGym tie-retention case"]
fn test_massspecgym_ground_truth_labeled_mces_partial_enumeration_case_0006() {
    let cases = load_massspecgym_ground_truth();
    let case = find_case(&cases, "massspecgym_default_0006");
    let result =
        run_labeled_case_with_default_orientation(case, true, McesSearchMode::PartialEnumeration);
    assert_labeled_result_matches_ground_truth(case, &result, "PartialEnumeration");
}

#[test]
#[ignore = "focused PartialEnumeration regressions for smaller-first partition-side cases"]
fn test_massspecgym_ground_truth_labeled_mces_partial_enumeration_smaller_first_cases() {
    let cases = load_massspecgym_ground_truth();
    for case_name in
        ["massspecgym_default_0018", "massspecgym_default_0029", "massspecgym_default_0054"]
    {
        let case = find_case(&cases, case_name);
        let result = run_labeled_case(case);
        assert_labeled_result_matches_ground_truth(case, &result, "PartialEnumeration");
    }
}

#[test]
#[ignore = "focused red regression for a remaining 1K default mismatch"]
fn test_massspecgym_ground_truth_labeled_mces_1000_case_0719() {
    let cases = load_massspecgym_ground_truth_1000();
    let case = find_case(&cases, "massspecgym_default_0719");
    let result = run_labeled_case(case);
    assert_labeled_result_matches_ground_truth(case, &result, "PartialEnumeration");
}

#[test]
#[ignore = "focused red regression for a remaining 1K default mismatch"]
fn test_massspecgym_ground_truth_labeled_mces_1000_case_0585() {
    let cases = load_massspecgym_ground_truth_1000();
    let case = find_case(&cases, "massspecgym_default_0585");
    let result = run_labeled_case(case);
    assert_labeled_result_matches_ground_truth(case, &result, "PartialEnumeration");
}

#[test]
#[ignore = "focused red regression for a remaining 1K default mismatch"]
fn test_massspecgym_ground_truth_labeled_mces_1000_case_0702() {
    let cases = load_massspecgym_ground_truth_1000();
    let case = find_case(&cases, "massspecgym_default_0702");
    let result = run_labeled_case(case);
    assert_labeled_result_matches_ground_truth(case, &result, "PartialEnumeration");
}

#[test]
#[ignore = "focused red regression for a remaining 1K default mismatch"]
fn test_massspecgym_ground_truth_labeled_mces_1000_case_0911() {
    let cases = load_massspecgym_ground_truth_1000();
    let case = find_case(&cases, "massspecgym_default_0911");
    let result = run_labeled_case(case);
    assert_labeled_result_matches_ground_truth(case, &result, "PartialEnumeration");
}

#[test]
#[ignore = "focused red regression for a remaining 10K allBest mismatch"]
fn test_massspecgym_ground_truth_labeled_mces_all_best_10000_case_5585() {
    let cases = load_massspecgym_all_best_ground_truth_10000();
    let case = find_case(&cases, "massspecgym_default_5585");
    let result = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
    assert_labeled_result_matches_ground_truth(case, &result, "AllBest");
}

#[test]
#[ignore = "focused red regression for a remaining 50K default mismatch"]
fn test_massspecgym_ground_truth_labeled_mces_50000_case_16725() {
    let cases = load_massspecgym_ground_truth_50000();
    let case = find_case(&cases, "massspecgym_default_16725");
    let result = run_labeled_case(case);
    assert_labeled_result_matches_ground_truth(case, &result, "PartialEnumeration");
}

#[test]
#[ignore = "focused red regression for a remaining 50K default mismatch"]
fn test_massspecgym_ground_truth_labeled_mces_50000_case_19932() {
    let cases = load_massspecgym_ground_truth_50000();
    let case = find_case(&cases, "massspecgym_default_19932");
    let result = run_labeled_case(case);
    assert_labeled_result_matches_ground_truth(case, &result, "PartialEnumeration");
}

#[test]
#[ignore = "focused red regression for a remaining 50K default mismatch"]
fn test_massspecgym_ground_truth_labeled_mces_50000_case_23080() {
    let cases = load_massspecgym_ground_truth_50000();
    let case = find_case(&cases, "massspecgym_default_23080");
    let result = run_labeled_case(case);
    assert_labeled_result_matches_ground_truth(case, &result, "PartialEnumeration");
}

#[test]
#[ignore = "focused proof that the first missing RDKit ranker flips the 5585 allBest top clique"]
fn test_massspecgym_ground_truth_labeled_mces_all_best_10000_case_5585_first_missing_ranker() {
    let cases = load_massspecgym_all_best_ground_truth_10000();
    let case = find_case(&cases, "massspecgym_default_5585");
    let result = run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
    let expected_index = result
        .all_cliques()
        .iter()
        .position(|info| labeled_info_mismatch(case, info).is_none())
        .expect("expected RDKit allBest clique must be present in the Rust AllBest set");
    let reordered = reordered_all_best_indices_by_first_missing_ranker(case, &result);

    assert_ne!(
        expected_index, 0,
        "raw AllBest should still disagree on 5585 before the test-only ranker is applied"
    );
    assert_eq!(
        reordered.first().copied(),
        Some(expected_index),
        "the first missing RDKit ranker should move the RDKit allBest clique to the top"
    );
}

#[test]
#[ignore = "exploratory parity check for the first missing RDKit allBest ranker over the fast 10K corpus"]
fn test_massspecgym_ground_truth_labeled_mces_all_best_10000_first_missing_ranker() {
    let cases = load_massspecgym_all_best_ground_truth_10000();
    assert!(
        !cases.is_empty(),
        "fast 10K allBest fixture must retain at least one non-timeout RDKit pair",
    );
    assert!(
        cases.iter().all(|case| !case.timed_out),
        "fast 10K allBest fixture must exclude timed-out RDKit pairs",
    );

    let mismatch = cases
        .par_iter()
        .try_for_each(|case| -> Result<(), String> {
            let result =
                run_labeled_case_with_search_mode(case, true, false, McesSearchMode::AllBest);
            let reordered = reordered_all_best_indices_by_first_missing_ranker(case, &result);
            let top = reordered
                .first()
                .map(|&index| &result.all_cliques()[index])
                .expect("AllBest should retain at least one clique");
            match labeled_info_mismatch(case, top) {
                Some(mismatch) => Err(mismatch),
                None => Ok(()),
            }
        })
        .err();

    println!(
        "checked {} MassSpecGym allBestMCESs=true 10K fast cases with the first missing RDKit ranker; mismatches={}",
        cases.len(),
        usize::from(mismatch.is_some())
    );

    assert!(
        mismatch.is_none(),
        "found mismatch in MassSpecGym allBestMCESs=true 10K fast corpus with the first missing RDKit ranker:\n{}",
        mismatch.unwrap()
    );
}
