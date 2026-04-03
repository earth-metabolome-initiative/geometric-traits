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
#[ignore = "100-case parity check against MassSpecGym-derived RDKit fixtures"]
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
#[ignore = "10K-corpus parity check against the committed MassSpecGym-derived RDKit default fixture"]
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
#[ignore = "200K-corpus parity check against the local-only MassSpecGym-derived RDKit default fixture"]
fn test_massspecgym_ground_truth_labeled_mces_200000() {
    let cases = load_massspecgym_ground_truth_200000();
    assert_eq!(cases.len(), 200000, "expected exactly 200000 fast large-corpus cases");
    assert!(
        cases.iter().all(|case| !case.timed_out),
        "fast 200K default fixture must exclude timed-out RDKit pairs",
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
        "checked {} MassSpecGym default-config 200K fast cases; mismatches={}",
        cases.len(),
        mismatches.len()
    );
    for mismatch in mismatches.iter().take(20) {
        println!("{mismatch}");
    }

    assert!(
        mismatches.is_empty(),
        "found {} mismatches in MassSpecGym default-config 200K fast corpus",
        mismatches.len()
    );
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
#[ignore = "100-case parity check against MassSpecGym-derived RDKit allBestMCESs=true fixtures"]
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
        run_labeled_case_with_search_mode(case, true, McesSearchMode::AllBest)
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
            let result = run_labeled_case_with_search_mode(case, true, McesSearchMode::AllBest);
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
