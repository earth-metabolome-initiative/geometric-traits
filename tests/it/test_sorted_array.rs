//! Test submodule for SortedError (integration-visible error type from sorted
//! data structures).
#![cfg(feature = "std")]

use geometric_traits::errors::SortedError;

#[test]
fn test_sorted_error_display_contains_value() {
    let err: SortedError<u32> = SortedError::UnsortedEntry(42u32);
    let msg = format!("{err}");
    assert!(msg.contains("42"), "Display message should contain the offending value, got: {msg}");
}

#[test]
fn test_sorted_error_debug_contains_value() {
    let err: SortedError<u32> = SortedError::UnsortedEntry(99u32);
    let msg = format!("{err:?}");
    assert!(msg.contains("99"), "Debug message should contain the offending value, got: {msg}");
}

#[test]
fn test_sorted_error_clone() {
    let err: SortedError<u32> = SortedError::UnsortedEntry(10u32);
    let cloned = err.clone();
    assert_eq!(err, cloned);
}

#[test]
fn test_sorted_error_eq() {
    let err1: SortedError<u32> = SortedError::UnsortedEntry(10u32);
    let err2: SortedError<u32> = SortedError::UnsortedEntry(10u32);
    let err3: SortedError<u32> = SortedError::UnsortedEntry(20u32);
    assert_eq!(err1, err2);
    assert_ne!(err1, err3);
}

#[test]
fn test_sorted_error_is_std_error() {
    fn check_is_error<E: std::error::Error>(_: E) {}
    check_is_error(SortedError::UnsortedEntry(1u32));
}
