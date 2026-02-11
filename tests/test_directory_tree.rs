//! Tests for DirectoryTree struct.

use geometric_traits::naive_structs::DirectoryTree;
use std::path::PathBuf;

#[test]
fn test_directory_tree_from_path() {
    // Use the current test directory structure
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let tree = DirectoryTree::from(path.clone());

    // The tree should be created successfully - we just verify it doesn't panic
    // and produces some output
    let display = format!("{tree}");
    assert!(!display.is_empty());
    assert!(display.contains("tests"));
}

#[test]
fn test_directory_tree_display() {
    // Create tree from src directory
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let tree = DirectoryTree::from(path);

    let display = format!("{tree}");
    // Should contain subdirectories
    assert!(display.contains("src"));
    // Should have some content
    assert!(display.len() > 10);
}

#[test]
fn test_directory_tree_single_file_dir() {
    // Use a subdirectory with known structure
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let tree = DirectoryTree::from(path);

    let display = format!("{tree}");
    // Should contain the root directory
    assert!(!display.is_empty());
}
