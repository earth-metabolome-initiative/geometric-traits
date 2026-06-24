//! Submodule providing the [`Forest`] type, a rooted forest stored as a parent
//! array and exposed as a directed child-to-parent sparse adjacency matrix.

#[cfg(feature = "mem_dbg")]
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Debug;

use multi_ranged::Step;
use num_traits::AsPrimitive;

use crate::traits::{
    Matrix, Matrix2D, PositiveInteger, SizedSparseMatrix, SparseMatrix, SparseMatrix2D,
    SparseSquareMatrix, SquareMatrix, TryFromUsize,
};

#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
#[derive(Clone, Eq, PartialEq, Hash)]
/// A rooted forest stored as a parent array.
///
/// Node `i` in `0..N` stores its parent in `parent[i]`. A root marks itself,
/// meaning `parent[i].as_() == i`. The forest is exposed as a directed
/// child-to-parent adjacency matrix: row `i` yields the single column
/// `parent[i]` when `i` is not a root, and an empty row when `i` is a root (no
/// self-loops are produced for roots).
pub struct Forest<T> {
    /// The parent of each node, indexed by node identifier.
    parent: Vec<T>,
}

impl<T: Debug> Debug for Forest<T> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Forest").field("parent", &self.parent).finish()
    }
}

impl<T> Forest<T> {
    #[inline]
    #[must_use]
    /// Builds a forest from a parent array.
    ///
    /// Node `i` is a root when `parent[i]` points to `i` itself.
    pub fn from_parent_array(parent: Vec<T>) -> Self {
        Self { parent }
    }

    #[inline]
    #[must_use]
    /// Returns a reference to the underlying parent array.
    pub fn parent_array(&self) -> &[T] {
        &self.parent
    }
}

impl<T: AsPrimitive<usize> + Copy> Forest<T> {
    #[inline]
    #[must_use]
    /// Returns the parent stored for node `i`.
    ///
    /// For a root, the returned parent equals `i` itself.
    pub fn parent(&self, i: T) -> T {
        self.parent[i.as_()]
    }

    #[inline]
    #[must_use]
    /// Returns whether node `i` is a root (it points to itself).
    pub fn is_root(&self, i: T) -> bool {
        self.parent[i.as_()].as_() == i.as_()
    }

    #[inline]
    #[must_use]
    /// Returns the number of nodes in the forest.
    pub fn number_of_nodes(&self) -> usize {
        self.parent.len()
    }

    #[inline]
    /// Returns an iterator over the root node identifiers.
    pub fn roots(&self) -> impl Iterator<Item = T> + '_ {
        self.parent
            .iter()
            .enumerate()
            .filter_map(|(node, parent)| (parent.as_() == node).then_some(*parent))
    }

    #[inline]
    #[must_use]
    /// Returns the number of roots in the forest.
    pub fn number_of_roots(&self) -> usize {
        self.parent.iter().enumerate().filter(|(node, parent)| parent.as_() == *node).count()
    }

    #[inline]
    #[must_use]
    /// Returns the number of connected components, which equals the number of
    /// roots.
    pub fn number_of_components(&self) -> usize {
        self.number_of_roots()
    }
}

/// Iterator over the non-root entries of a [`Forest`].
///
/// Each yielded item is the pair `(child, parent)` for a non-root node, scanned
/// in increasing node order. The iterator is double-ended via independent front
/// and back cursors.
#[derive(Clone)]
pub struct ForestNonRootEntries<'a, T> {
    /// The parent slice being scanned.
    parent: &'a [T],
    /// The next node index to consider from the front.
    front: usize,
    /// One past the next node index to consider from the back.
    back: usize,
}

impl<'a, T: AsPrimitive<usize> + Copy> ForestNonRootEntries<'a, T> {
    #[inline]
    fn new(parent: &'a [T]) -> Self {
        Self { parent, front: 0, back: parent.len() }
    }
}

impl<T: AsPrimitive<usize> + Copy + TryFromUsize> Iterator for ForestNonRootEntries<'_, T> {
    type Item = (T, T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.front < self.back {
            let node = self.front;
            self.front += 1;
            let parent = self.parent[node];
            if parent.as_() != node {
                let row = T::try_from_usize(node)
                    .unwrap_or_else(|_| unreachable!("a node index always fits its own type"));
                return Some((row, parent));
            }
        }
        None
    }
}

impl<T: AsPrimitive<usize> + Copy + TryFromUsize> DoubleEndedIterator
    for ForestNonRootEntries<'_, T>
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        while self.front < self.back {
            let node = self.back - 1;
            self.back -= 1;
            let parent = self.parent[node];
            if parent.as_() != node {
                let row = T::try_from_usize(node)
                    .unwrap_or_else(|_| unreachable!("a node index always fits its own type"));
                return Some((row, parent));
            }
        }
        None
    }
}

/// Extracts the parent (column) from a non-root entry.
#[inline]
fn entry_column<T>((_, parent): (T, T)) -> T {
    parent
}

/// Extracts the child (row) from a non-root entry.
#[inline]
fn entry_row<T>((row, _): (T, T)) -> T {
    row
}

impl<T> Matrix for Forest<T>
where
    T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
{
    type Coordinates = (T, T);

    #[inline]
    fn shape(&self) -> Vec<usize> {
        let n = self.parent.len();
        vec![n, n]
    }
}

impl<T> Matrix2D for Forest<T>
where
    T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
{
    type RowIndex = T;
    type ColumnIndex = T;

    #[inline]
    fn number_of_rows(&self) -> Self::RowIndex {
        T::try_from_usize(self.parent.len())
            .unwrap_or_else(|_| unreachable!("the number of nodes must fit the index type"))
    }

    #[inline]
    fn number_of_columns(&self) -> Self::ColumnIndex {
        self.number_of_rows()
    }
}

impl<T> SquareMatrix for Forest<T>
where
    T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
{
    type Index = T;

    #[inline]
    fn order(&self) -> Self::Index {
        self.number_of_rows()
    }
}

impl<T> SparseMatrix for Forest<T>
where
    T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
{
    type SparseIndex = T;
    type SparseCoordinates<'a>
        = ForestNonRootEntries<'a, T>
    where
        Self: 'a;

    #[inline]
    fn sparse_coordinates(&self) -> Self::SparseCoordinates<'_> {
        ForestNonRootEntries::new(&self.parent)
    }

    #[inline]
    fn last_sparse_coordinates(&self) -> Option<Self::Coordinates> {
        self.sparse_coordinates().next_back()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.sparse_coordinates().next().is_none()
    }
}

impl<T> SizedSparseMatrix for Forest<T>
where
    T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
{
    #[inline]
    fn number_of_defined_values(&self) -> Self::SparseIndex {
        let number_of_edges = self.number_of_nodes() - self.number_of_roots();
        T::try_from_usize(number_of_edges)
            .unwrap_or_else(|_| unreachable!("the number of edges must fit the index type"))
    }
}

impl<T> SparseSquareMatrix for Forest<T>
where
    T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
{
    #[inline]
    fn number_of_defined_diagonal_values(&self) -> Self::Index {
        T::zero()
    }

    #[inline]
    fn is_symmetric(&self) -> bool {
        false
    }
}

impl<T> SparseMatrix2D for Forest<T>
where
    T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
{
    type SparseRow<'a>
        = core::option::IntoIter<T>
    where
        Self: 'a;
    type SparseColumns<'a>
        = core::iter::Map<ForestNonRootEntries<'a, T>, fn((T, T)) -> T>
    where
        Self: 'a;
    type SparseRows<'a>
        = core::iter::Map<ForestNonRootEntries<'a, T>, fn((T, T)) -> T>
    where
        Self: 'a;

    #[inline]
    fn sparse_row(&self, row: Self::RowIndex) -> Self::SparseRow<'_> {
        if self.is_root(row) { None.into_iter() } else { Some(self.parent(row)).into_iter() }
    }

    #[inline]
    fn sparse_columns(&self) -> Self::SparseColumns<'_> {
        self.sparse_coordinates().map(entry_column as fn((T, T)) -> T)
    }

    #[inline]
    fn sparse_rows(&self) -> Self::SparseRows<'_> {
        self.sparse_coordinates().map(entry_row as fn((T, T)) -> T)
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    type TestForest = Forest<usize>;

    fn two_component_forest() -> TestForest {
        // Roots are nodes 0 and 3. Tree A: 0 <- {1 <- 2}. Tree B: 3 <- 4.
        Forest::from_parent_array(vec![0, 0, 1, 3, 3])
    }

    #[test]
    fn test_basic_counts() {
        let forest = two_component_forest();
        assert_eq!(forest.number_of_nodes(), 5);
        assert_eq!(forest.number_of_roots(), 2);
        assert_eq!(forest.number_of_components(), 2);
        let roots: Vec<usize> = forest.roots().collect();
        assert_eq!(roots, vec![0, 3]);
        assert!(forest.is_root(0));
        assert!(!forest.is_root(1));
        assert!(forest.is_root(3));
    }

    #[test]
    fn test_matrix_shape_and_order() {
        let forest = two_component_forest();
        assert_eq!(forest.shape(), vec![5, 5]);
        assert_eq!(forest.number_of_rows(), 5);
        assert_eq!(forest.number_of_columns(), 5);
        assert_eq!(forest.order(), 5);
        assert_eq!(forest.number_of_defined_values(), 3);
        assert_eq!(forest.number_of_defined_diagonal_values(), 0);
        assert!(!forest.is_symmetric());
        assert!(!forest.is_empty());
    }

    #[test]
    fn test_sparse_rows_roots_empty_children_point_to_parent() {
        let forest = two_component_forest();
        assert!(forest.sparse_row(0).next().is_none());
        assert!(forest.sparse_row(3).next().is_none());
        assert_eq!(forest.sparse_row(1).collect::<Vec<_>>(), vec![0]);
        assert_eq!(forest.sparse_row(2).collect::<Vec<_>>(), vec![1]);
        assert_eq!(forest.sparse_row(4).collect::<Vec<_>>(), vec![3]);
    }

    #[test]
    fn test_sparse_coordinates_and_views() {
        let forest = two_component_forest();
        let coords: Vec<(usize, usize)> = forest.sparse_coordinates().collect();
        assert_eq!(coords, vec![(1, 0), (2, 1), (4, 3)]);
        let columns: Vec<usize> = forest.sparse_columns().collect();
        assert_eq!(columns, vec![0, 1, 3]);
        let rows: Vec<usize> = forest.sparse_rows().collect();
        assert_eq!(rows, vec![1, 2, 4]);
    }

    #[test]
    fn test_double_ended_entries() {
        let forest = two_component_forest();
        let reversed: Vec<(usize, usize)> = forest.sparse_coordinates().rev().collect();
        assert_eq!(reversed, vec![(4, 3), (2, 1), (1, 0)]);
        assert_eq!(forest.last_sparse_coordinates(), Some((4, 3)));
    }

    #[test]
    fn test_single_root_is_empty() {
        let forest: TestForest = Forest::from_parent_array(vec![0]);
        assert!(forest.is_empty());
        assert_eq!(forest.number_of_defined_values(), 0);
        assert_eq!(forest.last_sparse_coordinates(), None);
    }

    #[test]
    fn test_parent_array_round_trips() {
        let forest = two_component_forest();
        assert_eq!(forest.parent_array(), &[0, 0, 1, 3, 3]);
    }

    #[test]
    fn test_debug_contains_type_name() {
        let forest = two_component_forest();
        let debug = alloc::format!("{forest:?}");
        assert!(debug.contains("Forest"));
    }
}
