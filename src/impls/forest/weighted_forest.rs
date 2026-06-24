//! Submodule providing the [`WeightedForest`] type, a rooted forest whose edges
//! carry weights, exposed as a directed valued child-to-parent sparse adjacency
//! matrix.

#[cfg(feature = "mem_dbg")]
use alloc::string::String;
use alloc::vec::Vec;
use core::{fmt::Debug, ops::Add};

use multi_ranged::Step;
use num_traits::{AsPrimitive, Zero};

use super::SymmetricForest;
use crate::{
    impls::{SymmetricCSR2D, ValuedCSR2D},
    traits::{
        Matrix, Matrix2D, PositiveInteger, SizedSparseMatrix, SparseMatrix, SparseMatrix2D,
        SparseSquareMatrix, SparseValuedMatrix, SparseValuedMatrix2D, SquareMatrix, TryFromUsize,
        ValuedMatrix, ValuedMatrix2D,
    },
};

#[cfg_attr(feature = "mem_size", derive(mem_dbg::MemSize))]
#[cfg_attr(feature = "mem_size", mem_size(rec))]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemDbg))]
#[derive(Clone, Eq, PartialEq, Hash)]
/// A rooted forest whose tree edges carry weights.
///
/// Node `i` in `0..N` stores `(parent_i, weight_i)` in `parent[i]`, where
/// `weight_i` is the weight of the edge connecting `i` to its parent. A root
/// marks itself, storing `(i, F::zero())`. The forest is exposed as a directed
/// valued child-to-parent adjacency matrix: row `i` holds the single value
/// `weight_i` at column `parent_i` when `i` is not a root, and an empty row
/// when `i` is a root.
pub struct WeightedForest<T, F> {
    /// The parent and edge weight of each node, indexed by node identifier.
    parent: Vec<(T, F)>,
}

impl<T: Debug, F: Debug> Debug for WeightedForest<T, F> {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WeightedForest").field("parent", &self.parent).finish()
    }
}

impl<T, F> WeightedForest<T, F> {
    #[inline]
    #[must_use]
    /// Builds a weighted forest from a parent array of `(parent, weight)`
    /// pairs.
    ///
    /// Node `i` is a root when its parent points to `i` itself.
    pub fn from_parent_array(parent: Vec<(T, F)>) -> Self {
        Self { parent }
    }

    #[inline]
    #[must_use]
    /// Returns a reference to the underlying parent array.
    pub fn parent_array(&self) -> &[(T, F)] {
        &self.parent
    }
}

impl<T: AsPrimitive<usize> + Copy, F: Clone> WeightedForest<T, F> {
    #[inline]
    #[must_use]
    /// Returns the parent stored for node `i`.
    ///
    /// For a root, the returned parent equals `i` itself.
    pub fn parent(&self, i: T) -> T {
        self.parent[i.as_()].0
    }

    #[inline]
    #[must_use]
    /// Returns the weight of the edge connecting node `i` to its parent.
    ///
    /// For a root, this is the stored zero weight.
    pub fn weight(&self, i: T) -> F {
        self.parent[i.as_()].1.clone()
    }

    #[inline]
    #[must_use]
    /// Returns whether node `i` is a root (it points to itself).
    pub fn is_root(&self, i: T) -> bool {
        self.parent[i.as_()].0.as_() == i.as_()
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
            .filter_map(|(node, (parent, _))| (parent.as_() == node).then_some(*parent))
    }

    #[inline]
    #[must_use]
    /// Returns the number of roots in the forest.
    pub fn number_of_roots(&self) -> usize {
        self.parent.iter().enumerate().filter(|(node, (parent, _))| parent.as_() == *node).count()
    }

    #[inline]
    #[must_use]
    /// Returns the number of connected components, which equals the number of
    /// roots.
    pub fn number_of_components(&self) -> usize {
        self.number_of_roots()
    }

    #[inline]
    #[must_use]
    /// Returns the number of edges, which equals the number of non-root nodes.
    pub fn len(&self) -> usize {
        self.number_of_nodes() - self.number_of_roots()
    }

    #[inline]
    #[must_use]
    /// Returns whether the forest has no edges (every node is a root).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    #[must_use]
    /// Returns whether the forest is a spanning tree, meaning it has at most
    /// one root.
    pub fn is_spanning_tree(&self) -> bool {
        self.number_of_roots() <= 1
    }

    #[inline]
    /// Returns an iterator over the canonical undirected edges of the forest.
    ///
    /// Each non-root node `i` yields `(min(i, parent_i), max(i, parent_i),
    /// weight_i)`.
    pub fn edge_iter(&self) -> impl Iterator<Item = (T, T, F)> + '_
    where
        T: TryFromUsize,
    {
        self.parent.iter().enumerate().filter_map(|(node, (parent, weight))| {
            if parent.as_() == node {
                return None;
            }
            let child = T::try_from_usize(node)
                .unwrap_or_else(|_| unreachable!("a node index always fits its own type"));
            let (low, high) =
                if child.as_() <= parent.as_() { (child, *parent) } else { (*parent, child) };
            Some((low, high, weight.clone()))
        })
    }
}

impl<T: AsPrimitive<usize> + Copy, F: Copy + Add<Output = F> + Zero> WeightedForest<T, F> {
    #[inline]
    #[must_use]
    /// Returns the total weight of the forest, summing every non-root edge
    /// weight.
    pub fn total_weight(&self) -> F {
        self.parent.iter().enumerate().fold(F::zero(), |accumulator, (node, (parent, weight))| {
            if parent.as_() == node { accumulator } else { accumulator + *weight }
        })
    }
}

impl<T, F> WeightedForest<T, F>
where
    T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize + TryFrom<T>,
    F: Clone,
{
    #[inline]
    #[must_use]
    /// Builds the undirected symmetric view of this forest.
    ///
    /// Each non-root node `i` contributes the upper-triangular entry
    /// `(min(i, parent_i), max(i, parent_i), weight_i)`. The entries are sorted
    /// lexicographically and handed to
    /// [`SymmetricCSR2D::from_sorted_upper_triangular_entries`]. Forest edges
    /// are duplicate-free and become upper-triangular and sorted, so the
    /// conversion is always feasible.
    pub fn symmetrize(&self) -> SymmetricForest<T, F> {
        let order = T::try_from_usize(self.number_of_nodes())
            .unwrap_or_else(|_| unreachable!("the number of nodes must fit the index type"));

        let mut entries: Vec<(T, T, F)> = self
            .parent
            .iter()
            .enumerate()
            .filter_map(|(node, (parent, weight))| {
                if parent.as_() == node {
                    return None;
                }
                let (low, high) =
                    if node <= parent.as_() { (node, parent.as_()) } else { (parent.as_(), node) };
                let low = T::try_from_usize(low)
                    .unwrap_or_else(|_| unreachable!("a node index always fits its own type"));
                let high = T::try_from_usize(high)
                    .unwrap_or_else(|_| unreachable!("a node index always fits its own type"));
                Some((low, high, weight.clone()))
            })
            .collect();

        entries.sort_by(|(row1, column1, _), (row2, column2, _)| {
            (row1.as_(), column1.as_()).cmp(&(row2.as_(), column2.as_()))
        });

        let matrix: SymmetricCSR2D<ValuedCSR2D<T, T, T, F>> =
            SymmetricCSR2D::from_sorted_upper_triangular_entries(order, entries)
                .expect("forest edges are duplicate-free, upper-triangular, and sorted");

        SymmetricForest::from_parts(matrix)
    }
}

/// Iterator over the non-root entries of a [`WeightedForest`].
///
/// Each yielded item is the triple `(child, parent, weight)` for a non-root
/// node, scanned in increasing node order. The iterator is double-ended via
/// independent front and back cursors.
#[derive(Clone)]
pub struct WeightedForestNonRootEntries<'a, T, F> {
    /// The parent slice being scanned.
    parent: &'a [(T, F)],
    /// The next node index to consider from the front.
    front: usize,
    /// One past the next node index to consider from the back.
    back: usize,
}

impl<'a, T, F> WeightedForestNonRootEntries<'a, T, F> {
    #[inline]
    fn new(parent: &'a [(T, F)]) -> Self {
        Self { parent, front: 0, back: parent.len() }
    }
}

impl<T: AsPrimitive<usize> + Copy + TryFromUsize, F: Clone> Iterator
    for WeightedForestNonRootEntries<'_, T, F>
{
    type Item = (T, T, F);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.front < self.back {
            let node = self.front;
            self.front += 1;
            let (parent, weight) = &self.parent[node];
            if parent.as_() != node {
                let row = T::try_from_usize(node)
                    .unwrap_or_else(|_| unreachable!("a node index always fits its own type"));
                return Some((row, *parent, weight.clone()));
            }
        }
        None
    }
}

impl<T: AsPrimitive<usize> + Copy + TryFromUsize, F: Clone> DoubleEndedIterator
    for WeightedForestNonRootEntries<'_, T, F>
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        while self.front < self.back {
            let node = self.back - 1;
            self.back -= 1;
            let (parent, weight) = &self.parent[node];
            if parent.as_() != node {
                let row = T::try_from_usize(node)
                    .unwrap_or_else(|_| unreachable!("a node index always fits its own type"));
                return Some((row, *parent, weight.clone()));
            }
        }
        None
    }
}

/// Extracts the child (row) and parent (column) coordinates from a non-root
/// entry.
#[inline]
fn entry_coordinates<T, F>((row, parent, _): (T, T, F)) -> (T, T) {
    (row, parent)
}

/// Extracts the parent (column) from a non-root entry.
#[inline]
fn entry_column<T, F>((_, parent, _): (T, T, F)) -> T {
    parent
}

/// Extracts the child (row) from a non-root entry.
#[inline]
fn entry_row<T, F>((row, _, _): (T, T, F)) -> T {
    row
}

/// Extracts the weight (value) from a non-root entry.
#[inline]
fn entry_value<T, F>((_, _, weight): (T, T, F)) -> F {
    weight
}

impl<T, F> Matrix for WeightedForest<T, F>
where
    T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    F: Clone,
{
    type Coordinates = (T, T);

    #[inline]
    fn shape(&self) -> Vec<usize> {
        let n = self.parent.len();
        vec![n, n]
    }
}

impl<T, F> Matrix2D for WeightedForest<T, F>
where
    T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    F: Clone,
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

impl<T, F> SquareMatrix for WeightedForest<T, F>
where
    T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    F: Clone,
{
    type Index = T;

    #[inline]
    fn order(&self) -> Self::Index {
        self.number_of_rows()
    }
}

impl<T, F> SparseMatrix for WeightedForest<T, F>
where
    T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    F: Clone,
{
    type SparseIndex = T;
    type SparseCoordinates<'a>
        = core::iter::Map<WeightedForestNonRootEntries<'a, T, F>, fn((T, T, F)) -> (T, T)>
    where
        Self: 'a;

    #[inline]
    fn sparse_coordinates(&self) -> Self::SparseCoordinates<'_> {
        WeightedForestNonRootEntries::new(&self.parent)
            .map(entry_coordinates as fn((T, T, F)) -> (T, T))
    }

    #[inline]
    fn last_sparse_coordinates(&self) -> Option<Self::Coordinates> {
        self.sparse_coordinates().next_back()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        WeightedForestNonRootEntries::new(&self.parent).next().is_none()
    }
}

impl<T, F> SizedSparseMatrix for WeightedForest<T, F>
where
    T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    F: Clone,
{
    #[inline]
    fn number_of_defined_values(&self) -> Self::SparseIndex {
        let number_of_edges = self.number_of_nodes() - self.number_of_roots();
        T::try_from_usize(number_of_edges)
            .unwrap_or_else(|_| unreachable!("the number of edges must fit the index type"))
    }
}

impl<T, F> SparseSquareMatrix for WeightedForest<T, F>
where
    T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    F: Clone,
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

impl<T, F> SparseMatrix2D for WeightedForest<T, F>
where
    T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    F: Clone,
{
    type SparseRow<'a>
        = core::option::IntoIter<T>
    where
        Self: 'a;
    type SparseColumns<'a>
        = core::iter::Map<WeightedForestNonRootEntries<'a, T, F>, fn((T, T, F)) -> T>
    where
        Self: 'a;
    type SparseRows<'a>
        = core::iter::Map<WeightedForestNonRootEntries<'a, T, F>, fn((T, T, F)) -> T>
    where
        Self: 'a;

    #[inline]
    fn sparse_row(&self, row: Self::RowIndex) -> Self::SparseRow<'_> {
        if self.is_root(row) { None.into_iter() } else { Some(self.parent(row)).into_iter() }
    }

    #[inline]
    fn sparse_columns(&self) -> Self::SparseColumns<'_> {
        WeightedForestNonRootEntries::new(&self.parent).map(entry_column as fn((T, T, F)) -> T)
    }

    #[inline]
    fn sparse_rows(&self) -> Self::SparseRows<'_> {
        WeightedForestNonRootEntries::new(&self.parent).map(entry_row as fn((T, T, F)) -> T)
    }
}

impl<T, F> ValuedMatrix for WeightedForest<T, F>
where
    T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    F: Clone,
{
    type Value = F;
}

impl<T, F> ValuedMatrix2D for WeightedForest<T, F>
where
    T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    F: Clone,
{
}

impl<T, F> SparseValuedMatrix for WeightedForest<T, F>
where
    T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    F: Clone,
{
    type SparseValues<'a>
        = core::iter::Map<WeightedForestNonRootEntries<'a, T, F>, fn((T, T, F)) -> F>
    where
        Self: 'a;

    #[inline]
    fn sparse_values(&self) -> Self::SparseValues<'_> {
        WeightedForestNonRootEntries::new(&self.parent).map(entry_value as fn((T, T, F)) -> F)
    }
}

impl<T, F> SparseValuedMatrix2D for WeightedForest<T, F>
where
    T: Step + PositiveInteger + AsPrimitive<usize> + TryFromUsize,
    F: Clone,
{
    type SparseRowValues<'a>
        = core::option::IntoIter<F>
    where
        Self: 'a;

    #[inline]
    fn sparse_row_values(&self, row: Self::RowIndex) -> Self::SparseRowValues<'_> {
        if self.is_root(row) { None.into_iter() } else { Some(self.weight(row)).into_iter() }
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    type TestWeightedForest = WeightedForest<usize, i64>;

    fn two_component_forest() -> TestWeightedForest {
        // Roots are nodes 0 and 3. Tree A: 0 <- {1 (w 5) <- 2 (w 7)}.
        // Tree B: 3 <- 4 (w 9).
        WeightedForest::from_parent_array(vec![(0, 0), (0, 5), (1, 7), (3, 0), (3, 9)])
    }

    #[test]
    fn test_basic_counts() {
        let forest = two_component_forest();
        assert_eq!(forest.number_of_nodes(), 5);
        assert_eq!(forest.number_of_roots(), 2);
        assert_eq!(forest.number_of_components(), 2);
        assert_eq!(forest.len(), 3);
        assert!(!forest.is_empty());
        assert!(!forest.is_spanning_tree());
        let roots: Vec<usize> = forest.roots().collect();
        assert_eq!(roots, vec![0, 3]);
    }

    #[test]
    fn test_matrix_stack() {
        let forest = two_component_forest();
        assert_eq!(forest.shape(), vec![5, 5]);
        assert_eq!(forest.order(), 5);
        assert_eq!(forest.number_of_defined_values(), 3);
        assert!(!forest.is_symmetric());
        assert!(forest.sparse_row(0).next().is_none());
        assert_eq!(forest.sparse_row(2).collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn test_sparse_values_and_rows() {
        let forest = two_component_forest();
        let values: Vec<i64> = forest.sparse_values().collect();
        assert_eq!(values, vec![5, 7, 9]);
        assert!(forest.sparse_row_values(0).next().is_none());
        assert_eq!(forest.sparse_row_values(1).collect::<Vec<_>>(), vec![5]);
        assert_eq!(forest.sparse_row_values(4).collect::<Vec<_>>(), vec![9]);
        let coords: Vec<(usize, usize)> = forest.sparse_coordinates().collect();
        assert_eq!(coords, vec![(1, 0), (2, 1), (4, 3)]);
    }

    #[test]
    fn test_total_weight_and_edge_iter() {
        let forest = two_component_forest();
        assert_eq!(forest.total_weight(), 21);
        let edges: Vec<(usize, usize, i64)> = forest.edge_iter().collect();
        assert_eq!(edges, vec![(0, 1, 5), (1, 2, 7), (3, 4, 9)]);
    }

    #[test]
    fn test_spanning_tree_single_root() {
        let forest: TestWeightedForest =
            WeightedForest::from_parent_array(vec![(0, 0), (0, 2), (0, 3)]);
        assert!(forest.is_spanning_tree());
        assert_eq!(forest.number_of_components(), 1);
        assert_eq!(forest.total_weight(), 5);
    }

    #[test]
    fn test_parent_array_round_trips() {
        let forest = two_component_forest();
        assert_eq!(forest.parent_array(), &[(0, 0), (0, 5), (1, 7), (3, 0), (3, 9)]);
    }

    #[test]
    fn test_last_sparse_coordinates_and_reverse_iteration() {
        let forest = two_component_forest();
        // The last non-root entry is node 4 pointing at its parent 3.
        assert_eq!(forest.last_sparse_coordinates(), Some((4, 3)));
        let reversed: Vec<(usize, usize)> = forest.sparse_coordinates().rev().collect();
        assert_eq!(reversed, vec![(4, 3), (2, 1), (1, 0)]);
        let reversed_values: Vec<i64> = forest.sparse_values().rev().collect();
        assert_eq!(reversed_values, vec![9, 7, 5]);
    }

    #[test]
    fn test_sparse_matrix_is_empty() {
        let forest = two_component_forest();
        assert!(!SparseMatrix::is_empty(&forest));
        // Two isolated nodes (both roots) have no edges.
        let empty: TestWeightedForest = WeightedForest::from_parent_array(vec![(0, 0), (1, 0)]);
        assert!(SparseMatrix::is_empty(&empty));
    }

    #[test]
    fn test_debug_contains_type_name() {
        let forest = two_component_forest();
        let debug = alloc::format!("{forest:?}");
        assert!(debug.contains("WeightedForest"));
    }

    #[test]
    fn test_matrix_columns_rows_and_diagonal() {
        let forest = two_component_forest();
        assert_eq!(forest.number_of_rows(), 5);
        assert_eq!(forest.number_of_columns(), 5);
        assert_eq!(forest.number_of_defined_diagonal_values(), 0);
        // The whole-matrix column and row views, one per non-root edge.
        let columns: Vec<usize> = forest.sparse_columns().collect();
        assert_eq!(columns, vec![0, 1, 3]);
        let rows: Vec<usize> = forest.sparse_rows().collect();
        assert_eq!(rows, vec![1, 2, 4]);
        // Reverse iteration of the column and row views.
        let columns_rev: Vec<usize> = forest.sparse_columns().rev().collect();
        assert_eq!(columns_rev, vec![3, 1, 0]);
        let rows_rev: Vec<usize> = forest.sparse_rows().rev().collect();
        assert_eq!(rows_rev, vec![4, 2, 1]);
    }
}
