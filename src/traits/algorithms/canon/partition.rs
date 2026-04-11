//! Backtrackable ordered partitions for dense vertex identifiers.
//!
//! This module provides the partition data structure that sits under
//! individualization-refinement canonization algorithms. The representation is
//! array-backed and supports:
//!
//! - ordered cells over `0..n`
//! - cell splitting and vertex individualization
//! - efficient backtracking to earlier partition states
//! - iteration over all cells or only non-singleton cells

use alloc::{vec, vec::Vec};

/// Identifier of a cell inside an ordered partition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PartitionCellId(usize);

impl PartitionCellId {
    /// Returns the internal dense cell index.
    #[must_use]
    #[inline]
    pub fn index(self) -> usize {
        self.0
    }
}

/// A previously recorded point that the partition can backtrack to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PartitionBacktrackPoint(usize);

impl PartitionBacktrackPoint {
    /// Returns the internal dense backtrack-point index.
    #[must_use]
    #[inline]
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug)]
struct PartitionCell {
    first: usize,
    length: usize,
    prev: Option<usize>,
    next: Option<usize>,
    prev_non_singleton: Option<usize>,
    next_non_singleton: Option<usize>,
    split_level: usize,
    component_level: usize,
}

impl PartitionCell {
    #[must_use]
    #[inline]
    fn inactive() -> Self {
        Self {
            first: 0,
            length: 0,
            prev: None,
            next: None,
            prev_non_singleton: None,
            next_non_singleton: None,
            split_level: 0,
            component_level: 0,
        }
    }

    #[must_use]
    #[inline]
    fn is_unit(&self) -> bool {
        self.length == 1
    }

    #[must_use]
    #[inline]
    fn is_non_singleton(&self) -> bool {
        self.length > 1
    }

    #[must_use]
    #[inline]
    fn is_active(&self) -> bool {
        self.length > 0
    }
}

#[derive(Clone, Debug)]
struct SplitRecord {
    left_cell: usize,
    right_cell: usize,
    prev_non_singleton: Option<usize>,
    next_non_singleton: Option<usize>,
}

#[derive(Clone, Debug)]
struct BacktrackRecord {
    split_trail_len: usize,
    component_level_trail_len: usize,
    max_component_level: usize,
}

#[derive(Clone, Debug)]
struct ComponentLevelRecord {
    cell_id: usize,
    previous_level: usize,
}

/// Read-only view of one ordered partition cell.
#[derive(Clone, Copy, Debug)]
pub struct PartitionCellView<'a> {
    id: PartitionCellId,
    elements: &'a [usize],
    split_level: usize,
}

impl<'a> PartitionCellView<'a> {
    /// Returns the identifier of the cell.
    #[must_use]
    #[inline]
    pub fn id(self) -> PartitionCellId {
        self.id
    }

    /// Returns the dense elements contained in this cell.
    #[must_use]
    #[inline]
    pub fn elements(self) -> &'a [usize] {
        self.elements
    }

    /// Returns the number of elements in this cell.
    #[must_use]
    #[inline]
    pub fn len(self) -> usize {
        self.elements.len()
    }

    /// Returns whether the cell contains no elements.
    #[must_use]
    #[inline]
    pub fn is_empty(self) -> bool {
        self.elements.is_empty()
    }

    /// Returns whether this cell is a singleton.
    #[must_use]
    #[inline]
    pub fn is_unit(self) -> bool {
        self.elements.len() == 1
    }

    /// Returns the split depth at which this cell was created.
    #[must_use]
    #[inline]
    pub fn split_level(self) -> usize {
        self.split_level
    }
}

/// Iterator over partition cells in cell order.
pub struct OrderedPartitionCells<'a> {
    partition: &'a BacktrackableOrderedPartition,
    next_cell: Option<usize>,
    non_singleton_only: bool,
}

impl<'a> Iterator for OrderedPartitionCells<'a> {
    type Item = PartitionCellView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let cell_id = self.next_cell?;
        let cell = &self.partition.cells[cell_id];
        self.next_cell = if self.non_singleton_only { cell.next_non_singleton } else { cell.next };
        let start = cell.first;
        let end = start + cell.length;
        Some(PartitionCellView {
            id: PartitionCellId(cell_id),
            elements: &self.partition.elements[start..end],
            split_level: cell.split_level,
        })
    }
}

/// Array-backed ordered partition with efficient backtracking.
///
/// The partition always contains the dense elements `0..order`, arranged into
/// contiguous cells in a single `elements` array. Splits only change cell
/// boundaries and element order inside the affected cell, which allows
/// backtracking to merge cells without restoring any historical permutation.
///
/// A backtrack therefore restores the same cell partition, but it does not
/// guarantee that elements inside a merged cell appear in the same order they
/// had before the split sequence.
#[derive(Clone, Debug)]
pub struct BacktrackableOrderedPartition {
    order: usize,
    elements: Vec<usize>,
    positions: Vec<usize>,
    element_to_cell: Vec<usize>,
    cells: Vec<PartitionCell>,
    free_cells: Vec<usize>,
    first_cell: Option<usize>,
    first_non_singleton_cell: Option<usize>,
    discrete_cell_count: usize,
    split_trail: Vec<SplitRecord>,
    component_level_trail: Vec<ComponentLevelRecord>,
    max_component_level: usize,
    backtrack_records: Vec<BacktrackRecord>,
}

impl BacktrackableOrderedPartition {
    /// Creates the unit partition over `0..order`.
    ///
    /// For `order == 0`, the partition is empty and already discrete.
    #[must_use]
    pub fn new(order: usize) -> Self {
        if order == 0 {
            return Self {
                order,
                elements: Vec::new(),
                positions: Vec::new(),
                element_to_cell: Vec::new(),
                cells: Vec::new(),
                free_cells: Vec::new(),
                first_cell: None,
                first_non_singleton_cell: None,
                discrete_cell_count: 0,
                split_trail: Vec::new(),
                component_level_trail: Vec::new(),
                max_component_level: 0,
                backtrack_records: Vec::new(),
            };
        }

        let mut cells = vec![PartitionCell::inactive(); order];
        cells[0] = PartitionCell {
            first: 0,
            length: order,
            prev: None,
            next: None,
            prev_non_singleton: None,
            next_non_singleton: None,
            split_level: 0,
            component_level: 0,
        };
        let free_cells = (1..order).rev().collect();

        Self {
            order,
            elements: (0..order).collect(),
            positions: (0..order).collect(),
            element_to_cell: vec![0; order],
            cells,
            free_cells,
            first_cell: Some(0),
            first_non_singleton_cell: (order > 1).then_some(0),
            discrete_cell_count: usize::from(order == 1),
            split_trail: Vec::new(),
            component_level_trail: Vec::new(),
            max_component_level: 0,
            backtrack_records: Vec::new(),
        }
    }

    /// Returns the number of elements in the partition universe.
    #[must_use]
    #[inline]
    pub fn order(&self) -> usize {
        self.order
    }

    /// Returns whether the partition contains no elements.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.order == 0
    }

    /// Returns whether every element is in a singleton cell.
    #[must_use]
    #[inline]
    pub fn is_discrete(&self) -> bool {
        self.discrete_cell_count == self.order
    }

    /// Returns the number of singleton cells currently present.
    #[must_use]
    #[inline]
    pub fn number_of_discrete_cells(&self) -> usize {
        self.discrete_cell_count
    }

    /// Returns the number of active cells in the ordered partition.
    #[must_use]
    #[inline]
    pub fn number_of_cells(&self) -> usize {
        self.order.saturating_sub(self.free_cells.len())
    }

    /// Returns the current non-singleton signature of the partition.
    ///
    /// This is the ordered list of cell sizes greater than one.
    #[must_use]
    pub fn signature(&self) -> Vec<usize> {
        self.non_singleton_cells().map(PartitionCellView::len).collect()
    }

    /// Returns the identifier of the cell currently containing `element`.
    ///
    /// # Panics
    ///
    /// Panics if `element >= self.order()`.
    #[must_use]
    pub fn cell_of(&self, element: usize) -> PartitionCellId {
        self.assert_valid_element(element);
        PartitionCellId(self.element_to_cell[element])
    }

    /// Returns the current elements of `cell` as a contiguous slice.
    ///
    /// # Panics
    ///
    /// Panics if `cell` is not active.
    #[must_use]
    pub fn cell_elements(&self, cell: PartitionCellId) -> &[usize] {
        let cell_data = self.cell(cell.0);
        &self.elements[cell_data.first..cell_data.first + cell_data.length]
    }

    /// Returns the dense partition order across all active cells.
    #[must_use]
    pub(crate) fn ordered_elements(&self) -> &[usize] {
        &self.elements
    }

    /// Returns the starting offset of `cell` in the dense partition array.
    ///
    /// # Panics
    ///
    /// Panics if `cell` is not active.
    #[must_use]
    pub(crate) fn cell_first(&self, cell: PartitionCellId) -> usize {
        self.cell(cell.0).first
    }

    /// Returns the current size of `cell`.
    ///
    /// # Panics
    ///
    /// Panics if `cell` is not active.
    #[must_use]
    pub(crate) fn cell_len(&self, cell: PartitionCellId) -> usize {
        self.cell(cell.0).length
    }

    /// Returns the current component-recursion level of `cell`.
    #[must_use]
    pub(crate) fn cell_component_level(&self, cell: PartitionCellId) -> usize {
        self.cell(cell.0).component_level
    }

    /// Returns the highest component-recursion level that still contains a
    /// non-singleton cell.
    #[must_use]
    pub(crate) fn highest_non_singleton_component_level(&self) -> Option<usize> {
        self.non_singleton_cells().map(|cell| self.cell_component_level(cell.id())).max()
    }

    /// Promotes the listed cells to a fresh component-recursion level.
    ///
    /// The returned level is strictly greater than all previous component
    /// levels and is reverted by `goto_backtrack_point()`.
    #[must_use]
    pub(crate) fn promote_cells_to_new_component_level(
        &mut self,
        cells: &[PartitionCellId],
    ) -> usize {
        let new_level = self.max_component_level + 1;
        self.max_component_level = new_level;
        for &cell in cells {
            let cell_id = cell.0;
            let previous_level = self.cell(cell_id).component_level;
            if previous_level == new_level {
                continue;
            }
            self.component_level_trail.push(ComponentLevelRecord { cell_id, previous_level });
            self.cell_mut(cell_id).component_level = new_level;
        }
        new_level
    }

    /// Returns an iterator over all cells in cell order.
    #[must_use]
    pub fn cells(&self) -> OrderedPartitionCells<'_> {
        OrderedPartitionCells {
            partition: self,
            next_cell: self.first_cell,
            non_singleton_only: false,
        }
    }

    /// Returns an iterator over all non-singleton cells in cell order.
    #[must_use]
    pub fn non_singleton_cells(&self) -> OrderedPartitionCells<'_> {
        OrderedPartitionCells {
            partition: self,
            next_cell: self.first_non_singleton_cell,
            non_singleton_only: true,
        }
    }

    /// Records the current state as a future backtracking target.
    #[must_use]
    pub fn set_backtrack_point(&mut self) -> PartitionBacktrackPoint {
        let point = PartitionBacktrackPoint(self.backtrack_records.len());
        self.backtrack_records.push(BacktrackRecord {
            split_trail_len: self.split_trail.len(),
            component_level_trail_len: self.component_level_trail.len(),
            max_component_level: self.max_component_level,
        });
        point
    }

    /// Restores the partition to a previously recorded backtracking point.
    ///
    /// The point itself is removed, so later points recorded after it are also
    /// discarded.
    ///
    /// This restores cell boundaries and membership, but it does not restore
    /// the historical order of elements inside merged cells.
    ///
    /// # Panics
    ///
    /// Panics if `point` does not refer to an existing backtrack point.
    pub fn goto_backtrack_point(&mut self, point: PartitionBacktrackPoint) {
        let target = self
            .backtrack_records
            .get(point.0)
            .expect("backtrack point must refer to an existing checkpoint")
            .clone();
        self.backtrack_records.truncate(point.0);

        while self.component_level_trail.len() > target.component_level_trail_len {
            let record = self
                .component_level_trail
                .pop()
                .expect("component-level trail must contain at least one update to undo");
            self.cell_mut(record.cell_id).component_level = record.previous_level;
        }
        self.max_component_level = target.max_component_level;

        while self.split_trail.len() > target.split_trail_len {
            let record = self
                .split_trail
                .pop()
                .expect("split trail must contain at least one split to undo");
            self.undo_split(record);
        }
    }

    /// Splits a non-singleton `cell` by turning `element` into its own
    /// singleton cell.
    ///
    /// The new singleton cell is placed immediately after the remaining part
    /// of the original cell.
    ///
    /// # Panics
    ///
    /// Panics if `cell` is inactive, singleton, or does not contain
    /// `element`.
    #[must_use]
    pub fn individualize(&mut self, cell: PartitionCellId, element: usize) -> PartitionCellId {
        self.assert_valid_element(element);
        let cell_data = self.cell(cell.0).clone();
        assert!(cell_data.is_non_singleton(), "individualization requires a non-singleton cell");
        assert!(
            self.element_to_cell[element] == cell.0,
            "the individualized element must belong to the selected cell",
        );

        let position = self.positions[element];
        let last = cell_data.first + cell_data.length - 1;
        self.elements.swap(position, last);
        let swapped_left = self.elements[position];
        let swapped_right = self.elements[last];
        self.positions[swapped_left] = position;
        self.positions[swapped_right] = last;

        let new_cell = self.split_cell_at(cell.0, cell_data.length - 1);
        self.element_to_cell[element] = new_cell;
        PartitionCellId(new_cell)
    }

    /// Splits `cell` into maximal contiguous groups of equal `key_of(element)`.
    ///
    /// The groups are ordered by key, with ties broken by the elements' prior
    /// positions inside the cell so the result is deterministic.
    ///
    /// The returned identifiers are in cell order after the split. If every
    /// element has the same key, the returned vector contains only `cell`.
    ///
    /// # Panics
    ///
    /// Panics if `cell` is inactive.
    #[must_use]
    pub fn split_cell_by_key<Key, F>(
        &mut self,
        cell: PartitionCellId,
        mut key_of: F,
    ) -> Vec<PartitionCellId>
    where
        Key: Ord,
        F: FnMut(usize) -> Key,
    {
        let cell_data = self.cell(cell.0).clone();
        if cell_data.length <= 1 {
            return vec![cell];
        }

        let start = cell_data.first;
        let end = start + cell_data.length;
        let mut keyed_elements = self.elements[start..end]
            .iter()
            .copied()
            .map(|element| (key_of(element), self.positions[element], element))
            .collect::<Vec<_>>();
        keyed_elements
            .sort_unstable_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));

        let mut group_lengths = Vec::new();
        let mut current_group_len = 0usize;
        for index in 0..keyed_elements.len() {
            if index > 0 && keyed_elements[index - 1].0 != keyed_elements[index].0 {
                group_lengths.push(current_group_len);
                current_group_len = 0;
            }
            current_group_len += 1;
        }
        group_lengths.push(current_group_len);

        if group_lengths.len() == 1 {
            return vec![cell];
        }

        for (offset, (_, _, element)) in keyed_elements.into_iter().enumerate() {
            self.elements[start + offset] = element;
            self.positions[element] = start + offset;
        }

        let mut produced_cells = Vec::with_capacity(group_lengths.len());
        produced_cells.push(cell);
        let mut left_cell = cell.0;
        let mut consumed = 0usize;
        for split_index in 0..group_lengths.len() - 1 {
            consumed += group_lengths[split_index];
            let right_cell = self.split_cell_at(left_cell, consumed);
            produced_cells.push(PartitionCellId(right_cell));
            left_cell = right_cell;
            consumed = 0;
        }

        for partition_cell in &produced_cells[1..] {
            let elements = self.cell_elements(*partition_cell).to_vec();
            for element in elements {
                self.element_to_cell[element] = partition_cell.0;
            }
        }

        produced_cells
    }

    /// Splits `cell` into an untouched prefix and a touched suffix by moving
    /// `touched_elements_in_order` to the end of the cell using the same
    /// repeated tail-swap pattern as `bliss` uses for unit-cell refinement.
    ///
    /// The touched elements must be distinct members of `cell`, listed in the
    /// encounter order used to discover them. The returned cells are in cell
    /// order after the split. If no split is needed, the returned vector
    /// contains only `cell`.
    #[must_use]
    pub(crate) fn split_cell_by_tail_elements_in_order(
        &mut self,
        cell: PartitionCellId,
        touched_elements_in_order: &[usize],
    ) -> Vec<PartitionCellId> {
        let cell_data = self.cell(cell.0).clone();
        if cell_data.length <= 1 {
            return vec![cell];
        }

        let touched_len = touched_elements_in_order.len();
        if touched_len == 0 {
            return vec![cell];
        }

        debug_assert!(
            touched_elements_in_order
                .iter()
                .all(|&element| self.element_to_cell[element] == cell.0)
        );

        for (offset, &element) in touched_elements_in_order.iter().enumerate() {
            let position = self.positions[element];
            let swap_position = cell_data.first + cell_data.length - offset - 1;
            if position == swap_position {
                continue;
            }
            self.elements.swap(position, swap_position);
            let swapped_left = self.elements[position];
            let swapped_right = self.elements[swap_position];
            self.positions[swapped_left] = position;
            self.positions[swapped_right] = swap_position;
        }

        if touched_len == cell_data.length {
            return vec![cell];
        }

        let new_cell = self.split_cell_at(cell.0, cell_data.length - touched_len);
        let new_cell_id = PartitionCellId(new_cell);
        let moved_elements = self.cell_elements(new_cell_id).to_vec();
        for element in moved_elements {
            self.element_to_cell[element] = new_cell;
        }

        vec![cell, new_cell_id]
    }

    /// Splits `cell` by a binary `0/1` invariant using the same in-place
    /// partitioning strategy as `bliss` `sort_and_split_cell1()`.
    ///
    /// Elements for which `is_one(element)` is true form the new right cell.
    /// The input `ones_count` must equal the number of ones in the cell.
    #[must_use]
    pub(crate) fn split_cell_by_binary_invariant_like_bliss<F>(
        &mut self,
        cell: PartitionCellId,
        ones_count: usize,
        mut is_one: F,
    ) -> Vec<PartitionCellId>
    where
        F: FnMut(usize) -> bool,
    {
        let cell_data = self.cell(cell.0).clone();
        if cell_data.length <= 1 || ones_count == 0 || ones_count == cell_data.length {
            return vec![cell];
        }

        let start = cell_data.first;
        let end = start + cell_data.length;
        let mut left = start;
        let mut right = start + cell_data.length - ones_count;

        if ones_count > cell_data.length / 2 {
            while right < end {
                while !is_one(self.elements[right]) {
                    self.elements.swap(right, left);
                    let swapped_left = self.elements[left];
                    let swapped_right = self.elements[right];
                    self.positions[swapped_left] = left;
                    self.positions[swapped_right] = right;
                    left += 1;
                }
                right += 1;
            }
        } else {
            let pivot = right;
            while left < pivot {
                while is_one(self.elements[left]) {
                    self.elements.swap(left, right);
                    let swapped_left = self.elements[left];
                    let swapped_right = self.elements[right];
                    self.positions[swapped_left] = left;
                    self.positions[swapped_right] = right;
                    right += 1;
                }
                left += 1;
            }
        }

        let new_cell = self.split_cell_at(cell.0, cell_data.length - ones_count);
        let new_cell_id = PartitionCellId(new_cell);
        let moved_elements = self.cell_elements(new_cell_id).to_vec();
        for element in moved_elements {
            self.element_to_cell[element] = new_cell;
        }

        vec![cell, new_cell_id]
    }

    /// Splits `cell` by an integer invariant using the same broad strategy as
    /// `bliss` `zplit_cell()`:
    /// binary `0/1` values use the dedicated fast path, larger values below
    /// 256 use an in-place counting sort, and the produced groups remain in
    /// ascending invariant order.
    #[must_use]
    pub(crate) fn split_cell_by_unsigned_invariant_like_bliss<F>(
        &mut self,
        cell: PartitionCellId,
        mut invariant_of: F,
    ) -> Vec<PartitionCellId>
    where
        F: FnMut(usize) -> usize,
    {
        let cell_data = self.cell(cell.0).clone();
        if cell_data.length <= 1 {
            return vec![cell];
        }

        let start = cell_data.first;
        let end = start + cell_data.length;
        let mut invariant_by_element = vec![0usize; self.order];
        let mut max_invariant = 0usize;
        let mut max_invariant_count = 0usize;
        let mut all_equal = true;
        let mut first_invariant = None::<usize>;

        for position in start..end {
            let element = self.elements[position];
            let invariant = invariant_of(element);
            invariant_by_element[element] = invariant;
            if let Some(first) = first_invariant {
                if invariant != first {
                    all_equal = false;
                }
            } else {
                first_invariant = Some(invariant);
            }
            if invariant > max_invariant {
                max_invariant = invariant;
                max_invariant_count = 1;
            } else if invariant == max_invariant {
                max_invariant_count += 1;
            }
        }

        if all_equal {
            return vec![cell];
        }

        if max_invariant == 1 {
            return self.split_cell_by_binary_invariant_like_bliss(
                cell,
                max_invariant_count,
                |element| invariant_by_element[element] == 1,
            );
        }

        if max_invariant < 256 {
            let mut counts = [0usize; 256];
            for position in start..end {
                counts[invariant_by_element[self.elements[position]]] += 1;
            }

            let mut starts = [0usize; 256];
            let mut sum = 0usize;
            for invariant in 0..=max_invariant {
                starts[invariant] = sum;
                sum += counts[invariant];
            }

            for invariant in 0..=max_invariant {
                let mut position = start + starts[invariant];
                for _ in 0..counts[invariant] {
                    loop {
                        let element = self.elements[position];
                        let element_invariant = invariant_by_element[element];
                        if element_invariant == invariant {
                            break;
                        }
                        let swap_position = start + starts[element_invariant];
                        self.elements.swap(position, swap_position);
                        let swapped_left = self.elements[position];
                        let swapped_right = self.elements[swap_position];
                        self.positions[swapped_left] = position;
                        self.positions[swapped_right] = swap_position;
                        starts[element_invariant] += 1;
                        counts[element_invariant] -= 1;
                    }
                    position += 1;
                }
                counts[invariant] = 0;
            }

            return self.split_cell_by_sorted_unsigned_invariants(cell, &invariant_by_element);
        }

        self.split_cell_by_key(cell, |element| invariant_by_element[element])
    }

    fn split_cell_by_sorted_unsigned_invariants(
        &mut self,
        cell: PartitionCellId,
        invariant_by_element: &[usize],
    ) -> Vec<PartitionCellId> {
        let elements = self.cell_elements(cell).to_vec();
        let mut group_lengths = Vec::new();
        let mut current_group_len = 0usize;
        let mut previous_invariant = None::<usize>;
        for element in elements {
            let invariant = invariant_by_element[element];
            if previous_invariant.is_some() && previous_invariant != Some(invariant) {
                group_lengths.push(current_group_len);
                current_group_len = 0;
            }
            previous_invariant = Some(invariant);
            current_group_len += 1;
        }
        group_lengths.push(current_group_len);

        if group_lengths.len() == 1 {
            return vec![cell];
        }

        let mut produced_cells = Vec::with_capacity(group_lengths.len());
        produced_cells.push(cell);
        let mut left_cell = cell.0;
        let mut consumed = 0usize;
        for split_index in 0..group_lengths.len() - 1 {
            consumed += group_lengths[split_index];
            let right_cell = self.split_cell_at(left_cell, consumed);
            produced_cells.push(PartitionCellId(right_cell));
            left_cell = right_cell;
            consumed = 0;
        }

        for partition_cell in &produced_cells[1..] {
            let moved_elements = self.cell_elements(*partition_cell).to_vec();
            for element in moved_elements {
                self.element_to_cell[element] = partition_cell.0;
            }
        }

        produced_cells
    }

    fn undo_split(&mut self, record: SplitRecord) {
        let left_id = record.left_cell;
        let right_id = record.right_cell;
        let right_next = self.cell(right_id).next;
        let right_first = self.cell(right_id).first;
        let right_length = self.cell(right_id).length;

        if self.cell(left_id).is_unit() {
            self.discrete_cell_count = self.discrete_cell_count.saturating_sub(1);
        }
        if self.cell(right_id).is_unit() {
            self.discrete_cell_count = self.discrete_cell_count.saturating_sub(1);
        }

        for position in right_first..right_first + right_length {
            let element = self.elements[position];
            self.element_to_cell[element] = left_id;
        }

        {
            let left_cell = self.cell_mut(left_id);
            left_cell.length += right_length;
            left_cell.next = right_next;
            left_cell.prev_non_singleton = record.prev_non_singleton;
            left_cell.next_non_singleton = record.next_non_singleton;
        }

        if let Some(next_id) = right_next {
            self.cell_mut(next_id).prev = Some(left_id);
        }

        if let Some(prev_non_singleton) = record.prev_non_singleton {
            self.cell_mut(prev_non_singleton).next_non_singleton = Some(left_id);
        } else {
            self.first_non_singleton_cell = Some(left_id);
        }

        if let Some(next_non_singleton) = record.next_non_singleton {
            self.cell_mut(next_non_singleton).prev_non_singleton = Some(left_id);
        }

        self.cells[right_id] = PartitionCell::inactive();
        self.free_cells.push(right_id);
    }

    #[must_use]
    fn split_cell_at(&mut self, cell_id: usize, first_half_size: usize) -> usize {
        let original = self.cell(cell_id).clone();
        assert!(first_half_size > 0, "split point must leave a non-empty left half");
        assert!(first_half_size < original.length, "split point must leave a non-empty right half");

        let new_cell_id = self.free_cells.pop().expect("no free cell slots remain");
        let new_cell = PartitionCell {
            first: original.first + first_half_size,
            length: original.length - first_half_size,
            prev: Some(cell_id),
            next: original.next,
            prev_non_singleton: None,
            next_non_singleton: None,
            split_level: self.split_trail.len() + 1,
            component_level: original.component_level,
        };
        self.cells[new_cell_id] = new_cell;

        {
            let left_cell = self.cell_mut(cell_id);
            left_cell.length = first_half_size;
            left_cell.next = Some(new_cell_id);
        }

        if let Some(next_id) = original.next {
            self.cell_mut(next_id).prev = Some(new_cell_id);
        }

        self.split_trail.push(SplitRecord {
            left_cell: cell_id,
            right_cell: new_cell_id,
            prev_non_singleton: original.prev_non_singleton,
            next_non_singleton: original.next_non_singleton,
        });

        if self.cell(new_cell_id).is_non_singleton() {
            let next_non_singleton = original.next_non_singleton;
            {
                let right_cell = self.cell_mut(new_cell_id);
                right_cell.prev_non_singleton = Some(cell_id);
                right_cell.next_non_singleton = next_non_singleton;
            }
            if let Some(next_id) = next_non_singleton {
                self.cell_mut(next_id).prev_non_singleton = Some(new_cell_id);
            }
            self.cell_mut(cell_id).next_non_singleton = Some(new_cell_id);
        } else {
            self.discrete_cell_count += 1;
        }

        if self.cell(cell_id).is_unit() {
            self.discrete_cell_count += 1;
            let previous = original.prev_non_singleton;
            let next = self.cell(cell_id).next_non_singleton;

            if let Some(previous_id) = previous {
                self.cell_mut(previous_id).next_non_singleton = next;
            } else {
                self.first_non_singleton_cell = next;
            }
            if let Some(next_id) = next {
                self.cell_mut(next_id).prev_non_singleton = previous;
            }

            let left_cell = self.cell_mut(cell_id);
            left_cell.prev_non_singleton = None;
            left_cell.next_non_singleton = None;
        }

        new_cell_id
    }

    fn assert_valid_element(&self, element: usize) {
        assert!(
            element < self.order,
            "element index {element} is out of bounds for order {}",
            self.order
        );
    }

    #[must_use]
    fn cell(&self, cell_id: usize) -> &PartitionCell {
        let cell = self.cells.get(cell_id).expect("cell id must refer to an allocated cell slot");
        assert!(cell.is_active(), "cell id {cell_id} must refer to an active cell");
        cell
    }

    #[must_use]
    fn cell_mut(&mut self, cell_id: usize) -> &mut PartitionCell {
        let cell =
            self.cells.get_mut(cell_id).expect("cell id must refer to an allocated cell slot");
        assert!(cell.is_active(), "cell id {cell_id} must refer to an active cell");
        cell
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_cell_by_tail_elements_in_order_reorders_fully_touched_cells() {
        let mut partition = BacktrackableOrderedPartition::new(2);
        let cell = partition.cell_of(0);

        let produced = partition.split_cell_by_tail_elements_in_order(cell, &[0, 1]);

        assert_eq!(produced, vec![cell]);
        assert_eq!(partition.cell_elements(cell), &[1, 0]);
    }
}
