//! Enumeration for the LAPMOD assignment state, including the variants of:
//!
//! * `Unassigned`
//! * `Assigned`
//! * `Conflict`

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// The LAPMOD assignment state.
pub enum AssignmentState<T> {
    /// The assignment is unassigned.
    Unassigned,
    /// The assignment is assigned.
    Assigned(T),
    /// The assignment is in conflict.
    Conflict(T),
}

impl<T> AssignmentState<T> {
    /// Returns true if the assignment is unassigned.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::traits::algorithms::AssignmentState;
    ///
    /// let state: AssignmentState<usize> = AssignmentState::Unassigned;
    /// assert!(state.is_unassigned());
    /// ```
    #[inline]
    pub fn is_unassigned(&self) -> bool {
        matches!(self, AssignmentState::Unassigned)
    }

    /// Returns true if the assignment is assigned.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::traits::algorithms::AssignmentState;
    ///
    /// let state: AssignmentState<usize> = AssignmentState::Assigned(1);
    /// assert!(state.is_assigned());
    /// ```
    #[inline]
    pub fn is_assigned(&self) -> bool {
        matches!(self, AssignmentState::Assigned(_))
    }
}

#[cfg(test)]
mod tests {
    use super::AssignmentState;

    #[test]
    fn test_state_predicates() {
        let assigned: AssignmentState<usize> = AssignmentState::Assigned(7);
        assert!(assigned.is_assigned());
        assert!(!assigned.is_unassigned());

        let unassigned: AssignmentState<usize> = AssignmentState::Unassigned;
        assert!(!unassigned.is_assigned());
        assert!(unassigned.is_unassigned());

        let conflict: AssignmentState<usize> = AssignmentState::Conflict(3);
        assert!(!conflict.is_assigned());
        assert!(!conflict.is_unassigned());
    }
}
