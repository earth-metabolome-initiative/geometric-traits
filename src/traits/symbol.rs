//! Trait defining an unique symbolic identifier.

/// Trait defining an unique symbolic identifier.
///
/// This is a marker trait that combines common requirements for symbols:
/// `PartialEq`, `Eq`, `Clone`, `Hash`, and `Debug`.
///
/// # Examples
///
/// ```
/// use geometric_traits::traits::Symbol;
///
/// // Common types automatically implement Symbol
/// fn accepts_symbol<S: Symbol>(s: S) -> S {
///     s.clone()
/// }
///
/// assert_eq!(accepts_symbol(42_u32), 42);
/// assert_eq!(accepts_symbol("hello"), "hello");
/// assert_eq!(accepts_symbol(String::from("world")), "world");
/// ```
pub trait Symbol: PartialEq + Eq + Clone + core::hash::Hash + core::fmt::Debug {}

impl<T> Symbol for T where T: PartialEq + Eq + Clone + core::hash::Hash + core::fmt::Debug {}
