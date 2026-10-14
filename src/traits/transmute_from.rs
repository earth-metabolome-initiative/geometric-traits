//! Trait for unsafe transmutation.

/// Trait for unsafe transmutation.
pub trait TransmuteFrom<T> {
    /// Transmutes the input into the output.
    ///
    /// # Safety
    /// The caller must ensure that the input is valid for the output type.
    unsafe fn transmute_from(from: T) -> Self;
}
