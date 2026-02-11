/// Trait defining the conversion into `usize`.
///
/// # Examples
///
/// ```
/// use geometric_traits::traits::IntoUsize;
///
/// assert_eq!(42u8.into_usize(), 42usize);
/// assert_eq!(1000u16.into_usize(), 1000usize);
/// assert_eq!(100_000u32.into_usize(), 100_000usize);
/// assert_eq!(42usize.into_usize(), 42usize);
/// ```
pub trait IntoUsize {
    /// Converts the value into `usize`.
    fn into_usize(self) -> usize;
}

impl IntoUsize for u8 {
    #[inline]
    fn into_usize(self) -> usize {
        self.into()
    }
}

impl IntoUsize for u16 {
    #[inline]
    fn into_usize(self) -> usize {
        self.into()
    }
}

impl IntoUsize for u32 {
    #[inline]
    fn into_usize(self) -> usize {
        self as usize
    }
}

#[cfg(target_pointer_width = "64")]
impl IntoUsize for u64 {
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    fn into_usize(self) -> usize {
        self as usize
    }
}

impl IntoUsize for usize {
    #[inline]
    fn into_usize(self) -> usize {
        self
    }
}
