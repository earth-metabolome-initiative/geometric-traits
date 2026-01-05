/// Trait defining the conversion into `usize`.
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
