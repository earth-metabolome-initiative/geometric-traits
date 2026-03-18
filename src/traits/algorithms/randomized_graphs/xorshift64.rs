//! XorShift64 pseudo-random number generator.
#![cfg(feature = "alloc")]

/// Struct for storing the `XorShift64` state.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct XorShift64(u64);

impl XorShift64 {
    /// Normalizes a seed so that zero maps to a fixed non-zero state.
    ///
    /// `XorShift64` with a zero state remains zero forever.
    /// This maps seed 0 to a fixed non-zero constant to avoid degenerate output.
    #[inline]
    #[must_use]
    pub fn normalize_seed(seed: u64) -> u64 {
        if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed }
    }
}

impl From<u64> for XorShift64 {
    #[inline]
    fn from(state: u64) -> Self {
        Self(state)
    }
}

impl Iterator for XorShift64 {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        Some(x)
    }
}
