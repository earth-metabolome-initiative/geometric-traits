//! [`BitSet`], a fixed-length set of indices stored as one bit per index.

use alloc::{vec, vec::Vec};
use core::{iter::FusedIterator, ops::BitAndAssign};

// `usize::BITS` is at most 128, so the widening cast loses nothing.
const WORD_BITS: usize = usize::BITS as usize;

/// A fixed-length set over the indices `0..len`, one bit per index.
///
/// Index `i` lives in word `i / usize::BITS` at bit `i % usize::BITS`. Bits at
/// or past `len` in the last word are always zero, so [`count_ones`], equality
/// and hashing depend only on the indices in the set.
///
/// [`count_ones`]: BitSet::count_ones
///
/// # Examples
///
/// ```
/// use geometric_traits::prelude::*;
///
/// let mut set = BitSet::zeros(10);
/// assert!(set.insert(3));
/// assert!(set.insert(7));
/// assert!(!set.insert(7));
/// assert_eq!(set.iter_ones().collect::<Vec<_>>(), vec![3, 7]);
/// assert_eq!(set.and_count(&BitSet::ones(5)), 1);
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct BitSet {
    words: Vec<usize>,
    len: usize,
}

impl BitSet {
    /// Creates a set of length `len` with no index in it.
    #[inline]
    #[must_use]
    pub fn zeros(len: usize) -> Self {
        Self { words: vec![0; len.div_ceil(WORD_BITS)], len }
    }

    /// Creates a set of length `len` holding every index in `0..len`.
    #[inline]
    #[must_use]
    pub fn ones(len: usize) -> Self {
        let mut words = vec![usize::MAX; len.div_ceil(WORD_BITS)];
        if let Some(last) = words.last_mut() {
            *last &= tail_mask(len);
        }
        Self { words, len }
    }

    /// Returns the number of indices the set ranges over.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the set ranges over no index at all.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns whether `index` is in the set. Indices at or past `len` are
    /// never in it.
    #[inline]
    #[must_use]
    pub fn contains(&self, index: usize) -> bool {
        Self::words_contain(&self.words, index)
    }

    /// Membership test on the words of a set, for loops that hold the word
    /// slice in a local so it is not reloaded through `&BitSet` after every
    /// store.
    #[inline]
    pub(crate) fn words_contain(words: &[usize], index: usize) -> bool {
        words.get(index / WORD_BITS).is_some_and(|&word| word >> (index % WORD_BITS) & 1 == 1)
    }

    /// Adds `index` to the set and returns whether it was absent.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    #[inline]
    pub fn insert(&mut self, index: usize) -> bool {
        let (word, mask) = self.locate(index);
        let absent = self.words[word] & mask == 0;
        self.words[word] |= mask;
        absent
    }

    /// Removes `index` from the set and returns whether it was present.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    #[inline]
    pub fn remove(&mut self, index: usize) -> bool {
        let (word, mask) = self.locate(index);
        let present = self.words[word] & mask != 0;
        self.words[word] &= !mask;
        present
    }

    /// Returns the number of indices in the set.
    #[inline]
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|&word| popcount(word)).sum()
    }

    /// Returns the number of indices in both `self` and `other`.
    ///
    /// Indices past the shorter of the two lengths count as absent.
    #[inline]
    #[must_use]
    pub fn and_count(&self, other: &Self) -> usize {
        self.words.iter().zip(&other.words).map(|(&left, &right)| popcount(left & right)).sum()
    }

    /// Returns the largest index in the set.
    #[inline]
    #[must_use]
    pub fn last_one(&self) -> Option<usize> {
        let word = self.words.iter().rposition(|&word| word != 0)?;
        Some(word * WORD_BITS + highest_one(self.words[word]))
    }

    /// Iterates the indices in the set in increasing order.
    #[inline]
    #[must_use]
    pub fn iter_ones(&self) -> BitSetOnes<'_> {
        BitSetOnes::new(&self.words)
    }

    /// Returns the words backing the set, least significant bit first.
    #[inline]
    #[must_use]
    pub fn as_words(&self) -> &[usize] {
        &self.words
    }

    #[inline]
    fn locate(&self, index: usize) -> (usize, usize) {
        assert!(index < self.len, "index {index} out of range for a BitSet of length {}", self.len);
        (index / WORD_BITS, 1 << (index % WORD_BITS))
    }
}

impl BitAndAssign<&BitSet> for BitSet {
    /// Keeps only the indices also in `rhs`. Indices past `rhs.len()` are
    /// removed, and the length of `self` is unchanged.
    #[inline]
    fn bitand_assign(&mut self, rhs: &BitSet) {
        let shared = self.words.len().min(rhs.words.len());
        for (left, &right) in self.words[..shared].iter_mut().zip(&rhs.words[..shared]) {
            *left &= right;
        }
        self.words[shared..].fill(0);
    }
}

/// Iterator over the indices of a [`BitSet`], from either end.
#[derive(Clone, Debug)]
pub struct BitSetOnes<'a> {
    words: &'a [usize],
    /// Word the front cursor is in.
    front: usize,
    /// Bits of `words[front]` not yet yielded. Once `front == back`, this
    /// holds every remaining bit.
    front_bits: usize,
    /// Word the back cursor is in.
    back: usize,
    /// Bits of `words[back]` not yet yielded. Zero once `front == back`.
    back_bits: usize,
}

impl<'a> BitSetOnes<'a> {
    #[inline]
    fn new(words: &'a [usize]) -> Self {
        let back = words.len().saturating_sub(1);
        let front_bits = words.first().copied().unwrap_or(0);
        let back_bits = if back == 0 { 0 } else { words[back] };
        Self { words, front: 0, front_bits, back, back_bits }
    }
}

impl Iterator for BitSetOnes<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        loop {
            if self.front_bits != 0 {
                let bit = lowest_one(self.front_bits);
                self.front_bits &= self.front_bits - 1;
                return Some(self.front * WORD_BITS + bit);
            }
            if self.front >= self.back {
                return None;
            }
            self.front += 1;
            if self.front == self.back {
                self.front_bits = core::mem::take(&mut self.back_bits);
            } else {
                self.front_bits = self.words[self.front];
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let between = if self.back > self.front {
            self.words[self.front + 1..self.back].iter().map(|&word| popcount(word)).sum()
        } else {
            0
        };
        let remaining = popcount(self.front_bits) + between + popcount(self.back_bits);
        (remaining, Some(remaining))
    }
}

impl DoubleEndedIterator for BitSetOnes<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<usize> {
        loop {
            if self.back_bits != 0 {
                let bit = highest_one(self.back_bits);
                self.back_bits &= !(1 << bit);
                return Some(self.back * WORD_BITS + bit);
            }
            if self.back <= self.front {
                if self.front_bits == 0 {
                    return None;
                }
                let bit = highest_one(self.front_bits);
                self.front_bits &= !(1 << bit);
                return Some(self.front * WORD_BITS + bit);
            }
            self.back -= 1;
            if self.back != self.front {
                self.back_bits = self.words[self.back];
            }
        }
    }
}

impl ExactSizeIterator for BitSetOnes<'_> {}

impl FusedIterator for BitSetOnes<'_> {}

/// Mask of the bits of the last word that lie inside a set of length `len`.
fn tail_mask(len: usize) -> usize {
    match len % WORD_BITS {
        0 => usize::MAX,
        used => (1 << used) - 1,
    }
}

#[inline]
fn popcount(word: usize) -> usize {
    // At most `usize::BITS`, so the widening cast loses nothing.
    word.count_ones() as usize
}

#[inline]
fn lowest_one(word: usize) -> usize {
    debug_assert_ne!(word, 0, "lowest_one of an empty word");
    // Below `usize::BITS` for a nonzero word, so the widening cast loses
    // nothing.
    word.trailing_zeros() as usize
}

#[inline]
fn highest_one(word: usize) -> usize {
    debug_assert_ne!(word, 0, "highest_one of an empty word");
    // Below `usize::BITS` for a nonzero word, so the widening cast loses
    // nothing.
    WORD_BITS - 1 - word.leading_zeros() as usize
}
