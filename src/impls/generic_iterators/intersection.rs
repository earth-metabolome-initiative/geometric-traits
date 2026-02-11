use core::cmp::Ordering;

/// Iterator that returns the intersection of two sorted iterators.
#[derive(Clone)]
pub struct Intersection<I, J>
where
    I: Iterator,
    J: Iterator<Item = I::Item>,
    I::Item: Ord,
{
    iter1: I,
    iter2: J,
    item1: Option<I::Item>,
    item2: Option<J::Item>,
    item1_back: Option<I::Item>,
    item2_back: Option<J::Item>,
}

impl<I, J> Intersection<I, J>
where
    I: Iterator,
    J: Iterator<Item = I::Item>,
    I::Item: Ord,
{
    /// Creates a new `Intersection` iterator.
    pub fn new(mut iter1: I, mut iter2: J) -> Self {
        let item1 = iter1.next();
        let item2 = iter2.next();
        Self { iter1, iter2, item1, item2, item1_back: None, item2_back: None }
    }
}

impl<I, J> Iterator for Intersection<I, J>
where
    I: Iterator,
    J: Iterator<Item = I::Item>,
    I::Item: Ord,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Ensure front candidates are present
            if self.item1.is_none() {
                self.item1 = self.iter1.next().or_else(|| self.item1_back.take());
            }
            if self.item2.is_none() {
                self.item2 = self.iter2.next().or_else(|| self.item2_back.take());
            }

            match (&self.item1, &self.item2) {
                (Some(val1), Some(val2)) => {
                    // Check for crossing with back candidates
                    if let Some(ref back1) = self.item1_back
                        && val1 > back1
                    {
                        return None;
                    }
                    if let Some(ref back2) = self.item2_back
                        && val2 > back2
                    {
                        return None;
                    }

                    match val1.cmp(val2) {
                        Ordering::Less => self.item1 = None,    // Discard val1
                        Ordering::Greater => self.item2 = None, // Discard val2
                        Ordering::Equal => {
                            let res = self.item1.take();
                            self.item2 = None; // Consume both
                            return res;
                        }
                    }
                }
                _ => return None,
            }
        }
    }
}

impl<I, J> DoubleEndedIterator for Intersection<I, J>
where
    I: DoubleEndedIterator,
    J: DoubleEndedIterator<Item = I::Item>,
    I::Item: Ord,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            // Ensure back candidates are present
            if self.item1_back.is_none() {
                self.item1_back = self.iter1.next_back().or_else(|| self.item1.take());
            }
            if self.item2_back.is_none() {
                self.item2_back = self.iter2.next_back().or_else(|| self.item2.take());
            }

            match (&self.item1_back, &self.item2_back) {
                (Some(val1), Some(val2)) => {
                    // Check for crossing with front candidates
                    if let Some(ref front1) = self.item1
                        && front1 > val1
                    {
                        return None;
                    }
                    if let Some(ref front2) = self.item2
                        && front2 > val2
                    {
                        return None;
                    }

                    match val1.cmp(val2) {
                        Ordering::Greater => self.item1_back = None, // val1 > val2, discard val1
                        Ordering::Less => self.item2_back = None,    // val2 > val1, discard val2
                        Ordering::Equal => {
                            let res = self.item1_back.take();
                            self.item2_back = None;
                            return res;
                        }
                    }
                }
                _ => return None,
            }
        }
    }
}

/// Trait for sorted iterators.
pub trait SortedIterator: Iterator {
    /// Returns an iterator over the intersection of two sorted iterators.
    fn sorted_intersection<J>(self, other: J) -> Intersection<Self, J>
    where
        J: Iterator<Item = Self::Item>,
        Self::Item: Ord,
        Self: Sized,
    {
        Intersection::new(self, other)
    }
}

impl<I: Iterator> SortedIterator for I {}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    #[test]
    fn test_intersection_empty_iterators() {
        let iter1 = core::iter::empty::<i32>();
        let iter2 = core::iter::empty::<i32>();
        let result: Vec<i32> = Intersection::new(iter1, iter2).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_intersection_first_empty() {
        let iter1 = core::iter::empty::<i32>();
        let iter2 = [1, 2, 3].into_iter();
        let result: Vec<i32> = Intersection::new(iter1, iter2).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_intersection_second_empty() {
        let iter1 = [1, 2, 3].into_iter();
        let iter2 = core::iter::empty::<i32>();
        let result: Vec<i32> = Intersection::new(iter1, iter2).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_intersection_no_common_elements() {
        let iter1 = [1, 3, 5].into_iter();
        let iter2 = [2, 4, 6].into_iter();
        let result: Vec<i32> = Intersection::new(iter1, iter2).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_intersection_all_common() {
        let iter1 = [1, 2, 3].into_iter();
        let iter2 = [1, 2, 3].into_iter();
        let result: Vec<i32> = Intersection::new(iter1, iter2).collect();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_intersection_some_common() {
        let iter1 = [1, 2, 3, 4, 5].into_iter();
        let iter2 = [2, 4, 6].into_iter();
        let result: Vec<i32> = Intersection::new(iter1, iter2).collect();
        assert_eq!(result, vec![2, 4]);
    }

    #[test]
    fn test_intersection_single_element_match() {
        let iter1 = [1, 2, 3].into_iter();
        let iter2 = [2].into_iter();
        let result: Vec<i32> = Intersection::new(iter1, iter2).collect();
        assert_eq!(result, vec![2]);
    }

    #[test]
    fn test_intersection_different_lengths() {
        let iter1 = [1, 2].into_iter();
        let iter2 = [1, 2, 3, 4, 5].into_iter();
        let result: Vec<i32> = Intersection::new(iter1, iter2).collect();
        assert_eq!(result, vec![1, 2]);
    }

    #[test]
    fn test_sorted_iterator_trait() {
        let iter1 = [1, 2, 3, 4, 5].into_iter();
        let iter2 = [2, 4].into_iter();
        let result: Vec<i32> = iter1.sorted_intersection(iter2).collect();
        assert_eq!(result, vec![2, 4]);
    }

    #[test]
    fn test_intersection_double_ended_next_back() {
        let iter1 = [1, 2, 3, 4, 5].into_iter();
        let iter2 = [2, 3, 4].into_iter();
        let mut intersection = Intersection::new(iter1, iter2);
        assert_eq!(intersection.next_back(), Some(4));
        assert_eq!(intersection.next_back(), Some(3));
        assert_eq!(intersection.next_back(), Some(2));
        assert_eq!(intersection.next_back(), None);
    }

    #[test]
    fn test_intersection_double_ended_mixed() {
        let iter1 = [1, 2, 3, 4, 5].into_iter();
        let iter2 = [1, 2, 3, 4, 5].into_iter();
        let mut intersection = Intersection::new(iter1, iter2);
        assert_eq!(intersection.next(), Some(1));
        assert_eq!(intersection.next_back(), Some(5));
        assert_eq!(intersection.next(), Some(2));
        assert_eq!(intersection.next_back(), Some(4));
        assert_eq!(intersection.next(), Some(3));
        assert_eq!(intersection.next(), None);
        assert_eq!(intersection.next_back(), None);
    }
}
