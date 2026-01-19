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
