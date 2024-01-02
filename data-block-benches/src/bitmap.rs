pub struct BitVecOnesIter<'a> {
    bits: &'a Vec<u64>,
    current: u64,
    index: usize,
}

impl<'a> BitVecOnesIter<'a> {
    pub fn new(bits: &'a Vec<u64>) -> Self {
        if bits.is_empty() {
            Self {
                bits,
                current: 0,
                index: 0,
            }
        } else {
            Self {
                current: unsafe { *bits.get_unchecked(0) },
                bits,
                index: 0,
            }
        }
    }
}

impl<'a> Iterator for BitVecOnesIter<'a> {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.current == 0 {
            self.index += 1;
            if self.index == self.bits.len() {
                return None;
            }

            self.current = unsafe { *self.bits.get_unchecked(self.index) };
        }

        let index = (self.index * 64) + self.current.trailing_zeros() as usize;
        self.current &= self.current - 1;

        Some(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iter_bitvec() {
        let v = vec![u64::MAX; 16];
        let iter = BitVecOnesIter::new(&v);
        let indexes = iter.collect::<Vec<usize>>();
        assert_eq!(indexes, (0..1024).collect::<Vec<_>>());
    }
}
