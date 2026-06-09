// ============ HELPERS ============

#[inline(always)]
pub fn pow2(k: usize) -> usize {
    1usize << k
}

#[inline(always)]
pub fn bit_at(x: usize, pos: usize) -> u8 {
    ((x >> pos) & 1) as u8
}

// #[inline(always)]
// pub fn remove_bit(i: usize, bitpos: usize) -> usize {
//     let low_mask = (1usize << bitpos) - 1;
//     (i & low_mask) | ((i >> (bitpos + 1)) << bitpos)
// }
