pub(crate) fn maximum_matching_size(order: usize, edges: &[(usize, usize)]) -> usize {
    fn solve(mask: usize, adjacency: &[usize], memo: &mut [Option<usize>]) -> usize {
        if mask == 0 {
            return 0;
        }
        if let Some(cached) = memo[mask] {
            return cached;
        }

        let i = mask.trailing_zeros() as usize;
        let rest = mask & !(1usize << i);
        let mut best = solve(rest, adjacency, memo);
        let mut candidates = rest & adjacency[i];

        while candidates != 0 {
            let j_bit = candidates & candidates.wrapping_neg();
            candidates &= candidates - 1;
            best = best.max(1 + solve(rest & !j_bit, adjacency, memo));
        }

        memo[mask] = Some(best);
        best
    }

    assert!(
        order < usize::BITS as usize,
        "maximum-matching oracle only supports graphs smaller than the machine word size"
    );

    let mut adjacency = vec![0usize; order];
    for &(u, v) in edges {
        adjacency[u] |= 1usize << v;
        adjacency[v] |= 1usize << u;
    }

    let mut memo = vec![None; 1usize << order];
    solve((1usize << order) - 1, &adjacency, &mut memo)
}
