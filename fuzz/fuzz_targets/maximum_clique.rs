//! Fuzz harness for the maximum clique algorithm.
//!
//! Verifies invariants:
//! - All returned cliques have the same size.
//! - All returned cliques are actual cliques (pairwise adjacent).
//! - No duplicates.
//! - ω(G) matches brute-force for small n.
//! - Single-mode result is a subset of enumerate-mode results.

use geometric_traits::{impls::BitSquareMatrix, prelude::*};
use honggfuzz::fuzz;

fn main() {
    loop {
        fuzz!(|data: &[u8]| {
            if data.is_empty() {
                return;
            }
            let n = (data[0] % 10) as usize + 1; // 1..10
            let max_edges = n * (n - 1) / 2;
            if data.len() < 1 + max_edges {
                return;
            }

            // Build a random symmetric graph.
            let mut edges = Vec::new();
            let mut idx = 1;
            for u in 0..n {
                for v in (u + 1)..n {
                    if data[idx] & 1 != 0 {
                        edges.push((u, v));
                    }
                    idx += 1;
                }
            }
            let g = BitSquareMatrix::from_symmetric_edges(n, edges);

            // Enumerate all maximum cliques.
            let all = g.all_maximum_cliques();
            let one = g.maximum_clique();

            // 1. All cliques must have the same size.
            if !all.is_empty() {
                let omega = all[0].len();
                for c in &all {
                    assert_eq!(c.len(), omega, "clique size mismatch");
                }
                // Single-mode result must have the same size.
                assert_eq!(one.len(), omega, "single-mode size mismatch");
            }

            // 2. All cliques must be actual cliques.
            for c in &all {
                for (i, &u) in c.iter().enumerate() {
                    for &v in &c[i + 1..] {
                        assert!(
                            g.has_entry(u, v) && g.has_entry(v, u),
                            "not a clique: {u} and {v} not adjacent"
                        );
                    }
                }
            }
            // Also verify the single-mode result.
            for (i, &u) in one.iter().enumerate() {
                for &v in &one[i + 1..] {
                    assert!(
                        g.has_entry(u, v) && g.has_entry(v, u),
                        "single not a clique: {u} and {v} not adjacent"
                    );
                }
            }

            // 3. No duplicates (cliques are sorted, so just check sorted outer).
            {
                let mut sorted_all: Vec<Vec<usize>> = all.clone();
                sorted_all.sort();
                sorted_all.dedup();
                assert_eq!(sorted_all.len(), all.len(), "duplicate cliques");
            }

            // 4. Single-mode result must appear in enumerate-mode results.
            if !one.is_empty() {
                let mut sorted_one = one.clone();
                sorted_one.sort();
                assert!(all.contains(&sorted_one), "single result not in all");
            }

            // 5. For small n, brute-force verify ω(G) and all maximum cliques.
            if n <= 8 {
                let bf_cliques = brute_force_all_maximum_cliques(&g, n);
                let omega = if all.is_empty() { 0 } else { all[0].len() };
                let omega_bf = if bf_cliques.is_empty() { 0 } else { bf_cliques[0].len() };
                assert_eq!(omega, omega_bf, "omega mismatch vs brute force");

                // 6. The set of all maximum cliques must match exactly.
                let mut sorted_all: Vec<Vec<usize>> = all.clone();
                sorted_all.sort();
                let mut sorted_bf = bf_cliques;
                sorted_bf.sort();
                assert_eq!(sorted_all, sorted_bf, "all_maximum_cliques mismatch vs brute force");
            }

            // 7. Partition-aware maximum clique.
            // Generate a random partition from remaining fuzzer data.
            if idx < data.len() && n > 0 {
                let num_groups = (data[idx] % (n as u8).max(1)).max(1) as usize;
                let partition: Vec<usize> = (0..n)
                    .map(|i| {
                        let byte_idx = idx + 1 + i;
                        if byte_idx < data.len() {
                            data[byte_idx] as usize % num_groups
                        } else {
                            i % num_groups
                        }
                    })
                    .collect();

                let part_all = g.all_maximum_cliques_with_partition(&partition);
                let part_one = g.maximum_clique_with_partition(&partition);

                // 7a. All partition-aware cliques must be actual cliques.
                for c in &part_all {
                    for (i, &u) in c.iter().enumerate() {
                        for &v in &c[i + 1..] {
                            assert!(
                                g.has_entry(u, v),
                                "partition clique not valid: {u}-{v} not adjacent"
                            );
                        }
                    }
                }

                // 7b. All partition-aware cliques must respect the partition.
                for c in &part_all {
                    let mut groups_used = Vec::new();
                    for &v in c {
                        let grp = partition[v];
                        assert!(
                            !groups_used.contains(&grp),
                            "partition violation: group {grp} appears twice in {c:?}"
                        );
                        groups_used.push(grp);
                    }
                }

                // 7c. Partition-aware max clique size ≤ regular max clique size.
                if !part_all.is_empty() && !all.is_empty() {
                    assert!(
                        part_all[0].len() <= all[0].len(),
                        "partition clique larger than regular"
                    );
                }

                // 7d. Single-mode matches enumerate-mode size.
                if !part_all.is_empty() {
                    assert_eq!(
                        part_one.len(),
                        part_all[0].len(),
                        "partition single vs all size mismatch"
                    );
                }

                // 7e. For small n, brute-force verify partition-aware cliques.
                if n <= 8 {
                    let bf_part = brute_force_partition_cliques(&g, n, &partition);
                    let omega_p = if part_all.is_empty() { 0 } else { part_all[0].len() };
                    let omega_bf_p = if bf_part.is_empty() { 0 } else { bf_part[0].len() };
                    assert_eq!(omega_p, omega_bf_p, "partition omega mismatch vs brute force");

                    let mut sorted_part: Vec<Vec<usize>> = part_all;
                    sorted_part.sort();
                    let mut sorted_bf_p = bf_part;
                    sorted_bf_p.sort();
                    assert_eq!(
                        sorted_part, sorted_bf_p,
                        "partition all_maximum_cliques mismatch vs brute force"
                    );
                }
            }
        });
    }
}

/// Brute-force enumeration of all maximum cliques respecting a partition
/// constraint (at most one vertex per group).
fn brute_force_partition_cliques(
    g: &BitSquareMatrix,
    n: usize,
    partition: &[usize],
) -> Vec<Vec<usize>> {
    let mut best_size = if n > 0 { 1 } else { 0 };
    let mut cliques: Vec<Vec<usize>> = Vec::new();

    for mask in 1u32..(1u32 << n) {
        let bits: Vec<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();
        let sz = bits.len();
        if sz < best_size {
            continue;
        }
        // Check partition constraint.
        let mut groups_seen = Vec::new();
        let mut partition_ok = true;
        for &v in &bits {
            let g = partition[v];
            if groups_seen.contains(&g) {
                partition_ok = false;
                break;
            }
            groups_seen.push(g);
        }
        if !partition_ok {
            continue;
        }
        // Check clique.
        let mut is_clique = true;
        'outer: for (i, &u) in bits.iter().enumerate() {
            for &v in &bits[i + 1..] {
                if !g.has_entry(u, v) {
                    is_clique = false;
                    break 'outer;
                }
            }
        }
        if is_clique {
            if sz > best_size {
                best_size = sz;
                cliques.clear();
            }
            cliques.push(bits);
        }
    }

    if cliques.is_empty() && n > 0 {
        cliques = (0..n).map(|v| vec![v]).collect();
    }

    cliques
}

/// Brute-force enumeration of all maximum cliques by checking all 2^n subsets.
fn brute_force_all_maximum_cliques(g: &BitSquareMatrix, n: usize) -> Vec<Vec<usize>> {
    let mut best_size = if n > 0 { 1 } else { 0 };
    let mut cliques: Vec<Vec<usize>> = Vec::new();

    for mask in 1u32..(1u32 << n) {
        let bits: Vec<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();
        let sz = bits.len();
        if sz < best_size {
            continue;
        }
        let mut is_clique = true;
        'outer: for (i, &u) in bits.iter().enumerate() {
            for &v in &bits[i + 1..] {
                if !g.has_entry(u, v) {
                    is_clique = false;
                    break 'outer;
                }
            }
        }
        if is_clique {
            if sz > best_size {
                best_size = sz;
                cliques.clear();
            }
            cliques.push(bits);
        }
    }

    // For n > 0 with no edges, every single vertex is a max clique of size 1.
    if cliques.is_empty() && n > 0 {
        cliques = (0..n).map(|v| vec![v]).collect();
    }

    cliques
}
