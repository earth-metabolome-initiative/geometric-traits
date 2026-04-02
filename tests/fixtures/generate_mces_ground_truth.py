#!/usr/bin/env python3
"""Generate MCES ground truth test cases from RDKit RASCAL.

Converts SMILES pairs from RDKit's mces_catch.cpp test suite into JSON
with explicit graph edge lists, atom types, bond types, aromatic-ring
contexts, and expected MCES results. The output is consumed by Rust
integration tests.

Usage:
    uv run --with rdkit python3 tests/fixtures/generate_mces_ground_truth.py
"""

import gzip
import json
from rdkit import Chem, rdBase
from rdkit.Chem.rdRascalMCES import FindMCES, RascalOptions


def suppress_rdkit_logs():
    """Keep fixture generation readable by silencing RDKit chatter."""
    rdBase.DisableLog("rdApp.debug")
    rdBase.DisableLog("rdApp.info")
    rdBase.DisableLog("rdApp.warning")


def _normalize_aromatic_ring_smiles(smiles):
    """Normalize aromatic ring signatures to align with RDKit's matching."""
    return smiles.replace("[nH]", "n").replace("[pH]", "p")


def _rdkit_aromatic_ring_contexts_by_bond(mol):
    """Mirror RDKit's extractRings(...)+MolToSmiles ring signatures.

    RASCAL does not use MolFragmentToSmiles on the ring atom/bond subset.
    It clones the molecule, removes atoms outside the ring, then canonicalizes
    that pruned ring molecule with MolToSmiles. For some fused / bridged
    systems this yields a different signature than MolFragmentToSmiles, and the
    MCES bond-pair admission depends on matching those exact signatures.
    """
    aromatic_contexts_by_bond = {bond.GetIdx(): [] for bond in mol.GetBonds()}
    ring_info = mol.GetRingInfo()

    for atom_ring, bond_ring in zip(ring_info.AtomRings(), ring_info.BondRings()):
        if not bond_ring:
            continue

        ring_mol = Chem.RWMol(mol)
        atoms_in_ring = set(atom_ring)
        atoms_to_remove = [
            atom.GetIdx()
            for atom in ring_mol.GetAtoms()
            if atom.GetIdx() not in atoms_in_ring
        ]
        ring_mol.BeginBatchEdit()
        for atom_idx in sorted(atoms_to_remove, reverse=True):
            ring_mol.RemoveAtom(atom_idx)
        ring_mol.CommitBatchEdit()

        signature = _normalize_aromatic_ring_smiles(Chem.MolToSmiles(ring_mol))
        for bond_idx in bond_ring:
            # RDKit's checkRings(..., aromaticRingsMatchOnly=true) compares
            # aromatic bonds against every enclosing ring signature that
            # survives extractRings(...), not only against fully aromatic
            # cycles. Preserve that behavior here.
            if mol.GetBondWithIdx(bond_idx).GetIsAromatic():
                aromatic_contexts_by_bond[bond_idx].append(signature)

    return aromatic_contexts_by_bond


def _canonical_bond_payload(mol):
    """Return bonds in the canonical order used by the Rust fixture loader."""
    aromatic_contexts_by_bond = _rdkit_aromatic_ring_contexts_by_bond(mol)

    payload = []
    for bond in mol.GetBonds():
        src = bond.GetBeginAtomIdx()
        dst = bond.GetEndAtomIdx()
        original_orientation = [src, dst]
        if src > dst:
            src, dst = dst, src
        contexts = sorted(set(aromatic_contexts_by_bond[bond.GetIdx()]))
        payload.append(
            ([src, dst], int(bond.GetBondType()), contexts, original_orientation, bond.GetIdx())
        )

    payload.sort(key=lambda row: (row[0][0], row[0][1], row[1]))
    return payload


def mol_to_graph(smiles, sanitize=True):
    """Convert SMILES to graph representation."""
    if sanitize:
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    n_atoms = mol.GetNumAtoms()
    bond_payload = _canonical_bond_payload(mol)
    edges = [edge for edge, _, _, _, _ in bond_payload]
    bond_types = [bond_type for _, bond_type, _, _, _ in bond_payload]
    aromatic_ring_contexts = [contexts for _, _, contexts, _, _ in bond_payload]
    bond_orientations = [orientation for _, _, _, orientation, _ in bond_payload]
    bond_original_indices = [original_index for _, _, _, _, original_index in bond_payload]

    atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atom_is_aromatic = [atom.GetIsAromatic() for atom in mol.GetAtoms()]
    atom_total_hs = [atom.GetTotalNumHs() for atom in mol.GetAtoms()]

    return {
        "n_atoms": n_atoms,
        "edges": edges,
        "atom_types": atom_types,
        "atom_is_aromatic": atom_is_aromatic,
        "atom_total_hs": atom_total_hs,
        "bond_types": bond_types,
        "aromatic_ring_contexts": aromatic_ring_contexts,
        "bond_orientations": bond_orientations,
        "bond_original_indices": bond_original_indices,
    }


def compute_mces(smiles1, smiles2, **kwargs):
    """Compute MCES between two molecules."""
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    opts = RascalOptions()
    opts.similarityThreshold = kwargs.get("similarityThreshold", 0.0)
    opts.returnEmptyMCES = True
    opts.timeout = kwargs.get("timeout", 60)
    if kwargs.get("ignoreBondOrders", False):
        opts.ignoreBondOrders = True
    if "ringMatchesRingOnly" in kwargs:
        opts.ringMatchesRingOnly = kwargs["ringMatchesRingOnly"]
    if "completeAromaticRings" in kwargs:
        opts.completeAromaticRings = kwargs["completeAromaticRings"]
    if "singleLargestFrag" in kwargs:
        opts.singleLargestFrag = kwargs["singleLargestFrag"]
    if "allBestMCESs" in kwargs:
        opts.allBestMCESs = kwargs["allBestMCESs"]
    if "exactConnectionsMatch" in kwargs:
        opts.exactConnectionsMatch = kwargs["exactConnectionsMatch"]
    if "equivalentAtoms" in kwargs:
        opts.equivalentAtoms = kwargs["equivalentAtoms"]
    if "ignoreAtomAromaticity" in kwargs:
        opts.ignoreAtomAromaticity = kwargs["ignoreAtomAromaticity"]
    if "minFragSize" in kwargs:
        opts.minFragSize = kwargs["minFragSize"]
    if "maxFragSeparation" in kwargs:
        opts.maxFragSeparation = kwargs["maxFragSeparation"]
    if "minCliqueSize" in kwargs:
        opts.minCliqueSize = kwargs["minCliqueSize"]
    if "completeSmallestRings" in kwargs:
        opts.completeSmallestRings = kwargs["completeSmallestRings"]

    results = FindMCES(mol1, mol2, opts)
    if not results:
        return 0, 0, 0.0, False

    r = results[0]
    return (
        len(r.bondMatches()),
        len(r.atomMatches()),
        r.similarity,
        r.timedOut,
    )


# All test cases from RDKit's mces_catch.cpp.
# Format: (name, smiles1, smiles2, options_dict)
# Options that are chemistry-specific (ring matching, equivalent atoms, etc.)
# are recorded but only affect the labeled MCES result. Our unlabeled result
# should be >= the labeled one.
TEST_CASES = [
    # Simple pairs
    ("very_small", "OCC(=O)N", "NC(=O)C=O", {}),
    ("very_small_similarity_threshold_reject", "OCC(=O)N", "NC(=O)C=O",
     {"similarityThreshold": 0.99}),
    ("delta_y_small", "CC1CC1", "CC(C)(C)C", {}),
    ("delta_y_large", "C1CCCCC12CC2", "C1CCCCC1(C)(C)", {}),
    ("chloro_vs_fluoro_benzene", "c1ccccc1Cl", "c1ccccc1F", {}),
    ("toluene_vs_phenylcyclopropane", "c1ccccc1C", "c1ccccc1C2CC2", {}),
    ("fmcs_test1_basics", "CC1CCC(N)CC1", "CC1CC(C)CC(C)C1", {}),
    ("juglone_vs_scopoletin", "Oc1cccc2C(=O)C=CC(=O)c12",
     "O1C(=O)C=Cc2cc(OC)c(O)cc12", {"similarityThreshold": 0.5}),
    ("testosterone_vs_estradiol",
     "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
     "CC12CCC3C(C1CCC2O)CCC4=C3C=CC(=C4)O", {}),
    ("methadone_vs_meperidine",
     "c1ccccc1C(C(=O)CC)(c1ccccc1)CC(C)N(C)C",
     "c1ccccc1C1(CCN(C)CC1)C(=O)OCC", {}),
    ("symmetrical_esters",
     "c1c(OC)c(OC)c(OC)cc1C(=O)OCCCOC(=O)c1cc(OC)c(OC)c(OC)c1",
     "c1c(OC)c(OC)c(OC)cc1C(=O)OCCOC(=O)c1cc(OC)c(OC)c(OC)c1", {}),
    ("ignore_bond_orders", "CC=CC", "CCCC", {"ignoreBondOrders": True}),

    # Dyphylline series
    ("dyphylline_vs_caffeine",
     "OCC(O)CN1C=NC2=C1C(=O)N(C)C(=O)N(C)2",
     "CN1C=NC2=C1C(=O)N(C)C(=O)N(C)2", {}),
    ("dyphylline_vs_uric_acid",
     "OCC(O)CN1C=NC2=C1C(=O)N(C)C(=O)N(C)2",
     "C12C(=O)NC(=O)NC(NC(=O)N2)=1", {}),
    ("dyphylline_vs_captagon",
     "OCC(O)CN1C=NC2=C1C(=O)N(C)C(=O)N(C)2",
     "c1ccccc1CC(C)N(C)CCN1C=NC2=C1C(=O)N(C)C(=O)N(C)2", {}),
    ("dyphylline_vs_enprofylline",
     "OCC(O)CN1C=NC2=C1C(=O)N(C)C(=O)N(C)2",
     "CCCN1C(=O)NC(=O)C2N=CNC1=2", {}),
    ("dyphylline_vs_viagra",
     "OCC(O)CN1C=NC2=C1C(=O)N(C)C(=O)N(C)2",
     "CN1CCN(CC1)S(=O)(=O)c1ccc(OCC)c(c1)C1=NC(=O)C2N(C)N=C(CCC)C(N1)=2",
     {}),
    ("dyphylline_vs_cafaminol",
     "OCC(O)CN1C=NC2=C1C(=O)N(C)C(=O)N(C)2",
     "OCCN(C)C1=NC2N(C)C(=O)N(C)C(=O)C(N1C)=2", {}),

    # Aromatic tests
    ("bad_aromatics_1a", "c1ccccc1C(=O)c1ccncc1", "c1ccccc1C(=O)c1ccccc1", {}),
    ("bad_aromatics_1b", "c1ccccc1C(=O)c1cc[nH]c1", "c1ccccc1C(=O)c1cnccc1", {}),
    ("bad_aromatics_1c", "c1ccccc1C(=O)c1ccncc1", "c1ccccc1C(=O)c1cnccc1", {}),

    # GitHub bug fixes
    ("github_8198_toluene_cyclopropyl", "c1ccccc1C", "c1ccccc1C2CC2", {}),
    ("github_8198_pyrazoline",
     "Cc1ccc(C2=NN(C(N)=S)C(c3ccc(F)cc3)C2)cc1C",
     "CC(=O)N1N=C(c2ccc(C)c(C)c2)CC1c1ccc(F)cc1", {}),

    # Zinc pair
    ("zinc_pair", "NC(=O)Nc1cccc2c1C(=O)c1c-2n[nH]c1-c1cccs1",
     "COc1ccc(-c2[nH]nc3c2C(=O)c2c(NC(N)=O)cccc2-3)cc1", {}),

    # Ring matching
    ("ring_matches_ring", "c1ccccc1C(C(=O)CC)(c1ccccc1)CC(C)N(C)C",
     "c1ccccc1C1(CCN(C)CC1)C(=O)OCC",
     {"ringMatchesRingOnly": True}),

    # Fragment tests
    ("single_fragment_1", "C1CC1CCC1NC1", "C1CC1CCCCC1NC1",
     {"ringMatchesRingOnly": True}),
    ("single_fragment_2", "c1cnccc1CCc1ncccc1", "c1cnccc1CCCCCCc1ncccc1",
     {"ringMatchesRingOnly": True}),

    # Atom aromaticity
    ("atom_aromaticity_ignore", "c1ccccc1NCC", "C1CCCCC1NCC",
     {"ignoreAtomAromaticity": True}),
    ("atom_aromaticity_respect", "c1ccccc1NCC", "C1CCCCC1NCC",
     {"ignoreAtomAromaticity": False}),

    # Memory/timeout edge cases
    ("github_8645_memory", "Fc1c(F)c(F)c(F)c(F)c1c1c(F)c(F)c(F)c(F)c1",
     "FC1(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C1(F)(F)",
     {"timeout": 10}),

    # Min clique size
    ("min_clique_size_15", "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
     "CC12CCC3C(C1CCC2O)CCC4=C3C=CC(=C4)O", {}),

    # Exact connection match
    ("exact_connections_1", "c1cccc(Cc2cccnc2)c1", "c1ncccc1Oc1ccccc1",
     {"exactConnectionsMatch": True}),
]


def main():
    suppress_rdkit_logs()
    cases = []
    skipped = 0
    for name, smi1, smi2, opts in TEST_CASES:
        try:
            g1 = mol_to_graph(smi1)
            g2 = mol_to_graph(smi2)
        except ValueError as e:
            print(f"SKIP {name}: {e}")
            skipped += 1
            continue

        try:
            bond_matches, atom_matches, similarity, timed_out = compute_mces(
                smi1, smi2, **opts
            )
        except Exception as e:
            print(f"SKIP {name}: MCES failed: {e}")
            skipped += 1
            continue

        case = {
            "name": name,
            "smiles1": smi1,
            "smiles2": smi2,
            "graph1": g1,
            "graph2": g2,
            "expected_bond_matches": bond_matches,
            "expected_atom_matches": atom_matches,
            "expected_similarity": round(similarity, 6),
            "timed_out": timed_out,
            "options": opts,
        }
        cases.append(case)
        print(f"{name}: {bond_matches} bonds, {atom_matches} atoms, sim={similarity:.4f}"
              f"{' [TIMEOUT]' if timed_out else ''}")

    output_path = "tests/fixtures/mces_ground_truth.json.gz"
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump({"version": 4, "cases": cases}, f, indent=2)

    print(f"\nWrote {len(cases)} cases to {output_path} (skipped {skipped})")


if __name__ == "__main__":
    main()
