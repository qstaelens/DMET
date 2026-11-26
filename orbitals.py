from pyscf import gto
import numpy as np
 
# -----------------------
# Build your molecule
# -----------------------

filename = "Fe-complex.txt"
with open(filename, "r") as f:
    atom_data = f.read().strip()

mol = gto.Mole()
mol.atom = atom_data
mol.basis  = 'cc-pvdz'
mol.spin   = 0
mol.charge = 0
mol.build(verbose=0)
 
print(f"Total raw AO count (mol.nao_nr): {mol.nao_nr()}")
 
# -----------------------
# Helper: AO slices per atom
# -----------------------
# aoslice_by_atom() -> array with rows: [ish0, ish1, iao0, iao1]
# iao0:iao1 are AO indices for that atom
ao_slices = mol.aoslice_by_atom()
 
atom_symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
atom_labels  = [f"{i}:{atom_symbols[i]}" for i in range(mol.natm)]
 
raw_ao_per_atom = []
for i in range(mol.natm):
    iao0, iao1 = ao_slices[i, 2], ao_slices[i, 3]
    raw_ao_per_atom.append(iao1 - iao0)
 
print("\nRaw AO count per atom:")
for lbl, n in zip(atom_labels, raw_ao_per_atom):
    print(f"  {lbl:>6}  -> {n:3d}")
 
# -----------------------
# Define fragments (by atom indices, 0-based)
# -----------------------
# Example grouping: tweak these to match your “left/right/axial/Cl” partition
# (Use print of atom_labels above to pick indices)
frag_defs = {
    "Impurity": [0, 1, 2, 3, 4, 5, 6],
    "Env1": list(range(7, 91)),
    "Env2": list(range(91, mol.natm)),
}

frag_defs = {
    "Impurity": [0, 1, 2],
    "Env1": list(range(3, 89)),
    "Env2": list(range(89, mol.natm)),
}
 
def atoms_to_ao_indices(atom_list):
    idx = []
    for i in atom_list:
        iao0, iao1 = ao_slices[i, 2], ao_slices[i, 3]
        idx.extend(range(iao0, iao1))
    return np.array(idx, dtype=int)
 
print("\nRaw AO count per fragment (sum of member-atom raw AOs):")
for name, atoms in frag_defs.items():
    nraw = sum(raw_ao_per_atom[i] for i in atoms)
    print(f"  {name:<10} -> {nraw:3d}  (atoms: {atoms})")
 
# -----------------------
# Effective counts after removing linear dependencies globally
# -----------------------
# Build overlap S in AO basis
S = mol.intor('int1e_ovlp')
# Canonical orthogonalization: diagonalize S
seig, U = np.linalg.eigh(S)
# Choose threshold for linear dependence
lindep = 1e-6
keep = seig > lindep
rank_eff = int(np.count_nonzero(keep))
print(f"\nGlobal effective AO dimension (rank of S with lindep={lindep}): {rank_eff} / {mol.nao_nr()}")
 
# Projectors for fragments and contribution (dimension) of each fragment to the kept subspace
# Let P_keep = U_keep U_keep^T be the projector onto the kept subspace.
U_keep = U[:, keep]
P_keep = U_keep @ U_keep.T   # (nao x nao)
 
def projector_from_ao_indices(ao_idx, nao):
    P = np.zeros((nao, nao))
    P[np.ix_(ao_idx, ao_idx)] = 1.0
    return P
 
nao = mol.nao_nr()
 
# Per-atom effective "dimension" (can be fractional because subspaces overlap)
print("\nEffective AO 'dimension' per atom (trace(P_atom * P_keep)):")
eff_per_atom = []
for i in range(mol.natm):
    ao_idx = atoms_to_ao_indices([i])
    P_atom = projector_from_ao_indices(ao_idx, nao)
    dim_atom = np.trace(P_atom @ P_keep)
    eff_per_atom.append(dim_atom)
    print(f"  {atom_labels[i]:>6}  -> {dim_atom:.3f} (raw {raw_ao_per_atom[i]})")
 
# Per-fragment effective "dimension"
print("\nEffective AO 'dimension' per fragment (trace(P_frag * P_keep)):")
for name, atoms in frag_defs.items():
    ao_idx = atoms_to_ao_indices(atoms)
    P_frag = projector_from_ao_indices(ao_idx, nao)
    dim_frag = np.trace(P_frag @ P_keep)
    nraw = sum(raw_ao_per_atom[i] for i in atoms)
    print(f"  {name:<10} -> {dim_frag:.3f}  (raw {nraw})")
 
# Optional sanity check: sum over all atoms ≈ global rank (within numerical noise)
print(f"\nSum of per-atom effective dims: {np.sum(eff_per_atom):.3f}  (should be close to {rank_eff})")