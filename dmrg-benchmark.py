from src import localintegrals, dmet_hungpham
from pyscf import gto, scf
import numpy as np
import time

filename = "H6.xyz"
basis = "cc-pvdz"
spin = 0
charge = 0
methods = ["DMRG", "DMRG-CheMPS2"]
methods = ["DMRG"]

t_total_start = time.perf_counter()

# Build molecule
mol = gto.Mole()
mol.atom = filename
mol.basis = basis
mol.spin = 2 * spin
mol.charge = charge
mol.build(verbose=0)

# RHF
t_rhf_start = time.perf_counter()
mf = scf.RHF(mol)
mf.verbose = 4
mf.max_cycle = 50
mf.conv_tol = 1e-7
mf.chkfile = "/app/output/H5.chk"
mf.init_guess = "chkfile"
mf.kernel()
t_rhf = time.perf_counter() - t_rhf_start

if not mf.converged:
    raise RuntimeError("RHF did not converge.")

# Localized integrals
t_integrals_start = time.perf_counter()
myInts = localintegrals.localintegrals(mf, range(mol.nao_nr()), "meta_lowdin")
t_integrals = time.perf_counter() - t_integrals_start

# Full system as one impurity
unit_sizes = np.array([mol.nao_nr()])
assert np.sum(unit_sizes) == mol.nao_nr()

impurityClusters = []
jump = 0

for fragment in range(len(unit_sizes)):
    impurity_orbitals = np.zeros(mol.nao_nr(), dtype=int)
    num_orb_in_imp = unit_sizes[fragment]

    if fragment > 0:
        impurity_orbitals[jump:jump + num_orb_in_imp] = -1
    else:
        impurity_orbitals[jump:jump + num_orb_in_imp] = 1

    impurityClusters.append(impurity_orbitals)
    jump += num_orb_in_imp

# DMET settings
isTranslationInvariant = False
SCmethod = "NONE"
fitImpBath = True
doDET = True

# Run DMRG solvers
results = {}

for method in methods:
    print(f"\nStarting {method}...")

    theDMET = dmet_hungpham.dmet(myInts, impurityClusters, isTranslationInvariant, method, SCmethod, fitImpBath, use_constrained_opt=False, doDET=doDET)
    theDMET.TwoS = mol.spin

    t_start = time.perf_counter()
    Energy = theDMET.doselfconsistent()
    elapsed = time.perf_counter() - t_start

    results[method] = {"energy": Energy, "time": elapsed}

total_time = time.perf_counter() - t_total_start

# Final comparison
print("\n============================================================")
print("FINAL DMRG BENCHMARK")
print("============================================================")
print(f"System       : {filename}")
print(f"Basis        : {basis}")
print(f"Orbitals     : {mol.nao_nr()}")
print(f"Electrons    : {mol.nelectron}")
print(f"TwoS         : {mol.spin}")
print()

for method in methods:
    print(f"{method:15s} E = {results[method]['energy']:.12f} Ha   time = {results[method]['time']:.2f} s")

E_block2 = results["DMRG"]["energy"]
E_chemps2 = results["DMRG-CheMPS2"]["energy"]
t_block2 = results["DMRG"]["time"]
t_chemps2 = results["DMRG-CheMPS2"]["time"]

print(f"\nEnergy difference (Block2 - CheMPS2) = {E_block2 - E_chemps2:.12e} Ha")

if t_block2 < t_chemps2:
    print(f"Block2 is {t_chemps2 / t_block2:.2f}x faster than CheMPS2.")
else:
    print(f"CheMPS2 is {t_block2 / t_chemps2:.2f}x faster than Block2.")

print(f"\nRHF time             = {t_rhf:.2f} s")
print(f"Integral setup time  = {t_integrals:.2f} s")
print(f"Total benchmark time = {total_time:.2f} s ({total_time / 60:.2f} min)")
print("============================================================")