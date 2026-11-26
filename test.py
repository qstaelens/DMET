from src import localintegrals, dmet_hungpham
from pyscf import gto, scf, ao2mo, fci, mcscf, cc
import numpy as np
import time

t0 = time.perf_counter()

# ---- choose methods ----
DO_RHF    = True
DO_UHF    = True
DO_CCSD = True
DO_DMET   = True

rhf_energies    = []
uhf_energies    = []
ccsd_energies = []
dmet_energies   = []

spins = [0]

filename = "H2.xyz"

for spin in spins:
    mol = gto.Mole()
    mol.atom = filename
    
    basis = '6-31g'  # 'sto-6g', '6-31g', '6-31g*', 'cc-pvdz'
    mol.basis  = basis
    mol.spin   = 2*spin
    mol.charge = 0 
    mol.build(verbose=0)

    print('#AOs=',mol.nao_nr())
    print('spin=',spin)

    mf = scf.RHF(mol)
    mf.verbose    = 4
    mf.max_cycle  = 100
    mf.conv_tol = 1e-7

    mf.chkfile = f"/app/output/H2_spin{spin}.chk"
    mf.init_guess = "chkfile"
    mf.kernel()

    # --- RHF total energy ---
    if DO_RHF:
        rhf_energy = mf.e_tot
        rhf_energies.append(rhf_energy)
        print(f"[RHF]   spin={spin}  E = {rhf_energy:.12f}")

    # --- UHF total energy ---
    if DO_UHF:
        mf_uhf = scf.UHF(mol)
        mf_uhf.verbose    = 4
        mf_uhf.max_cycle  = 100
        mf_uhf.conv_tol   = 1e-7
        mf_uhf.chkfile = f"/app/output/H2_UHF_spin{spin}.chk"
        mf_uhf.init_guess = "chkfile"
        mf_uhf.kernel()

        uhf_energy = mf_uhf.e_tot
        uhf_energies.append(uhf_energy)

        ss, multiplicity = mf_uhf.spin_square()

        print(f"[UHF]   spin={spin}  E = {uhf_energy:.12f}")
        print(f"[UHF]   <S^2> = {ss:.8f}")
        print(f"[UHF]   multiplicity = {multiplicity:.8f}")

    # --- CCSD total energy ---
    if DO_CCSD:
        mycc = cc.CCSD(mf)
        mycc.verbose = 4
        mycc.max_cycle = 50
        mycc.conv_tol = 1e-6

        ecc, t1, t2 = mycc.kernel()

        ccsd_energy = mycc.e_tot
        ccsd_energies.append(ccsd_energy)

        print(f"[CCSD]  spin={spin}  E = {ccsd_energy:.12f}")

    # --- DMET ---
    if DO_DMET:
        myInts = localintegrals.localintegrals(mf, range(mol.nao_nr()), 'meta_lowdin')

        unit_sizes = np.array([mol.nao_nr()])  #unit_sizes = np.array([28,230,230])
        assert np.sum(unit_sizes) == mol.nao_nr()

        impurityClusters = []
        jump = 0
        for fragment in range(1):    
            impurity_orbitals = np.zeros([mol.nao_nr()], dtype=int)    
            num_orb_in_imp = unit_sizes[fragment]
            if fragment > 0: 
                impurity_orbitals[jump : jump + num_orb_in_imp] = -1
            else:
                impurity_orbitals[jump : jump + num_orb_in_imp] = 1
            impurityClusters.append(impurity_orbitals)
            jump += num_orb_in_imp

        print(impurityClusters)
        isTranslationInvariant = False
        method   = 'DMRG'     # CC, MP2, FCI, CASSCF, RHF, DMRG-CheMPS2, DMRG
        SCmethod = 'NONE'   # LSTSQ, BFGS, NONE
        fitImpBath = True
        doDET     =  True
        bath_orb_per_bond = False #How many bath orbitals do you want for each fragment?
        theDMET = dmet_hungpham.dmet(myInts, impurityClusters,isTranslationInvariant, method, SCmethod, fitImpBath, use_constrained_opt=False, doDET = doDET)
        theDMET.TwoS = mol.spin

        Energy = theDMET.doselfconsistent()
        dmet_energies.append(Energy)
        print(f"[DMET]  spin={spin}  E = {Energy:.12f}")
        print("....................................................................................")

    elapsed_minutes = (time.perf_counter() - t0) / 60.0
    print(f"\nWall time: {elapsed_minutes:.2f} minutes")

elapsed_minutes = (time.perf_counter() - t0) / 60.0
print(f"\nTotal wall time: {elapsed_minutes:.2f} minutes")

# --- final summaries ---
print("Spins =", spins)
print("Basis set:", basis)
if DO_RHF:
    print("RHF energies =", rhf_energies)
if DO_UHF:
    print("UHF energies =", uhf_energies)
if DO_CCSD:
    print(f"CCSD energies =", ccsd_energies)
if DO_DMET and SCmethod == 'NONE':
    print(f"single-shot DMET energies with high-LOT:{method} = {dmet_energies}")
elif DO_DMET and theDMET.doDET:
    print(f"DET energies with high-LOT:{method} = {dmet_energies}")
elif DO_DMET:
    print(f"DMET energies with high-LOT:{method} = {dmet_energies}")
