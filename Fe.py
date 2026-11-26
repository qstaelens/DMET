from src import localintegrals, dmet_hungpham, rhf
from pyscf import gto, scf, ao2mo, fci, mcscf, cc
import numpy as np
import time

t0 = time.perf_counter()

# ---- choose methods ----
DO_RHF    = True
DO_UHF    = False
DO_CASSCF = False
DO_CCSD = False
DO_DMET   = True

ORBITALS = False

rhf_energies    = []
uhf_energies    = []
casscf_energies = []
ccsd_energies = []
dmet_energies   = []

spins = [1,2]

filename = "Fe.xyz"

for spin in spins:
    mol = gto.Mole()
    mol.atom = filename
    
    basis = '6-31g'  # 'sto-6g', '6-31g', '6-31g*', 'cc-pvdz', 'def2-TZVP'
    mol.basis  = basis
    mol.spin   = 2*spin
    mol.charge = 2 
    mol.build(verbose=0)

    print('#AOs=',mol.nao_nr())
    print('spin=',spin)

    if ORBITALS:
        ao_slices = mol.aoslice_by_atom()
        ao_labels = mol.ao_labels()
        for i in range(mol.natm):
            start, end = ao_slices[i, 2:4]
            print(f"\n{mol.atom_symbol(i)} (atom {i}):")
            for j in range(start, end):
                print(f"  AO {j:3d}: {ao_labels[j]}")

    mf = scf.RHF(mol)
    mf.verbose    = 4
    mf.max_cycle  = 100
    mf.conv_tol = 1e-7

    mf.chkfile = f"/app/output/Fe_{basis}_spin{spin}.chk"
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
        mf_uhf.chkfile = f"/app/output/Fe_UHF_{basis}_spin{spin}.chk"
        mf_uhf.init_guess = "chkfile"
        mf_uhf.kernel()

        uhf_energy = mf_uhf.e_tot
        uhf_energies.append(uhf_energy)

        ss, multiplicity = mf_uhf.spin_square()

        print(f"[UHF]   spin={spin}  E = {uhf_energy:.12f}")
        print(f"[UHF]   <S^2> = {ss:.8f}")
        print(f"[UHF]   multiplicity = {multiplicity:.8f}")

    # --- CASSCF total energy ---
    if DO_CASSCF:
        N_CAS_ORB  = 12
        N_CAS_ELEC = 10
        mc = mcscf.CASSCF(mf, N_CAS_ORB, N_CAS_ELEC)
        mc.verbose = 4
        mc.max_cycle = 50
        mc.conv_tol = 1e-6
        mc.kernel()
        casscf_energy = mc.e_tot
        casscf_energies.append(casscf_energy)
        print(f"[CASSCF({N_CAS_ELEC},{N_CAS_ORB})] spin={spin}  E = {casscf_energy:.12f}")

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

        def make_impurity_clusters(mol, active_orbitals):
            labels = mol.ao_labels()
            dmrg = np.zeros(mol.nao_nr(), dtype=int)

            for i, label in enumerate(labels):
                parts = label.split()
                element, orbital = parts[1], parts[2]

                if (element, orbital[:2]) in active_orbitals:
                    dmrg[i] = 1

            rhf = np.where(dmrg == 0, -1, 0)

            print("DMRG orbitals:", np.where(dmrg == 1)[0])
            print("RHF orbitals:", np.where(rhf == -1)[0])

            return [rhf, dmrg]

        impurityClusters = make_impurity_clusters(mol, active_orbitals=[
         ("Fe", "4d"), ("Fe", "3d"), ("O", "2p")]) #

        #print(impurityClusters)
        isTranslationInvariant = False
        method   = 'DMRG'     # CC, MP2, FCI, CASSCF, RHF, DMRG-CheMPS2, DMRG
        SCmethod = 'NONE'   # LSTSQ, BFGS, NONE
        fitImpBath = True
        doDET     =  True
        bath_orb_per_bond = False #How many bath orbitals do you want for each fragment?
        mu = 0.1
        theDMET = dmet_hungpham.dmet(myInts, impurityClusters,isTranslationInvariant, method, SCmethod, fitImpBath, use_constrained_opt=False, doDET = doDET, mu_imp=mu)
        theDMET.TwoS = mol.spin

        if method == 'CASSCF' or method =='SS-CASPDFT':
            theDMET.impCAS  = (10, 10)   # (2,2) is the standard
            #S = mol.spin / 2
            #theDMET.fix_spin_val = S * (S+1)
            #theDMET.CASlist = [23,24,25,26,27,28,29,30] #Which orbitals do you want in the active space?
        
        if (bath_orb_per_bond == True):
            theDMET.BATH_ORBS = np.ones([len(impurityClusters)], dtype=int)
            theDMET.BATH_ORBS[0] = 0 #How many bath orbitals do you want for each fragment?
            theDMET.BATH_ORBS[1] = 0
            theDMET.BATH_ORBS[2] = 0

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
if DO_CASSCF:
    print(f"CASSCF energies =", casscf_energies)
if DO_CCSD:
    print(f"CCSD energies =", ccsd_energies)
if DO_DMET and SCmethod == 'NONE':
    print(f"single-shot DMET energies with high-LOT:{method} = {dmet_energies}")
elif DO_DMET and theDMET.doDET:
    print(f"DET energies with high-LOT:{method} = {dmet_energies}")
elif DO_DMET:
    print(f"DMET energies with high-LOT:{method} = {dmet_energies}")
