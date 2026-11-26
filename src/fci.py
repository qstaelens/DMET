import numpy as np
import os
import sys
import qcdmet_paths
from pyscf import gto, scf, ao2mo, fci

def solve( CONST, OEI, FOCK, TEI, Norb, Nel, Nimp, TwoS=0, chempot_imp=0.0):

    FOCKcopy = FOCK.copy()
    if (chempot_imp != 0.0):
        for orb in range(Nimp):
            FOCKcopy[ orb, orb ] -= chempot_imp

    mol = gto.Mole()
    mol.build( verbose=0 )
    mol.atom.append(('H', (0, 0, 0)))
    mol.nelectron = Nel
    mol.incore_anyway = True
    mf = scf.RHF( mol )
    mf.get_hcore = lambda *args: FOCKcopy
    mf.get_ovlp = lambda *args: np.eye( Norb )
    mf._eri = ao2mo.restore(8, TEI, Norb)
    mf.scf()

    assert TwoS >= 0
    assert TwoS <= Nel
    assert (Nel - TwoS) % 2 == 0
    nalpha = (Nel + TwoS) // 2
    nbeta  = (Nel - TwoS) // 2
    nelec = (nalpha, nbeta)
    print("FCI: Nel =", Nel, "TwoS =", TwoS, "nalpha =", nalpha, "nbeta =", nbeta)

    cisolver = fci.direct_spin1.FCI()
    cisolver.verbose = 4
    cisolver.max_cycle = 50
    cisolver.conv_tol = 1e-8
    EnergyFCI, FCIvector = cisolver.kernel( FOCKcopy, TEI, Norb, nelec, ecore=CONST)
    TwoRDM = cisolver.make_rdm2( FCIvector, Norb, nelec)

    OneRDM = np.einsum( 'ijkk->ij', TwoRDM ) / ( Nel - 1 )

    ImpurityEnergy = CONST
    ImpurityEnergy += 0.5 * np.einsum( 'ij,ij->', OneRDM[:Nimp,:], OEI[:Nimp,:] + FOCK[:Nimp,:] )
    ImpurityEnergy += 0.5 * np.einsum( 'ijkl,ijkl->', TwoRDM[:Nimp,:,:,:], TEI[:Nimp,:,:,:] )
    return ( ImpurityEnergy, OneRDM )