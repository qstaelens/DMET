import numpy as np
from src import qcdmet_paths
from src import localintegrals
from pyscf import ao2mo, gto, scf

def solve( CONST, OEI, FOCK, TEI, Norb, Nel, Nimp, DMguessRHF, chempot_imp=0.0 ):

    # Augment the FOCK operator with the chemical potential
    FOCKcopy = FOCK.copy()
    if (chempot_imp != 0.0):
        for orb in range(Nimp):
            FOCKcopy[ orb, orb ] -= chempot_imp
    
    # Get the RHF solution
    mol = gto.Mole()
    mol.build( verbose=0 )
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = Nel
    mol.incore_anyway = True
    mf = scf.RHF( mol )
    mf.get_hcore = lambda *args: FOCKcopy
    mf.get_ovlp = lambda *args: np.eye( Norb )
    mf._eri = ao2mo.restore(8, TEI, Norb)
    mf.scf( DMguessRHF )
    DMloc = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
    if ( mf.converged == False ):
        mf = scf.newton(mf)
        mf.kernel(dm0=DMloc)
        DMloc = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
    
    ERHF = mf.e_tot
    RDM1 = mf.make_rdm1()
    JK   = mf.get_veff(None, dm=RDM1)
 
    # To calculate the impurity energy, rescale the JK matrix with a factor 0.5 to avoid double counting: 0.5 * ( OEI + FOCK ) = OEI + 0.5 * JK
    ImpurityEnergy = CONST \
                   + 0.25 * np.einsum('ji,ij->', RDM1[:,:Nimp], FOCK[:Nimp,:] + OEI[:Nimp,:]) \
                   + 0.25 * np.einsum('ji,ij->', RDM1[:Nimp,:], FOCK[:,:Nimp] + OEI[:,:Nimp]) \
                   + 0.25 * np.einsum('ji,ij->', RDM1[:,:Nimp], JK[:Nimp,:]) \
                   + 0.25 * np.einsum('ji,ij->', RDM1[:Nimp,:], JK[:,:Nimp])
    
    return ( ImpurityEnergy, RDM1 )

