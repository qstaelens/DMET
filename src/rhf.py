import numpy as np
import scipy.sparse.linalg
from src import qcdmet_paths
from pyscf import gto, scf, ao2mo

def solve_ERI( OEI, TEI, DMguess, numPairs ):

    mol = gto.Mole()
    mol.build(verbose=3)
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = 2 * numPairs

    L = OEI.shape[0]
    mf = scf.RHF( mol )
    mf.get_hcore = lambda *args: OEI
    mf.get_ovlp = lambda *args: np.eye( L )
    mf._eri = ao2mo.restore(8, TEI, L)
    
    mf.scf( DMguess )
    DMloc = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
    if ( mf.converged == False ):
        mf = scf.newton( mf, dm_guess=DMloc )
        DMloc = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
    return DMloc
    
def wrap_my_jk( mol_orig, ao2basis ): # mol_orig works in ao

    #get_jk(mol, dm, hermi=1, vhfopt=None)
    def my_jk( mol, dm, hermi=1, vhfopt=None ): # mol works in basis, dm is in basis
    
        dm_ao        = np.dot( np.dot( ao2basis, dm ), ao2basis.T )
        vj_ao, vk_ao = scf.hf.get_jk( mol_orig, dm_ao, hermi, vhfopt )
        vj_basis     = np.dot( np.dot( ao2basis.T, vj_ao ), ao2basis )
        vk_basis     = np.dot( np.dot( ao2basis.T, vk_ao ), ao2basis )
        return vj_basis, vk_basis
    
    return my_jk

def wrap_my_veff( mol_orig, ao2basis ): # mol_orig works in ao

    #get_veff(mol, dm, dm_last=0, vhf_last=0, hermi=1, vhfopt=None)
    def my_veff( mol, dm, dm_last=0, vhf_last=0, hermi=1, vhfopt=None ): # mol works in basis, dm is in basis
        
        ddm_basis    = np.array(dm, copy=False) - np.array(dm_last, copy=False)
        ddm_ao       = np.dot( np.dot( ao2basis, ddm_basis ), ao2basis.T )
        vj_ao, vk_ao = scf.hf.get_jk( mol_orig, ddm_ao, hermi, vhfopt )
        veff_ao      = vj_ao - 0.5 * vk_ao
        veff_basis   = np.dot( np.dot( ao2basis.T, veff_ao ), ao2basis ) + np.array( vhf_last, copy=False )
        return veff_basis
        
    return my_veff

def solve_JK( OEI, mol_orig, ao2basis, DMguess, numPairs ):

    mol = gto.Mole()
    mol.build(verbose=0)
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = 2 * numPairs

    L = OEI.shape[0]
    mf = scf.RHF( mol )
    mf.get_hcore = lambda *args: OEI
    mf.get_ovlp = lambda *args: np.eye( L )
    mf._eri = None
    mf.get_jk   = wrap_my_jk(   mol_orig, ao2basis )
    mf.get_veff = wrap_my_veff( mol_orig, ao2basis )
    mf.max_cycle = 500
    mf.damp_factor = 0.33
    
    mf.scf( DMguess )
    DMloc = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
    return DMloc
    
