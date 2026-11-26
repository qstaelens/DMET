import numpy as np
import os
import glob

def solve(CONST, OEI, FOCK, TEI, Norb, Nel, Nimp, TwoS=0, chempot_imp=0.0):
    """
    Solves the impurity problem using Block2 DMRG.
    Returns (ImpurityEnergy, OneRDM).
    """

    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    FOCKcopy = FOCK.copy()
    for orb in range(Nimp):
        FOCKcopy[orb, orb] -= chempot_imp

    scratch_dir = f"/app/output/block2_Fe1/Norb{Norb}_Nel{Nel}_TwoS{TwoS}"
    os.makedirs(scratch_dir, exist_ok=True)

    n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK") or (os.cpu_count() or 1))
    driver = DMRGDriver(scratch=scratch_dir, symm_type=SymmetryTypes.SU2, n_threads=n_threads)
    driver.initialize_system(n_sites=Norb, n_elec=Nel, spin=TwoS)

    print("Block2: Nel =", Nel, "TwoS =", TwoS)

    mpo = driver.get_qc_mpo(h1e=FOCKcopy, g2e=TEI, iprint=0)

    # DMRG settings
    bond_dims = [50]
    noises = [0.0]
    thrds = [1e-6]
    n_sweeps = 10

    D = bond_dims[0]

    # Find the largest saved bond dimension <= M
    checkpoints = [
        int(os.path.basename(f).split("_D")[-1].replace(".flag", ""))
        for f in glob.glob(os.path.join(scratch_dir, "checkpoint_D*.flag"))
    ]
    candidates = [m for m in checkpoints if m <= D]

    if candidates:
        previous_D = max(candidates)
        ket = driver.load_mps(tag=f"gs_D{previous_D}", nroots=1)
        print(f"Block2: Loaded MPS with bond dimension {previous_D}")
    else:
        ket = driver.get_random_mps(tag=f"gs_D{D}", bond_dim=D, nroots=1)
        print("Block2: Created new MPS")

    # Run DMRG
    EnergyDMRG = driver.dmrg(mpo, ket, n_sweeps=n_sweeps, bond_dims=bond_dims, noises=noises, thrds=thrds, iprint=1)

    # Save MPS under the requested bond dimension
    driver.copy_mps(ket, tag=f"gs_D{D}")

    with open(os.path.join(scratch_dir, f"checkpoint_D{D}.flag"), "w") as f:
        f.write(f"Norb={Norb}, Nel={Nel}, TwoS={TwoS}\n")

    print(f"Block2: D = {D}, Energy = {EnergyDMRG}")

    # Density matrices
    rdm1 = driver.get_1pdm(ket)
    rdm2 = driver.get_2pdm(ket).transpose(0, 3, 1, 2)

    # DMET impurity energy
    ImpurityEnergy = CONST
    ImpurityEnergy += 0.5 * np.einsum('ij,ij->', rdm1[:Nimp, :], OEI[:Nimp, :] + FOCK[:Nimp, :])
    ImpurityEnergy += 0.5 * np.einsum('ijkl,ijkl->', rdm2[:Nimp, :, :, :], TEI[:Nimp, :, :, :])

    return ImpurityEnergy, rdm1