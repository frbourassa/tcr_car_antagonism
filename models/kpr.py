# -*- coding:utf-8 -*-
""" Module containing functions to solve for the steady-state of
a pure KPR model (with receptor saturation though)."""
import numpy as np
from models.akpr_i_model import solution_CT, solve_CT_DT


## Main model functions
# Simple KPR model
def steady_kpr_1ligand(ratesp, tau1p, L1p, Rp, Np, large_l=False):
    """ Solving for the steady_state of the I model for a single ligand type.

    Args:
        ratesp (list): phi, kappa
        tau1p (float): binding time of the agonists
        L1p (float): number of agonist ligands
        Rp (float): R_tot, total number of receptors
        Np (int): Np, the number of phosphorylated complexes in the cascade
        large_l (bool): if True, take L_F -> infinity, implying C_T=R
    Returns:
        complexes (np.ndarray): 1D array of complexes, ordered as
            [C_0, C_1, ..., C_N, S]
    """
    # Rates
    phip, kappap = ratesp

    # Solve for C_T and D_T
    if large_l:
        CT = Rp
    else:
        CT = solution_CT(L1p, tau1p, kappap, Rp)
    # Quantities that come back often
    geofact = phip*tau1p / (phip*tau1p + 1.0)

    # Compute all Cns: apply same factor recursively
    complexes = np.zeros(Np + 1)
    complexes[0] = CT / (phip*tau1p + 1.0)
    for n in range(1, Np):
        complexes[n] = complexes[n-1]*geofact
    complexes[Np] = complexes[Np-1]*phip*tau1p

    return complexes
