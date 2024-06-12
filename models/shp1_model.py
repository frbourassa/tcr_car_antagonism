# -*- coding:utf-8 -*-
""" Module containing functions to solve for the steady-state of the
Francois 2013 model."""

import numpy as np
from scipy.optimize import root as rootsolve, root_scalar

## Solving for C_T and D_T numerically (without the non-saturation approximation)
# Without antagonism, solution of f(L) = 0
def solution_CT(L1p, tau1p, kappap, Rp):
    """ Solves the quadratic equation for C_T without antagonism"""
    bterm = L1p + Rp + 1/(kappap*tau1p)
    return 0.5 * bterm - 0.5*np.sqrt(bterm**2 - 4*L1p*Rp)


# This solves for C_T and D_T, switching to a simple quadratic equation if L2 is 0.
def solve_CT_DT(L1p, L2p, tau1p, tau2p, kappap, Rp):
    """ Use the np.roots function to solve the cubic equation
    that results for C_T after substitution of D_T(C_T).
    The roots are found within machine precision,
    since there is an exact formula for cubic. """
    # If there is a single ligand, really:
    if L2p < 1e-12:
        c_t, d_t = solution_CT(L1p, tau1p, kappap, Rp), 0.
        return c_t, d_t

    # Solving the first equation for D_T(C_T) and inserting in the second, we get
    # a cubic equation for C_T with coefficients
    p = np.zeros(4)
    p[0] = tau1p / tau2p - 1.
    p[1] = Rp + L1p - L2p - tau1p/tau2p * (Rp + 2.*L1p) - 1. / (kappap * tau2p) + 1. / (kappap * tau1p)
    p[2] = tau1p / tau2p * L1p * (2.*Rp + L1p) + L1p / (kappap * tau2p) - Rp * L1p + L1p * L2p
    p[3] = -tau1p / tau2p * Rp * L1p**2
    c_ts = np.roots(p)

    # Keep only real roots that are in [0, min(L_1, R)]
    isreal = (np.abs(np.imag(c_ts)) < 1e-14)
    isrange = (np.real(c_ts) >= 0.)*(np.real(c_ts) <= min(Rp, L1p))
    c_ts = np.real(c_ts[np.logical_and(isreal, isrange).astype(bool)])

    # There should already be a single root left in the correct range
    if len(c_ts) > 1:
        print("Many C_T solutions in the correct range:", c_ts)

    # Lastly, enforce D_T in the correct range too, to make sure.
    d_ts = Rp - c_ts - c_ts / (kappap * tau1p * (L1p - c_ts))
    d_isrange = ((d_ts >= 0.)*(d_ts <= min(Rp, L2p))).astype(bool)

    assert np.sum(d_isrange) == 1, "There are either no roots or too many"
    c_ts = float(c_ts[d_isrange])
    d_ts = float(d_ts[d_isrange])
    return c_ts, d_ts  # C_T and D_T


## Solving for S (closing the system)
# Without antagonism, h(S)
def equation_S_single(Sp, ratesp, tau1p, Tsol, STp, Np, mp=1):
    """ Polynomial equation for S without antagonism; we want the root between 0 and S_T"""
    phip, bp, gammap, kappap, Csp = ratesp

    invtau = 1. / tau1p
    dephospho = bp + gammap * Sp
    sumrates = phip + invtau + dephospho
    rootpart = np.sqrt(sumrates**2 - 4. * phip * dephospho)

    rminus = (sumrates - rootpart) / (2. * dephospho)
    rplus = (sumrates + rootpart) / (2. * dephospho)

    aminus = Tsol * (1.0 - rminus) / (1.0 - (rminus/rplus)**(Np + 1))
    aplus = Tsol * (rplus - 1.0) / ((rplus/rminus)**(Np + 1) - 1.0)

    Cmp = aminus * rminus**mp + aplus * rplus**mp
    return STp * Cmp / (Cmp + Csp) - Sp



# With antagonism, still one equation H(S), need C_T and D_T solutions
def equation_S_antagonism(Sp, ratesp, tau1p, tau2p, CTsol, DTsol, STp, Np, mp=1):
    """Polynomial equation for S with antagonism; we want the root between 0 and S_T"""
    phip, bp, gammap, kappap, Csp = ratesp
    invtau1 = 1. / tau1p
    invtau2 = 1. / tau2p
    sumrates1 = phip + invtau1 + bp + gammap*Sp
    sumrates2 = sumrates1 - invtau1 + invtau2
    dephospho = bp + gammap * Sp
    rootpart1 = np.sqrt(sumrates1**2 - 4. * phip * dephospho)
    rootpart2 = np.sqrt(sumrates2**2 - 4. * phip * dephospho)

    rminus1 = (sumrates1 - rootpart1) / (2. * dephospho)
    rminus2 = (sumrates2 - rootpart2) / (2. * dephospho)
    rplus1 = (sumrates1 + rootpart1) / (2. * dephospho)
    rplus2 = (sumrates2 + rootpart2) / (2. * dephospho)

    aminus1 = CTsol * (1.0 - rminus1) / (1.0 - (rminus1/rplus1)**(Np + 1))
    aminus2 = DTsol * (1.0 - rminus2) / (1.0 - (rminus2/rplus2)**(Np + 1))
    aplus1 = CTsol * (rplus1 - 1.0) / ((rplus1/rminus1)**(Np + 1) - 1.0)
    aplus2 = DTsol * (rplus2 - 1.0) / ((rplus2/rminus2)**(Np + 1) - 1.0)

    Cmp = aminus1 * rminus1**mp + aplus1 * rplus1**mp
    Dmp = aminus2 * rminus2**mp + aplus2 * rplus2**mp
    return STp * (Cmp + Dmp) / (Cmp + Dmp + Csp) - Sp


    # Special version of the Francois 2013 model solution with a single ligand, for faster computation
def steady_shp1_1ligand(ratesp, tau1p, L1p, rstots, nmp, large_l=False, precision=1e-6):
    """ Solving for the steady_state of the Francois 2013 model with a mixture of
    two ligands (or only one by inputting L2p=0).

    Args:
        ratesp (list): phi, b, gamma, kappa, beta/alpha
        tau1p (float): binding time of the agonists
        L1p (float): number of agonist ligands
        rstots (list of 2 floats): [total number of receptors,
            total number of phosphatase molecules]
        nmp (list of 2 ints): number of steps N, feedback m
        precision (float): relative tolerance for numerical
            solution to nonlinear equations
        ct_sol_numerical (bool): if False, use the no-saturation approximation
        large_l (bool): if True, take L_F -> infinity, implying C_T=R

    Returns:
        complexes (np.ndarray): 1D array of complexes, ordered as
            [C_0, C_1, ..., C_N, D_0, D_1, ..., D_N, S]
    """
    # Rates
    Rp, STp = rstots
    Np, mp = nmp
    phip, bp, gammap, kappap, Csp = ratesp
    invtau1 = 1. / tau1p

    # Solve for C_T and D_T
    if large_l:
        CT = Rp
    else:
        CT = solution_CT(L1p, tau1p, kappap, Rp)

    # Solve for S. Range of solutions between 0 and S_tot
    Ssol = root_scalar(equation_S_single, bracket=(0., STp), rtol=precision,
                      args=(ratesp, tau1p, CT, STp, Np, mp)).root

    # Compute all Cns and Dns. First, parts of the expressions:
    dephospho = bp + gammap * Ssol
    sumrates1 = phip + invtau1 + dephospho
    rootpart1 = np.sqrt(sumrates1**2 - 4. * phip * dephospho)
    rminus1 = (sumrates1 - rootpart1) / (2. * dephospho)
    rplus1 = (sumrates1 + rootpart1) / (2. * dephospho)
    aminus1 = CT * (1.0 - rminus1) / (1.0 - (rminus1/rplus1)**(Np + 1))
    aplus1 = CT * (rplus1 - 1.0) / ((rplus1/rminus1)**(Np + 1) - 1.0)

    # Then, loop over all N values
    complexes = np.zeros(Np + 2)
    for n in range(0, Np+1):
        complexes[n] = aminus1 * rminus1**n + aplus1 * rplus1**n
    complexes[-1] = Ssol
    return complexes


def steady_shp1_2ligands(ratesp, tau1p, tau2p, L1p, L2p, rstots, nmp,
                            precision=1e-6, ct_sol_numerical=True):
    """Solving for the steady_state of the Francois 2013 model with a mixture of two ligands
    So, we need to solve for S_tot,  C_tot and D_tot first.

    Args:
        ratesp (list): phi, b, gamma, kappa, beta/alpha
        tau1p (float): binding time of the agonists
        tau2p (float): binding time of the antagonists
        L1p (float): number of agonist ligands
        L2p (float): number of antagonist ligands
        rstots (list of 2 floats): [total number of receptors,
            total number of phosphatase molecules]
        nmp (list of 2 ints): number of steps N, feedback m
        precision (float): relative tolerance for numerical
            solution to nonlinear equations

    Returns:
        complexes (np.ndarray): 1D array of complexes, ordered as
            [C_0, C_1, ..., C_N, D_0, D_1, ..., D_N, S]
    """
    # Rates
    Rp, STp = rstots
    Np, mp = nmp
    phip, bp, gammap, kappap, Csp = ratesp
    invtau1 = 1. / tau1p
    invtau2 = 1. / tau2p

    # Solve for C_T
    CTsol, DTsol = solve_CT_DT(L1p, L2p, tau1p, tau2p, kappap, Rp)

    # Solve for S. Range of solutions between 0 and S_tot
    Ssol = root_scalar(equation_S_antagonism, bracket=(0., STp), rtol=precision,
                      args=(ratesp, tau1p, tau2p, CTsol, DTsol, STp, Np, mp)).root

    # Compute all Cns and Dns. First, parts of the expressions:
    sumrates1 = phip + invtau1 + bp + gammap*Ssol
    sumrates2 = sumrates1 - invtau1 + invtau2
    dephospho = bp + gammap * Ssol
    rootpart1 = np.sqrt(sumrates1**2 - 4. * phip * dephospho)
    rootpart2 = np.sqrt(sumrates2**2 - 4. * phip * dephospho)
    rminus1 = (sumrates1 - rootpart1) / (2. * dephospho)
    rminus2 = (sumrates2 - rootpart2) / (2. * dephospho)
    rplus1 = (sumrates1 + rootpart1) / (2. * dephospho)
    rplus2 = (sumrates2 + rootpart2) / (2. * dephospho)
    aminus1 = CTsol * (1.0 - rminus1) / (1.0 - (rminus1/rplus1)**(Np+1))
    aminus2 = DTsol * (1.0 - rminus2) / (1.0 - (rminus2/rplus2)**(Np+1))
    aplus1 = CTsol * (rplus1 - 1.0) / ((rplus1/rminus1)**(Np+1) - 1.0)
    aplus2 = DTsol * (rplus2 - 1.0) / ((rplus2/rminus2)**(Np+1) - 1.0)

    # Then, loop over all N values
    complexes = np.zeros(2*Np + 3)
    for n in range(0, Np+1):
        complexes[n] = aminus1 * rminus1**n + aplus1 * rplus1**n
        complexes[Np + 1 + n] = aminus2 * rminus2**n + aplus2 * rplus2**n
    complexes[-1] = Ssol
    return complexes
