""" Version of the Francois 2013 model extended to CAR and TCR.

@author: frbourassa
October 2021
"""
import numpy as np
from scipy.optimize import root_scalar


### CAR+TCR MODEL ###
## Solving for C_T or D_T numerically (without the non-saturation approximation)
# The equations for C_T and D_T are uncoupled because the two types of
# receptors are different and have different ligands.
def solution_CTs(Lp, taup, kappap, Rp):
    """ Solves the quadratic equation for C_T or D_T.
    Can also be used vectorially with Lp, taup, kappap, Rp being arrays
    of parameter values for the different receptor types.
    """
    bterm = Lp + Rp + 1/(kappap*taup)
    return 0.5 * bterm - 0.5*np.sqrt(bterm**2 - 4*Lp*Rp)


## Solving for S (closing the system)
# One equation involving C_m. Do not use complexes_given_I because it would
# be called many times and be slow. Compute only C_m here.
def equation_I_single_kind(Ip, ratesp, taup, C_tot_sol, i_params, n_params):
    """Nonlinear equation for S with antagonism; we want the root between 0 and I_T.
    Ip (float): current level of inhibitory species I.  I
    ratesp (list of floats): phi, b, gamma, kappa
    taup (float): binding time of the ligand
    CD_tot_sol (float): C_tot
    i_params (list of 3 floats):
        ITp (float): total number of phosphatase molecules
        cmthresh (list of floats): activation threshold of the
            inhibitory species I by each type of receptor
    n_params (list of ints): N and m
    """
    ITp, cmthresh = i_params
    # Compute C_m
    Np, mp = n_params
    phip, bp, gammap, kappap = ratesp
    invtau = 1.0 / taup
    sumrates = phip + invtau + bp + gammap*Ip
    dephospho = bp + gammap * Ip
    rootpart = np.sqrt(sumrates**2 - 4. * phip * dephospho)
    rminus = (sumrates - rootpart) / (2. * dephospho)
    rplus = (sumrates + rootpart) / (2. * dephospho)
    aminus = C_tot_sol * (1.0 - rminus) / (1.0 - (rminus/rplus)**(Np+1))
    cm = aminus * (1.0 + (rminus/rplus)**(Np-mp+1) * (rplus - 1.0)/(1.0 - rminus))*rminus**mp
    cm = cm / cmthresh
    return ITp * cm / (cm + 1.0) - Ip


## Solution when there is a single receptor type
# Given S, compute complexes for one kind of receptor
def complexes_given_I_single(Ip, ratep, taup, CTsoln, Np):
    """
    Args:
        i (int): index of the receptor type. The ith element of each
            array in the lists below will be used.
        Ip (float): level of inhibitory species I
        ratesp (list of arrays): phis, bs, gamma_mat, kappas,
            the ith element of each array will be used.
        taups (array of floats): tau for each receptor type
        CTsolns (np.ndarray): C_T for each receptor type
        Nps (np.ndarray): N for each receptor type.
    """
    phip, bp, gammap, kappap = ratep
    invtau = 1.0 / taup
    dephospho = bp + gammap*Ip
    sumrates = phip + invtau + dephospho
    rootpart = np.sqrt(sumrates**2 - 4. * phip * dephospho)
    rminus = (sumrates - rootpart) / (2. * dephospho)
    rplus = (sumrates + rootpart) / (2. * dephospho)
    aminus = CTsoln * (1.0 - rminus) / (1.0 - (rminus/rplus)**(Np+1))
    complexes = np.zeros(Np + 1)
    for n in range(0, Np+1):
        complexes[n] = aminus * (1.0 + (rminus/rplus)**(Np-n+1) * (rplus - 1.0)/(1.0 - rminus))*rminus**n
    return complexes


# Francois 2013 model solution with a single receptor type
def solution_francois2013_single_type(ratep, taup, Lp, Rp, iparams, nparams, precision=1e-6):
    """Solving for the steady_state of the Francois 2013 model with
    one receptor type.

    Args:
        ratep (list of floats): phi, b, gamma, kappa
        taup (float): binding time of the ligand
        Lp (float): number of ligands
        Rp (float): total number of receptors
        iparams (list):
            ITp (float): total number of phosphatase molecules
            cmthresh (float): activation threshold of inhibitory species I
        nparams (list of ints): N and m
        precision (float): relative tolerance for numerical
            solution to nonlinear equations

    Returns:
        complexes (np.ndarray): 1D array of complexes and I, ordered
            [C_0, C_1, ..., C_N, I]
    """
    kappap = ratep[-1]
    C_tot_sol = solution_CTs(Lp, taup, kappap, Rp)

    # Solve for S. Range of solutions between 0 and I_tot
    Isol = root_scalar(equation_I_single_kind, bracket=(0, iparams[0]),
                        x0=iparams[0]/2, rtol=precision,
                        args=(ratep, taup, C_tot_sol, iparams, nparams)).root

    # Compute all complexes for each kind of receptor
    c_n = complexes_given_I_single(Isol, ratep, taup, C_tot_sol, nparams[0])
    complexes = np.concatenate([c_n, [Isol]])
    return complexes


## Solution when there are multiple receptor types
# Solving for I (closing the system)
# One equation involving C_m for each receptor kind
def equation_I_kind_receptors(Ip, ratesp, tausp, CD_tot_sol, i_params, n_params):
    """Nonlinear equation for S with antagonism; we want the root between 0 and I_T.
    Args:
        Ip (float): current level of inhibitory species I
        ratesp (list of np.ndarrays): [phi_arr, b_arr, gamma_arr, kappa_arr] where x_arr is
            a 1d array of parameter x values for each receptor type.
        tausp (np.ndarray of floats): binding time of the ligands
            of each type of receptor
        CD_tot_sol (np.ndarray of floats): C_tot for each receptor type
        i_params (list):
            ITp (float): total number of phosphatase molecules
            cmthreshs (np.ndarray): 1d array of activation thresholds of the
                inhibitory species I by each type of receptor
        n_params (list of np.ndarrays): array of N and array of m
            for each receptor type
    """
    ITp, cmthreshs = i_params
    phips, bps, gammas, kappaps = ratesp
    cm_array = np.zeros(len(kappaps))
    Narr, marr = n_params
    # For each receptor type i, compute C_{m, i}. Vectorized.
    invtaus = 1.0 / tausp
    dephosphos = bps + np.dot(gammas, Ip)
    sumrates = phips + invtaus + dephosphos
    rootparts = np.sqrt(sumrates**2 - 4. * phips * dephosphos)
    rminus = (sumrates - rootparts) / (2. * dephosphos)
    rplus = (sumrates + rootparts) / (2. * dephosphos)
    aminus = CD_tot_sol * (1.0 - rminus) / (1.0 - (rminus/rplus)**(Narr+1))
    cm_array = aminus * (1.0 + (rminus/rplus)**(Narr-marr+1) * (rplus - 1.0)/(1.0 - rminus))*rminus**marr
    cms_sum = np.sum(cm_array / cmthreshs)
    return ITp * cms_sum / (cms_sum + 1.0) - Ip


# Given I value, compute complexes for the ith receptor type
def complexes_given_I(i, Ip, ratesp, taups, CTsolns, Nps):
    """
    Args:
        i (int): index of the receptor type. The ith element of each
            array in the lists below will be used.
        Ip (np.ndarray): inhibitory molecule
        ratesp (list of arrays): phis, bs, gamma_mat, kappas,
            the ith element of each array will be used.
        taups (array of floats): tau for each receptor type
        CTsolns (np.ndarray): C_T for each receptor type
        Nps (np.ndarray): N for each receptor type.
    """
    phip, bp, gamma, kappap = [a[i] for a in ratesp]
    Np = Nps[i]
    invtau = 1.0 / taups[i]
    dephospho = bp + gamma*Ip
    sumrates = phip + invtau + dephospho
    rootpart = np.sqrt(sumrates**2 - 4. * phip * dephospho)
    rminus = (sumrates - rootpart) / (2. * dephospho)
    rplus = (sumrates + rootpart) / (2. * dephospho)
    aminus = CTsolns[i] * (1.0 - rminus) / (1.0 - (rminus/rplus)**(Np+1))
    complexes = np.zeros(Np + 1)
    for n in range(0, Np+1):
        complexes[n] = aminus * (1.0 + (rminus/rplus)**(Np-n+1) * (rplus - 1.0)/(1.0 - rminus))*rminus**n
    return complexes


# Main functions solving the Francois 2013 model with multiple receptor types.
def solution_francois2013_many_receptor_types(ratesp, tausp, Lsp, Rsp, iparams, nparams, precision=1e-6):
    """Solving for the steady-state of the Francois 2013 model with two
    different receptors, each with its own ligand.
    So, we need to solve for I, C_tot and D_tot first.

    Args:
        ratesp (list of np.ndarrays): phis, bs, gammas, kappas
            Each array contains the values of one parameter for each
            type of receptor.
        tausp (np.ndarray of floats): binding time of the ligands
            of each type of receptor
        Lsp (np.ndarray of floats): number of ligands of each type
        Rsp (np.ndarray of floats): total number of receptors of each type
        iparams (list):
            ITp (float): total number of phosphatase molecules
            cmthreshs (np.ndarray): 1d array of activation thresholds of the
                inhibitory species I by each type of receptor
        nparams (list of 2 lists): Ns and ms for each receptor type
        precision (float): relative tolerance for numerical
            solution to nonlinear equations

    Returns:
        complexes (np.ndarray): list of 1D arrays of complexes or S, ordered
            [ [C_0, C_1, ..., C_N],
              [D_0, D_1, ..., D_N],
              I
            ]
    """
    # Exploit the vector-compatible form of the function solving for C_Ts
    kappasp = ratesp[-1]
    C_tot_sols = solution_CTs(Lsp, tausp, kappasp, Rsp)

    # Solve for I. Range of solutions between 0 and I_tot
    # Initial guess proportional to each C_T, total of I_tot/3.
    initial_i_guess = 0.33 * iparams[0]
    Isol = root_scalar(equation_I_kind_receptors, bracket=(0.0, iparams[0]),
                    x0=initial_i_guess, rtol=precision,
                    args=(ratesp, tausp, C_tot_sols, iparams, nparams)).root

    # Compute all complexes for each kind of receptor
    complexes = []  # List of arrays and I
    for i in range(len(tausp)):
        c_ns = complexes_given_I(i, Isol, ratesp, tausp, C_tot_sols, nparams[0])
        complexes.append(c_ns)
    complexes.append(Isol)
    return complexes


## Calculating the output of each receptor type
# Computing the thresholds of activation theta_n for each receptor type
def get_thresholds_francois2013(ratesp, tau_cs, Rsp, iparams, nparams):
    """ Compute activation threshold corresponding to threshold tau_cs
    for each receptor type.

    Args: see solution_francois2013_many_receptor_types documentation
    Returns:
        threshs (np.ndarray): thresholds on C_N for all receptor type.
    """
    threshs = []
    for i in range(len(tau_cs)):
        ratep = [a[i] for a in ratesp]
        npar = [a[i] for a in nparams]
        ipar = [iparams[0], iparams[1][i]]
        Rp = Rsp[i]
        comp = solution_francois2013_single_type(ratep, tau_cs[i],
                    Rsp[i]*100.0, Rsp[i], ipar, npar, precision=1e-6)
        threshs.append(comp[npar[0]])
    return np.asarray(threshs)
