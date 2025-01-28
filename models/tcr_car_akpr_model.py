""" Simplified version of the Voisinne 2015 model with both a CAR and a TCR.
The TCR model is the AKPR model where, instead of a kinase, the phosphatase I
plays the role of the negative feedback. The last f phosphorylation steps
are inhibited by I.

@author:frbourassa
October 2022
"""
import numpy as np
import scipy as sp
from scipy.optimize import root_scalar, root as rootsolve


### CAR+TCR MODEL ###
# Import functions for pure ligand mixtures from another module
# The equations for C_T and D_T are uncoupled because the two types of
# receptors are different and have different ligands. solution_CT can
# be called with arrays of parameters.
from models.akpr_i_model import steady_akpr_i_1ligand, solution_CT

# Can be called element-wise on ndarrays of c and thresh
def activation_function(c, thresh, pwr=2):
    return c**pwr / (c**pwr + thresh**pwr)
    #return (c / thresh)**pwr


# psi(S) function when there is a gamma matrix.
def psi_of_i_gamma(ivec, ithresh, ki, phip, gammat, psi0):
    #psi_s = phip * (ITp - complexes[-1])
    return phip / (1.0 + (gammat.dot(ivec) / ithresh)**ki) + psi0

## Solving for SHP-1 activated by different receptor types.
# For cases where at least two types have m >= N-f
def equation_ivec_types(ivec, ratesp, tausp, ctotsols, stot, nparams):
    """ Equation to solve for \vec{I} """
    phis, kappas, cmthreshs, ithreshs, ki, gamma_mat, psi0s = ratesp
    n_types = len(ivec)
    ns, ms, fs = nparams

    # Compute psi(S) of each type, given the tentative ivec
    # and the 1d arrays of parameters phis, psi0s, etc.
    psis = psi_of_i_gamma(ivec, ithreshs, ki, phis, gamma_mat, psi0s)

    # Compute the C_m of each type. Hard because m, n, f may differ.
    cmvec = np.zeros(n_types)
    for i in range(n_types):
        geofact = phis[i]*tausp[i] / (1.0 + phis[i]*tausp[i])
        if ms[i] < ns[i] - fs[i] or fs[i] == 0:  # feedforward case
            cmvec[i] = ctotsols[i] / (phis[i]*tausp[i]+ 1.0) * geofact**ms[i]
        elif ms[i] == ns[i] - fs[i]:  # Intermediate
            cmvec[i] = ctotsols[i] / (psis[i]*tausp[i]+ 1.0) * geofact**ms[i]
        elif ms[i] < ns[i]:  # Somewhere in the chain of complexes affected by psi(S)
            cmvec[i] = ctotsols[i] / (psis[i]*tausp[i]+ 1.0) * geofact**(ns[i] - fs[i])
            cmvec[i] *= (psis[i]*tausp[i] / (psis[i]*tausp[i] + 1.0))**(ms[i] - ns[i] + fs[i])
        elif ms[i] == ns[i]:  # Last complex
            cmvec[i] = ctotsols[i] * geofact**(ns[i] - fs[i])
            cmvec[i] *= (psis[i]*tausp[i] / (psis[i]*tausp[i] + 1.0))**fs[i]
        else:
            raise ValueError(("N = {}, m = {}, f = {}"
                + " for receptor type {}").format(ns[i], ms[i], fs[i], i))

    # Compute the RHS, function of S = S
    cmvec = cmvec / cmthreshs  # C_m / C_thresh
    ivec_new = stot * cmvec / (1.0 + np.sum(cmvec))
    return ivec - ivec_new  # Should be equal to zero


def equation_one_cm_types(cmii, ii, cmnorms, ratesp, tausp, c_tot_sols, stot, nparams):
    """ From a guess cm for receptor type ii, when other C_m^i are known and
    stored in cmnorms, compute the S vector and from there C_m^i to establish an equation.
    """
    # Extract parameters of receptor type ii
    phip, kappap, cmthresh, ithresh, kp, gamma_mati, psi0p  = [r[ii] for r in ratesp]
    c_tot_i = c_tot_sols[ii]
    n_p, mp, fp = [r[ii] for r in nparams]

    # Replace in cmnorms with the guess cm
    cmthreshs = ratesp[2]
    cmnorms[ii] = cmii / cmthreshs[ii]

    # Compute vector of S and psi_s from there
    sum_cmnorms = np.sum(cmnorms)
    ivec = stot * cmnorms / (1.0 + sum_cmnorms)
    # ithresh, ki, phip, gammat, psi0
    psivec = psi_of_i_gamma(ivec, *ratesp[3:5], ratesp[0], *ratesp[5:])

    # Recompute C_m from there
    geofact = phip*tausp[ii] / (phip*tausp[ii] + 1.0)

    # Compute all Cns and Dns below N-f: apply same factor recursively
    # Assuming m_i >= N_i - f_i so we can go to m=N-f at least
    cmii_eq = c_tot_i * geofact**(n_p - fp)
    geofact = psivec[ii]*tausp[ii] / (psivec[ii]*tausp[ii] + 1.0)
    if n_p > mp >= n_p - fp:
        cmii_eq *= geofact**(mp - n_p + fp) / (psivec[ii]*tausp[ii] + 1.0)
    elif mp == n_p:
        cmii_eq *= geofact**fp
    return cmii_eq - cmii


# Main functions solving the SHP-1 model with multiple receptor types.
def steady_akpr_i_receptor_types(ratesp, tausp, lsp, ri_tots, nparams):
    r"""Solving for the steady_state of the SHP-1 model with two
    different receptors, each with its own ligand.
    So, we need to solve for I_tot, C_tot and D_tot first.

    Args:
        ratesp (list of np.ndarrays): [phi_arr, kappa_arr, cmthresh, ithresh_arr,
            k_arr, gamma_mat, psi_arr] where x_arr is
            a 1d array of parameter x values for each receptor type.
            except cmthresh which is unique,
            and gammat is a KxK array for K receptor types,
            containing \gamma_{ij}, contribution of SHP-1 bound to receptor type j
            to the negative feedforward on receptor type i.
        tausp (np.ndarray of floats): binding time of the ligands
            of each type of receptor
        lsp (np.ndarray of floats): total number of ligands of each type
        ri_tots (list of 1 ndarray, 1 float):
            Rsp (np.ndarray of floats): total number of receptors of each type
            ITp (float): total number of SHP-1 molecules
        nparams (list of 3 np.ndarrays): Ns, ms, fs of all receptor types.

    Returns:
        complexes (np.ndarray): list of 1D arrays of complexes or S, ordered
            [ [C_0, C_1, ..., C_N],
              [D_0, D_1, ..., D_N'],
              [S_1, S_2]
            ]
    """
    # Exploit the vector-compatible form of the function solving for C_Ts
    phis, kappas, cmthreshs, ithreshs, ki, gamma_mat, psi0s  = ratesp
    Rsp, ITp = ri_tots
    n_types = len(Rsp)
    # C_n for each type, and lastly the S vector.
    complexes = [np.zeros(n+1) for n in nparams[0]] + [np.zeros(n_types)]

    # Special case: no input
    if np.sum(lsp) == 0.0 or np.sum(tausp) == 0.0:
        return complexes

    # Solve for C_Ts
    C_tot_sols = solution_CT(lsp, tausp, kappas, Rsp)
    ns, ms, fs = nparams

    # Quantities that come back often
    geofacts = phis*tausp / (phis*tausp + 1.0)

    # Compute all Cns and Dns below N-f: apply same factor recursively
    for i in range(n_types):
        if ns[i] == fs[i]: continue  # No complex to compute
        # Otherwise, we can at least compute C_0
        complexes[i][0] = C_tot_sols[i] / (phis[i]*tausp[i] + 1.0)
        for n in range(1, ns[i]-fs[i]):
            complexes[i][n] = complexes[i][n-1]*geofacts[i]

    # Now, compute the vector S and psi(S)
    # Case where all m < N - f
    number_implicit = sum([ms[i] >= ns[i]-fs[i] for i in range(n_types)])
    if number_implicit == 0:
        cmnorm_vec = np.asarray([complexes[i][ms[i]]/cmthreshs[i] for i in range(n_types)])
        sum_cm = np.sum(cmnorm_vec)
        ivec = ITp * cmnorm_vec / (1.0 + sum_cm)

    elif number_implicit == 1:
        # Only one receptor type is implicit; can reduce to a scalar equation
        # for the C_m of that type.
        impli = np.argmax(ms >= ns - fs)
        # Get a vector of the explicit known Cms, we don't recompute every time.
        # The unknown cm element will be 0 anyways
        cmnorm_vec = np.asarray([complexes[i][ms[i]]/cmthreshs[i] for i in range(n_types)])
        # Certainly, C_m^i can't be more than R^i minus the sum of already computed C_n^i
        bracket = (0.0, Rsp[impli])# - np.sum(complexes[impli]))
        args = (impli, cmnorm_vec, ratesp, tausp, C_tot_sols, ITp, nparams)
        sol = root_scalar(equation_one_cm_types, x0=np.mean(bracket),
                        args=args, bracket=bracket)
        if not sol.converged:
            raise RuntimeError("Could not solve for I. "
                + "Error flag: {}".format(sol.flag))
        else:
            # From the missing cm_i, compute the vector of S
            # Check that the found cm_i is consistent later
            cm_impli = sol.root
            cmnorm_vec[impli] = cm_impli / cmthreshs[impli]
            sum_cm = np.sum(cmnorm_vec)
            ivec = ITp * cmnorm_vec / (1.0 + sum_cm)

    else: # Two or more receptor types have a true feedback
        initial_i_guess = 0.5 * ITp * (C_tot_sols / (1.0 + np.sum(C_tot_sols)))
        sol = rootsolve(equation_ivec_types, x0=initial_i_guess, tol=1e-6,
                        args=(ratesp, tausp, C_tot_sols, ITp, nparams))
        if not sol.success:
            raise RuntimeError("Could not solve for I. "
                + "Error message: {}".format(sol.message))
        else:
            ivec = sol.x

    if np.sum(ivec) > ITp:
        raise ValueError("Found a I solution going over the max number of I")

    # Compute the psi(S) vector with the solution ivec.
    psis = psi_of_i_gamma(ivec, ithreshs, ki, phis, gamma_mat, psi0s)

    # Finally, compute C_{N-f}, ..., C_N for each receptor type
    for i in range(n_types):
        n = ns[i] - fs[i]
        if n > 0:
            complexes[i][n] = complexes[i][n-1] * phis[i]*tausp[i] / (psis[i]*tausp[i] + 1.0)
        else:
            complexes[i][n] = C_tot_sols[i] / (psis[i]*tausp[i] + 1.0)

        geofact = psis[i]*tausp[i] / (psis[i]*tausp[i] + 1.0)
        for n in range(ns[i] - fs[i] + 1, ns[i]):
            complexes[i][n] = complexes[i][n-1] * geofact
        complexes[i][ns[i]] = complexes[i][ns[i]-1] * psis[i]*tausp[i]

    # Check consistency
    if number_implicit == 1:
        if abs(cm_impli - complexes[impli][ms[impli]]) > 1e-6:
            print("Difference in implicit solution and final calculated C_m:",
                    cm_impli, complexes[impli][ms[impli]],
                    "for parameter values as follows:")
            print("ratesp:", ratesp, "\ntausp:", tausp, "\nlsp:", lsp,
                    "\nri_tots:", ri_tots, "\nnparams:", nparams)
    # Add the vector of S values to the returned variables
    complexes[-1] = ivec

    # Return
    return complexes


## Calculating the output of a receptor type
# Computing the threshold of activation theta_n for a receptor type
def get_threshold(nparams, ratep, ri_tots, tau_c=3.0):
    """
    Args:
        nparam (list of 3 ints): N, m, f
        ratep (list): phi, kappa, cmthresh, ithresh, k, gamma,
            psi0 for one receptor type.
        ri_tots (list of 2 floats): [R_tot, I_tot]
        tau_c (float): tau value for the threshold
    L is set equal to 10x the number of receptors: response at saturation.
    """
    # Thresholds for C_N
    thresh = steady_akpr_i_1ligand(ratep, tau_c, ri_tots[0]*10, ri_tots,
                            nparams, large_l=True)[nparams[0]]
    return thresh


# output total output of all receptor types, given thresholds on the C_N
# and the output of steady_akpr_i_receptor_types
def compute_zoutput_from_complexes(complexes, thresholds, pwr=2):
    r""" Compute \sum_i Z_i, where Z_i = C^i_N/(C^i_N + \theta^i_N) is the output
    of receptor type i.
    """
    c_n_vec = np.asarray([complexes[i][-1] for i in range(len(complexes)-1)])
    return activation_function(c_n_vec, thresholds, pwr=pwr)


# Normalization of C_N or D_N to get the next biochemical step (ZAP-70?)
# Version of get_thresholds that works from a parameter dictionary
# and explicitly looks for CAR or TCR parameters (receptype).
# Simpler use in the full IL-2 model.
def get_thresholds_tcr_car(allparams, receptype, tau_c=10.0):
    """
    Args:
        allparams (dict): dictionary with all TCR-CAR-IL-2 model parameters
        receptype (int): 0 for TCR, 1 for CAR
        tau_c (float): critical tau for the receptor type of interest
    L is set equal to 10x the number of receptors: response at saturation.
    """
    # Wrangle parameters into the correct format
    if receptype == 0:
        # Extract tcr parameter from each parameter vector
        ratep = [p[0] for p in allparams["all_rates"][:-1]]
        rsp = allparams["tcr_RStot"]
    elif receptype == 1:
        ratep = [p[1] for p in allparams["all_rates"][:-1]]
        rsp = allparams["car_RStot"]
    else:
        raise ValueError("Unknown receptor index {}".format(receptype))
    nparam = [a[receptype] for a in allparams["n_params"]]  # [N, m] for receptor type

    # Single receptor type
    # ratesp, tau1p, L1p, ri_tots, nparams, large_l=False
    c = steady_akpr_i_1ligand(ratep, tau_c, rsp[0]*10, rsp, nparam, large_l=True)
    thresh = c[nparam[0]]  # C_N

    return thresh
