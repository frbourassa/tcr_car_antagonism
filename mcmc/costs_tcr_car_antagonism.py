""" Module with cost functions to fit antagonism ratio with TCR-TCR receptor
models.

@author: frbourassa
December 2022
"""
import numpy as np
import pandas as pd

# Local modules
from models.tcr_car_akpr_model import (steady_akpr_i_receptor_types,
                                    steady_akpr_i_1ligand, activation_function)

# Useful constants, utility functions
from utils.preprocess import (
    ln10,
    eps_for_log
)


### Utility to wrap rates differently
def repackage_tcr_car_params(pvec, kmf, other_rates, ritot, nmf_fixed):
    """ See documentation of antag_ratio_panel_tcr_car for a list
    of args.

    Returns:
        all_rates
        tcr_rates
        car_rates
        ritot_both
        tcr_ri
        car_ri
        nmf_both
        tcr_nmf
        car_nmf
        threshold_taus
    """
    expvec = np.exp(pvec*ln10)

    # Assemble rates of CAR and TCR
    phi_arr = np.asarray([other_rates[0], other_rates[7]])
    kappa_arr = np.asarray([other_rates[1], other_rates[8]])
    cmthresh_arr = np.asarray([other_rates[2], expvec[0]])
    ithresh_arr = np.asarray([other_rates[3], expvec[1]])
    k_arr = np.asarray([other_rates[4], kmf[0]])
    gamma_mat = np.asarray([[other_rates[6], expvec[2]],
                            [expvec[3], other_rates[9]]])
    psi0_arr = np.asarray([other_rates[5], other_rates[10]])

    all_rates = [phi_arr, kappa_arr, cmthresh_arr, ithresh_arr, k_arr, gamma_mat, psi0_arr]

    # if gamma_cc or gamma_tt are not 1,
    # adjust i_thresh when computing CAR alone or TCR alone
    if np.any(np.diagonal(gamma_mat) != 1.0):
        raise NotImplementedError()

    # Rates for revised AKPR model with a single type, for either type
    tcr_rates = [p[0] for p in all_rates[:5]] + [psi0_arr[0]]
    car_rates = [p[1] for p in all_rates[:5]] + [psi0_arr[1]]

    # R, I params and n, m, f params
    ritot_vec = [np.asarray([ritot[0], ritot[2]]), ritot[1]]
    car_ri = [ritot[2], ritot[1]]
    tcr_ri = [ritot[0], ritot[1]]
    tcr_nmf = nmf_fixed[:3]
    car_nmf = (nmf_fixed[3], *kmf[1:])
    nmf_both = [np.asarray(a, dtype="int") for a in zip(tcr_nmf, car_nmf)]

    # Last fit parameters are threshold taus of TCR and CAR
    threshold_taus = expvec[4:6]

    return (all_rates, tcr_rates, car_rates,
            ritot_vec, tcr_ri, car_ri,
            nmf_both, tcr_nmf, car_nmf,
            threshold_taus)


### TCR-CAR REVISED AKPR model ###
def antag_ratio_panel_tcr_car(
        pvec, kmf, other_rates, ritot, nmf_fixed, cd19_tau_l, cond_index
    ):
    """
    Args:
        pvec (np.ndarray): log10 of parameters to fit, that is,
            cmthresh_car, ithresh_car, gamma_tc, gamma_ct, tau_c_tcr, tau_c_car
        kmf (list of 3 ints): grid search parameters, that is,
            k_{S,CAR}, m_{CAR}, f_{CAR}
        other_rates (list of parameters): parameters that are known, TCR first:
            phi_tcr, kappa_tcr, cmthresh_tcr, ithresh_tcr, k_tcr, psi0_tcr,
            gamma_tt=1.0, phi_car, kappa_car, gamma_cc=1.0, psi0_car
        ritot (list of 3 floats): R_tot_tcr, ITp, R_tot_car
            each R_tot (float): total number of receptors of some type
            ITp (float): total number of SHP-1 molecules
        nmf_fixed (list): n, m, f parameters that are fixed, namely
            n_tcr, m_tcr, f_tcr, n_car
        cd19_tau_l (list of 2 floats): CD19 (agonist here) tau and L
        cond_index (pd.MultiIndex): expecting two levels,
            TCR_Antigen_Density and TCR_Antigen (antagonist l and tau)

    Returns:
        sr_ratio (pd.Series): antagonism ratio for each case in cond_index
    """
    p_all = repackage_tcr_car_params(pvec, kmf, other_rates, ritot, nmf_fixed)
    (all_rates, tcr_rates, car_rates, ritot_vec,
    tcr_ri, car_ri, nmf_both, tcr_nmf, car_nmf, threshold_taus) = p_all

    # Evaluate CAR threshold; TCR is fixed and received as parameter
    tcr_thresh = steady_akpr_i_1ligand(tcr_rates, threshold_taus[0],
            10*tcr_ri[0], tcr_ri, tcr_nmf, large_l=True)[tcr_nmf[0]]
    car_thresh = steady_akpr_i_1ligand(car_rates, threshold_taus[1],
            10*car_ri[0], car_ri, car_nmf, large_l=True)[car_nmf[0]]

    # Agonist alone output
    ag_alone = steady_akpr_i_1ligand(car_rates, *cd19_tau_l, car_ri, car_nmf)[car_nmf[0]]
    ag_alone = activation_function(ag_alone, car_thresh)

    inames = cond_index.names
    sr_ratio = pd.Series(np.zeros(len(cond_index)), index=cond_index)
    # Now, for each condition, compute model output for the mixture
    for k in cond_index:
        l_tcr, tau_tcr = k[-2], k[-1]  # last two levels should be L, tau
        taus = np.asarray([tau_tcr, cd19_tau_l[0]])
        lvec = np.asarray([l_tcr, cd19_tau_l[1]])
        complexes_mix = steady_akpr_i_receptor_types(all_rates, taus, lvec, ritot_vec, nmf_both)
        # Normalize outputs to compare CAR and TCR properly, accounting for
        # their very different signaling potencies.
        out_tcr = activation_function(complexes_mix[0][-1], tcr_thresh)
        out_car = activation_function(complexes_mix[1][-1], car_thresh)
        out_mix = out_tcr + out_car
        sr_ratio[k] = out_mix / ag_alone
    return sr_ratio


# Main AKPR SHP-1 cost function
def cost_antagonism_tcr_car(pvec, pbounds, kmf, other_rates, ritot,
                    nmf_fixed, cd19_tau_l, df_ratio, df_err, verbose=False):
    """
    Args:
        pvec (np.ndarray): log10 of parameters to fit, that is,
            cmthresh_car, ithresh_car, gamma_tc, gamma_ct, tau_c_tcr, tau_c_car
        pbounds (list of 2 arrays): array of lower bounds, array of upper
            bounds on the log10 of parameters
        kmf (list): k_S, m, f of CARs
        other_rates (list of parameters): parameters that are known, TCR first:
            phi_tcr, kappa_tcr, cmthresh_tcr, ithresh_tcr, k_tcr, psi0_tcr,
            gamma_tt=1.0, phi_car, kappa_car, gamma_cc=1.0, psi0_car
        ritot (list of 3 floats): R_tot_tcr, ITp, R_tot_car
            each R_tot (float): total number of receptors of some type
            ITp (float): total number of SHP-1 molecules
        nmf_fixed (list): n, m, f parameters that are fixed, namely
            n_tcr, m_tcr, f_tcr, n_car
        cd19_tau_l (list of 2 floats): CD19 (agonist here) tau and L
        df_ratio (pd.DataFrame): antagonism ratio data for a fixed agonist.
            Should have its two last index levels be the L_TCR and tau_TCR.
        df_err (pd.DataFrame): log-scale error bars on the antagonism ratios.
        verbose (bool): change default to True to print all errors raised
            by extreme parameter values sampled in the absence of bounds

    Returns:
        cost (float): total scalar cost.
    """
    # Check parameter boundaries
    if np.any(pvec < pbounds[0]) or np.any(pvec > pbounds[1]):
        return -np.inf

    # Compute antagonism ratio for each data condition
    try:
        df_ratio_model = antag_ratio_panel_tcr_car(pvec, kmf, other_rates,
                    ritot, nmf_fixed, cd19_tau_l, df_ratio.index)
    except (ValueError, RuntimeError) as e:
        ratio_dists = np.inf
        if verbose:
            print(f"Error {type(e)} with log10 parameter values "
                + "{} and m,f={}".format(pvec, kmf[1:])
                + ". Error:\n" + str(e))
    else:
        ratio_dists = 0.5 * np.sum((np.log2(df_ratio_model/df_ratio+eps_for_log)/df_err)**2)

    return -ratio_dists


def cost_antagonism_tcr_car_priors(pvec, prior_p, kmf, other_rates, ritot,
                    nmf_fixed, cd19_tau_l, df_ratio, df_err, verbose=False):
    """
    Args:
        pvec (np.ndarray): phi, cm_thresh, i_thresh, psi0
        prior_p (list of 2 arrays): array of averages, variances of priors
            for each parameter.
        kmf (list): k_I, m, f
        other_rates (list): kappa
        ritot (list of 2 floats): R_tot, I_tot
        n_p (int): N
        ag_tau (float): tau of agonist. For the MI calculation, this will be used
            as one of the two taus to distinguish.
        df_ratio (pd.DataFrame): antagonism ratio data for a fixed agonist.
            Should have its three last index levels be the L1 (3rd to last), L2 (2nd to last)
            and tau2 (last).
        df_err (pd.DataFrame): log-scale error bars on the antagonism ratios.
        verbose (bool): change default to True to print all errors raised
            by extreme parameter values sampled in the absence of bounds

    Returns:
        cost (float): total scalar cost.
    """
    # Part 1: Compute the prior probability distribution, which is log-normal,
    # i.e., a normal distribution of the log parameters.
    logprior = -0.5 * np.sum((pvec - prior_p[0])**2 / prior_p[1])

    # Part 2: compute antagonism ratio for each data condition, return total
    try:
        df_ratio_model = antag_ratio_panel_tcr_car(pvec, kmf, other_rates,
                    ritot, nmf_fixed, cd19_tau_l, df_ratio.index)
    except (ValueError, RuntimeError) as e:
        ratio_dists = np.inf
        if verbose:
            print(f"Error {type(e)} with log10 parameter values "
                + "{} and m,f={}".format(pvec, kmf[1:])
                + ". Error:\n" + str(e))
    else:
        ratio_dists = 0.5 * np.sum((np.log2(df_ratio_model/df_ratio+eps_for_log)/df_err)**2)
    return logprior - ratio_dists
