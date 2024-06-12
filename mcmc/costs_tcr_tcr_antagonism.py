""" Module with cost functions to fit antagonism ratio with TCR-TCR receptor
models.

@author: frbourassa
November 2022
"""
import numpy as np
import pandas as pd

# Local modules
from models.shp1_model import steady_shp1_1ligand, steady_shp1_2ligands
from models.akpr_i_model import (steady_akpr_i_1ligand,
                                        steady_akpr_i_2ligands)
# Useful constants
from utils.preprocess import ln10, eps_for_log


### AKPR I model ###
def antag_ratio_panel_akpr_i(pvec, kmf, other_rates, ritot, n_p, ag_tau, cond_index):
    # Assemble rates in correct order: phi, kappa, cm_thresh, i_thresh, k_i, psi_0
    expvec = np.exp(pvec*ln10)
    all_rates = [expvec[0]] + list(other_rates[:1]) + list(expvec[1:3]) + [kmf[0], expvec[3]]
    nmf = (n_p,) + tuple(kmf[1:])

    inames = cond_index.names
    # AgonistConcentration, AntagonistConcentration, Antagonist
    df_ratio = pd.Series(np.zeros(len(cond_index)), index=cond_index)
    ag_alone = {}
    for l_ag in cond_index.get_level_values(inames[0]).unique():
        ag_alone[l_ag] = steady_akpr_i_1ligand(all_rates, ag_tau, l_ag, ritot, nmf)[n_p]
    # Now, for each condition, compute model output for the mixture
    for l_ag, l_antag, antag_tau in cond_index:
        taus = np.asarray([ag_tau, antag_tau])
        lvec = np.asarray([l_ag, l_antag])
        complexes_mix = steady_akpr_i_2ligands(all_rates, taus, lvec, ritot, nmf)
        out_mix = complexes_mix[nmf[0]] + complexes_mix[2*nmf[0]+1]
        df_ratio[(l_ag, l_antag, antag_tau)] = out_mix / ag_alone[l_ag]
    return df_ratio


# Main AKPR I cost function
def cost_antagonism_akpr_i(pvec, pbounds, kmf, other_rates, ritot, n_p,
    ag_tau, df_ratio, df_err, weight_smallagconc=3.0):
    """
    Args:
        pvec (np.ndarray): phi, cm_thresh, i_thresh, psi0
        pbounds (list of 2 arrays): array of lower bounds, array of upper
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

    Returns:
        cost (float): total scalar cost.
    """
    # Check parameter boundaries
    if np.any(pvec < pbounds[0]) or np.any(pvec > pbounds[1]):
        return -np.inf
    # Part 1: compute antagonism ratio for each data condition
    # For each agonist L, compute agonist alone output
    # Then for each L2, tau2, compute the ratio
    n_data = df_ratio.shape[0]
    try:
        df_ratio_model = antag_ratio_panel_akpr_i(pvec, kmf, other_rates,
                            ritot, n_p, ag_tau, df_ratio.index)
    except ValueError as e:
        print(e)
        ratio_dists = np.inf
        print("Error with parameter values {} and m,f={}".format(pvec, kmf[1:]))
    else:
        bonus_lvl = "AgonistConcentration"
        smallagconc = df_ratio.index.get_level_values(bonus_lvl).min()
        ratios = (np.log2(df_ratio_model/df_ratio+eps_for_log)/df_err)**2
        ratios.loc[ratios.index.isin([smallagconc], level=bonus_lvl)] *= weight_smallagconc
        ratio_dists = np.sum(ratios) / n_data

    return -ratio_dists


### Francois 2013 model ###
def antag_ratio_panel_shp1(pvec, mp, other_rates, rtot, n_p, ag_tau, cond_index):
    # Assemble rates in correct order: phi, b, gamma, kappa, cm_thresh
    expvec = np.exp(pvec*ln10)
    all_rates = [expvec[0]] + list(other_rates) + [expvec[1]]
    ritot = [rtot, expvec[2]]
    nm = (n_p, mp[0])

    inames = cond_index.names
    df_ratio = pd.Series(np.zeros(len(cond_index)), index=cond_index)
    ag_alone = {}
    for l_ag in cond_index.get_level_values(inames[0]).unique():
        ag_alone[l_ag] = steady_shp1_1ligand(all_rates, ag_tau, l_ag, ritot, nm)[nm[0]]
    # Now, for each condition, compute model output for the mixture
    for l_ag, l_antag, antag_tau in cond_index:
        taus = [ag_tau, antag_tau]
        lvec = [l_ag, l_antag]
        complexes_mix = steady_shp1_2ligands(all_rates, *taus, *lvec, ritot, nm)
        out_mix = complexes_mix[nm[0]] + complexes_mix[2*nm[0]+1]
        df_ratio[(l_ag, l_antag, antag_tau)] = out_mix / ag_alone[l_ag]
    return df_ratio


# Main Francois 2013 model cost function
def cost_antagonism_shp1(pvec, pbounds, mp, other_rates, rtot, n_p, ag_tau,
    df_ratio, df_err, weight_smallagconc=3.0):
    """
    Args:
        pvec (np.ndarray): phi, cm_thresh, itot
        pbounds (list of 2 arrays): array of lower bounds, array of upper
        mp (list of 1 int): [m]
        other_rates (list): b, gamma, kappa
        rtot (float): R_tot
        n_p (int): N
        ag_tau (float): tau of agonist. For the MI calculation, this will be used
            as one of the two taus to distinguish.
        df_ratio (pd.DataFrame): antagonism ratio data for a fixed agonist.
            Should have its three last index levels be the L1 (3rd to last), L2 (2nd to last)
            and tau2 (last).
        df_err (pd.DataFrame): log-scale error bars on the antagonism ratios.

    Returns:
        cost (float): total scalar cost.
    """
    # Check parameter boundaries
    if np.any(pvec < pbounds[0]) or np.any(pvec > pbounds[1]):
        return -np.inf
    n_data = df_ratio.shape[0]
    # Part 1: compute antagonism ratio for each data condition
    # For each agonist L, compute agonist alone output
    # Then for each L2, tau2, compute the ratio
    try:
        df_ratio_model = antag_ratio_panel_shp1(pvec, mp, other_rates, rtot,
                            n_p, ag_tau, df_ratio.index)
    except ValueError:
        ratio_dists = np.inf
        print("Error with parameter values {} and m={}".format(pvec, mp))
    else:
        bonus_lvl = "AgonistConcentration"
        smallagconc = df_ratio.index.get_level_values(bonus_lvl).min()
        ratios = (np.log2(df_ratio_model/df_ratio+eps_for_log)/df_err)**2
        ratios.loc[ratios.index.isin([smallagconc], level=bonus_lvl)] *= weight_smallagconc
        ratio_dists = np.sum(ratios) / n_data
    return -ratio_dists  # log likelihood, least negative is best.
