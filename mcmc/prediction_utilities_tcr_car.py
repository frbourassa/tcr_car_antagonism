""" Module with functions to compute predictions of TCR-CAR antagonism
and adjust parameters for different T cell lines based on separate data.

@author: frbourassa
May 2023
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import json
import sys, os

# Local modules
from models.tcr_car_akpr_model import (steady_akpr_i_receptor_types,
                                steady_akpr_i_1ligand, activation_function)
from mcmc.costs_tcr_car_antagonism import repackage_tcr_car_params
from utils.preprocess import (
    ln10,
    geo_mean_apply,
    geo_mean
)


# Useful constants
eps_for_log = 1e-8

### Functions for predictions ###
def compute_tcr_car_thresh(pvec, kmf, other_rates, ritot, nmf_fixed):
    """ Compute TCR and CAR thresholds given parameters formatted in the
    MCMC fashion. Consider writing the parameter rearranging code
    as a separate function if I keep adding functions here.

    Args:
        pvec, kmf, other_rates, ritot, nmf_fixed:
            see documentation of antag_ratio_panel_tcr_car

    Returns:
        tcr_thresh (float): threshold on T_N of TCR
        car_thresh (float): threshold on C_N of CAR
    """
    p_all = repackage_tcr_car_params(pvec, kmf, other_rates, ritot, nmf_fixed)
    (all_rates, tcr_rates, car_rates, ritot_vec,
    tcr_ri, car_ri, nmf_both, tcr_nmf, car_nmf, threshold_taus) = p_all

    # Evaluate CAR threshold; TCR is fixed and received as parameter
    tcr_thresh = steady_akpr_i_1ligand(tcr_rates, threshold_taus[0],
            10*tcr_ri[0], tcr_ri, tcr_nmf, large_l=True)[tcr_nmf[0]]
    car_thresh = steady_akpr_i_1ligand(car_rates, threshold_taus[1],
            10*car_ri[0], car_ri, car_nmf, large_l=True)[car_nmf[0]]
    return tcr_thresh, car_thresh


# Function where predictions are computed, given some correction factors
# for different cell lines, estimated from separate data
def antag_ratio_panel_tcr_car_predict(pvec, kmf, other_rates, ritot,
            nmf_fixed, cd19_tau_l, out_amplis, tcr_thresh_fact, cond_index):
    """
    Very similar to antag_ratio_panel_tcr_car_predict, except that
    the last two entries of pvec are directly thresholds on T_N and C_N
    rather than the ligand binding times.
    This allows to use fixed thresholds for new sets of parameters.
    Args:
        pvec (np.ndarray): log10 of parameters to fit, that is,
            cmthresh_car, ithresh_car, gamma_tc, gamma_ct, psi0_car, tcr_thresh, car_thresh
        kmf (list of 3 ints): grid search parameters, that is,
            k_{S,CAR}, m_{CAR}, f_{CAR}
        other_rates (list of parameters): parameters that are known, TCR first:
            phi_tcr, kappa_tcr, cmthresh_tcr, ithresh_tcr, k_tcr, psi0_tcr,
            gamma_tt=1.0, phi_car, kappa_car, gamma_cc=1.0
        ritot (list of 3 floats): R_tot_tcr, ITp, R_tot_car
            each R_tot (float): total number of receptors of some type
            ITp (float): total number of SHP-1 molecules
        nmf_fixed (list): n, m, f parameters that are fixed, namely
            n_tcr, m_tcr, f_tcr, n_car
        cd19_tau_l (list of 2 floats): CD19 (agonist here) tau and L
        out_amplis (list): for modified receptors, the max. amplitude of
            Z^T or Z^C can be less than 1, since with fewer ITAMs they activate
            less. Default is 1.
        tcr_thresh_fact (float): factor by which to multiply 6Y tau_c
            to get 6F's threshold. Should be 1 if doing prediction for 6Y TCR.
        cond_index (pd.MultiIndex): expecting two levels,
            TCR_Antigen_Density and TCR_Antigen (antagonist l and tau)

    Returns:
        sr_ratio (pd.Series): antagonism ratio for each case in cond_index
    """
    p_all = repackage_tcr_car_params(pvec, kmf, other_rates, ritot, nmf_fixed)
    (all_rates, tcr_rates, car_rates, ritot_vec,
    tcr_ri, car_ri, nmf_both, tcr_nmf, car_nmf, threshold_taus) = p_all

    # Evaluate TCR threshold with correction factor on threshold tau
    tcr_thresh = steady_akpr_i_1ligand(tcr_rates, threshold_taus[0]*tcr_thresh_fact,
        10*tcr_ri[0], tcr_ri, tcr_nmf, large_l=True)[tcr_nmf[0]]
    # Evaluate CAR threshold
    car_thresh = steady_akpr_i_1ligand(car_rates, threshold_taus[1],
            10*car_ri[0], car_ri, car_nmf, large_l=True)[car_nmf[0]]

    # Agonist alone output
    ag_alone = steady_akpr_i_1ligand(car_rates, *cd19_tau_l, car_ri, car_nmf)[car_nmf[0]]
    ag_alone = out_amplis[1] * activation_function(ag_alone, car_thresh)

    inames = cond_index.names
    sr_ratio = pd.Series(np.zeros(len(cond_index)), index=cond_index)
    # Now, for each condition, compute model output for the mixture
    for l_tcr, tau_tcr in cond_index:
        taus = np.asarray([tau_tcr, cd19_tau_l[0]])
        lvec = np.asarray([l_tcr, cd19_tau_l[1]])
        complexes_mix = steady_akpr_i_receptor_types(all_rates, taus, lvec, ritot_vec, nmf_both)
        # Normalize outputs to compare CAR and TCR properly, accounting for
        # their very different signaling potencies.
        out_tcr = out_amplis[0] * activation_function(complexes_mix[0][-1], tcr_thresh)
        out_car = out_amplis[1] * activation_function(complexes_mix[1][-1], car_thresh)
        out_mix = out_tcr + out_car
        sr_ratio[(l_tcr, tau_tcr)] = out_mix / ag_alone
    return sr_ratio


def find_1itam_car_ampli(df_cd19only, pvec, kmf, other_rates, ritot,
                                                nmf_fixed, cd19_tau_l):
    """ Find amplitude of output of 1-ITAM CAR compared to 3-ITAM CAR,
    based on data and outputs of the model for CD19 only.
    Args:
        df_cd19only (pd.DataFrame): raw cytokine dataframe, sliced for CD19
            only and no TCR antigen data points.
        pvec, kmf, other_rates, ritot, nmf_fixed, cd19_tau_l: as for
            e.g. antagonism ratio panel functions

    Returns:
        max_ampli (float): ratio of response of 1-ITAM CAR over 3-ITAM CAR
    """
    ## 1. Compute ratio of responses to CD19 for 1- and 3-ITAM CARs
    data_dir = os.path.join("data", "dataset_selection.json")
    if not os.path.split(os.getcwd())[-1] == "car_tcr_dev":  # we are in subfolder
        data_dir = os.path.join("..", data_dir)
    with open(data_dir, "r") as h:
        good_dsets = json.load(h).get("good_car_tcr_datasets")
    df = df_cd19only.loc[df_cd19only.index.isin(good_dsets, level="Data")]

    # Levels left in this df: "Cytokine", "Tumor", "TCR_ITAMs", "CAR_ITAMs",
    # "TCR_Antigen_Density", "Time"
    # Look at coupling with both 4- and 10-ITAM TCR
    df = (df.xs("E2APBX", level="Tumor")
            .xs("IL-2", level="Cytokine")
            .stack("Time"))
    # Look at the ratio of logs over time, averaged across times and repeats
    df = df.groupby("CAR_ITAMs").apply(geo_mean_apply)
    r13 = df["1"] / df["3"]

    ## 2. Compute the output for 3-ITAM CAR
    p_all = repackage_tcr_car_params(pvec, kmf, other_rates, ritot, nmf_fixed)
    (all_rates, tcr_rates, car_rates, ritot_vec,
    tcr_ri, car_ri, nmf_both, tcr_nmf, car_nmf, threshold_taus) = p_all
    # output for CD19 alone
    cn3 = steady_akpr_i_1ligand(car_rates, *cd19_tau_l, car_ri, car_nmf)
    cn3 = cn3[car_nmf[0]]
    # Threshold on CN for 3-ITAM CAR
    thresh3 = steady_akpr_i_1ligand(car_rates, threshold_taus[1],
            10*car_ri[0], car_ri, car_nmf, large_l=True)[car_nmf[0]]
    z3 = activation_function(cn3, thresh3)

    ## 3. Compute the Hill function output for 1-ITAM CAR
    car_nmf_1 = (1, 1, 1)
    cn1 = steady_akpr_i_1ligand(car_rates, *cd19_tau_l, car_ri, car_nmf_1)[1]
    thresh1 = steady_akpr_i_1ligand(car_rates, threshold_taus[1], 10*car_ri[0],
                        car_ri, car_nmf_1, large_l=True)[1]
    z1 = activation_function(cn1, thresh1)

    ## 4. Compute the relative amplitude the 6F Hill function must have
    # to match the ratio r13 found from data.
    rel_ampli_1itam = r13 * z3 / z1
    return rel_ampli_1itam


def find_1itam_effect_tcr_ampli(df_tcr_only):
    """ Find relative amplitude of TCR output in the presence of 1-ITAM CAR
    but without CAR antigen, compared to TCR output in the presence of 3-ITAM
    CAR. This looks at response to TCR antigen only, no CD19: separate
    data from what we are trying to predict.

    We can't just account for this by saying the 1-ITAM CAR antagonizes
    less the TCR, since:
        - This happens in the absence of CAR stimulation: just the presence
            of the CAR does that
        - This effect is significant in only one of the two datasets,
            so unsure it's really biological, but we need to correct for that
            to match the data.

    This amplitude factor effect will be multiplied with the 6F TCR amplitude
    effect for the double perturbed condition.

    We take the geometric average of this factor across 6F and 6Y cells.
    Just as for 6F correction, we average across CAR types.

    Args:
        df_tcr_only (pd.DataFrame): raw cytokine dataframe, sliced no CAR Ag

    Returns:
        effect_tcr_ampli_1itam (float): ratio of max. response of TCR with
            1 ITAM CAR over response with 3-ITAM CAR, averaged
            across 6Y and 6F TCR genotypes.
    """
    # Select relevant data
    data_dir = os.path.join("data", "dataset_selection.json")
    if not os.path.split(os.getcwd())[-1] == "car_tcr_dev":  # we are in subfolder
        data_dir = os.path.join("..", data_dir)
    with open(data_dir, "r") as h:
        good_dsets = json.load(h).get("good_car_tcr_datasets")
    df = df_tcr_only.loc[df_tcr_only.index.isin(good_dsets, level="Data")]

    # Levels left in this df: "Cytokine", "Tumor", "TCR_ITAMs", "CAR_ITAMs",
    #  "TCR_Antigen", "TCR_Antigen_Density", "Time".
    df = (df.xs("1uM", level="TCR_Antigen_Density")
            .xs("E2APBX", level="Tumor")
            .xs("IL-2", level="Cytokine")
            .stack("Time"))
    # Compute max amplitude based on A2 and N4 peptides, which both saturate
    # the TCR response in blast (mock) T cells
    df = df.loc[df.index.isin(['N4', 'A2'], level="TCR_Antigen")]
    # Look at the ratio of averaged logs across times, repeats and agonists
    df = df.groupby("CAR_ITAMs").apply(geo_mean_apply)
    # Taking the geometric average before or after ratios: no difference
    # since geo(a/b) = geo(a)/geo(b).
    effect_tcr_ampli_1itam = df["1"] / df["3"]
    return effect_tcr_ampli_1itam


def find_6f_tcr_ampli(df_tcr_only):
    """ Find amplitude of output of 6F TCR compared to 6Y CAR,
    based on mock CAR T cell data. This is model-independent,
    just comparing amplitude of response to CD19 only in both CAR types.
    Args:
        df_tcr_only (pd.DataFrame): raw cytokine dataframe, sliced no CAR Ag
    Returns:
        relative_ampli_6f (float): ratio of max. response of 6F TCR over 6Y TCR
    """
    # Select relevant data
    data_dir = os.path.join("data", "dataset_selection.json")
    if not os.path.split(os.getcwd())[-1] == "car_tcr_dev":  # we are in subfolder
        data_dir = os.path.join("..", data_dir)
    with open(data_dir, "r") as h:
        good_dsets = json.load(h).get("good_car_tcr_datasets")
    df = df_tcr_only.loc[df_tcr_only.index.isin(good_dsets, level="Data")]

    # Levels left in this df: "Cytokine", "Tumor", "TCR_ITAMs", "CAR_ITAMs",
    # "TCR_Antigen", "TCR_Antigen_Density", "Time".
    df = (df.xs("1uM", level="TCR_Antigen_Density")
            .xs("E2APBX", level="Tumor")
            .xs("IL-2", level="Cytokine")
            .stack("Time"))
    # Compute max amplitude based on A2 and N4 peptides, which both saturate
    # the TCR response in blast (mock) T cells
    df = df.loc[df.index.isin(['N4', 'A2'], level="TCR_Antigen")]
    # Look at the ratio of averaged logs across times, repeats and agonists
    df = df.groupby("TCR_ITAMs").apply(geo_mean_apply)
    relative_ampli_6f = df["4"] / df["10"]
    return relative_ampli_6f


def loglin_hill_fit(x, a, b, h, k):
    """ Given x, return log(Hill(x)). Fit a, b, h parameters """
    xnormk = (x / h)**k
    hill = a * xnormk / (xnormk + 1.0) + b
    return np.log(hill)


def find_6f_tcr_thresh_fact(df_tcr_only, pep_tau_map):
    """ Find ratio of TCR tau threshold of 6F vs 6Y, based on a Hill fit of cytokine
    versus peptide binding time for both TCR types.
    These thresholds may be different from those used to computed threshold
    on C_N, because C_N is a complicated function of tau; yet we assume
    the relative tau thresholds between 6F and 6Y is approx. conserved.
    This is simpler than computing C_N for both receptor types.

    Args:
        df_tcr_only (pd.DataFrame)
        pep_tau_map (dict)
    """
    # First, get appropriate data: response vs TCR Ag
    data_dir = os.path.join("data", "dataset_selection.json")
    if not os.path.split(os.getcwd())[-1] == "car_tcr_dev":  # we are in subfolder
        data_dir = os.path.join("..", data_dir)
    with open(data_dir, "r") as h:
        good_dsets = json.load(h).get("good_car_tcr_datasets")
    df = df_tcr_only.loc[df_tcr_only.index.isin(good_dsets, level="Data")]
    data_spleen_lvl = pd.Series(df.index.get_level_values("Data")
                                + "-" + df.index.get_level_values("Spleen"),
                                name="Data-spleen")
    df = df.set_index(data_spleen_lvl, append=True)
    df = df.droplevel(["Data", "Spleen"])

    # Levels left in this df: "Cytokine", "Tumor", "CAR_ITAMs", "TCR_Antigen",
    # "TCR_Antigen_Density", "Time", "Data-spleen".
    df = (df.xs("1uM", level="TCR_Antigen_Density")
            .xs("E2APBX", level="Tumor")
            .xs("IL-2", level="Cytokine"))
    # Geometric average of IL-2 over time is what we fit against tau
    df = geo_mean(df, axis=1)
    # Convert peptide names to taus
    df = df.rename(pep_tau_map, level="TCR_Antigen")

    # Fit a Hill curve on each data-spleen replicate for each TCR ITAM number
    df = df.reorder_levels(["Data-spleen", "TCR_ITAMs", "CAR_ITAMs", "TCR_Antigen"])
    gps = df.groupby(["Data-spleen", "TCR_ITAMs", "CAR_ITAMs"])
    tau_threshs = pd.Series(np.zeros(len(gps.groups)),
            index=pd.MultiIndex.from_tuples(gps.groups.keys(), names=gps.keys))
    for k in gps.groups.keys():
        lower, upper = df.loc[k].min(), df.loc[k].max()
        ampli = upper - lower
        # Limits on ampli, background, tau_thresh, hill power k
        pbounds = [(0.25*ampli, 4.0*ampli), (0.8*lower, 1.2*lower), (0.1, 10.0), (1.0, 16.0)]
        pbounds = np.asarray(list(zip(*pbounds)))
        p0 = (pbounds[0] + pbounds[1]) / 2.0
        popt, _ = curve_fit(loglin_hill_fit, xdata=df.loc[k].index.get_level_values("TCR_Antigen").values,
                           ydata=np.log(df.loc[k].values), p0=p0, bounds=pbounds)
        tau_threshs.loc[k] = popt[2]

    # Compute average tau threshold for 6F and 6Y, return their difference
    tau_threshs = tau_threshs.groupby("TCR_ITAMs").mean()
    tau_threshs.name = "tau^T_c"
    # Also return the actual fitted thresholds
    return tau_threshs["4"] / tau_threshs["10"], tau_threshs
