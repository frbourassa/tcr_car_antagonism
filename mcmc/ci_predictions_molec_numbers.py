"""
Function for confidence intervals on model predictions in other systems.
Similar to confidence_model_antagonism as defined in
mcmc.utilities_tcr_tcr_antagonism, but tweaked to facilitate TCR/CAR model
predictions with different TCR and APC types.

We define a special model panel function that computes thresholds with default
receptor numbers (for OT1-CD19-CAR T cells), because thresholds are thresholds
on signaling proteins, so they should not depend on the receptor numbers of
other CAR T cell types. Hence, this model panel function takes default receptor
numbers and updated ones.

However, for propagation of uncertainties on receptor numbers, the default
receptor numbers' uncertainties should be included in the threshold computation.
So we should vary within their uncertainties both the default and new cell
type receptor numbers standard deviations for sampling before calling this
model panel.

@author: frbourassa
June 2024
"""
import numpy as np
import pandas as pd
from multiprocessing import Pool

from mcmc.costs_tcr_car_antagonism import (
    antag_ratio_panel_tcr_car,
    repackage_tcr_car_params
)
from mcmc.prediction_utilities_tcr_car import antag_ratio_panel_tcr_car_predict
from utils.preprocess import michaelis_menten
from models.tcr_car_akpr_model import (
    steady_akpr_i_receptor_types,
    activation_function
)
from models.akpr_i_model import steady_akpr_i_1ligand
from models.conversion import convert_ec50_tau_relative
from mcmc.utilities_tcr_tcr_antagonism import (
    sample_log_student,
    sample_lognorm
)

from utils.cpu_affinity import count_parallel_cpu
n_cpu = count_parallel_cpu()



### TCR-CAR REVISED AKPR model, panel function dealing properly with
### surface molecule number changes
def antag_ratio_panel_tcr_car_numbers(
        pvec, kmf, other_rates, ritot, nmf_fixed, cd19_tau_l,
        baseline, scale_car_thresh, ritot_default, cond_index
    ):
    """
    Thresholds are computed with default receptor numbers,
    outputs are computed with the new cell lines' receptor numbers.
    Both numbers should be sampled within uncertainties to assess
    model confidence intervals fully.

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
        baseline (float): background activation level to add to Z^tot,
            pass 0.0 for default value.
        scale_car_thresh (float): to scale up or down the default CAR
            threshold, to account for strength of CAR signal depending
            on the experimental readout being proximal or distal. Pass 1.0
            for default value.
        ritot_default (list of 3 floats): default receptor numbers,
            used to compute the activation thresholds. Pass the same as
            ritot for default behavior.
        cond_index (pd.MultiIndex): expecting two levels,
            TCR_Antigen_Density and TCR_Antigen (antagonist l and tau)
    Returns:
        sr_ratio (pd.Series): antagonism ratio for each case in cond_index
    """
    if ritot_default is None:
        ritot_default = ritot
    p_all = repackage_tcr_car_params(pvec, kmf, other_rates, ritot_default, nmf_fixed)
    (all_rates, tcr_rates, car_rates, ritot_vec,
    tcr_ri, car_ri, nmf_both, tcr_nmf, car_nmf, threshold_taus) = p_all

    # Evaluate CAR threshold; TCR is fixed and received as parameter
    tcr_thresh = steady_akpr_i_1ligand(tcr_rates, threshold_taus[0],
            10*tcr_ri[0], tcr_ri, tcr_nmf, large_l=True)[tcr_nmf[0]]
    car_thresh = steady_akpr_i_1ligand(car_rates, threshold_taus[1],
            10*car_ri[0], car_ri, car_nmf, large_l=True)[car_nmf[0]]
    car_thresh *= scale_car_thresh

    # Parameters again, now with actual cell type receptor levels.
    p_all = repackage_tcr_car_params(pvec, kmf, other_rates, ritot, nmf_fixed)
    (all_rates, tcr_rates, car_rates, ritot_vec,
    tcr_ri, car_ri, nmf_both, tcr_nmf, car_nmf, threshold_taus) = p_all

    # Agonist alone output
    ag_alone = steady_akpr_i_1ligand(car_rates, *cd19_tau_l, car_ri, car_nmf)[car_nmf[0]]
    ag_alone = activation_function(ag_alone, car_thresh)

    inames = cond_index.names
    sr_ratio = pd.Series(np.zeros(len(cond_index)), index=cond_index)
    # Now, for each condition, compute model output for the mixture
    for k in cond_index:
        l_tcr, tau_tcr = k[-2], k[-1]
        taus = np.asarray([tau_tcr, cd19_tau_l[0]])
        lvec = np.asarray([l_tcr, cd19_tau_l[1]])
        complexes_mix = steady_akpr_i_receptor_types(all_rates, taus, lvec, ritot_vec, nmf_both)
        # Normalize outputs to compare CAR and TCR properly, accounting for
        # their very different signaling potencies.
        out_tcr = activation_function(complexes_mix[0][-1], tcr_thresh)
        out_car = activation_function(complexes_mix[1][-1], car_thresh)
        out_mix = out_tcr + out_car
        sr_ratio[k] = (out_mix + baseline) / (ag_alone + baseline)
    return sr_ratio


def compute_stats_ci(df_samples, series_best):
    """ Default statistical analysis of MCMC model samples,
    computing percentiles, median, mean, geometric mean, and
    aggregating with the best fit.
    """
    # Compute statistics of the ratios on a log scale, then back to linear scale
    stats = ["percentile_2.5", "median", "mean", "geo_mean", "best", "percentile_97.5"]
    df_stats = pd.DataFrame(np.zeros([df_samples.shape[0], len(stats)]),
                index=df_samples.index, columns=stats)
    df_stats["mean"] = np.log(df_samples.mean(axis=1))
    df_samples_log = np.log(df_samples)
    df_stats["percentile_2.5"] = df_samples_log.quantile(q=0.025, axis=1)
    df_stats["median"] = df_samples_log.quantile(q=0.5, axis=1)
    df_stats["geo_mean"] = df_samples_log.mean(axis=1)
    df_stats["best"] = np.log(series_best)
    df_stats["percentile_97.5"] = df_samples_log.quantile(q=0.975, axis=1)
    df_stats = np.exp(df_stats)
    return df_stats



def confidence_predictions_car_antagonism(
        model_panel, psamples, pbest, grid_pt, cond_index, **kwargs
    ):
    """ Receive parameter samples and a point on the grid search,
    as well as a function evaluating antagonism panel for a TCR/CAR model,
    and compute the confidence interval of model predictions compared to data.
    Uses multiprocessing to speed up sampling.

    The CAR parameters and receptor numbers are given in the cell_info
    dictionary; this function takes care of modifying default other_args
    according to the model panel function chosen.

    Not sure why I coded all this, since in the end, for all CAR T cell lines
    used in the paper (OT-1, NY-ESO, p8F, 6-11, 4196A), we used the OT-1 CAR
    receptor numbers, since they were similar in all T cell lines.

    Args:
        model_panel (callable): function evaluating model antagonism ratio
            for given (pvec, grid_pt, *other_args, df_ratio.index).
        psamples (np.ndarray): sampled parameters array,
            shaped [param, nwalkers, nsamples]
        pbest (np.ndarray): best parameter sample
        grid_pt (tuple of ints): integer parameters after grid search.
        cond_index (pd.MultiIndex): index of conditions for which to
            compute model FC. The last level should be pulse concentration,
            previous levels should specify the peptide to recover its
            logEC50s' mean and standard deviation.
            For model panel calculations, one level will be appended,
            to specify the L corresponding to the peptide pulse concentration.

    Keyword args:
        analysis_fct (callable): function taking (df_model, sr_best) as
            arguments and computing the aggregate statistics across MCMC
            samples that are returned by this function.
        other_args (tuple): other arguments passed to model_panel.
            Will be modified locally at each generated sample to contain
            the currently sampled L, tau, receptor numbers, etc.
        seed (np.random.SeedSequence or int): random number generator seed
        n_samp (int): number of samples to draw for confidence intervals
        antagonist_lvl (str): which level contains the TCR antagonist taus.
        tcr_pulse_lvl (str): level which contains the TCR antigen pulse
            concentration levels, should be last in the index levels order
        cell_info (dict): non-default receptor or ligand numbers, may have:
            tcr_number, car_number, car_ag_level, l_conc_mm_params
        molec_stds (list of 5 floats): standard deviation of the mean estimators
            for the TCR, CAR, CD19, MHC, and loading K_D
        molec_dofs (list of 5 ints): the number of degrees of freedom
            (cell or experimental repeats) from which the mean estimators
            for the TCR, CAR, CAR Ag, MHC, and loading K_D are calculated.
        baseline_std (float): standard deviation of the baseline output level,
            in log scale.
        scale_car_thresh_std (float): standard deviation of the CAR threshold
            scale factor, in log scale.
        ritot_default_stds_dofs (list of 2 lists of floats):
            first list: standard deviations of the
                default mean numbers of TCR and CAR receptors, in log scale.
            second list: number of degrees of freedom from which means
                have been estimated, for Student't distribution sampling.
        factors_stds (list of [list, float]): standard deviation of the
            estimated correction factors for [tcr_ampli, car_ampli]
            and TCR threshold. Not implemented.

    Returns:
        df_stats (pd.DataFrame): statistics on model antagonism predictions,
            including 90 % CI, median, best fit, as a function of tau.
    """
    # Get kwargs
    analysis_fct = kwargs.get("analysis_fct", compute_stats_ci)
    other_args = kwargs.get("other_args", ())
    seed = kwargs.get("seed", None)
    n_samp = kwargs.get("n_samp", 1000)
    tcr_pulse_lvl = kwargs.get("tcr_pulse_lvl", "TCR_Antigen_Pulse_uM")
    antagonist_lvl = kwargs.get("antagonist_lvl", "TCR_Antigen_tau")
    cell_info = kwargs.get("cell_info", {})
    molec_stds = kwargs.get("molec_stds", [0.0,]*5)
    molec_dofs = kwargs.get("molec_dofs", [1000,]*5)
    baseline_std = kwargs.get("baseline_std", 0.0)
    scale_car_thresh_std = kwargs.get("scale_car_thresh_std", 0.0)
    ritot_default_stds_dofs = kwargs.get("ritot_default_stds_dofs", [0.0,]*2)
    if kwargs.get("factors_stds", None) is not None:
        raise NotImplementedError("Did not implement prediction factors")

    # Initialization
    rgen = np.random.default_rng(seed)
    l_lvl_name = "TCR_Antigen_Density"
    tau_lvl_name = antagonist_lvl  # These should be taus directly.

    # Extract cell type information
    tcr_number = cell_info.get("tcr", None)
    car_number = cell_info.get("car", None)
    car_ag_number = cell_info.get("car_ag", None)
    # l_conc_mm_params contains the MHC number and loading EC50.
    l_conc_mm_params = cell_info.get("l_conc_mm_params", [1e5, 0.1])

    # Extract arguments according to the chosen panel function
    # If values not provided, get defaults (from OT1 and E2aPBX fits)
    error_msg = ("If receptor numbers change, use"
        + "antag_ratio_panel_tcr_car_numbers to ensure TCR & CAR"
        + "thresholds are computed with default receptor numbers")
    if model_panel == antag_ratio_panel_tcr_car:
        # ritot_ref: TCR number, I_tot, CAR number
        other_rates, ritot_ref, nmf_fixed, car_ag_tau_l = other_args
        if tcr_number is not None or car_number is not None:
            raise ValueError(error_msg)
    elif model_panel == antag_ratio_panel_tcr_car_predict:
        (other_rates, ritot_ref, nmf_fixed,
            car_ag_tau_l, out_amplis, tcr_thresh_fact) = other_args
        if tcr_number is not None or car_number is not None:
            raise ValueError(error_msg)
    elif model_panel == antag_ratio_panel_tcr_car_numbers:
        (other_rates, ritot_ref, nmf_fixed, car_ag_tau_l,
            baseline, scale_car_thresh, ritot_default) = other_args
    else:
        raise NotImplementedError("Not coded: {}".format(model_panel.__name__))
    if car_ag_number is None:
        car_ag_number = car_ag_tau_l[1]
        print("Loaded default CAR antigen number:", car_ag_number)
    if tcr_number is None:
        tcr_number = ritot_ref[0]
        print("Loaded default TCR number:", tcr_number)
    if car_number is None:
        car_number = ritot_ref[2]  # R_tcr, I_tot, R_car
        print("Loaded default CAR number:", car_number)

    # Build the new MultiIndex for model predictions
    # In the new index used to compute model outputs at each sample,
    # add levels for pulse concentrations to numbers, since numbers will
    # change for each MCMC sample of MHC number and loading EC50.
    cond_index = cond_index.sort_values()
    df_idx = cond_index.copy().to_frame()
    # For now, use default/best fit mm_params, but these L values
    # will be replaced for each sample.
    # Order of the last two levels should be L_tcr, tau_tcr
    error_msg =  "Need a TCR antigen pulse concentration level in the input"
    assert tcr_pulse_lvl in cond_index.names, error_msg
    df_idx[l_lvl_name] = michaelis_menten(df_idx[tcr_pulse_lvl], *l_conc_mm_params)

    new_cols = pd.RangeIndex(n_samp)
    df_model = pd.DataFrame(np.zeros([df_idx.shape[0], n_samp]),
                    index=cond_index, columns=new_cols)

    # Replace the ri_tot default by a local list to be modified each iteration
    # For TCR/CAR, other_args = list(cost_args_loaded) + extra args specific to panel
    # and cost_args_loaded = tcr_car_params, tcr_car_ritots, tcr_car_nmf, cd19_tau_l
    # and tcr_car_params = phi_tcr, kappa_tcr, cmthresh_tcr, S0p, kp, psi0_tcr, gamma_tt
    # then phi_car, kappa_car, gamma_cc, psi0_car
    # and tcr_car_ritots = [r_tcr, r_car, I_tot]
    # The extra panel fct args: for predictions, [[tcr_ampli, car_ampli], tcr_thresh] factors
    # For receptor numbers changing: [baseline, scale_car_thresh, ritot_default]
    # Replace elements in other_args by local copies
    other_args_ci = list(other_args)  # Local list, elements in it are refs
    other_args_ci[1] = [tcr_number, ritot_ref[1], car_number]  # ri_tots list: r_tcr, i_tot, r_car
    other_args_ci[3] = [car_ag_tau_l[0], car_ag_number]  # cd19_tau_l
    if model_panel == antag_ratio_panel_tcr_car_numbers:
        other_args_ci[6] = list(other_args_ci[6])  # ritot_default

    # Compute model output for best parameter vector.
    extra_lvl_names = list(cond_index.names)
    try:
        extra_lvl_names.remove(tau_lvl_name)
    except ValueError:
        extra_lvls_to_drop = [tau_lvl_name, l_lvl_name]
    else:
        extra_lvls_to_drop = [l_lvl_name]

    full_idx_order = extra_lvl_names + [l_lvl_name, tau_lvl_name]
    try:
        new_index = pd.MultiIndex.from_frame(df_idx[full_idx_order])
        sr_best = model_panel(pbest, grid_pt, *other_args_ci, new_index)
        sr_best = sr_best.droplevel(extra_lvls_to_drop)
    except RuntimeError:
        sr_best = np.nan

    # Randomly sample parameters from the MCMC chain
    samp_choices = rgen.choice(psamples.shape[1]*psamples.shape[2],
                    size=n_samp, replace=True)
    s_choices = samp_choices // psamples.shape[1]
    w_choices = samp_choices % psamples.shape[1]

    # Student's t samples for the TCR, CAR, CD19, MHC, and K_D mean estimators
    tcr_num_std, car_num_std, car_ag_l_std, max_mhc_std, kd_std = molec_stds
    n_dofs_tcr, n_dofs_car, n_dofs_cd19, n_dofs_mhc, n_dofs_kd = molec_dofs
    tcr_num_samples = sample_log_student(
        tcr_number, tcr_num_std, rgen, n_samp, n_dofs_tcr, base=10.0
    )
    car_num_samples = sample_log_student(
        car_number, car_num_std, rgen, n_samp, n_dofs_car, base=10.0
    )
    car_ag_num_samples = sample_log_student(
        car_ag_number, car_ag_l_std, rgen, n_samp, n_dofs_cd19, base=10.0
    )
    mhc_num_samples = sample_log_student(
        l_conc_mm_params[0], max_mhc_std, rgen, n_samp, n_dofs_mhc, base=10.0
    )
    load_kd_samples = sample_log_student(
        l_conc_mm_params[1], kd_std, rgen, n_samp, n_dofs_kd, base=10.0
    )

    # Also, sample extra args like baseline and ritot_default
    if model_panel == antag_ratio_panel_tcr_car_numbers:
        baseline_samples = sample_lognorm(
            baseline, baseline_std, rgen, n_samp, base=10.0
        )
        scale_car_thresh_samples = sample_lognorm(
            scale_car_thresh, scale_car_thresh_std, rgen, n_samp, base=10.0
        )
        tcr_default_samples = sample_log_student(
            ritot_default[0], ritot_default_stds_dofs[0][0], rgen, n_samp,
            ritot_default_stds_dofs[1][0], base=10.0
        )
        car_default_samples = sample_log_student(
            ritot_default[2], ritot_default_stds_dofs[0][1], rgen, n_samp,
            ritot_default_stds_dofs[1][1], base=10.0
        )

    # Did not implement ITAM factors, here would be the place to sample them

    # Compute the antagonism ratio panel for each sample parameter set
    pool = Pool(min(n_cpu, n_samp))
    all_procs = {}
    # Launch all processes
    for i in range(n_samp):
        nw, ns = w_choices[i], s_choices[i]
        pvec = psamples[:, nw, ns]
        # Place new TCR, CAR, CD19 numbers in other_args_ci
        # ri_tots: TCR, I_tot, CAR
        other_args_ci[1] = [tcr_num_samples[i], other_args[1][1], car_num_samples[i]]
        other_args_ci[3] = (other_args[3][0], car_ag_num_samples[i])  # car_ag_tau_l
        # Prepare the new index with the sampled mhc_num and load_kd
        df_idx[l_lvl_name] = michaelis_menten(
            df_idx[tcr_pulse_lvl], mhc_num_samples[i], load_kd_samples[i]
        )
        # Default ri_tots for threshold calculation, and other such factors
        if model_panel == antag_ratio_panel_tcr_car_numbers:
            other_args_ci[4] = baseline_samples[i]
            other_args_ci[5] = scale_car_thresh_samples[i]
            other_args_ci[6] = (tcr_default_samples[i], ritot_default[1],
                                car_default_samples[i])
        # Using df_idx, create the full index with TCR antigen L and tau
        # as the last two levels
        new_index_ci = df_idx.set_index(full_idx_order).index
        # Use this index containing all sampled parameters to compute predictions
        all_procs[i] = pool.apply_async(
            model_panel, args=(pvec, grid_pt, *other_args_ci, new_index_ci)
        )

    # Collect the results
    for i in range(n_samp):
        try:
            res = all_procs[i].get()
        except RuntimeError:
            df_model[i] = np.nan
        else:
            # Keep tau_lvl_name, this is also antagonist_lvl, should be there
            df_model[i] = res.droplevel(extra_lvls_to_drop, axis=0)
    print("Number NaN samples for {}:".format(grid_pt),
                df_model.isna().iloc[0, :].sum())

    df_stats = analysis_fct(df_model, sr_best)
    print("Finished computing model predictions for {}".format(grid_pt))
    return df_stats


### CI prediction calculation function for peptide libraries
def replace_taus(
        df, pep_ec50s, pep_concs, conc_lvl, tau_lvl, ec50_ref, tau_ref
    ):
    """ Utility function to update the log EC50 of each peptide,
    ensuring the same EC50 is given to a peptide at all concentrations.
    This requires a bit of index wrangling to ensure the EC50 df and
    the df to update have the same indexes to be aligned.
    """
    for conc in pep_concs:
        sl = df.xs(conc, level=conc_lvl).index
        sl_isin = df.index.isin([conc], level=conc_lvl)
        df.loc[sl_isin, tau_lvl] = convert_ec50_tau_relative(
            10.0**pep_ec50s.loc[sl] / ec50_ref, tau_ref, npow=6
        )
    return df


# Can use various panel functions, and also varies peptide EC50s.
# Does not yet implement both receptor number changes and peptide libraries.
# Not needed since we assumed CAR levels on Her2-CAR T cells were
# the same as on CD19-CAR T cells. And TCR levels too, we didn't quantify them.
def confidence_predictions_car_antagonism_ligands(
        model_panel, psamples, pbest, grid_pt,
        cond_index, pep_log10ec50s, **kwargs
    ):
    """ Receive parameter samples and a point on the grid search,
    as well as a function evaluating antagonism panel for some model,
    and compute the confidence interval of model predictions compared to data,
    taking into account uncertainty on ligand densities and binding times.
    Uses multiprocessing to speed up sampling.

    The CAR parameters and receptor numbers means are given in the cell_info
    dictionary; this function takes care of modifying default other_args
    according to the model panel function chosen.

    Args:
        model_panel (callable): function evaluating model antagonism ratio
            for given (pvec, grid_pt, *other_args, df_ratio.index).
        psamples (np.ndarray): sampled parameters array,
            shaped [param, nwalkers, nsamples]
        pbest (np.ndarray): best parameter sample
        grid_pt (tuple of ints): integer parameters after grid search.
        cond_index (pd.MultiIndex): index of conditions for which to
            compute model FC. The last level should be pulse concentration,
            previous levels should specify the peptide to recover its
            logEC50s' mean and standard deviation.
            For model panel calculations, two levels will be appended,
            to specify the L and tau corresponding to the peptide
            pulse concentration and peptide's EC50.
        pep_log10ec50s (pd.DataFrame): the log_10(EC50s) and their standard deviations
            of all peptides specified in cond_index. Should have two columns,
            "estimator" and "std". Should have the same index as cond_index.

    Keyword args:
        analysis_fct (callable): function taking (df_model, sr_best) as
            arguments and computing the aggregate statistics across MCMC
            samples that are returned by this function.
        other_args (tuple): other arguments passed to model_panel.
            Will be modified locally at each generated sample to contain
            the currently sampled L, tau, receptor numbers, etc.
        seed (np.random.SeedSequence or int): random number generator seed
        n_samp (int): number of samples to draw for confidence intervals
        tcr_pulse_lvl (str): level which contains the TCR antigen pulse
            concentration levels, should be last in the index levels order
        cell_info (dict): non-default receptor or ligand numbers, may have:
            tcr_number, car_number, car_ag_level, l_conc_mm_params
        molec_stds (list of 5 floats): standard deviation of the mean estimators
            for the TCR, CAR, CD19, MHC, and loading K_D
        molec_dofs (list of 3 ints): the number of degrees of freedom
            (cell or experimental repeats) from which the mean estimators
            for the TCR, CAR, CAR Ag, MHC, and loading K_D are calculated.
        factors_stds (list of [list, float]): standard deviation of the
            estimated correction factors for [tcr_ampli, car_ampli]
            and TCR threshold.
        ec50_tau_refs (list of 2 floats): ec50_ref, tau_ref.

    Returns:
        df_stats (pd.DataFrame): statistics on model antagonism predictions,
            including 90 % CI, median, best fit, as a function of tau.
    """
    # Get kwargs
    analysis_fct = kwargs.get("analysis_fct", compute_stats_ci)
    other_args = kwargs.get("other_args", ())
    seed = kwargs.get("seed", None)
    n_samp = kwargs.get("n_samp", 1000)
    tcr_pulse_lvl = kwargs.get("tcr_pulse_lvl", "TCR_Antigen_Pulse_uM")
    cell_info = kwargs.get("cell_info", {})
    molec_stds = kwargs.get("molec_stds", [0.0,]*5)
    molec_dofs = kwargs.get("molec_dofs", [1000,]*5)
    ec50_tau_refs = kwargs.get("ec50_tau_refs", [1e-10, 10.0])
    ec50_ref, tau_ref = ec50_tau_refs
    if "factors_stds" in kwargs.keys():
        raise NotImplementedError("Did not implement prediction factors")

    # Initialization
    rgen = np.random.default_rng(seed)
    l_lvl_name = "TCR_Antigen_Density"
    tau_lvl_name = "TCR_Antigen_tau"

    # Extract cell type information
    tcr_number = cell_info.get("tcr", None)
    car_number = cell_info.get("car", None)
    car_ag_number = cell_info.get("car_ag", None)
    l_conc_mm_params = cell_info.get("l_conc_mm_params", [1e5, 0.1])

    # Extract arguments according to the chosen panel function
    # If values not provided, get defaults (from OT1 and E2aPBX fits)
    error_msg = ("If receptor numbers change, use"
        + "antag_ratio_panel_tcr_car_numbers to ensure TCR & CAR"
        + "thresholds are computed with default receptor numbers")
    error_msg2 = ("Need to implement variation of default receptor "
                  + "numbers within their uncertainties")
    if model_panel == antag_ratio_panel_tcr_car:
        # ritot_ref: TCR number, I_tot, CAR number
        other_rates, ritot_ref, nmf_fixed, car_ag_tau_l = other_args
        if tcr_number is not None or car_number is not None:
            raise ValueError(error_msg)
    elif model_panel == antag_ratio_panel_tcr_car_predict:
        (other_rates, ritot_ref, nmf_fixed,
            car_ag_tau_l, out_amplis, tcr_thresh_fact) = other_args
        if tcr_number is not None or car_number is not None:
            raise ValueError(error_msg)
    elif model_panel == antag_ratio_panel_tcr_car_numbers:
        raise NotImplementedError(error_msg2)
        (other_rates, ritot_ref, nmf_fixed, car_ag_tau_l,
            baseline, scale_car_thresh, ritot_default) = other_args
    else:
        raise NotImplementedError("Not coded: {}".format(model_panel.__name__))
    if car_ag_number is None:
        car_ag_number = car_ag_tau_l[1]
        print("Loaded default CAR antigen number:", car_ag_number)
    if tcr_number is None:
        tcr_number = ritot_ref[0]
        print("Loaded default TCR number:", tcr_number)
    if car_number is None:
        car_number = ritot_ref[2]  # R_tcr, I_tot, R_car
        print("Loaded default CAR number:", car_number)

    # Build the new MultiIndex for model predictions
    # In the new index used to compute model outputs at each sample,
    # add levels for pulse concentrations to numbers
    cond_index = cond_index.sort_values()
    df_idx = cond_index.copy().to_frame()

    # use default/best fit mm_params, after computing the best fit value,
    # these will be replaced for each sample
    # Order of these last two levels should be L_tcr, tau_tcr
    error_msg =  "Need a TCR antigen pulse concentration level in the input"
    assert tcr_pulse_lvl in cond_index.names, error_msg
    df_idx[l_lvl_name] = michaelis_menten(df_idx[tcr_pulse_lvl], *l_conc_mm_params)

    # Tau index level
    pep_concs = cond_index.get_level_values(tcr_pulse_lvl).unique()
    df_idx = replace_taus(df_idx, pep_log10ec50s["estimator"], pep_concs,
                          tcr_pulse_lvl, tau_lvl_name, ec50_ref, tau_ref)
    new_cols = pd.RangeIndex(n_samp)
    df_model = pd.DataFrame(np.zeros([df_idx.shape[0], n_samp]),
                    index=cond_index, columns=new_cols)
    # Replace the ri_tot default by a local list to be modified each iteration
    # For TCR/CAR, other_args = list(cost_args_loaded) + [[tcr_ampli, car_ampli], tcr_thresh_fact]
    # and cost_args_loaded = tcr_car_params, tcr_car_ritots, tcr_car_nmf, cd19_tau_l
    # and tcr_car_params = phi_tcr, kappa_tcr, cmthresh_tcr, S0p, kp, psi0_tcr, gamma_tt
    # then phi_car, kappa_car, gamma_cc, psi0_car
    # and tcr_car_ritots = [r_tcr, r_car, I_tot]
    # Replace elements in other_args by local copies
    other_args_ci = list(other_args)  # Local list, elements in it are refs
    other_args_ci[1] = [tcr_number, ritot_ref[1], car_number]  # ri_tots list: r_tcr, i_tot, r_car
    other_args_ci[3] = [car_ag_tau_l[0], car_ag_number]  # cd19_tau_l

    # Compute model output for best parameter vector.
    full_idx_order = list(cond_index.names) + [l_lvl_name, tau_lvl_name]
    try:
        new_index = pd.MultiIndex.from_frame(df_idx[full_idx_order])
        sr_best = model_panel(pbest, grid_pt, *other_args_ci, new_index)
        sr_best = sr_best.droplevel([l_lvl_name, tau_lvl_name])
    except RuntimeError:
        sr_best = np.nan

    # Randomly sample parameters from the MCMC chain
    samp_choices = rgen.choice(psamples.shape[1]*psamples.shape[2],
                    size=n_samp, replace=True)
    s_choices = samp_choices // psamples.shape[1]
    w_choices = samp_choices % psamples.shape[1]

    # Student's t samples for the TCR, CAR, CD19, MHC, and K_D mean estimators
    tcr_num_std, car_num_std, car_ag_l_std, max_mhc_std, kd_std = molec_stds
    n_dofs_tcr, n_dofs_car, n_dofs_cd19, n_dofs_mhc, n_dofs_kd = molec_dofs
    tcr_num_samples = sample_log_student(
        tcr_number, tcr_num_std, rgen, n_samp, n_dofs_tcr, base=10.0
    )
    car_num_samples = sample_log_student(
        car_number, car_num_std, rgen, n_samp, n_dofs_car, base=10.0
    )
    car_ag_num_samples = sample_log_student(
        car_ag_number, car_ag_l_std, rgen, n_samp, n_dofs_cd19, base=10.0
    )
    mhc_num_samples = sample_log_student(
        l_conc_mm_params[0], max_mhc_std, rgen, n_samp, n_dofs_mhc, base=10.0
    )
    load_kd_samples = sample_log_student(
        l_conc_mm_params[1], kd_std, rgen, n_samp, n_dofs_kd, base=10.0
    )
    # Sample EC50s, then compute peptide taus from them
    # TODO: make sure that all peptide concentrations
    # receive the same peptide tau...
    noises = rgen.standard_normal(size=[n_samp, pep_log10ec50s.shape[0]])
    # Put noises in the DataFrame, make peptides=columns for broadcasting
    pep_ec50_samples = pd.DataFrame(noises, columns=pep_log10ec50s["estimator"].index,
                                    index=pd.RangeIndex(n_samp, name="Sample"))
    pep_ec50_samples = pep_log10ec50s["estimator"] + pep_ec50_samples*pep_log10ec50s["std"]
    pep_ec50_samples = pep_ec50_samples.T  # We want columns = samples ultimately

    # Compute the antagonism ratio panel for each sample parameter set
    pool = Pool(min(n_cpu, n_samp))
    all_procs = {}
    # Launch all processes
    for i in range(n_samp):
        nw, ns = w_choices[i], s_choices[i]
        pvec = psamples[:, nw, ns]
        # Place new TCR, CAR, CD19 numbers in other_args_ci
        # ri_tots: TCR, I_tot, CAR
        other_args_ci[1] = [tcr_num_samples[i], other_args[1][1], car_num_samples[i]]
        other_args_ci[3] = (other_args[3][0], car_ag_num_samples[i])  # car_ag_tau_l
        # Prepare the new index with the sampled mhc_num and load_kd
        df_idx[l_lvl_name] = michaelis_menten(
            df_idx[tcr_pulse_lvl], mhc_num_samples[i], load_kd_samples[i]
        )
        pep_ec50_sample = pep_ec50_samples[i]
        df_idx = replace_taus(df_idx, pep_ec50_sample, pep_concs,
                             tcr_pulse_lvl, tau_lvl_name, ec50_ref, tau_ref)

        # Using df_idx, create the full index with TCR antigen L and tau
        # as the last two levels
        new_index_ci = df_idx.set_index(full_idx_order).index
        # Use this index containing all sampled parameters to compute predictions
        all_procs[i] = pool.apply_async(
            model_panel, args=(pvec, grid_pt, *other_args_ci, new_index_ci)
        )

    # Collect the results, removing L and tau from the index
    # to have common peptide identifiers and pulse concentrations instead
    for i in range(n_samp):
        try:
            res = all_procs[i].get()
        except RuntimeError:
            df_model[i] = np.nan
        else:
            df_model[i] = res.droplevel([l_lvl_name, tau_lvl_name], axis=0)
    print("Number NaN samples for {}:".format(grid_pt),
                df_model.isna().iloc[0, :].sum())
    df_stats = analysis_fct(df_model, sr_best)
    print("Finished computing model predictions for {}".format(grid_pt))
    return df_stats, pep_ec50_samples
