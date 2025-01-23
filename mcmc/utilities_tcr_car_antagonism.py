"""
Functions to prepare data for MCMC fitting of antagonism ratios.
Code for generalization of the model to other cell lines than OT1-CAR T cells
is in separate modules.

@author: frbourassa
December 2022
"""
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json, h5py
import os

# Import local functions
from mcmc.utilities_tcr_tcr_antagonism import (
    sample_log_student,
    get_std_from_ci
)
from mcmc.plotting import (
    handles_properties_legend,
    prepare_hues,
    prepare_styles,
    prepare_subplots,
    data_model_handles_legend,
    change_log_ticks
)
from mcmc.mcmc_analysis import find_best_grid_point
from utils.preprocess import (
    groupby_others,
    geo_mean_levels,
    ln10,
    read_conc_uM,
    write_conc_uM,
    michaelis_menten,
    inverse_michaelis_menten,
    string_to_tuple
)


### UTILITY FUNCTIONS ###
def plot_cytokine_fit_data(df, do_save=False, do_show=False):
    def sortkey(idx):
        peps = ["None", "E1", "G4", "V4", "T4", "Q4", "Y3", "A2", "N4"]
        idx2 = pd.Index([peps.index(x) for x in idx], name=idx.name)
        return idx2

    df = df.sort_index(level="TCR_Antigen", key=sortkey)

    # Temporary plot for checkup
    palette = sns.color_palette("mako", )
    g = sns.relplot(data=df.reset_index(), x="TCR_Antigen", y=df.name,
            style="TCR_Antigen_Density", kind="line", marker="o", ms=8,
            hue="TCR_Antigen_Density", palette="Set2", col="Data-spleen", col_wrap=3,
            height=3.0)
    for ax in g.axes.flat:
        ax.set_yscale("log")
    if do_save:
        g.fig.tight_layout()
        g.fig.savefig("../figures/data_plots/datasets_fit_tcr_car_cd19_antagonism.pdf",
            transparent=True, bbox_inches="tight")
    if do_show:
        plt.show()
    plt.close()
    return None


### PREPARING MCMC RUNS ###
def prepare_car_antagonism_data(data, mm_params, pep_tau_map,
            cyto="IL-2", do_plot=False, tcr_conc=None, dropn4=False,
            tcr_itams=slice(None), car_itams=slice(None), data_fold="../data"):
    """ Take cytokine concentration dataframe and rearrange it to compute
    average antagonism ratio and errors on these ratios for each condition.
    We average over time and experiment after taking the ratio at each time
    point separately (gives more data points than averaging the time series
    and computing the ratio of averages).

    pep_tau_map (dict)
    tcr_conc: a list, or None. If a single string label is passed,
        it is converted to a length-1 list.
        If None is passed, all TCR_Antigen_Density labels are kept
    tcr_itams: either a slice, a list, or a single label (string),
        i.e. anything that can go into .loc
    car_itams: same
    data_fold (str): path to main data folder
    """
    def michaelis_menten_fitted(conc_axis):
        return michaelis_menten(conc_axis, *mm_params)

    # Create a data-spleen level
    df = data.stack().to_frame()
    df["Data-spleen"] = (df.index.get_level_values("Data") + "-"
                        + df.index.get_level_values("Spleen"))
    df = df.set_index("Data-spleen", append=True)
    df = df.droplevel(["Data", "Spleen"])

    # Select cytokine and conditions of choice for fitting
    df = df.reorder_levels(["Cytokine", "Tumor", "TCR_ITAMs", "CAR_ITAMs",
        "CAR_Antigen", "Data-spleen", "TCR_Antigen_Density", "TCR_Antigen", "Time"])
    df = df.sort_index()
    df = df.loc[(cyto, "E2APBX", tcr_itams, car_itams, "CD19")]
    # Levels are not dropped if tcr_itams or car_itams are lists
    if "Tumor" in df.index.names:
        df = df.droplevel(["Cytokine", "Tumor", "CAR_Antigen"])
    df.name = cyto

    if do_plot:  # Show geometric time average of time series in this data
        plot_cytokine_fit_data(geo_mean_levels(df, ["Time"], axis=0), do_save=True)

    # Compute time series of ratios. "No TCR antigen" is always
    # encoded as TCR_Antigen=None, with a copy for each TCR_Antigen_Density
    df_fit = df / df.xs("None", level="TCR_Antigen")
    df_fit = df_fit[0]
    df_fit.name = "Ratio"

    # Clip excessively large ratio values
    upper_lim = 2e1 if cyto == "IFNg" else 2e2
    df_fit = df_fit[df_fit < upper_lim]

    # Do not attempt to fit N4+CD19 mix, the non-monotonicity of enhancement
    # is not what we want to capture. Many causes are possible, including:
    #   we have a single dataset for 1 nM;
    #   killing is so strong in response to the mixture that cells don't
    #       have time to produce IL-2
    #   receptor downregulation (Coombs et al. 2002)
    # Drop N4 when fitting, but not when predicting (try to predict N4 too)
    if dropn4:
        df_fit = df_fit.drop("N4", level="TCR_Antigen")

    # Selecting only desired concentrations of TCR ligand
    if tcr_conc is not None:
        if isinstance(tcr_conc, str):
            tcr_conc = [tcr_conc]
        df_fit = df_fit.loc[df_fit.index.isin(tcr_conc, level="TCR_Antigen_Density")]

    # Transform index entries into L, tau values of the TCR antigen
    df_fit = df_fit.rename(pep_tau_map, axis=0, level="TCR_Antigen")
    df_fit = df_fit.rename(read_conc_uM, axis=0, level="TCR_Antigen_Density")
    df_fit = df_fit.rename(michaelis_menten_fitted, axis=0, level="TCR_Antigen_Density")
    # Compute log2-scale confidence intervals and geometric average
    # across Time and experimental repeat
    df_groups = groupby_others(np.log2(df_fit), ["Data-spleen", "Time"], axis=0)
    # Student's t critical value for 95 % CI
    n_dofs = df_groups.count()
    t_crits = sp.stats.t.ppf(0.975, n_dofs-1)
    df_ci = t_crits * df_groups.std(ddof=1) / np.sqrt(n_dofs)
    df_ci.dropna()
    # Clip ci to some small nonzero value, for the no antagonist
    # condition which has a ratio always identically 1.
    df_ci = df_ci.clip(lower=0.01)

    # Geometric average across experiments
    df_fit = geo_mean_levels(df_fit, ["Data-spleen", "Time"], axis=0)
    df_fit = df_fit.dropna()
    return df_fit, df_ci


def load_tcr_car_molec_numbers(molec_counts_fi, mtc, **kwargs):
    """
    Standard loading of surface molecule parameters for TCR-CAR antagonism
    for 1- or 3-ITAM CARs.

    Args:
        molec_counts_fi (str): path to file containing surface molecule summary stats
        mtc (str): metric/statistic to use, such as "Geometric mean"
    Keyword args:
        tcell_type (str): "OT1_CAR" by default,
            "OT1_Naive" or "OT1_Blast" are also available.
        tumor_type (str): "E2aPBX_WT" by default,
            also "B16", "Nalm6", "PC9", "BEAS2B" are available.
        tumor_antigen (str): "CD19" by default, varies for different tumors
        data_fold (str): path to main data folder. Typically ../data/
            because MCMC scripts in subfolder.

    Returns:
        tcr_number (float): number of TCRs per CAR T cell
        car_number (float): number of CARs per CAR T cell
        cd19_l (float): number of CD19 molecules per E2aPBX cell
        l_conc_mm_params (list of 2 floats): [max_mhc, pulse_kd]
            max_mhc (float): total number of MHC per E2aPBX
            pulse_kd (float): antigen pulse dissociation constant
        pep_tau_map_ot1 (dict): binding time estimated from KPR scaling law
            for each OVA variant peptide.
    """
    tcell_type = kwargs.get("tcell_type", "OT1_CAR")
    tumor_type = kwargs.get("tumor_type", "E2aPBX_WT")
    tumor_antigen = kwargs.get("tumor_antigen", "CD19")
    data_fold = kwargs.get("data_fold", "../data/")

    molec_stats = pd.read_hdf(molec_counts_fi, key="surface_numbers_stats")
    # Number of TCR per T cell
    tcr_number =  molec_stats.loc[(tcell_type, "TCR"), mtc]

    # Number of CARs per T cell (assume same for 1- and 3-ITAM, data for 3)
    try:
        car_number = molec_stats.loc[(tcell_type, "CAR"), mtc]
    except KeyError:
        car_number = molec_stats.loc[("OT1_CAR", "CAR"), mtc]

    # Number of MHCs per APC (B6 splenocyte)
    max_mhc = molec_stats.loc[(tumor_type, "MHC"), mtc]
    cd19_l = molec_stats.loc[(tumor_type, tumor_antigen), mtc]

    # Pulse concentration to ligand numbers conversion:
    # based on RMA-S 2019-2020 data
    mhc_pulse_kd = pd.read_hdf(molec_counts_fi, key="mhc_pulse_kd")
    # This pd.Series also contains covariance of the K_D parameter fit
    pulse_kd = mhc_pulse_kd[mtc]

    # Use average parameters.
    l_conc_mm_params_dict = {"amplitude":max_mhc, "ec50":pulse_kd}
    # amplitude, EC50
    l_conc_mm_params = [max_mhc, pulse_kd]

    # Mapping tau and EC50s
    with open(os.path.join(data_fold, "pep_tau_map_ot1.json"), "r") as handle:
        pep_tau_map_ot1 = json.load(handle)

    return tcr_number, car_number, cd19_l, l_conc_mm_params, pep_tau_map_ot1


def load_tcr_car_molec_numbers_ci(molec_counts_fi, mtc, **kwargs):
    """
    Standard loading of surface molecule parameters for TCR-CAR antagonism
    for 1- or 3-ITAM CARs. Also load standard deviation of the TCR, CAR, CAR
    and TCR antigens (MHC), and of the loading K_D parameter.
    Also load the number of d.o.f. on which the mean estimators are based.

    Args:
        molec_counts_fi (str): path to file containing surface molecule summary stats
        mtc (str): metric/statistic to use, such as "Geometric mean"
    Keyword args:
        tcell_type (str): "OT1_CAR" by default,
            "OT1_Naive" or "OT1_Blast" are also available.
        tumor_type (str): "E2aPBX_WT" by default,
            also "B16", "Nalm6", "PC9", "BEAS2B" are available.
        tumor_antigen (str): "CD19" by default, varies for different tumors
        data_fold (str): path to main data folder. Typically ../data/
            because MCMC scripts in subfolder.
        mhc_name (str): default "MHC", can be "HLA-A2" for instance.

    Returns:
        tcr_number (float): number of TCRs per CAR T cell
        car_number (float): number of CARs per CAR T cell
        cd19_l (float): number of CD19 molecules per E2aPBX cell
        l_conc_mm_params (list of 2 floats): [max_mhc, pulse_kd]
            max_mhc (float): total number of MHC per E2aPBX
            pulse_kd (float): antigen pulse dissociation constant
        pep_tau_map_ot1 (dict): binding time estimated from KPR scaling law
            for each OVA variant peptide.
        all_stds (list of 5 floats): std on TCR, CAR, CD19, MHC, K_D
        all_dofs (list of 5 floats): number dofs for TCR, CAR, CD19, MHC, KD
    """
    tcell_type = kwargs.get("tcell_type", "OT1_CAR")
    tumor_type = kwargs.get("tumor_type", "E2aPBX_WT")
    tumor_antigen = kwargs.get("tumor_antigen", "CD19")
    data_fold = kwargs.get("data_fold", "../data/")
    mhc_name = kwargs.get("mhc_name", "MHC")

    molec_stats = pd.read_hdf(molec_counts_fi, key="surface_numbers_stats")
    # Number of TCR per T cell
    tcr_number =  molec_stats.loc[(tcell_type, "TCR"), mtc]

    # Recover standard deviation of the mean estimator from the CI
    tcr_std, n_dofs_tcr = get_std_from_ci(molec_stats, (tcell_type, "TCR"), mtc)

    # Number of CARs per T cell (assume same for 1- and 3-ITAM, data for 3)
    try:
        car_number = molec_stats.loc[(tcell_type, "CAR"), mtc]
        car_std, n_dofs_car = get_std_from_ci(molec_stats, (tcell_type, "CAR"), mtc)
    except KeyError:
        car_number = molec_stats.loc[("OT1_CAR", "CAR"), mtc]
        car_std, n_dofs_car = get_std_from_ci(molec_stats, ("OT1_CAR", "CAR"), mtc)

    # Number of MHCs per APC (B6 splenocyte)
    max_mhc = molec_stats.loc[(tumor_type, mhc_name), mtc]
    mhc_std, n_dofs_mhc = get_std_from_ci(molec_stats, (tumor_type, mhc_name), mtc)
    cd19_l = molec_stats.loc[(tumor_type, tumor_antigen), mtc]
    cd19_std, n_dofs_cd19 = get_std_from_ci(molec_stats, (tumor_type, tumor_antigen), mtc)

    # Pulse concentration to ligand numbers conversion:
    # based on RMA-S 2019-2020 data
    mhc_pulse_kd = pd.read_hdf(molec_counts_fi, key="mhc_pulse_kd")
    # This pd.Series also contains covariance of the K_D parameter fit
    pulse_kd = mhc_pulse_kd[mtc]
    kd_std, n_dofs_kd = get_std_from_ci(mhc_pulse_kd, None, mtc)

    # Use average parameters.
    l_conc_mm_params_dict = {"amplitude":max_mhc, "ec50":pulse_kd}
    # amplitude, EC50
    l_conc_mm_params = [max_mhc, pulse_kd]

    # Mapping tau and EC50s
    with open(os.path.join(data_fold, "pep_tau_map_ot1.json"), "r") as handle:
        pep_tau_map_ot1 = json.load(handle)

    all_stds = [tcr_std, car_std, cd19_std, mhc_std, kd_std]
    all_dofs = [n_dofs_tcr, n_dofs_car, n_dofs_cd19, n_dofs_mhc, n_dofs_kd]
    return (tcr_number, car_number, cd19_l, l_conc_mm_params, pep_tau_map_ot1,
            all_stds, all_dofs)


def load_tcr_tcr_akpr_fits(
        res_file, analysis_file, klim=2, wanted_kmf=None
    ):
    """ Load all model parameters fitted on TCR-TCR antagonism data """
    # We need the following.
    # TCR parameters: phi, kappa, cmthresh, S0p, kp, psi0, gamma_tt
    # Some were fitted, some were "other_rates".
    # We also need TCR's N, m, f

    # Get best k, m, f
    with open(analysis_file, "r") as h:
        lysis = json.load(h)
    # Drop all points with k > klim, as large ks can be overfitted
    for p in list(lysis.keys()):
        kmf_tuple = string_to_tuple(p)
        if kmf_tuple[0] > klim:
            lysis.pop(p)
    if wanted_kmf is None:
        best_grid, best_p, _ = find_best_grid_point(lysis, strat="best")
        best_kmf = list(string_to_tuple(best_grid))
    else:
        best_p = lysis[str(tuple(wanted_kmf))]["param_estimates"]["MAP best"]
        best_p = np.asarray(best_p)
        best_kmf = list(wanted_kmf)
    # Get phi, cm_threshold, I_threshold, psi0 from best fit
    best_phi, best_cm, best_i, best_psi0 = np.exp(best_p * ln10)

    # Get other rates
    with h5py.File(res_file, "r") as fi:
        other_rates = (fi.get("data/rates_others")[:]).tolist()
        tcr_n = fi.get("data/N")[()]
        tcr_itot = fi.get("data/total_RI")[1]  # keep I, R may differ on CAR T

    # Arrange the loaded parameters as will be useful
    tcr_params = ([best_phi] + other_rates[:1]                     # phi, kappa
                    + [best_cm, best_i, best_kmf[0]]  # cmthresh, S0p, k_S
                    + [best_psi0]                # psi_0
                    + [1.0]                          # gamma_tt
                )
    tcr_nmf = [tcr_n] + best_kmf[1:]

    return tcr_params, tcr_nmf, tcr_itot


### ANALYZING MCMC RUNS ###

# Model CI generation, special case for TCR/CAR to propagate uncertainties
# from antigen numbers, receptor numbers, prediction factors for ITAM numbers.
def confidence_model_antagonism_car(
        model_panel, psamples, pbest, grid_pt, df_ratio, df_err, #psamples_tcr,
        other_args=(), n_taus=101, n_samp=1000, seed=None, antagonist_lvl="Antagonist",
        l_conc_mm_params=[1.0, 0.1], molec_stds=[0.1,]*5, molec_dofs=[1000,]*5,
        factors_stds=None
    ):
    """ Receive parameter samples and a point on the grid search,
    as well as a function evaluating antagonism panel for some model,
    and compute the confidence interval of model predictions compared to data.

    Rigorously, the mean TCR, MHC numbers and K_D must be sampled from
    a Student's t distribution, with scale given by the standard deviations
    provided in tcr_num_std and mm_params_std and numbers of degrees of freedom
    in dofs_rcpt_mhc_car_kd.

    If you want to use this function with other CAR T cell or tumor cell lines
    change the corresponding model parameters in the other_args (cost args)
    passed as an argument here. The elements in other_args are passed as
    individual arguments to model_panel.

    Args:
        model_panel (callable): function evaluating model antagonism ratio
            for given (pvec, grid_pt, *other_args, df_ratio.index).
        psamples (np.ndarray): sampled parameters array,
            shaped [param, nwalkers, nsamples]
        pbest (np.ndarray): best parameter sample
        grid_pt (tuple of ints): integer parameters after grid search.
        df_ratio (pd.DataFrame): antagonism ratio points
        df_err (pd.DataFrame): antagonism ratio error bars.
        #psamples_tcr (np.ndarray): sampled TCR/TCR parameter array,
        #    shaped [param_tcr, nwalkers_tcr, nsamples_tcr]
        other_args (tuple): other arguments passed to model_panel.
        seed (np.random.SeedSequence or int): random number generator seed
        n_samp (int): number of samples to draw for confidence intervals
        antagonist_lvl (str): which level contains the antagonist tau,
            will be dropped and replaced by a range of taus.
        l_conc_mm_params (list of 2 floats): mean MHC number per APC
            and peptide loading EC50 mean estimator.
        tcr_num_std (float): standard deviation of the mean estimator
            of the TCR number per T cell
        molec_stds (list of 5 floats): standard deviation of the mean estimators
            for the TCR, CAR, CD19, MHC, and loading K_D
        molec_dofs (list of 5 ints): the number of degrees of freedom
            (cell or experimental repeats) from which the mean estimators
            for the TCR, CAR, CD19, MHC, and loading K_D are calculated.
        factors_stds (list of [list, float]): standard deviation of the
            estimated correction factors for [tcr_ampli, car_ampli]
            and TCR threshold.

    Returns:
        df_stats (pd.DataFrame): statistics on model antagonism predictions,
            including 90 % CI, median, best fit, as a function of tau.
    """
    rgen = np.random.default_rng(seed)

    # Recover the pulse concentrations from the ligand numbers computed
    # with the best Michaelis-Menten parameters. These pulse concentrations
    # will correspond to slightly different ligand numbers for each
    # l_conc_mm_params values sampled when bootstrapping the CI.
    best_antag_nums = df_ratio.index.get_level_values("TCR_Antigen_Density").unique()
    antag_pulses = inverse_michaelis_menten(best_antag_nums, *l_conc_mm_params)
    antag_l_conc_map = dict(zip(best_antag_nums, antag_pulses))
    inv_l_antag_map = dict(zip(antag_l_conc_map.values(), antag_l_conc_map.keys()))

    # Create continuous antagonist tau range, add pulse concentration levels
    # to be converted to different ligand numbers for every sample.
    new_tau_range = np.linspace(0.001, df_ratio.index.get_level_values(antagonist_lvl).max(), n_taus)
    old_index = df_ratio.index.droplevel(antagonist_lvl).unique()
    if isinstance(old_index, pd.MultiIndex):
        product_tuples = [tuple(a)+(b,) for b in new_tau_range for a in old_index]
    elif isinstance(old_index, pd.Index):
        product_tuples = [(a,)+(b,) for b in new_tau_range for a in old_index]
    new_names = list(old_index.names) + [antagonist_lvl]
    # Turn into a DataFrame without an index to better add and map index levels
    # to deal with pulse concentration to ligand number conversions
    new_index = pd.MultiIndex.from_tuples(product_tuples, names=new_names)
    df_idx = new_index.to_frame()
    df_idx["TCR_Antigen_Pulse"] = df_idx["TCR_Antigen_Density"].map(antag_l_conc_map)

    # Replace the ri_tot default by a local list to be modified each iteration
    # For TCR/CAR, other_args = list(cost_args_loaded) + [[tcr_ampli, car_ampli], tcr_thresh_fact]
    # and cost_args_loaded = tcr_car_params, tcr_car_ritots, tcr_car_nmf, cd19_tau_l
    # and tcr_car_params = phi_tcr, kappa_tcr, cmthresh_tcr, S0p, kp, psi0_tcr, gamma_tt
    # then phi_car, kappa_car, gamma_cc, psi0_car
    # and tcr_car_ritots = [r_tcr, r_car, I_tot]
    # Replace elements in other_args by local copies
    other_args_ci = list(other_args)  # Local list, elements in it are refs
    other_args_ci[1] = list(other_args_ci[1])  # ri_tots list: r_tcr, i_tot, r_car
    other_args_ci[3] = list(other_args_ci[3])  # cd19_tau_l
    if factors_stds is not None:  # prediction factors
        other_args_ci[-2] = list(other_args_ci[-2])
        other_args_ci[-1] = float(other_args_ci[-1])
    # From the loaded cost_args, extract TCR, CAR, CD19 numbers
    tcr_num_estim = other_args[1][0]
    car_num_estim = other_args[1][2]
    cd19_l = other_args[3][1]

    new_cols = pd.RangeIndex(n_samp)
    # Replace TCR_Antigen_Density with TCR_Antigen_Pulse
    df_model_idx = df_idx.set_index(
            ["TCR_Antigen_Pulse"] + list(new_index.names[1:])).index
    df_model = pd.DataFrame(np.zeros([len(new_index), n_samp]),
                    index=df_model_idx, columns=new_cols)

    # Randomly sample parameters from the MCMC chain
    samp_choices = rgen.choice(psamples.shape[1]*psamples.shape[2], size=n_samp, replace=True)
    s_choices = samp_choices // psamples.shape[1]
    w_choices = samp_choices % psamples.shape[1]

    # Student's t samples for the TCR, CAR, CD19, MHC, and K_D mean estimators
    tcr_num_std, car_num_std, cd19_l_std, max_mhc_std, kd_std = molec_stds
    n_dofs_tcr, n_dofs_car, n_dofs_cd19, n_dofs_mhc, n_dofs_kd = molec_dofs
    tcr_num_samples = sample_log_student(
        tcr_num_estim, tcr_num_std, rgen, n_samp, n_dofs_tcr, base=10.0
    )
    car_num_samples = sample_log_student(
        car_num_estim, car_num_std, rgen, n_samp, n_dofs_car, base=10.0
    )
    cd19_l_samples = sample_log_student(
        cd19_l, cd19_l_std, rgen, n_samp, n_dofs_cd19, base=10.0
    )
    mhc_num_samples = sample_log_student(
        l_conc_mm_params[0], max_mhc_std, rgen, n_samp, n_dofs_mhc, base=10.0
    )
    load_kd_samples = sample_log_student(
        l_conc_mm_params[1], kd_std, rgen, n_samp, n_dofs_kd, base=10.0
    )
    if factors_stds is not None:
        factors_a = other_args[-2]
        factor_th = other_args[-1]
        tcr_ampli_samples = factors_a[0] + factors_stds[0][0]*rgen.standard_normal(size=n_samp)
        car_ampli_samples = factors_a[1] + factors_stds[0][1]*rgen.standard_normal(size=n_samp)
        tcr_thresh_samples = factor_th + factors_stds[1]*rgen.standard_normal(size=n_samp)
    else:
        factors, tcr_ampli_samples = None, None
        car_ampli_samples, tcr_thresh_samples = None, None

    # Compute the antagonism ratio panel for each sample parameter set
    # The last levels should be l_ag, l_antag, antag_tau,
    # but previous levels can be the pulse concentrations

    # last two levels should be L, tau: fine to prepend TCR_Antigen_Pulse
    full_idx_order = ["TCR_Antigen_Pulse"] + list(new_index.names)
    for i in range(n_samp):
        nw, ns = w_choices[i], s_choices[i]
        pvec = psamples[:, nw, ns]
        # Place new TCR, CAR, CD19 numbers in other_args_ci
        # ri_tots: TCR, I_tot, CAR
        other_args_ci[1] = [tcr_num_samples[i], other_args[1][1], car_num_samples[i]]
        other_args_ci[3] = (other_args[3][0], cd19_l_samples[i])
        if factors_stds is not None:
            other_args_ci[-2] = [tcr_ampli_samples[i], car_ampli_samples[i]]
            other_args_ci[-1] = tcr_thresh_samples[i]
        # Prepare the new index with the sampled mhc_num and load_kd
        df_idx["TCR_Antigen_Density"] = michaelis_menten(
            df_idx["TCR_Antigen_Pulse"], mhc_num_samples[i], load_kd_samples[i]
        )

        new_index_ci = df_idx.set_index(full_idx_order).index
        try:
            df_model_i = model_panel(pvec, grid_pt, *other_args_ci, new_index_ci)
        except RuntimeError:
            df_model_i = np.nan
        else:
            df_model_i = df_model_i.droplevel(
                ["TCR_Antigen_Density"], axis=0
            )
        df_model[i] = df_model_i
    print("Number NaN samples for {}:".format(grid_pt),
                df_model.isna().iloc[0, :].sum())

    # Compute model output for best parameter vector.
    # Need an index with pulse concentrations and the default L numbers
    df_idx["TCR_Antigen_Density"] = df_idx["TCR_Antigen_Pulse"].map(inv_l_antag_map)
    new_index_ci = df_idx.set_index(full_idx_order).index
    try:
        sr_best = model_panel(pbest, grid_pt, *other_args, new_index_ci)
    except RuntimeError:
        sr_best = np.nan
    else:
        sr_best = sr_best.droplevel(["TCR_Antigen_Density"], axis=0)

    # Compute statistics of the ratios on a log scale, then back to linear scale
    stats = ["percentile_2.5", "median", "mean", "geo_mean", "best", "percentile_97.5"]
    df_stats = pd.DataFrame(np.zeros([df_model.shape[0], len(stats)]),
                index=df_model_idx, columns=stats)
    df_stats["mean"] = np.log(df_model.mean(axis=1))
    df_model = np.log(df_model)
    df_stats["percentile_2.5"] = df_model.quantile(q=0.025, axis=1)
    df_stats["median"] = df_model.quantile(q=0.5, axis=1)
    df_stats["geo_mean"] = df_model.mean(axis=1)
    df_stats["best"] = np.log(sr_best)
    df_stats["percentile_97.5"] = df_model.quantile(q=0.975, axis=1)
    df_stats = np.exp(df_stats)

    # Put back default ligand numbers corresponding to Pulse levels for
    # output compatibility with other panel functions
    df_stats.index = (df_stats.index
        .rename(names="TCR_Antigen_Density", level="TCR_Antigen_Pulse")
    )
    df_stats = (df_stats
        .rename(inv_l_antag_map, axis=0, level="TCR_Antigen_Density")
    )

    print("Finished computing model predictions for {}".format(grid_pt))
    return df_stats


# check_fit_model_car_antagonism: same as check_fit_model_antagonism
# written for TCR-TCR, just pass antagonist_lvl = "TCR_Antigen"
from mcmc.utilities_tcr_tcr_antagonism import check_fit_model_antagonism


def plot_fit_car_antagonism(
        df_ratio, df_model, l_conc_mm_params, df_err, cost=None, model_ci=None,
    ):
     # Aesthetic parameters
    with open(os.path.join("..", "results", "for_plots", 
            "perturbations_palette.json"), "r") as f:
        pert_palette = json.load(f)
    pert_palette["None"] = [0., 0., 0., 1.]  # Black
    df_model_data = pd.concat({"Data":df_ratio, "Model": df_model}, names=["Source"])
    df_model_data.name = "Antagonism ratio"

    # Rename concentrations for nicer plotting
    def reverse_mm_fitted(x):
        return inverse_michaelis_menten(x, *l_conc_mm_params)
    def renamer(d):
        return (d.rename(reverse_mm_fitted, axis=0, level="TCR_Antigen_Density")
                 .rename(write_conc_uM, axis=0, level="TCR_Antigen_Density")
                )
    df_model_data = renamer(df_model_data).sort_index()
    df_err = renamer(df_err).sort_index()
    if model_ci is not None:
        model_ci = renamer(model_ci).sort_index()

    # Prepare palette, columns, etc.
    available_antagconc = list(df_model_data.index
                        .get_level_values("TCR_Antigen_Density").unique())
    available_antagconc = sorted(available_antagconc, key=read_conc_uM, reverse=True)
    palette = sns.dark_palette(pert_palette["AgDens"],
                               n_colors=len(available_antagconc))
    palette[0] = (0.0,)*3 + (1.0,)
    palette = {a:c for a, c in zip(available_antagconc, palette)}

    marker_bank = ["o", "s", "^", "X", "P", "*"]
    if len(available_antagconc) > len(marker_bank):
        raise NotImplementedError("Don't know which other markers to use")
    markers = {available_antagconc[i]:marker_bank[i] for i in range(len(available_antagconc))}

    styles_bank = ["-", "--", ":", "-."]
    if len(available_antagconc) > len(styles_bank):
        raise NotImplementedError("Don't know which other line styles to use")
    linestyles = {available_antagconc[i]:styles_bank[i] for i in range(len(available_antagconc))}

    # Make a nice plot. Don't use seaborn because it adds lines between data points
    fig, ax = plt.subplots()
    fig.set_size_inches(3.75, 3.0)

    index_antag = list(df_model_data.index.names).index("TCR_Antigen_Density")
    conc_key = [slice(None)]*len(df_model_data.index.names)
    # Then change default_slice[index_antag] at every iteration
    for i, antag_conc in enumerate(available_antagconc):
        conc_key[index_antag] = antag_conc
        conc_key[0] = "Data"
        data_pts = np.log2(df_model_data.loc[tuple(conc_key)]).sort_index()
        conc_key[0] = "Model"
        model_pts = np.log2(df_model_data.loc[tuple(conc_key)]).sort_index()
        err_pts = df_err.loc[tuple(conc_key[1:])].sort_index()
        if model_ci is not None:
            percentiles = sorted(list(model_ci.columns), 
                                key=lambda x: float(x.split("_")[-1]))
            model_ci_low = np.log2(model_ci.loc[tuple(conc_key[1:]), percentiles[0]]).sort_index()
            model_ci_hi = np.log2(model_ci.loc[tuple(conc_key[1:]), percentiles[1]]).sort_index()

        hue = palette.get(antag_conc)
        mark = markers.get(antag_conc)
        style = linestyles.get(antag_conc)
        if model_ci is not None:
            # First shade confidence interval
            ax.fill_between(model_ci_low.index.get_level_values("TCR_Antigen").values,
                        model_ci_low.values, model_ci_hi.values, color=hue, alpha=0.25)
        errbar = ax.errorbar(data_pts.index.get_level_values("TCR_Antigen").values, data_pts.values,
                   yerr=err_pts.values, marker=mark, ls="none", color=hue, mfc=hue, mec=hue)
        li, = ax.plot(model_pts.index.get_level_values("TCR_Antigen").values, model_pts.values,
                   color=hue, ls=style, label=antag_conc)

    # Label this plot
    ax.set_xlabel(r"Antagonist model $\tau$ (s)")
    if i == 0:
        ax.set_ylabel(r"FC$_{TCR \rightarrow CAR}$")
    # Change y tick labels to 2^x
    ax = change_log_ticks(ax, base=2, which="y")
    ax.axhline(0.0, ls="--", color="k", lw=1.0)
    ax.set_title(antag_conc + " TCR Antigen", y=0.95, va="top")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    # Annotate the last plot with the cost function
    if cost is not None:
        ax.annotate("Log posterior:{:.3f}".format(cost), xy=(0.95, 0.05),
                        xycoords="axes fraction", ha="right", va="bottom")

    # Add a figure legend, manual to indicate markers and hues, model and data
    fig.tight_layout()
    leg_handles, leg_handler = data_model_handles_legend(palette, markers, linestyles,
                        "TCR Antigen\nDensity", model_style="-",
                        model_lw=li.get_linewidth(),
                        data_size=errbar[0].get_markersize(), data_marker="o")
    fig.legend(handles=leg_handles, handler_map=leg_handler,
               loc="upper left", bbox_to_anchor=(0.975, 0.95), frameon=False)

    return fig, ax


### PREDICT FROM MCMC RUNS ###
# confidence_model_car_antagonism is the same as confidence_model_antagonism
# from the tcr_tcr_utilities, with antagonist_lvl="TCR_Antigen"

def perturb_concatenator(x):
    lbl = []
    if x["TCR_Antigen_Density"] == "1nM":
        lbl.append("AgDens")
    if x["CAR_ITAMs"] == "1":
        lbl.append("CARNum")
    if x["TCR_ITAMs"] == "4":
        lbl.append("TCRNum")
    lbl = "_".join(lbl)
    if lbl == "":
        lbl = "None"
    return lbl

def perturb_decoder(x):
    if x == "None":
        return "Default"
    xsplit = x.split("_")
    lbl = []
    for u in xsplit:
        if u == "AgDens":
            lbl.append("1 nM TCR Ag")
        elif u == "CARNum":
            lbl.append("1 CAR ITAM")
        elif u == "TCRNum":
            lbl.append("4 TCR ITAMs")

    return ",\n".join(lbl)

def plot_predict_car_antagonism(df_data, df_model, l_conc_mm_params, df_err):
    # Aesthetic parameters
    with open(os.path.join("..", "results", "for_plots", 
            "perturbations_palette.json"), "r") as f:
        perturb_palette = json.load(f)
    perturb_palette["None"] = [0., 0., 0., 1.]  # Black
    # Rename concentrations for nicer plotting
    def reverse_mm_fitted(x):
        return inverse_michaelis_menten(x, *l_conc_mm_params)
    dfdict = {"model":df_model, "data":df_data, "err":df_err}
    for k in dfdict:
        dfdict[k] = (dfdict[k]
                .rename(reverse_mm_fitted, axis=0, level="TCR_Antigen_Density")
                .rename(write_conc_uM, axis=0, level="TCR_Antigen_Density"))
        # Also a "Condition" level to each Df
        new_idx_lvl = (dfdict[k].index.to_frame()
                    .apply(perturb_concatenator, axis=1))
        new_idx = pd.MultiIndex.from_tuples([(new_idx_lvl.iat[i],
                *dfdict[k].index[i]) for i in range(len(new_idx_lvl))],
                names=["Condition", *dfdict[k].index.names])
        dfdict[k] = dfdict[k].set_axis(new_idx)
    dfdict["data"].name = "Antagonism ratio"

    # No chance using seaborn functions: need to combine scatter and line
    # plots on same relplot. Also need to fill_between pre-computed error
    # statistics for the model: not included in seaborn.
    # So I would spend just as long hacking the data into the right format
    # to exploit seaborn as I will use here to rewrite with just matplotlib.
    hue_lvl = "Condition"
    hue_vals, palette = prepare_hues(dfdict['model'], hue_lvl,
                            sortkws={"key":len, "reverse":False})
    palette = {k:perturb_palette[k] for k in hue_vals}
    # styles = Subset
    sty_lvl = "Subset"
    sty_vals, styles = prepare_styles(dfdict['model'], sty_lvl)
    # rows = CAR_ITAMs, if available
    row_lvl = "CAR_ITAMs"
    col_lvl = "TCR_ITAMs"
    x_lvl = "TCR_Antigen"
    row_vals, col_vals, fig, axes = prepare_subplots(dfdict['model'],
                row_lvl=row_lvl, col_lvl=col_lvl,
                sortkws_col={"key":int, "reverse":True},
                sortkws_row={"key":int, "reverse":True},
                sharey="row")
    legwidth = 1.5
    figwidth = len(col_vals)*3. + legwidth
    fig.set_size_inches(figwidth, max(1, len(row_vals))*3.)

    for i in range(max(1, len(row_vals))):
        if len(row_vals) > 0:
            data_row = dfdict['data'].xs(row_vals[i], level=row_lvl, drop_level=False)
            model_row = dfdict['model'].xs(row_vals[i], level=row_lvl, drop_level=False)
            err_row = dfdict['err'].xs(row_vals[i], level=row_lvl, drop_level=False)
        else:
            data_row, model_row, err_row = dfdict['data'], dfdict['model'], dfdict['err']
        for j in range(max(1, len(col_vals))):
            if len(col_vals) > 0:
                dat_loc = data_row.xs(col_vals[j], level=col_lvl, drop_level=False)
                mod_loc = model_row.xs(col_vals[j], level=col_lvl, drop_level=False)
                err_loc = err_row.xs(col_vals[j], level=col_lvl, drop_level=False)
            else:
                dat_loc, mod_loc, err_loc = data_row, model_row, err_row
            ax = axes[i, j]
            ax.axhline(1.0, ls=":", color="k")
            local_hue_vals = [h for h in hue_vals
                    if h in dat_loc.index.get_level_values(hue_lvl).unique()]
            for h in local_hue_vals:
                # Plot data +- error transformed to linear scale
                err_loc2 = err_loc.xs(h, level=hue_lvl)
                dat_loc2 = dat_loc.xs(h, level=hue_lvl)
                # Make sure data and error have same index order for plotting
                err_loc2 = err_loc2.reindex(index=dat_loc2.index, copy=False)
                dat_log = np.log2(dat_loc2)
                # Compute linear scale error bars (asymmetric)
                # from symmetric log-scale error bars
                yup = 2**(err_loc2 + dat_log) - dat_loc2
                ylo = dat_loc2 - 2**(-err_loc2 + dat_log)
                yerr = np.vstack([ylo.values, yup.values])
                xvals = dat_loc2.index.get_level_values(x_lvl).values
                ax.errorbar(xvals, dat_loc2, xerr=None, yerr=yerr, marker="o",
                    ecolor=palette[h], mfc=palette[h], mec=palette[h],
                    ms=6, ls="none",
                )

                # Fill between confidence interval of model, highlight median
                # Use different styles for fitted or predicted subsets
                mod_loc2 = mod_loc.xs(h, level=hue_lvl)
                # There should be only one style value left
                assert len(mod_loc2.index.get_level_values(sty_lvl).unique()) == 1
                sty_val = mod_loc2.index.get_level_values(sty_lvl)[0]
                xvals = mod_loc2.index.get_level_values(x_lvl).values
                ax.fill_between(xvals, mod_loc2["percentile_2.5"], mod_loc2["percentile_97.5"],
                    color=palette[h], alpha=0.3)
                lbl = perturb_decoder(h)
                ax.plot(xvals, mod_loc2["best"], color=palette[h],
                        lw=2.5, ls=styles[sty_val], label=lbl)
            ax.set_yscale("log", base=2)
            ax.set_title(col_lvl + " = " + col_vals[j] + "\n"
                        + row_lvl + " = " + row_vals[i], size=10)
    for ax in axes[-1]:
        ax.set_xlabel(r"TCR Antigen $\tau$ (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$FC_{\mathrm{TCR \rightarrow CAR}}$")
    fig.tight_layout()
    fig.subplots_adjust(right=1.0 - legwidth/figwidth)
    # Add custom legend
    hues = (hue_lvl, palette)
    styles = {k:(a, None) for k,a in styles.items()}
    styles["Data"] = ("none", "o")
    styles = (sty_lvl, styles)
    legend_handles, legend_handler_map = handles_properties_legend(hues, styles, None)
    fig.legend(handles=legend_handles, handler_map=legend_handler_map,
            bbox_to_anchor=(1.02 - legwidth/figwidth, 0.5), loc="center left")
    return fig, axes
