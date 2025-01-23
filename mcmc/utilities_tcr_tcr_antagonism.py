"""
Functions to prepare data for MCMC fitting of antagonism ratios.

@author: frbourassa
November 2022
"""
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings  # pandas has annoying futurewarnings about stack
# even though we call it in a way compatible with the future 3.0 version

from mcmc.plotting import data_model_handles_legend, change_log_ticks
from utils.preprocess import (
    geo_mean_apply,
    read_conc_uM,
    write_conc_uM,
    michaelis_menten,
    inverse_michaelis_menten
)

### PREPARING MCMC RUNS ###

def prepare_data(data, mm_params, pep_tau_map):
    """ Take cytokine concentration dataframe and rearrange it to compute
    average antagonism ratio and errors on these ratios for each condition.
    We use the geometric average over time, then we compute the ratio,
    and lastly, average the ratio across experimental replicates.
    """
    def michaelis_menten_fitted(conc_axis):
        return michaelis_menten(conc_axis, *mm_params)

    # Remove missing time points and format as Series to use groupby later
    # to average over time
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        df_fit = data.stack("Time").dropna().iloc[:, 0]

    # The 1nM agonist, 1nM antagonist but antagonist=None is off in
    # the experiment SingleCell_Antagonism_3. Probably due to this condition
    # being on the corner of plates and thus drying up a little.
    # This condition is biologically identical to 1uM antagonist, antagonist=None,
    # so we replace the former by the latter in every dataset for consistency.
    sln = slice(None)
    slc = (sln, sln, sln, "1nM", "None", "1nM")
    slc2 = (sln, sln, sln, "1nM", "None", "1uM")
    # Setting one slice of a Series with another slice doesn't work. Can only
    # set a slice with a value or another df (with same columns and index)
    # So we set with values, after reindexing to make sure values are aligned
    rest_of_index = df_fit.loc[slc].index
    df_fit.loc[slc] = df_fit.loc[slc2].reindex(rest_of_index).values

    # Keep only the interesting stuff
    df_fit = df_fit.xs("N4", level="Agonist")
    tau_agonist = pep_tau_map.get("N4", 10.0)

    # To have better error bars, consider time points as replicates
    # So compute the ratio at every time, then compute geometric average
    # and also log-scale error bars across times and replicates.
    grouplvls = ["AgonistConcentration", "AntagonistConcentration", "Antagonist"]
    # Compute ratios of average time series
    df_fit = df_fit / df_fit.xs("None", level="Antagonist")

    # Use IL-2 and drop the no-agonist condition.
    df_fit = df_fit.xs("IL-2", level="Cytokine")
    df_fit = df_fit.drop("0", level="AgonistConcentration")

    # Convert index values to Ls and taus
    df_fit = df_fit.rename(pep_tau_map, axis=0, level="Antagonist")
    df_fit = df_fit.rename(read_conc_uM, axis=0, level="AgonistConcentration")
    df_fit = df_fit.rename(read_conc_uM, axis=0, level="AntagonistConcentration")
    df_fit = df_fit.rename(michaelis_menten_fitted, axis=0, level="AgonistConcentration")
    df_fit = df_fit.rename(michaelis_menten_fitted, axis=0, level="AntagonistConcentration")

    # Prepare dataframe to average over time and replicates
    df_gp = df_fit.groupby(grouplvls)

    # Estimate 95 % confidence interval based on t_crit*standard error on mean
    # in log2 scale
    # See for instance Scott 2020, section 8.1.2, CI on mean for unknown var
    n_dofs = df_gp.count()
    t_crits = sp.stats.t.ppf(0.975, n_dofs-1)
    df_ci = t_crits * np.log2(df_fit).groupby(grouplvls).std(ddof=1) / np.sqrt(n_dofs)
    df_ci = df_ci.dropna()
    # Clip ci to some small nonzero value, for the no antagonist
    # condition which has a ratio always identically 1.
    df_ci = df_ci.clip(lower=0.01)

    # Geometric average across experiments, time points, replicates
    df_fit = df_gp.apply(geo_mean_apply)
    df_fit = df_fit.dropna()
    df_fit = df_fit.reorder_levels(grouplvls)

    return df_fit, df_ci, tau_agonist


def prepare_data_6f(data, mm_params, pep_tau_map):
    """ Take cytokine concentration dataframe and rearrange it to compute
    average antagonism ratio and errors on these ratios for each condition.
    We use the geometric average over time, then we compute the ratio,
    and lastly, average the ratio across experimental replicates.
    """
    def michaelis_menten_fitted(conc_axis):
        return michaelis_menten(conc_axis, *mm_params)

    # Keep only the interesting stuff
    df_fit = data.xs("N4", level="Agonist")
    df_fit = df_fit.xs("6F", level="Genotype")
    tau_agonist = pep_tau_map.get("N4", 10.0)

    # Compute ratios of average time series
    df_fit = df_fit / df_fit.xs("None", level="Antagonist")

    # Prepare dataframe of antagonism ratios
    df_fit = df_fit.xs("IL-2", level="Cytokine")
    df_fit = df_fit.drop("None", level="AgonistConcentration")

    # Convert index values to Ls and taus
    df_fit = df_fit.rename(pep_tau_map, axis=0, level="Antagonist")
    df_fit = df_fit.rename(read_conc_uM, axis=0, level="AgonistConcentration")
    df_fit = df_fit.rename(read_conc_uM, axis=0, level="AntagonistConcentration")
    df_fit = df_fit.rename(michaelis_menten_fitted, axis=0, level="AgonistConcentration")
    df_fit = df_fit.rename(michaelis_menten_fitted, axis=0, level="AntagonistConcentration")

    # Group by varying parameters, treat the rest as replicates
    grouplvls = ["AgonistConcentration", "AntagonistConcentration", "Antagonist"]
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        df_fit = df_fit.stack("Time").dropna()  # Remove missing time points
    df_gp = df_fit.groupby(grouplvls)

    # Estimate 95 % confidence interval based on t_crit*standard error on mean
    # in log2 scale
    # See for instance Scott 2020, section 8.1.2, CI on mean for unknown var
    n_dofs = df_gp.count()
    t_crits = sp.stats.t.ppf(0.975, n_dofs-1)
    df_ci = t_crits * np.log2(df_fit).groupby(grouplvls).std(ddof=1) / np.sqrt(n_dofs)
    df_ci = df_ci.dropna()
    # Clip ci to some small nonzero value, for the no antagonist
    # condition which has a ratio always identically 1.
    df_ci = df_ci.clip(lower=0.01)

    # Geometric average across experiments, time points, replicates
    df_fit = df_gp.apply(geo_mean_apply)
    df_fit = df_fit.dropna()
    df_fit = df_fit.reorder_levels(grouplvls)

    return df_fit, df_ci, tau_agonist


def load_tcr_tcr_molec_numbers(molec_counts_fi, mtc, **kwargs):
    """
    Standard loading of surface molecule parameters for TCR-TCR antagonism,
    both old and new model, and also 6F T cells.

    Args:
        molec_counts_fi (str): path to file containing surface molecule summary stats
        mtc (str): metric/statistic to use, such as "Geometric mean"
    Keyword args:
        data_fold (str): path to main data folder. Typically ../data/
            because MCMC scripts in subfolder.
        tcell_type (str): "OT1_Naive" by default,
            also "OT1_CAR" and "OT1_Blast" available.
    Returns:
        tcr_number (float): number of TCRs per naive OT-1 T cell
        l_conc_mm_params (list of 2 floats): [max_mhc, pulse_kd]
            max_mhc (float): total number of MHC per B6 splenocyte
            pulse_kd (float): antigen pulse dissociation constant
        pep_tau_map_ot1 (dict): binding time estimated from KPR scaling law
            for each OVA variant peptide.
    """
    data_fold = kwargs.get("data_fold", "../data/")
    tcell_type = kwargs.get("tcell_type", "OT1_Naive")

    molec_stats = pd.read_hdf(molec_counts_fi, key="surface_numbers_stats")
    # Number of TCR per T cell
    tcr_number =  molec_stats.loc[(tcell_type, "TCR"), mtc]

    # Number of MHCs per APC (B6 splenocyte)
    max_mhc = molec_stats.loc[("B6", "MHC"), mtc]

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

    return tcr_number, l_conc_mm_params, pep_tau_map_ot1


def get_std_from_ci(molec_stats, key, mtc, alph=0.05):
    """
    Recover standard deviation of the mean estimator from the CI,
    assuming an underlying log-Gaussian distribution, which makes the mean
    estimator follow a Student's t distribution, because the variance
    of the underlying Gaussian is unknown, having only its estimator S.
    The variance of the mean estimator is S / sqrt(n), where S is
    related to the unbiased estimator of the underlying Gaussian variance;
    it is related to the CI of the mean estimator at level 1-alpha as:
    CI(1-alpha/2) - CI(alpha/2) = 2 * t^{n-1}_{alpha/2} S / sqrt(n)
    where n is the number of samples,
    Use the number of DOFs (cells) to get correct Student's t critical value
    """
    if key is not None:
        n_dofs = molec_stats.at[key, "dof"]
    else:
        n_dofs = molec_stats.at["dof"]
    alph2 = 0.5 * alph
    # The factor 2 is to divide the 2.5-97.5% interval in two
    # each half on either side of the median is t_crit * standard dev.
    factor_times_sigma = 2.0 * sp.stats.t.ppf(1.0 - alph2, n_dofs-1)
    ci_up_lbl = "CI {}".format(1.0 - alph2)
    ci_lo_lbl = "CI {}".format(alph2)
    if key is not None:
        ci_upper = molec_stats.at[key, mtc + " " + ci_up_lbl]
        ci_lower = molec_stats.at[key, mtc + " " + ci_lo_lbl]
    else:
        ci_upper = molec_stats.at[ci_up_lbl]
        ci_lower = molec_stats.at[ci_lo_lbl]
    if "Geometric" in mtc:
        ci_range = np.log10(ci_upper) - np.log10(ci_lower)
    else:
        ci_range = ci_upper - ci_lower
    scale_std = ci_range / factor_times_sigma
    return scale_std, n_dofs


def load_tcr_tcr_molec_numbers_ci(molec_counts_fi, mtc, **kwargs):
    """
    Standard loading of surface molecule parameters for TCR-TCR antagonism,
    both old and new model, and also 6F T cells. Also load standard
    deviation of the TCR and MHC numbers and of the loading K_D parameter.

    Args:
        molec_counts_fi (str): path to file containing surface molecule summary stats
        mtc (str): metric/statistic to use, such as "Geometric mean"
    Keyword args:
        data_fold (str): path to main data folder. Typically ../data/
            because MCMC scripts in subfolder.
        tcell_type (str): "OT1_Naive" by default,
            also "OT1_CAR" and "OT1_Blast" available.
    Returns:
        tcr_number (float): number of TCRs per naive OT-1 T cell
        l_conc_mm_params (list of 2 floats): [max_mhc, pulse_kd]
            max_mhc (float): total number of MHC per B6 splenocyte
            pulse_kd (float): antigen pulse dissociation constant
        pep_tau_map_ot1 (dict): binding time estimated from KPR scaling law
            for each OVA variant peptide.
        tcr_log10_std (float): standard deviation of the mean estimator,
            either in log10 scale if mtc is "Geometric mean", or in linear
            scale if mtc is just "Mean" or something else.
        l_conc_mm_params_std (list of 2 floats): standard deviations of the
            mean estimators of the total MHC number and the loading EC50,
            either in log10 scale or linear scale, depending on mtc choice.
        dofs_tcr_mhc_kd (list of 3 ints): the number of degrees of freedom
            (cell or experimental repeats) from which the mean estimators
            for the TCR number, MHC number, and loading EC50 are derived.
    """
    data_fold = kwargs.get("data_fold", "../data/")
    tcell_type = kwargs.get("tcell_type", "OT1_Naive")

    molec_stats = pd.read_hdf(molec_counts_fi, key="surface_numbers_stats")
    # Number of TCR per T cell
    tcr_number =  molec_stats.at[(tcell_type, "TCR"), mtc]

    # Recover standard deviation of the mean estimator from the CI
    tcr_log10_std, n_dofs_tcr = get_std_from_ci(molec_stats, (tcell_type, "TCR"), mtc)

    # Number of MHCs per APC (B6 splenocyte)
    max_mhc = molec_stats.loc[("B6", "MHC"), mtc]

    # Recover standard deviation from CI
    mhc_log10_std, n_dofs_mhc = get_std_from_ci(molec_stats, ("B6", "MHC"), mtc)

    # Pulse concentration to ligand numbers conversion:
    # based on RMA-S 2019-2020 data
    mhc_pulse_kd = pd.read_hdf(molec_counts_fi, key="mhc_pulse_kd")
    # This pd.Series also contains covariance of the K_D parameter fit
    pulse_kd = mhc_pulse_kd[mtc]

    # Again, infer std from CI on the mean estimator
    # Use the number of DOFs (number of exp. loading EC50 curves replicates)
    kd_log10_std, n_dofs_kd = get_std_from_ci(mhc_pulse_kd, None, mtc)

    # Use average parameters. amplitude, EC50
    l_conc_mm_params = [max_mhc, pulse_kd]
    l_conc_mm_params_std = [mhc_log10_std, kd_log10_std]

    # Mapping tau and EC50s
    with open(os.path.join(data_fold, "pep_tau_map_ot1.json"), "r") as handle:
        pep_tau_map_ot1 = json.load(handle)

    # Also return dofs: TCR, MHC, KD
    dofs_tcr_mhc_kd = [n_dofs_tcr, n_dofs_mhc, n_dofs_kd]

    return (tcr_number, l_conc_mm_params, pep_tau_map_ot1,
                tcr_log10_std, l_conc_mm_params_std, dofs_tcr_mhc_kd)


### ANALYZING MCMC RUNS ###

def assemble_kf_vals(kf, m_axis, results_dict, kind="posterior_probs", stratkey="MAP best"):
    """ Put together results of some kind for a given k, f, as a
    function of m values in m_axis
    """
    pts_kept = []
    for m in m_axis:
        try:
            pts_kept.append(results_dict[str((kf[0], m, kf[1]))][kind][stratkey])
        except:
            continue
    return pts_kept


def check_fit_model_antagonism(model_panel, pvec, grid_pt, df_ratio, df_err,
                    other_args=(), n_taus=101, antagonist_lvl="Antagonist"):
    """ Receive a fitted parameter vector and a point on the grid search,
    as well as a function evaluating antagonism panel for some model,
    and compute the model prediction compared to data.
    Make a plot comparing model and data as a function of antagonist quality.
    Hence, create an artificial antagonist axis for the model predictions
    that is continuous.

    Args:
        model_panel (callable): function evaluating model antagonism ratio
            for given (pvec, grid_pt, *other_args, df_ratio.index).
        pvec (np.ndarray): fitted parameters
        grid_pt (tuple of ints): integer parameters after grid search.
        df_ratio (pd.DataFrame): antagonism ratio points
        df_err (pd.DataFrame): antagonism ratio error bars.
        other_args (tuple): other arguments passed to model_panel.
        n_taus (int): number of tau points to compute
        antagonist_lvl (str): name of the level containing the antagonist taus.
            Default: 'Antagonist'; use 'TCR_Antigen' for CAR-TCR antagonism.

    Returns:
        df_model (pd.DataFrame): model panel antagonism predictions
            at each condition in df_ratio's index and a range of
            closely spaced taus.
    """
    # Create continuous antagonist tau range.
    new_tau_range = np.linspace(0.001, df_ratio.index.get_level_values(antagonist_lvl).max(), n_taus)
    old_index = df_ratio.index.droplevel(antagonist_lvl).unique()
    if isinstance(old_index, pd.MultiIndex):  # 2 ore more levels left
        product_tuples = [tuple(a)+(b,) for b in new_tau_range for a in old_index]
    elif isinstance(old_index, pd.Index):  # index elements are not iterables
        product_tuples = [(a,)+(b,) for b in new_tau_range for a in old_index]
    new_names = list(old_index.names) + [antagonist_lvl]
    new_index = pd.MultiIndex.from_tuples(product_tuples, names=new_names)

    # Compute the antagonism ratio panel.
    df_model = model_panel(pvec, grid_pt, *other_args, new_index)

    return df_model


def plot_fit_antagonism(
        df_ratio, df_model, l_conc_mm_params, df_err, cost=None, model_ci=None
    ):
    # Aesthetic parameters
    with open(os.path.join("..", "results", "for_plots", 
            "perturbations_palette.json"), "r") as f:
        pert_palette = json.load(f)
    pert_palette["None"] = [0., 0., 0., 1.]  # Black
    # Prepare data to plot
    df_model_data = pd.concat({"Data":df_ratio, "Model": df_model}, names=["Source"])
    df_model_data.name = "Antagonism ratio"

    # Rename concentrations for nicer plotting
    def reverse_mm_fitted(x):
        return inverse_michaelis_menten(x, *l_conc_mm_params)
    def renamer(d):
        return (d.rename(reverse_mm_fitted, axis=0, level="AgonistConcentration")
                .rename(write_conc_uM, axis=0, level="AgonistConcentration")
                .rename(reverse_mm_fitted, axis=0, level="AntagonistConcentration")
                .rename(write_conc_uM, axis=0, level="AntagonistConcentration")
                )

    df_model_data = renamer(df_model_data).sort_index()
    df_err = renamer(df_err).sort_index()
    if model_ci is not None:
        model_ci = renamer(model_ci).sort_index()

    # Prepare color palette
    available_antagconc = list(df_model_data.index
                        .get_level_values("AntagonistConcentration").unique())
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

    available_agconc = df_model_data.index.get_level_values("AgonistConcentration").unique()
    available_agconc = sorted(list(available_agconc), key=read_conc_uM)

    # Now make nice plots. Don't use seaborn because it adds lines between data points
    fig, axes = plt.subplots(nrows=1, ncols=len(available_agconc), sharey="row")
    fig.set_size_inches(3*len(available_agconc), 3.0)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    index_antag = list(df_model_data.index.names).index("AntagonistConcentration")
    index_ag = list(df_model_data.index.names).index("AgonistConcentration")
    for i, ag_conc in enumerate(available_agconc):
        ax = axes[i]
        for j, antag_conc in enumerate(available_antagconc):
            conc_key = ((ag_conc, antag_conc) if index_ag < index_antag
                                    else (antag_conc, ag_conc))
            try:
                data_pts = np.log2(df_model_data.loc[("Data", *conc_key)]).sort_index()
            except KeyError:
                continue  # This agonist, antagonist pair isn't available
            else:
                model_pts = np.log2(df_model_data.loc[("Model", *conc_key)]).sort_index()
                err_pts = df_err.loc[conc_key].sort_index()
                if model_ci is not None:
                    percentiles = sorted(list(model_ci.columns), 
                                        key=lambda x: float(x.split("_")[-1]))
                    model_ci_low = np.log2(model_ci.loc[conc_key, percentiles[0]]).sort_index()
                    model_ci_hi = np.log2(model_ci.loc[conc_key, percentiles[1]]).sort_index()

            hue = palette.get(antag_conc)
            mark = markers.get(antag_conc)
            style = linestyles.get(antag_conc)
            if model_ci is not None:
                # First shade confidence interval
                ax.fill_between(model_ci_low.index.get_level_values("Antagonist").values,
                            model_ci_low.values, model_ci_hi.values, color=hue, alpha=0.25)
            errbar = ax.errorbar(data_pts.index.get_level_values("Antagonist").values, data_pts.values,
                       yerr=err_pts.values, marker=mark, ls="none", color=hue, mfc=hue, mec=hue)
            li, = ax.plot(model_pts.index.get_level_values("Antagonist").values, model_pts.values,
                       color=hue, ls=style, label=antag_conc)
        # Label this plot
        ax.set_xlabel(r"Antagonist model $\tau$ (s)")
        if i == 0:
            ax.set_ylabel(r"FC$_{TCR \rightarrow TCR}$")
        # Change y tick labels to 2^x
        ax = change_log_ticks(ax, base=2, which="y")
        ax.axhline(0.0, ls="--", color="k", lw=1.0)
        ax.set_title(ag_conc + " Agonist", y=0.95, va="top")
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)

    # Annotate the last plot with the cost function
    if cost is not None:
        axes[-1].annotate("Log posterior:{:.3f}".format(cost), xy=(0.95, 0.05),
                        xycoords="axes fraction", ha="right", va="bottom")

    # Add a figure legend, manual to indicate markers and hues, model and data
    fig.tight_layout()
    leg_handles, leg_handler = data_model_handles_legend(palette, markers, linestyles,
                        "Antagonist\nconcentration", model_style="-",
                        model_lw=li.get_linewidth(), data_size=errbar[0].get_markersize(), data_marker="o")
    fig.legend(handles=leg_handles, handler_map=leg_handler,
               loc="upper left", bbox_to_anchor=(0.975, 0.95), frameon=False)

    return fig, axes


### Confidence interval on model fits
# Small utility functions to generate samples in log-scale
def sample_lognorm(mean, std, rng, n_samp, base=10.0):
    """ Sample n_samp mean estimator samples from a log-scale Gaussian
    distribution with standard deviation std. The log-scale is in some base
    (10 by default). rng is a NumPy random number generator.
    """
    samples = np.log(mean)/np.log(base) + std*rng.standard_normal(size=n_samp)
    return base ** samples


def sample_log_student(mean, std, rng, n_samp, n_dof, base=10.0):
    """ Sample n_samp mean estimator samples from a log-scale Student's t
    distribution with n_dof-1 degrees of freedom and standard deviation
    std. The log-scale is in some base (10 by default).
    rng is a NumPy random number generator.
    """
    noises = std * rng.standard_t(n_dof-1, size=n_samp)
    samples = np.log(mean) / np.log(base) + noises
    return base ** samples

# Model CI estimation for TCR/TCR which includes uncertainty on ligand numbers
# and uncertainty on receptor numbers too.
def confidence_model_antagonism_tcr(
        model_panel, psamples, pbest, grid_pt, df_ratio, df_err, other_args=(),
        n_taus=101, n_samp=1000, seed=None, antagonist_lvl="Antagonist",
        l_conc_mm_params=[1e5, 0.1], tcr_num_std=0.1, mm_params_std=[0.1, 0.1],
        dofs_tcr_mhc_kd=[1000, 1000, 24]
    ):
    """ Receive parameter samples and a point on the grid search,
    as well as a function evaluating antagonism panel for some model,
    and compute the confidence interval of model predictions compared to data.

    Rigorously, the mean TCR, MHC numbers and K_D must be sampled from
    a Student's t distribution, with scale given by the standard deviations
    provided in tcr_num_std and mm_params_std, and number of degrees of
    freedom provided in dofs_tcr_mhc_kd.

    Args:
        model_panel (callable): function evaluating model antagonism ratio
            for given (pvec, grid_pt, *other_args, df_ratio.index).
        psamples (np.ndarray): sampled parameters array,
            shaped [param, nwalkers, nsamples]
        pbest (np.ndarray): best parameter sample
        grid_pt (tuple of ints): integer parameters after grid search.
        df_ratio (pd.DataFrame): antagonism ratio points
        df_err (pd.DataFrame): antagonism ratio error bars.
        other_args (tuple): other arguments passed to model_panel.
        seed (np.random.SeedSequence or int): random number generator seed
        n_samp (int): number of samples to draw for confidence intervals
        antagonist_lvl (str): which level contains the antagonist tau,
            will be dropped and replaced by a range of taus.
        l_conc_mm_params (list of 2 floats): mean MHC number per APC
            and peptide loading EC50 mean estimator.
        tcr_num_std (float): standard deviation of the mean estimator
            of the TCR number per T cell
        mm_params_std (float): standard deviation of the mean estimators
            for the MHC number and loading EC50 of an APC.
        dofs_tcr_mhc_kd (list of 3 ints): the number of degrees of freedom
            (cell or experimental repeats) from which the mean estimators
            for the TCR number, MHC number, and loading EC50 are derived.

    Returns:
        df_stats (pd.DataFrame): statistics on model antagonism predictions,
            including 90 % CI, median, best fit, as a function of tau.
    """
    rgen = np.random.default_rng(seed)

    # Recover the pulse concentrations from the ligand numbers computed
    # with the best Michaelis-Menten parameters. These pulse concentrations
    # will correspond to slightly different ligand numbers for each
    # l_conc_mm_params values sampled when bootstrapping the CI.
    best_ag_nums = df_ratio.index.get_level_values("AgonistConcentration").unique()
    best_antag_nums = df_ratio.index.get_level_values("AntagonistConcentration").unique()
    ag_pulses = inverse_michaelis_menten(best_ag_nums, *l_conc_mm_params)
    antag_pulses = inverse_michaelis_menten(best_antag_nums, *l_conc_mm_params)
    ag_l_conc_map = dict(zip(best_ag_nums, ag_pulses))
    antag_l_conc_map = dict(zip(best_antag_nums, antag_pulses))
    inv_l_ag_map = dict(zip(ag_l_conc_map.values(), ag_l_conc_map.keys()))
    inv_l_antag_map = dict(zip(antag_l_conc_map.values(), antag_l_conc_map.keys()))

    # Create continuous antagonist tau range, add pulse concentration levels
    # to be converted to different ligand numbers for every sample.
    new_tau_range = np.linspace(0.001, df_ratio.index.get_level_values(antagonist_lvl).max(), n_taus)
    old_index = df_ratio.index.droplevel(antagonist_lvl).unique()
    product_tuples = [tuple(a)+(b,) for b in new_tau_range for a in old_index]
    new_names = list(old_index.names) + [antagonist_lvl]
    # Turn into a DataFrame without an index to better add and map index levels
    # to deal with pulse concentration to ligand number conversions
    new_index = pd.MultiIndex.from_tuples(product_tuples, names=new_names)
    df_idx = new_index.to_frame()
    df_idx["AgonistPulse"] = df_idx["AgonistConcentration"].map(ag_l_conc_map)
    df_idx["AntagonistPulse"] = df_idx["AntagonistConcentration"].map(antag_l_conc_map)

    # Replace the ri_tot default by a local list to be modified each iteration
    other_args_ci = list(other_args)
    # From the loaded cost_args, extract TCR number and l_conc_mm_params
    r_arg_type = type(other_args[1])
    if r_arg_type in [int, float, np.float64]:
        tcr_num_estim = other_args[1]  # r_tot
    elif r_arg_type in [list, tuple, np.ndarray, set]:
        tcr_num_estim = other_args[1][0]  # ri_tot[0]
        # Replace the default list at other_args[1] by a local list
        other_args_ci[1] = [tcr_num_estim, other_args[1][1]]
    else:
        raise ValueError("other_args unexpected order: {}".format(other_args))

    new_cols = pd.RangeIndex(n_samp)
    df_model_idx = df_idx.set_index(["AgonistPulse", "AntagonistPulse"]
                                        + list(new_index.names[2:])).index
    df_model = pd.DataFrame(np.zeros([len(new_index), n_samp]),
                    index=df_model_idx, columns=new_cols)

    # Randomly sample parameters from the MCMC chain
    samp_choices = rgen.choice(psamples.shape[1]*psamples.shape[2], size=n_samp, replace=True)
    s_choices = samp_choices // psamples.shape[1]
    w_choices = samp_choices % psamples.shape[1]

    # Student's t samples for the TCR, MHC, and K_D mean estimators
    n_dofs_tcr, n_dofs_mhc, n_dofs_kd = dofs_tcr_mhc_kd
    tcr_num_samples = sample_log_student(
        tcr_num_estim, tcr_num_std, rgen, n_samp, n_dofs_tcr, base=10.0
    )
    mhc_num_samples = sample_log_student(
        l_conc_mm_params[0], mm_params_std[0], rgen,
        n_samp, n_dofs_mhc, base=10.0
    )
    load_kd_samples = sample_log_student(
        l_conc_mm_params[1], mm_params_std[1], rgen,
        n_samp, n_dofs_kd, base=10.0
    )

    # Compute the antagonism ratio panel for each sample parameter set
    # The last levels should be l_ag, l_antag, antag_tau,
    # but previous levels can be the pulse concentrations
    full_idx_order = ["AgonistPulse", "AntagonistPulse"] + list(new_index.names)
    for i in range(n_samp):
        nw, ns = w_choices[i], s_choices[i]
        pvec = psamples[:, nw, ns]
        # Place new TCR number in other_args_ci
        if r_arg_type in [int, float, np.float64]:
            other_args_ci[1] = tcr_num_samples[i]
        elif r_arg_type in [list, tuple, np.ndarray, set]:
            other_args_ci[1][0] = tcr_num_samples[i]
        # Prepare the new index with the sampled mhc_num and load_kd
        df_idx["AgonistConcentration"] = michaelis_menten(
            df_idx["AgonistPulse"], mhc_num_samples[i], load_kd_samples[i]
        )
        df_idx["AntagonistConcentration"] = michaelis_menten(
            df_idx["AntagonistPulse"], mhc_num_samples[i], load_kd_samples[i]
        )

        new_index_ci = df_idx.set_index(full_idx_order).index
        try:
            df_model_i = model_panel(pvec, grid_pt, *other_args_ci, new_index_ci)
        except RuntimeError:
            df_model_i = np.nan
        else:
            df_model_i = df_model_i.droplevel(
                ["AgonistConcentration", "AntagonistConcentration"], axis=0
            )
        df_model[i] = df_model_i
    print("Number NaN samples for {}:".format(grid_pt),
                df_model.isna().iloc[0, :].sum())

    # Compute model output for best parameter vector.
    # Need an index with pulse concentrations and the default L numbers
    df_idx["AgonistConcentration"] = df_idx["AgonistPulse"].map(inv_l_ag_map)
    df_idx["AntagonistConcentration"] = df_idx["AntagonistPulse"].map(inv_l_antag_map)
    new_index_ci = df_idx.set_index(full_idx_order).index
    try:
        sr_best = model_panel(pbest, grid_pt, *other_args, new_index_ci)
    except RuntimeError:
        sr_best = np.nan
    else:
        sr_best = sr_best.droplevel(
            ["AgonistConcentration", "AntagonistConcentration"], axis=0
        )

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
        .rename(names="AgonistConcentration", level="AgonistPulse")
        .rename(names="AntagonistConcentration", level="AntagonistPulse")
    )
    df_stats = (df_stats
        .rename(inv_l_antag_map, axis=0, level="AntagonistConcentration")
        .rename(inv_l_ag_map, axis=0, level="AgonistConcentration")
    )

    print("Finished computing model predictions for {}".format(grid_pt))
    return df_stats


# Function to check model outputs
def check_model_output(model, rates, rstot, nmf):
    # Define the range of L_i, tau_i to test
    tau_range = np.asarray([10.0, 7.0, 5.0, 3.0, 2.0])
    l_range = np.logspace(0, 5, 101)
    # Index each output array [l2, tau2]
    output = np.zeros([tau_range.size, l_range.size])
    for i in range(output.shape[0]):  # over tau
        taup = tau_range[i]
        for j in range(output.shape[1]):  # over L
            lp = l_range[j]
            output[i, j] = model(rates, taup, lp, rstot, nmf)[nmf[0]]
    return l_range, tau_range, output
