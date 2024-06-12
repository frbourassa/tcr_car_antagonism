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

from mcmc.plotting import data_model_handles_legend, change_log_ticks
from utils.preprocess import (
    geo_mean,
    geo_mean_apply,
    read_conc_uM,
    write_conc_uM,
    hill,
    michaelis_menten,
    loglog_michaelis_menten,
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


def plot_fit_antagonism(df_ratio, df_model, l_conc_mm_params, df_err, cost=None):
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

    df_model_data = renamer(df_model_data)
    df_err = renamer(df_err)
    df_model_data = df_model_data.sort_index()
    df_err = df_err.sort_index()

    # Prepare palette, columns, etc.
    available_antagconc = list(df_model_data.index
                        .get_level_values("AntagonistConcentration").unique())
    available_antagconc = sorted(available_antagconc, key=read_conc_uM)

    if len(available_antagconc) > 1:
        palette = sns.color_palette("BuPu", n_colors=len(available_antagconc))
    else:
        palette = [(0, 0, 0, 1)]
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

            hue = palette.get(antag_conc)
            mark = markers.get(antag_conc)
            style = linestyles.get(antag_conc)
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
# This function can also be used for TCR/CAR
def confidence_model_antagonism(model_panel, psamples, pbest, grid_pt,
        df_ratio, df_err, other_args=(), n_taus=101, n_samp=1000, seed=None,
        antagonist_lvl="Antagonist"):
    """ Receive parameter samples and a point on the grid search,
    as well as a function evaluating antagonism panel for some model,
    and compute the confidence interval of model predictions compared to data.

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
        antagonist_lvl (str):

    Returns:
        df_stats (pd.DataFrame): statistics on model antagonism predictions,
            including 90 % CI, median, best fit, as a function of tau.
    """
    rgen = np.random.default_rng(seed)

    # Create continuous antagonist tau range.
    new_tau_range = np.linspace(0.001, df_ratio.index.get_level_values(antagonist_lvl).max(), n_taus)
    old_index = df_ratio.index.droplevel(antagonist_lvl).unique()
    if isinstance(old_index, pd.MultiIndex):  # 2 ore more levels left
        product_tuples = [tuple(a)+(b,) for b in new_tau_range for a in old_index]
    elif isinstance(old_index, pd.Index):  # index elements are not iterables
        product_tuples = [(a,)+(b,) for b in new_tau_range for a in old_index]
    new_names = list(old_index.names) + [antagonist_lvl]
    new_index = pd.MultiIndex.from_tuples(product_tuples, names=new_names)

    new_cols = pd.RangeIndex(n_samp)
    df_model = pd.DataFrame(np.zeros([len(new_index), n_samp]),
                    index=new_index, columns=new_cols)

    # Randomly sample parameters from the MCMC chain
    samp_choices = rgen.choice(psamples.shape[1]*psamples.shape[2], size=n_samp, replace=True)
    s_choices = samp_choices // psamples.shape[1]
    w_choices = samp_choices % psamples.shape[1]

    # Compute the antagonism ratio panel for each sample parameter set
    for i in range(n_samp):
        nw, ns = w_choices[i], s_choices[i]
        pvec = psamples[:, nw, ns]
        try:
            df_model[i] = model_panel(pvec, grid_pt, *other_args, new_index)
        except RuntimeError:
            df_model[i] = np.nan
    print("Number NaN samples for {}:".format(grid_pt),
                df_model.isna().iloc[0, :].sum())

    # Compute model output for best parameter vector.
    sr_best = model_panel(pbest, grid_pt, *other_args, new_index)

    # Compute statistics of the ratios on a log scale, then back to linear scale
    stats = ["percentile_5", "median", "mean", "geo_mean", "best", "percentile_95"]
    df_stats = pd.DataFrame(np.zeros([df_model.shape[0], len(stats)]),
                index=new_index, columns=stats)
    df_stats["mean"] = np.log(df_model.mean(axis=1))
    df_model = np.log(df_model)
    df_stats["percentile_5"] = df_model.quantile(q=0.05, axis=1)
    df_stats["median"] = df_model.quantile(q=0.5, axis=1)
    df_stats["geo_mean"] = df_model.mean(axis=1)
    df_stats["best"] = np.log(sr_best)
    df_stats["percentile_95"] = df_model.quantile(q=0.95, axis=1)
    df_stats = np.exp(df_stats)
    print("Finished computing model predictions for {}".format(grid_pt))
    return df_stats



## Function to visualize the analysis results.
# TODO: finish coding this plotting function
def plot_fitted_params(popt, pvar, idx):
    """ Assuming we have two index levels: TCR_Antigen and CAR_Antigen """
    raise NotImplementedError()
    # Prepare data for plotting
    tcr_ag_order = ["N4", "A2", "Y3", "Q4", "T4", "V4", "G4", "E1", "None"]
    tcr_ag_order = [p for p in tcr_ag_order if p in
                        idx.get_level_values("TCR_Antigen").unique()]
    npar = popt.shape[1]
    ncol = 2
    nrow = npar // ncol + min(1, npar % ncol)

    fig, axes = plt.subplots(nrow, ncol, sharex=True)
    fig.set_size_inches(7, 5)
    axes  = axes.flatten()
    xaxis = np.arange(len(tcr_ag_order))
    for i, ax in enumerate(axes):
        par = popt.columns.values[i]
        ax.errorbar(xaxis, popt.loc[(tcr_ag_order, "None"), par],
            yerr=np.sqrt(pvar.loc[(tcr_ag_order, "None"), par]),
            label="Ag alone",ls="--", marker="o")
        ax.errorbar(xaxis+0.065, popt.loc[(tcr_ag_order, "CD19"), par],
            yerr=np.sqrt(pvar.loc[(tcr_ag_order, "None"), par]),
            label="Ag + CD19", ls="-", marker="s")
        # Determine decent y limits
        ymin = popt.loc[tcr_ag_order, par].min()
        ymax = popt.loc[tcr_ag_order, par].max()
        yrange = ymax - ymin
        ax.set_ylim(ymin - 0.2*yrange, ymax + 0.2*yrange)
        ax.set(ylabel=r"${}$".format(par))
        ax.set_xticks(xaxis)
        ax.set_xticklabels(tcr_ag_order)
        ax.legend()
    fig.tight_layout()
    return fig, axes


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
