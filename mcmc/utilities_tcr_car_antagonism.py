"""
Functions to prepare data for MCMC fitting of antagonism ratios.

@author: frbourassa
December 2022
"""
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json, h5py
import os

# Import local functions
from mcmc.utilities_tcr_tcr_antagonism import (
    assemble_kf_vals,
    check_fit_model_antagonism,
    confidence_model_antagonism
)
from mcmc.plotting import (
    handles_properties_legend,
    prepare_hues,
    prepare_styles,
    prepare_markers,
    prepare_subplots,
    data_model_handles_legend,
    change_log_ticks
    )
from mcmc.mcmc_analysis import find_best_grid_point
from utils.preprocess import (
    groupby_others,
    geo_mean_levels,
    ln10,
    time_dd_hh_mm_ss,
    geo_mean,
    geo_mean_apply,
    read_conc_uM,
    write_conc_uM,
    hill,
    michaelis_menten,
    loglog_michaelis_menten,
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

    # Keep only experiments 9 and 14, which are high-quality.
    # "20211027-OT1_CAR_11" is not too bad, but only 1 uM
    # and small amplitude compared to usual
    with open(os.path.join(data_fold, "dataset_selection.json"), "r") as h:
        good_dsets = json.load(h).get("good_car_tcr_datasets")
    df = data.loc[data.index.isin(good_dsets, level="Data")]

    # Create a data-spleen level
    df = df.stack().to_frame()
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
    # TODO: reconsider processing of the no-antagonist conditions when
    # there are duplicates that could introduce a bias.
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


def load_tcr_tcr_akpr_fits(res_file, analysis_file, klim=2, wanted_kmf=None):
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
# check_fit_model_car_antagonism: same as check_fit_model_antagonism
# written for TCR-TCR, just pass antagonist_lvl = "TCR_Antigen"

def plot_fit_car_antagonism(df_ratio, df_model, l_conc_mm_params, df_err, cost=None):
    df_model_data = pd.concat({"Data":df_ratio, "Model": df_model}, names=["Source"])
    df_model_data.name = "Antagonism ratio"

    # Rename concentrations for nicer plotting
    def reverse_mm_fitted(x):
        return inverse_michaelis_menten(x, *l_conc_mm_params)
    def renamer(d):
        return (d.rename(reverse_mm_fitted, axis=0, level="TCR_Antigen_Density")
                 .rename(write_conc_uM, axis=0, level="TCR_Antigen_Density")
                )
    df_model_data = renamer(df_model_data)
    df_err = renamer(df_err)
    df_model_data = df_model_data.sort_index()
    df_err = df_err.sort_index()

    # Prepare palette, columns, etc.
    available_antagconc = list(df_model_data.index
                        .get_level_values("TCR_Antigen_Density").unique())
    available_antagconc = sorted(available_antagconc, key=read_conc_uM)

    palette = sns.color_palette("BuPu", n_colors=len(available_antagconc))
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

        hue = palette.get(antag_conc)
        mark = markers.get(antag_conc)
        style = linestyles.get(antag_conc)
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


# Function to check TCR-CAR model outputs
def check_model_output_tcr_car(model, rates, rstot, nmf):
    raise NotImplementedError()
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
    with open("../results/for_plots/perturbations_palette.json", "r") as f:
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
                ax.fill_between(xvals, mod_loc2["percentile_5"], mod_loc2["percentile_95"],
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
