import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sys, os
import h5py, json
import itertools
import multiprocessing
from time import perf_counter

# Local modules
import sys, os
if not "../" in sys.path:
    sys.path.insert(1, "../")

from mcmc.mcmc_analysis import (
    analyze_mcmc_run,
    graph_mcmc_samples,
    find_best_grid_point,
    rebuild_grid,
    fit_summary_to_json
)
from mcmc.costs_tcr_tcr_antagonism import (
    cost_antagonism_akpr_i,
    cost_antagonism_shp1,
    cost_antagonism_akpr_priors,
    cost_antagonism_shp1_priors
)
from mcmc.costs_tcr_tcr_antagonism import (
    antag_ratio_panel_akpr_i,
    antag_ratio_panel_shp1,
    steady_akpr_i_1ligand
)
from mcmc.utilities_tcr_tcr_antagonism import (
    prepare_data,
    prepare_data_6f,
    plot_fit_antagonism,
    check_model_output,
    confidence_model_antagonism_tcr,
    load_tcr_tcr_molec_numbers_ci
)
from utils.preprocess import (
    string_to_tuple,
    inverse_michaelis_menten,
    ln10,
    write_conc_uM
)
from models.conversion import convert_tau_ec50_relative

# For multiprocessing
from utils.cpu_affinity import count_parallel_cpu
n_cpu = count_parallel_cpu()

# This means we are on the Physics servers
if sys.platform == 'linux':
    mpl.use("Agg")


### MCMC RUN ANALYSIS ###
def main_mcmc_analysis(fl, model_kwargs, do_save=False, do_show=False):
    """ Choose among 3 options for model:
        "akpr_i": TCR/TCR antagonism on 6Y T cells with updated model
        "shp1": TCR/TCR antagonism on 6Y T cells with Francois 2013 model
        "6f": TCR/TCR on 6F T cells with updated model
    Pass a dictionary of keyword arguments specifying file names etc.
    """
    # Load folder names, etc. for each model choice
    analysis_res_fname = model_kwargs.get("analysis_res_fname")
    fit_summary_fname = model_kwargs.get("fit_summary_fname")
    results_fname = model_kwargs.get("results_fname")
    fig_subfolder = model_kwargs.get("fig_subfolder")
    model = model_kwargs.get("model", "none")
    # Can't pickle functions for multiprocessing, but these function choices
    # are the same for all kinds of runs we want to do with each model.
    if model == "akpr_i":
        data_prep_fct = prepare_data
        cost_fct = (cost_antagonism_akpr_priors if "priors"
                    in analysis_res_fname else cost_antagonism_akpr_i)
    elif model == "shp1":
        data_prep_fct = prepare_data
        cost_fct = (cost_antagonism_shp1_priors if "priors" in analysis_res_fname
                        else cost_antagonism_shp1)
    elif model == "6f":
        data_prep_fct = prepare_data_6f
        cost_fct = (cost_antagonism_akpr_priors if "priors"
                    in analysis_res_fname else cost_antagonism_akpr_i)
    else:
        raise ValueError("Model {} unknown;".format(model)
                        + "choose among 'akpr_i', 'shp1', '6f'")
    results = h5py.File(os.path.join(fl, results_fname), "r")

    # Import MCMC results
    samples_group = results.get("samples")
    cost_group = results.get("cost")
    data_group = results.get("data")

    # Import data
    data_file_name = data_group.attrs.get("data_file_name")
    l_conc_mm_params = data_group.get("l_conc_mm_params")[()]
    df = pd.read_hdf(data_file_name)
    tau_file = os.path.join("..", "data", "pep_tau_map_ot1.json")
    with open(tau_file, "r") as handle:
        pep_tau_map_ot1 = json.load(handle)
    df_fit, df_ci_log2, _ = data_prep_fct(df,
                                    l_conc_mm_params, pep_tau_map_ot1)

    # Analyze each run
    all_results_dicts = {}
    # Reconstitute cost function arguments, including data
    cost_args_loaded = [data_group.get(a)[()]
                        for a in data_group.attrs.get("cost_args_names")]
    cost_args_loaded += [df_fit, df_ci_log2]
    for cond in list(samples_group.keys()):
        # In this case, we used the grid search point as the run_id.
        grid_point = samples_group.get(cond).attrs.get("run_id")
        # And it should be the first cost argument.
        cost_args = [grid_point] + cost_args_loaded
        results_dict = analyze_mcmc_run(cond, samples_group.get(cond),
                cost_group.get(cond), dict(samples_group.attrs),
                cost_fct, cost_args=cost_args)
        all_results_dicts[cond] = results_dict

    with open(os.path.join(fl, analysis_res_fname), "w") as jfile:
        json.dump(all_results_dicts, jfile, indent=2)

    # Write nice summary of all model parameters to JSON file
    k_limit = 1 if model == "6f" else 2
    fit_summ = fit_summary_to_json(
        data_group, samples_group, all_results_dicts, meth="MAP best",
        klim=k_limit
    )
    with open(os.path.join(fl, fit_summary_fname), "w") as h:
        json.dump(fit_summ, h, indent=2)

    # Use the analysis results to make plots
    figures_folder = os.path.join("..", "figures", fig_subfolder)
    cost_args_loaded = [data_group.get(a)[()]
                        for a in data_group.attrs.get("cost_args_names")]

    # Loop over grid points
    if not (do_save or do_show):  # No plots
        return 0
    for cond in all_results_dicts.keys():
        # Graphs of samples
        grid_point = samples_group.get(cond).attrs.get("run_id")
        graph_mcmc_samples(cond, samples_group.get(cond),
            cost_group.get(cond), dict(samples_group.attrs),
            all_results_dicts[cond], skip=5, walker_skip=4,
            figures_folder=figures_folder, do_save=do_save,
            do_show=do_show, plot_chain=False)

        print("Finished graphing condition {}".format(cond))
    return 0


### 6Y TCR MODEL ANALYSIS ###
def main_compare_models_figures(fl_res, fl_plots, cost_args,
                    do_save=False, do_show=False):
    # Load results files
    with open(os.path.join(fl_res, "mcmc_analysis_akpr_i.json"), "r") as h:
        results_akpr = json.load(h)
    with open(os.path.join(fl_res, "mcmc_analysis_shp1.json"), "r") as h:
        results_shp1 = json.load(h)

    # Find best kmf or m, print results
    strategy = "best"
    bests_akpr = find_best_grid_point(results_akpr, strat=strategy)
    bests_shp1 = find_best_grid_point(results_shp1, strat=strategy)
    # Exponentiate back parameters
    bests_akpr, bests_shp1 = list(bests_akpr), list(bests_shp1)
    bests_akpr[1] = np.exp(bests_akpr[1]*ln10)
    bests_shp1[1] = np.exp(bests_shp1[1]*ln10)
    # Print results
    print("Best AKPR k, m, f:", bests_akpr[0])
    print("With C_m_thresh, I_thresh =", bests_akpr[1])
    print("Cost function:", -bests_akpr[2])

    print("Best SHP-1 m:", bests_shp1[0])
    print("With C_m_thresh, I_tot =", bests_shp1[1])
    print("Cost function:", -bests_shp1[2])

    # Heatmap/plot of the best k or k, m, f for each model
    # Find extents, axes, mesh, sorted_pts_list of gridded parameters.
    akpr_grids = rebuild_grid(results_akpr.keys())
    akpr_cost_grid = np.asarray(
        [results_akpr[str(tuple(map(int, k)))]["posterior_probs"]["MAP "+strategy]
        for k in akpr_grids[-1]]).reshape(*akpr_grids[1])
    akpr_cost_grid *= -1  # Go back to cost to minimize
    shp1_grids = rebuild_grid(results_shp1.keys())
    shp1_cost_grid = np.asarray(
        [results_shp1[str(tuple(map(int, k)))]["posterior_probs"]["MAP "+strategy]
        for k in shp1_grids[-1]]).reshape(*shp1_grids[1])
    shp1_cost_grid *= -1

    # AKPR scores: 2D heatmaps of the best score with each k, m, f, for each k
    fig, axes = plt.subplots(1, akpr_grids[1][0])
    fig.set_size_inches(akpr_grids[1][0]*4, 4)
    axes = axes.flatten()
    for i in range(len(axes)):
        ks = akpr_grids[2][0][i]  # k_I axis
        # left-right=cols=f, bottom-top=rows=m
        img = axes[i].imshow(akpr_cost_grid[i], cmap="plasma_r",
                    vmin=akpr_cost_grid.min(), vmax=akpr_cost_grid.max(),
                    origin="lower",
                    extent=(akpr_grids[0][2][0]-0.5, akpr_grids[0][2][1]+0.5,
                            akpr_grids[0][1][0]-0.5, akpr_grids[0][1][1]+0.5))
        axes[i].set_title(r"$k_I = {}$".format(ks))
        axes[i].set(xlabel=r"$f$", ylabel=r"$m$")
        axes[i].set_xticks(akpr_grids[2][2])  # axes
        axes[i].set_yticks(akpr_grids[2][1])
        fig.colorbar(img, ax=axes[i], anchor=(0, 0.3),
        label="Cost (lower is better)", shrink=0.8)
    fig.tight_layout()
    if do_save:
        fig.savefig(os.path.join(fl_plots, "akpr_i_heatmap_cost_mcmc.pdf"),
                    transparent=True, bbox_inches="tight")
    if do_show:
        plt.show()
    plt.close()

    # SHP-1 scores: line plot vs m, comparing with AKPR SHP-1 too
    # Compute Akaike information = 2k - 2*ln(L)
    # ln(L) log of likelihood is -cost function. Lower is better.
    # number of parameters is 3 for SHP-1, 4 for AKPR SHP-1
    # Plus 1 for SHP-1 (m), 3 for AKPR (k, m, f)
    aic_akpr = 2*7.0 - 2*(-1*akpr_cost_grid)
    aic_shp1 = 2*4.0 - 2*(-1*shp1_cost_grid)

    fig, ax = plt.subplots()
    # Plot SHP-1 model first
    m_axis = shp1_grids[2][0]
    #models_colors = ["xkcd: jade", "xkcd: tangerine"]
    models_colors = ["#1fa774", "#ff9408"]
    ax.plot(m_axis, aic_shp1, label="SHP-1 model", mfc=models_colors[0],
        color=models_colors[0], lw=3.5, marker="s", ms=8, mec=models_colors[0],
        zorder=100)

    # Plot best condition AKPR SHP-1
    kf_grid = list(itertools.product(akpr_grids[2][0], akpr_grids[2][2]))
    m_axis = akpr_grids[2][1]
    #best_kf = (1, 2)  # Index of these k, f in the grid is 0 and 1, respectively
    #costs_vals = aic_akpr[0, :, 1]
    #best_kmf = (best_kf[0], m_axis[np.argmin(costs_vals)], best_kf[1])
    best_kmf = find_best_grid_point(results_akpr, strat=strategy)[0]
    best_kmf = string_to_tuple(best_kmf)
    best_kf = best_kmf[::2]
    # Index in the grid. k, m, f axes in akpr_grids[2]
    best_grid_idx = [list(akpr_grids[2][i]).index(best_kmf[i]) for i in range(len(best_kmf))]
    best_grid_idx[1] = slice(None)
    best_grid_idx = tuple(best_grid_idx)
    costs_vals = aic_akpr[best_grid_idx]
    ax.plot(m_axis, costs_vals, color=models_colors[1], lw=3.5, marker="o",
            label=r"AKPR, $k_I={}$, $f={}$".format(*best_kf), ms=8,
            mfc=models_colors[1], mec=models_colors[1],
            zorder=101)
    colormap = list(sns.light_palette(models_colors[1], n_colors=len(kf_grid)))
    colormap = colormap[1:]  # Drop original color
    markers = ["^", "x", ">", "*", "+", "v"]
    for kf in kf_grid:
        if kf == best_kf:
            continue
        kf_idx = [np.argmax(akpr_grids[2][0] == kf[0]),
                    np.argmax(akpr_grids[2][2] == kf[1])]

        costs_vals = aic_akpr[kf_idx[0], :, kf_idx[1]]
        col = colormap.pop()
        ax.plot(m_axis, costs_vals, lw=1.25, color=col,
            label=r"AKPR, $k_I={}$, $f={}$".format(*kf), ms=4,
            marker=markers.pop(), mfc=col, mec=col)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    lowlim = min(aic_akpr.min(), aic_shp1.min())
    ax.set_ylim([lowlim, aic_akpr.max()])  # Not the largest value
    ax.set_xticks(shp1_grids[2][0])
    ax.set_xticklabels(shp1_grids[2][0])
    ax.set(xlabel=r"Step $m$ controlling inhibition", ylabel=r"AIC (lower is better)")
    fig.tight_layout()
    if do_save:
        fig.savefig(os.path.join(fl_plots, "models_comparison_aic_mcmc_fits.pdf"),
                    transparent=True, bbox_inches="tight")
    if do_show:
        plt.show()
    plt.close()

    # Check model curves for single ligand of AKPR fit
    # Does it still have absolute discrimination?
    # cost_args: ["rates_others", "total_RI", "N", "tau_agonist"]

    # We changed best kmf to 1, 2, 2; uncomment to check 1, 2, 3 (true best by slight margin)
    #best_kmf = (1, 2, 3)

    pvec = results_akpr[str(best_kmf)]["param_estimates"]["MAP " + strategy]
    pvec = np.exp(np.asarray(pvec)*ln10)
    akpr_best_rates = [pvec[0]] + list(cost_args[0][0][:1]) + list(pvec[1:3]) + [best_kmf[0], pvec[3]]
    nmf = (cost_args[0][2],) + tuple(best_kmf[1:])
    res = check_model_output(steady_akpr_i_1ligand, akpr_best_rates, cost_args[0][1], nmf)
    l_range, tau_range, outputs = res
    fig, ax = plt.subplots()
    fig.set_size_inches(5.5, 4.0)
    for i, tau in enumerate(tau_range):
        ax.plot(l_range, outputs[i], label=r"$\tau = {:.0f}$ s".format(tau))
    ax.set(xscale="log", yscale="log", xlabel=r"$L$", ylabel=r"$C_N$", title="AKPR SHP-1 model")
    ax.legend(loc="lower right")
    ax.annotate(r"Best $k_I, m, f$ : $({}, {}, {})$".format(*best_kmf) + "\n"
                + r"Best $\varphi$ : " + "{:.3f}\n".format(pvec[0])
                + r"Best $C_{m, thresh}$ : " + "{:.2f}\n".format(pvec[1])
                + r"Best $S_{thresh}$ : " + "{:.2e}\n".format(pvec[2])
                + r"Best $\psi_{0,TCR}$: " + "{:.2e}\n".format(pvec[3]),
                xy=(0.05, 0.95), ha="left", va="top",
                xycoords="axes fraction")
    fig.tight_layout()
    if do_save:
        fig.savefig(os.path.join(fl_plots, "output_vs_L_akpr_i_mcmc_best_fit.pdf"),
                transparent=True, bbox_inches="tight")
    if do_show:
        plt.show()
    plt.close()

    return 0


### 6F TCR ANALYSIS ###
def compare_kmf_tcr_tcr_6f(fl_res, fl_plots, cost_args_6f, do_save=False, do_show=False):
    # Load results files
    with open(os.path.join(fl_res, "mcmc_analysis_tcr_tcr_6f.json"), "r") as h:
        results_6f = json.load(h)

    # Find best kmf or m, print results
    strategy = "best"
    bests_6f = find_best_grid_point(results_6f, strat=strategy)

    # Exponentiate back parameters
    bests_6f = list(bests_6f)
    bests_6f[1] = np.exp(bests_6f[1]*ln10)
    # Print results
    print("Best k, m, f for 6F TCRs:", bests_6f[0])
    print("With C_m_thresh, I_thresh =", bests_6f[1])
    print("Cost function:", -bests_6f[2])

    # Heatmap/plot of the best k or k, m, f for each model
    # Find extents, axes, mesh, sorted_pts_list of gridded parameters.
    akpr_grids = rebuild_grid(results_6f.keys())
    akpr_cost_grid = np.asarray(
        [results_6f[str(tuple(map(int, k)))]["posterior_probs"]["MAP "+strategy]
        for k in akpr_grids[-1]]).reshape(*akpr_grids[1])
    akpr_cost_grid *= -1  # Go back to cost to minimize

    # 2D heatmaps of the best score with each k, m, f, for each k
    fig, axes = plt.subplots(1, akpr_grids[1][0])
    fig.set_size_inches(akpr_grids[1][0]*4, 4)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    for i in range(len(axes)):
        ks = akpr_grids[2][0][i]  # k_I axis
        # left-right=cols=f, bottom-top=rows=m
        img = axes[i].imshow(akpr_cost_grid[i], cmap="plasma_r",
                    vmin=akpr_cost_grid.min(), vmax=akpr_cost_grid.max(),
                    origin="lower",
                    extent=(akpr_grids[0][2][0]-0.5, akpr_grids[0][2][1]+0.5,
                            akpr_grids[0][1][0]-0.5, akpr_grids[0][1][1]+0.5))
        axes[i].set_title(r"$k_I = {}$".format(ks))
        axes[i].set(xlabel=r"$f$", ylabel=r"$m$")
        axes[i].set_xticks(akpr_grids[2][2])  # axes
        axes[i].set_yticks(akpr_grids[2][1])
        fig.colorbar(img, ax=axes[i], anchor=(0, 0.3),
        label="Cost (lower is better)", shrink=0.8)
    fig.tight_layout()
    if do_save:
        fig.savefig(os.path.join(fl_plots, "tcr_tcr_6f_heatmap_cost_mcmc.pdf"),
                    transparent=True, bbox_inches="tight")
    if do_show:
        plt.show()
    plt.close()

    # Plotting model output as a function of L
    pvec = results_6f[str(bests_6f[0])]["param_estimates"]["MAP " + strategy]
    pvec = np.exp(np.asarray(pvec)*ln10)
    #kf_grid = list(itertools.product(akpr_grids[2][0], akpr_grids[2][2]))
    #m_axis = akpr_grids[2][1]
    #best_kf = (1, 2)  # Index of these k, f in the grid is 0 and 1, respectively
    #costs_vals = aic_akpr[0, :, 1]
    #best_kmf = (best_kf[0], m_axis[np.argmin(costs_vals)], best_kf[1])
    best_kmf = tuple(int(k) for k in bests_6f[0].strip("()").split(","))

    best_rates_6f = [pvec[0]] + list(cost_args_6f[0][:1]) + list(pvec[1:3]) + [best_kmf[0], pvec[3]]
    nmf = (cost_args_6f[2],) + tuple(best_kmf[1:])
    res = check_model_output(steady_akpr_i_1ligand, best_rates_6f, cost_args_6f[1], nmf)
    l_range, tau_range, outputs = res
    fig, ax = plt.subplots()
    fig.set_size_inches(5.5, 4.0)
    for i, tau in enumerate(tau_range):
        ax.plot(l_range, outputs[i], label=r"$\tau = {:.0f}$ s".format(tau))
    ax.set(xscale="log", yscale="log", xlabel=r"$L$", ylabel=r"$C_N$", title="AKPR SHP-1 model for 6F")
    ax.legend(loc="lower right")
    ax.annotate(r"Best $k_I, m, f$ : $({}, {}, {})$".format(*best_kmf) + "\n"
                + r"Best $\varphi$ : " + "{:.3f}\n".format(pvec[0])
                + r"Best $C_{m, thresh}$ : " + "{:.2f}\n".format(pvec[1])
                + r"Best $S_{thresh}$ : " + "{:.2e}\n".format(pvec[2])
                + r"Best $\psi_{0,TCR}$: " + "{:.2e}\n".format(pvec[3]),
                xy=(0.05, 0.95), ha="left", va="top",
                xycoords="axes fraction")
    fig.tight_layout()
    if do_save:
        fig.savefig(os.path.join(fl_plots, "output_vs_L_6f_mcmc_best_fit.pdf"),
                transparent=True, bbox_inches="tight")
    if do_show:
        plt.show()
    plt.close()


### Confidence intervals on the model fits ###
# For final plotting in the paper, sample from the MCMC distributions.
# Revision 1: use 95 % CI instead of 90 % CI
# Revision 2: propagate uncertainty on MHC levels: sample a MHC level
# for each MCMC parameter sample.
# This repeats a bit the data_model_comparison plots, but whatever.
# The main goal is to save these CIs to disk for plotting elsewhere
def main_tcr_tcr_confidence(
        fl_res, fl_for_plots, fnames_dict, molec_counts_fi, **kwargs
    ):
    """ Compute model confidence intervals for Francois 2013 model, revised
    AKPR model, and 6F TCR fit. Call once per model

    fl_res (str): path to the results folder
    fl_for_plots (str): path to the results/for_plots folder
    fnames_dict (dict): dictionary with 
    molec_counts_fi (str): name of the surface molecule summary counts

    kwargs:
        do_save
        do_show
        n_boot
        ci_res_fname
        plotting_data_fname
        fig_subfolder (str): folder where model fit plots will be saved
        tcell_type
        mtc
    """
    do_save = kwargs.get("do_save", False)
    do_show = kwargs.get("do_show", False)
    n_boot = kwargs.get("n_boot", 1000)
    # Arguments for molecule numbers CIs
    tcell_type = kwargs.get("tcell_type", "OT1_Naive")
    mtc = kwargs.get("mtc", "Geometric mean")

    # Name of the file that will contain the results of this main function
    ci_res_fname = kwargs.get("ci_res_fname",
        "model_confidence_intervals_tcr_tcr.h5")
    ci_res_file = os.path.join(fl_res, ci_res_fname)
    # Also, name of the file that will contain all plotting data
    plotting_data_fname = kwargs.get("plotting_data_fname",
                            "dfs_model_data_ci_mcmc_tcr_tcr.h5")
    all_model_dfs = {}
    all_data_dfs = {}
    all_ci_dfs = {}

    # One main SeedSequence will spawn one seedsequence per model
    # which will spawn enough seedsequences to cover all kmf of that model
    main_seedseq = kwargs.get("main_seedseq", None)
    if main_seedseq is None:
        main_seedseq = np.random.SeedSequence(
                entropy=0xe4b0e76433027d424189f4edea6f88ab,
                spawn_key=(0x6af3ffba3469cf4ff8381cab2a3de3f6,)
        )
    models_list = kwargs.get("models_list", ["akpr_i", "shp1", "tcr_tcr_6f"])
    model_seedseqs = main_seedseq.spawn(len(models_list))

    for model_suffix in models_list:
        seedseq = model_seedseqs.pop()
        # Model-specific arguments
        mod = "6f" if model_suffix == "tcr_tcr_6f" else model_suffix
        analysis_res_fname = fnames_dict[mod].get("analysis_res_fname")
        samples_fname = fnames_dict[mod].get("results_fname")
        if model_suffix == "tcr_tcr_6f":
            data_prep_fct = prepare_data_6f
        else:
            data_prep_fct = prepare_data

        if model_suffix == "shp1":
            panel_fct = antag_ratio_panel_shp1
        else:
            panel_fct = antag_ratio_panel_akpr_i

        # Import MCMC results
        results_tcr_tcr = h5py.File(os.path.join(fl_res, samples_fname), "r")
        samples_group = results_tcr_tcr.get("samples")
        data_group = results_tcr_tcr.get("data")

        # Prepare all data: keep ITAM numbers levels and TCR Antigen Density
        data_file_name = data_group.attrs.get("data_file_name")  # full path
        l_conc_mm_params = data_group.get("l_conc_mm_params")[()]

        tau_file = os.path.join("..", "data", "pep_tau_map_ot1.json")
        with open(tau_file, "r") as handle:
            pep_tau_map_ot1 = json.load(handle)
        df = pd.read_hdf(data_file_name)
        df_fit, df_ci_log2, tau_agonist = data_prep_fct(df,
                                        l_conc_mm_params, pep_tau_map_ot1)

        # Load standard deviation of the mean estimators of TCR, MHC, and
        # loading EC50 parameters
        molec_loads = load_tcr_tcr_molec_numbers_ci(
            molec_counts_fi, mtc, tcell_type=tcell_type
        )
        _, _, _, tcr_num_std, mm_params_std, all_n_dofs = molec_loads

        # Load analysis results
        try:
            jfile = open(os.path.join(fl_res, analysis_res_fname), "r")
            all_results_dicts = json.load(jfile)
            jfile.close()
        except FileNotFoundError:
            raise FileNotFoundError("Run main_mcmc_analysis first.")

        # Launching CI calculations in parallel
        print("Starting to generate {} CI samples...".format(model_suffix))
        start_t = perf_counter()
        # Load constant model parameters
        cost_args_loaded = [data_group.get(a)[()]
                            for a in data_group.attrs.get("cost_args_names")]

        # Launch calculation of predictions
        pool = multiprocessing.Pool(min(n_cpu, len(all_results_dicts)))
        all_processes = {}
        seeds = seedseq.spawn(len(all_results_dicts))
        for cond in all_results_dicts.keys():
            # Graphs of model prediction with confidence intervals from samples
            grid_point = samples_group.get(cond).attrs.get("run_id")
            p_samp = samples_group.get(cond)[()]
            p_best = np.asarray(all_results_dicts[cond]
                            .get("param_estimates")["MAP best"])
            # Compute model ratios
            res = pool.apply_async(
                    confidence_model_antagonism_tcr,
                    args=(panel_fct, p_samp, p_best,
                            grid_point, df_fit, df_ci_log2),
                    kwds=dict(other_args=cost_args_loaded, n_taus=200,
                            n_samp=n_boot, seed=seeds.pop(),
                            antagonist_lvl="Antagonist",
                            l_conc_mm_params=l_conc_mm_params,
                            tcr_num_std=tcr_num_std,
                            mm_params_std=mm_params_std,
                            dofs_tcr_mhc_kd=all_n_dofs)
                )
            all_processes[cond] = res

        # Collect model predictions for each kmf into one DataFrame
        df_fit_model = pd.concat({k:a.get() for k, a in all_processes.items()},
                        names=["kmf"]).sort_index().sort_index(axis=1)
        df_fit_model.to_hdf(ci_res_file, key=model_suffix)
        pool.close()
        delta_t = perf_counter() - start_t
        print("Total time to generate {} CI: {} s".format(model_suffix, delta_t))

        # Make plots of model fits with CIs
        fig_folder = os.path.join("..", "figures", 
                                  fnames_dict[mod].get("fig_subfolder"))
        for cond in all_results_dicts.keys():
            cond_nice = (str(cond).replace(" ", "-").replace(")", "")
                        .replace("(", "").replace(",", ""))
            # Posterior probability
            posts_cond = all_results_dicts[cond].get("posterior_probs")

            percentiles = [a for a in df_fit_model.columns 
                           if a.startswith("percentile")]
            fig, _ = plot_fit_antagonism(df_fit, df_fit_model.loc[cond, "best"],
                l_conc_mm_params, df_ci_log2, cost=posts_cond.get("MAP best"), 
                model_ci=df_fit_model.loc[cond, percentiles])
            if do_save:
                fig.savefig(os.path.join(fig_folder,
                        "data_model_comparison_{}.pdf".format(cond_nice)),
                        bbox_inches="tight", transparent=True)
            if do_show:
                plt.show()
            plt.close()
        
        # Reformat further for final plotting elsewhere
        # Rename concentrations to uM, nM for plotting convenience
        def reverse_mm_fitted(x):
            return inverse_michaelis_menten(x, *l_conc_mm_params)
        with open(os.path.join("..", "data", "reference_pep_tau_maps.json"), "r") as h:
            pep_tau_refs = json.load(h)
        def reverse_tau_to_ec50(x):
            return convert_tau_ec50_relative(x, pep_tau_refs["N4"],
                                            npow=pep_tau_refs["npow"])
        def renamer(d):
            # Add a level for EC50 so we have the choice when plotting
            new_col = reverse_tau_to_ec50(d.index.get_level_values("Antagonist").values)
            new_idx = pd.MultiIndex.from_tuples(
                        [tuple(d.index[i]) + (new_col[i],) for i in range(len(new_col))],
                        names=list(d.index.names) + ["Antagonist_EC50"])
            return (d.set_axis(new_idx)
                     .rename(reverse_mm_fitted, axis=0, level="AgonistConcentration")
                     .rename(write_conc_uM, axis=0, level="AgonistConcentration")
                     .rename(reverse_mm_fitted, axis=0, level="AntagonistConcentration")
                     .rename(write_conc_uM, axis=0, level="AntagonistConcentration")
                    )

        df_fit_model = renamer(df_fit_model)
        df_fit = renamer(df_fit)
        df_ci_log2 = renamer(df_ci_log2)
        all_model_dfs[model_suffix] = df_fit_model.copy()
        all_data_dfs[model_suffix] = df_fit.copy()
        all_ci_dfs[model_suffix] = df_ci_log2.copy()

    ### Join all predictions into one model dataframe, same for data and errors
    dfs_model = pd.concat(all_model_dfs, names=["Model"])
    dfs_data = pd.concat(all_data_dfs, names=["Model"])
    dfs_ci = pd.concat(all_ci_dfs, names=["Model"])

    # Save all final plotting data to disk for final figures production
    dfs_data.name = "Antagonism ratio"

    plot_res_file = os.path.join(fl_for_plots, plotting_data_fname)
    dfs_model.to_hdf(plot_res_file, key="model")
    dfs_data.to_hdf(plot_res_file, key="data")
    dfs_ci.to_hdf(plot_res_file, key="ci")
    print("Finished saving")

    return 0


if __name__ == "__main__":
    # We provide file names and paths separately to analysis scripts. 
    # This is because there are many file names in the same results folder. 
    # However, data files are always provided as complete paths. 
    analysis_kwarguments = {
        "akpr_i": {
            "analysis_res_fname": "mcmc_analysis_akpr_i.json",
            "fit_summary_fname": "fit_summary_akpr_i.json",
            "results_fname": "mcmc_results_akpr_i.h5",
            "fig_subfolder": "mcmc_akpr_i",
            "model": "akpr_i"
        },
        "shp1": dict(
            analysis_res_fname = "mcmc_analysis_shp1.json",
            fit_summary_fname = "fit_summary_shp1.json",
            results_fname = "mcmc_results_shp1.h5",
            fig_subfolder = "mcmc_shp1",
            model = "shp1"
        ),
        "6f": dict(
            analysis_res_fname = "mcmc_analysis_tcr_tcr_6f.json",
            fit_summary_fname = "fit_summary_tcr_tcr_6f.json",
            results_fname = "mcmc_results_tcr_tcr_6f.h5",
            fig_subfolder = "mcmc_tcr_tcr_6f",
            model = "6f"
        )
    }
    folder_results = os.path.join("..", "results", "mcmc")
    folder_results_for_plots = os.path.join("..", "results", "for_plots")
    # Results file for model CI
    ci_res_filename = "model_confidence_intervals_tcr_tcr.h5"
    ci_res_filepath = os.path.join(folder_results, ci_res_filename)

    # Multiprocess
    all_processes = {"akpr_i":None, "shp1":None, "6f":None}
    skip_analysis = False
    # Check that files do not already exist to avoid overwriting results
    for mod in all_processes.keys():
        analysis_filename = analysis_kwarguments[mod]["analysis_res_fname"]
        analysis_filepath = os.path.join(folder_results, analysis_filename)
        if os.path.isfile(analysis_filepath):
            if os.path.isfile(ci_res_filepath):
                raise RuntimeError("Existing model analysis file found at "
                    + str(analysis_filepath) + " and a model CI file at"
                    + str(ci_res_filepath) + " . If you want to re-run, "
                    + "delete these existing files first. ")
             # This is being re-run just to get model CI; skip analysis
            else: 
                skip_analysis = True
    
    ### Launch main analysis
    if not skip_analysis:
        print("Starting MCMC results analysis...")
        pool = multiprocessing.Pool(min(n_cpu, len(all_processes)))
        for mod in all_processes.keys():
            res = pool.apply_async(
                    main_mcmc_analysis,
                    args=(folder_results, analysis_kwarguments[mod]),
                    kwds=dict(do_save=True)
                )
            all_processes[mod] = res
        # There is no return but need to call get() to wait for the end
        for mod in all_processes.keys():
            all_processes[mod].get()
        pool.close()
    else:
        print("...\nUsing existing analysis files, "
              + "not generating analysis plots, "
              + "and skipping to model CI generation. \n...\n"
        )

    
    ### Model comparison, SHP-1 vs 6F
    folder_graphs = os.path.join("..", "figures", "model_comparison")
    cost_args = []
    # Get other cost function arguments.
    for fname in ["mcmc_results_akpr_i.h5", "mcmc_results_shp1.h5"]:
        f = os.path.join(folder_results, fname)
        samples_file = h5py.File(f, "r")
        cost_args_names = samples_file.get("data").attrs.get("cost_args_names")
        # ["rates_others", "total_RI", "N", "tau_agonist"]
        cost_args.append([samples_file.get("data").get(a)[()] for a in cost_args_names])
    
    main_compare_models_figures(folder_results, folder_graphs, cost_args,
                                do_save=True, do_show=False)

    
    ### Choosing right k, m, f for 6F T cells
    f = os.path.join(folder_results, "mcmc_results_tcr_tcr_6f.h5")
    samples_file = h5py.File(f, "r")
    cost_args_names = samples_file.get("data").attrs.get("cost_args_names")
    # ["rates_others", "total_RI", "N", "tau_agonist"]
    cost_args_6f = [samples_file.get("data").get(a)[()] for a in cost_args_names]
    
    compare_kmf_tcr_tcr_6f(folder_results, folder_graphs, cost_args_6f,
                            do_save=True, do_show=False)

    ### Confidence intervals simulations
    # Do not run if the results file already exists, 
    # to avoid deleting good results. 
    if os.path.isfile(ci_res_filepath):
        raise RuntimeError("Existing model confidence intervals file found at "
            + str(ci_res_filepath) + ". If you want to re-run, "
            + "delete this existing file first. ")
    molec_counts_fi = os.path.join("..", "data", "surface_counts", 
                                   "surface_molecule_summary_stats.h5")
    mtc = "Geometric mean"
    
    main_tcr_tcr_confidence(folder_results, folder_results_for_plots,
        analysis_kwarguments, molec_counts_fi=molec_counts_fi,
        do_save=True, n_boot=1000, mtc=mtc, tcell_type="OT1_Naive", 
        ci_res_fname=ci_res_filename
    )
    # kwargs do_show, plotting_data_fname, main_seedseq
    # kept to default values
    if skip_analysis:
        print("Note: MCMC analysis was skipped before model CI calculation "
              + "because existing results/mcmc/analysis*.json were found. "
              + "Delete these files to reanalyze or analyze new simulations. ")
