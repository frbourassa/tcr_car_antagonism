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
    cost_antagonism_shp1
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
    check_fit_model_antagonism,
    check_model_output,
    assemble_kf_vals,
    confidence_model_antagonism
)
from utils.preprocess import (
    string_to_tuple,
    loglog_michaelis_menten,
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
def main_mcmc_analysis(fl, model, do_save=False, do_show=False):
    """ Choose among 3 options for model:
        "akpr_i": TCR/TCR antagonism on 6Y T cells with updated model
        "shp1": TCR/TCR antagonism on 6Y T cells with Francois 2013 model
        "6f": TCR/TCR on 6F T cells with updated model
    """
    if model == "akpr_i":
        analysis_res_fname = "mcmc_analysis_akpr_i_test.json"
        fit_summary_fname = "fit_summary_akpr_i_test.json"
        results = h5py.File(os.path.join(fl, "mcmc_results_akpr_i.h5"), "r")
        fig_subfolder = "mcmc_akpr_i"
        data_prep_fct = prepare_data
        cost_fct = cost_antagonism_akpr_i
        panel_fct = antag_ratio_panel_akpr_i
    elif model == "shp1":
        analysis_res_fname = "mcmc_analysis_shp1_test.json"
        fit_summary_fname = "fit_summary_shp1_test.json"
        results = h5py.File(fl+"mcmc_results_shp1.h5", "r")
        fig_subfolder = "mcmc_shp1"
        data_prep_fct = prepare_data
        cost_fct = cost_antagonism_shp1
        panel_fct = antag_ratio_panel_shp1
    elif model == "6f":
        analysis_res_fname = "mcmc_analysis_tcr_tcr_6f_test.json"
        fit_summary_fname = "fit_summary_tcr_tcr_6f_test.json"
        # Import MCMC results
        results = h5py.File(fl+"mcmc_results_tcr_tcr_6f.h5", "r")
        fig_subfolder = "mcmc_tcr_tcr_6f"
        data_prep_fct = prepare_data_6f
        cost_fct = cost_antagonism_akpr_i
        panel_fct = antag_ratio_panel_akpr_i
    else:
        raise ValueError("Model {} unknown;".format(model)
                        + "choose among 'akpr_i', 'shp1', '6f'")

    # Import MCMC results
    samples_group = results.get("samples")
    cost_group = results.get("cost")
    data_group = results.get("data")

    # Import data
    data_file_name = data_group.attrs.get("data_file_name")
    l_conc_mm_params = data_group.get("l_conc_mm_params")[()]
    df = pd.read_hdf(data_file_name)
    with open("../data/pep_tau_map_ot1.json", "r") as handle:
        pep_tau_map_ot1 = json.load(handle)
    df_fit, df_ci_log2, tau_agonist = data_prep_fct(df,
                                    l_conc_mm_params, pep_tau_map_ot1)

    # Analyze each run
    if os.path.exists(os.path.join(fl, analysis_res_fname)):
        jfile = open(os.path.join(fl, analysis_res_fname), "r")
        all_results_dicts = json.load(jfile)
        jfile.close()

    else:
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
    fit_summ = fit_summary_to_json(
        data_group, samples_group, all_results_dicts, meth="MAP best"
    )
    with open(os.path.join("..","results","mcmc",fit_summary_fname), "w") as h:
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

        # Compute model ratios
        p_est = all_results_dicts[cond].get("param_estimates")
        dfs_model = {}
        for strat in p_est:
            dfs_model[strat] = check_fit_model_antagonism(
                panel_fct, np.asarray(p_est.get(strat)), grid_point,
                df_fit, df_ci_log2, other_args=cost_args_loaded, n_taus=101
            )
        dfs_model = pd.concat(dfs_model, names=["Estimate"])
        cond_nice = (str(cond).replace(" ", "-").replace(")", "")
                        .replace("(", "").replace(",", ""))
        posts_cond = all_results_dicts[cond].get("posterior_probs")
        fig, _ = plot_fit_antagonism(df_fit, dfs_model.loc["MAP best"],
                l_conc_mm_params, df_ci_log2, cost=posts_cond.get("MAP best"))
        if do_save:
            fig.savefig(os.path.join(figures_folder,
                        "data_model_comparison_{}.pdf".format(cond_nice)),
                        bbox_inches="tight", transparent=True)
        if do_show:
            plt.show()
        plt.close()

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
    print("With C_m_thresh, S_thresh =", bests_akpr[1])
    print("Cost function:", -bests_akpr[2])

    print("Best SHP-1 m:", bests_shp1[0])
    print("With C_m_thresh, S_tot =", bests_shp1[1])
    print("Cost function:", -bests_shp1[2])

    # Heatmap/plot of the best k or k, m, f for each model
    # Find extents, axes, mesh, sorted_pts_list of gridded parameters.
    akpr_grids = rebuild_grid(results_akpr.keys())
    akpr_cost_grid = np.asarray(
        [results_akpr[str(tuple(k))]["posterior_probs"]["MAP "+strategy]
        for k in akpr_grids[-1]]).reshape(*akpr_grids[1])
    akpr_cost_grid *= -1  # Go back to cost to minimize
    shp1_grids = rebuild_grid(results_shp1.keys())
    shp1_cost_grid = np.asarray(
        [results_shp1[str(tuple(k))]["posterior_probs"]["MAP "+strategy]
        for k in shp1_grids[-1]]).reshape(*shp1_grids[1])
    shp1_cost_grid *= -1

    # AKPR scores: 2D heatmaps of the best score with each k, m, f, for each k
    fig, axes = plt.subplots(1, akpr_grids[1][0])
    fig.set_size_inches(akpr_grids[1][0]*4, 4)
    axes = axes.flatten()
    for i in range(len(axes)):
        ks = akpr_grids[2][0][i]  # k_S axis
        # left-right=cols=f, bottom-top=rows=m
        img = axes[i].imshow(akpr_cost_grid[i], cmap="plasma_r",
                    vmin=akpr_cost_grid.min(), vmax=akpr_cost_grid.max(),
                    origin="lower",
                    extent=(akpr_grids[0][2][0]-0.5, akpr_grids[0][2][1]+0.5,
                            akpr_grids[0][1][0]-0.5, akpr_grids[0][1][1]+0.5))
        axes[i].set_title(r"$k_S = {}$".format(ks))
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
    # ln(L) log of likelihood is -cost function.
    # number of parameters is 2 for SHP-1, 2 for AKPR SHP-1
    # Lower is better.
    # Also, I'm not counting k, m, f as parameters because they are gridded
    # over, not fitted. They describe the structure of the model,
    # they're not really parameters that one could over-fit.
    aic_akpr = 2*3.0 - 2*(-1*akpr_cost_grid)
    aic_shp1 = 2*2.0 - 2*(-1*shp1_cost_grid)

    fig, ax = plt.subplots()
    # Plot SHP-1 model first
    m_axis = shp1_grids[2][0]
    #models_colors = ["xkcd: jade", "xkcd: tangerine"]
    models_colors = ["#1fa774", "#ff9408"]
    ax.plot(m_axis, aic_shp1, label="SHP-1 model", mfc=models_colors[0],
        color=models_colors[0], lw=3.5, marker="s", ms=8, mec=models_colors[0],
        zorder=100)

    # Plot best condition AKPR SHP-1
    # TODO: add colormap
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
            label=r"AKPR, $k_S={}$, $f={}$".format(*best_kf), ms=8,
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
            label=r"AKPR, $k_S={}$, $f={}$".format(*kf), ms=4,
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
    ax.annotate(r"Best $k_S, m, f$ : $({}, {}, {})$".format(*best_kmf) + "\n"
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
    with open(os.path.join(fl_res, "mcmc_analysis_tcr_tcr_6f_test.json"), "r") as h:
        results_6f = json.load(h)

    # Find best kmf or m, print results
    strategy = "best"
    bests_6f = find_best_grid_point(results_6f, strat=strategy)

    # Exponentiate back parameters
    bests_6f = list(bests_6f)
    bests_6f[1] = np.exp(bests_6f[1]*ln10)
    # Print results
    print("Best k, m, f for 6F TCRs:", bests_6f[0])
    print("With C_m_thresh, S_thresh =", bests_6f[1])
    print("Cost function:", -bests_6f[2])

    # Heatmap/plot of the best k or k, m, f for each model
    # Find extents, axes, mesh, sorted_pts_list of gridded parameters.
    akpr_grids = rebuild_grid(results_6f.keys())
    akpr_cost_grid = np.asarray(
        [results_6f[str(tuple(k))]["posterior_probs"]["MAP "+strategy]
        for k in akpr_grids[-1]]).reshape(*akpr_grids[1])
    akpr_cost_grid *= -1  # Go back to cost to minimize

    # 2D heatmaps of the best score with each k, m, f, for each k
    fig, axes = plt.subplots(1, akpr_grids[1][0])
    fig.set_size_inches(akpr_grids[1][0]*4, 4)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    for i in range(len(axes)):
        ks = akpr_grids[2][0][i]  # k_S axis
        # left-right=cols=f, bottom-top=rows=m
        img = axes[i].imshow(akpr_cost_grid[i], cmap="plasma_r",
                    vmin=akpr_cost_grid.min(), vmax=akpr_cost_grid.max(),
                    origin="lower",
                    extent=(akpr_grids[0][2][0]-0.5, akpr_grids[0][2][1]+0.5,
                            akpr_grids[0][1][0]-0.5, akpr_grids[0][1][1]+0.5))
        axes[i].set_title(r"$k_S = {}$".format(ks))
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
    ax.annotate(r"Best $k_S, m, f$ : $({}, {}, {})$".format(*best_kmf) + "\n"
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
# This repeats a bit the data_model_comparison plots, but whatever.
# The main goal is to save these CIs to disk for plotting elsewhere
def main_tcr_tcr_confidence(fl_res, fl_for_plots, do_save=False,
                        do_show=False, n_boot=1000):
    """ Compute model confidence intervals for Francois 2013 model, revised
    AKPR model, and 6F TCR fit. Call once per model
    """
    if not (do_save or do_show):  # No plots
        return 0
    # Name of the file that will contain the results of this main function
    ci_res_fname = "model_confidence_intervals_tcr_tcr_test.h5"
    # Also, name of the file that will contain all plotting data
    plotting_data_fname = "dfs_model_data_ci_mcmc_tcr_tcr_test.h5"
    all_model_dfs = {}
    all_data_dfs = {}
    all_ci_dfs = {}

    # One main SeedSequence will spawn one seedsequence per model
    # which will spawn enough seedsequences to cover all kmf of that model
    main_seedseq = np.random.SeedSequence(
                entropy=0xe4b0e76433027d424189f4edea6f88ab,
                spawn_key=(0x6af3ffba3469cf4ff8381cab2a3de3f6,)
                )
    models_list = ["akpr_i", "shp1", "tcr_tcr_6f"]
    model_seedseqs = main_seedseq.spawn(len(models_list))

    for model_suffix in models_list:
        seedseq = model_seedseqs.pop()
        # Model-specific arguments
        analysis_res_fname = "mcmc_analysis_{}_test.json".format(model_suffix)
        samples_fname = "mcmc_results_{}_test.h5".format(model_suffix)
        if model_suffix == "tcr_tcr_6f":
            data_prep_fct = prepare_data_6f
        else:
            data_prep_fct = prepare_data

        if model_suffix == "shp1":
            panel_fct = antag_ratio_panel_shp1
        else:
            panel_fct = antag_ratio_panel_akpr_i

        # Import MCMC results
        results_tcr_tcr = h5py.File(fl_res + samples_fname, "r")
        samples_group = results_tcr_tcr.get("samples")
        cost_group = results_tcr_tcr.get("cost")
        data_group = results_tcr_tcr.get("data")

        # Prepare all data: keep ITAM numbers levels and TCR Antigen Density
        data_file_name = data_group.attrs.get("data_file_name")
        l_conc_mm_params = data_group.get("l_conc_mm_params")[()]
        with open("../data/pep_tau_map_ot1.json", "r") as handle:
            pep_tau_map_ot1 = json.load(handle)
        df = pd.read_hdf(data_file_name)
        df_fit, df_ci_log2, tau_agonist = data_prep_fct(df,
                                        l_conc_mm_params, pep_tau_map_ot1)

        # Load analysis results
        try:
            jfile = open(os.path.join(fl_res, analysis_res_fname), "r")
            all_results_dicts = json.load(jfile)
            jfile.close()
        except FileNotFoundError:
            raise FileNotFoundError("Run main_mcmc_analysis first.")

        # Launching CI calculations in parallel
        try:
            df_fit_model = pd.read_hdf(
                        os.path.join(fl_res, ci_res_fname), key=model_suffix)
            print("Loaded existing model predictions file")
        except (FileNotFoundError, KeyError):
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
                        confidence_model_antagonism,
                        args=(panel_fct, p_samp, p_best,
                                grid_point, df_fit, df_ci_log2),
                        kwds=dict(other_args=cost_args_loaded, n_taus=200,
                                n_samp=n_boot, seed=seeds.pop(),
                                antagonist_lvl="Antagonist")
                    )
                all_processes[cond] = res

            # Collect model predictions for each kmf into one DataFrame
            df_fit_model = pd.concat({k:a.get() for k, a in all_processes.items()},
                            names=["kmf"])
            df_fit_model.to_hdf(os.path.join(fl_res, ci_res_fname), key=model_suffix)
            pool.close()
            delta_t = perf_counter() - start_t
            print("Total time to generate {} CI: {} s".format(model_suffix, delta_t))

        # Reformat further for plotting
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

    # Don't plot here
    # Save all final plotting data to disk for final figures production
    dfs_data.name = "Antagonism ratio"

    plot_res_file = os.path.join(fl_for_plots, plotting_data_fname)
    dfs_model.to_hdf(plot_res_file, key="model")
    dfs_data.to_hdf(plot_res_file, key="data")
    dfs_ci.to_hdf(plot_res_file, key="ci")

    return 0


if __name__ == "__main__":
    folder_results = "../results/mcmc/"
    folder_results_for_plots = "../results/for_plots/"
    # Multiprocess
    all_processes = {"akpr_i":None, "shp1":None, "6f":None}
    pool = multiprocessing.Pool(min(n_cpu, len(all_processes)))
    for mod in all_processes.keys():
        res = res = pool.apply_async(
                main_mcmc_analysis,
                args=(folder_results, mod),
                kwds=dict(do_save=False)
            )
        all_processes[mod] = res
    # There is no return but need to call get() to wait for the end
    for mod in all_processes.keys():
        all_processes[mod].get()
    pool.close()

    # Model comparison, SHP-1 vs 6F
    folder_graphs = "../figures/model_comparison/"
    cost_args = []
    # Get other cost function arguments.
    for f in [os.path.join(folder_results, "mcmc_results_akpr_i.h5"),
                os.path.join(folder_results, "mcmc_results_shp1.h5")]:
        samples_file = h5py.File(f, "r")
        cost_args_names = samples_file.get("data").attrs.get("cost_args_names")
        # ["rates_others", "total_RI", "N", "tau_agonist"]
        cost_args.append([samples_file.get("data").get(a)[()] for a in cost_args_names])
    main_compare_models_figures(folder_results, folder_graphs, cost_args,
                                do_save=False, do_show=False)

    # Choosing right k, m, f for 6F T cells
    f = os.path.join(folder_results, "mcmc_results_tcr_tcr_6f.h5")
    samples_file = h5py.File(f, "r")
    cost_args_names = samples_file.get("data").attrs.get("cost_args_names")
    # ["rates_others", "total_RI", "N", "tau_agonist"]
    cost_args_6f = [samples_file.get("data").get(a)[()] for a in cost_args_names]
    compare_kmf_tcr_tcr_6f(folder_results, folder_graphs, cost_args_6f,
                            do_save=False, do_show=False)

    main_tcr_tcr_confidence(folder_results, folder_results_for_plots,
                            do_save=False, n_boot=1000)
