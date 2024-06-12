"""
General analysis scripts for MCMC runs. Just based on sample distributions
and other stuff included in return files of MCMC runs.

@author: frbourassa
November 2022
"""
import numpy as np
import pandas as pd
import emcee
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Local modules
from mcmc.convergence import (check_autocorr_convergence, rel_error,
                                    check_acceptance_fractions)
from mcmc.estimation import find_max_posterior, find_confidence_interval
from mcmc.plotting import hexpair
from utils.preprocess import string_to_tuple


### UTILITY FUNCTIONS ###


def rebuild_grid(keys):
    """ Find grid bounds and grid axes based on a list of string grid points.
    Also place the points into a grid.
    Uses ij indexing, less confusing in general dimension.
    """
    pts_list = [string_to_tuple(s) for s in keys]  # Might not be sorted
    bounds = [(min(a), max(a)) for a in zip(*pts_list)]
    extents = [a[1]-a[0]+1 for a in bounds]
    axes = [np.arange(a, b+1, dtype="int") for a, b in bounds]
    mesh = np.meshgrid(*axes, indexing="ij")
    mesh = np.moveaxis(np.stack(mesh, 0), 0, -1)  # n-d array of tuples
    # mesh is a hypercube where each point is the coordinate tuple (last dim).
    sorted_pts_list = mesh.reshape(-1, len(bounds))  # flattened
    return bounds, extents, axes, mesh, sorted_pts_list


# Utility function to find best grid point given analysis results
def find_best_grid_point(all_res_dict, strat="best"):
    stratkey = "MAP " + strat
    conditions = list(all_res_dict.keys())
    best_cond = conditions[0]
    best_cost = all_res_dict[best_cond]["posterior_probs"][stratkey]
    best_p = all_res_dict[best_cond]["param_estimates"][stratkey]
    for cond in conditions:
        cost = all_res_dict[cond]["posterior_probs"][stratkey]
        if cost > best_cost:
            best_cost = cost
            best_cond = cond
            best_p = all_res_dict[cond]["param_estimates"][stratkey]

    return best_cond, np.asarray(best_p), best_cost


def read_metadata(meta):
    """ Returns
    nsamples, nwalkers, parambounds, n_pm, param_names, gridbounds, params0
    """
    data = [None]*7
    data[0] = meta["n_samples"]
    data[1] = meta["n_walkers"]
    data[2] = meta["param_bounds"]
    data[3] = len(data[2][0])
    data[4] = meta.get("param_names",
                        ["p_{}".format(i) for i in range(data[3])])
    data[5] = meta["grid_bounds"]
    data[6] = meta["p0"]  # Starting point, if any
    return data


### PARAMETER SAMPLING ANALYSIS ###
# General, not model-specific
def analyze_mcmc_run(cond, samples, costs, metadata, cost_fct,
                                    cost_args=(), cost_kwargs={}):
    # Find parameter estimates
    # Deal with keyword arguments
    cond_nice = str(cond)#.replace("&", "-")

    # Extract metadata
    (nsamples, nwalkers, parambounds, n_pm,
            param_names, gridbounds, params0) = read_metadata(metadata)
    accept_frac = samples.attrs.get("acceptance_fraction", np.nan)

    # Extract the arrays, drop metadata of that dataset
    samples = samples[()]
    costs = costs[()]

    # Dictionary to add to JSON file in an entry labeled cond
    results_dict = {}
    print()
    print("*** STARTING ANALYSIS OF CONDITION {} ***".format(cond_nice))

    ## Check acceptance fraction
    check_acceptance_fractions(accept_frac, cond_nice, lo=0.05, hi=0.95)
    # Min, max, median acceptance fractions
    results_dict["acceptance_fractions"] = [
        np.min(accept_frac),
        np.median(accept_frac),
        np.amax(accept_frac)
    ]

    ## Check autocorrelation of each variable, compare to chain duration
    taus_corr = check_autocorr_convergence(samples, param_names, cond_nice)
    total_corr_periods = [nsamples / a for a in taus_corr]
    burn_in_frac =  min(20.0 / min(total_corr_periods), 0.5)
    burn_in_steps = int(nsamples * burn_in_frac)

    results_dict["taus_corr"] = list(taus_corr)
    results_dict["total_corr_periods"] = list(total_corr_periods)
    results_dict["burn_in_frac"] = burn_in_frac
    results_dict["burn_in_steps"] = burn_in_steps

    ## Find maximum a posteriori estimates.
    # Reshape results so all walkers are contiguous in a flattened last dim.
    samples2 = np.moveaxis(samples[:, :, burn_in_steps:], 1, 2).reshape(n_pm, -1)
    costs2 = costs[:, burn_in_steps:].T.flatten()
    param_estimates, param_costs = {}, {}
    # Just use the best sample found, that's all we care about, the best fit.
    # The hist method is bad: marginal modes are not multidimensional modes
    for strat in ["hist", "best"]:
        # Burn-in already removed from samples2 and costs2
        map_strat, post_strat = find_max_posterior(samples2, costs2,
                                    strat=strat, burn=0.0)
        param_estimates["MAP " + strat] = list(map_strat)
        param_costs["MAP " + strat] = post_strat

    # Store parameter estimates in the results dictionary
    results_dict["param_estimates"] = param_estimates

    # Compute log-posterior probability of each estimate
    # and compare to recorded value during the MCMC run
    # to make sure there is no significant discrepancy (> 1 %)
    # There will be with the hist method because we use
    # the log-prob of the nearest simulation point
    param_costs_full = {}
    for k in param_estimates:
        cost_recomp = cost_fct(np.asarray(param_estimates[k]), parambounds,
                                *cost_args, **cost_kwargs)
        cost_discrep = rel_error(cost_recomp, param_costs[k], 1, k)
        param_costs_full[k] = cost_recomp

    # Store log-posterior probabilities in the results dictionary
    results_dict["posterior_probs"] = param_costs_full
    del param_costs

    ## Compute confidence intervals (use quantiles on marginal distributions)
    # Although that doesn't mean much when parameters are correlated
    params_ci = {}
    for i in range(n_pm):
        ci = find_confidence_interval(samples2[i],
                                lower=0.05, upper=0.95, burn=0.0)
        params_ci[param_names[i]] = list(ci)
    results_dict["confidence_intervals"] = params_ci

    return results_dict

## Fit summary
# Easily readable file with all model parameters and fit results
def fit_summary_to_json(d_gp, s_gp, all_results_dicts, meth="MAP best", nkeep=2):
    # Global parameters first
    summary = {}
    summary["common_parameters"] = {
        "fit_param_names": list(s_gp.attrs.get("param_names")[()]),
        #"fit_bounds": list(list(a) for a in s_gp.attrs.get("param_bounds"))
    }
    # Cost arguments
    cost_args_loaded = {}
    for a in d_gp.keys():
        arg = d_gp.get(a)[()]
        if isinstance(arg, np.ndarray):
            if arg.dtype == np.int64:
                arg = list(map(int, arg))
            else:
                arg = list(arg)
        elif isinstance(arg, np.int64):
            arg = int(arg)
        elif isinstance(arg, np.float64):
            arg = float(arg)
        cost_args_loaded[a] = arg
    summary["common_parameters"].update(cost_args_loaded)

    # Keep only the best two fits, no need for bad ones in this simple summary
    all_pvecs = {}
    for k in all_results_dicts.keys():
        all_pvecs[k] = {
            "pvec": [10.0**a for a in all_results_dicts[k]["param_estimates"][meth]],
            "cost": all_results_dicts[k]["posterior_probs"][meth]
        }
    sorted_grid = sorted(list(all_pvecs.keys()), reverse=True,
                    key=lambda x: all_pvecs[x].get("cost"))

    for k in sorted_grid[:nkeep]:
        summary[k] = all_pvecs[k]

    return summary



### PLOTTING FUNCTIONS ###

def graph_mcmc_samples(cond, samples, costs, metadata, analysis_res,  **kwds):
    # Deal with keyword arguments
    skp = kwds.get("skip", 5)  # Skipping sample points on plots
    wskp = kwds.get("walker_skip", 4)  # Skipping walkers
    cond_nice = (str(cond).replace(" ", "-").replace(")", "")
                .replace("(", "").replace(",", ""))
    figures_folder = kwds.get("figures_folder", "figures/")
    do_save = kwds.get("do_save", False)
    do_show = kwds.get("do_show", True)
    plot_chain = kwds.get("plot_chain", True)

    # Extract metadata
    (nsamples, nwalkers, parambounds, n_pm,
        param_names, gridbounds, params0) = read_metadata(metadata)

    # Aesthetical parameters
    map_colors = kwds.get("map_colors", ["xkcd:cornflower", "xkcd:sage",
                                        "xkcd:salmon", "xkcd:mustard"])
    linestyles = ["-", "-.", "--", ":"]
    markers = ["o", "s", "^", "*"]

    ## Check the Markov series of each parameter, compare to start value
    # I usually don't look at these Markov chain plots, drop them.
    if plot_chain:
        ncols = 2
        nrows = n_pm // ncols + max(n_pm % ncols, 1)
        fig, ax = plt.subplots(nrows, ncols)
        fig.set_size_inches(2.75*ncols, 2.25*nrows)
        ax = ax.flatten()

        markov_times = np.arange(nsamples)
        walker_colors = sns.color_palette("YlGnBu", n_colors=nwalkers)
        for i in range(n_pm):
            for j in range(0, nwalkers, wskp):
                ax[i].plot(markov_times[::skp], samples[i, j, ::skp], lw=1.0,
                    alpha=max(0.1, 1-0.5*nwalkers/100), color=walker_colors[j])
            ax[i].axhline(parambounds[0][i], color="k", lw=2.0)
            ax[i].axhline(parambounds[1][i], color="k", lw=2.0)
            ax[i].set(ylabel=param_names[i], xlabel="Markov time")
        for i in range(n_pm, ax.size):
            ax[i].set_axis_off()
        # Annotate MAP estimates as horizontal lines.
        for k, strat in enumerate(analysis_res["param_estimates"]):
            ls = linestyles.pop(0)
            linestyles.append(ls)
            map_strat = analysis_res["param_estimates"][strat]
            strat_nice = "MAP marginal" if strat=="MAP hist" else strat
            for i in range(n_pm):
                ax[i].axhline(map_strat[i], color=map_colors[k], lw=1.5,
                    ls=ls, label=strat_nice)
            ax[i].legend()  # last plot
        fig.tight_layout()
        figname = "markov_chains_course_{}.pdf".format(cond_nice)
        if do_save:
            fig.savefig(os.path.join(figures_folder, figname),
                transparent=True, bbox_inches="tight")
        if do_show:
            plt.show()
        plt.close()

    # Find the maximum a posteriori estimate.
    burn_in_frac = analysis_res["burn_in_frac"]
    burn_end = analysis_res["burn_in_steps"]
    # Reshape results so all walkers are contiguous in a flattened last dim.
    samples2 = np.moveaxis(samples[:, :, burn_end:], 1, 2).reshape(samples.shape[0], -1)
    costs2 = costs[:, burn_end:].T.flatten()

    ## Pairplot with hexbins for 2D density plots.
    # Flatten all steady-state samples of all walkers and remove burn-in phase
    idx =  pd.RangeIndex(0, nwalkers*(nsamples-burn_end), name="Sample")
    df_samples = pd.DataFrame(samples2.T, index=idx,
                        columns=pd.Index(param_names, name="Parameter"))
    g = hexpair(data=df_samples.reset_index(),
            grid_kws={"vars": param_names, "layout_pad": 0.4},
            hexbin_kws={"alpha": 0.8, "color":"k"},
            diag_kws={"color": "k", "legend": False, "bins":"doane"})

    # Annotate MAP on the diagonals and hexbin plots.
    for k, strat in enumerate(analysis_res["param_estimates"]):
        ls = linestyles.pop(0)
        mk = markers.pop(0)
        linestyles.append(ls)
        markers.append(mk)
        map_strat = analysis_res["param_estimates"][strat]
        strat_nice = "MAP marginal" if strat=="MAP hist" else strat
        for i in range(n_pm):
            for j in range(0, i):
                g.axes[i, j].plot(map_strat[j], map_strat[i], marker=mk, ms=4,
                    mfc=map_colors[k], mec=map_colors[k], ls="none",
                    label=strat_nice)
            # PairGrid uses special diagonal axes on top of the default subplot
            # axes; need to get those, otherwise will always plot under
            g.diag_axes[i].axvline(map_strat[i], ls=ls, color=map_colors[k],
                        label=strat_nice, lw=1.5, alpha=1.0)
    g.diag_axes[0].legend()
    g.tight_layout()
    figname = "mcmc_samples_histograms_{}.pdf".format(cond_nice)
    if do_save:
        g.fig.savefig(os.path.join(figures_folder, figname),
            transparent=True, bbox_inches="tight")
    if do_show:
        plt.show()
    plt.close()

    return 0
