import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys, os
import h5py, json
import multiprocessing
from time import perf_counter

# Local modules
import sys, os
if not "../" in sys.path:
    sys.path.insert(1, "../")

from mcmc.mcmc_analysis import (
    analyze_mcmc_run, graph_mcmc_samples,
    fit_summary_to_json
)
from mcmc.costs_tcr_car_antagonism import (
    cost_antagonism_tcr_car,
    cost_antagonism_tcr_car_priors,
    antag_ratio_panel_tcr_car
    )
from mcmc.prediction_utilities_tcr_car import (
    antag_ratio_panel_tcr_car_predict,
    find_1itam_car_ampli,
    find_6f_tcr_ampli,
    find_6f_tcr_thresh_fact,
    find_1itam_effect_tcr_ampli
)
from mcmc.utilities_tcr_car_antagonism import (
    prepare_car_antagonism_data,
    load_tcr_tcr_akpr_fits,
    plot_fit_car_antagonism,
    confidence_model_antagonism_car,
    plot_predict_car_antagonism,
    load_tcr_car_molec_numbers_ci
)
from utils.preprocess import (
    michaelis_menten,
    inverse_michaelis_menten,
    read_conc_uM,
    write_conc_uM,
    ln10
)
from models.conversion import convert_tau_ec50_relative

# Number of CPUS, for multiprocessing predictions
from utils.cpu_affinity import count_parallel_cpu
n_cpu = count_parallel_cpu()

# This means we are on the Physics servers
if sys.platform == 'linux':
    mpl.use("Agg")


### TCR-CAR ANALYSIS ###
def main_tcr_car_analysis(fl, **kwargs):
    """
    kwargs:
        do_save=False
        do_show=False
        tcr_conc="1uM" or list of concentrations
        analysis_res_fname
        samples_fname (str): name (not full path) of the 
            HDF file containing full MCMC results
        fit_summary_fname
        figures_folder
    """
    # Load keyword arguments
    do_save = kwargs.get("do_save", False)
    do_show = kwargs.get("do_show", False)
    tcr_conc = kwargs.get("tcr_conc", ["1uM", "1nM"])
    tcr_conc_name = "both_conc" if isinstance(tcr_conc, list) else tcr_conc
    analysis_res_fname = kwargs.get("analysis_res_fname", 
        "mcmc_analysis_tcr_car_{}.json".format(tcr_conc_name))
    analysis_res_file = os.path.join(fl, analysis_res_fname)
    samples_fname = kwargs.get("samples_fname",
        "mcmc_results_tcr_car_{}.h5".format(tcr_conc_name))
    fit_summary_fname = kwargs.get("fit_summary_fname",
        "fit_summary_tcr_car_{}.json".format(tcr_conc_name))
    figures_folder = kwargs.get("figures_folder",
        os.path.join("..", "figures", "mcmc_tcr_car"))

    # Import MCMC results
    results_tcr_car = h5py.File(os.path.join(fl, samples_fname), "r")
    samples_group = results_tcr_car.get("samples")
    cost_group = results_tcr_car.get("cost")
    data_group = results_tcr_car.get("data")
    prior_dist = cost_group.attrs.get("prior_dist", "uniform")

    # Get correct cost function
    cost_fct = (cost_antagonism_tcr_car if prior_dist == "uniform"
                else cost_antagonism_tcr_car_priors)

    # Import data
    data_file_name = data_group.attrs.get("data_file_name")
    l_conc_mm_params = data_group.get("l_conc_mm_params")[()]
    with open(os.path.join("..","data","pep_tau_map_ot1.json"), "r") as handle:
        pep_tau_map_ot1 = json.load(handle)
    df = pd.read_hdf(data_file_name)
    df_fit, df_ci_log2 = prepare_car_antagonism_data(df, l_conc_mm_params,
                            pep_tau_map_ot1, tcr_conc=tcr_conc,
                            tcr_itams="10", car_itams="3")

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

    with open(analysis_res_file, "w") as jfile:
        json.dump(all_results_dicts, jfile, indent=2)

    # Write nice summary of all model parameters to JSON file
    fit_summ = fit_summary_to_json(
        data_group, samples_group, all_results_dicts, meth="MAP best"
    )
    with open(os.path.join(fl, fit_summary_fname), "w") as h:
        json.dump(fit_summ, h, indent=2)

    # Use the analysis results to make plots
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

    # Close the results file
    results_tcr_car.close()

    return 0


def main_tcr_car_predictions(fl, **kwargs):
    """ Make predictions for different TCR antigen densities and
    different ITAM numbers from the parameter samples found by mcmc.

    Try to predict N4 too.
    args:
        fl: folder where MCMC simulation results are saved
    kwargs:
        do_save=False
        do_show=False
        fit_conc=["1uM", "1nM"]
        n_boot=1000
        analysis_res_fname
        samples_fname
        predictions_res_fname
        analysis_6f
        results_6f
        main_seedseq
        figures_folder
        plot_res_file (to save confidence intervals)
        for_plots_folder (where the CI file will be saved)
        molec_counts_fi (str): file with molecule counts, default
            "../data/surface_counts/surface_molecule_summary_stats.h5"
        mtc (str): default "Geometric mean"
        tcell_type (str): default "OT1_Naive"
        tumor_type (str): default "E2aPBX_WT"
        tumor_antigen (str): default "CD19"

    """
    # Load the keyword arguments
    do_save = kwargs.get("do_save", False)
    do_show = kwargs.get("do_show", False)
    fit_conc = kwargs.get("fit_conc", ["1uM", "1nM"])
    fit_conc_name = "both_conc" if isinstance(fit_conc, list) else fit_conc
    n_boot = kwargs.get("n_boot", 1000)

    # Various files and folders arguments
    if not (do_save or do_show):  # No plots
        return 0
    # Get all file names, either default locations or kwargs ones.
    analysis_res_fname = kwargs.get("analysis_res_fname",
        "mcmc_analysis_tcr_car_{}.json".format(fit_conc_name))
    analysis_res_file = os.path.join(fl, analysis_res_fname)
    samples_fname = kwargs.get("samples_fname",
        "mcmc_results_tcr_car_{}.h5".format(fit_conc_name))
    predictions_res_fname = kwargs.get("predictions_res_fname",
        "model_predictions_tcr_car_{}.h5".format(fit_conc_name))
    predictions_res_file = os.path.join(fl, predictions_res_fname)
    figures_folder = kwargs.get("figures_folder",
        os.path.join("..", "figures", "model_predictions"))
    figures_folder_fit = kwargs.get("figures_folder_fit", 
        os.path.join("..", "figures", "mcmc_tcr_car"))
    for_plots_folder = kwargs.get("for_plots_folder", 
                            os.path.join("..", "results", "for_plots"))
    plot_res_file = kwargs.get("plot_res_file",
        "dfs_model_data_ci_mcmc_{}.h5".format(fit_conc_name))
    plot_res_file = os.path.join(for_plots_folder, plot_res_file)
    analysis_6f = kwargs.get("analysis_6f", "mcmc_analysis_tcr_tcr_6f.json")
    analysis_6f = os.path.join(fl, analysis_6f)
    results_6f = kwargs.get("results_6f", "mcmc_results_tcr_tcr_6f.h5")
    results_6f = os.path.join(fl, results_6f)
    molec_counts_fi = kwargs.get("molec_counts_fi", os.path.join("..", "data",
        "surface_counts", "surface_molecule_summary_stats.h5"))

    # Arguments for molecule number
    mtc = kwargs.get("mtc", "Geometric mean")
    tumor_type = kwargs.get("tumor_type", "E2aPBX_WT")
    tcell_type = kwargs.get("tcell_type", "OT1_CAR")
    tumor_antigen = kwargs.get("tumor_antigen", "CD19")

    # Import MCMC results
    results_tcr_car = h5py.File(os.path.join(fl, samples_fname), "r")
    samples_group = results_tcr_car.get("samples")
    cost_group = results_tcr_car.get("cost")
    data_group = results_tcr_car.get("data")

    main_seedseq = kwargs.get("main_seedseq", None)
    if main_seedseq is None:
        # Original version had four different manually generated seeds
        spawned_sequences = [
            np.random.SeedSequence(0x32393bbb64883bf94c7f39b5a8f2fd69,
                spawn_key=(0xbad2d1339b15b6ce4db649b277cf84b,)),
            np.random.SeedSequence(0x99fb375952c5f85114e6b146dc5f031,
                spawn_key=(0x3fabc1f4d66953fb097b4e3d2913e5b3,)),
            np.random.SeedSequence(0x1e66a3ba12430a8c672474761d8de9b4,
                spawn_key=(0x88272fc85e8711e12ddb09576c6a4a02,)),
            np.random.SeedSequence(0x2f8c461221ed205e3f05e6e8f9a6823b,
                spawn_key=(0xac1383a71f34032cdcaf056395bbfd54,))
        ]
    # For new simulations, more clever to spawn from one main seedsequence
    else:
        spawned_sequences = main_seedseq.spawn(4)

    # Prepare all data: keep ITAM numbers levels and TCR Antigen Density
    data_file_name = data_group.attrs.get("data_file_name")
    l_conc_mm_params = data_group.get("l_conc_mm_params")[()]
    with open(os.path.join("..","data","pep_tau_map_ot1.json"), "r") as handle:
        pep_tau_map_ot1 = json.load(handle)
    df = pd.read_hdf(data_file_name)
    df_fit, df_ci_log2 = prepare_car_antagonism_data(df, l_conc_mm_params,
                            pep_tau_map_ot1, tcr_conc=["1nM", "1uM"],
                            tcr_itams=slice(None), car_itams=slice(None))

    # This data point, CD19 only and no TCR antigen, has trivially FC=1
    # so it's not data we are trying to predict; we use it to set the tau
    # threshold of 1-ITAM CAR threshold
    df_cyto_cd19only = (df.xs("None", level="TCR_Antigen")
                        .xs("CD19", level="CAR_Antigen"))
    # TCR Ag, no CD19 measurements have not been used; also independent data
    # to tune 6F and make real predictions.
    df_cyto_tcronly = df.xs("None", level="CAR_Antigen")

    # Load standard deviation of the mean estimators of TCR, CAR, CD19, MHC,
    # and loading EC50 parameters
    molec_loads = load_tcr_car_molec_numbers_ci(
        molec_counts_fi, mtc, tumor_type=tumor_type,
        tcell_type=tcell_type, tumor_antigen=tumor_antigen
    )
    all_stds, all_dofs = molec_loads[5], molec_loads[6]

    # Load the analysis of each run
    try:
        jfile = open(analysis_res_file, "r")
        all_results_dicts = json.load(jfile)
        jfile.close()
    except FileNotFoundError:
        raise FileNotFoundError("Run main_tcr_car_analysis first.")

    ## Function launching predictions in parallel, with special tweaks
    # for each kind of predictions. Uses a lot of things loaded above
    # hence why we are defining this function inside the main.
    def launch_predictions(pred_key, seedseq, df_dat, df_ci):
        """ pred_key: AgDens, TCRNum or CARNum
            seedseq: Seed Sequence
        """
        print("Starting to generate {} model prediction samples...".format(pred_key))
        start_t = perf_counter()
        # Load constant model parameters
        cost_args_loaded = [data_group.get(a)[()]
                            for a in data_group.attrs.get("cost_args_names")]
        # Special tweaks for some kinds of predictions: 1-ITAM and/or 6F TCR
        if "CARNum" in pred_key:
            # Make necessary parameter adjustments: setting n, m, f = 1
            # for every (k, m, f) tried for 3-ITAM CARs.
            for i, a in enumerate(data_group.attrs.get("cost_args_names")):
                if a == "tcr_car_nmf":  # n_tcr, m_tcr, f_tcr, n_car
                    cost_args_loaded[i][3] = 1
            # If nothing works, just re-fit CAR tau threshold for 1-ITAM, the other
            # parameters should remain fixed.
        elif "TCRNum" in pred_key:
            # Load 6F TCR parameters, hope they work with the CAR/TCR params
            tcr_6f_loads = load_tcr_tcr_akpr_fits(results_6f, analysis_6f, klim=1)
            # params: phi, kappa, cmthresh, I0p, kp, psi0, gamma_tt
            # then [N, m, f] of TCR, and I_tot.
            tcr_6f_params, tcr_6f_nmf, tcr_6f_itot = tcr_6f_loads
            # Replace cost args: ['fixed_rates', 'tcr_car_ritots', 'tcr_car_nmf', 'cd19_tau_l']
            # fixed_rates: phi_tcr, kappa_tcr, cmthresh_tcr, ithresh_tcr, k_tcr, psi0_tcr,
            # gamma_tt=1.0
            cost_args_loaded[0][0:7] = tcr_6f_params
            # tcr_car_ritots:  R_tot_tcr, ITp, R_tot_car
            cost_args_loaded[1][1] = tcr_6f_itot
            # tcr_car_nmf changes to tcr_6f_nmf
            cost_args_loaded[2][0:3] = tcr_6f_nmf

        # Untweaked model parameters for 1-ITAM vs 3-ITAM amplitude correction
        cost_args_thresh = [data_group.get(a)[()]
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
            # Printing some info here
            if "AgDens" in pred_key:
                print("For {}, best parameters {} are:".format(cond,
                                        samples_group.attrs.get("param_names")))
                p_best_exp = np.exp(p_best * ln10)
                print(p_best_exp, "\n")
                cost_args_loc = cost_args_loaded
                grid_point_loc = grid_point
                panel_fct = antag_ratio_panel_tcr_car
                # Stdev of amplitude and threshold factors: none here
                factors_stds = None
            # Here, more tweaks for predictions in different cell lines
            elif pred_key == "CARNum":
                # Change m, f to 1, 1
                grid_point_loc = (grid_point[0], 1, 1)
                # Determine amplitude of 1-ITAM CAR response
                car_ampli, car_ampli_std = find_1itam_car_ampli(
                    df_cyto_cd19only, p_best, grid_point, *cost_args_thresh)
                # Also determine impact of 1-ITAM CAR on TCR response amplitude
                ta = find_1itam_effect_tcr_ampli(df_cyto_tcronly)
                tcr_ampli_factor, tcr_ampli_factor_std = ta
                cost_args_loc = list(cost_args_loaded) + [[tcr_ampli_factor, car_ampli], 1.0]
                panel_fct = antag_ratio_panel_tcr_car_predict
                factors_stds = [[tcr_ampli_factor_std, car_ampli_std], 0.0]
            elif pred_key == "TCRNum":
                # Determine max. output amplitude of 6F relative to 6Y
                tcr_ampli_6f, tcr_ampli_6f_std = find_6f_tcr_ampli(df_cyto_tcronly)
                # Also relative difference in tau threshold based on Hill fits
                th = find_6f_tcr_thresh_fact(df_cyto_tcronly, pep_tau_map_ot1)
                tcr_thresh_fact, tcr_thresh_fact_std = th[:2]
                cost_args_loc = list(cost_args_loaded) + [[tcr_ampli_6f, 1.0], tcr_thresh_fact]
                grid_point_loc = grid_point
                panel_fct = antag_ratio_panel_tcr_car_predict
                factors_stds = [[tcr_ampli_6f_std, 0.0], tcr_thresh_fact_std]
            elif pred_key == "CARNum_TCRNum":
                # Determine max. output amplitude of 6F relative to 6Y
                tcr_ampli_6f, tcr_ampli_6f_std = find_6f_tcr_ampli(df_cyto_tcronly)
                # Also relative difference in tau threshold based on Hill fits
                th = find_6f_tcr_thresh_fact(df_cyto_tcronly, pep_tau_map_ot1)
                tcr_thresh_fact, tcr_thresh_fact_std = th[:2]
                # Determine amplitude of 1-ITAM CAR response
                car_ampli, car_ampli_std = find_1itam_car_ampli(
                    df_cyto_cd19only, p_best, grid_point, *cost_args_thresh)
                # Also determine impact of 1-ITAM CAR on TCR response amplitude
                ta = find_1itam_effect_tcr_ampli(df_cyto_tcronly)
                tcr_ampli_factor, tcr_ampli_factor_std = ta
                tcr_ampli = 1.0 * tcr_ampli_factor * tcr_ampli_6f
                # Sum the individual variances of these ampli factors
                tcr_ampli_std = np.sqrt(tcr_ampli_factor_std**2 + tcr_ampli_6f_std**2)
                # Change m, f of the CAR to 1, 1
                grid_point_loc = (grid_point[0], 1, 1)
                cost_args_loc = list(cost_args_loaded) + [[tcr_ampli, car_ampli], tcr_thresh_fact]
                panel_fct = antag_ratio_panel_tcr_car_predict
                factors_stds = [[tcr_ampli_std, car_ampli_std], tcr_thresh_fact_std]

            # Compute model ratios
            res = pool.apply_async(
                    confidence_model_antagonism_car,
                    args=(panel_fct, p_samp, p_best,
                            grid_point_loc, df_dat, df_ci),
                    kwds=dict(other_args=cost_args_loc, n_taus=200,
                            n_samp=n_boot, seed=seeds.pop(),
                            antagonist_lvl="TCR_Antigen",
                            l_conc_mm_params=l_conc_mm_params,
                            molec_stds=all_stds, molec_dofs=all_dofs,
                            factors_stds=factors_stds)
            )
            all_processes[cond] = res

        # Collect model predictions for each kmf into one DataFrame
        df_pred_model = pd.concat({k:a.get() for k, a in all_processes.items()},
                        names=["kmf"]).sort_index().sort_index(axis=1)
        # Add extra level telling what condition has been fitted
        if pred_key == "AgDens":  # Some fits in there
            fit_conc_l = [michaelis_menten(read_conc_uM(c), *l_conc_mm_params)
                        for c in fit_conc]
            df_pred_model.loc[df_pred_model.index.isin(fit_conc_l,
                        level="TCR_Antigen_Density"), "Subset"] = "Fit"
        else:
            df_pred_model["Subset"] = ["Prediction"]*df_pred_model.shape[0]
        df_pred_model = df_pred_model.set_index("Subset", append=True)
        df_pred_model.to_hdf(predictions_res_file, key=pred_key, mode="a")
        delta_t = perf_counter() - start_t
        pool.close()
        print("Total time to generate {} predictions: {} s".format(pred_key, delta_t))

        return df_pred_model


    ### PART 1: Plot 1 nM TCR Antigen density for 10-ITAM TCR, 3-ITAM CAR
    df_agdens = df_fit.loc[("10", "3")]
    df_agdens_ci = df_ci_log2.loc[("10", "3")]
    df_agdens_model = launch_predictions("AgDens", spawned_sequences[0],
                                        df_agdens, df_agdens_ci)
    # Make plots comparing the data and the model fits with CIs
    for cond in all_results_dicts.keys():
        cond_nice = (str(cond).replace(" ", "-").replace(")", "")
                        .replace("(", "").replace(",", ""))
        posts_cond = all_results_dicts[cond].get("posterior_probs")
        df_model = df_agdens_model.loc[cond, "best"].droplevel("Subset")
        percentiles = [a for a in df_agdens_model.columns 
                       if a.startswith("percentile")]
        df_model_ci = df_agdens_model.loc[cond, percentiles].droplevel("Subset")
        fig, _ = plot_fit_car_antagonism(df_agdens, df_model,
                    l_conc_mm_params, df_agdens_ci, cost=posts_cond.get("MAP best"), 
                    model_ci=df_model_ci)
        if do_save:
            fig.savefig(os.path.join(figures_folder_fit,
                        "data_model_comparison_{}.pdf".format(cond_nice)),
                        bbox_inches="tight", transparent=True)
        if do_show:
            plt.show()
        plt.close()

    ### PART 2: Predict 6F TCR, using same TCR-CAR interaction parameters
    # but TCR parameters from separate fits
    df_6f = df_fit.loc[("4", "3")]
    df_6f_ci = df_ci_log2.loc[("4", "3")]
    # Collect model predictions for each kmf into one DataFrame
    df_6f_model = launch_predictions("TCRNum", spawned_sequences[1],
                                            df_6f, df_6f_ci)


    ### PART 3: Predict 1-ITAM CAR, both concentrations
    # Import relevant data (really only need the 1-ITAM part)
    df_carnum = df_fit.loc[("10", "1")]
    df_carnum_ci = df_ci_log2.loc[("10", "1")]
    # Collect model predictions for each kmf into one DataFrame
    df_carnum_model = launch_predictions("CARNum", spawned_sequences[2],
                                        df_carnum, df_carnum_ci)

    ### PART 4: Predict 1-ITAM CAR, 6F TCR (only high concentration available)
    df_6f_carnum = df_fit.loc[("4", "1")]
    df_6f_carnum_ci = df_ci_log2.loc[("4", "1")]
    # Collect model predictions for each kmf into one DataFrame
    df_6f_carnum_model = launch_predictions("CARNum_TCRNum", spawned_sequences[3],
                                            df_6f_carnum, df_6f_carnum_ci)


    ### Join all predictions into one model dataframe, same for data and errors
    dfs_model = pd.concat({("10", "3"):df_agdens_model,
                    ("10", "1"):df_carnum_model,
                    ("4", "3"): df_6f_model,
                    ("4", "1"): df_6f_carnum_model},
                    names=["TCR_ITAMs", "CAR_ITAMs"])
    dfs_data = pd.concat({("10", "3"):df_agdens,
                    ("10", "1"):df_carnum,
                    ("4", "3"): df_6f,
                    ("4", "1"): df_6f_carnum},
                    names=["TCR_ITAMs", "CAR_ITAMs"])
    dfs_ci = pd.concat({("10", "3"):df_agdens_ci,
                    ("10", "1"):df_carnum_ci,
                    ("4", "3"): df_6f_ci,
                    ("4", "1"): df_6f_carnum_ci},
                    names=["TCR_ITAMs", "CAR_ITAMs"])

    # Use the analysis results to make plots
    for cond in dfs_model.index.get_level_values("kmf").unique():
        df_plot = dfs_model.xs(cond, level="kmf")
        fig, axes = plot_predict_car_antagonism(dfs_data, df_plot,
            l_conc_mm_params, dfs_ci)
        if do_save:
            cond_nice = (str(cond).replace(" ", "-").replace(")", "")
                            .replace("(", "").replace(",", ""))
            fig.savefig(os.path.join(figures_folder,
                        "model_prediction_tcr_density_{}.pdf".format(cond_nice)),
                        bbox_inches="tight", transparent=True)
        if do_show:
            plt.show()
        plt.close()

        print("Finished graphing prediction of condition {}".format(cond))

    # Close the results file
    results_tcr_car.close()

    # Save all final plotting data to disk for final figures production
    # Rename concentrations to uM, nM for plotting convenience
    # Also convert taus to ec50s
    def reverse_mm_fitted(x):
        return inverse_michaelis_menten(x, *l_conc_mm_params)
    with open(os.path.join("..", "data", "reference_pep_tau_maps.json"), "r") as h:
        pep_tau_refs = json.load(h)
    def reverse_tau_to_ec50(x):
        return convert_tau_ec50_relative(x, pep_tau_refs["N4"],
                                        npow=pep_tau_refs["npow"])
    def renamer(d):
        # Add a level for EC50 so we have the choice when plotting
        new_col = reverse_tau_to_ec50(d.index.get_level_values("TCR_Antigen").values)
        new_idx = pd.MultiIndex.from_tuples(
                    [tuple(d.index[i]) + (new_col[i],) for i in range(len(new_col))],
                    names=list(d.index.names) + ["TCR_Antigen_EC50"])
        return (d.set_axis(new_idx)
                 .rename(reverse_mm_fitted, axis=0, level="TCR_Antigen_Density")
                 .rename(write_conc_uM, axis=0, level="TCR_Antigen_Density")
                )

    dfs_model = renamer(dfs_model)
    dfs_data = renamer(dfs_data)
    dfs_ci = renamer(dfs_ci)
    dfs_data.name = "Antagonism ratio"

    dfs_model.to_hdf(plot_res_file, key="model", mode="w")
    dfs_data.to_hdf(plot_res_file, key="data", mode="a")
    dfs_ci.to_hdf(plot_res_file, key="ci", mode="a")

    return 0


def save_z_correction_factors(fl, fit_conc, **kwargs):
    """
    fl: folder where MCMC results are saved
    kwargs:
        analysis_res_fname
        samples_fname
        factors_fname
    """
    fit_conc_name = "both_conc" if isinstance(fit_conc, list) else fit_conc
    analysis_res_fname = kwargs.get("analysis_res_fname", 
            "mcmc_analysis_tcr_car_{}.json".format(fit_conc_name))
    analysis_res_file = os.path.join(fl, analysis_res_fname)
    
    samples_fname = kwargs.get("samples_fname",
        "mcmc_results_tcr_car_{}.h5".format(fit_conc_name))
    factors_fname = kwargs.get("factors_fname",
        "prediction_factors_tcr_car_{}.json".format(fit_conc_name))
    # Import MCMC results
    results_tcr_car = h5py.File(os.path.join(fl, samples_fname), "r")
    data_group = results_tcr_car.get("data")
    samples_group = results_tcr_car.get("samples")

    cost_args = [data_group.get(a)[()]
                    for a in data_group.attrs.get("cost_args_names")]

    # Data for alphas estimation
    data_file_name = data_group.attrs.get("data_file_name")
    with open(os.path.join("..","data","pep_tau_map_ot1.json"), "r") as handle:
        pep_tau_map_ot1 = json.load(handle)
    df = pd.read_hdf(data_file_name)
    df_cyto_cd19only = (df.xs("None", level="TCR_Antigen")
                        .xs("CD19", level="CAR_Antigen"))
    # TCR Ag, no CD19 measurements have not been used; also independent data
    # to tune 6F and make real predictions.
    df_cyto_tcronly = df.xs("None", level="CAR_Antigen")

    # Load best parameter fits
    try:
        jfile = open(analysis_res_file, "r")
        all_results_dicts = json.load(jfile)
        jfile.close()
    except FileNotFoundError:
        raise FileNotFoundError("Run main_tcr_car_analysis first.")

    # Determine max. output amplitude of 6F relative to 6Y
    tcr_ampli_6f, tcr_ampli_6f_std = find_6f_tcr_ampli(df_cyto_tcronly)
    # Also relative difference in tau threshold based on Hill fits
    th = find_6f_tcr_thresh_fact(df_cyto_tcronly, pep_tau_map_ot1)
    tcr_thresh_fact_6f, tcr_thresh_fact_6f_std, _ = th
    # Determine amplitude of 1-ITAM CAR response, for each set of parameters
    car_ampli_factors = {}
    car_ampli_factors_std = {}
    for cond in all_results_dicts.keys():
        grid_point = samples_group.get(cond).attrs.get("run_id")
        p_best = np.asarray(all_results_dicts[cond]
                        .get("param_estimates")["MAP best"])
        car_ampli, car_ampli_std = find_1itam_car_ampli(df_cyto_cd19only, p_best,
            grid_point, *cost_args)
        car_ampli_factors[cond] = car_ampli
        car_ampli_factors_std[cond] = car_ampli_std

    # Also determine impact of 1-ITAM CAR on TCR response amplitude
    tcr_ampli_factor, tcr_ampli_factor_std = find_1itam_effect_tcr_ampli(df_cyto_tcronly)

    results_tcr_car.close()

    # Save results to dict
    alphas_dict = {
        'tcr_ampli_6f': tcr_ampli_6f,
        'tcr_ampli_6f_std': tcr_ampli_6f_std,
        'tcr_thresh_factor_6f': tcr_thresh_fact_6f,
        'tcr_thresh_factor_6f_std': tcr_thresh_fact_6f_std,
        'car_ampli_1itam': car_ampli_factors,
        'car_ampli_1itam_std': car_ampli_factors_std,
        'tcr_ampli_1itam': tcr_ampli_factor,
        'tcr_ampli_1itam_std': tcr_ampli_factor_std
    }
    with open(os.path.join(fl, factors_fname), "w") as h:
        json.dump(alphas_dict, h, indent=2)
    return alphas_dict


if __name__ == "__main__":
    # We provide file names and paths separately to analysis scripts. 
    # This is because there are many file names in the same results folder. 
    # However, data files are always provided as complete paths. 
    chosen_fit_conc = ["1uM", "1nM"]
    folder_results = os.path.join("..", "results", "mcmc")
    figures_folder_fit = os.path.join("..", "figures", "mcmc_tcr_car")
    folder_for_plots = os.path.join("..", "results", "for_plots")

    # Check if there are existing results and skip analysis if that's the case
    skip_analysis = False
    tcr_conc_name = ("both_conc" if isinstance(chosen_fit_conc, list) 
                     else chosen_fit_conc)
    analysis_filename = "mcmc_analysis_tcr_car_{}.json".format(tcr_conc_name)
    analysis_filepath = os.path.join(folder_results, analysis_filename)
    ci_res_filename = "model_predictions_tcr_car_{}.h5".format(tcr_conc_name)
    ci_res_filepath = os.path.join(folder_results, ci_res_filename)

    if os.path.isfile(analysis_filepath):
        if os.path.isfile(ci_res_filepath):
            raise RuntimeError("Existing model analysis file found at "
                + str(analysis_filepath) + " and a model CI file at "
                + str(ci_res_filepath) + ". If you want to re-run, "
                + "delete these existing files first. ")
            # This is being re-run just to get model CI; skip analysis
        else: 
            skip_analysis = True
    
    if not skip_analysis:
        print("Launching TCR/CAR analysis...")
        main_tcr_car_analysis(
            folder_results, do_save=True, tcr_conc=chosen_fit_conc,
            figures_folder=os.path.join("..", "figures", "mcmc_tcr_car"), 
            analysis_res_fname=analysis_filename
        )
    else:
        print("...\nUsing existing TCR/CAR analysis file, "
              + "not generating analysis plots, "
              + "and skipping to model CI generation. \n...\n"
        )

    # Make prediction about other cell types and compare to data.
    # Do not run if the results file already exists, 
    # to avoid deleting good results. 
    if os.path.isfile(ci_res_filepath):
        raise RuntimeError("Existing model confidence intervals file found at "
            + str(ci_res_filepath) + ". If you want to re-run, "
            + "delete this existing file first. ")
    
    molec_counts_file = os.path.join("..", "data", "surface_counts",
                                "surface_molecule_summary_stats.h5")
    main_tcr_car_predictions(
        folder_results, do_save=True, fit_conc=chosen_fit_conc,
        n_boot=1000, tumor_type="E2aPBX_WT", tcell_type="OT1_CAR",
        tumor_antigen="CD19", mtc="Geometric mean",
        molec_counts_fi=molec_counts_file, 
        figures_folder_fit=figures_folder_fit, 
        analysis_res_fname=analysis_filename, 
        predictions_res_fname=ci_res_filename, 
        for_plots_folder=folder_for_plots
    )

    # Recompute and save the correction factors applied to Z^C and Z^T
    save_z_correction_factors(
        folder_results, fit_conc=chosen_fit_conc, 
        analysis_res_fname=analysis_filename
    )
    print("Finished analysis and saving of predictions!")
