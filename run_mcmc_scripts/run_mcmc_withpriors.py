"""
Alternate MCMC parameter estimation simulations without boundaries on
parameters but prior distributions defaulting to parameter values in the
classical AKPR SHP-1 model.
Sequence of runs:
    - TCR/TCR antagonism on regular TCRs
    - TCR/TCR antagonism on 6F TCRs
    - TCR/CAR antagonism
    - TCR/CAR predictions

The point is to show that no good parameter fits have been excluded
by the parameter boundaries chosen by default. 

Warning: this script will only run if there are no existing results 
for the starting point chosen or later in results/mcmc_withpriors/, 
to avoid erasing existing results. Just empty the folder of these
results before running (delete or move files elsewhere). 

Warning: this script takes several hours to run on a 32-CPU cluster, 
it needs to be run on a dedicated computer. 

Warning: since the prior is unbounded, MCMC walkers run into 
extreme parameter values which raise various errors in 
the numerical model solution functions. These events are
assigned -infinity log-likelihood so they do not disrupt the
parameter optimization, but we filter out these warnings, 
also using the verbose=False option of the cost functions. 

@author: frbourassa
July 2024
"""

import numpy as np
import pandas as pd
import json
from time import perf_counter
import emcee
import multiprocessing
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# Local modules
import sys, os
if not "../" in sys.path:
    sys.path.insert(1, "../")

# For multiprocessing
from utils.cpu_affinity import count_parallel_cpu
n_cpu = count_parallel_cpu()

# This means we are on the Physics servers
import matplotlib as mpl
import matplotlib.pyplot as plt
if sys.platform == 'linux':
    mpl.use("Agg")


from mcmc.utilities_tcr_tcr_antagonism import (
    prepare_data,
    prepare_data_6f,
    load_tcr_tcr_molec_numbers
)
from mcmc.utilities_tcr_car_antagonism import (
    prepare_car_antagonism_data,
    load_tcr_car_molec_numbers
)

from run_mcmc_scripts.run_mcmc_tcr_tcr_akpr_i import main_akpr_i_run
from run_mcmc_scripts.run_mcmc_tcr_tcr_6f import main_tcr_tcr_run_6f
from run_mcmc_scripts.run_mcmc_tcr_car import main_tcr_car_antagonism
from run_mcmc_scripts.analyze_mcmc_tcr_tcr import (
    main_mcmc_analysis,
    main_tcr_tcr_confidence
)
from run_mcmc_scripts.analyze_mcmc_tcr_car import (
    main_tcr_car_analysis,
    main_tcr_car_predictions,
    save_z_correction_factors
)
from mcmc.costs_tcr_tcr_antagonism import cost_antagonism_akpr_priors
from mcmc.costs_tcr_car_antagonism import cost_antagonism_tcr_car_priors
from mcmc.utilities_tcr_tcr_antagonism import (
    prepare_data_6f,
    load_tcr_tcr_molec_numbers
)
from mcmc.mcmc_analysis import find_best_grid_point


if __name__ == "__main__":
    # Start points of this main script (for re-running after crash)
    # 0:all, 1:6F, 2:TCR analysis, 3:TCR CI, 4: TCR/CAR run,
    # 5: TCR/CAR analysis, 6: TCR/CAR predictions
    start_point = 0
    # Define all file names first
    # For main MCMC run functions, we provide complete paths
    # For main analysis functions, we provide file names
    # and folder names separately (for conciseness, 
    # since many files are in the same folder)
    # Data files are always provided as full paths. 

    # Folders for results and figures
    folder_results = os.path.join("..", "results", "mcmc_withpriors")
    folder_results_for_plots = os.path.join(
        "..", "results", "for_plots_withpriors")
    car_fig_subfolder = os.path.join("..", "figures", "mcmc_tcr_car_withpriors")
    predictions_figure_folder = os.path.join(
        "..", "figures", "model_predictions_withpriors")
    
    # Data files
    molec_counts_fi = os.path.join(
        "..", "data", "surface_counts", "surface_molecule_summary_stats.h5")
    data_file_name = os.path.join(
        "..", "data", "antagonism", "allManualSameCellAntagonismDfs_v3.h5")
    data_file_name_6f = os.path.join(
        "..", "data", "antagonism", "combined_6f_tcr_tcr_antagonism_datasets.h5")
    data_file_name_car = os.path.join("..", "data", "antagonism",
                            "combined_OT1-CAR_antagonism.hdf")
    
    # Results files for TCR/TCR antagonism, full paths
    results_file_akpr = os.path.join(
        folder_results, "mcmc_results_akpr_i_withpriors.h5")
    results_file_6f = os.path.join(
        folder_results, "mcmc_results_tcr_tcr_6f_withpriors.h5")
    
    # Analysis file names for TCR/TCR antagonism, file names only
    analysis_kwarguments = {
        "akpr_i": {
            "analysis_res_fname": "mcmc_analysis_akpr_i_withpriors.json",
            "fit_summary_fname": "fit_summary_akpr_i_withpriors.json",
            "results_fname": os.path.split(results_file_akpr)[-1],
            "fig_subfolder": "mcmc_akpr_i_withpriors",
            "model": "akpr_i"
        },
        "6f": {
            "analysis_res_fname": "mcmc_analysis_tcr_tcr_6f_withpriors.json",
            "fit_summary_fname": "fit_summary_tcr_tcr_6f_withpriors.json",
            "results_fname": os.path.split(results_file_6f)[-1],
            "fig_subfolder": "mcmc_tcr_tcr_6f_withpriors",
            "model": "6f"
        }
    }
    tcr_ci_res_fname = "model_confidence_intervals_tcr_tcr_withpriors.h5"
    tcr_ci_res_fpath = os.path.join(folder_results, tcr_ci_res_fname)
    tcr_plotting_data_fname = "dfs_model_data_ci_mcmc_tcr_tcr_withpriors.h5"

    # Results files for TCR/CAR antagonism, full paths
    tcr_car_conc = "both_conc"
    # Provide the file name only (not path) to analysis functions
    car_results_fname = "mcmc_results_tcr_car_{}_withpriors.h5".format(tcr_car_conc)
    # And the full path to the main run functions
    car_results_file = os.path.join(folder_results, car_results_fname)
    

    # Analysis files for TCR/CAR antagonism, file names only
    car_analysis_fname = "mcmc_analysis_tcr_car_both_conc_withpriors.json"
    car_summary_fname = "fit_summary_tcr_car_both_conc_withpriors.json"
    car_predictions_fname = "model_predictions_tcr_car_withpriors.h5"
    car_ci_fname = "dfs_model_data_ci_mcmc_withpriors.h5"
    car_ci_fpath = os.path.join(folder_results, car_ci_fname)
    car_zfactors_fname = "prediction_factors_tcr_car_withpriors.json"


    # The number of steps taken is thin_param times number_samples
    # As long as number_samples is below 10^5 there should be no RAM issues
    # For larger number_samples, will need to use HDF5 backend of emcee.
    number_samples = 10000
    thin_param = 0

    # Number of TCR per T cell, L-pulse conversion parameters, peptide taus
    mtc = "Geometric mean"
    nums = load_tcr_tcr_molec_numbers(molec_counts_fi, mtc,
                                        tcell_type="OT1_Naive")
    tcr_number, l_conc_mm_params, pep_tau_map_ot1 = nums

    ## TCR/TCR antagonism ratio fitting for regular TCRs
    # Prepare data for fitting antagonism ratios
    df = pd.read_hdf(data_file_name)
    data_prep = prepare_data(df, l_conc_mm_params, pep_tau_map_ot1)
    print(data_prep[0])
    seed_sequence = np.random.SeedSequence(0xc679cd3e946a22b5e541c11476758433,
                    spawn_key=(0xaa5bd6b629eae90f1b5d98ed13802f2,))

    # Arguments specific to wider bounds for AKPR model
    # Use values of these parameters in the original SHP-1 model as priors
    I_total = 1.0
    # varphi: 0.09 in classical model
    # C_m,th: SHP-1 activation half-max. at C_m = 500 in classical model
    # I_th: take 0.01, no such threshold in the original model, but a small
    # I_th is equivalent to a strong feedback, as in the original model
    # Psi_0: just say 100x less than phi, ensure antagonism of 2^-4 is possible
    classic_akpr_pvec = [0.09, 500.0, 0.01, 0.0009]
    classic_akpr_pvec = np.log10(np.asarray(classic_akpr_pvec))
    # Not really boundaries, but passed as this argument.
    # Gaussian prior widths: tight on varphi because exists in classical model,
    # more relaxed on parameters without exact equivalent in the classical model
    akpr_prior_stds = np.asarray([0.1, 2.0, 2.0, 2.0])
    param_bounds = [classic_akpr_pvec, akpr_prior_stds**2]
    # Special moves, like for TCR/CAR simulations, which give faster
    # convergence while avoiding walkers stuck in local minima
    special_moves = [
        (emcee.moves.DEMove(gamma0=2/np.sqrt(2*len(classic_akpr_pvec))), 0.6),
        (emcee.moves.DESnookerMove(gammas=1.5), 0.2),
        (emcee.moves.WalkMove(), 0.2)
    ]
    emcee_kwargs = {"moves": special_moves}
    cost_fct = cost_antagonism_akpr_priors

    # Main run
    if start_point <= 0:
        if  os.path.isfile(results_file_akpr):
            raise RuntimeError(f"Existing {results_file_akpr} found; "
                + "delete before running from start_point <= 0")
        else:
            print("Starting revised AKPR fits...")
        nsteps = main_akpr_i_run(data_prep, l_conc_mm_params,
                data_file_name, n_samp=number_samples, thin_by=thin_param,
                R_tot=tcr_number, results_file=results_file_akpr,
                cost_fct=cost_fct, fit_bounds=param_bounds,
                seed_sequence=seed_sequence, prior_dist="gaussian",
                emcee_kwargs=emcee_kwargs)

    ## TCR/TCR antagonism ratio fitting for 6F TCRs
    # Prepare data for fitting antagonism ratios
    df = pd.read_hdf(data_file_name_6f, key="df")
    data_prep = prepare_data_6f(df, l_conc_mm_params, pep_tau_map_ot1)
    seed_sequence = np.random.SeedSequence(0x9708cf1fb4e87d2db1ccb88c0163dcc3,
                    spawn_key=(0x23b68d3589d18ef7d20b8f7bd60d7ae,))

    # Arguments specific to wider bounds for AKPR model, 6F TCRs
    # Use fits from our regular MCMC simulation as priors
    number_samples = 10000
    thin_param = 8
    with open(os.path.join("..", "results", "mcmc",
        "mcmc_analysis_akpr_i.json"), "r") as h:
        analysis_6y = json.load(h)
    best_akpr_pvec = find_best_grid_point(analysis_6y)[1]
    # Means and variances, not really boundaries, but passed as this arg
    tcr_6f_stds = np.asarray([0.2, 1.0, 1.0, 1.0])
    param_bounds = [best_akpr_pvec, tcr_6f_stds**2]
    cost_fct = cost_antagonism_akpr_priors
    special_moves = [
        (emcee.moves.DEMove(gamma0=2/np.sqrt(2*len(best_akpr_pvec))), 0.6),
        (emcee.moves.DESnookerMove(gammas=1.5), 0.2),
        (emcee.moves.WalkMove(), 0.2)
    ]
    emcee_kwargs = {"moves": special_moves}

    # Main run
    if start_point <= 1:
        if  os.path.isfile(results_file_6f):
            raise RuntimeError(f"Existing {results_file_6f} found; "
                + "delete before running from start_point <= 1")
        else:
            print("\nStarting MCMC simulations on 4-ITAM TCR/TCR antagonism...")
        nsteps = main_tcr_tcr_run_6f(data_prep, l_conc_mm_params,
                data_file_name_6f, n_samp=number_samples, thin_by=thin_param,
                R_tot=tcr_number, results_file=results_file_6f,
                cost_fct=cost_fct, fit_bounds=param_bounds,
                seed_sequence=seed_sequence, prior_dist="gaussian",
                emcee_kwargs=emcee_kwargs
        )


    ## TCR/TCR antagonism analysis
    # Multiprocess
    all_processes = {"akpr_i":None, "6f":None}
    pool = multiprocessing.Pool(min(n_cpu, len(all_processes)))
    if start_point <= 2:
        for mod in all_processes.keys():
            filepath = os.path.join(folder_results, 
                analysis_kwarguments[mod]["analysis_res_fname"])
            if os.path.isfile(filepath):
                pool.close()
                raise RuntimeError(f"Existing {filepath} found; "
                    + "delete before running from start_point <= 2")
        
        print("\nStarting analysis of MCMC results for TCR/TCR antagonism...")
        for mod in all_processes.keys():
            # main_mcmc_analysis infers the proper cost fct
            #  from the analysis file name
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

    # Confidence intervals simulations
    confid_seedseq = np.random.SeedSequence(
        0x3fbd66e969c68721471e2ca0d8ee170,
        spawn_key=(0xc8683058625e84715546c9e5cb06dd7d,)
    )
    if start_point <= 3:
        if os.path.isfile(tcr_ci_res_fpath):
            raise RuntimeError(f"Existing {tcr_ci_res_fpath} found; "
                + "delete before running from start_point <= 3")
        else:
            print("\nStarting model CI generation for TCR/TCR antagonism...")
        main_tcr_tcr_confidence(folder_results, folder_results_for_plots,
            analysis_kwarguments, do_save=True, n_boot=1000,
            ci_res_fname=tcr_ci_res_fname, models_list=["akpr_i", "tcr_tcr_6f"],
            plotting_data_fname=tcr_plotting_data_fname,
            main_seedseq=confid_seedseq, molec_counts_fi=molec_counts_fi,
            mtc=mtc, tcell_type="OT1_Naive")

    ## TCR/CAR antagonism
    # Load best fits for TCR/TCR antagonism with revised AKPR model
    tcr_analysis_file = os.path.join(folder_results,
                    analysis_kwarguments["akpr_i"]["analysis_res_fname"])

    number_samples = 10000
    thin_param = 8
    # Number of TCR and CAR per T cell, CD19 per tumor, pulse KD, peptide taus
    mtc = "Geometric mean"
    res = load_tcr_car_molec_numbers(molec_counts_fi, mtc, tumor_type="E2aPBX_WT",
                        tcell_type="OT1_CAR", tumor_antigen="CD19")
    # Surface_nums contains [tcr_number, car_number, cd19_number]
    surface_nums = res[:3]
    l_conc_mm_params, pep_tau_map = res[3:]

    ## Antagonism ratio fitting
    # Prepare data for fitting antagonism ratios
    df = pd.read_hdf(data_file_name_car)
    chosen_fit_conc = ["1uM", "1nM"]
    data_prep = prepare_car_antagonism_data(df, l_conc_mm_params,
                    pep_tau_map, cyto="IL-2", do_plot=False, dropn4=True,
                    tcr_conc=chosen_fit_conc, tcr_itams="10", car_itams="3")

    file_names = [data_file_name_car, results_file_akpr, tcr_analysis_file]
    car_seedseq = np.random.SeedSequence(
        0x183afaabb0cb7be3edb1e7fa2d17f5f,
        spawn_key=(0xeecb6315d88ed7452ea5cb63dab6e7fd,)
    )
    # C_mth, I_th, gamma^T_C, gamma^C_T, tau^T, tau^C
    # C_mth: 100x more than TCR default, CAR poor at activating inhibition
    # I_th: again 0.01, no such threshold in the classical model
    # gamma_TC: 0.1, assuming some asymmetry where TCR is more efficient
    # gamma_CT: 10.0, assuming some asymmetry
    # tau^T: 3s, critical ligand
    # tau^C: 300 s, critical ligand, x100 for scale of Ab affinity
    tcr_car_priors = [50000.0, 0.01, 0.1, 10.0, 3.0, 300.0]
    tcr_car_priors = np.log10(np.asarray(tcr_car_priors))
    # Widths: first four are wide, critical taus are more well-defined
    tcr_car_prior_vari = np.asarray([2.0, 2.0, 1.0, 1.0, 0.5, 0.5])**2
    tcr_car_bounds = [tcr_car_priors, tcr_car_prior_vari]
    cost_fct_car = cost_antagonism_tcr_car_priors

    if start_point <= 4:
        if os.path.isfile(car_results_file):
            raise RuntimeError(f"Existing {car_results_file} found; "
                + "delete before running from start_point <= 4")
        else:
            print("\nStarting TCR/CAR antagonism MCMC runs...")
        nsteps = main_tcr_car_antagonism(data_prep, surface_nums,
            l_conc_mm_params, file_names, n_samp=number_samples,
            thin_by=thin_param, do_plot=False, tcr_conc="both_conc",
            seed_sequence=car_seedseq, results_car_file=car_results_file,
            cost_fct=cost_fct_car, fit_bounds=tcr_car_bounds,
            prior_dist="gaussian"
        )

    # Finally, analyze TCR/CAR results and generate predictions.
    if start_point <= 5:
        filepath = os.path.join(folder_results, car_analysis_fname)
        if os.path.isfile(filepath):
            raise RuntimeError(f"Existing {filepath} found; "
                + "delete before running from start_point <= 5")
        else:
            print("\nStarting analysis of MCMC simulations for TCR/CAR...")
        main_tcr_car_analysis(
            folder_results, do_save=True, do_show=False, tcr_conc=["1uM", "1nM"],
            analysis_res_fname=car_analysis_fname,
            samples_fname=car_results_fname,
            fit_summary_fname=car_summary_fname,
            figures_folder=car_fig_subfolder,
        )
    if start_point <= 6:
        fname = os.path.join(folder_results, car_predictions_fname)
        if os.path.isfile(fname):
            raise RuntimeError(f"Existing {fname} found; "
                + "delete before running from start_point <= 6")
        else:
            print("\nStarting generation of model predictions for TCR/CAR...")
        main_tcr_car_predictions(
            folder_results, do_save=True, do_show=False, n_boot=1000,
            fit_conc=["1uM", "1nM"],
            analysis_res_fname=car_analysis_fname,
            samples_fname=car_results_fname,
            figures_folder=predictions_figure_folder,
            predictions_res_fname=car_predictions_fname,
            analysis_6f=analysis_kwarguments["6f"]["analysis_res_fname"],
            results_6f=analysis_kwarguments["6f"]["results_fname"],
            main_seedseq=np.random.SeedSequence(
                0x23c68d3d82bbc82919286a0263454b80,
                spawn_key=(0x8e2bd466a72f7b9091f00e0599f6aca9,)
            ),
            plot_res_file=car_ci_fname,
            for_plots_folder=folder_results_for_plots,
            tumor_antigen="CD19",
            mtc="Geometric mean",
            molec_counts_fi=molec_counts_fi
        )
        save_z_correction_factors(folder_results, fit_conc=chosen_fit_conc,
            analysis_res_fname=car_analysis_fname,
            samples_fname=car_results_fname,
            predictions_res_fname=car_predictions_fname,
            factors_fname=car_zfactors_fname
        )
    # Save prior distribution parameters to a JSON file
    if start_point <= 7:
        print("\nSaving prior distributions of model parameters...")
        prior_params_dict = {
            "akpr": {
                "means": list(classic_akpr_pvec),
                "stds": list(akpr_prior_stds)
            },
            "6f": {
                "means": list(best_akpr_pvec),
                "stds": list(tcr_6f_stds)
            },
            "car":{
                "means": list(tcr_car_priors),
                "stds": list(np.sqrt(tcr_car_prior_vari))
            }
        }
        with open(os.path.join("..", "results", "for_plots_withpriors", 
                "mcmc_prior_parameters.json"), "w") as f:
            json.dump(prior_params_dict, f)
        print("Finished saving prior parameters to JSON:")
        print(prior_params_dict)
