"""
Fit antagonism ratio data for CAR T cells responding to CD19 tumors
with 1 uM TCR antigen. Grid search integer parameters $m$, $f$, $k_S$
and for each, optimize $C_{m, thresh}$, $I_0$, and $\psi_0$.

Keeping separate main script for each model so we can launch MCMC in parallel
on different clusters.

@author: frbourassa
December 2022
"""

import numpy as np
import pandas as pd
import scipy as sp
from scipy import optimize
import h5py, json
import seaborn as sns
import matplotlib.pyplot as plt
from time import perf_counter
import emcee

# Local modules
import sys, os
if not "../" in sys.path:
    sys.path.insert(1, "../")

from mcmc.costs_tcr_car_antagonism import cost_antagonism_tcr_car
from mcmc.mcmc_run import grid_search_emcee
from models.tcr_car_akpr_model import steady_akpr_i_1ligand
from mcmc.utilities_tcr_car_antagonism import (
    prepare_car_antagonism_data,
    inverse_michaelis_menten,
    load_tcr_car_molec_numbers,
    load_tcr_tcr_akpr_fits
)
from mcmc.utilities_tcr_tcr_antagonism import geo_mean_apply
from utils.preprocess import (
    time_dd_hh_mm_ss,
    geo_mean_levels,
    loglog_michaelis_menten,
    inverse_michaelis_menten,
    write_conc_uM,
    ln10
)

### UTILITY FUNCTIONS ###

def plot_check_data(df, df_err, mm_params, do_save=False, do_show=False):
    # Temporary plot for checkup
    def reverse_mm_fitted(x):
        return inverse_michaelis_menten(x, *mm_params)

    df_model_data = (df.rename(reverse_mm_fitted, axis=0, level="TCR_Antigen_Density")
                        .rename(write_conc_uM, axis=0, level="TCR_Antigen_Density"))
    df_err_data = (df_err.rename(reverse_mm_fitted, axis=0, level="TCR_Antigen_Density")
                        .rename(write_conc_uM, axis=0, level="TCR_Antigen_Density"))
    palette = sns.color_palette("mako", n_colors=2)
    palette = {"1nM":palette[0], "1uM":palette[1]}
    g = sns.relplot(data=df_model_data.reset_index(), x="TCR_Antigen", y=df.name,
                hue="TCR_Antigen_Density", kind="line", marker="o", ms=8,
                palette=palette)
    for ax in g.axes.flat:
        ax.set_yscale("log", base=2)
        ax.axhline(1.0, ls="--", color="k", lw=1.0)
    nm_dat = df_model_data.name
    for conc in palette:
        df_err_loc2 = df_err_data.xs(conc, level="TCR_Antigen_Density")
        df_dat_loc2 = df_model_data.xs(conc, level="TCR_Antigen_Density")
        df_dat_log = np.log2(df_dat_loc2)
        # Compute linear scale error bars (asymmetric)
        # from symmetric log-scale error bars
        yup = 2**(df_err_loc2 + df_dat_log) - df_dat_loc2
        ylo = df_dat_loc2 - 2**(-df_err_loc2 + df_dat_log)
        yerr = np.vstack([ylo.values, yup.values])
        df_dat_loc2 = df_dat_loc2.reset_index()
        df_err_loc2 = df_err_loc2.reset_index()
        ax.errorbar(df_err_loc2["TCR_Antigen"], df_dat_loc2[nm_dat],
            xerr=None, yerr=yerr, ecolor=palette[conc], ls="none"
        )
    if do_save:
        g.fig.tight_layout()
        g.fig.savefig("../figures/data_plots/data_fit_tcr_car_cd19_antagonism_mean.pdf")
    if do_show:
        plt.show()
    plt.close()
    return None



### MAIN FUNCTIONS ###
def main_tcr_car_antagonism(data_list, surface_nums, mm_params, file_names,
                        n_samp, thin_by, do_plot=False, tcr_conc="1uM"):
    """ Returns the total number of steps taken """
    ## Data preparation. Unwrap input args
    df_fit, df_ci_log2 = data_list  # Processed antagonism data
    data_fname, res_file, analysis_file = file_names  # File names
    tcr_number, car_number, cd19_number = surface_nums  # surface molecules
    if do_plot:
        plot_check_data(df_fit, df_ci_log2, mm_params, do_save=False)

    ## Load/choose model parameters for 3-ITAM CAR, 10-ITAM TCR
    # Naive parameters for TCR. Use values fitted on TCR-TCR antagonism data
    tcr_loads = load_tcr_tcr_akpr_fits(res_file, analysis_file)
    # params: phi, kappa, cmthresh, S0p, kp, psi0, gamma_tt
    # Then, [N, m, f] and I_tot
    tcr_params, tcr_nmf, tcr_itot = tcr_loads

    ## Wrap parameters properly
    # Choose tau, and use experimentally calibrated L for CD19
    # tau 100x larger than TCR, since this is an antibody-target pair
    # In total, dissociation constant K_D = 1/kappa/tau should be 1000x smaller
    # than for TCR, since Ab affinities are in the nM range,
    # while TCR are in uM range. Harris 2018.
    cd19_tau_l = (500.0, cd19_number)

    # psi0 fixed because it is sloppy, ill-determined by data when MCMC'ed too.
    # Chosen to make sure that max. antagonism (psi = psi0) can give a FC
    # at least as small as the max. fold-change in the data, assuming CD19
    # only has psi = phi: this would require very large cmthresh^CAR,
    # so we make psi0 smaller than this limit to allow the CAR itself to
    # activate feedback.
    phi_car = tcr_params[0]*0.005   # phi < 100x slower for CAR than TCR, as suggested in Harris, 2018
    fcmin = df_fit.min()
    # psi0_factor = psi0*tau / (psi0*tau + 1) = FC_min * phi_factor
    # Invert: psi0 = psi0_factor / (1 - psi0_factor)
    psi0_factor_car = fcmin * phi_car / (1.0 + phi_car*cd19_tau_l[0])
    psi0_car = psi0_factor_car / (1.0 - psi0_factor_car*cd19_tau_l[0])

    # And then smaller, so cmthresh is not pushed too high
    # and there can still be antagonism at high CAR antigen densities.
    psi0_car /= 3.0

    # Fixed parameters for CAR: phi, kappa, gamma_cc
    # Fix gamma_cc = 1.0, because s_thresh_car can compensate it.
    # Faster binding, slower phosphorylation (Harris et al., 2018)
    car_params = [
        phi_car,
        tcr_params[1]*10.0,  # kappa 10x larger, so antigen's KD 1000x larger
        tcr_params[6],        # gamma_tt = gamma_cc = 1.0
        psi0_car
    ]
    # Wrapping up parameters. Fixed ones:
    # phi_tcr, kappa_tcr, cmthresh_tcr, S0p, kp, psi0_tcr, gamma_tt
    # then phi_car, kappa_car, gamma_cc, psi0_car
    tcr_car_params = tcr_params + car_params

    # Total R (both types) and S: fixed. R, S of TCR, then R of CAR
    tcr_rs = [tcr_number, tcr_itot]
    tcr_car_ritots = tcr_rs + [car_number]

    # N, m, f for 6Y TCR and 3-ITAM CAR: fixed ones, n_tcr, m_tcr, f_tcr, n_car
    tcr_car_nmf = tcr_nmf + [3]

    # if gamma_tt is not 1, adjust s_tresh
    if tcr_params[-1] != 1.0:
        raise NotImplementedError()

    ## MCMC setup
    # Parameter boundaries: log10 of C_m_thresh_car, I_thresh_car,
    # gamma_{TC}, gamma_{CT}, tau_c_tcr, tau_c_car
    # Limit gamma_TC to 1
    fit_bounds = [(1, 1000*tcr_car_ritots[2]), (1e-5, 1000.0*tcr_car_ritots[1])]
    fit_bounds += [(0.1, 1.0), (1e-2, 1e4), (1.0, 30.0), (50.0, 5e3)]
    # Rearrange as one array of lower, one array of upper bounds
    fit_bounds = [np.log10(np.asarray(a)) for a in zip(*fit_bounds)]

    # Fitted ones:
    fit_params_vec = (fit_bounds[0] + fit_bounds[1])/2
    fit_param_names = [r"$C^C_{m,th}$", r"$I^C_{th}$",
                       r"${\gamma^T}_C$", r"${\gamma^C}_T$",
                       r"$\tau^T_c$", r"$\tau^C_c$"]

    # Skip computing MI in the cost function
    cost_args = (tcr_car_params, tcr_car_ritots, tcr_car_nmf,
                cd19_tau_l, df_fit, df_ci_log2)
    # List only the names of cost function args to save to the hdf5 file.
    cost_args_names = ["fixed_rates", "tcr_car_ritots", "tcr_car_nmf", "cd19_tau_l"]
    print(tcr_car_params)
    print(tcr_car_ritots)
    print(tcr_car_nmf)
    print(cd19_tau_l)

    ## Fit with MCMC
    # Grid search over m, f, k_S of the CAR, running MCMC in each case
    # to adjust C_m_thresh_car, S_thresh_car, gamma_{TC}, gamma_{CT}, tau_c_tcr, tau_c_car
    # Note: this can take a long time; run on the physics cluster
    kmf_bounds = [(1, 2), (1, 3), (1, 3)]
    n_grid_pts = np.product([a[1]-a[0]+1 for a in kmf_bounds])
    nwalkers = 32

    # Use special moves for this higher-dimensional space.
    # Make steps a bit smaller than default to avoid missing narrow minima.
    moves = [
        (emcee.moves.DEMove(gamma0=2/np.sqrt(2*len(fit_params_vec))), 0.6),
        (emcee.moves.DESnookerMove(gammas=1.5), 0.2),
        (emcee.moves.WalkMove(), 0.2)
    ]

    seed_sequence = np.random.SeedSequence(0x183afaabb0cb7be3edb1e7fa2d17f5f,
                            spawn_key=(0xeecb6315d88ed7452ea5cb63dab6e7fd,))
    results_file = "../results/mcmc/mcmc_results_tcr_car_{}_test.h5".format(tcr_conc)
    try:
        start_t = perf_counter()
        grid_search_emcee(cost_antagonism_tcr_car,
            kmf_bounds, fit_bounds, results_file, p0=None, nwalkers=nwalkers,
            nsamples=n_samp, seed_sequence=seed_sequence, cost_args=cost_args,
            cost_kwargs={}, emcee_kwargs={"moves":moves}, param_names=fit_param_names,
            run_kwargs={"tune":True, "thin_by":thin_by})
    except Exception as e:
        os.remove(results_file)
        print("Carried exception over")
        print(e)
    else:
        print("Finished grid search without uncontrolled exception")
        end_t = perf_counter()
        delta_t = end_t - start_t
        nsteps = nwalkers * n_samp * thin_by * n_grid_pts
        print("Total run time for MCMC: {} s".format(time_dd_hh_mm_ss(delta_t)))
        print("Time per step: {} s".format(delta_t / nsteps))
        # Add the cost function args to the results file, except the data.
        # Add the name of the data file, though.
        results_obj = h5py.File(results_file, "r+")
        args_group = results_obj.create_group("data")
        args_group.attrs["cost_args_names"] = cost_args_names
        args_group.attrs["data_file_name"] = data_fname
        args_group.attrs["thin_by"] = thin_by
        args_group.attrs["run_time"] = delta_t
        args_group.attrs["nsteps"] = nsteps
        for i in range(len(cost_args_names)):
            args_group.create_dataset(cost_args_names[i], data=cost_args[i])
        # Add the L-pulse conversion parameters too.
        args_group.create_dataset("l_conc_mm_params", data=mm_params)
        results_obj.close()

    return nsteps


if __name__ == "__main__":
    # The number of steps taken is thin_param times number_samples
    # As long as number_samples is below 10^5 there should be no RAM issues
    # For larger number_samples, will need to use HDF5 backend of emcee.
    # Run for a short time first, see if it works kind of OK...
    number_samples = 10000
    thin_param = 8

    # Number of TCR and CAR per T cell, CD19 per tumor, pulse KD, peptide taus
    molec_counts_fi = "../data/surface_counts/surface_molecule_summary_stats.h5"
    mtc = "Geometric mean"
    res = load_tcr_car_molec_numbers(molec_counts_fi, mtc, tumor_type="E2aPBX_WT",
                        tcell_type="OT1_CAR", tumor_antigen="CD19")
    # Surface_nums contains [tcr_number, car_number, cd19_number]
    surface_nums = res[:3]
    l_conc_mm_params, pep_tau_map = res[3:]

    # Load TCR best fit parameters
    tcr_results_file = "../results/mcmc/mcmc_results_akpr_i.h5"
    tcr_analysis_file = "../results/mcmc/mcmc_analysis_akpr_i.json"

    ## Antagonism ratio fitting
    # Prepare data for fitting antagonism ratios
    data_file_name = "../data/antagonism/combined-OT1_CAR-dataframe_2022-01-19.hdf"
    df = pd.read_hdf(data_file_name)
    chosen_fit_conc = ["1uM", "1nM"]
    data_prep = prepare_car_antagonism_data(df, l_conc_mm_params,
                    pep_tau_map, cyto="IL-2", do_plot=False, dropn4=True,
                    tcr_conc=chosen_fit_conc, tcr_itams="10", car_itams="3")

    file_names = [data_file_name, tcr_results_file, tcr_analysis_file]
    nsteps = main_tcr_car_antagonism(data_prep, surface_nums, l_conc_mm_params,
                file_names, number_samples, thin_param,
                do_plot=False, tcr_conc="both_conc")
