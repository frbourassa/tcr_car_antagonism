"""
Module with a wrapper around emcee's MCMC sampler that allows it to be
called with a cost function requiring extra arguments.
Also includes general analysis functions,
not specific to antagonism cost functions.

These arguments may include a list of discrete parameters over which a
grid search is performed; code the cost function accordingly.

Also includes a main function launching MCMC runs in parallel in a
grid search over discrete parameters.

@author: frbourassa
November 2022
"""
import numpy as np
import emcee
import h5py

# Multiprocessing modules
import itertools
import multiprocessing
from utils.cpu_affinity import count_parallel_cpu
n_cpu = count_parallel_cpu()


### MCMC RUN FUNCTIONS ###
def run_emcee(cost, pbounds, p0=None, nwalkers=32, nsamples=1000,
            cost_args=(), cost_kwargs={}, run_id=None,
            rgen=None, emcee_kwargs={}, run_kwargs={}):
    """
    cost: should take pvec, pbounds as first two arguments.
        Call signature: cost(pvec, pbounds, *args, **kwargs)
    emcee_kwargs: should contain in particular args and kwargs that
        are passed to the cost function after the vector of parameters.
    Run results are stored to a file.
    """
    # Get number of dimensions
    ndim = len(pbounds[0])

    # If rgen is none, create one
    if rgen is None:
        rgen = np.random.default_rng()
    # If p0 is None, create initial parameter values scattered over space
    if p0 is None:
        # limit to [0.05, 0.95] times the extent of each parameter interval
        extents = (pbounds[1] - pbounds[0]).reshape(1, -1)
        p0 = 0.9 * extents * rgen.random(size=(nwalkers, len(pbounds[0])))
        p0 = p0 + 0.05*extents + pbounds[0]
    # Prepare initial state
    state_init = emcee.State(p0, random_state=rgen)

    # Treat emcee kwargs and run kwargs. Make sure rgen, cost_args, cost_kwargs
    # are not duplicated in these kwargs, but do not add stuff to these
    # dictionaries because that could create side effects in multiprocessing.
    emcee_kwargs.pop("args", cost_args)
    emcee_kwargs.pop("kwargs", cost_kwargs)
    emcee_kwargs.pop("nwalkers", nwalkers)  # This shouldn't be a keyword argument
    run_kwargs.pop("nsamples", nsamples)  # Neither this
    run_kwargs.pop("rstate0", rgen)

    # Force the progress bar to be hidden unless specified otherwise
    run_kwargs.setdefault("progress", False)

    # Create an EnsembleSampler with extra args and kwargs
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, cost, args=(pbounds,) + cost_args, kwargs=cost_kwargs,
        **emcee_kwargs
    )
    # Run MCMC
    # TODO: here, use the .sample method and check autocorrelation regularly.
    sampler.run_mcmc(
        state_init, nsamples, **run_kwargs
    )

    # Return relevant results from the run
    return (sampler.get_chain(), sampler.get_log_prob(),
            sampler.acceptance_fraction, run_id)


def grid_search_emcee(cost, grid_bounds, pbounds, results_fname,
        p0=None, nwalkers=32, nsamples=1000, seed_sequence=None, cost_args=(),
        cost_kwargs={}, emcee_kwargs={}, run_kwargs={"tune":True}, **kwds):
    """
    cost: should take pvec, pgrid, pbounds as first three arguments.
        pgrid is prepended to extra args passed to cost after pvec, pbounds.
        Call signature: cost(pvec, pbounds, pgrid, *args, **kwargs)
    Run results are stored to a file.
    """
    # Avoid having each process opening and writing to the same HDF5 file,
    # because that is not allowed. Instead, open one file here and define
    # here a callback function writing the results to that file
    # which will be applied to the result of each process
    # In other languages we would have needed to write to separate files,
    # or keep all in RAM before writing at the end.
    # NB: that may not work after all. In that case, write to separate files.
    results_file = h5py.File(results_fname, "w")
    samples_group = results_file.create_group("samples")
    cost_group = results_file.create_group("cost")

    # Add metadata to the results file
    param_names = kwds.get("param_names",
                    ["p_{}".format(i) for i in range(len(pbounds[0]))])
    samples_group.attrs["param_names"] = param_names
    samples_group.attrs["param_bounds"] = pbounds
    samples_group.attrs["p0"] = p0 if p0 is not None else []


    samples_group.attrs["grid_bounds"] = grid_bounds
    for g in [samples_group, cost_group]:
        g.attrs["n_walkers"] = nwalkers
        g.attrs["n_samples"] = nsamples

    # Up to the user to add more specialized metadata/data to the files

    # Call back function to save returns
    def callback(result):
        # Get the result being returned.
        samples, costs, acpt, run_id = result
        # Reshape samples so the first axis is the variable
        # get_chain() returns array indexed [sample, walker, var]
        # while we want [var, walker, sample]
        samples = np.moveaxis(samples, source=[0,2], destination=[2,0])
        costs = np.moveaxis(costs, source=1, destination=0)

        # Save to the file opened above at the appropriate key.
        dset = samples_group.create_dataset(str(run_id), data=samples)
        cost_group.create_dataset(str(run_id), data=costs)
        # Add, as metadata, acceptance fraction
        dset.attrs["run_id"] = run_id
        dset.attrs["acceptance_fraction"] = acpt
        # Print some info, return ik
        print("Fitted {}".format(run_id))
        return run_id

    # Error callback
    def error_callback(excep):
        print()
        print(excep)
        print()
        return -1

    # Loop over grid points
    grid = list(itertools.product(*[range(a, b+1) for a, b in grid_bounds]))
    all_processes = []
    if seed_sequence is None:
        seed_sequence = np.random.SeedSequence()
    seeds = seed_sequence.spawn(len(grid))

    # Launch MCMC for each grid point
    pool = multiprocessing.Pool(min(n_cpu, len(grid)))
    all_returns = []
    for gpt in grid:
        rgen_loc = np.random.default_rng(seeds.pop())
        cost_args_loc = (gpt,) + cost_args
        apply_kw = {"p0":p0, "nwalkers":nwalkers, "nsamples":nsamples,
                        "cost_args":cost_args_loc, "cost_kwargs":cost_kwargs,
                        "rgen":rgen_loc, "emcee_kwargs":emcee_kwargs,
                        "run_kwargs":run_kwargs, "run_id":gpt}
        ret = pool.apply_async(run_emcee, args=(cost, pbounds), kwds=apply_kw,
            callback=callback, error_callback=error_callback)
        all_returns.append(ret)

    for p in all_returns:
        try:
            p.get()
        # Exception has already been handled by the callback_error
        # so avoid printing the message twice.
        except RuntimeError:
            pass

    # Close Pool and results file.
    results_file.close()
    pool.close()

    return 0
