""" To report accurately all statistics in the manuscript,
print how many experimental repeats and biological or technical
replicates are in each dataset used for MCMC or other fitting.

This means:
    - TCR/TCR antagonism, 6Y cells
    - TCR/TCR antagonism, 6F cells
    - TCR/CAR antagonism, 6Y or 6F cells.

To be sure I use exactly the same as during the actual fitting,
I read the data file name from MCMC results.

I also export all this to a JSON file, in results/for_plots/.

@author: frbourassa
June 14, 2023
"""
import numpy as np
import pandas as pd
import h5py, json, os

def prints_tcr_6y(df):
    # Level order: 'Experiment', 'Cytokine', 'Agonist', 'AgonistConcentration',
    # 'Antagonist', 'AntagonistConcentration'
    print("------ TCR/TCR, 6Y ------")
    df = df.loc[(s, "IL-2", "N4")]
    # Number of datasets
    print("Number of datasets:", len(df.index.get_level_values("Experiment").unique()))
    print("These are:")
    print(df.index.get_level_values("Experiment").unique())
    # Number of replicates per dataset and condition
    print("No Replicates/Spleen level; 1 replicate per condition per dataset")
    # Number of time points, per dataset.
    df_tpts = df.stack("Time").groupby(["Experiment", "AgonistConcentration", "Antagonist", "AntagonistConcentration"]).count()
    print("Number of timepoints:", df_tpts.min().iat[0], " - ", df_tpts.max().iat[0])
    # number of data points per condition
    df_np = (df.drop("0", level="AgonistConcentration").stack("Time")
        .groupby(["AgonistConcentration", "Antagonist", "AntagonistConcentration"]).count())
    print("n_p per condition:", df_np.min().iat[0], " - ", df_np.max().iat[0])
    return 0


def prints_tcr_6f(df):
    print("------ TCR/TCR, 6F ------")
    # 'Experiment', 'Cytokine', 'Genotype', 'Agonist', 'AgonistConcentration',
    # 'Antagonist', 'AntagonistConcentration', 'Spleen', 'Replicate'
    df = df.loc[(s, "IL-2", "6F", "N4")]  # We only used 6F here.
    try:
        df = df.drop("0", level="AgonistConcentration")
    except KeyError:
        pass
    print("Number of datasets:", len(df.index.get_level_values("Experiment").unique()))
    print("These are:")
    print(df.index.get_level_values("Experiment").unique())

    # Number of replicates and spleens per condition per dataset
    df_spleens_reps = df.index.to_frame().set_index("Experiment")
    print("Number of spleens per condition/dataset:")
    print(df_spleens_reps["Spleen"].groupby("Experiment").nunique())
    print("Number of replicates per condition/dataset:")
    print(df_spleens_reps["Replicate"].groupby("Experiment").nunique())

    # Number of time points, per dataset.
    df_tpts = df.stack("Time").index.to_frame().set_index("Experiment")
    print("Number of timepoints per dataset:")
    print(df_tpts["Time"].groupby("Experiment").nunique())

    # Number of data points n_p per condition, in the end
    df_np = df.stack("Time").groupby(["AgonistConcentration", "Antagonist", "AntagonistConcentration"]).count()
    print("Number of points per condition:", df_np.min(), "-", df_np.max())

    return 0


def prints_tcr_car(df):
    print("------ TCR/CAR ------")
    # Level order: 'Data', 'Cytokine', 'TCR_Antigen_Density', 'Tumor',
    # 'Spleen', 'TCR_ITAMs', 'CAR_ITAMs', 'TCR_Antigen', 'CAR_Antigen'
    df = df.loc[(s, "IL-2", s, "E2APBX")]

    # Number of datasets
    good_dsets = df.index.get_level_values("Data").unique()
    print("Number of datasets:", len(good_dsets))
    print("These are:")
    print(good_dsets)

    # Number of spleens per condition per dataset
    df_index = df.index.to_frame()
    print("Number of spleens per dataset:")
    print(df_index.set_index("Data").groupby("Data").nunique()["Spleen"])

    # Number of experiments having each condition
    print("Number of experiments having each condition:")
    print(df.loc[df.index.isin(["1", "3"], level="CAR_ITAMs")].index.to_frame()
            .set_index(["TCR_Antigen_Density", "TCR_ITAMs", "CAR_ITAMs"])
            .groupby(["TCR_ITAMs", "TCR_Antigen_Density", "CAR_ITAMs"])
            .nunique()["Data"])

    # Number of time points per dataset
    print("Number of time points in each dataset")
    print(df.stack("Time").index.to_frame().set_index("Data")
            .groupby("Data").nunique()["Time"])

    # Finally, number of data points per condition
    print("Data points per condition:")
    print(df.xs("CD19", level="CAR_Antigen").xs("V4", level="TCR_Antigen").stack("Time")
            .groupby(["TCR_ITAMs", "TCR_Antigen_Density", "CAR_ITAMs"])
            .count())

    return 0

if __name__ == "__main__":
    s = slice(None)  # Useful to slice
    # TCR/TCR, 6Y
    tcr_tcr_6y_mcmc = h5py.File("../results/mcmc/mcmc_results_akpr_i.h5", "r")
    filename = tcr_tcr_6y_mcmc.get("data").attrs.get("data_file_name")
    #prints_tcr_6y(pd.read_hdf(filename))
    tcr_tcr_6y_mcmc.close()
    print()

    # TCR/TCR, 6F
    tcr_tcr_6f_mcmc = h5py.File("../results/mcmc/mcmc_results_tcr_tcr_6f.h5", "r")
    filename = tcr_tcr_6f_mcmc.get("data").attrs.get("data_file_name")
    #prints_tcr_6f(pd.read_hdf(filename))
    tcr_tcr_6f_mcmc.close()
    print()


    # TCR/CAR, any construct. This is tricky because some double perturbations
    # are only present in one dataset. Maybe split per condition.
    tcr_car_mcmc = h5py.File("../results/mcmc/mcmc_results_tcr_car_both_conc.h5", "r")
    filename = tcr_car_mcmc.get("data").attrs.get("data_file_name")
    prints_tcr_car(pd.read_hdf(filename))
    tcr_car_mcmc.close()
