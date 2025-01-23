"""
Plotting script to show that the condition 1nM agonist, 1nM antagonist
but antagonist=None is off in the experiment SingleCell_Antagonism_3.
Probably due to this condition being on the corner of plates and thus drying
up a little. This condition is biologically identical to 1uM antagonist,
antagonist=None, so when using this data for model fitting,
we replace the former by the latter in every dataset for consistency.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

if __name__ == "__main__":
    df = pd.read_hdf("allManualSameCellAntagonismDfs_v3.h5")
    df = df.xs("N4", level="Agonist")
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        df = df.stack()
    df.columns = pd.Index(["Concentration"])
    print(df)
    df = df.xs("1nM", level="AgonistConcentration")
    df = df.xs("SingleCell_Antagonism_3", level="Experiment", drop_level=False)

    g = sns.relplot(data=df.reset_index(), x="Time", y="Concentration",
                    hue="Antagonist", style="Experiment",
                    row='AntagonistConcentration', col="Cytokine",
                    kind="line", height=3, facet_kws={"margin_titles":True})
    for ax in g.axes.flat:
        ax.set_yscale("log")
    plt.show()
    plt.close()
