""" Analysis tools producing generic plots to assess the convergence
and quality of parameter estimation algorithm results, like MCMC.

@author: frbourassa
May 2022
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import corner

import matplotlib.text as mtext
from matplotlib.legend_handler import HandlerBase
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import itertools


# For adding subtitles in legends.
# Artist class: handle, containing a string
class LegendSubtitle(object):
    def __init__(self, message, **text_properties):
        self.text = message
        self.text_props = text_properties
        text_len = max(map(len, self.text.split("\n")))
        self.labelwidth = " "*int(text_len*1.33) + "\n"*self.text.count("\n")
    def get_label(self, *args, **kwargs):
        return self.labelwidth  # no label, the artist itself is the text

# Handler class, give it text properties
class LegendSubtitleHandler(HandlerBase):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle.text, size=fontsize,
                    ha="left", va="center", **orig_handle.text_props)
        handlebox.add_artist(title)
        # Make the legend box wider if needed
        return title


# Utility for log ticks in scientific notation
def change_log_ticks(axis, base=2, which="y"):
    if which == "y" or which == "both":
        ogyticks = axis.get_yticks()
        newyticks = list(np.unique([int(x) for x in ogyticks]))
        newyticklabels = [str(base)+'$^{'+str(x)+'}$' for x in newyticks]
        axis.set_yticks(newyticks)
        axis.set_yticklabels(newyticklabels)
    if which == "x" or which == "both":
        ogxticks = axis.get_xticks()
        newxticks = list(np.unique([int(x) for x in ogxticks]))
        newxticklabels = [str(base)+'$^{'+str(x)+'}$' for x in newxticks]
        axis.set_xticks(newxticks)
        axis.set_xticklabels(newxticklabels)
    return axis


### Automatic style generation functions
# Replace old core seaborn functions, which were moved
# around in the package since then
def unique_markers(n):
    """Build an arbitrarily long list of unique marker styles for points.
    Parameters
    ----------
    n : int
        Number of unique marker specs to generate.
    Returns
    -------
    markers : list of string or tuples
        Values for defining :class:`matplotlib.markers.MarkerStyle` objects.
        All markers will be filled.
    """
    # Start with marker specs that are well distinguishable
    markers = [
        "o",
        "s",
        "X",
        (4, 0, 0),
        "^",
        "v",
        (4, 1, 45),
    ]

    # Now generate more from regular polygons of increasing order
    s = 5
    while len(markers) < n:
        a = 360 / (s + 1) / 2
        markers.extend([
            (s + 1, 1, a),
            (s + 1, 0, a),
            (s, 1, 0),
            (s, 0, 0),
        ])
        s += 1

    # Convert to MarkerStyle object, using only exactly what we need
    # markers = [mpl.markers.MarkerStyle(m) for m in markers[:n]]

    return markers[:n]


def unique_dashes(n):
    """Build an arbitrarily long list of unique dash styles for lines.
    Parameters
    ----------
    n : int
        Number of unique dash specs to generate.
    Returns
    -------
    dashes : list of strings or tuples
        Valid arguments for the ``dashes`` parameter on
        :class:`matplotlib.lines.Line2D`. The first spec is a solid
        line (``""``), the remainder are sequences of long and short
        dashes.
    """
    # Start with dash specs that are well distinguishable
    dashes = [
        "",
        (4, 1.5),
        (1, 1),
        (3, 1.25, 1.5, 1.25),
        (5, 1, 1, 1),
    ]

    # Now programatically build as many as we need
    p = 3
    while len(dashes) < n:

        # Take combinations of long and short dashes
        a = itertools.combinations_with_replacement([3, 1.25], p)
        b = itertools.combinations_with_replacement([4, 1], p)

        # Interleave the combinations, reversing one of the streams
        segment_list = itertools.chain(*zip(
            list(a)[1:-1][::-1],
            list(b)[1:-1]
        ))

        # Now insert the gaps
        for segments in segment_list:
            gap = min(segments)
            spec = tuple(itertools.chain(*((seg, gap) for seg in segments)))
            dashes.append(spec)

        p += 1

    return dashes[:n]


### General legend generation functions ###

def handles_properties_legend(hues, styles, sizes):
    """ Creates handles (with labels) for a categorical legend with a section
    for hues, one for styles and markers, one for sizes.
    Pass None to skip that legend section.
    Args:
        hues (tuple): hue_title, hue_entries
            hue_entries: dictionary of label: hue
        styles (tuple): style_title, style_entries
            style_entries: dictionary of label: (linestyle, marker)
        sizes (tuple): size_title, size_entries
            size_entries: dictionary of label: (linesize, markersize)
    Returns:
        legend_handles
        handler_map

    """
    legend_handles = []
    legend_handler_map = {LegendSubtitle: LegendSubtitleHandler()}

    # Hues section
    if hues is not None:
        hue_title, hue_dict = hues
        legend_handles.append(LegendSubtitle(hue_title))
        for lbl in hue_dict:
            legend_handles.append(Line2D([0], [0], color=hue_dict[lbl],
                label=lbl, linestyle="-"))

    # Line styles section
    if styles is not None:
        style_title, style_dict = styles
        legend_handles.append(LegendSubtitle(style_title))
        for lbl in style_dict:
            legend_handles.append(Line2D([0], [0], color="k", label=lbl,
                linestyle=style_dict[lbl][0], marker=style_dict[lbl][1],
                mfc="k", mec="k"))
    if sizes is not None:
        size_title, size_dict = sizes
        legend_handles.append(LegendSubtitle(size_title))
        for lbl in size_dict:
            legend_handles.append(Line2D([0], [0], color="k", label=lbl,
                linestyle="-", marker="o", mfc="k", mec="k",
                linesize=size_dict[lbl][0], markersize=size_dict[lbl][1]))
    return legend_handles, legend_handler_map


def prepare_hues(df, lvl, sortkws={}, colmap=None):
    """ Prepare color palette for the entries in the level lvl of
    df. Sort them according to sortkey. Use colors from colmap.
    """
    try:
        hue_entries = list(df.index.get_level_values(lvl).unique())
    except ValueError:
        palette = {}
        hue_entries = []
    else:
        hue_entries = sorted(hue_entries, **sortkws)
        palette = sns.color_palette(colmap, n_colors=len(hue_entries))
        palette = {hue_entries[i]:palette[i] for i in range(len(hue_entries))}
    return hue_entries, palette


def prepare_styles(df, lvl, sortkws={}):
    """ Prepare dictionary of line styles for the entries in the level lvl
    of df. Sort them according to sortkey.
    """
    try:
        style_entries = list(df.index.get_level_values(lvl).unique())
    except KeyError:
        stdict = {}
        style_entries = []
    else:
        style_entries = sorted(style_entries, **sortkws)
        stdict = unique_dashes(len(style_entries))
        stdict = {style_entries[i]:(0, stdict[i]) for i in range(len(style_entries))}
    return style_entries, stdict


def prepare_markers(df, lvl, sortkws={}):
    """ Prepare dictionary of line styles for the entries in the level lvl
    of df. Sort them according to sortkey.
    """
    try:
        marker_entries = list(df.index.get_level_values(lvl).unique())
    except KeyError:
        markdict = {}
        marker_entries = []
    else:
        marker_entries = sorted(marker_entries, **sortkws)
        markdict = unique_markers(len(marker_entries))
        markdict = {marker_entries[i]:markdict[i] for i in range(len(marker_entries))}
    return marker_entries, markdict


def prepare_subplots(df, row_lvl, col_lvl, sortkws_row={},
                sortkws_col={}, **kwargs):
    """ Extra kwargs are passed to plt.subplots """
    try:
        row_vals = list(df.index.get_level_values(row_lvl).unique())
    except KeyError:
        nrows = 1
        row_vals = []
    else:
        row_vals = sorted(row_vals, **sortkws_row)
        nrows = len(row_vals)

    try:
        col_vals = list(df.index.get_level_values(col_lvl).unique())
    except KeyError:
        ncols = 1
        col_vals = []
    else:
        col_vals = sorted(col_vals, **sortkws_col)
        ncols = len(col_vals)
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    # Make sure axes is a 2d array
    axes = np.atleast_2d(axes)
    return row_vals, col_vals, fig, axes


### Specific legend creation function for data vs model fits
def data_model_handles_legend(palette, markers, styles, palette_name,
            model_style="-", model_lw=2.0, data_size=8, data_marker="o"):
    """ Prepare a list of legend handles with legend Subtitles,
    return a legend handler map too

    Args:
        palette (dict): {condition: color} dictionary
        markers (dict): {condition: marker} dictionary of data points
        styles (str): line style of model curves
        palette_name (str): name of the property delineated
            by palette/markers/styles

    Note: the palette, markers, and styles dictionaries are supposed to
        have the same keys.

    Returns:
        legend_handes, legend_handler_map
    """
    legend_handles = []
    legend_handler_map = {LegendSubtitle: LegendSubtitleHandler()}

    # First, clarify data vs model
    legend_handles.append(Line2D([0], [0], ls="none", marker=data_marker,
                mfc="grey", mec="grey", label="Data", ms=data_size))
    legend_handles.append(Line2D([0], [0], ls=model_style,
                color="grey", marker=None, label="Model", lw=model_lw))

    # Colors, linestyles and markers together
    # TODO: ideally, the marker and line would be separated, e.g. with marker
    # above the line, rather than stacked.
    legend_handles.append(LegendSubtitle(palette_name))
    for k in palette.keys():
        color = palette.get(k)
        legend_handles.append(Line2D([0], [0], ls=styles.get(k),
            marker=markers.get(k), color=color, mfc=color,
            mec=color, ms=data_size, lw=model_lw, label=k))

    return legend_handles, legend_handler_map


### Corner plots for model parameter posterior distributions
def share_axis(axes, idx, which="x", reverse=False):
    """ Force sharing of an axis (e.g. after putting log-scale ticks) between
    an axis and all other axes of the same row (which=="y") or
    column (which=='y'), taking into account whether the corner plots are in
    the lower left (reverse == False) or upper right half of the grid
    (reverse == True).

    idx: tuple of 2 ints, the reference axis to share
    """
    iref, jref = idx
    axref = axes[iref, jref]

    if which == "x" or which == "both":  # Same column
        rng = range(0, jref) if reverse else range(jref, axes.shape[0])
        for i in rng:
            if i != iref:
                axes[i, jref].sharex(axref)
                axes[i, jref].tick_params(axis="x", labelbottom=False)
    if which == "y" or which == "both":
        rng = range(iref, axes.shape[1]) if reverse else range(0, iref)
        for j in rng:
            if j != jref:
                axes[iref, j].sharey(axref)
                axes[iref, j].tick_params(axis="y", labelleft=False)
    return axes


# For log-transformed data, manually put back log-scale ticks
def put_log_ticks(ax, axis="both", basis=10, max_ticks=5):
    """ axis: either "x", "y", or "both" (default). """
    # Do x axis
    max_minor = max_ticks - 1
    if axis in ["x", "both"]:
        xlims = ax.get_xlim()
        xmajor = np.arange(np.floor(xlims[0]), np.ceil(xlims[1])+1, 1, dtype="int")
        xmajor_visible = xmajor[np.logical_and(xmajor <= xlims[1], xmajor >= xlims[0])]
        xmajor_visible2 = xmajor_visible[::len(xmajor_visible)//max_ticks+1]
        if len(xmajor_visible) == 0:
            xlims = [xmajor[0], xmajor[1]]
            xmajor_visible = np.asarray([xmajor[0], xmajor[1]])
            xmajor_visible2 = xmajor_visible
        ax.set_xticks(
            ticks=xmajor_visible2,
            labels=[r"$10^{" + str(k) + "}$" for k in xmajor_visible2]
        )

        # If less than 3 decades, put minor ticks too
        if len(xmajor_visible) < max_minor and basis > 2:
            decades = np.arange(2.0, basis, dtype="float")
            xminor = []
            for j in range(len(xmajor)):
                xminor.append(np.log(10.0**xmajor[j] * decades)/np.log(basis))
            xminor = np.concatenate(xminor)
            ax.set_xticks(xminor, minor=True)
        # Put back original limits
        ax.set_xlim(xlims)

    # Do y axis
    if axis in ["y", "both"]:
        ylims = ax.get_ylim()
        ymajor = np.arange(np.floor(ylims[0]), np.ceil(ylims[1])+1, 1, dtype="int")
        ymajor_visible = ymajor[np.logical_and(ymajor <= ylims[1], ymajor >= ylims[0])]
        ymajor_visible2 = ymajor_visible[::len(ymajor_visible)//max_ticks+1]
        if len(ymajor_visible) == 0:
            ylims = [ymajor[0], ymajor[1]]
            ymajor_visible = np.asarray(ylims)
            ymajor_visible2 = ymajor_visible
        ax.set_yticks(
            ticks=ymajor_visible2,
            labels=[r"$10^{" + str(k) + "}$" for k in ymajor_visible2],
            rotation=0
        )
        # If less than 3 decades, put minor ticks too
        if len(ymajor_visible) < max_minor and basis > 2:
            decades = np.arange(2.0, basis, dtype="float")
            yminor = []
            for j in range(len(ymajor)):
                yminor.append(np.log(10.0**ymajor[j] * decades)/np.log(basis))
            yminor = np.concatenate(yminor)
            ax.set_yticks(yminor, minor=True)
        # Put back original limits
        ax.set_ylim(ylims)

    return ax


# Corner plots for analysis of MCMC simulations, not as polished as published,
# and takes actual samples and analysis results rather than file names
# Keep this one to a minimum, just making corner plots, leave annotation to
# the calling function
def corner_plot_mcmc_analysis(
        df_samples, analysis_res, pdims,
        sizes_kwargs={}, line_props_kwargs={}, **kwargs
    ):
    """
    Args:
        df_samples (pd.DataFrame): MCMC run samples, burn-in already dropped
            and thinning applied, for some condition k, m, f,
            with columns containing parameter names
        analysis_res (dict): MCMC analysis results for some kmf
        pdims (dict): dimensions of panels
        sizes_kwargs (dict): things like scaleup, small_lw, truth_lw,
            small_markersize.
        line_props_kwargs (dict): for annotation of best parameter estimates,
            lists: map_colors, linestyles, markers;
            dict: strat_names_map

    Other kwargs are passed to corner.corner.

    Returns:
        fig (matplotlib.figure.Figure): cornerplot figure
    """
    # Labels and line styles
    param_names = df_samples.columns.values
    param_labels = list(map(lambda a: a.replace(r"\log ", ""), param_names))
    param_labels = list(map(lambda a: a.replace("thresh", "th"), param_labels))
    colors = line_props_kwargs.get("map_colors",
            ["xkcd:cornflower", "xkcd:sage", "xkcd:salmon", "xkcd:mustard"]
    )
    linestyles = line_props_kwargs.get("linestyles", ["-", "-.", "--", ":"])
    markers = line_props_kwargs.get("markers", ["o", "s", "^", "*"])
    strat_names_map = line_props_kwargs.get("strat_names_map",
        {"MAP hist": "MAP marginal"}
    )

    # Bins for histograms: doane
    doane_bins = np.histogram_bin_edges(df_samples.iloc[:, 0], bins="doane").size - 1
    pvec_best = np.asarray(analysis_res.get("param_estimates").get("MAP best"))

    # Make the corner plot, using aesthetical parameters in sizes_kwargs
    scaleup = sizes_kwargs.get("scaleup", 1.0)
    small_lw = sizes_kwargs.get("small_lw", 0.8) * scaleup
    truth_lw = sizes_kwargs.get("truth_lw", 1.25) * scaleup
    small_markersize = sizes_kwargs.get("small_markersize", 1.0) * scaleup
    reverse_plots = False
    labelpad = sizes_kwargs.get("labelpad", len(pvec_best)**3/100.0)

    # Corner plot
    hist2d_kwargs = {"contour_kwargs":{"linewidths":small_lw},
                     "data_kwargs":{"ms":small_markersize}}
    fig = corner.corner(
        data=df_samples.values,
        labels=param_labels,
        reverse=reverse_plots,
        # Plot truths manually below to control line width
        nbins=doane_bins,
        labelpad=labelpad,
        hist_kwargs={"linewidth": small_lw},
        **hist2d_kwargs,
        **kwargs
    )
    # Annotate MAP estimates as horizontal lines.
    leg_handles = []
    for k, strat in enumerate(analysis_res["param_estimates"]):
        pvec_best = analysis_res["param_estimates"][strat]
        truth_color = colors.pop(0)
        colors.append(truth_color)
        truth_ls = linestyles.pop(0)
        linestyles.append(truth_ls)
        truth_marker = markers.pop(0)
        markers.append(truth_marker)
        strat_nice_name = strat_names_map.get(strat, strat)
        corner.overplot_lines(fig, pvec_best, reverse=reverse_plots,
            color=truth_color, lw=truth_lw, ls=truth_ls
        )
        corner.overplot_points(
            fig, [[p for p in pvec_best]], reverse=reverse_plots,
            color=truth_color, ms=2.5*small_markersize, marker=truth_marker
        )
        leg_handles.append(Line2D([0], [0], label=strat_nice_name,
            ls=truth_ls, lw=truth_lw, color=truth_color, marker=truth_marker
        ))
    n = sizes_kwargs.get("n_times_height", len(param_labels))
    mx = sizes_kwargs.get("n_extra_x_labels", 2)
    my = sizes_kwargs.get("n_extra_y_labels", 0)

    # Data is log-transformed, put back log ticks and labels
    # Treat y axes of the leftmost column, excluding first which is histogram
    gs = fig.axes[0].get_gridspec()
    axes = np.asarray(fig.axes).reshape(gs.nrows, gs.ncols)
    for i in range(1, axes.shape[0]):
        ax = axes[i, 0]
        put_log_ticks(ax, axis="y", max_ticks=4)
        share_axis(axes, (i, 0), which="y", reverse=reverse_plots)
        ax.set_ylabel(param_names[i], labelpad=labelpad)
    axes[0, 0].set_ylabel("Marginal", labelpad=labelpad + 10.0)

    # Treat x axes of the bottom row, including last
    max_x_ticks = 5 if axes.shape[1] < 5 else 4
    for j in range(axes.shape[1]):
        ax = axes[-1, j]
        put_log_ticks(ax, axis="x", max_ticks=max_x_ticks)
        share_axis(axes, (axes.shape[0]-1, j), which="x", reverse=reverse_plots)
        ax.set_xlabel(param_names[j])

    fig.legend(
        handles=leg_handles, title="Parameter estimate",
        bbox_to_anchor=(1, 1), loc="upper right"
    )

    fig.set_size_inches(pdims["panel_width"]*n + pdims["axes_label_width"]*mx,
                       pdims["panel_height"]*n + pdims["axes_label_width"]*my)
    fig.tight_layout()
    return fig


def average_time_series(t, y):
    # Use the trapeze rule with potentially unequal time intervals
    dt = np.diff(t)
    return np.sum((y[1:] + y[:-1])*dt*0.5) / (t.max() - t.min())


if __name__ == "__main__":
    # Test corner plot with dummy data
    import pandas as pd
    vars = ["w", "x", "y", "z"]
    numbers, letters = np.arange(2000), list("abcd")
    idx = pd.MultiIndex.from_product([numbers, letters],
            names=["Number", "Letter"])
    cols = pd.Index(vars, name="Variable")
    rgen = np.random.default_rng(seed=0xf6478c6135e8ff7cfca9cacb35ff916b)
    arr = rgen.standard_normal(size=(len(idx), len(cols)))
    dat = pd.DataFrame(arr, index=idx, columns=cols)

    dummy_analysis = {
        "param_estimates":{
            "MAP best": 0.2*np.random.standard_normal(size=len(vars)),
            "MAP hist": 0.2*np.random.standard_normal(size=len(vars))
        }
    }
    panel_dims = {
        "panel_width": 3.0,
        "panel_height": 3.0,
        "axes_label_width": 0.25
    }
    sizes_args = {
        "scaleup": 1.0,
        "small_lw": 1.0,
        "truth_lw": 1.5,
        "small_markersize": 1.5,
        "labelpad": 0.1
    }
    corner_plot_mcmc_analysis(
        dat, dummy_analysis, panel_dims, sizes_args
    )
    plt.show()
    plt.close()
