""" Analysis tools producing generic plots to assess the convergence
and quality of parameter estimation algorithm results, like MCMC.

@author: frbourassa
May 2022
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy import stats

import matplotlib.text as mtext
from matplotlib.legend_handler import HandlerBase
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import itertools

import corner, h5py, json

from scripts.preprocess import read_conc_uM, write_conc_uM
from scripts.analysis import autocorr_avg, autocorr_func_avg


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
    #axes = np.atleast_2d(axes)
    if ncols == 1:
        axes = axes[:, None]
    if nrows == 1:
        axes = axes[None, :]
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


def plot_tcr_tcr_fits(plot_res_file, model_suffix, kmf, pdims, pert_palette,
                    col_wrap=3, level="supplement"):
    """ level: "main" or "supplement". "main" means simpler labels. """
    # Read data to plot first
    df_model = pd.read_hdf(plot_res_file, key="model").xs(model_suffix).xs(kmf).sort_index()
    df_data = pd.read_hdf(plot_res_file, key="data").xs(model_suffix).sort_index()
    df_ci = pd.read_hdf(plot_res_file, key="ci").xs(model_suffix).sort_index()

    # Find how many plots we will need = number of tcr antagonist antigen densities
    available_agconc = df_model.index.get_level_values("AgonistConcentration").unique()
    available_agconc = sorted(list(available_agconc), key=read_conc_uM)

    # Prepare plots, using col_wrap
    n_plots = len(available_agconc)
    n_cols = min(n_plots, col_wrap)
    n_rows = n_plots // col_wrap + min(n_plots % col_wrap, 1)
    fig, axes_grid = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    fig.set_size_inches(n_cols*pdims["panel_width"] + pdims["axes_label_width"],
                       n_rows*(pdims["panel_height"] + pdims["axes_label_width"]))  # For titles
    axes = axes_grid.flatten()
    # Leave the top left corner empty if possible
    if n_rows * n_cols > n_plots:
        axes = axes[n_rows*n_cols - n_plots:]
    # Remove last axes, leave blank
    for i in range(n_rows*n_cols - n_plots):
        axes_grid.flat[i].set_axis_off()

    # Prepare color palette
    available_antagconc = df_model.index.get_level_values("AntagonistConcentration").unique()
    available_antagconc = sorted(list(available_antagconc), key=read_conc_uM, reverse=True)
    # 1uM is first, make it black if supplementary plot
    #if level == "main":
    #    palette = sns.dark_palette(pert_palette["AgDens"],
    #                               n_colors=len(available_antagconc))
    palette = sns.dark_palette(pert_palette["AgDens"],
                               n_colors=len(available_antagconc))
    palette[0] = (0.0,)*3 + (1.0,)
    palette = {a:c for a, c in zip(available_antagconc, palette)}

    marker_bank = ["^", "P", "X", "*"]
    markers = ["o", "s"]
    if len(available_antagconc) > len(marker_bank) + len(markers):
        raise NotImplementedError("Don't know which other markers to use")
    while len(available_antagconc) > len(markers):
        markers.insert(-1, marker_bank.pop(0))
    markers = {available_antagconc[i]:markers[i] for i in range(len(available_antagconc))}

    styles_bank = [":", "-.", (0, (3, 1, 1, 1)), (0, (2.5, 1, 5, 1))]  # Bank for 6F TCR
    styles = ["-", "--"]  # Main styles
    if len(available_antagconc) > len(styles_bank) + len(styles):
        raise NotImplementedError("Don't know which other line styles to use")
    while len(available_antagconc) > len(styles):
        styles.insert(-1, styles_bank.pop(0))
    linestyles = {available_antagconc[i]:styles[i] for i in range(len(available_antagconc))}

    index_antag = list(df_model.index.names).index("AntagonistConcentration")
    index_ag = list(df_model.index.names).index("AgonistConcentration")
    densities = ["Low", "Med", "High"]
    for i in range(n_plots):
        ag_conc = available_agconc[i]
        ax = axes[i]
        ax.axhline(0.0, ls=(0, (5, 2.5, 2.5, 2.5)), color="grey", lw=1.0)
        for j, antag_conc in enumerate(available_antagconc):
            conc_key = ((ag_conc, antag_conc) if index_ag < index_antag
                                    else (antag_conc, ag_conc))
            try:
                data_pts = np.log2(df_data.loc[conc_key]).sort_index()
            except KeyError:
                continue  # This agonist, antagonist pair isn't available
            else:
                model_pts = np.log2(df_model.loc[conc_key, "best"]).sort_index()
                model_ci_low = np.log2(df_model.loc[conc_key, "percentile_2.5"]).sort_index()
                model_ci_hi = np.log2(df_model.loc[conc_key, "percentile_97.5"]).sort_index()
                err_pts = df_ci.loc[conc_key].sort_index()

            hue = palette.get(antag_conc)
            mark = markers.get(antag_conc)
            style = linestyles.get(antag_conc)
            # First shade confidence interval
            ax.fill_between(model_ci_low.index.get_level_values("Antagonist").values,
                            model_ci_low.values, model_ci_hi.values, color=hue, alpha=0.25)
            # Then plot data, and finally model on top
            errbar = ax.errorbar(data_pts.index.get_level_values("Antagonist").values, data_pts.values,
                       yerr=err_pts.values, marker=mark, ls="none", color=hue, mfc=hue, mec=hue)
            li, = ax.plot(model_pts.index.get_level_values("Antagonist").values, model_pts.values,
                       color=hue, ls=style, label=antag_conc)

        # Change y tick labels to 2^x
        if level == "supplement":
            ax.set_title(ag_conc + " Agonist", y=0.95, va="top",
                     size=plt.rcParams["font.size"])
        else:
            ax.set_title(densities[i] + " agonist density", y=1.005, va="top",
                     size=plt.rcParams["font.size"])
    # For each plot in the last row of its column, label x axis.
    for j in range(n_plots-1, n_plots-1-n_cols, -1):
        axes[j].set_xlabel(r"TCR antagonist"+"\n" + r"strength, $\tau$ (s)")
        axes[j].xaxis.set_tick_params(labelbottom=True)

    # For each plot which is first in its row, label y axis
    fc_label = r"$FC_{\mathrm{TCR/TCR}}$"
    axes[0].set_ylabel(fc_label)
    for j in range(n_plots-n_cols, -1, -n_cols):
        axes[j].set_ylabel(fc_label)
        axes[j].yaxis.set_tick_params(labelleft=True)

    # Set log ticks everywhere
    ylim = axes[0].get_ylim()
    for j in range(n_plots):
        change_log_ticks(axes[j], base=2, which="y")
    # Undesirable effect of change_log_ticks: changes y lims. Undo it.
    for j in range(n_plots):
        axes[j].set_ylim(*ylim)
    # Get legend handles and labels, manually generated
    # to indicate markers and hues, model and data
    fig.tight_layout()
    leg_handles, leg_handler = data_model_handles_legend(palette, markers, linestyles,
                        "Antagonist\nconcentration", model_style="-", model_lw=li.get_linewidth(),
                        data_size=plt.rcParams["lines.markersize"], data_marker="o")

    return fig, axes, leg_handles, leg_handler


# Hexbin plot for use with pairgrid taken from StackOverflow:
# https://stackoverflow.com/questions/40495093/hexbin-plot-in-pairgrid-with-seaborn
def hexbin(x, y, color, max_ser=None, min_ser=None, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    ax = plt.gca()
    xmin, xmax = min_ser[x.name], max_ser[x.name]
    ymin, ymax = min_ser[y.name], max_ser[y.name]
    artist = ax.hexbin(x, y, gridsize=15, cmap=cmap,
            extent=[xmin, xmax, ymin, ymax], **kwargs)
    return artist


## Pairplot with hexbins off-diagonal
def hexpair(data, grid_kws={}, hexbin_kws={}, diag_kws={}):
    """ Wrapper around FacetGrid, mapping sns.jointplot to lower
    half and histogram to diagonal. grid_kws contain arguments about
    the variables to be plotted, hues, and so on.

    Args:
        data (pd.DataFrame)
        grid_kws (dict): keyword arguments passed to seaborn.PairGrid.
            Accepted values and defaults: hue=None, hue_order=None,
            palette=None, hue_kws=None, vars=None, x_vars=None, y_vars=None,
            corner=False, diag_sharey=True, height=2.5, aspect=1,
            layout_pad=0.5, despine=True, dropna=False, size=None

        hexbin_kws (dict): keyword arguments passed to sns.jointplot,
            apart from kind="hex". Accepted values and defaults:


        diag_kws (dict): keyword arguments passed to sns.histogram.
            Accepted values and defaults: hue=None, weights=None, stat='count',
            bins='auto', binwidth=None, binrange=None, discrete=None,
            cumulative=False, common_bins=True, common_norm=True,
            multiple='layer', element='bars', fill=True, shrink=1, kde=False,
            kde_kws=None, line_kws=None, thresh=0, pthresh=None, pmax=None,
            cbar=False, cbar_ax=None, cbar_kws=None, palette=None,
            hue_order=None, hue_norm=None, color=None, log_scale=None,
            legend=True, ax=None, **kwargs

    Returns:
        g (sns.PairGrid)
    """
    g = sns.PairGrid(data, **grid_kws)
    g = g.map_diag(sns.histplot, **diag_kws)
    g = g.map_lower(hexbin, min_ser=data.min(), max_ser=data.max(), **hexbin_kws)
    for i in range(g.axes.shape[0]):
        g.axes[i, i].set_ylabel("Density")
        for j in range(i+1, g.axes.shape[1]):
            g.axes[i, j].set_axis_off()
    return g


def average_time_series(t, y):
    # Use the trapeze rule with potentially unequal time intervals
    dt = np.diff(t)
    return np.sum((y[1:] + y[:-1])*dt*0.5) / (t.max() - t.min())


def standalone_legend(*leg_args, **leg_kwargs):
    fig1, ax1 = plt.subplots()
    ax1.set_axis_off()
    leg1 = ax1.legend(*leg_args, **leg_kwargs)
    fig1.canvas.draw()
    # First dummy drawing to get the legend size in inches
    leg_width = leg1.get_window_extent().width / fig1.dpi
    leg_height = leg1.get_window_extent().height / fig1.dpi
    plt.close()

    # Now, actual figure which we set at the right size
    fig, ax = plt.subplots()
    fig.set_size_inches(leg_width*1.05, leg_height*1.05)
    ax.set_axis_off()
    leg = ax.legend(*leg_args, **leg_kwargs)
    fig.tight_layout()
    return fig, ax, leg


def standalone_parameter_values(variable_names, values, n_per_line=1):
    figt, axt = plt.subplots()
    axt.set_axis_off()
    # Build nice string of variable values
    msg = ""
    for i in range(len(variable_names)):
        lbl, n = variable_names[i], values[i]
        msg += "$" + lbl + " = " + str(n) + "$"
        if (i+1) % n_per_line == 0 and i < len(variable_names)-1:
            msg += "\n"
        elif i < len(variable_names)-1:
            msg += ", "

    # Use the figure renderer to get the bbox extent; no need to plot and erase
    txt = axt.annotate(msg, xy=(0.5, 0.5), xycoords="axes fraction", ha="center", va="center")
    txt_width = txt.get_tightbbox(renderer=figt.canvas.get_renderer()).width / figt.dpi
    txt_height = txt.get_tightbbox(renderer=figt.canvas.get_renderer()).height / figt.dpi

    # Now, actual figure which we set at the right size
    figt.set_size_inches(txt_width*1.05, txt_height*1.05)
    axt.set_axis_off()
    figt.tight_layout()
    return figt, axt, txt


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


def add_prior_over_corner(fig, prior_means, prior_stds,
    lw=0.5, color="xkcd:sage", levels=6, alpha=0.8):
    n_params = len(prior_means)
    gs = fig.axes[0].get_gridspec()
    axes = np.asarray(fig.axes).reshape(gs.nrows, gs.ncols)
    for i in range(n_params):
        for j in range(i+1):
            ax = axes[i, j]
            xlims = ax.get_xlim()
            ylims = ax.get_ylim()
            xrange = np.linspace(prior_means[j]-3*prior_stds[j], prior_means[j]+3*prior_stds[j], 100)
            yrange = np.linspace(prior_means[i]-3*prior_stds[i], prior_means[i]+3*prior_stds[i], 100)
            #print("For i, j = {}, xlims={}, ylims={}".format((i, j), xlims, ylims))
            if i == j:  # marginal
                zvals = sp.stats.norm.pdf(xrange, loc=prior_means[i], scale=prior_stds[i])
                zvals *= 0.9*ylims[1] / zvals.max()
                ax.plot(xrange, zvals, color=color, lw=lw)
                # Extend ax limits manually if the prior doesn't fit:
                # combat corner's default behavior
                ax.set_xlim(min(xlims[0], xrange.min()), max(xlims[1], xrange.max()))
            else:
                xgrid, ygrid = np.meshgrid(xrange, yrange)
                #print(xgrid, ygrid)
                zvals = sp.stats.norm.pdf(xgrid, loc=prior_means[j], scale=prior_stds[j])
                zvals *= sp.stats.norm.pdf(ygrid, loc=prior_means[i], scale=prior_stds[i])
                ax.contour(xgrid, ygrid, zvals, levels=levels, colors=color, alpha=alpha, linewidths=lw)
                ax.set_xlim(min(xlims[0], xrange.min()), max(xlims[1], xrange.max()))
                ax.set_ylim(min(ylims[0], yrange.min()), max(ylims[1], yrange.max()))
    return fig


def corner_plot_mcmc(samples_fname, analysis_fname, kmf, pdims,
    pnames=None, prior_means=None, prior_stds=None, prior_kwargs={},
    sizes_kwargs={}, **kwargs):
    """
    Args:
        samples_fname (str): HDF5 file containing MCMC run results
        analysis_fname (str): JSON file containing MCMC analysis results
        pnames (list): list of fitted parameter names to override the
            ones in results files
        pdims (dict): dimensions of panels
        sizes_kwargs (dict): things like scaleup, small_lw, truth_lw, small_markersize.
            Also truth_color.
        prior_means (np.ndarray): means of the priors
        prior_stds (np.ndarray): standard deviations of the priors
        prior_kwargs (dict): kwargs passed to add_prior_over_corner,
            such as color and lw

    Other kwargs are passed to corner.corner.

    Returns:
        fig (matplotlib.figure.Figure): cornerplot figure
    """
    # Load relevant data: samples, parameter names, burn in fraction
    samples_file = h5py.File(samples_fname, "r")
    samples = samples_file.get("samples").get(str(kmf))
    param_names = samples_file.get("samples").attrs.get("param_names")
    if pnames is None:
        # Prepare nicer parameter labels (add log10, etc.)
        # Python >= 3.9 would have a nice string method .removeprefix to remove leading \log...
        param_labels = list(map(lambda a: r"$\log_{10}\," + a.strip("$").replace(r"\log ", "") + "$", param_names))
        param_labels = list(map(lambda a: a.replace("thresh", "th"), param_labels))
    else:
        assert len(pnames) == len(param_names)
        param_labels = pnames

    # Drop the burn_in fraction.
    with open(analysis_fname, "r") as h:
        results_dict = json.load(h).get(str(kmf))
    burn_in_steps = results_dict["burn_in_steps"]
    # Flatten the walker dimension.
    processed_samples = (samples[:, :, burn_in_steps:]
                         .reshape(samples.shape[0], -1))
    # Bins for histograms: doane
    doane_bins = np.histogram_bin_edges(processed_samples[0], bins="doane").size - 1

    # Get the best sample
    pvec_best = np.asarray(results_dict.get("param_estimates").get("MAP best"))

    # Make the corner plot, using aesthetical parameters in sizes_kwargs
    scaleup = sizes_kwargs.get("scaleup", 1.0)
    small_lw = sizes_kwargs.get("small_lw", 0.8) * scaleup
    truth_lw = sizes_kwargs.get("truth_lw", 1.25) * scaleup
    small_markersize = sizes_kwargs.get("small_markersize", 1.0) * scaleup
    tcr_color = np.asarray((0.0, 156.0, 75.0, 255.0)) / 255.0  # deep key lime green
    truth_color = sizes_kwargs.get("truth_color", tcr_color)
    #"xkcd:cornflower", #"xkcd:sage"
    reverse_plots = sizes_kwargs.get("reverse_plots", False)
    labelpad = sizes_kwargs.get("labelpad", len(pvec_best)**3/200.0)

    # Corner plot
    hist2d_kwargs = {"contour_kwargs":{"linewidths":small_lw},
                     "data_kwargs":{"ms":small_markersize}}
    fig = corner.corner(
        data=processed_samples.T,
        labels=param_labels,
        reverse=reverse_plots,
        # Plot truths manually below to control line width
        nbins=doane_bins,
        labelpad=labelpad,
        hist_kwargs={"linewidth": small_lw},
        **hist2d_kwargs,
        **kwargs
    )
    # Add truths manually to control line width
    corner.overplot_lines(fig, pvec_best, reverse=reverse_plots, color=truth_color, lw=truth_lw)
    corner.overplot_points(fig, [[p for p in pvec_best]], reverse=reverse_plots,
                           color=truth_color, ms=2.5*small_markersize, marker="s")

    n = sizes_kwargs.get("n_times_height", 2)
    mx = sizes_kwargs.get("n_extra_x_labels", 2)
    my = sizes_kwargs.get("n_extra_y_labels", 0)

    # If Gaussian prior distributions are used, plot them before setting ticks
    if prior_means is not None and prior_stds is not None:
        fig = add_prior_over_corner(fig, prior_means, prior_stds, **prior_kwargs)

    # Data is log-transformed, put back log ticks and labels
    # Treat y axes of the leftmost column, excluding first which is histogram
    gs = fig.axes[0].get_gridspec()
    axes = np.asarray(fig.axes).reshape(gs.nrows, gs.ncols)
    for i in range(1, axes.shape[0]):
        ax = axes[i, 0]
        put_log_ticks(ax, axis="y", max_ticks=4)
        share_axis(axes, (i, 0), which="y", reverse=reverse_plots)

    # Treat x axes of the bottom row, including last
    max_x_ticks = 5 if axes.shape[1] < 5 else 4
    for j in range(axes.shape[1]):
        ax = axes[-1, j]
        put_log_ticks(ax, axis="x", max_ticks=max_x_ticks)
        share_axis(axes, (axes.shape[0]-1, j), which="x", reverse=reverse_plots)

    fig.set_size_inches(pdims["panel_width"]*n + pdims["axes_label_width"]*mx,
                       pdims["panel_height"]*n + pdims["axes_label_width"]*my)
    #fig.tight_layout()
    samples_file.close()
    return fig


# Compute and plot autocorrelation function of each parameter
# Annotate with autocorrelation time and compare it to burn-in fraction
def plot_autocorr(samples_fname, analysis_fname, kmf, pdims, pnames=None, show_50=True, show_tmax=False, figax=None):
    # Load relevant data: samples, parameter names, burn in fraction
    # Express time in fraction of MCMC simulation to prove convergence.
    samples_file = h5py.File(samples_fname, "r")
    samples = samples_file.get("samples").get(str(kmf))
    param_names = samples_file.get("samples").attrs.get("param_names")
    thin_by = samples_file.get("data").attrs["thin_by"]
    nsteps_recorded = samples.shape[2]
    nsteps_simul = nsteps_recorded * thin_by
    accept_frac = samples_file.get("samples").get(str(kmf)).attrs.get("acceptance_fraction")
    print("Acceptance fraction (average across walkers):", np.mean(accept_frac))
    accept_frac = np.mean(accept_frac)
    if pnames is None:
        # Prepare nicer parameter labels (add log10, etc.)
        # Python >= 3.9 would have a nice string method .removeprefix to remove leading \log...
        param_labels = list(map(lambda a: r"$\log_{10}\," + a.strip("$").replace(r"\log ", "") + "$", param_names))
        param_labels = list(map(lambda a: a.replace("thresh", "th"), param_labels))
    else:
        assert len(pnames) == len(param_names)
        param_labels = pnames

    # Compute autocorrelation function of each parameter, averaging across walkers
    with open(analysis_fname, "r") as h:
        analysis_file = json.load(h)
        analysis_taus = [t * thin_by for t in analysis_file.get(str(kmf)).get("taus_corr")]
        burn_in_frac = analysis_file.get(str(kmf)).get("burn_in_frac")
        burn_in_step = analysis_file.get(str(kmf)).get("burn_in_steps") * thin_by
    corr_fct_mid, corr_fct_full = [], []
    taus_mid, taus_full = [], []
    for i in range(len(param_names)):
        # Compute also with only first 50 % of points, should agree nicely with full run
        # This proves that the autocorrelation estimator has converged.
        fct_mid = autocorr_func_avg(samples[i, :, :nsteps_recorded//2])
        tau_mid = autocorr_avg(samples[i, :, :nsteps_recorded//2], c=5.0) * thin_by
        corr_fct_mid.append(fct_mid)
        taus_mid.append(tau_mid)

        fct_full = autocorr_func_avg(samples[i])
        tau_full = autocorr_avg(samples[i], c=5.0) * thin_by
        corr_fct_full.append(fct_full)
        taus_full.append(tau_full)

        # Tolerate a 1-step difference, although should be exactly the same
        if abs(taus_full[i] - analysis_taus[i]) > 1.0:
            print("Discrepancy in tau_int computed")
            print("Here, found tau of {} = {}".format(param_labels[i], taus_full[i]))
            print("While the analysis file saved = {}".format(analysis_taus[i]))

    # Now plot.
    n_step_range = np.arange(0, nsteps_recorded*thin_by, thin_by) / nsteps_simul
    max_tau = max(taus_full)
    plot_edge = max(2*int(max_tau), int(burn_in_step*1.1)) // thin_by  # Got to 2*tau or burn_in
    n_step_range = n_step_range[:plot_edge]

    # Number of legend columns determines figure width
    leg_columns = (len(param_labels)+1) // 5 + min(1, (len(param_labels)+1) % 5)
    if figax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches((pdims["panel_width"] + pdims["axes_label_width"]
                             + pdims["legend_width"]*leg_columns),
                        pdims["panel_height"] + pdims["axes_label_width"])
    else:
        fig, ax = figax
    palette = sns.color_palette("plasma_r", n_colors=len(param_names))
    for i in range(len(param_names)):
        # Do not label 50 % since we will use a custom legend to indicate line style.
        if show_50:
            li, = ax.plot(n_step_range, corr_fct_mid[i][:plot_edge], ls="--", lw=1.0,
                     color=palette[i])
        ax.plot(n_step_range, corr_fct_full[i][:plot_edge], label=param_labels[i],
               ls="-", color=palette[i], lw=0.75)

    # Vertical line for the largest autocorrelation time across parameters
    tau_max_i = np.argmax(taus_full)
    if show_tmax:
        ax.axvline(max_tau / nsteps_simul, ls=":", color=palette[tau_max_i],
                   lw=1.0, label="Largest $T_{int}$")
    ax.axvline(burn_in_step / nsteps_simul, ls="--", color="k", lw=1.0)
    ylims = ax.get_ylim()
    xlims = ax.get_xlim()
    ax.fill_betweenx(y=ylims, x1=xlims[0], x2=burn_in_frac,  #burn_in_step / nsteps_simul,
                     label="Burn-in", color="k", alpha=0.15, zorder=-100)
    # Set xlim and ylim back to what it was
    ax.set_ylim(ylims)
    ax.set_xlim(xlims)

    # Labeling
    ax.set(xlabel="Fraction of MCMC \nsimulation time",
           ylabel="Normalized auto-\ncorrelation function")
    ax.annotate("Acceptance\nfraction:\n{:.3f}".format(np.mean(accept_frac)),
                xy=(n_step_range.max()/2.0, 1.0),
                ha="center", va="top")

    # Legend
    if show_50:
        # Custom legend if we ever need to show 50 %...
        pass
    leg = ax.legend(loc="upper left", bbox_to_anchor=(burn_in_step / plot_edge / thin_by, 1.0),
              labelspacing=0.1, frameon=False, ncol=leg_columns, columnspacing=0.4, handletextpad=0.3,
              borderaxespad=0.0, borderpad=0.1, handlelength=1.5)
    fig.tight_layout()

    samples_file.close()
    return fig, ax, leg, corr_fct_full, taus_full
