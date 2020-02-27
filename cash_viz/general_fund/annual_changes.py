import seaborn as sns
from matplotlib import pyplot as plt
from .. import utils
from ..style import default_style

__all__ = ["annual_changes_in_spending", "annual_changes_in_revenue"]


def annual_changes_in_spending(report, filename):
    """
    Plot a heatmap of the year to year changes in General 
    Fund spending categories.

    Parameters
    ----------
    report : CashReport
        the cash report object
    filename : str
        the name of the file to save to
    """
    _plot(report, filename, "spending")


def annual_changes_in_revenue(report, filename):
    """
    Plot a heatmap of the year to year changes in General 
    Fund revenue categories.

    Parameters
    ----------
    report : CashReport
        the cash report object
    filename : str
        the name of the file to save to
    """
    _plot(report, filename, "revenue")


def _plot(report, filename, kind):
    """
    Internal function to plot the heatmap.
    """
    kind = kind.lower()
    assert kind in ["revenue", "spending"]

    # Load annual General Fund numbers
    data = report.annual_general_fund_totals()
    data = data.loc[data["Kind"] == kind.capitalize()]
    data = data.pivot(columns="Fiscal Year", index="Name", values="Total")
    order = data[report.year].sort_values().index

    with plt.style.context(default_style):
        # create figure
        grid_kws = dict(left=0.3, right=0.98, bottom=0.2)
        fig, ax = plt.subplots(figsize=(8, 5), gridspec_kw=grid_kws)

        # difference
        Y = data.diff(axis=1)
        Y = Y.T.dropna(how="all").T
        Y = Y.loc[order]

        # get the colormap
        if kind == "revenue":
            cmap = plt.get_cmap("RdGy")
        else:
            cmap = plt.get_cmap("RdGy_r")

        # plot the heatmap
        ymax = Y.max().max()
        ax = sns.heatmap(
            Y,
            fmt=".1f",
            annot=True,
            linewidths=0.5,
            cmap=cmap,
            vmin=-1.0 * ymax,
            vmax=ymax,
            annot_kws={"fontsize": 7},
        )

        # Format
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=90)
        plt.setp(ax.get_yticklabels(), fontsize=11)
        cbar = ax.collections[0].colorbar
        ticks = cbar.ax.get_yticks()
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels(
            [utils.format_currency(x, "{:.0f}") for x in ticks], fontsize=11
        )

        title = f"Year-over-Year Changes in General Fund {kind.capitalize()}"
        fig.text(
            0.5,
            0.95,
            "%s\n(Amounts in Millions)" % title,
            fontsize=12,
            weight="bold",
            ha="center",
            va="center",
        )

        # save
        plt.savefig(filename, dpi=300)

