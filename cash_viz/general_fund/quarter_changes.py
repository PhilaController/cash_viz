import seaborn as sns
from matplotlib import pyplot as plt
from .. import utils
from ..style import default_style

__all__ = [
    "revenue_changes_since_last_quarter",
    "spending_changes_since_last_quarter",
    "balance_changes_since_last_quarter",
]


def revenue_changes_since_last_quarter(report, filename):
    """
    Plot a heatmap of the changes to the monthly revenue 
    totals since last quarter.

    Parameters
    ----------
    report : CashReport
        the cash report object
    filename : str
        the file name to save the file to
    """
    _plot_changes_since_last_quarter(report, filename, "revenue")


def spending_changes_since_last_quarter(report, filename):
    """
    Plot a heatmap of the changes to the monthly spending 
    totals since last quarter.

    Parameters
    ----------
    report : CashReport
        the cash report object
    filename : str
        the file name to save the file to
    """
    _plot_changes_since_last_quarter(report, filename, "spending")


def balance_changes_since_last_quarter(report, filename):
    """
    Plot a heatmap of the changes to the monthly fund balance 
    totals since last quarter.

    Parameters
    ----------
    report : CashReport
        the cash report object
    filename : str
        the file name to save the file to
    """
    _plot_changes_since_last_quarter(report, filename, "fund balance")


def _plot_changes_since_last_quarter(report, filename, kind):
    """
    Internal function to plot a heatmap of the changes to the
    monthly revenue/spending/fund balance totals since last quarter.
    """
    kind = kind.lower()
    assert kind in ["revenue", "spending", "fund balance"]

    this_quarter = f"FY{str(report.year)[-2:]} Q{report.quarter}"
    if report.quarter == 1:
        last_quarter = f"FY{str(report.year-1)[-2:]} Q4"
    else:
        last_quarter = f"FY{str(report.year)[-2:]} Q{report.quarter-1}"

    # get the data
    data = report.compare_to_last_quarter()
    data = data.loc[data["Kind"] == " ".join([w.capitalize() for w in kind.split()])]
    data["Change"] = data[this_quarter] - data[last_quarter]

    if kind != "fund balance":
        annual_total = data.groupby("Name")[
            [this_quarter, last_quarter, "Change"]
        ].sum()

    data = data.pivot(columns="Month", index="Name", values="Change")
    if kind != "fund balance":
        data["Annual"] = annual_total["Change"]

    # reorder the columns
    order = [
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
    ]
    if kind != "fund balance":
        order += ["Annual"]
    data = data[order]

    # order y axis
    if kind != "fund balance":
        order = annual_total.sort_values(by=this_quarter).index.tolist()
    else:
        order = [
            "Total Capital Funds",
            "Grants Fund",
            "General Fund",
            "Consolidated Cash",
        ]

    with plt.style.context(default_style):

        # create figure
        grid_kws = dict(left=0.3, right=0.98, bottom=0.2)
        fig, ax = plt.subplots(figsize=(8, 5), gridspec_kw=grid_kws)

        # difference
        Y = data.loc[order]

        ymax = Y.max().max()
        cmap = plt.get_cmap("coolwarm")
        ax = sns.heatmap(
            Y,
            fmt=".1f",
            annot=True,
            linewidths=0.5,
            center=0,
            square=True,
            cmap=cmap,
            vmin=-1.0 * ymax,
            vmax=ymax,
            annot_kws={"fontsize": 7},
        )

        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=90)
        plt.setp(ax.get_yticklabels(), fontsize=11)
        cbar = ax.collections[0].colorbar
        ticks = cbar.ax.get_yticks()
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels(
            [utils.format_currency(x, "{:.0f}") for x in ticks], fontsize=11
        )

        if kind != "fund balance":
            title = f"Difference between {this_quarter} and {last_quarter} Projections for General Fund {kind.capitalize()}"
        else:
            title = f"Difference between {this_quarter} and {last_quarter} Projections for Monthly Cash Balances"

        fig.text(
            0.5,
            0.95,
            "%s\n(Amounts in Millions)" % title,
            fontsize=12,
            weight="bold",
            ha="center",
            va="center",
        )

        # add a key
        fig.text(
            0.01,
            0.05,
            r"$\bf{Key}$"
            + f"\nRed: {this_quarter} value higher\nBlue: {this_quarter} value lower",
            fontsize=10,
            ha="left",
        )

        # save
        plt.savefig(filename, dpi=300)
