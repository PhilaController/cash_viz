import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from .. import utils
from ..style import default_style, palette
from qcmr.parse.utils import get_fiscal_months

__all__ = [
    "historical_EOQ_balances",
    "this_year_vs_last_year",
    "historical_accuracy_of_EOY_projection",
    "change_in_EOY_balance_projection",
    "actual_balance_vs_last_quarters_projection",
    "historical_annual_balance_changes",
]

fund_colors = {}
fund_colors["General Fund"] = palette["ben-franklin-blue"]
fund_colors["Consolidated Cash"] = palette["kelly-drive-green"]
fund_colors["Grants Fund"] = palette["bell-yellow"]
fund_colors["Total Capital Funds"] = palette["flyers-orange"]


def historical_annual_balance_changes(report, filename):
    """
    Plot a bar graph of the change in end-of-year balance for the current
    year relative to last year.

    Parameters
    ----------
    report : CashReport
        the cash report object
    filename : str
        the file name to save the file to
    """
    # load the data
    data = report.historical_balance_by_quarter()

    values = []
    labels = []
    for col in fund_colors:
        values.append(data[col].diff())
        label = col
        labels.append([label] * len(data[col]))

    years0 = sorted(data["Fiscal Year"].unique())
    years = np.concatenate([years0] * len(fund_colors), axis=0)
    values = pd.concat(values, axis=0)
    labels = np.concatenate(labels, axis=0)
    df = pd.DataFrame({"Fiscal Year": years, "value": values, "label": labels})

    with plt.style.context(default_style):

        # Initialize
        fig, ax = plt.subplots(
            figsize=(6, 3.75), gridspec_kw=dict(left=0.13, bottom=0.17, top=0.85)
        )

        # plot monthly
        N = len(data["Fiscal Year"].unique()) - 1
        df = df.dropna()
        df = df.set_index(["label", "Fiscal Year"])

        # Set position of bar on X axis
        barWidth = 0.15
        r0 = r = np.arange(N)

        # Make the plot
        for col in [
            "General Fund",
            "Grants Fund",
            "Total Capital Funds",
            "Consolidated Cash",
        ]:
            label = col
            ax.bar(
                r,
                df.loc[label]["value"].values,
                color=fund_colors[col],
                width=barWidth,
                edgecolor="none",
                label=label,
            )
            r = [x + barWidth for x in r]

        # Add y=0 line
        ax.axhline(y=0, c="k", lw=1, zorder=2)

        # Format x-axis
        ax.set_xticks([r + 1.5 * barWidth for r in range(N)])
        ax.set_xticklabels(years0[1:], rotation=90, fontsize=13)
        ax.set_xlim(-0.25, N)
        ax.set_xlabel("Fiscal Year", fontsize=13, weight="bold")
        ax.xaxis.labelpad = 5
        ax.tick_params(axis="x", pad=0)
        ax.grid(False, axis="x")

        # Format y-axis
        ax.set_ylim(-650, 700)
        ax.set_yticks([-600, -400, -200, 0, 200, 400, 600])
        ax.set_yticklabels(
            [utils.format_currency(x, "{:,.0f}M") for x in ax.get_yticks()], fontsize=13
        )
        ax.set_ylabel("")

        # Y-axis label
        ax.text(
            0.005,
            0.97,
            "Year-Over-Year Change\nin Cash Balances",
            weight="bold",
            fontsize=13,
            ha="left",
            va="top",
            transform=fig.transFigure,
        )
        leg = plt.legend(
            ncol=2,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            bbox_transform=fig.transFigure,
            fontsize=10,
            frameon=False,
        )
        leg.set_title("")

        # save
        plt.savefig(filename, dpi=300)


def actual_balance_vs_last_quarters_projection(report, filename):
    """
    Plot a bar graph of the difference between the fund balance at the 
    end of the current quarter vs. the projection from last quarter.

    Parameters
    ----------
    report : CashReport
        the cash report object
    filename : str
        the file name to save the file to
    """
    CURRENT_FY = report.year
    CURRENT_Q = report.quarter
    if CURRENT_Q == 1:
        raise ValueError("This chart is not valid for Q1")

    # comparison to last quarter
    df = report.compare_to_last_quarter()

    # calculate the change
    this_quarter = utils.make_tag(CURRENT_FY, CURRENT_Q)
    last_quarter = utils.make_tag(CURRENT_FY, CURRENT_Q - 1)
    df["Change"] = df[this_quarter] - df[last_quarter]

    # filter by end-of-quarter fund balances
    month = get_fiscal_months()[CURRENT_Q * 3 - 1]
    df = df.query(f"Kind == 'Fund Balance' & Month == '{month.capitalize()}'")

    # re-order
    df = df.set_index("Name")
    order = ["General Fund", "Grants Fund", "Total Capital Funds", "Consolidated Cash"]
    df = df.loc[order]

    with plt.style.context(default_style):

        # initalize the figure/axes
        grid_kws = dict(top=0.85, left=0.33, right=0.88, bottom=0.17, hspace=0.45)
        fig, ax = plt.subplots(figsize=(5, 2.5), gridspec_kw=grid_kws)

        # plot
        sns.barplot(
            ax=ax,
            y=df.index,
            x=df["Change"],
            zorder=10,
            saturation=1.0,
            palette=[fund_colors[col] for col in df.index],
        )

        # add texts
        for i, (name, row) in enumerate(df.iterrows()):
            x = row["Change"]
            ax.text(
                x + 5,
                i,
                utils.format_currency(x, "{:,.1f}M", plus_sign=True),
                ha="left",
                va="center",
                fontsize=10,
                zorder=5,
                bbox=dict(facecolor="white", pad=0, edgecolor="none"),
            )

        # x = 0 line
        ax.axvline(x=0, lw=0.5, c="k", zorder=12)

        # x-axis
        ax.tick_params(axis="y", pad=2)

        # ticks
        ax.set_xticklabels(
            [
                utils.format_currency(x, "{:,.0f}M", plus_sign=True)
                for x in ax.get_xticks()
            ],
            fontsize=12,
        )
        plt.setp(ax.get_yticklabels(), fontsize=12)

        # labels
        ax.set_xlabel(
            f"Increase from Q{CURRENT_Q-1} Projection", weight="bold", fontsize=12
        )
        ax.set_ylabel("")

        # add a title
        fig.text(
            0.5,
            0.92,
            f"Actual Cash Balances at the End of Q{CURRENT_Q} vs. Q{CURRENT_Q-1} Projections",
            weight="bold",
            fontsize=12,
            ha="center",
        )

        # save
        plt.savefig(filename, dpi=300)


def change_in_EOY_balance_projection(report, filename):
    """
    Plot a bar graph of the change in end-of-year balances since 
    last quarter.

    Parameters
    ----------
    report : CashReport
        the cash report object
    filename : str
        the file name to save the file to
    """
    CURRENT_FY = report.year
    CURRENT_Q = report.quarter
    if CURRENT_Q == 1:
        raise ValueError("This chart is not valid for Q1")

    # comparison to last quarter
    df = report.compare_to_last_quarter()

    # calculate the change
    this_quarter = utils.make_tag(CURRENT_FY, CURRENT_Q)
    last_quarter = utils.make_tag(CURRENT_FY, CURRENT_Q - 1)
    df["Change"] = df[this_quarter] - df[last_quarter]

    # Get the end-of-year balance
    df = df.query(f"Kind == 'Fund Balance' & Month == 'Jun'")

    # re-order
    df = df.set_index("Name")
    order = ["General Fund", "Grants Fund", "Total Capital Funds", "Consolidated Cash"]
    df = df.loc[order]

    with plt.style.context(default_style):

        # initalize the figure/axes
        grid_kws = dict(top=0.85, left=0.33, right=0.9, bottom=0.17, hspace=0.45)
        fig, ax = plt.subplots(figsize=(5, 2.5), gridspec_kw=grid_kws)

        # plot
        sns.barplot(
            ax=ax,
            y=df.index,
            x=df["Change"],
            zorder=10,
            saturation=1.0,
            palette=[fund_colors[col] for col in df.index],
        )

        # add texts
        for i, (name, row) in enumerate(df.iterrows()):
            x = row["Change"]
            ax.text(
                x,
                i,
                utils.format_currency(x, "{:,.0f}M", plus_sign=True),
                ha="left",
                va="center",
                fontsize=10,
                zorder=3,
                bbox=dict(facecolor="white", pad=0, edgecolor="none"),
            )

        # x = 0 line
        ax.axvline(x=0, lw=0.5, c="black", zorder=12)

        # x-axis
        ax.tick_params(axis="y", pad=2)

        # ticks
        ax.set_xticklabels(
            [
                utils.format_currency(x, "{:,.0f}M", plus_sign=True)
                for x in ax.get_xticks()
            ],
            fontsize=12,
        )
        plt.setp(ax.get_yticklabels(), fontsize=12)

        # labels
        ax.set_xlabel(
            f"Increase from Q{CURRENT_Q-1} Projection", weight="bold", fontsize=12
        )
        ax.set_ylabel("")

        # add a title
        fig.text(
            0.5,
            0.92,
            f"Change in End-of-Year Balance from Q{CURRENT_Q-1} to Q{CURRENT_Q}",
            weight="bold",
            fontsize=12,
            ha="center",
        )

        # save
        plt.savefig(filename, dpi=300)


def historical_EOQ_balances(report, filename, include_TRAN=True):
    """
    Plot the historical balances at the end of the current quarter.

    This is a line chart showing the end-of-quarter balance for the 
    General Fund, Consolidated Cash, Capital Fund, and Grants Fund
    since 2007.

    Parameters
    ----------
    report : CashReport
        the cash report object
    filename : str
        the file name to save the file to
    include_TRAN : bool, optional
        whether to include the TRAN in the balances
    """
    # get the data
    data = report.historical_balance_by_quarter()

    with plt.style.context(default_style):

        # create figure
        grid_kws = dict(left=0.03, bottom=0.22, top=0.85, right=0.98)
        fig, ax = plt.subplots(figsize=(5.5, 3.5), gridspec_kw=grid_kws)

        # the colors per funds
        cols = ["Grants Fund", "Total Capital Funds"]
        _cols = ["General Fund", "Consolidated Cash"]
        if not include_TRAN:
            _cols += [col + " (No TRAN)" for col in _cols]
        cols += _cols

        ymax = 0
        for i, col in enumerate(cols):
            label = col
            color = fund_colors[col]
            ax.plot(
                data["Fiscal Year"],
                data[col],
                label=label,
                color=color,
                marker="o",
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.5,
                markersize=5,
                lw=2,
                zorder=i + 3,
            )
            if data[col].max() > ymax:
                ymax = data[col].max()

        # put a line at zero
        ax.axhline(y=0, c="k", lw=1, zorder=1)

        # format x ticks
        ax.set_xticks(data["Fiscal Year"].tolist())
        plt.setp(ax.get_xticklabels(), fontsize=13, rotation=90)
        ax.set_xlabel("Fiscal Year", fontsize=13, weight="bold")
        ax.set_xlim(-2, len(data) - 0.5)

        # format y ticks
        ax.set_yticks(np.arange(-200, ymax + 200, 200))

        yticklabels = []
        for x in ax.get_yticks():
            if x < 1000:
                yticklabels.append(utils.format_currency(x, "{:,.0f}M"))
            else:
                yticklabels.append(utils.format_currency(x / 1e3, "{:,.1f}B"))
        ax.set_yticklabels(yticklabels, fontsize=10)
        plt.setp(ax.get_yticklabels(), ha="left")

        # add labels
        if report.quarter != 4:
            title = f"End-of-Q{report.quarter}\nCash Balances"
        else:
            title = "End-of-Fiscal-Year\nCash Balances"

        if not include_TRAN:
            title += " (No TRAN)"
        ax.text(
            0.005,
            0.99,
            title,
            weight="bold",
            fontsize=13,
            transform=fig.transFigure,
            ha="left",
            va="top",
        )

        # add a legend
        leg = plt.legend(
            ncol=2,
            loc="upper right",
            bbox_to_anchor=(0.98, 1.0),
            bbox_transform=fig.transFigure,
            frameon=False,
            fontsize=9,
        )

        # save
        plt.savefig(filename, dpi=300)


def this_year_vs_last_year(report, filename):
    """
    Plot the fund balances for the General Fund and Consolidated Cash
    for this fiscal year, comparing to last fiscal year.

    This is a two-panel chart showing the comparison for the General 
    Fund and Consolidated Cash.

    Parameters
    ----------
    report : CashReport
        the cash report object
    filename : str
        the file name to save the file to
    """

    def _plot(ax, data, col, color, label=None, linestyle="solid", months=None):
        if months is not None:
            data = data.loc[data["Month"].isin(months)]

        if label is None:
            label = col
        ax.plot(
            data["Month"],
            data[col],
            label=label,
            color=color,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.5,
            markersize=5,
            linestyle=linestyle,
            lw=2,
            zorder=i + 1,
        )

    # load the data
    df = report.compare_to_last_year()

    with plt.style.context(default_style):

        # Initialize
        grid_kws = dict(left=0.05, right=0.98, top=0.85, bottom=0.15, hspace=1.2)
        fig, axs = plt.subplots(
            figsize=(6, 3.75), nrows=2, ncols=1, gridspec_kw=grid_kws
        )

        colors = ["kelly-drive-green", "ben-franklin-blue"]
        funds = ["Consolidated Cash", "General Fund"]
        for i, col in enumerate(funds):

            ax = axs[i]

            # trim the data
            data = df.loc[df["Name"] == col]

            # color
            this_color = palette[colors[i]]

            # plot last year as gray
            _plot(ax, data, f"FY{str(report.year-1)[-2:]}", palette["medium-gray"])

            # plot actual data for this year
            _plot(
                ax,
                data,
                f"FY{str(report.year)[-2:]}",
                this_color,
                months=utils.get_actual_months(report.quarter),
            )
            # plot propjected data for this year
            _plot(
                ax,
                data,
                f"FY{str(report.year)[-2:]}",
                this_color,
                linestyle="dashed",
                label="",
                months=utils.get_projected_months(report.quarter),
            )

            # format
            if i != 0:
                ax.set_xticks(data["Month"].tolist())

            if i == 0:
                ax.set_ylim(450, 1750)
            else:
                ax.set_ylim(450, 1200)
                ax.xaxis.labelpad = 3
            ax.set_xlim(left=-3)

            plt.setp(ax.get_xticklabels(), fontsize=13, rotation=90)
            if i == 0:
                ax.set_yticks(np.arange(500, 1600, 500))
            else:
                ax.set_yticks(np.arange(500, 1100, 500))

            labels = []
            for x in ax.get_yticks():
                labels.append(utils.format_currency(x, "{:,.0f}M"))

            ax.set_yticklabels(labels, fontsize=12)
            plt.setp(ax.get_yticklabels(), ha="left")

            # Label the y-axis
            if col == "General Fund":
                xlabel = col + "\nCash Balance"
            else:
                xlabel = col + "\nBalance"
            ax.text(
                -0.05,
                1.1,
                xlabel,
                weight="bold",
                fontsize=13,
                ha="left",
                va="bottom",
                transform=ax.transAxes,
            )

            # Add a legend
            leg = ax.legend(
                ncol=2,
                loc="upper right",
                bbox_to_anchor=(1, 1.4),
                frameon=False,
                bbox_transform=ax.transAxes,
                fontsize=10,
            )

        # add a footnote
        if report.quarter != 4:
            tag = str(report.year)[-2:]
            fig.text(
                0.002,
                0.01,
                f"Note: Projections beyond FY{tag} Q{report.quarter} shown as a dashed line",
                color="#444444",
                fontsize=7,
            )

        # save
        plt.savefig(filename, dpi=300)


def historical_accuracy_of_EOY_projection(report, filename):
    """
    Plot the historical accuracy of the balance projection for the current
    quarter for the General Fund and Consolidated Cash.

    This is a two-panel chart with bar graphs showing the difference between
    the projection and actual over time.

    Parameters
    ----------
    report : CashReport
        the cash report object
    filename : str
        the file name to save the file to
    """
    # load the data
    df = report.annual_projection_accuracy("Fund Balance")

    with plt.style.context(default_style):

        # Initialize
        grid_kws = dict(
            left=0.14, right=0.98, top=0.7, bottom=0.2, hspace=0, wspace=0.4
        )
        fig, axs = plt.subplots(
            nrows=1, ncols=2, figsize=(6, 3.5), gridspec_kw=grid_kws
        )

        funds = ["Consolidated Cash", "General Fund"]
        for i, col in enumerate(funds):

            ax = axs[i]

            # trim the data
            data = df.loc[df["Name"] == col]
            positive = data[f"Actual - Q{report.quarter} Projection"] >= 0
            negative = data[f"Actual - Q{report.quarter} Projection"] < 0

            # plot
            colors = [palette[c] for c in ["black", "love-park-red"]]
            labels = ["Actual Balance Higher", "Actual Balance Lower"]
            sels = [positive, negative]
            for (color, label, sel) in zip(colors, labels, sels):

                # Zero out positive/negative
                data2 = data.copy()
                data2.loc[~sel, f"Actual - Q{report.quarter} Projection"] = 0

                # Plot
                sns.barplot(
                    ax=ax,
                    x="Fiscal Year",
                    color=color,
                    saturation=1.0,
                    y=f"Actual - Q{report.quarter} Projection",
                    label=label,
                    data=data2,
                )

                # Add a y=0 line
                ax.axhline(y=0, lw=1, c="k")

            # Format x-axis
            ax.set_xlabel("Fiscal Year", fontsize=13, weight="bold")
            plt.setp(ax.get_xticklabels(), fontsize=13, rotation=90)

            # Format y-axis
            ax.set_ylabel("")
            ax.set_ylim(-110, 425)
            ax.set_yticks(np.arange(-100, 425, 100))
            ax.set_yticklabels(
                [
                    utils.format_currency(x, "{:,.0f}M", plus_sign=True)
                    for x in ax.get_yticks()
                ],
                fontsize=13,
            )

            # Add a sub-title
            ax.text(
                0.5,
                1.05,
                "%s" % col,
                weight="bold",
                fontsize=13,
                ha="center",
                transform=ax.transAxes,
            )

            # Add the legend
            if i == 0:
                leg = ax.legend(
                    ncol=1,
                    loc="lower right",
                    bbox_to_anchor=(1.0, 0.82),
                    frameon=False,
                    bbox_transform=fig.transFigure,
                    fontsize=10,
                )

        # Add title
        fig.text(
            0.03,
            0.85,
            f"Year-End Cash Balances:\nActual vs. the Q{report.quarter} Projection",
            ha="left",
            weight="bold",
            fontsize=15,
        )

        # save
        plt.savefig(filename, dpi=300)
