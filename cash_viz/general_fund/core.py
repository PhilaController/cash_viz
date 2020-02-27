from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from pandas.tseries.offsets import MonthEnd
from qcmr.parse.utils import get_fiscal_months
from .. import utils
from ..style import default_style, palette

__all__ = [
    "monthly_actuals_this_quarter",
    "historical_accuracy_of_revenue_projection",
    "historical_accuracy_of_spending_projection",
    "historical_annual_cash_flows",
    "fund_balance_revisions",
    "annual_TRAN",
    "historical_monthly_cash_flows",
]


def historical_monthly_cash_flows(report, filename):
    """
    """
    from qcmr.cash import get_GF_revenues, get_GF_spending

    def add_date(df):
        year = df["fiscal_year"].where(df["month"] < 7, df["fiscal_year"] - 1)
        df["Date"] = df["month"].astype(str).str.cat(year.astype(str), sep="/")
        df["Date"] = pd.to_datetime(df["Date"]) + MonthEnd(1)
        return df

    # Revenues
    R = get_GF_revenues().query("quarter == 4")[
        ["Total Cash Receipts", "fiscal_year", "month"]
    ]
    R = add_date(R)[["Date", "Total Cash Receipts"]].rename(
        columns={"Total Cash Receipts": "Revenue"}
    )

    # Spending
    S = get_GF_spending().query("quarter == 4")[
        ["Total Disbursements", "fiscal_year", "month"]
    ]
    S = add_date(S)[["Date", "Total Disbursements"]].rename(
        columns={"Total Disbursements": "Expenditures"}
    )

    data = pd.merge(R, S, on="Date")
    xticks = data["Date"].dt.year[(data["Date"].dt.month == 7)]
    dates = data["Date"].dt.strftime("%m/%y")

    with plt.style.context(default_style):

        # Initialize the figure/axes
        fig, axs = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(6, 5),
            gridspec_kw=dict(top=0.88, left=0.13, bottom=0.15, right=0.99, hspace=1.0),
        )

        for i, col in enumerate(["Revenue", "Expenditures"]):

            ax = axs[i]

            # spending
            if i == 1:
                label = "Expenditures"
                ax.plot(dates, data[col].values, color=palette["love-park-red"], lw=3)
            else:  # revenue
                label = "Revenues"
                ax.plot(dates, data[col].values, color=palette["black"], lw=3)

            # format grid
            sns.despine(left=True, ax=ax)
            ax.grid(b=False, axis="x")

            # x label
            ax.text(
                -0.13,
                1.15,
                "Monthly General Fund\nCash " + label,
                weight="bold",
                fontsize=11,
                transform=ax.transAxes,
            )

            ax.set_xticks(xticks.index.tolist())
            ax.set_xticklabels(xticks.values + 1, fontsize=11, rotation=90)

            ax.set_ylim(0, 1050)
            ax.set_yticks([0, 200, 400, 600, 800, 1000])
            ax.set_yticklabels(
                [utils.format_currency(x, "{:,.0f}M") for x in ax.get_yticks()],
                fontsize=11,
            )
            ax.set_xlabel("Fiscal Year", fontsize=12, weight="bold")
            ax.grid(True)

        fig.savefig(filename, dpi=300)


def annual_TRAN(report, filename):
    """
    Plot a bar graph showing the total amount of TRAN taken out 
    each fiscal year.

    Parameters
    ----------
    report : CashReport
        the cash report object
    filename : str
        the file name to save the file to
    """
    from qcmr.cash import get_GF_balance_sheet

    # Load the data
    df = get_GF_balance_sheet()
    df = (
        df.query("quarter==4")
        .groupby("fiscal_year")["TRAN"]
        .apply(lambda x: sum(abs(x)))
        // 2
    )
    df = (
        df.reset_index()
        .assign(fiscal_year=lambda df: df.fiscal_year.astype(int))
        .rename(columns={"fiscal_year": "Fiscal Year"})
    )

    with plt.style.context(default_style):

        # Initialize the figure/axes
        fig, ax = plt.subplots(
            figsize=(5, 3),
            gridspec_kw=dict(top=0.8, left=0.13, bottom=0.25, right=0.98),
        )

        # bar plot
        sns.barplot(
            ax=ax,
            x=df["Fiscal Year"],
            y=df["TRAN"],
            color=palette["dark-ben-franklin"],
            saturation=1.0,
        )

        # Add the y-axis label
        ax.text(
            -0.13,
            1.12,
            "Short-term General Fund\nBorrowing Amounts",
            weight="bold",
            fontsize=12,
            transform=ax.transAxes,
        )

        # Format
        ax.set_ylim(0, 410)
        plt.setp(ax.get_xticklabels(), fontsize=11, rotation=90)
        ax.set_yticks([0, 100, 200, 300, 400])
        ax.set_yticklabels(["$%.0fM" % x for x in ax.get_yticks()], fontsize=11)
        ax.set_xlabel("Fiscal Year", fontsize=12, weight="bold")
        ax.set_ylabel("")
        ax.xaxis.labelpad = 7

        plt.savefig(filename, dpi=300)


def fund_balance_revisions(report, filename):
    """
    Plot a scatter chart with estimated uncertainty showing the relationship between 
    the modified accrual and cash General Fund balances.

    Parameters
    ----------
    report : CashReport
        the cash report object
    filename : str
        the file name to save the file to
    """
    CURRENT_FY = report.year
    CURRENT_Q = report.quarter

    this_year = f"FY{str(CURRENT_FY)[2:]}"
    last_year = f"FY{str(CURRENT_FY - 1)[2:]}"

    # load the data
    df = report.fund_balance_revisions()

    with plt.style.context(default_style):

        # Initialize the figure/axes
        fig, ax = plt.subplots(
            figsize=(6, 4),
            gridspec_kw=dict(top=0.85, left=0.03, bottom=0.12, right=0.85),
        )

        # Plot the historical data points
        X = df.dropna()
        color = palette["black"]
        fmt = {
            "c": "white",
            "zorder": 11,
            "marker": "o",
            "edgecolors": color,
            "linewidth": 2,
        }
        min_year = f"FY{str(X['Fiscal Year'].astype(int).min())[2:]}"
        max_year = f"FY{str(X['Fiscal Year'].astype(int).max())[2:]}"
        label = f"Annual Historical Data\nfrom {min_year} to {max_year}"
        ax.scatter(X["Q4 Cash Balance"], X["Q1 Actual"], label=label, **fmt)

        # Plot the uncertainty
        ax.fill_between(
            df["Q4 Cash Balance"],
            df["Q1 Actual (Lower)"],
            df["Q1 Actual (Upper)"],
            color=palette["light-ben-franklin"],
            alpha=0.5,
            zorder=10,
            label="Estimated Uncertainty",
        )
        ax.plot(
            df["Q4 Cash Balance"],
            df["Q1 Actual (Upper)"],
            color=palette["light-ben-franklin"],
            lw=3,
            zorder=10,
        )
        ax.plot(
            df["Q4 Cash Balance"],
            df["Q1 Actual (Lower)"],
            color=palette["light-ben-franklin"],
            lw=3,
            zorder=10,
        )

        # Vertical line for this year
        df_this_year = df.query(f"`Fiscal Year` == {report.year}")
        ax.axvline(
            x=df_this_year["Q4 Cash Balance"].squeeze(),
            lw=2,
            color=palette["love-park-red"],
            zorder=11,
        )

        # Mark upper/lower estimates for this year
        labels = ["Lower estimate", "Upper estimate"]
        offsets = [(10, -20), (-30, 15)]
        has = ["left", "right"]
        vas = ["top", "bottom"]
        for i, col in enumerate(["Q1 Actual (Lower)", "Q1 Actual (Upper)"]):
            color = palette["love-park-red"]

            x = df_this_year["Q4 Cash Balance"]
            y = df_this_year[col]
            ax.scatter(
                x,
                y,
                color="white",
                zorder=12,
                marker="o",
                edgecolors=color,
                linewidth=2,
            )
            ax.annotate(
                labels[i] + f"\nfor {this_year}: " + "$%.0fM" % y,
                xy=(x, y),
                xycoords="data",
                xytext=offsets[i],
                textcoords="offset points",
                ha=has[i],
                va=vas[i],
                fontsize=10,
                zorder=12,
                weight="bold",
                arrowprops=dict(
                    arrowstyle="->", color="k", lw=2, connectionstyle="arc3,rad=0.1"
                ),
                bbox=dict(facecolor="white", pad=0),
            )

        # Format
        ax.set_xlim(-395, 1100)
        ax.set_yticklabels(
            [
                utils.format_currency(x, "{:,.0f}M", plus_sign=True)
                for x in ax.get_yticks()
            ]
        )
        ax.set_xticklabels(
            [
                utils.format_currency(x, "{:,.0f}M", plus_sign=False)
                for x in ax.get_xticks()
            ]
        )
        plt.setp(ax.get_yticklabels(), ha="left")

        # Axis labels
        ax.set_xlabel("Q4 Cash Balance", weight="bold", fontsize=11)
        ax.text(
            -0.02,
            1.06,
            "Final Modified Accrual\nFund Balance",
            fontsize=11,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            weight="bold",
        )

        ax.text(
            df_this_year["Q4 Cash Balance"].squeeze(),
            370.0,
            f"{this_year} Cash Balance",
            fontsize=10,
            ha="right",
            va="top",
            weight="bold",
            rotation=90,
            bbox=dict(facecolor="white"),
        )

        # Add arrow for last year
        df_last_year = df.query(f"`Fiscal Year` == {report.year-1}")
        x1 = df_last_year["Q4 Cash Balance"].squeeze()
        y1 = df_last_year["Q1 Actual"].squeeze()
        x2, y2 = 550, 550
        ax.annotate(
            last_year,
            xy=(x1, y1),
            xycoords="data",
            xytext=(x2, y2),
            textcoords="data",
            ha="left",
            va="bottom",
            zorder=12,
            fontsize=10,
            weight="bold",
            arrowprops=dict(
                arrowstyle="->", color="k", lw=2, connectionstyle="arc3,rad=-0.3"
            ),
            bbox=dict(facecolor="white", pad=0),
        )

        # Make the y=0,x=0 grid lines darker
        ax.axhline(y=0, lw=1, color="#2a3135", zorder=1)
        ax.axvline(x=0, lw=1, color="#2a3135", zorder=1)

        # Add the legend
        ax.legend(
            ncol=2,
            fontsize=10,
            bbox_transform=fig.transFigure,
            loc="lower right",
            bbox_to_anchor=(1.01, 0.87),
        )

        plt.savefig(filename, dpi=300)


def historical_annual_cash_flows(report, filename):
    """
    Plot the historical annual revenue and spending cash flows.

    Parameters
    ----------
    report : CashReport
        the cash report object
    filename : str
        the file name to save the file to
    """
    # load the data
    data = (
        report.annual_general_fund_totals()
        .query("Name in ['Total Disbursements', 'Total Cash Receipts']")
        .pivot(index="Fiscal Year", values="Total", columns="Name")
        .rename(
            columns={
                "Total Cash Receipts": "Revenue",
                "Total Disbursements": "Expenditures",
            }
        )
    )

    with plt.style.context(default_style):

        # Initialize
        fig, ax = plt.subplots(
            figsize=(6, 3.75), gridspec_kw=dict(left=0.05, bottom=0.20, top=0.82)
        )

        # revenue and spending
        y1 = data["Revenue"]
        y2 = data["Expenditures"]

        # revenue
        color = palette["black"]
        ax.plot(y1.index, y1.values, color=color, lw=3, label="Revenues")
        fmt = {
            "c": "white",
            "zorder": 10,
            "marker": "o",
            "edgecolors": color,
            "linewidth": 2,
        }
        ax.scatter(y1.index, y1.values, **fmt)

        # spending
        color = palette["love-park-red"]
        ax.plot(y2.index, y2.values, color=color, lw=3, label="Expenditures")
        fmt = {
            "c": "white",
            "zorder": 10,
            "marker": "o",
            "edgecolors": color,
            "linewidth": 2,
        }
        ax.scatter(y2.index, y2.values, **fmt)

        # in the black
        ax.fill_between(
            y1.index,
            y1,
            y2,
            where=y1 > y2,
            color=palette["medium-gray"],
            interpolate=True,
            label="Fund Balance Increases",
        )

        # in the red
        ax.fill_between(
            y1.index,
            y1,
            y2,
            where=y1 < y2,
            color=palette["light-red"],
            interpolate=True,
            label="Fund Balance Decreases",
        )

        # format grid
        sns.despine(left=True, bottom=True, ax=ax)

        # ylabel and y limits
        ax.text(
            0.005,
            0.99,
            "Annual General Fund\nCash Flows",
            weight="bold",
            fontsize=14,
            ha="left",
            va="top",
            transform=fig.transFigure,
        )

        # Format y-axis
        ax.set_ylim(3.45e3, 5.05e3)
        ax.set_yticklabels(["$%.1fB" % (x / 1e3) for x in ax.get_yticks()], fontsize=14)
        plt.setp(ax.get_yticklabels(), ha="left")

        # Format x-axis
        ax.set_xticks(y1.index.tolist())
        plt.setp(ax.get_xticklabels(), fontsize=14, rotation=90)
        ax.set_xlabel("Fiscal Year", fontsize=14, weight="bold")
        ax.set_xlim(left=2005)

        # Add a legend
        leg = plt.legend(
            ncol=2,
            loc="upper right",
            frameon=False,
            fontsize=11.5,
            bbox_to_anchor=(1.01, 0.97),
            bbox_transform=fig.transFigure,
        )

        # Save
        plt.savefig(filename, dpi=300)


def monthly_actuals_this_quarter(report, filename):
    """
    Plot the monthly revenue/spending actuals for this quarter.

    This is a two panel chart with line charts showing the monthly totals 
    for the total cash receipts and total disbursements from the General Fund.

    Parameters
    ----------
    report : CashReport
        the cash report object
    filename : str
        the file name to save the file to
    """
    CURRENT_FY = report.year
    CURRENT_Q = report.quarter

    this_year = f"FY{str(CURRENT_FY)[2:]}"
    last_year = f"FY{str(CURRENT_FY - 1)[2:]}"

    quarter_months = [
        month.capitalize()
        for month in get_fiscal_months()[(CURRENT_Q - 1) * 3 : CURRENT_Q * 3]
    ]

    # load the data
    data = report.compare_to_last_year()
    sel = data["Name"].isin(["Total Disbursements", "Total Cash Receipts"])
    sel &= data["Month"].isin(quarter_months)
    data = data.loc[sel]

    with plt.style.context(default_style):

        # loop over tags
        tags = ["Revenue", "Spending"]
        labels = ["Cash Revenue", "Cash Spending"]
        for i, tag in enumerate(tags):

            # initalize the figure/axes
            grid_kws = dict(
                top=0.8,
                left=0.12,
                right=0.9,
                bottom=0.12,
                width_ratios=[3, 1],
                wspace=0.3,
            )
            fig, axs = plt.subplots(
                ncols=2, nrows=1, figsize=(5, 3), gridspec_kw=grid_kws
            )

            # select this type of data
            df = data.query(f"Kind == '{tag}'")

            ax = axs[0]

            # plot last year
            color = palette["medium-gray"]
            ax.plot(
                df["Month"],
                df[last_year],
                label=last_year,
                color=color,
                marker="o",
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.5,
                markersize=6,
                clip_on=False,
                lw=2.25,
                zorder=10,
            )

            # plot this year
            if tag == "Revenue":
                color = "black"
            else:
                color = palette["love-park-red"]
            ax.plot(
                df["Month"],
                df[this_year],
                label=this_year,
                color=color,
                marker="o",
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.5,
                markersize=6,
                lw=2.25,
                zorder=10,
                clip_on=False,
            )

            # format
            ax.set_xticks(df["Month"].tolist())
            plt.setp(ax.get_xticklabels(), fontsize=14)
            ax.set_yticklabels(
                [utils.format_currency(x, "{:,.0f}M") for x in ax.get_yticks()],
                fontsize=11,
            )

            # add a legend
            leg = ax.legend(
                ncol=2,
                loc="lower right",
                bbox_to_anchor=(1.05, 0.95),
                frameon=False,
                bbox_transform=ax.transAxes,
                fontsize=12,
            )

            # add the text
            ax = axs[1]
            ax.axis("off")
            total = df[this_year].sum()
            diff = df[this_year].sum() - df[last_year].sum()
            growth = diff / df[last_year].sum() * 100

            ax.text(
                0.5,
                0.95,
                f"Total Q{report.quarter} {tag}",
                ha="center",
                va="center",
                fontsize=14,
                weight="bold",
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                0.7,
                this_year + "\n" + "${:,.1f}M".format(total),
                ha="center",
                va="center",
                fontsize=14,
            )
            color = palette["love-park-red"] if diff < 0 else palette["phanatic-green"]
            ax.text(
                0.5,
                0.4,
                "${:,.1f}M".format(abs(diff)),
                color=color,
                fontsize=14,
                va="center",
                ha="center",
            )
            if diff < 0:
                t = f"less than {last_year}"
            else:
                t = f"more than {last_year}"

            ax.text(0.5, 0.3, t, fontsize=14, va="center", ha="center", color=color)

            ax.text(
                0.5,
                0.2,
                "(+%.1f%%)" % growth,
                ha="center",
                color=color,
                fontsize=14,
                va="top",
            )

            # add a title
            fig.text(
                0.5,
                0.94,
                f"General Fund Cash {tag} in {this_year} Q{CURRENT_Q} vs. {last_year} Q{CURRENT_Q}",
                weight="bold",
                fontsize=14,
                ha="center",
            )

            basename, ext = os.path.splitext(filename)
            plt.savefig(f"{basename}_{tag}{ext}", dpi=300)


def historical_accuracy_of_revenue_projection(report, filename):
    """
    Plot the historical accuracy of the current quarter's projection 
    for the annual revenue total.

    This is a line chart which shows the projected year-over-year change
    from the current quarter as well as the actual year-over-year change.

    Parameters
    ----------
    report : CashReport
        the cash report object
    filename : str
        the file name to save the file to
    """
    _historical_projection_accuracy(report, filename, "revenue")


def historical_accuracy_of_spending_projection(report, filename):
    """
    Plot the historical accuracy of the current quarter's projection 
    for the annual spending total.

    This is a line chart which shows the projected year-over-year change
    from the current quarter as well as the actual year-over-year change.

    Parameters
    ----------
    report : CashReport
        the cash report object
    filename : str
        the file name to save the file to
    """
    _historical_projection_accuracy(report, filename, "spending")


def _historical_projection_accuracy(report, filename, kind):
    """
    Internal function to plot the historical accuracy of the 
    current quarter's projection for the annual revenue/spending total.
    """
    kind = kind.lower()
    assert kind in ["revenue", "spending"]

    # load the data
    df = report.actual_vs_projected_changes()

    with plt.style.context(default_style):

        bottom = 0.28 if report.quarter == 1 else 0.24

        # initialize the figure
        grid_kws = dict(left=0.18, right=0.95, top=0.825, bottom=bottom, hspace=0.95)
        fig, ax = plt.subplots(figsize=(4, 2.7), gridspec_kw=grid_kws)

        if kind == "revenue":
            name = "Total Cash Receipts"
            label = "Year-over-Year Change in\nGeneral Fund Revenue"
        else:
            name = "Total Disbursements"
            label = "Year-over-Year Change in\nGeneral Fund Expenditures"

        # trim the data
        data = df.loc[(df["Name"] == name)]

        # plot projected change
        color = palette["dark-gray"]
        ax.plot(
            data["Fiscal Year"],
            data["Projected Change"],
            label=f"Q{report.quarter} Projection",
            color=color,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.5,
            markersize=5,
            lw=2,
            linestyle="dashed",
            zorder=2,
        )

        # plot actual change
        color = palette["electric-blue"]
        ax.plot(
            data["Fiscal Year"],
            data["Actual Change"],
            label="Actual Change",
            color=color,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.5,
            markersize=5,
            lw=2,
            zorder=10,
        )

        # add zero line
        ax.axhline(y=0, c=palette["medium-gray"], lw=2, zorder=1)

        # Format x-axis
        ax.set_xticks(data["Fiscal Year"].tolist())
        ax.set_xlabel("Fiscal Year", fontsize=11, weight="bold")
        plt.setp(ax.get_xticklabels(), fontsize=11, rotation=90)

        # Format y-axis
        if kind == "revenue":
            ylims = (-400, 400)
        else:
            ylims = (-400, 600)
        PAD = 25
        ax.set_ylim(ylims[0] - PAD, ylims[1] + PAD)
        ax.set_yticks(np.arange(ylims[0], ylims[1] + 1, 200))
        ax.set_yticklabels(
            [utils.format_currency(x, "{:,.0f}M") for x in ax.get_yticks()], fontsize=11
        )

        # Add a y-axis label
        ax.text(
            0.005,
            0.99,
            label,
            weight="bold",
            fontsize=10,
            transform=fig.transFigure,
            ha="left",
            va="top",
        )

        # Add a legend
        ax.legend(
            ncol=1,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.02),
            frameon=False,
            bbox_transform=fig.transFigure,
            fontsize=9,
        )

        if report.quarter == 1:
            caption = "Note: Q1 projection for Fiscal Year 2015 now shown due to the budgeted sale of Philadelphia Gas Works that did not occur."
            fig.text(
                0.0, 0.005, caption, ha="left", va="bottom", color="#666666", fontsize=5
            )

        # save
        plt.savefig(filename, dpi=300)
