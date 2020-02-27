from . import general_fund, fund_balances
import os
import shutil


def make_charts(fiscalYear, currentQuarter, output_dir):
    """

    """
    from qcmr.analysis import CashReport

    # Skip these plots if the current quarter is Q4
    Q4_skips = [
        "historical_accuracy_of_EOY_projection",
        "change_in_EOY_balance_projection",
        "historical_accuracy_of_spending_projection",
        "historical_accuracy_of_revenue_projection",
    ]
    Q1_skips = [
        "actual_balance_vs_last_quarters_projection",
        "change_in_EOY_balance_projection",
        "balance_changes_since_last_quarter",
        "revenue_changes_since_last_quarter",
        "spending_changes_since_last_quarter",
    ]

    # Initialize the cash report object
    report = CashReport(fiscalYear, currentQuarter)

    def _make_charts(label, module):
        dirname = os.path.join(output_dir, label)
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.makedirs(dirname)

        for f in dir(module):
            if not f.startswith("__"):
                if currentQuarter == 4 and f in Q4_skips:
                    continue
                if f == "fund_balance_revisions" and currentQuarter != 4:
                    continue
                if currentQuarter == 1 and f in Q1_skips:
                    continue
                plotter = getattr(module, f)
                if callable(plotter):

                    path = os.path.join(dirname, f + ".png")
                    plotter(report, path)

    # Make the charts for general fund and fund balances
    for label, module in zip(
        ["general_fund", "fund_balances"], [general_fund, fund_balances]
    ):
        _make_charts(label, module)

