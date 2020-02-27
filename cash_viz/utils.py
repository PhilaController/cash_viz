MONTHS = [
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


def format_currency(x, fmt, plus_sign=False):
    """
    Format the input value as currency, optionally including a
    plus sign.
    """
    negative = x < 0
    x = abs(x)
    if not plus_sign:
        sign = "$" if not negative else "-$"
    else:
        if x == 0:
            sign = "$"
        else:
            sign = "+$" if not negative else "-$"
    return sign + fmt.format(x)


def get_quarter_months(quarter):
    """
    Get the months for the specified quarter
    """
    return MONTHS[3 * (quarter - 1) : 3 * quarter]


def get_actual_months(quarter):
    """
    Get the months with actual values.
    """
    return MONTHS[: 3 * quarter]


def get_projected_months(quarter):
    """
    Get the months with projected values.
    """
    return MONTHS[3 * quarter - 1 :]


def make_tag(fiscalYear, quarter):
    """
    Utility function to make the fiscal year tag.
    """
    return f"FY{str(fiscalYear)[2:]} Q{quarter}"
