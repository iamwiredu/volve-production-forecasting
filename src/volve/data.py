"""Load and clean the Volve daily production data."""
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "Volve production data.xlsx"
DAILY_SHEET = "Daily Production Data"


def load_daily(path=DEFAULT_PATH):
    """Read the daily production sheet with dates parsed."""
    df = pd.read_excel(path, sheet_name=DAILY_SHEET)
    df["DATEPRD"] = pd.to_datetime(df["DATEPRD"])
    return df.sort_values(["NPD_WELL_BORE_NAME", "DATEPRD"])


def producer(df, well):
    """Return one production well's days on stream with positive oil rate.

    `well` is the NPD wellbore name, e.g. "15/9-F-12".
    """
    w = df[(df["NPD_WELL_BORE_NAME"] == well) & (df["WELL_TYPE"] == "OP")]
    w = w[(w["ON_STREAM_HRS"] > 0) & (w["BORE_OIL_VOL"] > 0)]
    return w.set_index("DATEPRD")


def monthly_oil_rate(well_df):
    """Average calendar-day oil rate (Sm3/d) per month.

    Monthly averaging smooths shut-ins and choke changes, which decline-curve
    analysis assumes away.
    """
    monthly = well_df["BORE_OIL_VOL"].resample("MS").sum()
    days = monthly.index.days_in_month
    return (monthly / days).loc[lambda s: s > 0]
