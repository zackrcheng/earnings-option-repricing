import numpy as np
import pandas as pd

from src.config import NW_LAGS
from src.stats import newey_west_mean_tstat


def assign_quantile_by_date(data, sort_col, nport=5, date_col="event_date"):
    out = data.sort_values([date_col, sort_col]).copy()

    def _qcut_safe(s):
        valid = s.dropna()
        if valid.nunique() < nport:
            return pd.Series(np.nan, index=s.index)
        try:
            return pd.qcut(valid, q=nport, labels=False, duplicates="drop") + 1
        except Exception:
            ranks = valid.rank(method="first")
            return pd.qcut(ranks, q=nport, labels=False, duplicates="drop") + 1

    out["portfolio"] = out.groupby(date_col)[sort_col].transform(_qcut_safe)
    return out


def portfolio_sort_table(data, sort_col, outcome_col, nport=5, date_col="event_date"):
    tmp = data[[date_col, sort_col, outcome_col]].dropna().copy()
    tmp = assign_quantile_by_date(tmp, sort_col=sort_col, nport=nport, date_col=date_col)
    tmp = tmp.dropna(subset=["portfolio"]).copy()
    tmp["portfolio"] = tmp["portfolio"].astype(int)

    if tmp.empty:
        return pd.DataFrame()

    port_ts = (
        tmp.groupby([date_col, "portfolio"])[outcome_col]
        .mean()
        .reset_index()
        .pivot(index=date_col, columns="portfolio", values=outcome_col)
        .sort_index()
    )

    rows = []
    for p in port_ts.columns:
        mean_, t_, n_ = newey_west_mean_tstat(port_ts[p], lags=NW_LAGS)
        rows.append({"portfolio": f"P{p}", "mean": mean_, "t_stat_NW": t_, "T": n_})

    if 1 in port_ts.columns and nport in port_ts.columns:
        hl = port_ts[nport] - port_ts[1]
        mean_, t_, n_ = newey_west_mean_tstat(hl, lags=NW_LAGS)
        rows.append({"portfolio": f"P{nport}-P1", "mean": mean_, "t_stat_NW": t_, "T": n_})

    return pd.DataFrame(rows)
