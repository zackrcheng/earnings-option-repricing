import numpy as np
import pandas as pd

from src.config import (
    REQUIRED_COLS,
    WINSOR_LOWER,
    WINSOR_UPPER,
    WINSOR_MIN_GROUP_N,
)


def safe_log(series, lower=1e-8):
    return np.log(pd.to_numeric(series, errors="coerce").clip(lower=lower))


def winsorize_by_group(
    data, cols, group_col="event_date", lower=0.01, upper=0.99, min_group_n=20
):
    out = data.copy()
    for col in cols:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out.groupby(group_col)[col].transform(
            lambda s: (
                s.clip(s.quantile(lower), s.quantile(upper))
                if s.notna().sum() >= min_group_n
                else s
            )
        )
    return out


def load_panel(path):
    df = pd.read_csv(path)
    for col in ("event_date", "date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["event_id"] = df["secid"].astype(str) + "_" + df["event_date"].astype(str)
    return df


def build_event_level_dataset(data, pre_day=-1, post_day=1):
    flow = [
        {"step": "Raw panel rows", "N": int(len(data))},
        {"step": "Raw unique events", "N": int(data["event_id"].nunique())},
    ]

    pre = data.loc[data["rel_day"] == pre_day].copy()
    post = data.loc[data["rel_day"] == post_day].copy()

    flow.append({"step": f"Events with rel_day = {pre_day}", "N": int(pre["event_id"].nunique())})
    flow.append({"step": f"Events with rel_day = {post_day}", "N": int(post["event_id"].nunique())})

    pre_keep = [
        "event_id", "secid", "ticker", "event_date",
        "LN_IMPSKEW", "LN_IMPVOL", "LN_VIX", "LN_MRKCAP",
        "ret", "vol", "prc", "saleq", "atq",
        "SIC2", "QUARTER_FE",
        "Dispersion", "LN_PC_OI", "LN_PC_VLM",
        "TOTALVAR", "LN_TOTALVAR", "IMPKURT",
        "dte", "LN_EXPTIME",
    ]
    post_keep = ["event_id", "LN_IMPSKEW", "LN_IMPVOL", "ret", "vol", "prc"]

    pre = pre[[c for c in pre_keep if c in pre.columns]]
    post = post[[c for c in post_keep if c in post.columns]]

    event = pre.merge(post, on="event_id", suffixes=("_pre", "_post"), how="inner")
    flow.append({"step": "Events surviving pre/post merge", "N": int(len(event))})

    # Outcome variables
    event["SR"] = -(event["LN_IMPSKEW_post"] - event["LN_IMPSKEW_pre"])
    event["UR"] = -(event["LN_IMPVOL_post"] - event["LN_IMPVOL_pre"])
    event["UR_norm_by_PreIV"] = event["UR"] / event["LN_IMPVOL_pre"].abs().replace(0, np.nan)
    event["IV_post"] = event["LN_IMPVOL_post"]

    # Main regressors
    event["PreSkew"] = event["LN_IMPSKEW_pre"]
    event["PreIV"] = event["LN_IMPVOL_pre"]
    event["LN_VIX_pre"] = event["LN_VIX"]
    event["LN_MRKCAP_pre"] = event["LN_MRKCAP"]

    # Derived controls
    event["ret_pre"] = pd.to_numeric(event["ret_pre"], errors="coerce")
    event["LN_VOL_pre"] = safe_log(event["vol_pre"], lower=1)
    event["LN_PRC_pre"] = safe_log(event["prc_pre"].abs(), lower=0.01)
    event["Abs_Ret"] = event["ret_post"].abs()

    if {"saleq", "atq"}.issubset(event.columns):
        event["AssetTurnover_pre"] = (
            pd.to_numeric(event["saleq"], errors="coerce")
            / pd.to_numeric(event["atq"], errors="coerce")
        )

    rename_map = {
        "Dispersion": "Dispersion_pre",
        "LN_PC_OI": "LN_PC_OI_pre",
        "LN_PC_VLM": "LN_PC_VLM_pre",
        "LN_TOTALVAR": "LN_TOTALVAR_pre",
        "IMPKURT": "IMPKURT_pre",
        "LN_EXPTIME": "LN_EXPTIME_pre",
        "SIC2": "SIC2_pre",
        "QUARTER_FE": "QUARTER_FE_pre",
    }
    for old, new in rename_map.items():
        if old in event.columns and new not in event.columns:
            event[new] = event[old]

    event = event.replace([np.inf, -np.inf], np.nan)

    winsor_cols = [
        "SR", "UR", "UR_norm_by_PreIV", "IV_post",
        "PreSkew", "PreIV", "LN_VIX_pre", "LN_MRKCAP_pre",
        "ret_pre", "LN_VOL_pre", "LN_PRC_pre", "Abs_Ret",
        "Dispersion_pre", "LN_PC_OI_pre", "LN_PC_VLM_pre",
        "LN_TOTALVAR_pre", "IMPKURT_pre", "LN_EXPTIME_pre",
        "AssetTurnover_pre",
    ]
    event = winsorize_by_group(
        event,
        cols=[c for c in winsor_cols if c in event.columns],
        group_col="event_date",
        lower=WINSOR_LOWER,
        upper=WINSOR_UPPER,
        min_group_n=WINSOR_MIN_GROUP_N,
    )

    event["event_month"] = event["event_date"].dt.to_period("M").astype(str)
    event["event_quarter"] = event["event_date"].dt.to_period("Q").astype(str)

    key_vars = ["SR", "UR", "PreSkew", "PreIV", "LN_VIX_pre"]
    flow.append({"step": "Non-missing key vars", "N": int(event.dropna(subset=key_vars).shape[0])})
    flow.append({"step": "Final event-level observations", "N": int(len(event))})

    return event, pd.DataFrame(flow)


def build_h2_segment_dataset(data):
    """Extends the base [-1,+1] event dataset with IV at days +3 and +5
    to decompose UR into three sequential segments."""
    base_event, _ = build_event_level_dataset(data, pre_day=-1, post_day=1)

    d3 = (
        data.loc[data["rel_day"] == 3, ["event_id", "LN_IMPVOL"]]
        .rename(columns={"LN_IMPVOL": "LN_IMPVOL_3"})
    )
    d5 = (
        data.loc[data["rel_day"] == 5, ["event_id", "LN_IMPVOL"]]
        .rename(columns={"LN_IMPVOL": "LN_IMPVOL_5"})
    )

    seg = (
        base_event
        .merge(d3, on="event_id", how="left")
        .merge(d5, on="event_id", how="left")
    )
    seg["UR_1"] = -(seg["LN_IMPVOL_post"] - seg["PreIV"])
    seg["UR_23"] = -(seg["LN_IMPVOL_3"] - seg["LN_IMPVOL_post"])
    seg["UR_45"] = -(seg["LN_IMPVOL_5"] - seg["LN_IMPVOL_3"])
    return seg
