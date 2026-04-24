import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def missingness_table(data, columns):
    n = len(data)
    rows = []
    for col in columns:
        if col in data.columns:
            miss = int(data[col].isna().sum())
            rows.append({
                "variable": col,
                "missing_count": miss,
                "missing_pct": miss / n if n > 0 else np.nan,
            })
    return pd.DataFrame(rows).sort_values(
        ["missing_pct", "variable"], ascending=[False, True]
    )


def summary_table(data, columns):
    existing = [c for c in columns if c in data.columns]
    if not existing:
        return pd.DataFrame()
    desc = data[existing].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    desc = desc.rename(columns={
        "count": "N", "mean": "Mean", "std": "Std", "min": "Min",
        "1%": "P1", "5%": "P5", "50%": "Median",
        "95%": "P95", "99%": "P99", "max": "Max",
    })
    keep = [c for c in ["N", "Mean", "Std", "Min", "P1", "P5", "Median", "P95", "P99", "Max"]
            if c in desc.columns]
    return desc[keep].reset_index().rename(columns={"index": "variable"})


def newey_west_mean_tstat(series, lags=4):
    s = pd.Series(series).dropna()
    if len(s) < 5:
        return np.nan, np.nan, len(s)
    model = sm.OLS(s.values, np.ones((len(s), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    return float(model.params[0]), float(model.tvalues[0]), int(len(s))


def available_controls(data, candidates):
    return [c for c in candidates if c in data.columns]


def formula_escape(name):
    return f'Q("{name}")' if any(ch in name for ch in ("-", "+", " ", "/")) else name


def run_clustered_pooled_ols(data, depvar, regressors, cluster_var="secid", spec_name=None):
    rhs = list(dict.fromkeys(regressors))
    needed = [depvar, cluster_var, "event_quarter"] + rhs
    if "SIC2_pre" in data.columns:
        needed.append("SIC2_pre")

    sample = data[[c for c in needed if c in data.columns]].dropna().copy()

    fe_terms = []
    if "SIC2_pre" in sample.columns:
        fe_terms.append("C(SIC2_pre)")
    fe_terms.append("C(event_quarter)")

    rhs_formula = " + ".join([formula_escape(x) for x in rhs] + fe_terms)
    formula = f"{formula_escape(depvar)} ~ {rhs_formula}"

    model = smf.ols(formula, data=sample).fit(
        cov_type="cluster",
        cov_kwds={"groups": sample[cluster_var]},
    )

    coef_rows = [
        {
            "spec": spec_name,
            "depvar": depvar,
            "term": term,
            "coef": model.params.get(term, np.nan),
            "std_err": model.bse.get(term, np.nan),
            "t_stat": model.tvalues.get(term, np.nan),
            "p_value": model.pvalues.get(term, np.nan),
        }
        for term in rhs
    ]

    meta = pd.DataFrame([{
        "spec": spec_name,
        "depvar": depvar,
        "N": int(model.nobs),
        "adj_R2": model.rsquared_adj,
        "clusters": int(sample[cluster_var].nunique()),
        "quarters": (
            int(sample["event_quarter"].nunique())
            if "event_quarter" in sample.columns else np.nan
        ),
    }])

    return pd.DataFrame(coef_rows), meta, model


def run_fama_macbeth(
    data, depvar, regressors, date_col="event_month", min_obs=20, lags=4, spec_name=None
):
    rhs = list(dict.fromkeys(regressors))
    needed = [date_col, depvar] + rhs
    sample = data[[c for c in needed if c in data.columns]].dropna().copy()

    threshold = max(min_obs, len(rhs) + 5)
    coef_rows = []

    for dt, g in sample.groupby(date_col):
        if len(g) < threshold:
            continue
        X = sm.add_constant(g[rhs].astype(float), has_constant="add")
        y = g[depvar].astype(float)
        try:
            fit = sm.OLS(y, X).fit()
        except Exception:
            continue
        row = {date_col: dt, "N_cs": len(g), "adj_R2_cs": fit.rsquared_adj}
        for term in ["const"] + rhs:
            row[term] = fit.params.get(term, np.nan)
        coef_rows.append(row)

    coef_ts = pd.DataFrame(coef_rows)
    if coef_ts.empty:
        raise ValueError(f"No valid cross-sections for FMB: {depvar} ~ {rhs}")

    summary = []
    for term in ["const"] + rhs:
        mean_, t_, T_ = newey_west_mean_tstat(coef_ts[term], lags=lags)
        summary.append({
            "spec": spec_name,
            "depvar": depvar,
            "term": term,
            "coef_mean": mean_,
            "t_stat_NW": t_,
            "T": T_,
            "avg_cs_N": coef_ts["N_cs"].mean(),
            "avg_cs_adj_R2": coef_ts["adj_R2_cs"].mean(),
        })

    return pd.DataFrame(summary), coef_ts
