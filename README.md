# State-Dependent Option Market Repricing Around Earnings Announcements

## Abstract

This paper examines how pre-announcement market conditions shape the options market's repricing response to earnings announcements. Using a panel of ~19,000 earnings events, we test two hypotheses: (H1) pre-announcement implied skew predicts the magnitude of skew revision around the event, and (H2) aggregate uncertainty (VIX) predicts implied volatility compression. Results from clustered pooled OLS and Fama-MacBeth regressions show that both effects are economically large and highly significant after controlling for firm characteristics and option-market microstructure variables.

---

## Data Sources

| Source | Variable group | Access |
|---|---|---|
| **I/B/E/S** (via WRDS) | EPS actuals, analyst forecasts, dispersion | WRDS subscription |
| **OptionMetrics** (via WRDS) | Implied volatility, implied skew, implied kurtosis, open interest, volume | WRDS subscription |
| **CRSP** (via WRDS) | Daily returns, volume, price, market cap | WRDS subscription |
| **Compustat** (via WRDS) | Quarterly sales, assets | WRDS subscription |
| **CBOE VIX** | Daily VIX index | Free — `scripts/04_download_vix.py` |

> **Raw data is not committed to this repo.** Place downloaded files in `data/raw/` and the final merged panel at the path set in `src/config.py → DATA_PATH`.

---

## Repository Structure

```
.
├── src/                        # Shared Python modules (imported by all notebooks)
│   ├── config.py               # All parameters: paths, window bounds, thresholds
│   ├── data_utils.py           # Panel loading, event-dataset construction, winsorisation
│   ├── stats.py                # OLS, Fama-MacBeth, Newey-West, summary/missingness tables
│   └── portfolio.py            # Quintile sort construction and NW t-stat tables
│
├── notebooks/                  # Analysis notebooks (run in order)
│   ├── 01_data_validation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_portfolio_sorts.ipynb
│   ├── 04_main_regressions.ipynb
│   └── 05_robustness.ipynb
│
├── scripts/                    # WRDS data pull scripts (run once on a WRDS machine)
│   ├── 01_download_ibes_actuals.py
│   ├── 02_download_ibes_summary.py
│   ├── 03_download_crsp.py
│   ├── 04_download_vix.py
│   ├── 05_download_compustat.py
│   ├── 06_build_secid_map.py
│   ├── 07_merge_ibes_secid.py
│   ├── 08_filter_ibes.py
│   ├── 09_merge_actuals_summary.py
│   ├── 10a_optionmetrics_phase2_v7.py
│   ├── 10b_optionmetrics_phase2_v11.py
│   ├── 11_build_controls.py
│   └── 12_build_impkurt.py
│
├── data/                       # Raw data (gitignored)
│   └── raw/
│
├── outputs/                    # Generated tables and figures (gitignored)
│   ├── tables/
│   └── figures/
│
├── docs/                       # Supporting documents
│   ├── panel_column_definitions.pdf
│   └── final_report.docx
│
└── archive/                    # Stale scripts kept for reference
    └── diagnostics/
```

---

## Reproducing Results

### 1. Install dependencies

```bash
pip install pandas numpy statsmodels matplotlib
```

### 2. Configure paths

Edit `src/config.py`:
```python
DATA_PATH = Path("path/to/panel_stage5_V2_final.csv")
OUT_DIR   = Path("outputs")
```

### 3. Pull data (WRDS access required)

Run scripts in `scripts/` in numbered order on a machine with WRDS access:
```bash
python scripts/01_download_ibes_actuals.py
# ... through ...
python scripts/12_build_impkurt.py
```
This produces the final panel file pointed to by `DATA_PATH`.

### 4. Run notebooks in order

```bash
jupyter nbconvert --to notebook --execute notebooks/01_data_validation.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_eda.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_portfolio_sorts.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_main_regressions.ipynb
jupyter nbconvert --to notebook --execute notebooks/05_robustness.ipynb
```

Or open them interactively in JupyterLab and run top-to-bottom.

Output CSVs land in `outputs/tables/`.

### 5. Toggle appendix sections

In `src/config.py`, set any of these to `False` to skip that appendix:
```python
RUN_APPENDIX_FMB         = True   # Fama-MacBeth
RUN_APPENDIX_ALT_WINDOWS = True   # windows [-2,+2] and [-1,+5]
RUN_APPENDIX_IMPKURT     = True   # implied kurtosis robustness
```

---

## Variable Glossary

See `docs/panel_column_definitions.pdf` for the full panel schema.

| Variable | Definition |
|---|---|
| `SR` | Skew Revision: −(LN_IMPSKEW_post − LN_IMPSKEW_pre) |
| `UR` | Uncertainty Revision: −(LN_IMPVOL_post − LN_IMPVOL_pre) |
| `PreSkew` | Log implied skew at day −1 |
| `PreIV` | Log implied volatility at day −1 |
| `LN_VIX_pre` | Log VIX at day −1 |
| `event_window` | Main window: [−1, +1] relative to announcement date |
