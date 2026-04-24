from pathlib import Path

# ---------------------------------------------------------------------------
# Paths  (anchored to repo root so notebooks run from any working directory)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = REPO_ROOT / "data" / "panel_stage5_V2_final.csv"
OUT_DIR   = REPO_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Main event window
# ---------------------------------------------------------------------------
PRE_DAY = -1
POST_DAY = 1
WINDOW_LABEL = f"[{PRE_DAY},{POST_DAY}]"

# ---------------------------------------------------------------------------
# Robustness / appendix windows
# ---------------------------------------------------------------------------
ALT_WINDOWS = [(-2, 2), (-1, 5)]

# ---------------------------------------------------------------------------
# Winsorization
# ---------------------------------------------------------------------------
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99
WINSOR_MIN_GROUP_N = 20

# ---------------------------------------------------------------------------
# Regression / inference settings
# ---------------------------------------------------------------------------
MAIN_CLUSTER_VAR = "secid"
FMB_DATE_COL = "event_month"
MIN_CS_OBS = 20
NW_LAGS = 4

# ---------------------------------------------------------------------------
# Control variable lists (order matters for table display)
# ---------------------------------------------------------------------------
BASE_CONTROLS = [
    "PreIV", "LN_MRKCAP_pre", "ret_pre", "LN_VOL_pre", "LN_PRC_pre"
]
RICH_OPTION_CONTROLS = [
    "Dispersion_pre", "LN_PC_OI_pre", "LN_PC_VLM_pre",
    "LN_TOTALVAR_pre", "IMPKURT_pre", "LN_EXPTIME_pre",
]
ROBUSTNESS_CONTROLS = ["AssetTurnover_pre"]
SHOCK_PROXY = ["Abs_Ret"]

# ---------------------------------------------------------------------------
# Appendix toggles
# ---------------------------------------------------------------------------
RUN_APPENDIX_ALT_WINDOWS = True
RUN_APPENDIX_DOUBLE_SORT = True
RUN_APPENDIX_FMB = True
RUN_APPENDIX_IMPKURT = True

# ---------------------------------------------------------------------------
# Required panel columns (validated on load)
# ---------------------------------------------------------------------------
REQUIRED_COLS = [
    "secid", "event_date", "date", "rel_day",
    "LN_IMPSKEW", "LN_IMPVOL", "LN_VIX", "LN_MRKCAP",
    "ret", "vol", "prc",
]
