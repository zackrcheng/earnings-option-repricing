import pandas as pd
import numpy as np

stage2 = pd.read_csv("optionm/optionm_stage2_skew.csv")

# 1. Verify delta_put is OTM (should be around -0.10 to -0.30)
print("delta_put  :", stage2["delta_put"].describe())
print("delta_call :", stage2["delta_call"].describe())

# 2. Verify LN_IMPSKEW ≈ ln(iv_put) - LN_IMPVOL
# If this identity holds, LN_IMPVOL = ln(IV_ATM) is confirmed
stage2["check_skew"] = np.log(stage2["iv_put"]) - stage2["LN_IMPVOL"]
diff = (stage2["check_skew"] - stage2["LN_IMPSKEW"]).abs()
print(f"\nMax deviation of IMPSKEW identity: {diff.max():.6f}")
print(f"Mean deviation                   : {diff.mean():.6f}")
# Should be ~0.0 if interpretation is correct