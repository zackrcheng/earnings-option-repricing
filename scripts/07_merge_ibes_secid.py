import pandas as pd

# ── Phase 1.5 — Production merge: attach date-validated SECID to events ──────
secid_map = pd.read_csv("optionm/secid_map.csv",
                        parse_dates=["effect_date", "expire_date"])

events = pd.read_csv("ibes/ibes_stage1_events.csv",
                     parse_dates=["anndats"])

events["cusip8"] = events["cusip_x"].astype(str).str.strip().str[:8]

print(f"Events before merge : {len(events):,}")

# Left merge — one row per event × per effective SECID window
merged = events.merge(secid_map, on="cusip8", how="left")

# Date-constrained: keep only the window where the event date actually falls
merged = merged[
    (merged["anndats"] >= merged["effect_date"]) &
    (merged["anndats"] <= merged["expire_date"])
].copy()

n_lost = len(events) - len(merged)
print(f"Events after merge  : {len(merged):,}")
print(f"Events dropped      : {n_lost}  (no valid SECID for announcement date)")
print(merged[["ticker", "cname", "cusip8", "secid", "anndats"]].head(10))

merged.to_csv("ibes/ibes_stage2_with_secid.csv", index=False)
print(f"\n✅  Saved {len(merged):,} rows → ibes/ibes_stage2_with_secid.csv")