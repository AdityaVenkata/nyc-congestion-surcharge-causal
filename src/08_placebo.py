"""
Step 9: Can we break our own result?

Two attacks on the short-trip finding.

  1. PLACEBO IN TIME. Re-run the analysis on pre-period data only, pretending
     the surcharge started on a date when nothing happened. If a fake treatment
     date produces an effect, the method is picking up noise and the real
     estimate means nothing.

  2. RANDOMISATION INFERENCE (placebo in space). Randomly reassign which zones
     are "treated", keeping the group sizes, and re-estimate. Repeat 1,000
     times to build the distribution of estimates under the null that zone
     assignment does not matter. The exact p-value is the share of placebo
     estimates at least as extreme as the real one.

     WHY THIS MATTERS HERE: we have only 8 control zones, and cluster-robust
     standard errors are unreliable with few clusters. They tend to be far too
     small, which makes p-values look better than they are. Randomisation
     inference makes no asymptotic assumption about cluster count, so it is the
     right tool for this sample. If the clustered p-value and the exact p-value
     agree, the finding is solid. If they diverge, trust the exact one.

SPEED NOTE: fitting 56 zone dummies and 243 date dummies a thousand times is
slow. The panel is perfectly balanced (56 zones x 243 days), so two-way fixed
effects can be absorbed exactly by demeaning:

    y~ = y - zone_mean - date_mean + grand_mean

We verify this reproduces the statsmodels estimate before relying on it.

Run:  python src/08_placebo.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from zones import TREATMENT_DATE

PANEL = "data/processed/panel.parquet"
FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

METRIC = "share_short_trips"
SPEC_B_ESTIMATE = -0.01225        # from src/05_estimate.py, spec (B)
N_PERM = 1000
RNG = np.random.default_rng(7)

panel = pd.read_parquet(PANEL).dropna(subset=[METRIC])
panel["trip_date"] = pd.to_datetime(panel["trip_date"])
treat_date = pd.Timestamp(TREATMENT_DATE)


def demean(df, series):
    """Absorb zone and date fixed effects on a balanced panel."""
    s = pd.Series(series, index=df.index)
    return (s
            - s.groupby(df["zone_id"]).transform("mean")
            - s.groupby(df["trip_date"]).transform("mean")
            + s.mean()).values


def estimate(df, treated, post, with_trend=True):
    """Two-way FE estimate of the treated-by-post coefficient."""
    y = demean(df, df[METRIC].values)
    cols = [demean(df, treated * post)]
    if with_trend:
        cols.append(demean(df, treated * df["day_index"].values))
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta[0]


# ---------------------------------------------------------------------------
# Verify the fast path against the known statsmodels answer.
# ---------------------------------------------------------------------------
print("=" * 74)
print("VERIFICATION")
print("=" * 74)

check = estimate(panel, panel["treated"].values.astype(float),
                 panel["post"].values.astype(float))
print(f"statsmodels spec (B):    {SPEC_B_ESTIMATE:+.5f}")
print(f"demeaned fast path:      {check:+.5f}")
print(f"difference:              {abs(check - SPEC_B_ESTIMATE):.2e}")
if abs(check - SPEC_B_ESTIMATE) > 5e-4:
    print("WARNING: fast path does not match. Do not trust what follows.")
else:
    print("match confirmed, proceeding")

# ---------------------------------------------------------------------------
# 1. Placebo in time.
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("PLACEBO IN TIME (pre-period only, fake treatment dates)")
print("=" * 74)
print("the surcharge did not exist on any of these dates, so every estimate")
print("below should be indistinguishable from zero\n")

pre = panel[panel.trip_date < treat_date].copy()
print(f"pre-period sample: {len(pre):,} zone-days "
      f"({pre.trip_date.min().date()} to {pre.trip_date.max().date()})\n")

print(f"{'fake date':>14} {'estimate':>12}   {'vs real':>10}")
for fake in ["2018-11-01", "2018-11-15", "2018-12-01", "2018-12-15", "2019-01-01"]:
    fd = pd.Timestamp(fake)
    sub = pre.copy()
    post = (sub["trip_date"] >= fd).values.astype(float)
    if post.sum() == 0 or post.sum() == len(post):
        continue
    est = estimate(sub, sub["treated"].values.astype(float), post)
    ratio = est / SPEC_B_ESTIMATE
    print(f"{fake:>14} {est:>+12.5f}   {ratio:>9.2f}x")

print(f"\nreal estimate (2 Feb 2019): {SPEC_B_ESTIMATE:+.5f}")

# ---------------------------------------------------------------------------
# 2. Randomisation inference.
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print(f"RANDOMISATION INFERENCE ({N_PERM:,} random zone assignments)")
print("=" * 74)

zone_ids = panel["zone_id"].unique()
n_treated = panel.loc[panel.treated == 1, "zone_id"].nunique()
post_vec = panel["post"].values.astype(float)

print(f"{len(zone_ids)} zones, reassigning {n_treated} to treatment each draw")

placebos = np.empty(N_PERM)
for i in range(N_PERM):
    fake_treated_zones = RNG.choice(zone_ids, size=n_treated, replace=False)
    treated = panel["zone_id"].isin(fake_treated_zones).values.astype(float)
    placebos[i] = estimate(panel, treated, post_vec)
    if (i + 1) % 250 == 0:
        print(f"  {i+1:,} draws done")

exact_p = np.mean(np.abs(placebos) >= abs(SPEC_B_ESTIMATE))

print(f"\nplacebo distribution")
print(f"  mean:                  {placebos.mean():+.5f}")
print(f"  std deviation:         {placebos.std(ddof=1):.5f}")
print(f"  2.5th percentile:      {np.percentile(placebos, 2.5):+.5f}")
print(f"  97.5th percentile:     {np.percentile(placebos, 97.5):+.5f}")
print(f"\nreal estimate:           {SPEC_B_ESTIMATE:+.5f}")
print(f"placebos at least as extreme: "
      f"{int(exact_p * N_PERM)} of {N_PERM}")
print(f"EXACT P-VALUE:           {exact_p:.4f}")
print(f"clustered p-value was:   0.0212")

if exact_p < 0.05:
    print("\nThe finding survives. Randomly chosen zone assignments almost never")
    print("produce an effect this large, so the real assignment is doing the work.")
else:
    print("\nThe finding does NOT survive. Random zone assignments produce effects")
    print("this large often enough that the clustered p-value was overstated.")

# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.hist(placebos, bins=45, color="#bdc3c7", edgecolor="white", linewidth=0.5)
ax.axvline(SPEC_B_ESTIMATE, color="#c0392b", linewidth=2)
ax.axvline(-SPEC_B_ESTIMATE, color="#c0392b", linewidth=1, linestyle=":")
ax.annotate(f"actual estimate\n{SPEC_B_ESTIMATE:+.5f}\nexact p = {exact_p:.3f}",
            xy=(SPEC_B_ESTIMATE, ax.get_ylim()[1] * 0.72),
            xytext=(-118, 0), textcoords="offset points",
            fontsize=9, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", linewidth=1.2))
ax.set_xlabel("estimated effect under random zone assignment")
ax.set_ylabel("frequency")
ax.set_title(f"Randomisation inference: {N_PERM:,} placebo assignments vs the real one",
             fontsize=10)
ax.grid(alpha=0.22)
plt.tight_layout()
path = os.path.join(FIGDIR, "05_randomisation_inference.png")
plt.savefig(path, dpi=140)
print(f"\nchart written to {path}")

pd.DataFrame({"placebo_estimate": placebos}).to_csv(
    "data/processed/placebo_draws.csv", index=False)

print("\n\nDone.")
