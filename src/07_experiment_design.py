"""
Step 8: If we could have run this as a real experiment, how would we size it?

The city could not randomise the surcharge. Uber can randomise a price change.
This step asks what that experiment would have needed to look like.

RANDOMISATION UNIT: the zone-day. This is a SWITCHBACK design, and it is what
marketplace companies actually use for pricing tests. Randomising individual
riders would violate SUTVA: change the price for one rider and you change
driver availability for everyone nearby, so the control group is contaminated
by the treatment. Randomising geographic-time cells keeps the interference
inside the unit.

Three pieces:

  1. POWER / MDE   How small an effect could we detect, given N zone-days?
                   Equivalently: how many days must the test run?

  2. PEEKING       What happens if you check for significance every day and
                   stop at the first significant result? Simulated under the
                   null, where the true effect is exactly zero, so every
                   "significant" result is a false positive.

  3. CUPED         Use each zone's pre-experiment behaviour to strip out
                   predictable variance, then report how much shorter the
                   experiment becomes.

Run:  python src/07_experiment_design.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from zones import TREATMENT_DATE

PANEL = "data/processed/panel.parquet"
FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

METRIC = "share_short_trips"
ALPHA = 0.05
POWER = 0.80
OBSERVED_EFFECT = 0.0127          # from spec (B), the preferred estimate
RNG = np.random.default_rng(42)

panel = pd.read_parquet(PANEL)
panel["trip_date"] = pd.to_datetime(panel["trip_date"])
pre = panel[panel.trip_date < pd.Timestamp(TREATMENT_DATE)].dropna(subset=[METRIC])

n_zones = pre.zone_id.nunique()
per_arm_per_day = n_zones // 2

print("=" * 74)
print("BASELINE (pre-period, zone-day level)")
print("=" * 74)
sigma_raw = pre[METRIC].std(ddof=1)
print(f"metric:                  {METRIC}")
print(f"zone-days available:     {len(pre):,}")
print(f"zones:                   {n_zones}  ->  {per_arm_per_day} per arm per day")
print(f"mean:                    {pre[METRIC].mean():.5f}")
print(f"standard deviation:      {sigma_raw:.5f}")

# ---------------------------------------------------------------------------
# 1. CUPED. Split the pre-period in half: the first half gives each zone a
#    covariate, the second half is the outcome we would be measuring.
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("CUPED VARIANCE REDUCTION")
print("=" * 74)

midpoint = pre.trip_date.quantile(0.5)
covariate = (pre[pre.trip_date <= midpoint]
             .groupby("zone_id")[METRIC].mean().rename("x"))
outcome = pre[pre.trip_date > midpoint][["zone_id", METRIC]].copy()
joined = outcome.join(covariate, on="zone_id").dropna()

y, x = joined[METRIC].values, joined["x"].values
corr = np.corrcoef(y, x)[0, 1]
theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
y_adj = y - theta * (x - x.mean())

sigma_cuped = y_adj.std(ddof=1)
sigma_plain = y.std(ddof=1)
reduction = 1 - (sigma_cuped ** 2) / (sigma_plain ** 2)

print(f"correlation(pre, post):  {corr:+.4f}")
print(f"theta:                   {theta:.5f}")
print(f"sd without CUPED:        {sigma_plain:.5f}")
print(f"sd with CUPED:           {sigma_cuped:.5f}")
print(f"variance reduction:      {100*reduction:.1f}%")
print(f"mean unchanged:          {y.mean():.5f} -> {y_adj.mean():.5f}")
print("\nCUPED works here because a zone's past behaviour predicts its future")
print("behaviour. The correlation above is what makes the technique pay off;")
print("a weak correlation would leave the variance untouched.")

# ---------------------------------------------------------------------------
# 2. Minimum detectable effect.
# ---------------------------------------------------------------------------
z_a = stats.norm.ppf(1 - ALPHA / 2)
z_b = stats.norm.ppf(POWER)


def mde(n_per_arm, sigma):
    """Smallest effect detectable at ALPHA with POWER, two-sample."""
    return (z_a + z_b) * sigma * np.sqrt(2.0 / n_per_arm)


def days_needed(effect, sigma):
    """Days of switchback running required to detect `effect`."""
    n = 2 * ((z_a + z_b) * sigma / effect) ** 2
    return n / per_arm_per_day


print("\n" + "=" * 74)
print(f"MINIMUM DETECTABLE EFFECT  (alpha={ALPHA}, power={POWER:.0%})")
print("=" * 74)
print(f"{'days':>6} {'n/arm':>8} {'MDE plain':>12} {'MDE CUPED':>12}")
for d in [7, 14, 21, 28, 42, 60, 90]:
    n = d * per_arm_per_day
    print(f"{d:>6} {n:>8,} {mde(n, sigma_plain):>12.5f} {mde(n, sigma_cuped):>12.5f}")

d_plain = days_needed(OBSERVED_EFFECT, sigma_plain)
d_cuped = days_needed(OBSERVED_EFFECT, sigma_cuped)

print(f"\nTo detect the effect we actually observed ({OBSERVED_EFFECT:.4f}):")
print(f"  without CUPED:  {d_plain:.1f} days")
print(f"  with CUPED:     {d_cuped:.1f} days")
print(f"  saving:         {d_plain - d_cuped:.1f} days "
      f"({100*(1 - d_cuped/d_plain):.0f}% shorter)")

# ---------------------------------------------------------------------------
# 3. Peeking. Simulate under the null: there is genuinely no effect, so every
#    significant result is a false positive by construction.
# ---------------------------------------------------------------------------
N_SIMS = 10_000
MAX_DAYS = 28
FIRST_LOOK = 3

print("\n" + "=" * 74)
print(f"PEEKING SIMULATION  ({N_SIMS:,} experiments, true effect = 0)")
print("=" * 74)

k = per_arm_per_day
treat = RNG.normal(0, sigma_plain, size=(N_SIMS, MAX_DAYS, k))
ctrl = RNG.normal(0, sigma_plain, size=(N_SIMS, MAX_DAYS, k))

# Running sums so each day's test uses all data collected so far.
cum_t = np.cumsum(treat.sum(axis=2), axis=1)
cum_c = np.cumsum(ctrl.sum(axis=2), axis=1)
cum_t2 = np.cumsum((treat ** 2).sum(axis=2), axis=1)
cum_c2 = np.cumsum((ctrl ** 2).sum(axis=2), axis=1)

days = np.arange(1, MAX_DAYS + 1)
n = days * k

mean_t, mean_c = cum_t / n, cum_c / n
var_t = (cum_t2 - n * mean_t ** 2) / (n - 1)
var_c = (cum_c2 - n * mean_c ** 2) / (n - 1)
se = np.sqrt(var_t / n + var_c / n)
tstat = (mean_t - mean_c) / se

crit = stats.norm.ppf(1 - ALPHA / 2)
sig = np.abs(tstat) > crit

fixed_fpr = sig[:, MAX_DAYS - 1].mean()
peek_any = sig[:, FIRST_LOOK - 1:]
peek_fpr = peek_any.any(axis=1).mean()

print(f"test once at day {MAX_DAYS}:               "
      f"false positive rate {100*fixed_fpr:.2f}%")
print(f"test daily from day {FIRST_LOOK} and stop early: "
      f"false positive rate {100*peek_fpr:.2f}%")
print(f"inflation factor:                     {peek_fpr/fixed_fpr:.1f}x")

print("\nfalse positive rate by number of looks allowed:")
for n_looks in [1, 2, 5, 10, 26]:
    idx = np.linspace(FIRST_LOOK - 1, MAX_DAYS - 1, n_looks).astype(int)
    print(f"  {n_looks:>3} look(s): {100*sig[:, idx].any(axis=1).mean():.2f}%")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

dd = np.arange(3, 91)
nn = dd * per_arm_per_day
axes[0].plot(dd, [mde(v, sigma_plain) for v in nn],
             label="Standard", color="#c0392b", linewidth=1.8)
axes[0].plot(dd, [mde(v, sigma_cuped) for v in nn],
             label="With CUPED", color="#1e8449", linewidth=1.8)
axes[0].axhline(OBSERVED_EFFECT, color="grey", linestyle=":", linewidth=1.3)
axes[0].annotate(f"observed effect ({OBSERVED_EFFECT:.4f})",
                 xy=(60, OBSERVED_EFFECT), xytext=(0, 6),
                 textcoords="offset points", fontsize=8, color="grey")
axes[0].set_xlabel("experiment duration (days)")
axes[0].set_ylabel("minimum detectable effect")
axes[0].set_title("How long must the switchback run?", fontsize=10)
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.25)

looks = np.arange(1, 27)
fprs = [sig[:, np.linspace(FIRST_LOOK - 1, MAX_DAYS - 1, L).astype(int)]
        .any(axis=1).mean() * 100 for L in looks]
axes[1].plot(looks, fprs, marker="o", markersize=3.5,
             color="#c0392b", linewidth=1.6)
axes[1].axhline(5, color="black", linestyle="--", linewidth=1.2)
axes[1].annotate("nominal 5% rate", xy=(14, 5), xytext=(0, 5),
                 textcoords="offset points", fontsize=8)
axes[1].set_xlabel("number of times results are checked")
axes[1].set_ylabel("false positive rate (%)")
axes[1].set_title("Peeking inflates false positives (true effect = 0)",
                  fontsize=10)
axes[1].grid(alpha=0.25)

plt.tight_layout()
path = os.path.join(FIGDIR, "04_experiment_design.png")
plt.savefig(path, dpi=140)
print(f"\nchart written to {path}")

pd.DataFrame({
    "days": dd,
    "n_per_arm": nn,
    "mde_plain": [mde(v, sigma_plain) for v in nn],
    "mde_cuped": [mde(v, sigma_cuped) for v in nn],
}).to_csv("data/processed/power_curve.csv", index=False)
print("power curve written to data/processed/power_curve.csv")

print("\n\nDone. Send me this output and the chart.")
