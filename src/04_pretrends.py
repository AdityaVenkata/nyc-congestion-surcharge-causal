"""
Step 5: Does the parallel trends assumption hold?

This is the decision point. Difference-in-differences assumes that, absent the
surcharge, treated and control zones would have moved together. We cannot prove
that. What we can do is check whether they moved together in the four months
BEFORE the surcharge existed. If they did, the assumption is credible. If they
were already diverging, the design is not valid and we say so.

Two checks:

  1. Visual. Plot both groups over the pre-period, indexed so the levels are
     comparable. Parallel lines are what we want.

  2. Formal. Using pre-period data only, regress the outcome on a treated-group
     indicator interacted with a linear time trend, with zone fixed effects and
     standard errors clustered by zone. If the interaction is significant, the
     groups were already diverging.

WHY CLUSTER: two trips in the same zone on the same day share weather, traffic
and local events, so observations are not independent. Ignoring that makes the
standard errors far too small and the conclusions overconfident.

Run:  python src/04_pretrends.py
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from zones import TREATMENT_DATE

PANEL = "data/processed/panel.parquet"
FIGDIR = "figures"

OUTCOMES = [
    ("log_trips",         "Log daily trips"),
    ("mean_distance",     "Mean trip distance (miles)"),
    ("share_short_trips", "Share of trips under 1 mile"),
    ("tip_rate_base",     "Tip rate (corrected base)"),
]

os.makedirs(FIGDIR, exist_ok=True)

panel = pd.read_parquet(PANEL)
panel["trip_date"] = pd.to_datetime(panel["trip_date"])
treat_date = pd.Timestamp(TREATMENT_DATE)

pre = panel[panel["trip_date"] < treat_date].copy()

print("=" * 72)
print("PRE-PERIOD")
print("=" * 72)
print(f"dates:      {pre.trip_date.min().date()} to {pre.trip_date.max().date()}")
print(f"zone-days:  {len(pre):,}")
print(f"zones:      {pre.zone_id.nunique()}  "
      f"({pre[pre.treated==1].zone_id.nunique()} treated, "
      f"{pre[pre.treated==0].zone_id.nunique()} control)")

# ---------------------------------------------------------------------------
# Formal test, one outcome at a time.
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("FORMAL PARALLEL TRENDS TEST (pre-period only)")
print("interaction term = difference in slope between the two groups")
print("=" * 72)

results = []
for col, label in OUTCOMES:
    d = pre.dropna(subset=[col]).copy()
    d["weeks"] = d["day_index"] / 7.0

    model = smf.ols(f"{col} ~ treated * weeks + C(zone_id) + C(dow)", data=d)
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": d["zone_id"]})

    coef = fit.params["treated:weeks"]
    se = fit.bse["treated:weeks"]
    p = fit.pvalues["treated:weeks"]
    lo, hi = fit.conf_int().loc["treated:weeks"]

    verdict = "FAILS - groups already diverging" if p < 0.05 else "holds"
    results.append((label, coef, se, p, verdict))

    print(f"\n{label}")
    print(f"  slope difference per week: {coef:+.6f}")
    print(f"  std error (clustered):     {se:.6f}")
    print(f"  95% CI:                    [{lo:+.6f}, {hi:+.6f}]")
    print(f"  p-value:                   {p:.4f}")
    print(f"  verdict:                   {verdict}")

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
summary = pd.DataFrame(results,
                       columns=["outcome", "slope_diff", "std_err", "p_value", "verdict"])
print(summary.to_string(index=False))

# ---------------------------------------------------------------------------
# Visual check. Levels differ hugely between groups (Midtown vs Inwood), so we
# index each group to its own pre-period mean and compare shapes, not levels.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(OUTCOMES), 1, figsize=(11, 4 * len(OUTCOMES)))

for ax, (col, label) in zip(axes, OUTCOMES):
    daily = (panel.dropna(subset=[col])
                  .groupby(["trip_date", "treated"])[col]
                  .mean().reset_index())

    for grp, name, colour in [(1, "Treated (below 96th)", "#c0392b"),
                              (0, "Control (above 96th)", "#2874a6")]:
        g = daily[daily.treated == grp].sort_values("trip_date")
        base = g[g.trip_date < treat_date][col].mean()
        series = (g[col] / base) * 100
        ax.plot(g["trip_date"], series.rolling(7, center=True).mean(),
                label=name, color=colour, linewidth=1.6)

    ax.axvline(treat_date, color="black", linestyle="--", linewidth=1.2)
    ax.annotate("surcharge begins\n2 Feb 2019",
                xy=(treat_date, ax.get_ylim()[1]),
                xytext=(6, -30), textcoords="offset points", fontsize=8)
    ax.set_title(f"{label}  (indexed to pre-period mean = 100, 7-day average)",
                 fontsize=10)
    ax.set_ylabel("index")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

plt.tight_layout()
path = os.path.join(FIGDIR, "01_pretrends.png")
plt.savefig(path, dpi=140)
print(f"\nchart written to {path}")

# ---------------------------------------------------------------------------
# The gap between groups over time. If parallel trends holds, this line should
# be flat before the cutoff and then step.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(OUTCOMES), 1, figsize=(11, 3.2 * len(OUTCOMES)))

for ax, (col, label) in zip(axes, OUTCOMES):
    daily = (panel.dropna(subset=[col])
                  .groupby(["trip_date", "treated"])[col]
                  .mean().unstack())
    base_t = daily.loc[daily.index < treat_date, 1].mean()
    base_c = daily.loc[daily.index < treat_date, 0].mean()
    gap = (daily[1] / base_t) - (daily[0] / base_c)

    ax.plot(gap.index, gap.rolling(7, center=True).mean(),
            color="#6c3483", linewidth=1.6)
    ax.axvline(treat_date, color="black", linestyle="--", linewidth=1.2)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_title(f"Treated minus control, {label}", fontsize=10)
    ax.grid(alpha=0.25)

plt.tight_layout()
path2 = os.path.join(FIGDIR, "02_gap.png")
plt.savefig(path2, dpi=140)
print(f"chart written to {path2}")

print("\n\nDone. Send me this output and both charts.")
