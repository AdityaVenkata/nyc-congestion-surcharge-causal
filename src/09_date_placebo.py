"""
Step 10: Is 2 February actually special?

The placebo-in-time test produced estimates at fake autumn dates that rivalled
the real one, while randomisation inference over zones passed easily. Those two
results together suggest a persistent difference between the zone groups that
is not switched on by the surcharge. But the placebo dates were estimated on
short, unequal windows with no uncertainty attached, so they may simply be
noisy.

This script settles it properly. For EVERY feasible date in the sample, take a
fixed window of WINDOW_WEEKS before and after, estimate the same specification,
and record the result. That produces a distribution of estimates under "some
arbitrary date", against which the real date can be ranked.

Every candidate uses an identical window length and an identical number of
observations, so the estimates are directly comparable. That was the flaw in
the earlier placebo.

HOW TO READ IT:

  If 2 February is a clear minimum, or sits in the extreme tail, the surcharge
  is doing the work and the finding stands.

  If 2 February is unremarkable among dozens of other dates, the effect is a
  persistent group difference and this design cannot attribute it to the
  surcharge. That is a null result, and it is the correct answer if that is
  what the data says.

Run:  python src/09_date_placebo.py
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

METRICS = [
    ("share_short_trips", "Share of trips under 1 mile"),
    ("mean_distance",     "Mean trip distance (miles)"),
]
WINDOW_WEEKS = 8
RNG = np.random.default_rng(11)

panel = pd.read_parquet(PANEL)
panel["trip_date"] = pd.to_datetime(panel["trip_date"])
treat_date = pd.Timestamp(TREATMENT_DATE)
window = pd.Timedelta(days=WINDOW_WEEKS * 7)

start, end = panel.trip_date.min(), panel.trip_date.max()
candidates = pd.date_range(start + window, end - window, freq="D")

print("=" * 74)
print("SLIDING WINDOW DATE PLACEBO")
print("=" * 74)
print(f"sample:            {start.date()} to {end.date()}")
print(f"window:            {WINDOW_WEEKS} weeks each side")
print(f"candidate dates:   {len(candidates)}")
print(f"real date:         {treat_date.date()}")


def demean(df, values):
    s = pd.Series(values, index=df.index)
    return (s
            - s.groupby(df["zone_id"]).transform("mean")
            - s.groupby(df["trip_date"]).transform("mean")
            + s.mean()).values


def estimate(df, metric, cutoff):
    """Two-way FE estimate with a group-specific linear trend."""
    d = df.dropna(subset=[metric])
    treated = d["treated"].values.astype(float)
    post = (d["trip_date"] >= cutoff).values.astype(float)
    if post.sum() < 100 or (1 - post).sum() < 100:
        return np.nan
    y = demean(d, d[metric].values)
    X = np.column_stack([
        demean(d, treated * post),
        demean(d, treated * d["day_index"].values),
    ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta[0]


results = {}
for metric, label in METRICS:
    print(f"\n{'-' * 74}\n{label}\n{'-' * 74}")

    ests = []
    for c in candidates:
        sub = panel[(panel.trip_date >= c - window) &
                    (panel.trip_date < c + window)]
        ests.append(estimate(sub, metric, c))
    ests = np.array(ests)
    results[metric] = ests

    sub_real = panel[(panel.trip_date >= treat_date - window) &
                     (panel.trip_date < treat_date + window)]
    real = estimate(sub_real, metric, treat_date)

    valid = ests[~np.isnan(ests)]
    rank = (np.abs(valid) >= abs(real)).mean()
    more_neg = (valid <= real).mean()

    print(f"real estimate at {treat_date.date()}:  {real:+.5f}")
    print(f"placebo distribution across {len(valid)} dates:")
    print(f"  mean                {valid.mean():+.5f}")
    print(f"  std deviation       {valid.std(ddof=1):.5f}")
    print(f"  min                 {valid.min():+.5f}")
    print(f"  max                 {valid.max():+.5f}")
    print(f"  5th percentile      {np.percentile(valid, 5):+.5f}")
    print(f"\n  share of dates at least as extreme (abs):  {rank:.3f}")
    print(f"  share of dates at least as negative:      {more_neg:.3f}")

    if more_neg < 0.05:
        print("\n  VERDICT: 2 February is in the extreme tail. The date matters.")
    elif more_neg < 0.20:
        print("\n  VERDICT: suggestive but not decisive. Report with caution.")
    else:
        print("\n  VERDICT: 2 February is unremarkable. This design cannot")
        print("           attribute the difference to the surcharge.")

    results[f"{metric}_real"] = real

# ---------------------------------------------------------------------------
# Chart: estimate as a function of assumed treatment date.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(METRICS), 1, figsize=(11.5, 4.2 * len(METRICS)))
if len(METRICS) == 1:
    axes = [axes]

for ax, (metric, label) in zip(axes, METRICS):
    ests = results[metric]
    real = results[f"{metric}_real"]

    ax.plot(candidates, ests, color="#7f8c8d", linewidth=1.4,
            label="estimate at each assumed date")
    ax.axvline(treat_date, color="#c0392b", linewidth=1.8, linestyle="--")
    ax.scatter([treat_date], [real], color="#c0392b", zorder=5, s=45,
               label=f"actual date ({real:+.5f})")
    ax.axhline(0, color="black", linewidth=0.8)

    valid = ests[~np.isnan(ests)]
    ax.axhspan(np.percentile(valid, 5), np.percentile(valid, 95),
               color="#d6eaf8", alpha=0.55, zorder=0,
               label="5th to 95th percentile of placebo dates")

    ax.set_title(f"{label}: estimated effect if the surcharge had started on "
                 f"each date ({WINDOW_WEEKS}-week windows)", fontsize=10)
    ax.set_ylabel("estimated effect")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)

plt.tight_layout()
path = os.path.join(FIGDIR, "06_date_placebo.png")
plt.savefig(path, dpi=140)
print(f"\nchart written to {path}")

out = pd.DataFrame({"date": candidates})
for metric, _ in METRICS:
    out[metric] = results[metric]
out.to_csv("data/processed/date_placebo.csv", index=False)
print("estimates written to data/processed/date_placebo.csv")

print("\n\nDone. Send me this output and the chart.")
