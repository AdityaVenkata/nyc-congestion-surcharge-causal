"""
Step 6: The main estimates.

The parallel trends test failed for three of four outcomes, so a single naive
difference-in-differences would not be credible on its own. We therefore report
three specifications side by side and let the reader see how each estimate moves.

  (A) Naive TWFE          outcome ~ treated:post + zone FE + date FE
                          What you get if you ignore the pre-trend. Reported
                          so the reader can see what we are correcting.

  (B) Group time trends   adds treated:day_index
                          Lets treated and control zones follow their own linear
                          trajectories, so the treatment coefficient picks up
                          only a DISCRETE BREAK at February, not accumulated
                          drift. This is our preferred specification.

  (C) Narrow border ring  spec (B) restricted to zones hugging 96th Street
                          Neighbouring zones share weather, transit and local
                          conditions, so comparability is far higher. Fewer
                          observations, wider intervals.

Plus an EVENT STUDY, which is the honest way to present a design with pre-trend
problems: instead of one number, plot the treated-control gap week by week. The
reader sees the pre-period wobble and can judge whether February is a real break.

A NOTE ON CLUSTERING: standard errors are clustered by zone throughout, because
trips in the same zone on the same day share weather and traffic. But we have
only 8 control zones, and cluster-robust inference is known to be unreliable
with few clusters. Treat the intervals as indicative rather than exact, and say
so in the write-up.

Run:  python src/05_estimate.py
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
from zones import TREATMENT_DATE, BORDER_TREATED, BORDER_CONTROL

PANEL = "data/processed/panel.parquet"
FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

OUTCOMES = [
    ("share_short_trips", "Share of trips under 1 mile"),
    ("mean_distance",     "Mean trip distance (miles)"),
    ("log_trips",         "Log daily trips"),
    ("tip_rate_base",     "Tip rate (corrected base)"),
]

panel = pd.read_parquet(PANEL)
panel["trip_date"] = pd.to_datetime(panel["trip_date"])
treat_date = pd.Timestamp(TREATMENT_DATE)
treat_day = (treat_date - pd.Timestamp("2018-10-01")).days

panel["rel_day"] = panel["day_index"] - treat_day
panel["rel_week"] = np.floor(panel["rel_day"] / 7).astype(int)


def fit(formula, data):
    """OLS with standard errors clustered by zone."""
    m = smf.ols(formula, data=data)
    return m.fit(cov_type="cluster", cov_kwds={"groups": data["zone_id"]})


def report(fit_result, term, label):
    """Pull one coefficient out with its interval."""
    coef = fit_result.params[term]
    se = fit_result.bse[term]
    p = fit_result.pvalues[term]
    lo, hi = fit_result.conf_int().loc[term]
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    print(f"  {label:<24} {coef:+.5f}  se {se:.5f}  "
          f"[{lo:+.5f}, {hi:+.5f}]  p={p:.4f} {stars}")
    return dict(spec=label, coef=coef, se=se, p=p, lo=lo, hi=hi)


# ---------------------------------------------------------------------------
# Three specifications per outcome.
# ---------------------------------------------------------------------------
print("=" * 78)
print("DIFFERENCE-IN-DIFFERENCES ESTIMATES")
print("standard errors clustered by zone")
print("=" * 78)

border = panel[panel.zone_id.isin(BORDER_TREATED + BORDER_CONTROL)].copy()
print(f"\nborder ring: {border.zone_id.nunique()} zones "
      f"({border[border.treated==1].zone_id.nunique()} treated, "
      f"{border[border.treated==0].zone_id.nunique()} control), "
      f"{len(border):,} zone-days")

all_rows = []
for col, label in OUTCOMES:
    print(f"\n{'-' * 78}\n{label}\n{'-' * 78}")

    d = panel.dropna(subset=[col]).copy()
    b = border.dropna(subset=[col]).copy()

    fa = fit(f"{col} ~ treated:post + C(zone_id) + C(trip_date)", d)
    ra = report(fa, "treated:post", "(A) naive TWFE")

    fb = fit(f"{col} ~ treated:post + treated:day_index "
             f"+ C(zone_id) + C(trip_date)", d)
    rb = report(fb, "treated:post", "(B) + group trends")

    fc = fit(f"{col} ~ treated:post + treated:day_index "
             f"+ C(zone_id) + C(trip_date)", b)
    rc = report(fc, "treated:post", "(C) border ring")

    for r in (ra, rb, rc):
        r["outcome"] = label
        all_rows.append(r)

    # For log outcomes the coefficient reads as an approximate percent change.
    if col == "log_trips":
        print(f"  preferred spec (B) as percent: "
              f"{100*(np.exp(rb['coef'])-1):+.2f}%")

summary = pd.DataFrame(all_rows)[["outcome", "spec", "coef", "se", "p", "lo", "hi"]]
print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(summary.round(5).to_string(index=False))
summary.to_csv("data/processed/estimates.csv", index=False)

# ---------------------------------------------------------------------------
# Event study. One coefficient per week relative to treatment, week -1 omitted
# as the reference. Weeks with too little data at the edges are trimmed.
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("EVENT STUDY")
print("=" * 78)

LO, HI = -16, 16
fig, axes = plt.subplots(len(OUTCOMES), 1, figsize=(11, 3.6 * len(OUTCOMES)))

for ax, (col, label) in zip(axes, OUTCOMES):
    d = panel.dropna(subset=[col]).copy()
    d = d[(d.rel_week >= LO) & (d.rel_week <= HI)].copy()
    d["wk"] = d["rel_week"].astype(int)

    f = fit(f"{col} ~ C(wk, Treatment(reference=-1)):treated "
            f"+ C(zone_id) + C(trip_date)", d)

    weeks, coefs, los, his = [], [], [], []
    for w in range(LO, HI + 1):
        term = f"C(wk, Treatment(reference=-1))[T.{w}]:treated"
        if term in f.params.index:
            lo, hi = f.conf_int().loc[term]
            weeks.append(w); coefs.append(f.params[term])
            los.append(lo); his.append(hi)
        elif w == -1:
            weeks.append(w); coefs.append(0.0); los.append(0.0); his.append(0.0)

    ax.errorbar(weeks, coefs, yerr=[np.array(coefs) - np.array(los),
                                    np.array(his) - np.array(coefs)],
                fmt="o", markersize=3.5, linewidth=1, capsize=2,
                color="#c0392b", ecolor="#e8a49c")
    ax.axvline(-0.5, color="black", linestyle="--", linewidth=1.2)
    ax.axhline(0, color="grey", linewidth=0.9)
    ax.set_title(f"Event study: {label}", fontsize=10)
    ax.set_xlabel("weeks relative to surcharge")
    ax.set_ylabel("treated - control")
    ax.grid(alpha=0.25)

    pre = [c for w, c in zip(weeks, coefs) if w < -1]
    post = [c for w, c in zip(weeks, coefs) if w >= 0]
    print(f"\n{label}")
    print(f"  mean pre-period coefficient:  {np.mean(pre):+.5f}")
    print(f"  mean post-period coefficient: {np.mean(post):+.5f}")
    print(f"  break at treatment:           {np.mean(post) - np.mean(pre):+.5f}")

plt.tight_layout()
path = os.path.join(FIGDIR, "03_event_study.png")
plt.savefig(path, dpi=140)
print(f"\nchart written to {path}")

print("\n\nDone. Send me this output and the chart.")
