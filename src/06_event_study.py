"""
Step 7: Event study.

Instead of collapsing everything into one before-versus-after number, estimate a
separate treated-minus-control gap for each week relative to the surcharge.
Week -1 is the reference, so every coefficient reads as "how different was the
gap that week compared to the week before treatment".

What to look for:
  - coefficients scattered around zero BEFORE week 0  -> pre-trends are fine
  - a visible step AT week 0 that persists            -> a real effect
  - a slope running through the pre-period            -> the design is confounded

This is the most honest way to present a design whose parallel trends test
failed: rather than asserting the assumption holds, show the reader the whole
path and let them judge.

NOTE ON THE PREVIOUS BUG: writing the interaction as
C(wk, Treatment(reference=-1)):treated let patsy choose its own column names,
which did not match what the lookup expected, so every coefficient came back
empty. Here the dummies are built explicitly, so the names are known.

Run:  python src/06_event_study.py
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

LO, HI = -16, 16
REF = -1

panel = pd.read_parquet(PANEL)
panel["trip_date"] = pd.to_datetime(panel["trip_date"])
treat_day = (pd.Timestamp(TREATMENT_DATE) - pd.Timestamp("2018-10-01")).days
panel["rel_week"] = np.floor((panel["day_index"] - treat_day) / 7).astype(int)


def wk_name(k):
    """Stable column name for a week offset."""
    return f"ev_m{abs(k)}" if k < 0 else f"ev_p{k}"


def build(df):
    """Add one treated-by-week indicator per week, omitting the reference."""
    d = df[(df.rel_week >= LO) & (df.rel_week <= HI)].copy()
    terms = []
    for k in range(LO, HI + 1):
        if k == REF:
            continue
        name = wk_name(k)
        d[name] = ((d["rel_week"] == k) & (d["treated"] == 1)).astype(float)
        terms.append(name)
    return d, terms


def run(df, col, tag):
    """Fit the event study and return a tidy frame of coefficients."""
    d, terms = build(df.dropna(subset=[col]))
    formula = f"{col} ~ " + " + ".join(terms) + " + C(zone_id) + C(trip_date)"
    fit = smf.ols(formula, data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["zone_id"]})

    rows = []
    for k in range(LO, HI + 1):
        if k == REF:
            rows.append(dict(week=k, coef=0.0, lo=0.0, hi=0.0))
            continue
        name = wk_name(k)
        lo, hi = fit.conf_int().loc[name]
        rows.append(dict(week=k, coef=fit.params[name], lo=lo, hi=hi))

    out = pd.DataFrame(rows).sort_values("week")
    out["outcome"], out["sample"] = col, tag
    return out


print("=" * 78)
print("EVENT STUDY")
print(f"weeks {LO} to {HI}, reference week {REF}, SEs clustered by zone")
print("=" * 78)

border = panel[panel.zone_id.isin(BORDER_TREATED + BORDER_CONTROL)]

fig, axes = plt.subplots(len(OUTCOMES), 1, figsize=(11.5, 3.8 * len(OUTCOMES)))
collected = []

for ax, (col, label) in zip(axes, OUTCOMES):
    full = run(panel, col, "full")
    ring = run(border, col, "border")
    collected += [full, ring]

    ax.axhspan(-1e9, 1e9, xmin=0, xmax=(REF + 0.5 - LO) / (HI - LO),
               color="#f2f2f2", zorder=0)

    for frame, name, colour, off in [(full, "Full sample", "#c0392b", -0.12),
                                     (ring, "Border ring", "#2874a6", 0.12)]:
        ax.errorbar(frame["week"] + off, frame["coef"],
                    yerr=[frame["coef"] - frame["lo"], frame["hi"] - frame["coef"]],
                    fmt="o", markersize=3.2, linewidth=0.9, capsize=1.8,
                    color=colour, alpha=0.85, label=name)

    ax.axvline(REF + 0.5, color="black", linestyle="--", linewidth=1.2)
    ax.axhline(0, color="grey", linewidth=0.9)
    ax.set_xlim(LO - 1, HI + 1)
    lim = max(abs(full["lo"].min()), abs(full["hi"].max())) * 1.15
    ax.set_ylim(-lim, lim)
    ax.set_title(f"Event study: {label}", fontsize=10)
    ax.set_xlabel("weeks relative to surcharge (2 Feb 2019)")
    ax.set_ylabel("treated - control")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.22)

    pre = full[(full.week < REF)]["coef"]
    post = full[(full.week >= 0)]["coef"]
    pre_b = ring[(ring.week < REF)]["coef"]
    post_b = ring[(ring.week >= 0)]["coef"]

    print(f"\n{label}")
    print(f"  full sample   pre {pre.mean():+.5f}   post {post.mean():+.5f}"
          f"   break {post.mean()-pre.mean():+.5f}")
    print(f"  border ring   pre {pre_b.mean():+.5f}   post {post_b.mean():+.5f}"
          f"   break {post_b.mean()-pre_b.mean():+.5f}")
    print(f"  pre-period spread (full, sd of coefs): {pre.std():.5f}")

plt.tight_layout()
path = os.path.join(FIGDIR, "03_event_study.png")
plt.savefig(path, dpi=140)
print(f"\nchart written to {path}")

pd.concat(collected).to_csv("data/processed/event_study.csv", index=False)
print("coefficients written to data/processed/event_study.csv")

print("\n\nDone.")
