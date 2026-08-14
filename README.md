# Did NYC's congestion surcharge change taxi behaviour?

**Short answer: I could not establish that it did, and the way that conclusion emerged is the point of this project.**

In February 2019 New York City added a flat $2.50 surcharge to metered taxi trips touching Manhattan below 96th Street. Trips entirely outside that zone were unaffected. That looks like a clean natural experiment, so I set out to measure the effect on rider behaviour using 57 million yellow taxi trips.

Four candidate findings emerged. All four were discarded, each for a different reason. What survives is a complete causal pipeline that returns an honest null, plus an experiment-design analysis of what it would have taken to answer the question properly.

---

## What I found, and why each finding died

| Outcome | Initial estimate | Killed by |
|---|---|---|
| Tip rate | +3.7 pp | Measurement artifact. The denominator moved, not behaviour |
| Trip volume | −5.6% | Pre-existing trend larger than the estimate itself |
| Mean distance | +0.11 miles | Collapsed to zero once group trends were included |
| Short trips (<1 mi) | −1.2 pp | Sliding-window placebo. February was not a special date |

### 1. The tipping result was arithmetic, not generosity

Measuring `tip / fare_amount` showed tipping jumping 3.7 percentage points in charged zones. That is backwards: riders charged more do not become more generous.

Taxi payment terminals suggest tips as a percentage of the **total** charge. The surcharge raised that total by $2.50, so a rider pressing the identical button tips more in dollars. Dividing by `fare_amount`, which excludes the surcharge, inflates the ratio mechanically.

Recomputing against `total_amount − tip_amount`, the base the terminal actually applies its percentage to, flips the sign to −0.6 pp. The apparent effect was entirely a denominator problem.

### 2. Volume was a pre-existing trend

The naive estimate was −5.6%. But treated zones were already losing ridership relative to uptown at 0.63% per week **before** the surcharge existed (p < 0.0001). Over the 17-week pre-period that is a 10.8% divergence, nearly double the measured effect. Adding group-specific time trends moves the estimate to +1.4% and insignificant.

This was yellow taxis declining faster in Midtown than in Harlem, which was true with or without the fee.

### 3. Distance did not survive specification

Mean distance passed the formal parallel trends test at p = 0.18, so I initially kept it. That was a mistake worth recording: **passing a pre-trend test does not mean there is no trend, it means you lacked power to detect one.** Its pre-trend of +0.0046 per week accumulated to +0.078 over the pre-period, nearly the whole +0.105 estimate. Group trends reduced it to −0.005.

### 4. Short trips: the one that took four tests to kill

This was the last one standing, and it looked strong:

- Stable across three specifications: −0.0127, −0.0123, −0.0195
- Significant in all three
- Larger in the tighter border design, not smaller
- Pre-trend ran *opposite* to the effect, making the estimate conservative
- Randomisation inference over zones: **0 of 1,000** random assignments came close

Then the placebo in time failed. A fake treatment date of 15 November produced −0.0131, larger than the real estimate.

The sliding-window test settled it. Estimating the same specification at every feasible date with identical 8-week windows, February 2 gives −0.0087, comfortably inside the placebo distribution. Mid-February gives −0.0175. Mid-January gives **+0.0185**, an equally large effect with the opposite sign.

The series oscillates continuously with no break at the treatment date.

---

## Why randomisation inference passed while the finding was wrong

This is the most useful thing in the project.

Randomisation inference shuffles **zone assignment** while holding the date fixed. It answers: *does the grouping matter?* Answer: emphatically yes, 0 of 1,000.

But it cannot answer: *does the timing matter?* The two questions come apart precisely when a group difference is persistent rather than event-driven, which is what is happening here. Below-96th and above-96th Manhattan differ in short-trip composition throughout the sample. They did not start differing in February.

The same applies to specification robustness. The estimate barely moved across three specifications, and I read that as strength. But specification stability tests whether the **model** is fragile, not whether the **identification** is. Three specifications resting on the same flawed assumption will agree with each other perfectly.

**Robustness across specifications is not robustness of identification.**

---

## What would be needed to answer this properly

The design fails because treated and control zones follow different trajectories for reasons unrelated to the surcharge. Options:

1. **Synthetic control.** Construct a weighted combination of untreated zones that reproduces the treated group's pre-period path, rather than assuming any natural comparison group works.
2. **Higher-frequency identification.** The surcharge began at 12:01 a.m. on a Saturday. An hourly regression discontinuity around that instant would use hours rather than months, leaving far less room for trends to accumulate.
3. **A real experiment.** Which is what the second half of this project designs.

---

## Experiment design: what a proper test would have required

The city could not randomise. A marketplace operator can. This half stands independent of the causal result above.

**Randomisation unit: the zone-day (a switchback design).** Randomising individual riders would violate SUTVA, since changing one rider's price changes driver availability for everyone nearby. Geographic-time cells contain the interference. This is what marketplace companies actually use for pricing tests.

### Power

With 56 zones split into two arms, detecting an effect of 0.0127 at 80% power and alpha 0.05 requires **19 days**.

### CUPED

Using each zone's prior behaviour as a covariate:

| | Value |
|---|---|
| Correlation, pre vs post | 0.918 |
| Theta | 1.021 |
| Variance reduction | 84.3% |
| Days required | 19 → 3 |

**An honest caveat.** Theta came out at 1.021, essentially exactly 1. With `Y − 1.0 × (X − X̄)`, the adjustment is arithmetically just subtracting each zone's own mean. CUPED here **is** zone demeaning, and recovers what zone fixed effects would give anyway. Its marginal value over a correctly specified regression is real but far below 84%.

A second caveat: "3 days" is not a shippable recommendation. Taxi demand has a hard weekly cycle, so any switchback shorter than a full week systematically over-samples some days of the week. The practical floor is one to two complete weeks regardless of what the power math says.

### Peeking

10,000 simulated experiments with a true effect of exactly zero, so every significant result is a false positive by construction:

| Looks allowed | False positive rate |
|---|---|
| 1 | 5.01% |
| 2 | 9.25% |
| 5 | 15.40% |
| 10 | 19.54% |
| 26 (daily) | 23.68% |

Checking twice nearly doubles the error rate. This is the entire argument for experimentation platforms over dashboard-watching.

---

## Method notes

### Deriving the treatment zone from data, not a map

Rather than eyeballing which taxi zones sit below 96th Street, I grouped March 2019 trips by pickup zone and measured what share incurred a surcharge. Three tiers emerged:

| Tier | Charge rate | Examples |
|---|---|---|
| Inside | 0.93 – 0.99 | Midtown, SoHo, Upper East Side South |
| Straddling 96th | 0.45 – 0.90 | Manhattan Valley, Bloomingdale, East Harlem South |
| Outside | 0.12 – 0.28 | Central Harlem, Hamilton Heights, Inwood |

The middle tier consists precisely of the zones that physically straddle 96th Street. Nothing geographic was supplied to the script; it recovered the street from charge records alone. Straddling zones were dropped as a buffer.

### Data and cleaning

Source: [NYC TLC trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), yellow taxi, October 2018 to May 2019.

**Effective date is 2 February 2019, not 1 January.** The law took effect 1 January but a court injunction delayed collection. Coding the earlier date contaminates the pre-period.

Filters, with rows lost:

| Filter | Reason |
|---|---|
| Pickup within nominal month | Raw files contain dates from 2002 to 2041 |
| `RatecodeID = 1` | Flat airport fares do not respond to a surcharge normally |
| Not JFK, LGA, EWR | Separate fare logic |
| Distance 0 to 100 miles | Data errors |
| Fare > 0 | Refunds and voided trips |
| Surcharge ≥ 0 | 7,124 March trips carry a negative surcharge |
| Duration 1 min to 4 hr | Meter left running, or never started |

57.0 million trips survive, collapsed to a balanced 13,608 zone-day panel.

**Cash tips are never recorded in this dataset.** Any tip rate computed over all trips measures payment method, not tipping. All tip metrics are card-only.

### Inference

Standard errors clustered by zone throughout. With only 8 control zones this is unreliable, since cluster-robust inference generally wants 30 or more clusters and is anticonservative below that. Randomisation inference is reported alongside for this reason.

---

## Repository

```
src/
  00_inspect.py            schema and data quality inspection
  01_derive_zone.py        derive treatment zone from surcharge records
  zones.py                 zone definitions with observed charge rates
  02_ingest.py             download, clean, log every filter
  03_build_panel.py        zone-day panel construction
  04_pretrends.py          parallel trends: visual and formal
  05_estimate.py           three difference-in-differences specifications
  06_event_study.py        week-by-week coefficients
  07_experiment_design.py  power, CUPED, peeking simulation
  08_placebo.py            placebo in time, randomisation inference
  09_date_placebo.py       sliding-window date placebo
figures/                   six charts
```

Aggregations in SQL via DuckDB. Estimation in statsmodels. Two-way fixed effects absorbed by demeaning where speed mattered, verified against statsmodels to 4e-07 before use.

**Reproduce:**

```bash
pip install -r requirements.txt
python src/02_ingest.py        # downloads ~800 MB
python src/03_build_panel.py
python src/05_estimate.py
python src/09_date_placebo.py
```

---

## What I would do differently

- **Run the placebo first, not last.** It was the only test that caught the problem, and it was cheap. I ran it after four analyses had already built confidence in a finding.
- **Treat specification stability with more suspicion.** Agreement across specifications sharing an assumption is not evidence about that assumption.
- **Attach uncertainty to placebo estimates.** My first placebo produced point estimates with no intervals, which made a noisy result look like a contradiction and delayed the right test.
