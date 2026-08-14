"""
Step 4 (revised): Collapse trips into a zone-day panel.

One row per neighbourhood per day. Every downstream step runs on this table.

TIP MEASUREMENT NOTE
--------------------
The first version measured tip_amount / fare_amount. That is misleading here.
Taxi payment terminals suggest tips as a percentage of the TOTAL charge, and
the congestion surcharge raised that total by $2.50. So a rider pressing the
exact same button tips more in dollars, and tip/fare rises mechanically with
no change in behaviour at all.

We therefore compute two versions:

  tip_rate_fare  tip / fare_amount        inflates mechanically, kept only to
                                          demonstrate the artifact
  tip_rate_base  tip / (total - tip)      the base the terminal actually applies
                                          its percentage to, so a flat reading
                                          means unchanged behaviour

Both exclude cash. Cash tips are never recorded in this dataset, so a tip rate
computed over all trips measures payment method, not tipping.

Run:  python src/03_build_panel.py
"""

import os
import sys
import duckdb

sys.path.insert(0, os.path.dirname(__file__))
from zones import CONGESTION_ZONE, UPTOWN_CONTROL, TREATMENT_DATE, sql_list

TRIPS = "data/processed/trips_clean.parquet"
LOOKUP = "data/raw/taxi_zone_lookup.csv"
OUT = "data/processed/panel.parquet"

con = duckdb.connect()

treated_ids = sql_list(CONGESTION_ZONE)
control_ids = sql_list(UPTOWN_CONTROL)

con.execute(f"""
    CREATE TABLE panel AS
    SELECT
        t.trip_date,
        t.pu_zone                                    AS zone_id,
        z.Zone                                       AS zone_name,
        CASE WHEN t.pu_zone IN ({treated_ids}) THEN 1 ELSE 0 END  AS treated,
        CASE WHEN t.trip_date >= DATE '{TREATMENT_DATE}'
             THEN 1 ELSE 0 END                       AS post,
        dayofweek(t.trip_date)                       AS dow,
        date_diff('day', DATE '2018-10-01', t.trip_date) AS day_index,

        count(*)                                     AS n_trips,
        ln(count(*))                                 AS log_trips,
        round(avg(t.trip_distance), 4)               AS mean_distance,
        round(avg(t.fare_amount), 4)                 AS mean_fare,
        round(avg(t.duration_min), 4)                AS mean_duration,

        -- short hops are where a flat $2.50 bites hardest
        round(avg(CASE WHEN t.trip_distance <= 1.0
                       THEN 1.0 ELSE 0.0 END), 5)    AS share_short_trips,

        -- payment mix, in case riders shifted toward cash
        count(*) FILTER (WHERE t.payment_type = 1)   AS n_card_trips,
        round(avg(CASE WHEN t.payment_type = 1
                       THEN 1.0 ELSE 0.0 END), 5)    AS share_card,

        -- naive tip rate: inflates mechanically once the surcharge exists
        round(avg(CASE WHEN t.payment_type = 1 AND t.fare_amount > 0
                       THEN t.tip_amount / t.fare_amount END), 5)
                                                     AS tip_rate_fare,

        -- corrected: divide by the base the payment terminal actually uses
        round(avg(CASE WHEN t.payment_type = 1
                        AND (t.total_amount - t.tip_amount) > 0
                       THEN t.tip_amount / (t.total_amount - t.tip_amount) END), 5)
                                                     AS tip_rate_base,

        round(avg(CASE WHEN t.congestion_surcharge > 0
                       THEN 1.0 ELSE 0.0 END), 4)    AS share_charged

    FROM '{TRIPS}' t
    LEFT JOIN '{LOOKUP}' z ON t.pu_zone = z.LocationID
    WHERE t.pu_zone IN ({treated_ids}) OR t.pu_zone IN ({control_ids})
    GROUP BY 1, 2, 3, 4, 5, 6, 7
""")

rows, zones, d0, d1 = con.execute("""
    SELECT count(*), count(DISTINCT zone_id), min(trip_date), max(trip_date)
    FROM panel
""").fetchone()

print("=" * 72)
print(f"PANEL: {rows:,} zone-days, {zones} zones, {d0} to {d1}")
print("=" * 72)

print("\nRAW FOUR-BUCKET MEANS")
print("=" * 72)
for metric in ["log_trips", "mean_distance", "share_short_trips",
               "share_card", "tip_rate_fare", "tip_rate_base"]:
    df = con.execute(f"""
        SELECT CASE WHEN treated = 1 THEN 'treated' ELSE 'control' END AS grp,
               round(avg(CASE WHEN post = 0 THEN {metric} END), 5) AS before,
               round(avg(CASE WHEN post = 1 THEN {metric} END), 5) AS after
        FROM panel GROUP BY 1 ORDER BY 1 DESC
    """).df()
    df["change"] = (df["after"] - df["before"]).round(5)
    t = df[df.grp == "treated"]["change"].values[0]
    c = df[df.grp == "control"]["change"].values[0]
    print(f"\n{metric}")
    print(df.to_string(index=False))
    print(f"  DiD = {t:.5f} - ({c:.5f}) = {t - c:+.5f}")

print("\n" + "=" * 72)
print("TIP ARTIFACT CHECK")
print("=" * 72)
print("If tip_rate_fare jumps but tip_rate_base stays flat, the apparent rise")
print("in tipping is arithmetic rather than behaviour.")

con.execute(f"COPY panel TO '{OUT}' (FORMAT PARQUET)")
print(f"\nwritten to {OUT}")
con.close()
print("\n\nDone.")
