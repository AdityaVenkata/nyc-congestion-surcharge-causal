"""
Step 2: Derive the congestion zone from the data.

Rather than guessing which taxi zones sit below 96th Street, we look at which
zones actually got charged. Every March 2019 trip records whether a congestion
surcharge was applied, so grouping by pickup zone and measuring the share charged
reveals the real boundary.

We also quantify how large a clean control group is available, which determines
whether a citywide comparison is viable or whether we need the border design.

Run:  python src/01_derive_zone.py
"""

import duckdb

MARCH = "data/raw/yellow_tripdata_2019-03.parquet"
LOOKUP = "data/raw/taxi_zone_lookup.csv"

con = duckdb.connect()

# ---------------------------------------------------------------------------
# Build a date-clean view. The raw file contains pickups from 2002 to 2041,
# so we keep only rows whose pickup actually falls in March 2019.
# ---------------------------------------------------------------------------
con.execute(f"""
    CREATE VIEW march AS
    SELECT *
    FROM '{MARCH}'
    WHERE tpep_pickup_datetime >= TIMESTAMP '2019-03-01'
      AND tpep_pickup_datetime <  TIMESTAMP '2019-04-01'
""")

raw_n = con.execute(f"SELECT count(*) FROM '{MARCH}'").fetchone()[0]
clean_n = con.execute("SELECT count(*) FROM march").fetchone()[0]

print("=" * 72)
print("DATE FILTER")
print("=" * 72)
print(f"raw rows:           {raw_n:,}")
print(f"in March 2019:      {clean_n:,}")
print(f"dropped:            {raw_n - clean_n:,}  ({100*(raw_n-clean_n)/raw_n:.4f}%)")

# ---------------------------------------------------------------------------
# What does the surcharge actually look like? If it is a flat 2.50 for taxis,
# the distinct values should be few.
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("SURCHARGE VALUES")
print("=" * 72)
print(con.execute("""
    SELECT congestion_surcharge AS value,
           count(*)             AS n_trips,
           round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct
    FROM march
    GROUP BY 1
    ORDER BY n_trips DESC
    LIMIT 10
""").df().to_string(index=False))

# ---------------------------------------------------------------------------
# Charge rate by pickup zone. Zones inside the congestion zone should sit near
# 100%. Zones outside should be much lower, driven only by trips whose
# destination or route entered the zone.
# ---------------------------------------------------------------------------
con.execute(f"""
    CREATE VIEW pu_rates AS
    SELECT
        m.PULocationID                                        AS location_id,
        z.Zone                                                AS zone_name,
        z.Borough                                             AS borough,
        z.service_zone                                        AS service_zone,
        count(*)                                              AS n_trips,
        round(avg(CASE WHEN m.congestion_surcharge > 0
                       THEN 1.0 ELSE 0.0 END), 4)             AS share_charged
    FROM march m
    LEFT JOIN '{LOOKUP}' z ON m.PULocationID = z.LocationID
    WHERE m.congestion_surcharge IS NOT NULL
    GROUP BY 1, 2, 3, 4
    HAVING count(*) >= 500          -- ignore zones with too little data to judge
""")

print("\n" + "=" * 72)
print("DISTRIBUTION OF CHARGE RATE ACROSS ZONES")
print("(looking for a clean gap that separates inside from outside)")
print("=" * 72)
print(con.execute("""
    SELECT
        CASE
            WHEN share_charged >= 0.95 THEN 'a. 0.95 - 1.00'
            WHEN share_charged >= 0.80 THEN 'b. 0.80 - 0.95'
            WHEN share_charged >= 0.50 THEN 'c. 0.50 - 0.80'
            WHEN share_charged >= 0.20 THEN 'd. 0.20 - 0.50'
            ELSE                            'e. 0.00 - 0.20'
        END          AS band,
        count(*)     AS n_zones,
        sum(n_trips) AS n_trips
    FROM pu_rates
    GROUP BY 1
    ORDER BY 1
""").df().to_string(index=False))

print("\n" + "=" * 72)
print("ALL MANHATTAN ZONES BY CHARGE RATE")
print("=" * 72)
print(con.execute("""
    SELECT location_id, zone_name, service_zone, n_trips, share_charged
    FROM pu_rates
    WHERE borough = 'Manhattan'
    ORDER BY share_charged DESC, n_trips DESC
""").df().to_string(index=False))

print("\n" + "=" * 72)
print("NON-MANHATTAN ZONES WITH CHARGE RATE ABOVE 20%")
print("(these are outside the zone but often travel into it)")
print("=" * 72)
print(con.execute("""
    SELECT location_id, zone_name, borough, n_trips, share_charged
    FROM pu_rates
    WHERE borough <> 'Manhattan' AND share_charged >= 0.20
    ORDER BY share_charged DESC
    LIMIT 25
""").df().to_string(index=False))

# ---------------------------------------------------------------------------
# Provisional zone definition: pickup zones charged at least 95% of the time.
# ---------------------------------------------------------------------------
zone_ids = [r[0] for r in con.execute("""
    SELECT location_id FROM pu_rates WHERE share_charged >= 0.95 ORDER BY location_id
""").fetchall()]

print("\n" + "=" * 72)
print("PROVISIONAL CONGESTION ZONE (share_charged >= 0.95)")
print("=" * 72)
print(f"\n{len(zone_ids)} zones:\n")
print(zone_ids)

# ---------------------------------------------------------------------------
# How much clean data does each group have? This decides the design.
# ---------------------------------------------------------------------------
if zone_ids:
    ids = ",".join(str(i) for i in zone_ids)
    print("\n" + "=" * 72)
    print("GROUP SIZES UNDER THE CITYWIDE DESIGN")
    print("=" * 72)
    print(con.execute(f"""
        SELECT
            CASE
                WHEN PULocationID IN ({ids}) AND DOLocationID IN ({ids})
                    THEN 'treated  (both ends inside)'
                WHEN PULocationID NOT IN ({ids}) AND DOLocationID NOT IN ({ids})
                    THEN 'control  (neither end inside)'
                ELSE 'mixed    (one end inside)'
            END          AS grp,
            count(*)     AS n_trips,
            round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct,
            round(avg(CASE WHEN congestion_surcharge > 0
                           THEN 1.0 ELSE 0.0 END), 4)          AS share_charged
        FROM march
        WHERE congestion_surcharge IS NOT NULL
        GROUP BY 1
        ORDER BY n_trips DESC
    """).df().to_string(index=False))

    print("\n" + "=" * 72)
    print("CONTROL GROUP COMPOSITION (where are the untreated trips?)")
    print("=" * 72)
    print(con.execute(f"""
        SELECT z.Borough AS borough,
               count(*)  AS n_trips,
               round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct
        FROM march m
        LEFT JOIN '{LOOKUP}' z ON m.PULocationID = z.LocationID
        WHERE m.PULocationID NOT IN ({ids})
          AND m.DOLocationID NOT IN ({ids})
        GROUP BY 1
        ORDER BY n_trips DESC
    """).df().to_string(index=False))

con.close()
print("\n\nDone. Send me this output.")
