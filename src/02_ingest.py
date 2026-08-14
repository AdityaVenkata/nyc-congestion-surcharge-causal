"""
Step 3: Download eight months, clean them, write one tidy file.

Filters applied, and why each one:

  valid month        raw files contain pickups from 2002 to 2041
  RatecodeID = 1     standard metered fare only; flat airport fares and
                     negotiated rates do not respond to a surcharge normally
  not an airport     JFK, LaGuardia, Newark have their own fare logic
  distance 0 to 100  zero-distance and 500-mile taxi trips are errors
  fare > 0           refunds and voided trips
  surcharge >= 0     7,124 March trips carry a NEGATIVE surcharge (refunds)
  duration 1min-4hr  meter left running, or never started

Every filter is counted and reported, because the drop log goes in the README.

Run:  python src/02_ingest.py
"""

import os
import sys
import requests
import duckdb

sys.path.insert(0, os.path.dirname(__file__))
from zones import MONTHS, AIRPORTS, sql_list

BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
RAW = "data/raw"
PROCESSED = "data/processed"
OUT = os.path.join(PROCESSED, "trips_clean.parquet")


def download(month):
    """Fetch one month of yellow taxi data if not already on disk."""
    fname = f"yellow_tripdata_{month}.parquet"
    path = os.path.join(RAW, fname)
    if os.path.exists(path):
        return path
    os.makedirs(RAW, exist_ok=True)
    print(f"  downloading {fname} ...", flush=True)
    with requests.get(BASE + fname, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"    {os.path.getsize(path)/1e6:.0f} MB")
    return path


def month_bounds(month):
    """Return the first day of this month and of the next, as SQL literals."""
    year, mon = int(month[:4]), int(month[5:7])
    nxt_y, nxt_m = (year + 1, 1) if mon == 12 else (year, mon + 1)
    return f"{year}-{mon:02d}-01", f"{nxt_y}-{nxt_m:02d}-01"


def clean_one(con, month, path):
    """Apply filters to one month, log the drops, return a cleaned relation."""
    start, end = month_bounds(month)
    airports = sql_list(AIRPORTS)

    con.execute(f"CREATE OR REPLACE VIEW raw AS SELECT * FROM '{path}'")

    # Count survivors after each filter is added, cumulatively.
    steps = [
        ("raw rows", "TRUE"),
        ("in nominal month",
         f"tpep_pickup_datetime >= TIMESTAMP '{start}' "
         f"AND tpep_pickup_datetime < TIMESTAMP '{end}'"),
        ("standard rate", "RatecodeID = 1"),
        ("not airport",
         f"PULocationID NOT IN ({airports}) AND DOLocationID NOT IN ({airports})"),
        ("distance sane", "trip_distance > 0 AND trip_distance < 100"),
        ("fare positive", "fare_amount > 0"),
        ("surcharge not negative",
         "(congestion_surcharge IS NULL OR congestion_surcharge >= 0)"),
        ("duration sane",
         "date_diff('second', tpep_pickup_datetime, tpep_dropoff_datetime) "
         "BETWEEN 60 AND 14400"),
    ]

    conditions = []
    prev = None
    print(f"\n{month}")
    for label, cond in steps:
        conditions.append(cond)
        where = " AND ".join(conditions)
        n = con.execute(f"SELECT count(*) FROM raw WHERE {where}").fetchone()[0]
        if prev is None:
            print(f"  {label:<24} {n:>10,}")
        else:
            print(f"  {label:<24} {n:>10,}   (-{prev-n:,})")
        prev = n

    where = " AND ".join(conditions)
    con.execute(f"""
        CREATE OR REPLACE VIEW clean_month AS
        SELECT
            CAST(tpep_pickup_datetime AS DATE)          AS trip_date,
            tpep_pickup_datetime                        AS pickup_ts,
            PULocationID                                AS pu_zone,
            DOLocationID                                AS do_zone,
            trip_distance,
            fare_amount,
            tip_amount,
            total_amount,
            payment_type,
            congestion_surcharge,
            date_diff('second', tpep_pickup_datetime,
                      tpep_dropoff_datetime) / 60.0     AS duration_min
        FROM raw
        WHERE {where}
    """)
    return prev


def main():
    os.makedirs(PROCESSED, exist_ok=True)
    con = duckdb.connect()

    print("=" * 60)
    print("DOWNLOADING")
    print("=" * 60)
    paths = {m: download(m) for m in MONTHS}

    print("\n" + "=" * 60)
    print("CLEANING (cumulative survivors after each filter)")
    print("=" * 60)

    first = True
    total = 0
    for month in MONTHS:
        n = clean_one(con, month, paths[month])
        total += n
        mode = "CREATE OR REPLACE TABLE trips AS" if first else "INSERT INTO trips"
        con.execute(f"{mode} SELECT * FROM clean_month")
        first = False

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"clean rows across all months: {total:,}")

    con.execute(f"COPY trips TO '{OUT}' (FORMAT PARQUET)")
    print(f"written to {OUT}  ({os.path.getsize(OUT)/1e6:.0f} MB)")

    print("\nrows per month:")
    print(con.execute("""
        SELECT strftime(trip_date, '%Y-%m') AS month,
               count(*)                     AS n_trips,
               round(avg(trip_distance), 2) AS avg_distance,
               round(avg(fare_amount), 2)   AS avg_fare
        FROM trips GROUP BY 1 ORDER BY 1
    """).df().to_string(index=False))

    con.close()


if __name__ == "__main__":
    main()
