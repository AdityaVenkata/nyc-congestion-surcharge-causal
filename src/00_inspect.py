import os
import requests
import duckdb

BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
RAW = "data/raw"
FILES = ["yellow_tripdata_2018-11.parquet", "yellow_tripdata_2019-03.parquet"]
LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"


def download(filename):
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, filename)
    if os.path.exists(path):
        print(f"already have {filename}")
        return path
    print(f"downloading {filename} ...")
    with requests.get(BASE + filename, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"  saved ({os.path.getsize(path)/1e6:.1f} MB)")
    return path


def inspect(path):
    con = duckdb.connect()
    print("\n" + "=" * 70)
    print(path)
    print("=" * 70)
    schema = con.execute(f"DESCRIBE SELECT * FROM '{path}'").fetchall()
    print(f"\n{len(schema)} columns:\n")
    for row in schema:
        print(f"  {row[0]:<28} {row[1]}")
    n = con.execute(f"SELECT count(*) FROM '{path}'").fetchone()[0]
    print(f"\nrow count: {n:,}")
    rng = con.execute(
        f"SELECT min(tpep_pickup_datetime), max(tpep_pickup_datetime) FROM '{path}'"
    ).fetchone()
    print(f"pickup range: {rng[0]}  to  {rng[1]}")
    cols = [c[0] for c in schema]
    if "congestion_surcharge" in cols:
        s = con.execute(f"""
            SELECT count(*) FILTER (WHERE congestion_surcharge IS NOT NULL),
                   count(*) FILTER (WHERE congestion_surcharge > 0),
                   round(avg(congestion_surcharge), 4)
            FROM '{path}'
        """).fetchone()
        print("\ncongestion_surcharge PRESENT")
        print(f"  non-null: {s[0]:,}")
        print(f"  charged (>0): {s[1]:,}")
        print(f"  mean: {s[2]}")
    else:
        print("\ncongestion_surcharge ABSENT (expected pre-2019)")
    print("\nfirst 2 rows:")
    print(con.execute(f"SELECT * FROM '{path}' LIMIT 2").df().to_string())
    con.close()


def inspect_lookup():
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, "taxi_zone_lookup.csv")
    if not os.path.exists(path):
        print("\ndownloading taxi_zone_lookup.csv ...")
        r = requests.get(LOOKUP_URL, timeout=60)
        r.raise_for_status()
        open(path, "wb").write(r.content)
    con = duckdb.connect()
    print("\n" + "=" * 70)
    print("taxi_zone_lookup.csv")
    print("=" * 70)
    n = con.execute(f"SELECT count(*) FROM '{path}'").fetchone()[0]
    print(f"\n{n} zones total\n")
    print(con.execute(
        f"SELECT Borough, count(*) AS n FROM '{path}' GROUP BY Borough ORDER BY n DESC"
    ).df().to_string(index=False))
    print("\nManhattan zones:")
    print(con.execute(
        f"SELECT LocationID, Zone, service_zone FROM '{path}' "
        f"WHERE Borough = 'Manhattan' ORDER BY Zone"
    ).df().to_string(index=False))
    con.close()


if __name__ == "__main__":
    for fname in FILES:
        inspect(download(fname))
    inspect_lookup()
    print("\n\nDone. Send me this output.")
