"""
Zone definitions for the congestion surcharge analysis.

These are not read off a map. They were derived in src/01_derive_zone.py by
measuring what share of each zone's March 2019 trips actually incurred a
congestion surcharge, then cross-checked against the 96th Street boundary.

Three groups:

  CONGESTION_ZONE  charged on 90%+ of trips. Unambiguously below 96th Street.
  BUFFER           charged on 45% to 90% of trips. These zones physically
                   straddle 96th Street, so they are neither cleanly treated
                   nor cleanly untreated. Dropped from the analysis entirely.
  UPTOWN_CONTROL   Manhattan zones north of 96th Street. Charged on only
                   12% to 28% of trips, and only because residents ride
                   downtown.

Dropping the buffer is standard practice in border designs: units near the
cutoff are contaminated, so you exclude a band around it rather than forcing
each one to a side.
"""

# Charged on 90%+ of trips in March 2019. Below 96th Street.
CONGESTION_ZONE = [
    4, 12, 13, 43, 45, 48, 50, 68, 79, 87, 88, 90, 100, 107, 113, 114,
    125, 137, 140, 141, 142, 143, 144, 148, 158, 161, 162, 163, 164, 170,
    186, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 239, 246,
    249, 261, 262, 263,
]

# Straddle the 96th Street line. Excluded from both groups.
BUFFER = [
    24,    # Bloomingdale            0.538
    75,    # East Harlem South       0.509
    151,   # Manhattan Valley        0.603
    166,   # Morningside Heights     0.448
    238,   # Upper West Side North   0.895
]

# Manhattan, north of 96th Street. Our control group.
UPTOWN_CONTROL = [
    41,    # Central Harlem          0.275
    42,    # Central Harlem North    0.142
    74,    # East Harlem North       0.281
    116,   # Hamilton Heights        0.237
    127,   # Inwood                  0.119
    152,   # Manhattanville          0.249
    243,   # Washington Heights N    0.162
    244,   # Washington Heights S    0.246
]

# Airport zones. Flat-fare structures do not respond to a surcharge the way
# metered fares do, so these are excluded regardless of group.
AIRPORTS = [1, 132, 138]   # Newark, JFK, LaGuardia

# The eight months we analyse: four before the surcharge, four after.
MONTHS = [
    "2018-10", "2018-11", "2018-12", "2019-01",
    "2019-02", "2019-03", "2019-04", "2019-05",
]

# Collection began 12:01 a.m. Saturday 2 February 2019. Note this is NOT
# 1 January, when the law nominally took effect. A court injunction delayed
# actual collection by a month.
TREATMENT_DATE = "2019-02-02"


def sql_list(ids):
    """Render a Python list of ints as a SQL IN-clause body."""
    return ",".join(str(i) for i in ids)


# ---------------------------------------------------------------------------
# Narrow border ring, added after the parallel trends test.
#
# The full control group (8 zones stretching to Inwood) diverges from downtown
# in the pre-period. A tighter ring keeps only zones hugging 96th Street on
# each side, which should be far more comparable. East Harlem South (75) sits
# between them and stays in the buffer.
# ---------------------------------------------------------------------------

# Northern tier of the congestion zone: immediately south of 96th Street.
BORDER_TREATED = [
    236,   # Upper East Side North
    262,   # Yorkville East
    263,   # Yorkville West
    239,   # Upper West Side South
    143,   # Lincoln Square West
]

# Immediately north of 96th Street.
BORDER_CONTROL = [
    41,    # Central Harlem
    74,    # East Harlem North
    116,   # Hamilton Heights
    152,   # Manhattanville
]
