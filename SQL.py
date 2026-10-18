import os
import pandas as pd
import sqlite3

# =============================================================
# 1. LOAD CSV AND CONVERT TO IN-MEMORY SQL DATABASE
# =============================================================
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
csv_path = os.path.join(script_dir, "ev_cars_subset.csv")

print("Loading CSV and initializing SQLite database...")
df_ev = pd.read_csv(csv_path, low_memory=False)

# Create in-memory SQLite connection
conn = sqlite3.connect(":memory:")

# Load Pandas DataFrame into a SQL table named 'ev_cars'
df_ev.to_sql("ev_cars", conn, index=False, if_exists="replace")
print("Data successfully loaded into SQL table 'ev_cars'!\n")

# =============================================================
# 2. DEFINE SQL QUERIES
# =============================================================
queries = {
    "1. Fleet Aging vs. Battery Health & Mileage": """
        SELECT 
            CAST(vehicle_age AS INT) AS age_years,
            COUNT(*) AS total_vehicles,
            ROUND(AVG(battery_health_percent), 2) AS avg_battery_health_pct,
            ROUND(AVG(electric_range_km), 2) AS avg_range_km,
            ROUND(AVG(odometer_km), 0) AS avg_odometer_km
        FROM ev_cars
        WHERE vehicle_age IS NOT NULL
        GROUP BY CAST(vehicle_age AS INT)
        ORDER BY age_years ASC;
    """,

    "2. Fast-Charging Speed & Battery Health Impact": """
        SELECT 
            CASE 
                WHEN dc_fast_charge_time_minutes < 30 THEN 'Fast Charger (<30m)'
                WHEN dc_fast_charge_time_minutes BETWEEN 30 AND 45 THEN 'Standard DC (30-45m)'
                ELSE 'Slow DC (>45m)'
            END AS charging_tier,
            COUNT(*) AS vehicle_count,
            ROUND(AVG(battery_health_percent), 2) AS avg_battery_health_pct,
            ROUND(AVG(odometer_km), 0) AS avg_odometer_km
        FROM ev_cars
        WHERE battery_health_percent IS NOT NULL
        GROUP BY charging_tier
        ORDER BY avg_battery_health_pct DESC;
    """,

    "3. Brand Depreciation & Resale Value": """
        SELECT 
            make,
            COUNT(*) AS total_listings,
            ROUND(AVG(listing_price), 2) AS avg_listing_price,
            ROUND(AVG(depreciation_percent), 2) AS avg_depreciation_pct,
            ROUND(AVG(battery_health_percent), 2) AS avg_battery_health_pct
        FROM ev_cars
        WHERE make IS NOT NULL
        GROUP BY make
        HAVING COUNT(*) >= 50
        ORDER BY avg_listing_price DESC;
    """,

    "4. Top Degraded Vehicles Per Brand (CTE & Window Functions)": """
        WITH BrandDegradation AS (
            SELECT 
                make,
                model,
                vehicle_year,
                battery_health_percent,
                odometer_km,
                DENSE_RANK() OVER (
                    PARTITION BY make 
                    ORDER BY battery_health_percent ASC, odometer_km DESC
                ) AS degradation_rank
            FROM ev_cars
            WHERE battery_health_percent IS NOT NULL
        )
        SELECT 
            make,
            model,
            vehicle_year,
            battery_health_percent,
            odometer_km,
            degradation_rank
        FROM BrandDegradation
        WHERE degradation_rank <= 2
        ORDER BY make, degradation_rank;
    """,

    "5. Second-Life Battery Candidates (<80% Health)": """
        SELECT 
            vehicle_id,
            make,
            model,
            vehicle_year,
            odometer_km,
            battery_health_percent,
            listing_price
        FROM ev_cars
        WHERE battery_health_percent < 80.0
          AND odometer_km > 100000
        ORDER BY battery_health_percent ASC, odometer_km DESC
        LIMIT 10;
    """
}

# =============================================================
# 3. EXECUTE QUERIES AND DISPLAY RESULTS
# =============================================================
for title, sql in queries.items():
    print("=" * 70)
    print(f"QUERY: {title}")
    print("=" * 70)

    # Run SQL query using pandas
    result_df = pd.read_sql_query(sql, conn)

    # Display results cleanly
    print(result_df.to_string(index=False))
    print("\n\n")

# Close connection
conn.close()