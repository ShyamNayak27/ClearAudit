"""
Velocity Features — DuckDB-powered rolling window aggregations.

Computes per-sender transaction velocity metrics over 1h, 6h, 24h, and 7d windows
using DuckDB's high-performance SQL window functions.
"""
import duckdb
import pandas as pd


def compute_velocity_features(con: duckdb.DuckDBPyConnection, table_name: str = 'transactions') -> pd.DataFrame:
    """
    Compute rolling velocity features using DuckDB window functions.
    
    Args:
        con: Active DuckDB connection with the transactions table registered.
        table_name: Name of the registered table/view.
    
    Returns:
        DataFrame with transaction_id + all velocity feature columns.
    """
    print("  [Velocity] Computing rolling window metrics via DuckDB...")

    query = f"""
    SELECT
        transaction_id,

        -- ======== TRANSACTION COUNT (per sender, rolling windows) ========
        COUNT(*) OVER (
            PARTITION BY sender_cif
            ORDER BY ts
            RANGE BETWEEN INTERVAL '1 HOUR' PRECEDING AND CURRENT ROW
        ) AS tx_count_1h,

        COUNT(*) OVER (
            PARTITION BY sender_cif
            ORDER BY ts
            RANGE BETWEEN INTERVAL '6 HOURS' PRECEDING AND CURRENT ROW
        ) AS tx_count_6h,

        COUNT(*) OVER (
            PARTITION BY sender_cif
            ORDER BY ts
            RANGE BETWEEN INTERVAL '24 HOURS' PRECEDING AND CURRENT ROW
        ) AS tx_count_24h,

        COUNT(*) OVER (
            PARTITION BY sender_cif
            ORDER BY ts
            RANGE BETWEEN INTERVAL '7 DAYS' PRECEDING AND CURRENT ROW
        ) AS tx_count_7d,

        -- ======== TRANSACTION SUM (per sender, rolling windows) ========
        SUM(amount_vnd) OVER (
            PARTITION BY sender_cif
            ORDER BY ts
            RANGE BETWEEN INTERVAL '1 HOUR' PRECEDING AND CURRENT ROW
        ) AS tx_sum_1h,

        SUM(amount_vnd) OVER (
            PARTITION BY sender_cif
            ORDER BY ts
            RANGE BETWEEN INTERVAL '6 HOURS' PRECEDING AND CURRENT ROW
        ) AS tx_sum_6h,

        SUM(amount_vnd) OVER (
            PARTITION BY sender_cif
            ORDER BY ts
            RANGE BETWEEN INTERVAL '24 HOURS' PRECEDING AND CURRENT ROW
        ) AS tx_sum_24h,

        SUM(amount_vnd) OVER (
            PARTITION BY sender_cif
            ORDER BY ts
            RANGE BETWEEN INTERVAL '7 DAYS' PRECEDING AND CURRENT ROW
        ) AS tx_sum_7d,

        -- ======== TRANSACTION AVG (per sender) ========
        AVG(amount_vnd) OVER (
            PARTITION BY sender_cif
            ORDER BY ts
            RANGE BETWEEN INTERVAL '1 HOUR' PRECEDING AND CURRENT ROW
        ) AS tx_avg_1h,

        AVG(amount_vnd) OVER (
            PARTITION BY sender_cif
            ORDER BY ts
            RANGE BETWEEN INTERVAL '24 HOURS' PRECEDING AND CURRENT ROW
        ) AS tx_avg_24h,

        -- ======== TRANSACTION MAX (per sender, 24h) ========
        MAX(amount_vnd) OVER (
            PARTITION BY sender_cif
            ORDER BY ts
            RANGE BETWEEN INTERVAL '24 HOURS' PRECEDING AND CURRENT ROW
        ) AS tx_max_24h,

        -- ======== AMOUNT Z-SCORE (24h window) ========
        CASE
            WHEN STDDEV_POP(amount_vnd) OVER (
                PARTITION BY sender_cif
                ORDER BY ts
                RANGE BETWEEN INTERVAL '24 HOURS' PRECEDING AND CURRENT ROW
            ) = 0 THEN 0.0
            ELSE (
                amount_vnd - AVG(amount_vnd) OVER (
                    PARTITION BY sender_cif
                    ORDER BY ts
                    RANGE BETWEEN INTERVAL '24 HOURS' PRECEDING AND CURRENT ROW
                )
            ) / STDDEV_POP(amount_vnd) OVER (
                PARTITION BY sender_cif
                ORDER BY ts
                RANGE BETWEEN INTERVAL '24 HOURS' PRECEDING AND CURRENT ROW
            )
        END AS amt_zscore_24h

    FROM {table_name}
    ORDER BY ts
    """

    df_velocity = con.execute(query).fetchdf()
    print(f"  [Velocity] Computed {len(df_velocity.columns) - 1} velocity features.")
    return df_velocity


def compute_distinct_count_features(con: duckdb.DuckDBPyConnection, table_name: str = 'transactions') -> pd.DataFrame:
    """
    Compute COUNT(DISTINCT) features over 24h windows via self-join.
    DuckDB doesn't support COUNT(DISTINCT) in window functions, so we use a
    range-join approach.
    
    Returns:
        DataFrame with transaction_id + unique_receivers_24h + unique_devices_24h.
    """
    print("  [Velocity] Computing distinct-count features via range self-join...")

    query = f"""
    WITH lookback AS (
        SELECT
            t1.transaction_id,
            COUNT(DISTINCT t2.receiver_cif) AS unique_receivers_24h,
            COUNT(DISTINCT t2.device_mac_hash) AS unique_devices_24h
        FROM {table_name} t1
        JOIN {table_name} t2
            ON t1.sender_cif = t2.sender_cif
            AND t2.ts BETWEEN t1.ts - INTERVAL '24 HOURS' AND t1.ts
        GROUP BY t1.transaction_id
    )
    SELECT * FROM lookback
    """

    df_distinct = con.execute(query).fetchdf()
    print(f"  [Velocity] Computed unique_receivers_24h & unique_devices_24h.")
    return df_distinct
