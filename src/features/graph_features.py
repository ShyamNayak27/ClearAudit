"""
Graph / Network Features — DuckDB-powered sender-receiver network metrics.

Computes node degree (in/out), repeat-pair indicators, pair transaction counts,
and sender bank diversity using efficient SQL aggregations.
"""
import duckdb
import pandas as pd


def compute_graph_features(con: duckdb.DuckDBPyConnection, table_name: str = 'transactions') -> pd.DataFrame:
    """
    Compute graph/network features using DuckDB SQL aggregations.
    
    Args:
        con: Active DuckDB connection with the transactions table registered.
        table_name: Name of the registered table/view.
    
    Returns:
        DataFrame with transaction_id + all graph feature columns.
    """
    print("  [Graph] Computing network topology features via DuckDB...")

    query = f"""
    WITH

    -- Pre-compute sender out-degree (total unique receivers per sender)
    sender_stats AS (
        SELECT
            sender_cif,
            COUNT(DISTINCT receiver_cif) AS sender_out_degree,
            COUNT(DISTINCT receiver_bank_code) AS sender_bank_diversity
        FROM {table_name}
        GROUP BY sender_cif
    ),

    -- Pre-compute receiver in-degree (total unique senders per receiver)
    receiver_stats AS (
        SELECT
            receiver_cif,
            COUNT(DISTINCT sender_cif) AS receiver_in_degree
        FROM {table_name}
        GROUP BY receiver_cif
    ),

    -- Pre-compute pair-level stats
    pair_stats AS (
        SELECT
            sender_cif,
            receiver_cif,
            COUNT(*) AS pair_tx_count
        FROM {table_name}
        GROUP BY sender_cif, receiver_cif
    )

    SELECT
        t.transaction_id,

        -- Node-level features
        COALESCE(ss.sender_out_degree, 0) AS sender_out_degree,
        COALESCE(rs.receiver_in_degree, 0) AS receiver_in_degree,
        COALESCE(ss.sender_bank_diversity, 0) AS sender_bank_diversity,

        -- Pair-level features
        COALESCE(ps.pair_tx_count, 0) AS pair_tx_count,
        CASE WHEN COALESCE(ps.pair_tx_count, 0) > 1 THEN 1 ELSE 0 END AS is_repeat_pair

    FROM {table_name} t
    LEFT JOIN sender_stats ss ON t.sender_cif = ss.sender_cif
    LEFT JOIN receiver_stats rs ON t.receiver_cif = rs.receiver_cif
    LEFT JOIN pair_stats ps ON t.sender_cif = ps.sender_cif AND t.receiver_cif = ps.receiver_cif
    ORDER BY t.ts
    """

    df_graph = con.execute(query).fetchdf()
    feature_count = len(df_graph.columns) - 1
    print(f"  [Graph] Computed {feature_count} graph/network features.")
    return df_graph
