"""
Feature Engineering Pipeline — Main Orchestrator.

Loads the 550k merged transaction dataset (CTGAN + WGAN-GP) into DuckDB,
runs all feature modules (velocity, temporal, graph), merges results, and
saves the feature-enriched dataset to data/processed/features_550k.csv.
"""
import os
import time
import duckdb
import pandas as pd

# Feature modules
from velocity_features import compute_velocity_features, compute_distinct_count_features
from temporal_features import compute_temporal_features
from graph_features import compute_graph_features


def run_pipeline():
    start_time = time.time()

    # ─── 1. Resolve paths ───
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '../../'))
    input_path = os.path.join(project_root, 'data', 'synthetic', 'vietnam_transactions_550k.csv')
    output_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'features_550k.csv')

    # ─── 2. Load data into DuckDB ───
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    print(f"\nLoading data from: {input_path}")

    con = duckdb.connect()

    # Read CSV and create a table with a proper TIMESTAMP column
    con.execute(f"""
        CREATE TABLE transactions AS
        SELECT
            *,
            CAST(timestamp AS TIMESTAMP) AS ts
        FROM read_csv_auto('{input_path.replace(os.sep, '/')}')
    """)

    row_count = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    print(f"Loaded {row_count:,} transactions into DuckDB.\n")

    # ─── 3. Load as Pandas for temporal features ───
    df_base = con.execute("""
        SELECT transaction_id, timestamp, sender_cif
        FROM transactions
        ORDER BY ts
    """).fetchdf()

    # ─── 4. Run feature modules ───

    # 4a. Velocity features (DuckDB window functions)
    print("[1/4] Velocity Features...")
    t0 = time.time()
    df_velocity = compute_velocity_features(con)
    print(f"       Done in {time.time() - t0:.1f}s\n")

    # 4b. Distinct-count features (DuckDB self-join)
    print("[2/4] Distinct-Count Features...")
    t0 = time.time()
    df_distinct = compute_distinct_count_features(con)
    print(f"       Done in {time.time() - t0:.1f}s\n")

    # 4c. Temporal features (Pandas)
    print("[3/4] Temporal Features...")
    t0 = time.time()
    df_temporal = compute_temporal_features(df_base)
    print(f"       Done in {time.time() - t0:.1f}s\n")

    # 4d. Graph features (DuckDB)
    print("[4/4] Graph / Network Features...")
    t0 = time.time()
    df_graph = compute_graph_features(con)
    print(f"       Done in {time.time() - t0:.1f}s\n")

    # ─── 5. Merge all features ───
    print("Merging all feature sets...")

    # Get the full base data (all original columns)
    df_full = con.execute("""
        SELECT
            transaction_id, timestamp, amount_vnd, sender_cif, receiver_cif,
            receiver_bank_code, transaction_type, is_biometric_verified,
            device_mac_hash, is_fraud, fraud_type
        FROM transactions
        ORDER BY ts
    """).fetchdf()

    # Merge on transaction_id
    df_final = df_full.copy()
    for df_feat, name in [
        (df_velocity, 'velocity'),
        (df_distinct, 'distinct'),
        (df_temporal, 'temporal'),
        (df_graph, 'graph'),
    ]:
        df_final = df_final.merge(df_feat, on='transaction_id', how='left', suffixes=('', f'_dup_{name}'))

    # Drop any accidental duplicate columns
    dup_cols = [c for c in df_final.columns if c.endswith(('_dup_velocity', '_dup_distinct', '_dup_temporal', '_dup_graph'))]
    if dup_cols:
        df_final.drop(columns=dup_cols, inplace=True)

    # ─── 6. Save ───
    print(f"Saving to: {output_path}")
    df_final.to_csv(output_path, index=False)

    # ─── 7. Report ───
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total time:          {elapsed:.1f}s")
    print(f"Output rows:         {len(df_final):,}")
    print(f"Output columns:      {len(df_final.columns)}")
    print(f"Output file:         {output_path}")
    print(f"\nFeature columns ({len(df_final.columns) - 11} engineered + 11 original):")
    for col in df_final.columns:
        null_pct = df_final[col].isnull().mean() * 100
        print(f"  {col:<30} nulls: {null_pct:.2f}%")
    print("=" * 60)

    con.close()


if __name__ == "__main__":
    run_pipeline()
