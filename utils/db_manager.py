import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database", "jeju_energy.db")


def _ensure_columns(con, table, columns):
    """구버전 스키마(더 적은 컬럼)로 이미 테이블이 있던 경우를 위한 안전장치 —
    필요한 컬럼이 없으면 ALTER TABLE 로 추가한다(기존 값은 건드리지 않음)."""
    existing = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
    for name, coltype in columns:
        if name not in existing:
            con.execute(f'ALTER TABLE {table} ADD COLUMN "{name}" {coltype}')


def init_db(db_path=DB_PATH):
    """historical_data(실측)·forecast_data(jeju_model 동기화) 테이블을 준비한다."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS historical_data (
            timestamp TEXT PRIMARY KEY,
            real_demand REAL,
            updated_at TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS forecast_data (
            timestamp TEXT PRIMARY KEY,
            est_demand REAL,
            horizon_d INTEGER,
            base TEXT,
            updated_at TEXT
        )
    """)
    _ensure_columns(con, "historical_data", [("real_demand", "REAL"), ("updated_at", "TEXT")])
    _ensure_columns(con, "forecast_data",
                    [("est_demand", "REAL"), ("horizon_d", "INTEGER"), ("base", "TEXT"), ("updated_at", "TEXT")])
    con.commit()
    con.close()


def save_historical(df, db_path=DB_PATH):
    """실측 수요 upsert — 기존 값은 COALESCE 로 보존(부분 재수집 시 NULL 로 덮지 않음)."""
    if df.empty:
        return
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    con = sqlite3.connect(db_path)
    for timestamp, row in df.iterrows():
        con.execute("""
            INSERT INTO historical_data (timestamp, real_demand, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(timestamp) DO UPDATE SET
                real_demand = COALESCE(excluded.real_demand, historical_data.real_demand),
                updated_at = excluded.updated_at
        """, (timestamp, row['real_demand'], now))
    con.commit()
    con.close()


def save_forecast(df, db_path=DB_PATH):
    """예측 수요 upsert — jeju_model 에서 새로 받은 값으로 항상 덮어쓴다(freshest-wins 이미 적용됨)."""
    if df.empty:
        return
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    con = sqlite3.connect(db_path)
    for row in df.itertuples(index=False):
        con.execute("""
            INSERT INTO forecast_data (timestamp, est_demand, horizon_d, base, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(timestamp) DO UPDATE SET
                est_demand = excluded.est_demand,
                horizon_d = excluded.horizon_d,
                base = excluded.base,
                updated_at = excluded.updated_at
        """, (row.timestamp, row.est_demand, row.horizon_d, row.base, now))
    con.commit()
    con.close()


def load_range(start, end, db_path=DB_PATH):
    """[start, end] 구간의 실측+예측 수요를 timestamp 기준으로 합쳐 반환한다."""
    import pandas as pd
    con = sqlite3.connect(db_path)
    hist = pd.read_sql_query(
        "SELECT timestamp, real_demand FROM historical_data WHERE timestamp BETWEEN ? AND ?",
        con, params=(start, end))
    fcst = pd.read_sql_query(
        "SELECT timestamp, est_demand FROM forecast_data WHERE timestamp BETWEEN ? AND ?",
        con, params=(start, end))
    con.close()
    base = pd.DataFrame({"timestamp": pd.date_range(start, end, freq="h").strftime('%Y-%m-%d %H:%M:%S')})
    df = base.merge(hist, on="timestamp", how="left").merge(fcst, on="timestamp", how="left")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
