import sqlite3
import os
import pandas as pd
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database", "jeju_energy.db")

HIST_COLS = ["real_demand", "real_renew_gen", "real_solar_gen", "real_wind_gen"]
FCST_COLS = ["est_demand", "est_renew_gen", "est_solar_gen", "est_wind_gen", "est_net_load"]


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
            updated_at TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS forecast_data (
            timestamp TEXT PRIMARY KEY,
            horizon_d INTEGER,
            base TEXT,
            updated_at TEXT
        )
    """)
    _ensure_columns(con, "historical_data", [(c, "REAL") for c in HIST_COLS] + [("updated_at", "TEXT")])
    _ensure_columns(con, "forecast_data",
                    [(c, "REAL") for c in FCST_COLS] + [("horizon_d", "INTEGER"), ("base", "TEXT"), ("updated_at", "TEXT")])
    con.commit()
    con.close()


def save_historical(df, db_path=DB_PATH):
    """실측 upsert — 기존 값은 COALESCE 로 보존(부분 재수집 시 NULL 로 덮지 않음)."""
    if df.empty:
        return
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    con = sqlite3.connect(db_path)
    collist = ", ".join(HIST_COLS)
    placeholders = ", ".join(["?"] * len(HIST_COLS))
    coalesce_set = ", ".join(f"{c} = COALESCE(excluded.{c}, historical_data.{c})" for c in HIST_COLS)
    for timestamp, row in df.iterrows():
        con.execute(f"""
            INSERT INTO historical_data (timestamp, {collist}, updated_at)
            VALUES (?, {placeholders}, ?)
            ON CONFLICT(timestamp) DO UPDATE SET
                {coalesce_set},
                updated_at = excluded.updated_at
        """, (timestamp, *[row.get(c) for c in HIST_COLS], now))
    con.commit()
    con.close()


def save_forecast(df, db_path=DB_PATH):
    """예측 upsert — jeju_model 에서 새로 받은 값으로 항상 덮어쓴다(freshest-wins 이미 적용됨)."""
    if df.empty:
        return
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    con = sqlite3.connect(db_path)
    collist = ", ".join(FCST_COLS)
    placeholders = ", ".join(["?"] * len(FCST_COLS))
    overwrite_set = ", ".join(f"{c} = excluded.{c}" for c in FCST_COLS)
    for row in df.itertuples(index=False):
        row_d = row._asdict()
        con.execute(f"""
            INSERT INTO forecast_data (timestamp, {collist}, horizon_d, base, updated_at)
            VALUES (?, {placeholders}, ?, ?, ?)
            ON CONFLICT(timestamp) DO UPDATE SET
                {overwrite_set},
                horizon_d = excluded.horizon_d,
                base = excluded.base,
                updated_at = excluded.updated_at
        """, (row_d["timestamp"], *[row_d.get(c) for c in FCST_COLS], row_d["horizon_d"], row_d["base"], now))
    con.commit()
    con.close()


def load_range(start, end, db_path=DB_PATH):
    """[start, end] 구간의 실측+예측을 timestamp 기준으로 합쳐 반환한다.

    jeju_model pages/common.py 의 jeju_range_compare 와 같은 방식으로 신재생 합계·순부하는
    저장하지 않고 조회 시점에 구성한다: est_renew_gen 은 이미 sync_forecast.py 가 담아 오지만
    real_renew_gen 은 KPX 원본값을 그대로 쓰고, real_net_load 는 real_demand - real_renew_gen 으로 계산한다.
    """
    con = sqlite3.connect(db_path)
    hist = pd.read_sql_query(
        f"SELECT timestamp, {', '.join(HIST_COLS)} FROM historical_data WHERE timestamp BETWEEN ? AND ?",
        con, params=(start, end))
    fcst = pd.read_sql_query(
        f"SELECT timestamp, base, {', '.join(FCST_COLS)} FROM forecast_data WHERE timestamp BETWEEN ? AND ?",
        con, params=(start, end))
    con.close()
    base = pd.DataFrame({"timestamp": pd.date_range(start, end, freq="h").strftime('%Y-%m-%d %H:%M:%S')})
    df = base.merge(hist, on="timestamp", how="left").merge(fcst, on="timestamp", how="left")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["real_net_load"] = df["real_demand"] - df["real_renew_gen"]
    return df
