"""sync_forecast.py — jeju_model 의 예측(수요·순부하·신재생)과 기상 예보를 읽어와 자체 DB에 복사하는 cron 진입점.

jeju_model 은 이미 매일 12z(00:20 KST)·18z(08:00 KST) cron 으로 est_horizon_jeju 에
수요·태양광·풍력·순부하 예측을 쌓는다. 이 스크립트는 그 DB를 read-only 로 열어(직접
조회하지 않고 주기 복사하는 쪽을 선택함 — 두 앱을 결합시키지 않기 위해) freshest-wins
(목표 시각마다 지평이 가장 짧고 가장 최근 발표인 값)로 뽑아 자체 forecast_data 테이블에 넣는다.

    python sync_forecast.py

기상 예보(구름·바람 탭의 표)는 jeju_model 예측에 실제로 들어가는 입력과 같은 출처로 맞춘다:
일사량·강수량 = forecast_horizon(KIMG, 같은 freshest-wins), 운량 = forecast_jma(JMA, 최신 실행).

.env 의 JEJU_MODEL_DB_PATH 로 jeju_model의 input_data_jeju.db 절대경로를 지정한다.
"""
import sys
import os
import sqlite3
import time
import logging
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.db_manager import init_db, save_forecast, save_weather_forecast, WEATHER_ZONES

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('sync_forecast')

JEJU_MODEL_DB_PATH = os.getenv("JEJU_MODEL_DB_PATH")
HORIZON_MIN, HORIZON_MAX = 1, 5   # jeju_model 운영 지평(JEJU_HZ_MAX)과 동일 — 당일(0)은 제외
DAYS_AHEAD = HORIZON_MAX          # 동기화 대상 창 = 오늘 00시 ~ 오늘+5일 23시

# jeju_model pages/common.py 의 _hz_select(mode="latest") 와 동일한 freshest-wins 쿼리:
# 목표 시각마다 지평이 가장 짧고(horizon_d ASC) 가장 최근 발표(base DESC)인 값 1건만 선택.
# est_renew_gen 은 jeju_model 에 저장된 컬럼이 아니라 태양광+풍력 합으로 여기서 구성한다
# (jeju_model pages/common.py jeju_range_compare 와 동일한 방식).
FRESHEST_SQL = """
    SELECT timestamp, base, horizon_d,
           est_demand_jeju AS est_demand,
           est_solar_gen_jeju AS est_solar_gen,
           est_wind_gen_jeju AS est_wind_gen,
           (est_solar_gen_jeju + est_wind_gen_jeju) AS est_renew_gen,
           est_net_load_jeju AS est_net_load
    FROM (
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY timestamp ORDER BY horizon_d ASC, base DESC
        ) rn
        FROM est_horizon_jeju
        WHERE horizon_d BETWEEN ? AND ? AND timestamp BETWEEN ? AND ?
    )
    WHERE rn = 1
    ORDER BY timestamp
"""


# 일사량(MJ/m²)·강수량(mm) — 위 FRESHEST_SQL 과 같은 기준으로 목표 시각마다 1건
WEATHER_KIMG_SQL = f"""
    SELECT timestamp,
           {", ".join(f"radiation_{zone}, rainfall_{zone}" for zone in WEATHER_ZONES)}
    FROM (
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY timestamp ORDER BY horizon_d ASC, base DESC
        ) rn
        FROM forecast_horizon
        WHERE horizon_d BETWEEN ? AND ? AND timestamp BETWEEN ? AND ?
    )
    WHERE rn = 1
"""

# 운량(0~1) — JMA 는 실행(run_time_utc)마다 여러 시각을 내므로 목표 시각마다 가장 최근 실행 1건
WEATHER_JMA_SQL = f"""
    SELECT timestamp,
           {", ".join(f"total_cloud_{zone}" for zone in WEATHER_ZONES)}
    FROM (
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY timestamp ORDER BY run_time_utc DESC
        ) rn
        FROM forecast_jma
        WHERE timestamp BETWEEN ? AND ?
    )
    WHERE rn = 1
"""


def _sync_window():
    """동기화 대상 창 = 어제 00시 ~ 오늘+5일 23시 (어제분은 구름·바람 탭의 최근 24시간 표시용)."""
    today = pd.Timestamp.now().normalize()
    start = (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")
    end = (today + pd.Timedelta(days=DAYS_AHEAD)).strftime("%Y-%m-%d 23:00:00")
    return start, end


def _query_jeju_model(sql, params, retries=3, retry_wait=2):
    """jeju_model DB 를 read-only 로 열어 조회한다(그쪽 cron 이 쓰는 중이면 잠깐 기다렸다 재시도)."""
    if not JEJU_MODEL_DB_PATH:
        raise RuntimeError("JEJU_MODEL_DB_PATH 가 .env 에 설정되어 있지 않습니다.")
    if not os.path.exists(JEJU_MODEL_DB_PATH):
        raise FileNotFoundError(f"jeju_model DB 를 찾을 수 없습니다: {JEJU_MODEL_DB_PATH}")

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            con = sqlite3.connect(f"file:{JEJU_MODEL_DB_PATH}?mode=ro", uri=True)
            try:
                df = pd.read_sql_query(sql, con, params=params)
            finally:
                con.close()
            return df
        except sqlite3.OperationalError as e:
            last_err = e
            logger.warning(f"jeju_model DB 조회 실패({attempt}/{retries}), {retry_wait}s 후 재시도: {e}")
            time.sleep(retry_wait)
    raise last_err


def fetch_freshest_forecast():
    start, end = _sync_window()
    return _query_jeju_model(FRESHEST_SQL, (HORIZON_MIN, HORIZON_MAX, start, end))


def fetch_weather_forecast():
    start, end = _sync_window()
    kimg = _query_jeju_model(WEATHER_KIMG_SQL, (HORIZON_MIN, HORIZON_MAX, start, end))
    jma = _query_jeju_model(WEATHER_JMA_SQL, (start, end))
    return kimg.merge(jma, on="timestamp", how="outer").sort_values("timestamp")


def main():
    init_db()
    df = fetch_freshest_forecast()
    if df.empty:
        logger.warning("jeju_model 에 동기화할 예측 데이터가 없습니다.")
        return
    save_forecast(df)
    logger.info(f"예측 {len(df)}행 동기화 완료 (base 최신: {df['base'].max()})")

    weather = fetch_weather_forecast()
    save_weather_forecast(weather)
    logger.info(f"기상 예보 {len(weather)}행 동기화 완료")


if __name__ == "__main__":
    main()
