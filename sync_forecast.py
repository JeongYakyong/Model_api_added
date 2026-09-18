"""sync_forecast.py — jeju_model 의 수요 예측을 읽어와 자체 DB에 복사하는 cron 진입점.

jeju_model 은 이미 매일 12z(00:20 KST)·18z(08:00 KST) cron 으로 est_horizon_jeju 에
수요 예측을 쌓는다. 이 스크립트는 그 DB를 read-only 로 열어(직접 조회하지 않고 주기
복사하는 쪽을 선택함 — 두 앱을 결합시키지 않기 위해) '내일·모레'(horizon_d 1, 2) 만
freshest-wins 로 뽑아 자체 forecast_data 테이블에 넣는다.

    python sync_forecast.py

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
from utils.db_manager import init_db, save_forecast

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('sync_forecast')

JEJU_MODEL_DB_PATH = os.getenv("JEJU_MODEL_DB_PATH")
HORIZON_DAYS = (1, 2)   # 내일·모레만 — 당일(0)은 제외 (사용자 확정)
DAYS_AHEAD = 2          # 동기화 대상 창 = 오늘 00시 ~ 오늘+2일 23시

# jeju_model pages/common.py 의 _hz_select(mode="latest") 와 동일한 freshest-wins 쿼리:
# 목표 시각마다 지평이 가장 짧고(horizon_d ASC) 가장 최근 발표(base DESC)인 값 1건만 선택.
FRESHEST_SQL = """
    SELECT timestamp, base, horizon_d, est_demand_jeju AS est_demand FROM (
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY timestamp ORDER BY horizon_d ASC, base DESC
        ) rn
        FROM est_horizon_jeju
        WHERE horizon_d IN (?, ?) AND timestamp BETWEEN ? AND ?
    )
    WHERE rn = 1
    ORDER BY timestamp
"""


def fetch_freshest_demand_forecast(retries=3, retry_wait=2):
    if not JEJU_MODEL_DB_PATH:
        raise RuntimeError("JEJU_MODEL_DB_PATH 가 .env 에 설정되어 있지 않습니다.")
    if not os.path.exists(JEJU_MODEL_DB_PATH):
        raise FileNotFoundError(f"jeju_model DB 를 찾을 수 없습니다: {JEJU_MODEL_DB_PATH}")

    today = pd.Timestamp.now().normalize()
    start = today.strftime("%Y-%m-%d 00:00:00")
    end = (today + pd.Timedelta(days=DAYS_AHEAD)).strftime("%Y-%m-%d 23:00:00")

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            con = sqlite3.connect(f"file:{JEJU_MODEL_DB_PATH}?mode=ro", uri=True)
            try:
                df = pd.read_sql_query(FRESHEST_SQL, con, params=(*HORIZON_DAYS, start, end))
            finally:
                con.close()
            return df
        except sqlite3.OperationalError as e:
            last_err = e
            logger.warning(f"jeju_model DB 조회 실패({attempt}/{retries}), {retry_wait}s 후 재시도: {e}")
            time.sleep(retry_wait)
    raise last_err


def main():
    init_db()
    df = fetch_freshest_demand_forecast()
    if df.empty:
        logger.warning("jeju_model 에 동기화할 예측 데이터가 없습니다.")
        return
    save_forecast(df)
    logger.info(f"수요 예측 {len(df)}행 동기화 완료 (base 최신: {df['base'].max()})")


if __name__ == "__main__":
    main()
