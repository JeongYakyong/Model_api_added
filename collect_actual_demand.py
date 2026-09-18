"""collect_actual_demand.py — KPX 실측 수요 수집 cron 진입점.

기존에는 관리자 화면 버튼을 눌러야만 실행됐다. 이제 crontab 이 매시 자동으로 돌린다.
    python collect_actual_demand.py
"""
import sys
import os
import logging
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.api_fetchers import fetch_kpx_past
from utils.db_manager import init_db, save_historical

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('collect_actual_demand')


def main():
    init_db()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')  # 결측 자동 보정용 여유분

    df = fetch_kpx_past(start_date, end_date)
    if df.empty:
        logger.error("KPX 실측 수요 수집 실패 또는 빈 응답")
        sys.exit(1)

    save_historical(df)
    logger.info(f"실측 수요 {len(df)}행 저장 완료 ({start_date} ~ {end_date})")


if __name__ == "__main__":
    main()
