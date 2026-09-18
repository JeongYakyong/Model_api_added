import pandas as pd
import numpy as np
import requests
import io
import logging

logger = logging.getLogger('jejucr.api')
logger.setLevel(logging.DEBUG)


def fetch_kpx_past(start_date, end_date):
    """KPX 제주 수급현황에서 실측 수요·신재생 발전(MW)을 가져온다 (인증키 불필요).

    start_date/end_date: 'YYYY-MM-DD' (하이픈 포함).
    반환: timestamp 인덱스(1시간 단위) + real_demand/real_renew_gen/real_solar_gen/real_wind_gen.
    """
    url = "https://openapi.kpx.or.kr/downloadChejuSukubCSV.do"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://openapi.kpx.or.kr/chejusukub.do"
    }
    payload = {'startDate': start_date, 'endDate': end_date}
    power_cols = ['real_demand', 'real_renew_gen', 'real_solar_gen', 'real_wind_gen']

    try:
        resp = requests.post(url, data=payload, headers=headers, timeout=30)
        resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = df.columns.str.strip()

        # 1시간 단위 필터링 (0000으로 끝나는 행)
        df = df[df['기준일시'].astype(str).str.endswith('0000')].copy()

        df['timestamp'] = pd.to_datetime(
            df['기준일시'].astype(str), format='%Y%m%d%H%M%S'
        ).dt.strftime('%Y-%m-%d %H:%M:%S')

        df = df.rename(columns={
            '현재수요(MW)': 'real_demand',
            '신재생총합(MW)': 'real_renew_gen',
            '신재생태양광(MW)': 'real_solar_gen',
            '신재생풍력(MW)': 'real_wind_gen',
        })
        result = df.set_index('timestamp')[power_cols].apply(pd.to_numeric, errors='coerce')

        # demand=0 은 계측 오류 (실제 수요가 0이 될 수 없음) → 해당 행 전체 NaN 처리 후 양방향 보간(최대 3개 연속)
        zero_mask = result['real_demand'] == 0
        if zero_mask.any():
            result.loc[zero_mask, power_cols] = np.nan
            logger.warning(f"[KPX Past] demand=0 오류 {zero_mask.sum()}행 → NaN 양방향 보간")
        result.index = pd.to_datetime(result.index)
        result = result.interpolate(method='time', limit=3, limit_direction='both')
        result.index = result.index.strftime('%Y-%m-%d %H:%M:%S')

        logger.info(f"[KPX Past] {len(result)}행 수집")
        return result

    except Exception as e:
        logger.error(f"[KPX Past] 실패: {e}")
        return pd.DataFrame()
