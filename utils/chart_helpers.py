import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# chart_helpers.py 맨 아래에 추가
# ==========================================

def draw_danger_zones(fig, df, condition_series, fill_color,
                      annotation_text=None, show_legend_label=None,
                      layer_pos="below", fill_opacity=0.15):
    """
    Plotly figure에 위험 구간 음영(vrect)을 추가하는 헬퍼.

    Parameters
    ----------
    fig : go.Figure          — 음영을 추가할 Plotly 차트
    df  : pd.DataFrame       — condition_series와 같은 인덱스를 공유하는 데이터프레임
    condition_series : pd.Series[bool] — True인 구간에 음영 표시
    fill_color : str         — 음영 색상 (예: "red", "blue")
    annotation_text : str    — 음영 위에 표시할 텍스트 (None이면 생략)
    show_legend_label : str  — 범례에 표시할 이름 (None이면 생략)
    layer_pos : str          — "below" 또는 "above"
    fill_opacity : float     — 음영 투명도 (기본 0.15)
    """
    if not condition_series.any():
        return

    danger_df = df[condition_series].copy()
    danger_df['group'] = (condition_series != condition_series.shift()).cumsum()
    danger_df['temp_time'] = danger_df.index

    danger_zones = danger_df.groupby('group').agg(
        start=('temp_time', 'min'),
        end=('temp_time', 'max')
    )

    for _, row in danger_zones.iterrows():
        start_time = row['start'] - timedelta(hours=1)
        end_time = row['end'] + timedelta(hours=1)
        fig.add_vrect(
            x0=start_time, x1=end_time,
            fillcolor=fill_color, opacity=fill_opacity,
            layer=layer_pos, line_width=0,
            annotation_text=annotation_text,
            annotation_position="top left" if annotation_text else None
        )

    if show_legend_label:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=fill_color, symbol='square'),
            name=show_legend_label, showlegend=True
        ))
        
# ==========================================
# 전역 상수
# ==========================================
EDA_ONLY_COLUMNS = {'HVDC_Total', 'LNG_Gen', 'Oil_Gen'}
PREDICTION_OUTPUT_COLUMNS = {'est_Solar_Utilization', 'est_Wind_Utilization'}

COLORS = {
    'solar_real': 'darkorange',
    'solar_est': 'orange',          # dark mode 호환성 향상을 위한 색변경
    'wind_real': 'dodgerblue',      # darkblue → dodgerblue
    'wind_est': 'lightskyblue',     # skyblue → lightskyblue
    'demand': 'darkgray',           # gray → darkgray
    'net_demand': 'tomato',         # red → tomato
    'renew_total': 'limegreen',     # green → limegreen
    'smp': 'orchid',                # purple → orchid
    'error': 'tomato',              # red → tomato
}

# est_ 변수별 색상 (COLORS 기반)
EST_COLORS = {
    'est_demand':      COLORS['demand'],
    'est_net_demand':  COLORS['net_demand'],
    'est_solar_gen':   COLORS['solar_est'],
    'est_wind_gen':    COLORS['wind_est'],
    'est_renew_total': COLORS['renew_total'],
}

# 예측 항목 레이블
PLOT_OPTIONS = {
    'est_demand':      '총 전력수요 예측 (est_demand)',
    'est_net_demand':  '순부하 예측 (est_net_demand)',
    'est_solar_gen':   '태양광 발전량 예측 (est_solar_gen)',
    'est_wind_gen':    '풍력 발전량 예측 (est_wind_gen)',
    'est_renew_total': '총 재생에너지 발전량 (est_renew_total)',
}

# 실측 항목 레이블
ACTUAL_LABEL_MAP = {
    'real_demand':      '수요 실측',
    'real_solar_gen':   '태양광 실측',
    'real_wind_gen':    '풍력 실측',
    'real_renew_total': '재생E 실측 합계',
    'real_net_demand':  '순부하 실측',
}

# 예측 → 실측 컬럼 매핑
ACTUAL_MAP = {
    'est_solar_gen':   'real_solar_gen',
    'est_wind_gen':    'real_wind_gen',
    'est_renew_total': 'real_renew_total',
    'est_net_demand':  'real_net_demand',
    'est_demand':      'real_demand',
}


# ==========================================
# 데이터 무결성 검사
# ==========================================
def check_data_status(df, key_columns=None):
    """
    데이터프레임의 무결성을 검사하는 함수.
    EDA_ONLY_COLUMNS에 포함된 컬럼은 결측치 검사에서 제외됩니다.
    """
    if df.empty:
        return {
            "status": "Empty",
            "missing_timestamps": 0,
            "nan_counts": {},
            "incomplete_rows": 0,
            "incomplete_details": {}
        }

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    if key_columns is None:
        key_columns = [
            c for c in df.select_dtypes(include=[np.number]).columns.tolist()
            if c not in EDA_ONLY_COLUMNS
        ]
    key_columns = [c for c in key_columns if c in df.columns]

    # 1. 시계열 누락
    expected_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1h')
    missing_timestamps = expected_range.difference(df.index)

    # 2. 전체 컬럼 결측치 (EDA 참고용 제외)
    check_cols = [c for c in df.columns if c not in EDA_ONLY_COLUMNS]
    nan_counts = df[check_cols].isna().sum()
    nan_counts = nan_counts[nan_counts > 0].to_dict()

    # 3. 주요 컬럼 기준 불완전 행
    if key_columns:
        key_df = df[key_columns]
        incomplete_mask = key_df.isna().any(axis=1)
        incomplete_rows = int(incomplete_mask.sum())
        incomplete_details = key_df.isna().sum()
        incomplete_details = incomplete_details[incomplete_details > 0].to_dict()
    else:
        incomplete_rows = 0
        incomplete_details = {}

    has_problem = len(missing_timestamps) > 0 or nan_counts or incomplete_rows > 0

    return {
        "status": "Warning" if has_problem else "Good",
        "missing_timestamps": len(missing_timestamps),
        "missing_dates": missing_timestamps.tolist()[:5],
        "nan_counts": nan_counts,
        "incomplete_rows": incomplete_rows,
        "incomplete_details": incomplete_details,
        "key_columns_checked": key_columns
    }


# ==========================================
# 기간 선택 위젯
# ==========================================
def date_range_selector(key_prefix, allow_future_days=0, default_option="1주"):
    """
    버튼 한 줄로 큰 기간을 선택 → 슬라이더로 세부 구간 조절하는 공통 헬퍼 함수.
    """
    today = datetime.now().date()
    max_date = today + timedelta(days=allow_future_days)

    state_key = f"date_range_{key_prefix}"
    if state_key not in st.session_state:
        st.session_state[state_key] = default_option

    options = {
        "하루": 1,
        "1주": 7,
        "2주": 14,
        "30일": 30,
        "90일": 90,
        "1년": 365,
    }

    cols = st.columns(len(options) + 1)

    for i, (label, _) in enumerate(options.items()):
        is_active = st.session_state[state_key] == label
        button_type = "primary" if is_active else "secondary"
        if cols[i].button(label, key=f"{key_prefix}_btn_{label}", type=button_type, width="stretch"):
            st.session_state[state_key] = label
            st.rerun()

    is_custom = st.session_state[state_key] == "기간선택"
    custom_type = "primary" if is_custom else "secondary"
    if cols[-1].button("기간선택", key=f"{key_prefix}_btn_custom", type=custom_type, width="stretch"):
        st.session_state[state_key] = "기간선택"
        st.rerun()

    current_selection = st.session_state[state_key]

    if current_selection == "기간선택":
        date_range = st.date_input(
            "달력에서 직접 선택",
            [today - timedelta(days=7), today],
            max_value=max_date,
            key=f"{key_prefix}_custom_date"
        )
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = date_range[0], date_range[0]
    else:
        days_back = options[current_selection]
        range_start = today - timedelta(days=days_back)
        range_end = today if allow_future_days == 0 else min(today + timedelta(days=allow_future_days), max_date)

        if days_back <= 1:
            return range_start, range_end

        start_date, end_date = st.slider(
            "📅 구간 조절",
            min_value=range_start,
            max_value=range_end,
            value=(range_start, range_end),
            format="YYYY-MM-DD",
            key=f"{key_prefix}_slider",
            label_visibility="collapsed"
        )

    return start_date, end_date


# ==========================================
# 실측 + 예보 머지
# ==========================================
def merge_actual_and_forecast(db, start_str, end_str):
    """
    실측+예보 머지 후 발전량 계산까지 완료된 DataFrame 반환.
    """
    hist_df = db.get_historical(start_str, end_str)
    fore_df = db.get_forecast(start_str, end_str)

    if hist_df.empty or fore_df.empty:
        return pd.DataFrame()

    merged = pd.merge(hist_df, fore_df, left_index=True, right_index=True, how='inner', suffixes=('', '_fore'))

    if merged.empty:
        return pd.DataFrame()

    cap_solar_col = 'Solar_Capacity_Est_fore' if 'Solar_Capacity_Est_fore' in merged.columns else 'Solar_Capacity_Est'
    cap_wind_col = 'Wind_Capacity_Est_fore' if 'Wind_Capacity_Est_fore' in merged.columns else 'Wind_Capacity_Est'

    for col in ['est_Solar_Utilization', 'est_Wind_Utilization', cap_solar_col, cap_wind_col]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce')

    merged['est_solar_gen'] = merged['est_Solar_Utilization'] * merged[cap_solar_col]
    merged['est_wind_gen'] = merged['est_Wind_Utilization'] * merged[cap_wind_col]

    return merged


# ==========================================
# 실제 vs 예측 비교 차트
# ==========================================
def plot_actual_vs_pred(df, date_title, radio_key):
    """실제 vs 예측 비교 차트."""
    source_choice = st.radio("발전원 선택:", ["태양광 (Solar)", "풍력 (Wind)"], horizontal=True, key=radio_key)

    fig = go.Figure()
    if source_choice == "태양광 (Solar)":
        real_col, est_col = 'real_solar_gen', 'est_solar_gen'
        color_real, color_est = COLORS['solar_real'], COLORS['solar_est']
    else:
        real_col, est_col = 'real_wind_gen', 'est_wind_gen'
        color_real, color_est = COLORS['wind_real'], COLORS['wind_est']

    fig.add_trace(go.Scatter(x=df.index, y=df[real_col], name="실제 발전량", line=dict(color=color_real, width=3)))
    fig.add_trace(go.Scatter(x=df.index, y=df[est_col], name="예측 발전량", line=dict(color=color_est, width=3, dash='dash')))

    df_plot = df.copy()
    df_plot['error'] = df_plot[est_col] - df_plot[real_col]
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['error'], name="오차", marker_color=COLORS['error'], opacity=0.3))

    fig.update_layout(
        title=f"{date_title} {source_choice} 실제 vs 예측 비교",
        hovermode="x unified",
        yaxis_title="발전량 (MW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        dragmode='zoom',
        yaxis=dict(fixedrange=True)
    )
    fig.update_traces(hovertemplate='%{y:,.1f}')
    st.plotly_chart(fig, width="stretch")