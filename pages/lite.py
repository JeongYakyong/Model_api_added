"""
pages/lite.py — 경량 버전 (사이드바 메뉴 3개: 예측확인 / 예측실행 / DB현황)

app.py의 Option C(시각화), Option B(예측), Option A(DB관리) 코드를 기반으로
사이드바 radio + 작은 글씨 + 컴팩트 레이아웃 컨셉.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from utils.data_pipeline import (
    daily_historical_kpx, daily_historical_kma, daily_historical_kpx_smp,
    run_model_prediction, daily_forecast_kpx, daily_forecast_kma
)
from utils.chart_helpers import (
    EDA_ONLY_COLUMNS, PREDICTION_OUTPUT_COLUMNS, COLORS,
    merge_actual_and_forecast, draw_danger_zones
)

from utils.gemini import generate_energy_narrative



# ==========================================
# 공유 리소스
# ==========================================
db = st.session_state['shared_db']
assets = st.session_state['shared_assets']

EXCLUDE = EDA_ONLY_COLUMNS | PREDICTION_OUTPUT_COLUMNS

import json
import os

# 저장할 파일 이름 설정
BRIEFING_FILE = "briefing_storage.json"

def load_briefings_from_file():
    """로컬 JSON 파일에서 브리핑 데이터를 읽어옵니다."""
    if os.path.exists(BRIEFING_FILE):
        try:
            with open(BRIEFING_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_briefing_to_file(date_key, text):
    """특정 날짜의 브리핑을 로컬 JSON 파일에 저장합니다."""
    data = load_briefings_from_file()
    data[date_key] = text
    try:
        with open(BRIEFING_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"File save error: {e}")
        

# ==========================================
# 헬퍼 함수
# ==========================================

def get_data_status(target_date):
    """예측 대상일 기준 과거 실측 / 미래 예보 데이터 상태 점검"""
    past_end   = f"{target_date - timedelta(days=1)} 23:00:00"
    past_start = f"{target_date - timedelta(days=14)} 00:00:00"
    fut_start  = f"{target_date} 00:00:00"
    fut_end    = f"{target_date} 23:00:00"

    past_df = db.get_historical(past_start, past_end)
    fut_df  = db.get_forecast(fut_start, fut_end)

    past_hours = len(past_df) if not past_df.empty else 0
    fut_hours  = len(fut_df)  if not fut_df.empty  else 0

    past_missing = int(
        past_df.drop(columns=EXCLUDE, errors='ignore').isna().any(axis=1).sum()
    ) if not past_df.empty else 0
    fut_missing = int(
        fut_df.drop(columns=EXCLUDE, errors='ignore').isna().any(axis=1).sum()
    ) if not fut_df.empty else 0

    past_ok  = (past_hours >= 336) and (past_missing == 0)
    fut_ok   = (fut_hours  >= 24)  and (fut_missing  == 0)
    past_gap = max(336 - past_hours, 0)
    fut_gap  = max(24  - fut_hours,  0)
    can_quick = (past_gap <= 48)

    return dict(
        past_df=past_df, fut_df=fut_df,
        past_hours=past_hours, fut_hours=fut_hours,
        past_missing=past_missing, fut_missing=fut_missing,
        past_ok=past_ok, fut_ok=fut_ok,
        past_gap=past_gap, fut_gap=fut_gap,
        can_quick=can_quick,
    )


def render_metrics(s):
    """4개 메트릭 카드 렌더링"""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("과거 실측",  f"{s['past_hours']} / 336h",
              "정상" if s['past_ok'] else "부족",
              delta_color="normal" if s['past_ok'] else "inverse")
    c2.metric("미래 예보",  f"{s['fut_hours']} / 24h",
              "정상" if s['fut_ok'] else "부족",
              delta_color="normal" if s['fut_ok'] else "inverse")
    c3.metric("실측 결측",  f"{s['past_missing']}h",
              "없음" if s['past_missing'] == 0 else "있음",
              delta_color="normal" if s['past_missing'] == 0 else "inverse")
    c4.metric("예보 결측",  f"{s['fut_missing']}h",
              "없음" if s['fut_missing'] == 0 else "있음",
              delta_color="normal" if s['fut_missing'] == 0 else "inverse")


def render_heatmap(df, n_cells, cell_unit_hours, axis_labels):
    """결측치 히트맵 렌더링"""
    if df is None or df.empty:
        st.caption("데이터 없음")
        return
    cols = [c for c in df.columns if c not in EXCLUDE]
    if not cols:
        return

    matrix, col_labels = [], []
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        n_rows = len(s)
        blocks = []
        for i in range(n_cells):
            start_i = i * cell_unit_hours
            end_i = min((i + 1) * cell_unit_hours, n_rows)
            if start_i >= n_rows:
                blocks.append(1)
            else:
                blocks.append(1 if s.iloc[start_i:end_i].isna().any() else 0)
        matrix.append(blocks)
        col_labels.append(col)

    if not matrix:
        return

    custom = [['결측' if v else '정상' for v in row] for row in matrix]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=list(range(n_cells)), y=col_labels,
        colorscale=[[0, '#d4edda'], [1, '#f8d7da']],
        showscale=False, xgap=2, ygap=3,
        zmin=0, zmax=1,
        hovertemplate='%{y} · 블록 %{x}: %{customdata}<extra></extra>',
        customdata=custom,
    ))
    tick_pos = [0, n_cells // 2, n_cells - 1]
    fig.update_layout(
        height=max(80, len(col_labels) * 28 + 40),
        margin=dict(l=0, r=0, t=6, b=30),
        xaxis=dict(tickvals=tick_pos, ticktext=axis_labels, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=11)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, width='stretch')


# ==========================================
# 사이드바 메뉴
# ==========================================
lite_menu = st.sidebar.radio("메뉴:", [
    "📈 예측 확인",
    "🚀 예측 실행",
    "🗂️ DB 수집현황",
], key="lite_menu")


# ══════════════════════════════════════════
# 📈 예측 결과 시각화
# (app.py Option C 코드 기반)
# ══════════════════════════════════════════
if lite_menu == "📈 예측 확인":
    st.caption(" ")
    # ── KPX 실측 자동 수집 (50분 쿨다운) ──
    _now = datetime.now()
    _last = st.session_state.get('_lite_kpx_last', None)
    if _last is None or (_now - _last).total_seconds() >= 3000:
        try:
            daily_historical_kpx(
                (_now - timedelta(days=1)).strftime("%Y-%m-%d"),
                _now.strftime("%Y-%m-%d"),
            )
            st.session_state['_lite_kpx_last'] = _now
        except Exception:
            pass

    # ── session_state 초기화 ──
    ss_defaults = {
        'lite_vis_vars':    ['est_demand', 'est_net_demand', 'est_solar_gen', 'est_wind_gen'],
        'lite_vis_actual':  ['real_demand', 'real_solar_gen', 'real_wind_gen', 'real_net_demand'],
        'lite_show_actual': False,
        'lite_warn_low':    250,
        'lite_warn_high':   750,
        'lite_smp_low':     10,
        'lite_warn_min_enabled': False,
        'lite_warn_min':    150,
        'lite_warn_max_enabled': False,
        'lite_warn_max':    900,
    }
    for k, v in ss_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── 날짜 / 표시항목 / 오버레이 / 오늘예측 — 한 줄 ──
    col_date, col_btn, col_overlay, col_today = st.columns([2, 1, 1.5, 1.5])
    st.markdown("---")
    with col_date:
        vis_date = st.date_input(
            "조회 날짜",
            st.session_state.get('lite_last_pred_date', datetime.now().date()),
            key="lite_vis_date",
            label_visibility="collapsed"
        )

    start_str = f"{vis_date} 00:00:00"
    end_str   = f"{vis_date} 23:00:00"
    df_res    = db.get_forecast(start_str, end_str)

    if df_res.empty or 'est_Solar_Utilization' not in df_res.columns \
            or df_res['est_Solar_Utilization'].isnull().all():
        with col_btn:
            st.button("⚙️ 표시 항목", width='stretch', key="lite_btn_plot_items_empty", disabled=True)
        with col_overlay:
            st.checkbox("📊 실측값 오버레이", key='lite_show_actual_empty', disabled=True)
        with col_today:
            if st.button("🔮 바로 예측", type="primary", width='stretch', key="lite_btn_today_nodata"):
                today = datetime.now().date()
                s_today = get_data_status(today)
                if not s_today['can_quick']:
                    st.session_state['_today_pred_error'] = (
                        "과거 데이터가 48시간 이상 부족합니다. "
                        "[🗂️ DB 수집 현황] 메뉴에서 수동 수집을 먼저 진행해 주세요."
                    )
                    st.rerun()
                else:
                    with st.spinner("① KPX 실측 수집 중..."):
                        try:
                            h_end_q   = (today - timedelta(days=1)).strftime("%Y-%m-%d")
                            h_start_q = (today - timedelta(days=3)).strftime("%Y-%m-%d")
                            if s_today['past_gap'] > 0 or s_today['past_missing'] > 0:
                                daily_historical_kpx(h_start_q, h_end_q)
                        except Exception:
                            pass
                    with st.spinner("② KMA 실측 수집 중..."):
                        try:
                            if s_today['past_gap'] > 0 or s_today['past_missing'] > 0:
                                daily_historical_kma(h_start_q, h_end_q)
                                daily_historical_kpx_smp(h_start_q, h_end_q)
                        except Exception:
                            pass
                    with st.spinner("③ KPX 예보 수집 중..."):
                        try:
                            tgt = today.strftime("%Y-%m-%d")
                            daily_forecast_kpx(tgt, tgt)
                        except Exception:
                            pass
                    with st.spinner("④ KMA 예보 수집 중 (다소 오래 걸릴 수 있습니다)..."):
                        try:
                            daily_forecast_kma(tgt, tgt)
                        except Exception:
                            pass
                    with st.spinner("⑤ 모델 예측 실행 중..."):
                        ok, msg, _ = run_model_prediction(
                            today.strftime('%Y-%m-%d'), db, assets
                        )
                    if ok:
                        st.session_state['lite_last_pred_date'] = today
                        st.session_state['_today_pred_success'] = msg
                    else:
                        st.session_state['_today_pred_error'] = f"예측 실패: {msg}"
                    st.rerun()

        # ── 바로 예측 결과 메시지 (rerun 후 토스트로 표시) ──
        if '_today_pred_success' in st.session_state:
            st.toast(st.session_state.pop('_today_pred_success'), icon="✅")
        if '_today_pred_error' in st.session_state:
            st.toast(st.session_state.pop('_today_pred_error'), icon="❌")

        st.warning("예측 데이터가 없습니다. [🔮 바로 예측] 버튼을 눌러 예측을 실행하거나, [🚀 예측 실행] 메뉴를 이용해 주세요.")
    else:
        df = df_res.copy()
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)

        # 파생 컬럼 계산 (app.py Option C와 동일)
        df['est_solar_gen'] = df['est_Solar_Utilization'] * df['Solar_Capacity_Est']
        df['est_wind_gen'] = df['est_Wind_Utilization'] * df['Wind_Capacity_Est']
        df['est_renew_total'] = df['est_solar_gen'] + df['est_wind_gen']
        if 'est_demand' in df.columns:
            df['est_net_demand'] = df['est_demand'] - df['est_renew_total']

        smp_col = 'smp_jeju' if 'smp_jeju' in df.columns else (
            'est_smp_jeju' if 'est_smp_jeju' in df.columns else None
        )

        # ── 실측 데이터 병합 (app.py Option C와 동일) ──
        hist_df = db.get_historical(start_str, f"{vis_date} 23:59:59")
        has_actual = False

        if not hist_df.empty:
            if not pd.api.types.is_datetime64_any_dtype(hist_df.index):
                hist_df.index = pd.to_datetime(hist_df.index)
            actual_cols = ['real_solar_gen', 'real_wind_gen', 'real_demand', 'real_renew_gen']
            for col in actual_cols:
                if col in hist_df.columns:
                    df[col] = hist_df[col].reindex(df.index)
            if 'real_solar_gen' in df.columns and df['real_solar_gen'].notna().any():
                has_actual = True
                df['real_renew_total'] = df.get('real_solar_gen', 0) + df.get('real_wind_gen', 0)
                if 'real_demand' in df.columns:
                    df['real_net_demand'] = df['real_demand'] - df['real_renew_total']

        # ── 표시 항목 설정 dialog ──
        plot_options = {
            'est_demand':      '총 전력수요 예측 (est_demand)',
            'est_net_demand':  '순부하 예측 (est_net_demand)',
            'est_solar_gen':   '태양광 발전량 예측 (est_solar_gen)',
            'est_wind_gen':    '풍력 발전량 예측 (est_wind_gen)',
            'est_renew_total': '총 재생에너지 발전량 (est_renew_total)',
        }
        if smp_col:
            plot_options[smp_col] = f'제주 SMP 가격 ({smp_col})'

        actual_label_map = {
            'real_demand':      '수요 실측',
            'real_solar_gen':   '태양광 실측',
            'real_wind_gen':    '풍력 실측',
            'real_renew_total': '재생E 실측 합계',
            'real_net_demand':  '순부하 실측',
        }
        available_actual = {
            col: label for col, label in actual_label_map.items()
            if has_actual and col in df.columns and df[col].notna().any()
        }

        @st.dialog("📊 표시 항목 설정")
        def select_plot_items():
            col1, col2 = st.columns(2)
            with col1:
                st.write("**예측 데이터**")
                current_est = st.session_state.get('lite_vis_vars', [])
                est_selections = {}
                for key, label in plot_options.items():
                    est_selections[key] = st.checkbox(label, value=(key in current_est), key=f"lite_dlg_est_{key}")
            with col2:
                st.write("**실측 항목 선택**")
                if available_actual:
                    current_act = st.session_state.get('lite_vis_actual', [])
                    act_selections = {}
                    for key, label in available_actual.items():
                        act_selections[key] = st.checkbox(label, value=(key in current_act), key=f"lite_dlg_act_{key}")
                else:
                    st.caption("실측 데이터 없음")
                    act_selections = {}
            st.markdown("---")
            if st.button("적용", type="primary", width='stretch'):
                st.session_state['lite_vis_vars'] = [k for k, v in est_selections.items() if v]
                if available_actual:
                    st.session_state['lite_vis_actual'] = [k for k, v in act_selections.items() if v]
                st.rerun()

        # ── 표시항목 버튼 (col_btn) ──
        with col_btn:
            if st.button("⚙️ 표시 항목", width='stretch', key="lite_btn_plot_items"):
                select_plot_items()

        # ── 오버레이 체크박스 (col_overlay) ──
        with col_overlay:
            if has_actual and available_actual:
                st.checkbox("📊 실측값 오버레이", key='lite_show_actual')
            else:
                st.caption("실측 데이터 없음")

        # ── 바로 예측 버튼 (col_today) ──
        with col_today:
            if st.button("🔮 바로 예측", type="primary", width='stretch', key="lite_btn_today"):
                today = datetime.now().date()
                s_today = get_data_status(today)

                # 48h 이상 부족하면 수동 수집 안내
                if not s_today['can_quick']:
                    st.session_state['_today_pred_error'] = (
                        "과거 데이터가 48시간 이상 부족합니다. "
                        "[] 메뉴에서 수동 수집을 먼저 진행해 주세요."
                    )
                    st.rerun()
                else:
                    with st.spinner("① KPX 실측 수집 중..."):
                        try:
                            h_end_q   = (today - timedelta(days=1)).strftime("%Y-%m-%d")
                            h_start_q = (today - timedelta(days=3)).strftime("%Y-%m-%d")
                            if s_today['past_gap'] > 0 or s_today['past_missing'] > 0:
                                daily_historical_kpx(h_start_q, h_end_q)
                        except Exception:
                            pass

                    with st.spinner("② KMA 실측 수집 중..."):
                        try:
                            if s_today['past_gap'] > 0 or s_today['past_missing'] > 0:
                                daily_historical_kma(h_start_q, h_end_q)
                                daily_historical_kpx_smp(h_start_q, h_end_q)
                        except Exception:
                            pass

                    with st.spinner("③ KPX 예보 수집 중..."):
                        try:
                            tgt = today.strftime("%Y-%m-%d")
                            daily_forecast_kpx(tgt, tgt)
                        except Exception:
                            pass

                    with st.spinner("④ KMA 예보 수집 중 (다소 오래 걸릴 수 있습니다)..."):
                        try:
                            daily_forecast_kma(tgt, tgt)
                        except Exception:
                            pass

                    with st.spinner("⑤ 모델 예측 실행 중..."):
                        ok, msg, _ = run_model_prediction(
                            today.strftime('%Y-%m-%d'), db, assets
                        )

                    if ok:
                        st.session_state['lite_last_pred_date'] = today
                        st.session_state['_today_pred_success'] = msg
                    else:
                        st.session_state['_today_pred_error'] = f"예측 실패: {msg}"
                    st.rerun()

        # ── 바로 예측 결과 메시지 (rerun 후 표시) ──
        if '_today_pred_success' in st.session_state:
            st.success(st.session_state.pop('_today_pred_success'))
        if '_today_pred_error' in st.session_state:
            st.error(st.session_state.pop('_today_pred_error'))

        selected_vars   = st.session_state['lite_vis_vars']
        selected_actual = st.session_state['lite_vis_actual']
        show_actual     = st.session_state.get('lite_show_actual', False)
        warn_low  = st.session_state['lite_warn_low']
        warn_high = st.session_state['lite_warn_high']
        smp_threshold = st.session_state['lite_smp_low'] if smp_col else 0

        if not selected_vars:
            st.info("👉 [⚙️ 표시 항목] 버튼을 눌러 시각화할 데이터를 선택하세요.")
        else:
            overlay_active = show_actual and has_actual and len(selected_actual) > 0

            # ── 메인 차트 ──
            fig = go.Figure()
            colors = {
                'est_demand':      COLORS['demand'],
                'est_net_demand':  COLORS['net_demand'],
                'est_solar_gen':   COLORS['solar_est'],
                'est_wind_gen':    COLORS['wind_est'],
                'est_renew_total': COLORS['renew_total'],
            }
            if smp_col:
                colors[smp_col] = COLORS['smp']

            actual_map = {
                'est_solar_gen':   'real_solar_gen',
                'est_wind_gen':    'real_wind_gen',
                'est_renew_total': 'real_renew_total',
                'est_net_demand':  'real_net_demand',
                'est_demand':      'real_demand',
            }

 
            for var in selected_vars:
                if var not in df.columns:
                    continue
                is_net = (var == 'est_net_demand')

                # 예측: 오버레이 꺼짐→solid, 켜짐→dash
                line_style = dict(
                    color=colors.get(var, 'black'),
                    width=4 if is_net else 2,
                    dash='dash' if overlay_active else 'solid'
                )
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[var],
                    name=plot_options.get(var, var).split('(')[0].strip(),
                    line=line_style, hovertemplate='%{y:,.1f}'
                ))

                # 실측: 전부 solid (오버레이 켜졌을 때만 표시)
                if overlay_active:
                    actual_col = actual_map.get(var)
                    if actual_col and actual_col in selected_actual and actual_col in df.columns:
                        is_actual_net = (actual_col == 'real_net_demand')
                        actual_line = dict(
                            color=colors.get(var, 'gray'),
                            width=3 if is_actual_net else 2,
                            dash='solid'
                        )
                        fig.add_trace(go.Scatter(
                            x=df.index, y=df[actual_col],
                            name=actual_label_map.get(actual_col, actual_col),
                            line=actual_line, opacity=0.7,
                            hovertemplate='%{y:,.1f}'
                        ))

            # 위험 구간 표시
            if 'est_net_demand' in df.columns:
                draw_danger_zones(fig, df, df['est_net_demand'] < warn_low,
                                  'red', annotation_text='LNG 저발전 구간', show_legend_label='저발전 구간')
                draw_danger_zones(fig, df, df['est_net_demand'] > warn_high,
                                  'blue', annotation_text='LNG 고발전 구간', show_legend_label='고발전 구간')

            if st.session_state.get('lite_warn_min_enabled') and 'est_renew_total' in df.columns:
                draw_danger_zones(fig, df, df['est_renew_total'] < st.session_state['lite_warn_min'],
                                  'brown', show_legend_label='최저발전 경고')
            if st.session_state.get('lite_warn_max_enabled') and 'est_renew_total' in df.columns:
                draw_danger_zones(fig, df, df['est_renew_total'] > st.session_state['lite_warn_max'],
                                  'purple', show_legend_label='최대발전 경고')

            # 현재시각 세로선
            now = datetime.now()
            day_start = datetime.combine(vis_date, datetime.min.time())
            day_end   = datetime.combine(vis_date, datetime.max.time())
            if day_start <= now <= day_end:
                now_str = now.strftime('%Y-%m-%d %H:%M:%S')
                fig.add_vline(x=now_str, line_width=1, line_dash="solid", line_color="tomato")
                fig.add_annotation(
                    x=now_str, y=-0.038, yref="paper",
                    text="현재", showarrow=False,
                    font=dict(size=10, color="tomato"),
                )

            fig.update_layout(
                title=f"{vis_date} 예측 결과",
                hovermode="x unified",
                yaxis_title="MW",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                dragmode='zoom',
                yaxis=dict(fixedrange=True)
            )
            st.plotly_chart(fig, width="stretch")
            st.caption("매일 자정 이후 [🔮 바로 예측] 버튼을 눌러 예보 업데이트가 가능합니다. **LNG발전은 순부하에 따라 제어되나 기력발전기 or 연계선의 정비로 실제보다 발전량이 증가할 수 있습니다.**")
            # ── AI 예측 브리핑 섹션 (lite.py 내부) ──
            with st.expander("✨ AI 예측 브리핑", expanded=True):
                date_key = str(vis_date)
                
                # 1. 파일에서 모든 브리핑 데이터를 로드 (세션에 없다면)
                if 'lite_briefings_storage' not in st.session_state:
                    st.session_state['lite_briefings_storage'] = load_briefings_from_file()

                # 2. 현재 선택된 날짜에 해당하는 브리핑이 있는지 확인
                saved_briefing = st.session_state['lite_briefings_storage'].get(date_key)

                # 3. 브리핑 생성 버튼
                if st.button("AI 브리핑 생성 / 갱신", key="lite_btn_ai_briefing"):
                    with st.spinner("AI가 데이터를 분석하고 있습니다..."):
                        briefing_text = generate_energy_narrative(
                            df=df, 
                            warn_low=warn_low, 
                            warn_high=warn_high,
                            smp_threshold=smp_threshold
                        )
                        
                        # 로컬 파일에 저장
                        save_briefing_to_file(date_key, briefing_text)
                        
                        # 세션 상태 업데이트 (화면 즉시 반영용)
                        st.session_state['lite_briefings_storage'][date_key] = briefing_text
                        st.rerun()
                
                # 4. 브리핑 내용 표시
                if saved_briefing:
                    st.markdown(saved_briefing)
                else:
                    st.caption("위 버튼을 눌러 해당 날짜의 브리핑을 생성하세요. 생성된 내용은 로컬에 자동 저장됩니다.")

            # ── 경고 설정 expander ──
            with st.expander("⚠️ 경고 임계값 설정", expanded=False):
                ec1, ec2 = st.columns(2)
                with ec1:
                    st.number_input("저발전 임계값 (MW)", value=warn_low, step=10, key='lite_warn_low')
                    st.number_input("고발전 임계값 (MW)", value=warn_high, step=10, key='lite_warn_high')
                with ec2:
                    st.checkbox("🟣 최저발전 경고", key='lite_warn_min_enabled')
                    if st.session_state['lite_warn_min_enabled']:
                        st.number_input("최저발전 임계값 (MW)", value=st.session_state['lite_warn_min'],
                                        step=10, key='lite_warn_min_input')
                        st.session_state['lite_warn_min'] = st.session_state.get('lite_warn_min_input',
                                                                                  st.session_state['lite_warn_min'])
                    st.checkbox("🟤 최대발전 경고", key='lite_warn_max_enabled')
                    if st.session_state['lite_warn_max_enabled']:
                        st.number_input("최대발전 임계값 (MW)", value=st.session_state['lite_warn_max'],
                                        step=10, key='lite_warn_max_input')
                        st.session_state['lite_warn_max'] = st.session_state.get('lite_warn_max_input',
                                                                                  st.session_state['lite_warn_max'])

            # ── 데이터 테이블 ──
            with st.expander("📋 데이터 테이블", expanded=False):
                display_cols = [c for c in selected_vars if c in df.columns]
                if overlay_active and selected_actual:
                    display_cols += [c for c in selected_actual if c in df.columns and c not in display_cols]

                def highlight_warnings(row):
                    styles = [''] * len(row)
                    if 'est_net_demand' in row.index:
                        nd = row['est_net_demand']
                        if pd.notna(nd):
                            idx = row.index.get_loc('est_net_demand')
                            if nd < warn_low:
                                styles[idx] = 'background-color: #cce5ff'
                            elif nd > warn_high:
                                styles[idx] = 'background-color: #f8d7da'
                    if smp_col and smp_col in row.index:
                        if pd.notna(row[smp_col]) and row[smp_col] < smp_threshold:
                            idx = row.index.get_loc(smp_col)
                            styles[idx] = 'background-color: #ffe5b4'
                    return styles

                st.dataframe(
                    df[display_cols].style.apply(highlight_warnings, axis=1).format(precision=2),
                    width="stretch"
                )


# ══════════════════════════════════════════
# 🚀 예측 실행
# (app.py Option B 코드 기반, 경량화)
# ══════════════════════════════════════════
elif lite_menu == "🚀 예측 실행":
    col_d, col_info = st.columns([1, 2])
    with col_d:
        default_date = st.session_state.get('lite_last_pred_date', datetime.now().date())
        target_date  = st.date_input("예측 대상일", default_date, key="lite_pred_date")
    with col_info:
        st.caption(" ")
        day_offset = (target_date - datetime.now().date()).days
        if day_offset <= 1:
            st.caption(f"과거 336h + 미래 24h 데이터 필요 (대상일: {target_date})")
        else:
            st.caption(f"과거 336h + 미래 24h 데이터 필요 (대상일: {target_date}, D+{day_offset} 롤링)")

    s = get_data_status(target_date)
    render_metrics(s)
    st.markdown("---")

    # ── 예측 실행 버튼 ──
    if st.button("🚀 예측 실행", type="primary", width='stretch', key="lite_btn_predict"):
        st.session_state.pop('lite_pred_ok', None)
        st.session_state.pop('lite_pred_msg', None)

        if not (s['past_ok'] and s['fut_ok']):
            st.error("데이터가 부족합니다. 아래 '빠른 수집'을 먼저 실행해 주세요.")
        else:
            with st.spinner(f"{target_date} 예측 중..."):
                ok, msg, _ = run_model_prediction(
                    target_date.strftime('%Y-%m-%d'), db, assets
                )
                st.session_state['lite_last_pred_date'] = target_date
                st.session_state['lite_pred_ok']  = ok
                st.session_state['lite_pred_msg'] = msg
            st.rerun()

    # ── 예측 결과 메시지 ──
    if 'lite_pred_ok' in st.session_state:
        if st.session_state['lite_pred_ok']:
            st.success(st.session_state['lite_pred_msg'])
            with st.expander("📊 예측 가동률 차트 미리보기", expanded=False):
                res_df = db.get_forecast(f"{target_date} 00:00:00", f"{target_date} 23:00:00")
                if not res_df.empty:
                    fig = px.line(
                        res_df,
                        x=res_df.index,
                        y=['est_Solar_Utilization', 'est_Wind_Utilization'],
                        title="예측 가동률",
                        labels={"value": "가동률 (0~1)", "timestamp": "시간", "variable": "발전원"},
                        color_discrete_map={
                            "est_Solar_Utilization": "orange",
                            "est_Wind_Utilization": "skyblue"
                        }
                    )
                    fig.update_layout(hovermode="x unified")
                    fig.update_traces(hovertemplate='%{y:,.3f}')
                    st.plotly_chart(fig, width="stretch")
        else:
            st.error(f"예측 실패: {st.session_state['lite_pred_msg']}")

    st.markdown("---")

    # ── 빠른 수집 ──
    st.caption("**📡 빠른 수집** — 선택한 날짜 기준 부족 데이터를 자동 수집합니다.")

    parts = []
    if s['past_gap']     > 0: parts.append(f"실측 {s['past_gap']}h 부족")
    if s['past_missing'] > 0: parts.append(f"실측 결측 {s['past_missing']}h")
    if s['fut_gap']      > 0: parts.append(f"예보 {s['fut_gap']}h 부족")
    if s['fut_missing']  > 0: parts.append(f"예보 결측 {s['fut_missing']}h")

    if parts:
        st.warning(" / ".join(parts))
    else:
        st.success("과거 실측 · 미래 예보 모두 정상입니다.")

    if s['can_quick']:
        if st.button("📡 부족 데이터 빠른 수집", width='stretch', key="lite_quick_fetch"):
            with st.spinner("수집 중..."):
                try:
                    tgt       = target_date.strftime("%Y-%m-%d")
                    h_end_q   = (target_date - timedelta(days=1)).strftime("%Y-%m-%d")
                    h_start_q = (target_date - timedelta(days=3)).strftime("%Y-%m-%d")
                    if s['past_gap'] > 0 or s['past_missing'] > 0:
                        daily_historical_kpx(h_start_q, h_end_q)
                        daily_historical_kma(h_start_q, h_end_q)
                        daily_historical_kpx_smp(h_start_q, h_end_q)
                    if s['fut_gap'] > 0 or s['fut_missing'] > 0:
                        daily_forecast_kpx(tgt, tgt)
                        daily_forecast_kma(tgt, tgt)
                    st.success("수집 완료! 다시 예측 실행 버튼을 눌러주세요.")
                    st.rerun()
                except Exception as e:
                    st.error(f"수집 실패: {e}")
    else:
        st.info("부족한 데이터가 많습니다. [🗂️ DB 수집현황] 메뉴에서 수동 수집해 주세요.")


# ══════════════════════════════════════════
# 🗂️ DB 수집 현황
# (app.py Option A 코드 기반, 핵심만 추출)
# ══════════════════════════════════════════
elif lite_menu == "🗂️ DB 수집현황":
    col_d, col_r = st.columns([2, 1])
    with col_d:
        db_date = st.date_input("기준 예측일", datetime.now().date(), key="lite_db_date",
                                label_visibility="collapsed")
    with col_r:
        st.button("🔄 새로고침", width='stretch', key="lite_db_refresh")

    s = get_data_status(db_date)
    render_metrics(s)

    with st.expander("🔍 결측치 히트맵 — 과거 실측 (336h · 1칸 = 1일)", expanded=False):
        render_heatmap(s['past_df'], n_cells=14, cell_unit_hours=24,
                       axis_labels=["-14일", "-7일", "-1일"])
    with st.expander("🔍 결측치 히트맵 — 미래 예보 (24h · 1칸 = 1h)", expanded=False):
        render_heatmap(s['fut_df'], n_cells=24, cell_unit_hours=1,
                       axis_labels=["00시", "12시", "23시"])

    st.markdown("---")

    # ── API 수동 수집: 실측/예보 2열 ──
    st.caption("**API 수동 수집**")
    col_hist, col_fore = st.columns(2)

    with col_hist:
        with st.container(border=True):
            st.caption("**실측 (Historical)**")
            hc1, hc2 = st.columns(2)
            with hc1:
                h_start = st.date_input("시작일",
                                        datetime.now().date() - timedelta(days=7),
                                        key="lite_h_start")
            with hc2:
                h_end = st.date_input("종료일",
                                      datetime.now().date() - timedelta(days=1),
                                      key="lite_h_end")
            h_invalid = h_start > h_end or (h_end - h_start).days > 30
            if h_start > h_end:
                st.error("시작일이 종료일보다 늦을 수 없습니다.")
            elif (h_end - h_start).days > 30:
                st.error("최대 30일까지 수집 가능합니다.")
            if st.button("📡 KPX + KMA 수집", type="primary",
                         width='stretch', disabled=h_invalid, key="lite_btn_hist"):
                with st.spinner("실측 데이터 수집 중..."):
                    try:
                        daily_historical_kpx(h_start.strftime("%Y-%m-%d"), h_end.strftime("%Y-%m-%d"))
                        daily_historical_kma(h_start.strftime("%Y-%m-%d"), h_end.strftime("%Y-%m-%d"))
                        daily_historical_kpx_smp(h_start.strftime("%Y-%m-%d"), h_end.strftime("%Y-%m-%d"))
                        st.success("실측 수집 완료!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"수집 실패: {e}")

    with col_fore:
        with st.container(border=True):
            st.caption("**예보 (Forecast)**")
            fc1, fc2 = st.columns(2)
            with fc1:
                f_start = st.date_input("시작일",
                                        datetime.now().date(),
                                        key="lite_f_start")
            with fc2:
                f_end = st.date_input("종료일",
                                      datetime.now().date() + timedelta(days=2),
                                      key="lite_f_end")
            f_invalid = f_start > f_end or (f_end - f_start).days > 7
            if f_start > f_end:
                st.error("시작일이 종료일보다 늦을 수 없습니다.")
            elif (f_end - f_start).days > 7:
                st.error("예보는 최대 7일까지 수집 가능합니다.")
            if st.button("📡 KPX + KMA 수집", type="primary",
                         width='stretch', disabled=f_invalid, key="lite_btn_fore"):
                with st.spinner("예보 데이터 수집 중..."):
                    try:
                        daily_forecast_kpx(f_start.strftime("%Y-%m-%d"), f_end.strftime("%Y-%m-%d"))
                        daily_forecast_kma(f_start.strftime("%Y-%m-%d"), f_end.strftime("%Y-%m-%d"))
                        st.success("예보 수집 완료!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"수집 실패: {e}")