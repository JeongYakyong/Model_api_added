import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import pvlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import warnings
import os
from components.system_info import render_system_info

from utils.db_manager import JejuEnergyDB
from utils.data_pipeline import (
    add_capacity_features,
    daily_historical_update, daily_forecast_and_predict,
    daily_historical_kpx, daily_historical_kma, daily_historical_kpx_smp,
    run_model_prediction, prepare_model_input,
    daily_forecast_kpx, daily_forecast_kma
)
from models.architecture import PatchTST_Weather_Model
from utils.chart_helpers import (
    EDA_ONLY_COLUMNS, PREDICTION_OUTPUT_COLUMNS, COLORS,
    check_data_status, date_range_selector,
    merge_actual_and_forecast, plot_actual_vs_pred, draw_danger_zones
)

# ==========================================
# 공유 리소스 가져오기
# ==========================================
db = st.session_state['shared_db']
assets = st.session_state['shared_assets']

# ==========================================
# 사이드바 메뉴
# ==========================================
st.sidebar.title("✔️ Side Bar")

if '_navigate_to' in st.session_state:
    st.session_state['main_menu'] = st.session_state.pop('_navigate_to')

menu = st.sidebar.radio("메뉴 선택:", [
    "Option A : DB 관리",
    "Option B : 발전량 예측",
    "Option C : 예측 결과 시각화",
    "Option D : 예측 정확도 검증",
    "Option E : 데이터 분석 (EDA)",
    "Option F : 시스템 안내"
], key="main_menu")

# ==========================================
# Option A : DB 관리
# ==========================================
if menu == "Option A : DB 관리":
    st.subheader("🗂️ DB 관리 및 Data Status")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Data Status", "API 데이터 수집", "데이터 조회", "CSV 업로드"])
    
    # --- Tab 1: Data Status ---
    with tab1:
        header_col1, header_col2 = st.columns([8, 2])
        with header_col1:
            st.subheader("DB 상태 점검 (전체 기간)")
        with header_col2:
            if st.button("🔄 새로고침", help="최신 데이터베이스 정보를 다시 불러옵니다.", width="stretch", key="refresh_db_button1"):
                st.rerun()

        with st.spinner("전체 실측 데이터를 불러오고 무결성을 검사하는 중입니다..."):
            full_hist_df = db.get_historical()
        
        if not full_hist_df.empty:
            st.write("### 📊 데이터 저장 현황")
            
            if not pd.api.types.is_datetime64_any_dtype(full_hist_df.index):
                full_hist_df.index = pd.to_datetime(full_hist_df.index)
                
            min_date = full_hist_df.index.min()
            max_date = full_hist_df.index.max()
            
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            max_date_only = max_date.replace(hour=0, minute=0, second=0, microsecond=0)
            gap_days = (today - max_date_only).days
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("총 데이터 수", f"{len(full_hist_df):,} 행")
            col2.metric("시작 날짜", min_date.strftime('%Y-%m-%d'))
            col3.metric("최근 날짜", max_date.strftime('%Y-%m-%d'))
            today_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            check_df = full_hist_df[full_hist_df.index < today_midnight]
            status_info = check_data_status(check_df)

            if gap_days > 0:
                col4.metric(
                    label="업데이트 필요 (오늘 기준)", 
                    value=f"{gap_days}일 분량", 
                    delta=f"{gap_days}일 지연됨", 
                    delta_color="inverse"
                )
            elif gap_days == 0 and status_info['status'] == "Good":
                col4.metric(
                    label="업데이트 필요 (오늘 기준)", 
                    value="최신 상태", 
                    delta="결측 없음", 
                    delta_color="normal"
                )
            elif gap_days == 0:
                col4.metric(
                    label="업데이트 필요 (오늘 기준)", 
                    value="최신 상태", 
                    delta=f"결측 {status_info['incomplete_rows']}건", 
                    delta_color="inverse"
                )
            else:
                col4.metric("업데이트 필요", "미래 데이터 포함", f"+{abs(gap_days)}일")
                        
            # 불완전 행 요약 (주요 컬럼 기준)
            if status_info['incomplete_rows'] > 0:
                st.warning(f"⚠️ timestamp는 존재하지만 주요 컬럼 값이 비어있는 행: **{status_info['incomplete_rows']}건**")
            
            if status_info['missing_timestamps'] > 0:
                st.warning(f"⚠️ 시계열 누락 (timestamp 자체가 빠진 시간대): **{status_info['missing_timestamps']}건**")
            
            if status_info['status'] == "Good":
                st.success("✅ 모든 주요 컬럼의 데이터가 빈틈없이 채워져 있습니다!")
            
            with st.expander("🔍 전체 컬럼별 결측치 상세 확인", expanded=False):
                st.write("각 컬럼별로 비어있는(Null, NaN) 데이터의 개수를 보여줍니다.")
                
                status_check_df = check_df.drop(columns=EDA_ONLY_COLUMNS, errors='ignore')
                missing_info = status_check_df.isnull().sum().reset_index()
                
                missing_info.columns = ["컬럼명", "결측치 개수"]
                
                st.dataframe(missing_info, width="stretch", hide_index=True)
                total_missing = missing_info["결측치 개수"].sum()

                if total_missing > 0:
                    if st.button("✨ 결측치 자동 보간 (최대 3건 제한) 및 DB 적용", help="최대 2개의 연속된 껴있는 결측치까지만 시간 비례로 채웁니다."):
                        with st.spinner("결측치를 보간하고 DB에 저장하는 중입니다..."):
                            try:
                                interpolated_df = check_df.interpolate(method='time', limit=2, limit_direction='both', limit_area='inside')
                                interpolated_df.index = interpolated_df.index.strftime('%Y-%m-%d %H:%M:%S')
                                db.save_historical(interpolated_df)
                                st.success("🎉 결측치 보간 및 DB 업데이트가 완료되었습니다!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"보간 처리 중 오류가 발생했습니다: {e}")
                                    
            with st.expander("👀 결측치가 발생한 시간대 직접 확인하기", expanded=False):
                st.info("어느 시간대(timestamp)의 데이터가 비어있는지 확인해 보세요.")
                display_df = check_df.drop(columns=EDA_ONLY_COLUMNS, errors='ignore')
                missing_rows = display_df[display_df.isna().any(axis=1)]
                st.dataframe(missing_rows, width="stretch")
        else:
            st.error("데이터베이스가 비어있습니다. [API 데이터 수집] 탭에서 데이터를 수집해 주세요.")
            
    # --- Tab 2: API 데이터 수집 ---
    with tab2:
        st.subheader("API를 통한 데이터 수집")
        st.info("Data Status에서 확인한 결측 구간을 지정하여 데이터를 채워 넣으세요.")
        today = datetime.now().date()
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 📈 실측 데이터 (Historical)")
            st.caption("1회 최대 수집가능 기간은 30일입니다.")
            hist_start = st.date_input("시작일", today - timedelta(days=7), key='h_start')
            hist_end = st.date_input("종료일", today, key='h_end')
            
            invalid_date = False
            if hist_start > hist_end:
                st.error("시작일이 종료일보다 늦을 수 없습니다.")
                invalid_date = True
            elif (hist_end - hist_start).days > 30:
                st.error("실측 데이터 조회는 최대 30일까지만 가능합니다.")
                invalid_date = True
            elif hist_end > today or hist_start > today:
                st.warning("⚠️ 현재시간 이후의 데이터는 정상적으로 수집되지 않을 수 있습니다.")
            
            if st.button("실측 데이터 수집", width="stretch", disabled=invalid_date):
                with st.spinner("모든 실측 데이터를 수집하고 있습니다..."):
                    try:
                        daily_historical_kpx(hist_start.strftime("%Y-%m-%d"), hist_end.strftime("%Y-%m-%d"))
                        daily_historical_kma(hist_start.strftime("%Y-%m-%d"), hist_end.strftime("%Y-%m-%d"))
                        daily_historical_kpx_smp(hist_start.strftime("%Y-%m-%d"), hist_end.strftime("%Y-%m-%d"))
                        st.success("실측 데이터 수집 완료!")
                    except Exception as e:
                        st.error(f"API 호출 실패: {e}")
            
            with st.expander("🛠️ 개별 API 수집"):
                st.caption("필요한 특정 데이터만 개별적으로 수집할 수 있습니다.")
                
                h_start_str = hist_start.strftime("%Y-%m-%d")
                h_end_str = hist_end.strftime("%Y-%m-%d")
                
                if st.button("KPX 발전량 수집", key="btn_kpx_hist", disabled=invalid_date):
                    with st.spinner("KPX 발전량 데이터를 수집 중입니다..."):
                        try:
                            daily_historical_kpx(h_start_str, h_end_str) 
                            st.success("KPX 발전량 데이터 수집 완료!")
                        except Exception as e:
                            st.error(f"KPX 발전량 API 호출 실패: {e}")
                        
                if st.button("KPX SMP 수집", key="btn_kpx_smp", disabled=invalid_date):
                    with st.spinner("KPX SMP 데이터를 수집 중입니다..."):
                        try:
                            daily_historical_kpx_smp(h_start_str, h_end_str)
                            st.success("KPX SMP 데이터 수집 완료!")
                        except Exception as e:
                            st.error(f"KPX SMP API 호출 실패: {e}")
                        
                if st.button("KMA 기상 데이터 수집", key="btn_kma_hist", disabled=invalid_date):
                    with st.spinner("KMA 기상 데이터를 수집 중입니다..."):
                        try:
                            daily_historical_kma(h_start_str, h_end_str)
                            st.success("KMA 기상 데이터 수집 완료!")
                        except Exception as e:
                            st.error(f"KMA 기상 API 호출 실패: {e}")
                        
        with col2:
            st.write("### 🌤️ Forecast 데이터 (예보)")
            st.caption("주의: 과거 90일 전 ~ 미래 3일 후 (최대 30일 간격)")
            
            fore_start = st.date_input("시작일", today - timedelta(days=1), key='f_start')
            fore_end = st.date_input("종료일", today + timedelta(days=1), key='f_end')
            invalid_fore_date = False
            
            if fore_start > fore_end:
                st.error("시작일이 종료일보다 늦을 수 없습니다.")
                invalid_fore_date = True
            elif fore_start < today - timedelta(days=90):
                limit_past = (today - timedelta(days=90)).strftime("%Y-%m-%d")
                st.error(f"예보는 과거 90일 전({limit_past})까지만 조회 가능합니다.")
                invalid_fore_date = True
            elif fore_end > today + timedelta(days=3):
                limit_future = (today + timedelta(days=3)).strftime("%Y-%m-%d")
                st.error(f"예보는 3일 후({limit_future})까지만 조회 가능합니다.")
                invalid_fore_date = True
            elif (fore_end - fore_start).days > 30:
                st.error("예보 데이터 조회는 최대 30일까지만 가능합니다.")
                invalid_fore_date = True
                
            if st.button("Forecast 데이터 수집", width="stretch", disabled=invalid_fore_date):
                with st.spinner("모든 예보 데이터를 수집하고 있습니다..."):
                    try:
                        daily_forecast_and_predict(fore_start.strftime("%Y-%m-%d"), fore_end.strftime("%Y-%m-%d"))
                        st.success("Forecast 데이터 수집 완료!")
                    except Exception as e:
                        st.error(f"Forecast API 호출 실패: {e}")
            
            with st.expander("🛠️ 개별 API 수집"):
                st.caption("필요한 특정 예보 데이터만 개별적으로 수집할 수 있습니다.")
                
                f_start_str = fore_start.strftime("%Y-%m-%d")
                f_end_str = fore_end.strftime("%Y-%m-%d")
                
                if st.button("KPX 발전량 Forecast 수집", key="btn_kpx_fore_ind", disabled=invalid_fore_date):
                    with st.spinner("KPX Forecast 데이터를 수집 중입니다..."):
                        try:
                            daily_forecast_kpx(f_start_str, f_end_str)
                            st.success("KPX Forecast 수집 완료!")
                        except Exception as e:
                            st.error(f"KPX Forecast API 호출 실패: {e}")
                        
                if st.button("KMA 기상 Forecast 수집", key="btn_kma_fore_ind", disabled=invalid_fore_date):
                    with st.spinner("KMA 기상 Forecast 데이터를 수집 중입니다..."):
                        try:
                            daily_forecast_kma(f_start_str, f_end_str)
                            st.success("KMA Forecast 수집 완료!")
                        except Exception as e:
                            st.error(f"KMA Forecast API 호출 실패: {e}")
        st.markdown("---")
        st.info("💡 호출 시점 기준 최신 예보를 자동 반영합니다.")
        
        with st.expander("📌 Forecast 수집 상세 안내"):
            st.markdown(
                "**KPX 예보**\n"
                "- 전날 23시경 업로드. 내일(+1일)까지만 조회 가능\n\n"
                "**KMA 기상 예보**\n"
                "- 6시간 주기 갱신, 최대 3일 후(+3일)까지 조회 가능\n"
                "- 2일 이후 예보의 신뢰도는 보장되지 않음\n\n"
                "**과거 날짜 소급**\n"
                "- 조회 속도가 느리고, 오래된 날짜일수록 결측 가능성 있음\n"
                "- 기간을 나누어 수집 권장\n\n"
            )
            
    # --- Tab 3: 데이터 조회 ---
    with tab3:
        st.subheader("데이터 조회")
        st.caption("LNG, HVDC, 기력 발전량은 실시간 업데이트가 불가능합니다. 전력거래소 CSV 별도 다운로드 필요.")
        
        ctrl_col1, ctrl_col2 = st.columns([5, 1])
        with ctrl_col1:
            table_choice = st.radio("조회할 테이블:", ["실측 데이터 (Historical)", "Forecast 데이터"], horizontal=True, label_visibility="collapsed")
        with ctrl_col2:
            if st.button("🔄 새로고침", width="stretch", key="refresh_tab3"):
                st.rerun()
        
        future_days = 2 if table_choice == "Forecast 데이터" else 1
        start_date, end_date = date_range_selector("db_table", allow_future_days=future_days, default_option="1주")
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        if table_choice == "실측 데이터 (Historical)":
            df = db.get_historical(start_str, end_str)
        else:
            df = db.get_forecast(start_str, end_str)
        
        if not df.empty:
            st.caption(f"📅 {start_str} ~ {end_str}  |  총 {len(df):,}행")
            st.dataframe(df.style.format(precision=2), width="stretch")
        else:
            st.warning(f"조회하신 기간({start_str} ~ {end_str})에 해당하는 데이터가 없습니다.")

    # --- Tab 4: CSV 업로드 ---
    with tab4:
        st.subheader("과거 CSV 파일 일괄 적재 (백업/복구용)")
        st.info("💡 초기 셋팅을 하거나 DB가 손실되었을 때, 과거 CSV 파일을 올려서 한 번에 복구할 수 있습니다.")
        
        uploaded_file = st.file_uploader("과거 데이터 CSV 파일을 올려주세요", type=['csv'])
        
        if uploaded_file is not None:
            preview_df = pd.read_csv(uploaded_file)
            st.write("### 📊 업로드된 파일 결측치 분석")
            missing_preview = preview_df.isnull().sum().reset_index()
            missing_preview.columns = ["컬럼명", "결측치 개수"]
            
            if missing_preview['결측치 개수'].sum() > 0:
                st.warning("⚠️ 업로드된 파일에 결측치가 존재합니다. 아래 표를 확인하세요.")
            else:
                st.success("✅ 결측치가 없는 깔끔한 데이터입니다!")
                
            st.dataframe(missing_preview, width="stretch", hide_index=True)
            
            if st.button("DB에 적재하기", type="primary"):
                with st.spinner("데이터를 분석하고 DB에 기록하는 중입니다..."):
                    try:
                        df = preview_df.copy()
                        
                        if 'timestamp' not in df.columns:
                            st.error("CSV 파일에 'timestamp' 컬럼이 없습니다. 파일 형식을 확인해주세요!")
                        else:
                            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                            df = df.set_index('timestamp')
                            
                            df = add_capacity_features(df)
                            saved_rows = db.save_historical(df)
                            status_info = check_data_status(df)
                            
                            st.success(f"🎉 '{uploaded_file.name}' 파일 업로드 및 DB 적재 완료! (총 {saved_rows:,}행)")
                            st.info("추가/갱신된 파생 변수: Solar_Capacity_Est, Wind_Capacity_Est, Solar_Utilization, Wind_Utilization")
                            
                            if status_info["status"] == "Good":
                                st.success("✅ 업로드된 데이터의 시계열이 촘촘하고 결측치가 없습니다.")
                            else:
                                st.warning(f"⚠️ 업로드된 데이터에 {status_info['missing_timestamps']}개의 누락된 시간이 있거나 결측치가 포함되어 있습니다.")
                                
                    except Exception as e:
                        st.error(f"데이터 적재 중 오류가 발생했습니다: {e}")

    pass
                        
# ==========================================
# Option B : 발전량 예측
# ==========================================
elif menu == "Option B : 발전량 예측":
    st.subheader("🔮 발전량 예측 및 DB 저장")
    st.write("선택한 날짜의 예보 데이터를 바탕으로 태양광 및 풍력 발전 가동률을 예측하고 저장합니다.")
    
    default_pred_date = st.session_state.get('last_predicted_date', datetime.now().date())
    target_date = st.date_input("예측 대상 날짜", default_pred_date)
    
    # ==========================================
    # 날짜 선택 즉시: 데이터 상태 사전 점검
    # ==========================================
    st.markdown("---")
    st.write("### 🔍 입력 데이터 상태 점검")
    
    # D+N 판별
    today = datetime.now().date()
    day_offset = (target_date - today).days  # 0=오늘, 1=내일, 2=모레...
    
    if day_offset <= 1:
        st.caption(f"모델 추론에 필요한 데이터: 과거 실측 336시간 + 미래 예보 24시간 (대상일: {target_date})")
    else:
        st.caption(f"모델 추론에 필요한 데이터: 과거 실측 336시간 + 미래 예보 24시간 (대상일: {target_date}, **D+{day_offset} 롤링 예측**)")
    
    # 과거 실측 데이터 범위
    past_end = f"{target_date - timedelta(days=1)} 23:00:00"
    past_start = f"{target_date - timedelta(days=14)} 00:00:00"
    past_df = db.get_historical(past_start, past_end)
    
    # 미래 예보 데이터 범위
    future_start = f"{target_date} 00:00:00"
    future_end = f"{target_date} 23:00:00"
    future_df = db.get_forecast(future_start, future_end)
    
    past_hours = len(past_df) if not past_df.empty else 0
    future_hours = len(future_df) if not future_df.empty else 0
    
    EXCLUDE_FROM_CHECK = EDA_ONLY_COLUMNS | PREDICTION_OUTPUT_COLUMNS

    # past 구간에 forecast 보충이 필요한지 판단
    # D+0/D+1이라도 오늘 아직 안 지난 시간이 past에 포함되면 보충 필요
    needs_forecast_supplement = past_hours < 336
    
    if not needs_forecast_supplement:
        # ── historical만으로 336시간 충족 ──
        past_missing = (
            int(past_df.drop(columns=EXCLUDE_FROM_CHECK, errors='ignore').isna().any(axis=1).sum())
            if not past_df.empty else 0
        )
        future_missing = (
            int(future_df.drop(columns=EXCLUDE_FROM_CHECK, errors='ignore').isna().any(axis=1).sum())
            if not future_df.empty else 0
        )
        
        past_ok = past_hours >= 336 and past_missing == 0
        past_label = f"{past_hours} / 336시간"
        future_ok = future_hours >= 24 and future_missing == 0
        
    else:
        # ── forecast 보충 필요 (D+1 야간 또는 D+2/D+3) ──
        past_fore_df = db.get_forecast(past_start, past_end)
        past_hist_hours = set(past_df.index) if not past_df.empty else set()
        past_fore_hours = set(past_fore_df.index) if not past_fore_df.empty else set()
        past_combined_hours = len(past_hist_hours | past_fore_hours)
        
        # forecast 보충 구간에 est_Utilization이 있는지 확인
        forecast_only_hours = past_fore_hours - past_hist_hours
        if forecast_only_hours and not past_fore_df.empty:
            fore_only_df = past_fore_df.loc[past_fore_df.index.isin(forecast_only_hours)]
            est_util_missing = fore_only_df[['est_Solar_Utilization', 'est_Wind_Utilization']].isnull().sum().sum()
        else:
            est_util_missing = 0
        
        past_missing = 0  # historical 결측은 forecast가 보충하므로 무시
        future_missing = (
            int(future_df.drop(columns=EXCLUDE_FROM_CHECK, errors='ignore').isna().sum().sum())
            if not future_df.empty else 0
        )
        
        past_ok = past_combined_hours >= 336 and est_util_missing == 0
        forecast_supplement = past_combined_hours - past_hours
        past_label = f"{past_hours} + {forecast_supplement}보충"
        future_ok = future_hours >= 24 and future_missing == 0

    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        "과거 실측 (Historical)", 
        past_label,
        "정상" if past_ok else "부족",
        delta_color="normal" if past_ok else "inverse"
    )
    
    col2.metric(
        "미래 예보 (Forecast)", 
        f"{future_hours} / 24시간",
        "정상" if future_ok else "부족",
        delta_color="normal" if future_ok else "inverse"
    )
    
    col3.metric(
        "실측 결측치", 
        f"{past_missing}건",
        "없음" if past_missing == 0 else "부족",
        delta_color="normal" if past_missing == 0 else "inverse"
    )
    
    col4.metric(
        "예보 결측치", 
        f"{future_missing}건",
        "없음" if future_missing == 0 else "API 재수집 필요",
        delta_color="normal" if future_missing == 0 else "inverse"
    )
    # ── 예보 최신화 버튼 (데이터 정상이어도 KMA 최신 사이클 반영) ──
    if past_ok and future_ok:
        with st.expander("🔄 예보 데이터 최신화", expanded=False):
            st.caption("KMA 기상 예보는 6시간 주기로 갱신됩니다. (00시, 06시, 12시, 18시)최신 사이클을 반영하려면 아래 버튼을 눌러주세요.")
            if st.button("🌤️ KMA 예보 최신화", use_container_width=True):
                with st.spinner("최신 KMA 예보를 수집하고 있습니다..."):
                    try:
                        target_str = target_date.strftime("%Y-%m-%d")
                        daily_forecast_kma(target_str, target_str)
                        st.success(f"{target_date} KMA 예보 최신화 완료!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"KMA 예보 수집 실패: {e}")

    # 문제가 있을 때 구체적 안내 + 빠른 수집
    if not past_ok or not future_ok:
    
        # 빠른 수집 가능 여부 판단
        past_gap = max(336 - past_hours, 0) if not needs_forecast_supplement else max(336 - past_combined_hours, 0)
        future_gap = max(24 - future_hours, 0)
        
        has_past_missing = past_missing > 0
        has_future_missing = future_missing > 0  # 이 부분 추가
        
        needs_past_fetch = past_gap > 0 or has_past_missing
        needs_future_fetch = future_gap > 0 or has_future_missing  # future_gap이 0이어도 결측치가 있으면 True
        
        # est_util_missing이 있으면 이전 날짜 예측이 먼저 필요 -> 빠른 수집 불가
        has_util_issue = (est_util_missing if needs_forecast_supplement else 0) > 0
        # 대규모 실측 부족(48시간 초과)도 빠른 수집 불가
        can_quick_fix = (past_gap <= 48) and not has_util_issue
        
        # -- 빠른 수집 버튼 --
        if can_quick_fix and (needs_past_fetch or needs_future_fetch):
            target_str = target_date.strftime("%Y-%m-%d")
            
            # 과거 실측: 항상 -3일 ~ -1일 고정 (API가 빠르므로 넉넉하게)
            hist_start = (target_date - timedelta(days=3)).strftime("%Y-%m-%d")
            hist_end = (target_date - timedelta(days=1)).strftime("%Y-%m-%d")
            
            # 안내 메시지
            fetch_desc = []
            if needs_past_fetch:
                if has_past_missing and past_gap == 0:
                    fetch_desc.append(f"실측 결측치 {past_missing}건 보충 ({hist_start}~{hist_end})")
                else:
                    fetch_desc.append(f"실측 {past_gap}시간 부족 ({hist_start}~{hist_end})")
            
            if needs_future_fetch:
                # 미래 데이터 결측치 안내 로직 추가
                if has_future_missing and future_gap == 0:
                    fetch_desc.append(f"예보 결측치 {future_missing}건 보충 ({target_str})")
                else:
                    fetch_desc.append(f"예보 {future_gap}시간 부족 ({target_str})")
            
            if st.button("📡 부족 데이터 빠른 수집", type="primary", width='stretch'):
                with st.spinner("부족한 데이터를 수집하고 있습니다..."):
                    try:
                        collected = []
                        
                        if needs_past_fetch:
                            daily_historical_kpx(hist_start, hist_end)
                            daily_historical_kma(hist_start, hist_end)
                            daily_historical_kpx_smp(hist_start, hist_end)
                            collected.append(f"실측 {hist_start}~{hist_end}")
                        
                        if needs_future_fetch:
                            daily_forecast_kpx(target_str, target_str)
                            daily_forecast_kma(target_str, target_str)
                            collected.append(f"예보 {target_str}")
                        
                        st.success(f"수집 완료! ({', '.join(collected)})")
                        st.rerun()
                    except Exception as e:
                        st.error(f"수집 실패: {e}")
            
            st.caption(f"💡 {' / '.join(fetch_desc)}")
        
        # ── 상세 안내 (expander) ──
        with st.expander("⚠️ 부족한 데이터 상세 확인", expanded=False):
            
            if not needs_forecast_supplement:
                if past_hours < 336:
                    st.warning(f"📈 **실측 데이터 부족**: {past_hours}시간 수집됨 (필요: 336시간)")
                if past_missing > 0 and not past_df.empty:
                    missing_hours = int(past_df.drop(columns=EXCLUDE_FROM_CHECK, errors='ignore').isna().any(axis=1).sum())
                    st.caption(f"결측 발생 시간: {missing_hours}h")
                #if past_missing > 0 and not past_df.empty:
                #    missing_cols = past_df.drop(columns=EXCLUDE_FROM_CHECK, errors='ignore').isna().sum()
                #    missing_cols = missing_cols[missing_cols > 0]
                #    if not missing_cols.empty:
                #        st.caption(" " + ", ".join([f"{col}: {cnt}건" for col, cnt in missing_cols.items()]))
            else:
                if past_combined_hours < 336:
                    st.warning(
                        f"📈 **데이터 부족**: 실측 {past_hours}시간 + 예보 보충 {forecast_supplement}시간 = "
                        f"총 {past_combined_hours}시간 (필요: 336시간)")
                if has_util_issue:
                    need_dates = sorted(forecast_only_hours)
                    if need_dates:
                        first_date = need_dates[0][:10]
                        st.warning(
                            f"⚡ **이전 날짜 예측 필요**: 보충 구간에 예측값(est_Utilization)이 {int(est_util_missing)}건 비어있습니다.\n\n"
                            f"**{first_date}부터 순서대로 예측을 먼저 실행**해 주세요.")
            
            if future_hours < 24:
                st.warning(f"🌤️ **Forecast 데이터 부족**: {future_hours}시간 수집됨 (필요: 24시간)")
            if future_missing > 0 and not future_df.empty:
                missing_hours = int(future_df.drop(columns=EXCLUDE_FROM_CHECK, errors='ignore').isna().any(axis=1).sum())
                st.caption(f"결측 발생 시간: {missing_hours}h")
            #if future_missing > 0 and not future_df.empty:
            #    missing_cols_f = future_df.drop(columns=EXCLUDE_FROM_CHECK, errors='ignore').isna().sum()
            #    missing_cols_f = missing_cols_f[missing_cols_f > 0]
            #    if not missing_cols_f.empty:
            #        st.caption(" " + ", ".join([f"{col}: {cnt}건" for col, cnt in missing_cols_f.items()]))
            
            if not can_quick_fix:
                st.info("📌 부족량이 48시간을 초과하거나 이전 날짜 예측이 필요합니다. Option A에서 수동으로 수집해 주세요.")
            
            if st.button("📡 API 데이터 수집 페이지로 이동", type="secondary"):
                st.session_state["_navigate_to"] = "Option A : DB 관리"
                st.rerun()

    # ==========================================
    # 예측 실행 버튼
    # ==========================================
    st.markdown("---")
    st.caption("**2일 이상 미래 예측 시** 이전 날짜의 예측이 먼저 완료되어야 함.")
    
    if st.button("🚀 예측 실행", type="primary", width="stretch"):
        with st.spinner(f"{target_date} 예측을 진행 중입니다... (데이터 검증 및 모델 추론)"):
            success, message, input_info = run_model_prediction(
                target_date.strftime('%Y-%m-%d'), db, assets
            )
            if success:
                st.session_state['last_predicted_date'] = target_date
                st.session_state['_pred_success'] = True
                st.session_state['_pred_message'] = message
                st.rerun()
            else:
                st.error(f"예측 실패: {message}")
                
    # ── 예측 성공 후 결과 표시 (rerun 후에도 유지) ──
    if st.session_state.get('_pred_success', False):
        st.success(st.session_state.get('_pred_message', '예측 완료!'))
        
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
                st.info("예측 결과 조회에 실패했습니다.")
        
        if st.button("📈 예측 결과 확인으로 이동 →", type="primary"):
            st.session_state['_pred_success'] = False
            st.session_state["_navigate_to"] = "Option C : 예측 결과 시각화"
            st.rerun()
    pass
            

# ==========================================
# Option C : 예측 결과 시각화
# ==========================================
elif menu == "Option C : 예측 결과 시각화":
    
    # --- KPX 실측 자동 수집 (50분 쿨다운) ---
    _now = datetime.now()
    _last_kpx = st.session_state.get('_kpx_last_fetch', None)
    if _last_kpx is None or (_now - _last_kpx).total_seconds() >= 3000:
        try:
            daily_historical_kpx(
                (_now - timedelta(days=1)).strftime("%Y-%m-%d"),
                _now.strftime("%Y-%m-%d")
            )
            st.session_state['_kpx_last_fetch'] = _now
        except Exception:
            pass
    st.subheader("📈 예측 결과 및 Net Demand 분석")
    st.caption("매일 자정마다 업데이트 되는 데이터로 모델이 예측한 가동률을 실측 발전량과 순부하(Net Demand)를 시각화합니다. 실측 데이터 오버레이 선택가능.")
    st.markdown("---")
    
    # 💡 session_state에서 마지막 예측 날짜를 기본값으로 사용
    default_vis_date = st.session_state.get('last_predicted_date', datetime.now().date())
    target_date = st.date_input("조회할 예측 날짜 선택", default_vis_date)
    
    start_str = f"{target_date} 00:00:00"
    end_str = f"{target_date} 23:00:00"
    df_res = db.get_forecast(start_str, end_str)
    
    if df_res.empty or 'est_Solar_Utilization' not in df_res.columns or df_res['est_Solar_Utilization'].isnull().all():
        st.warning(f"{target_date}의 예측 데이터가 없습니다. [Option B : 발전량 예측]에서 먼저 예측을 실행해 주세요.")
        if st.button("🔮 발전량 예측으로 이동 →", key="val_to_pred"):
            st.session_state['last_predicted_date'] = target_date
            st.session_state["_navigate_to"] = "Option B : 발전량 예측"
            st.rerun()
    else:
        df = df_res.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
            
        df['est_solar_gen'] = df['est_Solar_Utilization'] * df['Solar_Capacity_Est']
        df['est_wind_gen'] = df['est_Wind_Utilization'] * df['Wind_Capacity_Est']
        df['est_renew_total'] = df['est_solar_gen'] + df['est_wind_gen']
        df['est_net_demand'] = df['est_demand'] - df['est_renew_total']
        
        smp_col = 'smp_jeju' if 'smp_jeju' in df.columns else ('est_smp_jeju' if 'est_smp_jeju' in df.columns else None)
        
        # --- 실측 데이터 병합 시도 (historical 테이블에서 직접 조회) ---
        hist_df = db.get_historical(start_str, f"{target_date} 23:59:59")
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
                df['real_renew_total'] = df['real_solar_gen'] + df['real_wind_gen']
                df['real_net_demand'] = df['real_demand'] - df['real_renew_total']
        
        # ── session_state 초기화 ──
        if 'vis_selected_vars' not in st.session_state:
            st.session_state['vis_selected_vars'] = ['est_demand', 'est_net_demand', 'est_solar_gen', 'est_wind_gen']
        if 'vis_warn_low' not in st.session_state:
            st.session_state['vis_warn_low'] = 250
        if 'vis_warn_high' not in st.session_state:
            st.session_state['vis_warn_high'] = 750
        if 'vis_smp_low' not in st.session_state:
            st.session_state['vis_smp_low'] = 10
        if 'vis_warn_min_enabled' not in st.session_state:
            st.session_state['vis_warn_min_enabled'] = False
        if 'vis_warn_min' not in st.session_state:
            st.session_state['vis_warn_min'] = 150
        if 'vis_warn_max_enabled' not in st.session_state:
            st.session_state['vis_warn_max_enabled'] = False
        if 'vis_warn_max' not in st.session_state:
            st.session_state['vis_warn_max'] = 900
        if 'vis_show_actual' not in st.session_state:
            st.session_state['vis_show_actual'] = False
        if 'vis_actual_cols' not in st.session_state:
            st.session_state['vis_actual_cols'] = ['real_demand', 'real_solar_gen', 'real_wind_gen', 'real_renew_total', 'real_net_demand']
        
        plot_options = {
            'est_demand': '총 전력수요 예측 (est_demand)',
            'est_net_demand': '순부하 예측 (est_net_demand)',
            'est_solar_gen': '태양광 발전량 예측 (est_solar_gen)',
            'est_wind_gen': '풍력 발전량 예측 (est_wind_gen)',
            'est_renew_total': '총 재생에너지 발전량 (est_renew_total)'
        }
        if smp_col:
            plot_options[smp_col] = f'제주 SMP 가격 ({smp_col})'
        
        # ── 표시 항목 설정 dialog ──
                # ── 표시 항목 설정 dialog (예측 + 실측 통합) ──
        actual_label_map = {
            'real_demand': '수요 실측',
            'real_solar_gen': '태양광 실측',
            'real_wind_gen': '풍력 실측',
            'real_renew_total': '재생E 실측 합계',
            'real_net_demand': '순부하 실측',
        }
        available_actual = {col: label for col, label in actual_label_map.items() 
                            if col in df.columns and df[col].notna().any()} if has_actual else {}
        
        @st.dialog("📊 표시 항목 설정")
        def select_plot_items():
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**예측 데이터**")
                current_est = st.session_state.get('vis_selected_vars', [])
                est_selections = {}
                for key, label in plot_options.items():
                    est_selections[key] = st.checkbox(label, value=(key in current_est), key=f"dlg_est_{key}")
            
            with col2:
                st.write("**실측 데이터**")
                if available_actual:
                    current_act = st.session_state.get('vis_actual_cols', list(available_actual.keys()))
                    act_selections = {}
                    for key, label in available_actual.items():
                        act_selections[key] = st.checkbox(label, value=(key in current_act), key=f"dlg_act_{key}")
                else:
                    st.caption("실측 데이터가 없습니다.")
                    act_selections = {}
            
            st.markdown("---")
            if st.button("적용", type="primary", use_container_width=True):
                st.session_state['vis_selected_vars'] = [k for k, v in est_selections.items() if v]
                if available_actual:
                    st.session_state['vis_actual_cols'] = [k for k, v in act_selections.items() if v]
                st.rerun()
                
        # ── 2탭 구성 ──
        tab_chart, tab_table = st.tabs(["📈 시각화", "📋 데이터 테이블"])
        
        # 설정값 로드
        selected_vars = st.session_state['vis_selected_vars']
        warning_threshold = st.session_state['vis_warn_low']
        warning_threshold2 = st.session_state['vis_warn_high']
        smp_threshold = st.session_state['vis_smp_low'] if smp_col else 0
        
        # --- Tab 1: 시각화 차트 ---
        with tab_chart:
            # ── 차트 위: 표시 항목 설정 버튼 ──
            btn_col1, btn_col2 = st.columns([1, 5])
            with btn_col1:
                if st.button("⚙️ 표시 항목 설정", width='stretch'):
                    select_plot_items()
            with btn_col2:
                # 현재 선택된 항목 요약 표시
                if selected_vars:
                    labels = [plot_options.get(v, v).split('(')[0].strip() for v in selected_vars]
                    st.caption(f"📌 {', '.join(labels)}")
                else:
                    st.caption("📌 표시 항목 없음 — 버튼을 눌러 선택하세요")
            
            if not selected_vars:
                st.info("👉 [⚙️ 표시 항목 설정] 버튼을 눌러 시각화할 데이터를 선택해 주세요.")
            else:
                # 실측 오버레이 관련 설정 읽기
                selected_actual_cols = st.session_state.get('vis_actual_cols', [])
                show_actual = st.session_state.get('vis_show_actual', True) if has_actual else False
                overlay_active = show_actual and len(selected_actual_cols) > 0
                
                fig = go.Figure()
                colors = {
                    'est_demand': COLORS['demand'] if 'COLORS' in globals() else 'black', 
                    'est_net_demand': COLORS['net_demand'] if 'COLORS' in globals() else 'blue', 
                    'est_solar_gen': COLORS['solar_est'] if 'COLORS' in globals() else 'orange', 
                    'est_wind_gen': COLORS['wind_est'] if 'COLORS' in globals() else 'green',
                    'est_renew_total': COLORS['renew_total'] if 'COLORS' in globals() else 'cyan'
                }
                if smp_col: colors[smp_col] = COLORS['smp'] if 'COLORS' in globals() else 'red'
                
                actual_map = {
                    'est_solar_gen': 'real_solar_gen',
                    'est_wind_gen': 'real_wind_gen',
                    'est_renew_total': 'real_renew_total',
                    'est_net_demand': 'real_net_demand',
                    'est_demand': 'real_demand',
                }
                           
                for var in selected_vars:
                    # --- 예측 트레이스 ---
                    line_style = dict(
                        color=colors.get(var, 'black'),
                        width=2 if var != 'est_net_demand' else 4,
                        dash='dot' if overlay_active else 'solid'
                    )
                    if var in ['est_solar_gen', 'est_wind_gen'] and not overlay_active:
                        line_style['dash'] = 'dash'
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df[var],
                        mode='lines+markers',
                        name=plot_options[var],
                        line=line_style,
                        hovertemplate='%{y:,.1f}',
                        legendgroup=var
                    ))
                    
                    # --- 실측 트레이스 ---
                    if overlay_active:
                        actual_col = actual_map.get(var)
                        if actual_col and actual_col in selected_actual_cols and actual_col in df.columns:
                            actual_name_map = {
                                'real_solar_gen': '태양광 실측',
                                'real_wind_gen': '풍력 실측',
                                'real_renew_total': '재생E 실측 합계',
                                'real_net_demand': '순부하 실측',
                                'real_demand': '수요 실측',
                            }                          
                            fig.add_trace(go.Scatter(
                                x=df.index, y=df[actual_col],
                                mode='lines+markers',
                                name=actual_name_map.get(actual_col, actual_col),
                                line=dict(
                                    color=colors.get(var, 'black'),
                                    width=3 if var != 'est_net_demand' else 5,
                                    dash='solid'
                                ),
                                marker=dict(size=6),
                                hovertemplate='%{y:,.1f}',
                                legendgroup=var
                            ))
                            
                # --- 현재 시각 세로선 ---
                now = datetime.now()
                target_start = datetime.combine(target_date, datetime.min.time())
                target_end = datetime.combine(target_date, datetime.max.time())

                if target_start <= now <= target_end:
                    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
                    fig.add_vline(x=now_str, line_width=1, line_dash="solid", line_color="tomato")
                    fig.add_annotation(
                        x=now_str, y=-0.06, yref="paper",
                        text="현재", showarrow=False,
                        font=dict(size=10, color="tomato")
                    )
                
                # --- 경고 음영 ---
                low_gen_condition = df['est_net_demand'] < warning_threshold
                if smp_col and smp_col in df.columns:
                    low_gen_condition = low_gen_condition | (df[smp_col] < smp_threshold)
                    
                draw_danger_zones(fig, df, low_gen_condition, "red", "LNG 저발전 경고🚨", layer_pos="below", fill_opacity=0.15)
                draw_danger_zones(fig, df, df['est_net_demand'] > warning_threshold2, "blue", "LNG 고발전 경고🚨", layer_pos="below", fill_opacity=0.15)
                
                if st.session_state.get('vis_warn_min_enabled', False):
                    draw_danger_zones(
                        fig, df,
                        df['est_net_demand'] < st.session_state['vis_warn_min'],
                        "purple", annotation_text=" ",
                        show_legend_label="최저발전구간",
                        layer_pos="above", fill_opacity=0.3
                    )
                if st.session_state.get('vis_warn_max_enabled', False):
                    draw_danger_zones(
                        fig, df,
                        df['est_net_demand'] > st.session_state['vis_warn_max'],
                        "brown", annotation_text=" ",
                        show_legend_label="최대발전구간",
                        layer_pos="above", fill_opacity=0.3
                    )
                
                fig.update_layout(
                    title=f"{target_date} 전력수급 및 재생에너지 예측 결과" + (" (실측 오버레이)" if overlay_active else ""),
                    xaxis_title="시간",
                    yaxis_title="발전량 / 전력량 (MW)",
                    hovermode="x unified", 
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    dragmode='zoom',
                    yaxis=dict(fixedrange=True)
                )
                
                st.plotly_chart(fig, width="stretch")
                # ── 차트 아래: 실측 오버레이 체크박스 ──
                if has_actual:
                    st.checkbox("📊 실측 데이터 오버레이", key='vis_show_actual')
                else:
                    st.info("ℹ️ 실측 데이터가 없습니다. 새로고침 혹은 API 데이터 호출이 필요합니다.")
                st.caption("순부하(net_demand) : 전체 부하에서 신재생발전을 제외한 내연 발전기 및 연계선이 담당해야 할 총 전력부하로 전력거래소에서 판단하여 제어.")
            
            # ── 차트 아래: 경고 기준 설정 (expander, 기본 접힘) ──
            with st.expander("🚨 경고 기준 설정", expanded=False):
                st.caption("est_net_demand 기준으로 위험 구간을 음영 처리합니다.")
                warn_col1, warn_col2, warn_col3 = st.columns(3)
                with warn_col1:
                    w_low = st.number_input("🔴 저발전 경고 (MW)", value=st.session_state['vis_warn_low'], step=10, key='vis_warn_low_input')
                    st.session_state['vis_warn_low'] = w_low
                with warn_col2:
                    w_high = st.number_input("🔵 고발전 경고 (MW)", value=st.session_state['vis_warn_high'], step=10, key='vis_warn_high_input')
                    st.session_state['vis_warn_high'] = w_high
                with warn_col3:
                    if smp_col:
                        w_smp = st.number_input("🟡 SMP 하한 (원)", value=st.session_state['vis_smp_low'], step=10, key='vis_smp_low_input')
                        st.session_state['vis_smp_low'] = w_smp
                    else:
                        st.info("SMP 데이터가 없습니다.")
                
                st.caption("💡 저발전 경고: est_net_demand < 저발전 임계값 **또는** SMP < SMP 하한일 때 발동됩니다.")
                
                st.markdown("---")
                st.write("**추가 경고 (선택)**")
                
                extra_col1, extra_col2 = st.columns(2)
                with extra_col1:
                    warn_min_on = st.checkbox("🟣 최저발전 경고 활성화", value=st.session_state['vis_warn_min_enabled'], key='vis_warn_min_cb')
                    st.session_state['vis_warn_min_enabled'] = warn_min_on
                    if warn_min_on:
                        warn_min_val = st.number_input("최저발전 임계값 (MW)", value=st.session_state['vis_warn_min'], step=10, key='vis_warn_min_input')
                        st.session_state['vis_warn_min'] = warn_min_val
                with extra_col2:
                    warn_max_on = st.checkbox("🟤 최대발전 경고 활성화", value=st.session_state['vis_warn_max_enabled'], key='vis_warn_max_cb')
                    st.session_state['vis_warn_max_enabled'] = warn_max_on
                    if warn_max_on:
                        warn_max_val = st.number_input("최대발전 임계값 (MW)", value=st.session_state['vis_warn_max'], step=10, key='vis_warn_max_input')
                        st.session_state['vis_warn_max'] = warn_max_val
        
        # --- Tab 2: 데이터 테이블 ---
        with tab_table:
            st.subheader(f"{target_date} 상세 데이터")
            display_cols = ['est_demand', 'est_solar_gen', 'est_wind_gen', 'est_renew_total', 'est_net_demand']
            
            if has_actual:
                actual_display = ['real_solar_gen', 'real_wind_gen', 'real_renew_total', 'real_net_demand']
                if 'real_demand' in df.columns:
                    actual_display.insert(0, 'real_demand')
                display_cols += [c for c in actual_display if c in df.columns]
            
            if smp_col: display_cols.append(smp_col)
            display_cols = [c for c in display_cols if c in df.columns]
            
            def highlight_warnings(row):
                styles = [''] * len(row)
                if 'est_net_demand' in row.index and row['est_net_demand'] < warning_threshold:
                    idx = row.index.get_loc('est_net_demand')
                    styles[idx] = 'background-color: #ffcccc'
                if smp_col and smp_col in row.index and row[smp_col] < smp_threshold:
                    idx = row.index.get_loc(smp_col)
                    styles[idx] = 'background-color: #ffe5b4'
                return styles

            st.dataframe(df[display_cols].style.apply(highlight_warnings, axis=1).format(precision=2), width="stretch")
            
    pass


# ==========================================
# Option D : 예측 정확도 검증
# ==========================================
elif menu == "Option D : 예측 정확도 검증":
    
    # --- KPX 실측 자동 수집 (50분 쿨다운) ---
    _now = datetime.now()
    _last_kpx = st.session_state.get('_kpx_last_fetch', None)
    if _last_kpx is None or (_now - _last_kpx).total_seconds() >= 3000:
        try:
            daily_historical_kpx(
                (_now - timedelta(days=1)).strftime("%Y-%m-%d"),
                _now.strftime("%Y-%m-%d")
            )
            st.session_state['_kpx_last_fetch'] = _now
        except Exception:
            pass
    
    st.subheader("✅ 예측 정확도 검증")
    st.caption("예측 모델의 결과와 실제 발전량을 비교하여 정확도를 평가합니다. 실시간 비교는 [Option C : 예측 결과 시각화]에서 확인할 수 있습니다.")
    with st.expander("📅 분석 기간 설정", expanded=False):
        common_start, common_end = date_range_selector("val_common", allow_future_days=0, default_option="1주")

    common_df = merge_actual_and_forecast(
        db,
        common_start.strftime("%Y-%m-%d"),
        common_end.strftime("%Y-%m-%d 23:59:59"),
    )

 
    tab1, tab2, tab3, tab4 = st.tabs(["📊 일간 비교", "📈 기간별 정확도 평가","☀️ 태양광 Bias", "💨 풍력 Bias"])
       
    # ── Bias 탭용 전처리 ──
    _bias_ready = False
    daytime = pd.DataFrame()
    if not common_df.empty:
        common_df.index = pd.to_datetime(common_df.index)
        daytime = common_df[common_df.index.hour.isin(range(8, 21))].copy()
        if len(daytime) >= 24:
            daytime['date'] = daytime.index.date
            _bias_ready = True

    # ── 날씨 분류 헬퍼 ──
    def _classify_sky_daily(group):
        """08~20시 전운량 → 맑음/구름많음/흐림 + 변동 태그"""
        mean_v = group.mean()
        std_v = group.std()
        if mean_v > 1.0:
            mean_v, std_v = mean_v / 10.0, std_v / 10.0
        if mean_v <= 0.3:
            label = "맑음"
        elif mean_v <= 0.7:
            label = "구름많음"
        else:
            label = "흐림"
        if std_v > 0.25:
            label += "(변동성큼)"
        return label

    # --- Tab 1: 일간 비교 ---
    with tab1:     
        default_val_date = st.session_state.get('last_predicted_date', datetime.now().date() - timedelta(days=1))
        target_date = st.date_input("비교할 날짜를 선택하세요", default_val_date, key="val_daily_date")
        
        val_df = merge_actual_and_forecast(db, f"{target_date} 00:00:00", f"{target_date} 23:59:59")
        
        if val_df.empty:
            st.warning(f"{target_date}의 실측 데이터 또는 예측 데이터가 부족합니다.")
            if st.button("🔮 발전량 예측으로 이동 →", key="val_to_pred"):
                st.session_state['last_predicted_date'] = target_date
                st.session_state["_navigate_to"] = "Option B : 발전량 예측"
                st.rerun()
        else:
            plot_actual_vs_pred(val_df, str(target_date), radio_key="daily_radio")
            
            
            # ========================================
            # 📋 주요 피처 비교 테이블 (06시~20시)
            # ========================================
            with st.expander("📋 주요 입력 피처 비교 (06:00 ~ 20:00)", expanded=False):
                st.caption("모델에 입력된 기상 피처의 실측값과 예보값을 비교합니다. 예측 오차의 원인을 파악하는 데 활용하세요.")

                # 시간 필터링 (06시 ~ 20시)
                feature_df = val_df.copy()
                feature_df.index = pd.to_datetime(feature_df.index)
                feature_df = feature_df[(feature_df.index.hour >= 6) & (feature_df.index.hour <= 20)]
                
                if feature_df.empty:
                    st.info("06시~20시 구간의 데이터가 없습니다.")
                else:
                    # 컬럼 존재 여부에 따라 동적으로 
                    # merge_actual_and_forecast 결과: 겹치는 컬럼은 _fore suffix
                    table_cols = {}
                    display_cols = {}
                    
                    # 풍속: historical=wind_spd, forecast=wind_spd_fore
                    # 풍속 비교: 실측 north vs 예보 north
                    if 'wind_spd_north' in feature_df.columns:
                        table_cols['wind_spd_north'] = '풍속 실측-북쪽 (m/s)'
                    if 'wind_spd_north_fore' in feature_df.columns:
                        table_cols['wind_spd_north_fore'] = '풍속 예보-북쪽 (m/s)'
                    
                    # 일사량: historical=solar_rad, forecast=solar_rad_fore
                    if 'solar_rad' in feature_df.columns:
                        table_cols['solar_rad'] = '일사량 실측 (MJ/m²)'
                    if 'solar_rad_fore' in feature_df.columns:
                        table_cols['solar_rad_fore'] = '일사량 예보 (MJ/m²)'
                    
                    # 기온 (참고용)
                    if 'temp_c' in feature_df.columns:
                        table_cols['temp_c'] = '기온 실측 (°C)'
                    if 'temp_c_fore' in feature_df.columns:
                        table_cols['temp_c_fore'] = '기온 예보 (°C)'
                    
                    # 전운량
                    if 'total_cloud' in feature_df.columns:
                        table_cols['total_cloud'] = '전운량 실측'
                    if 'total_cloud_fore' in feature_df.columns:
                        table_cols['total_cloud_fore'] = '전운량 예보'
                    
                    # 강수량
                    if 'rainfall' in feature_df.columns:
                        table_cols['rainfall'] = '강수량 실측 (mm)'
                    if 'rainfall_fore' in feature_df.columns:
                        table_cols['rainfall_fore'] = '강수량 예보 (mm)'
                    
                    available_cols = [c for c in table_cols.keys() if c in feature_df.columns]
                    
                    if available_cols:
                        display_df = feature_df[available_cols].copy()
                        display_df = display_df.rename(columns=table_cols)
                        display_df.index = pd.to_datetime(display_df.index).strftime('%H:%M')
                        display_df.index.name = '시간'
                        
                        # 소수점 정리
                        display_df = display_df.round(2)
                        
                        st.dataframe(display_df, width='stretch')
                        
                        # 피처별 오차 요약
                        st.write("#### 📊 피처 오차 요약 (06~20시 평균)")
                        
                        error_data = []
                        feature_pairs = [
                            ('wind_spd', 'wind_spd_fore', '풍속 (m/s)'),
                            ('solar_rad', 'solar_rad_fore', '일사량 (MJ/m²)'),
                            ('temp_c', 'temp_c_fore', '기온 (°C)'),
                            ('total_cloud', 'total_cloud_fore', '전운량'),
                            ('rainfall', 'rainfall_fore', '강수량 (mm)'),
                        ]
                        
                        for actual_col, forecast_col, label in feature_pairs:
                            if actual_col in feature_df.columns and forecast_col in feature_df.columns:
                                diff = feature_df[forecast_col] - feature_df[actual_col]
                                error_data.append({
                                    '피처': label,
                                    '실측 평균': round(feature_df[actual_col].mean(), 2),
                                    '예보 평균': round(feature_df[forecast_col].mean(), 2),
                                    '평균 오차 (예보-실측)': round(diff.mean(), 2),
                                    'MAE': round(diff.abs().mean(), 2),
                                })
                        
                        if error_data:
                            error_summary = pd.DataFrame(error_data)
                            st.dataframe(error_summary, width='stretch', hide_index=True)
                    else:
                        st.info("비교 가능한 피처 컬럼이 없습니다. 데이터를 확인해주세요.")

    # =========================================================================
    # Tab 2: 📈 기간별 정확도 평가
    # =========================================================================
    with tab2:
        if common_df.empty:
            st.warning("선택한 기간의 데이터가 부족합니다.")
        else:
            w_val_df = common_df.dropna(subset=['real_solar_gen', 'est_solar_gen', 'real_wind_gen', 'est_wind_gen'])

            if len(w_val_df) == 0:
                st.warning("실측·예측 데이터가 모두 있는 시간이 없습니다.")
            else:
                st.write("### 🎯 평가지표")
                st.caption("💡 **RMSE**: 큰 오차에 패널티 부여 / **MAE**: 평균적으로 몇 MW 차이나는지 직관적으로 표현")

                solar_rmse = np.sqrt(mean_squared_error(w_val_df['real_solar_gen'], w_val_df['est_solar_gen']))
                solar_mae = mean_absolute_error(w_val_df['real_solar_gen'], w_val_df['est_solar_gen'])
                wind_rmse = np.sqrt(mean_squared_error(w_val_df['real_wind_gen'], w_val_df['est_wind_gen']))
                wind_mae = mean_absolute_error(w_val_df['real_wind_gen'], w_val_df['est_wind_gen'])

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("태양광 RMSE", f"{solar_rmse:.2f} MW")
                m2.metric("태양광 MAE", f"{solar_mae:.2f} MW")
                m3.metric("풍력 RMSE", f"{wind_rmse:.2f} MW")
                m4.metric("풍력 MAE", f"{wind_mae:.2f} MW")

                st.markdown("---")

                st.write("### 📉 실제 vs 예측 추이")
                st.caption("전체 기간 동안의 모델 예측이 실제 트렌드를 잘 따라가는지 확인합니다.")

                fig_w_line = px.line(
                    w_val_df,
                    x=w_val_df.index,
                    y=['real_solar_gen', 'est_solar_gen', 'real_wind_gen', 'est_wind_gen'],
                    labels={"value": "발전량 (MW)", "index": "시간", "variable": "항목"}
                )
                fig_w_line.update_layout(hovermode="x unified", dragmode='zoom', yaxis=dict(fixedrange=True))
                st.plotly_chart(fig_w_line, width="stretch")

                st.markdown("---")

                st.write("### 🌌 실제값 vs 예측값 산점도")
                st.caption("점이 대각선(y=x)에 가깝게 모여 있을수록 정확한 예측입니다.")

                sc_col1, sc_col2 = st.columns(2)
                with sc_col1:
                    fig_s = px.scatter(w_val_df, x='real_solar_gen', y='est_solar_gen', opacity=0.5,
                                       title="태양광 (Solar)", color_discrete_sequence=[COLORS['solar_est']])
                    fig_s.add_shape(type="line", line=dict(dash="dash", color="gray"),
                                   x0=0, y0=0, x1=w_val_df['real_solar_gen'].max(), y1=w_val_df['real_solar_gen'].max())
                    st.plotly_chart(fig_s, width="stretch")
                with sc_col2:
                    fig_w = px.scatter(w_val_df, x='real_wind_gen', y='est_wind_gen', opacity=0.5,
                                       title="풍력 (Wind)", color_discrete_sequence=[COLORS['wind_est']])
                    fig_w.add_shape(type="line", line=dict(dash="dash", color="gray"),
                                   x0=0, y0=0, x1=w_val_df['real_wind_gen'].max(), y1=w_val_df['real_wind_gen'].max())
                    st.plotly_chart(fig_w, width="stretch")

    # ── 날씨 분류 헬퍼 (rainfall 추가) ──
    def _classify_sky_daily(cloud_group, rainfall_group=None):
        """08~20시 전운량 → 맑음/구름많음/흐림/비 + 변동 태그
        
        rainfall_group이 주어지면, 일 합산 강수량 > 3.0 일 때 '비'로 우선 분류
        """
        # 비 판정: 해당 날의 강수량 합계가 0보다 크면 '비'
        if rainfall_group is not None and rainfall_group.sum() > 3.0:
            return "비"
        
        mean_v = cloud_group.mean()
        std_v = cloud_group.std()
        if mean_v > 1.0:
            mean_v, std_v = mean_v / 10.0, std_v / 10.0
        if mean_v <= 0.3:
            label = "맑음"
        elif mean_v <= 0.7:
            label = "구름많음"
        else:
            label = "흐림"
        if std_v > 0.25:
            label += "(변동성큼)"
        return label


    # =========================================================================
    # Tab 3: ☀️ 태양광 Bias (수정본)
    # =========================================================================
    with tab3:
        st.subheader("☀️ 태양광 — 날씨 조건별 예측 검증")
        st.caption(
            "08~20시 전운량과 강수량으로 날씨를 맑음 / 구름많음 / 흐림 / 비로 구분하고, "
            "각 조건에서 일사량 예보와 발전량 예측이 얼마나 정확했는지 비교합니다."
        )

        if not _bias_ready:
            st.warning("분석 가능한 08~20시 데이터가 부족합니다.")
        else:
            has_cloud_actual = 'total_cloud' in daytime.columns
            has_cloud_fore = 'total_cloud_fore' in daytime.columns
            has_solar_pair = 'solar_rad' in daytime.columns and 'solar_rad_fore' in daytime.columns
            has_solar_gen = 'real_solar_gen' in daytime.columns and 'est_solar_gen' in daytime.columns
            has_rainfall = 'rainfall' in daytime.columns  # ← 추가

            if not (has_cloud_actual and has_solar_pair):
                st.info("전운량(total_cloud) 또는 일사량(solar_rad / solar_rad_fore) 데이터가 없습니다.")
            else:
                solar_cols = ['total_cloud', 'solar_rad', 'solar_rad_fore']
                if has_cloud_fore:
                    solar_cols.append('total_cloud_fore')
                if has_solar_gen:
                    solar_cols += ['real_solar_gen', 'est_solar_gen']
                if has_rainfall:                          # ← 추가
                    solar_cols.append('rainfall')
                solar_cols = [c for c in solar_cols if c in daytime.columns]
                solar_day = daytime[solar_cols].dropna(subset=['total_cloud', 'solar_rad', 'solar_rad_fore'])

                if solar_day.empty:
                    st.info("유효한 태양광 데이터가 부족합니다.")
                else:
                    solar_day['date'] = solar_day.index.date
                    grouped = solar_day.groupby('date')

                    rows = []
                    for dt, grp in grouped:
                        # rainfall_group 전달
                        rain_grp = grp['rainfall'] if has_rainfall and 'rainfall' in grp.columns else None

                        r = {
                            '날짜': dt,
                            '실측 날씨': _classify_sky_daily(grp['total_cloud'], rain_grp),
                            '일사량 실측': round(grp['solar_rad'].mean(), 2),
                            '일사량 예보': round(grp['solar_rad_fore'].mean(), 2),
                        }
                        if has_cloud_fore and 'total_cloud_fore' in grp.columns:
                            # 예보 날씨는 rainfall 예보를 안 쓰므로 전운량만으로 판별
                            r['예보 날씨'] = _classify_sky_daily(grp['total_cloud_fore'])
                        else:
                            r['예보 날씨'] = r['실측 날씨']

                        r['일사량 Bias'] = round(r['일사량 예보'] - r['일사량 실측'], 2)

                        if has_solar_gen and 'real_solar_gen' in grp.columns:
                            r['실측 발전량(MW)'] = round(grp['real_solar_gen'].mean(), 1)
                            r['예측 발전량(MW)'] = round(grp['est_solar_gen'].mean(), 1)
                            r['발전량 오차(MW)'] = round(r['예측 발전량(MW)'] - r['실측 발전량(MW)'], 1)

                        forecasting = str(r['예보 날씨']).replace('(변동성큼)', '')
                        real_value = str(r['실측 날씨']).replace('(변동성큼)', '')
                        
                        r['일치'] = '✅' if forecasting == real_value else '⚠️'
                        rows.append(r)

                    daily_solar = pd.DataFrame(rows).sort_values('날짜', ascending=False)

                    # ── 날씨 조건별 요약 ──
                    st.write("#### 날씨 조건별 요약")
                    sky_order = ["맑음", "구름많음", "흐림", "비"]  # ← "비" 추가
                    summary_rows = []
                    for sky in sky_order:
                        mask = daily_solar['실측 날씨'].str.startswith(sky)
                        subset = daily_solar[mask]
                        if len(subset) == 0:
                            continue
                        sr = {
                            '날씨 조건': sky, '일수': len(subset),
                            '일사량 실측 평균': round(subset['일사량 실측'].mean(), 2),
                            '일사량 예보 평균': round(subset['일사량 예보'].mean(), 2),
                            '일사량 Bias': round(subset['일사량 Bias'].mean(), 2),
                        }
                        if '실측 발전량(MW)' in subset.columns:
                            sr['실측 발전량(MW)'] = round(subset['실측 발전량(MW)'].mean(), 1)
                            sr['예측 발전량(MW)'] = round(subset['예측 발전량(MW)'].mean(), 1)
                            sr['발전량 오차(MW)'] = round(subset['발전량 오차(MW)'].mean(), 1)
                        summary_rows.append(sr)

                    if summary_rows:
                        st.dataframe(pd.DataFrame(summary_rows).style.format(precision=2), hide_index=True, width='stretch')

                    # ── 일별 상세 테이블 ──
                    st.write("#### 일별 상세")
                    st.caption("예보 날씨와 실측 날씨가 다른 날(⚠️)이 발전량 오차의 주요 원인입니다.")

                    disp_s = ['날짜', '일치', '예보 날씨', '실측 날씨', '일사량 예보', '일사량 실측', '일사량 Bias']
                    if '실측 발전량(MW)' in daily_solar.columns:
                        disp_s += ['예측 발전량(MW)', '실측 발전량(MW)', '발전량 오차(MW)']
                    disp_s = [c for c in disp_s if c in daily_solar.columns]

                    def _hl_solar(row):
                        s = [''] * len(row)
                        if row.get('일치') == '⚠️':
                            s = ['background-color: #fff3cd'] * len(row)
                        return s

                    st.dataframe(
                        daily_solar[disp_s].head(14).style.apply(_hl_solar, axis=1).format(precision=2),
                        hide_index=True, width='stretch'
                    )

    # =========================================================================
    # Tab 4: 💨 풍력 Bias
    # =========================================================================
    with tab4:
        st.subheader("💨 풍력 — 풍향·풍속 예측 검증")
        st.caption(
            "08~20시 기준 일별 풍향과 풍속(최대/평균)을 예보와 실측으로 비교하고, "
            "발전량 예측 정확도를 함께 확인합니다."
        )

        if not _bias_ready:
            st.warning("분석 가능한 08~20시 데이터가 부족합니다.")
        else:
            has_wd = 'wd_sin' in daytime.columns and 'wd_cos' in daytime.columns
            has_wind_pair = 'wind_spd_north' in daytime.columns and 'wind_spd_north_fore' in daytime.columns
            has_wind_gen = 'real_wind_gen' in daytime.columns and 'est_wind_gen' in daytime.columns

            if not (has_wd and has_wind_pair):
                st.info("풍향(wd_sin, wd_cos) 또는 풍속(wind_spd_north / wind_spd_north_fore) 데이터가 없습니다.")
            else:
                w_cols = ['wd_sin', 'wd_cos', 'wind_spd_north', 'wind_spd_north_fore']
                if has_wind_gen:
                    w_cols += ['real_wind_gen', 'est_wind_gen']
                w_cols = [c for c in w_cols if c in daytime.columns]
                wind_day = daytime[w_cols].dropna(subset=['wd_sin', 'wd_cos', 'wind_spd_north', 'wind_spd_north_fore'])

                if len(wind_day) < 24:
                    st.info("풍력 분석에 충분한 데이터가 없습니다.")
                else:
                    wind_day['wind_dir'] = (np.degrees(np.arctan2(wind_day['wd_sin'], wind_day['wd_cos'])) + 360) % 360
                    dir_bins = [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360]
                    dir_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N2']
                    wind_day['direction'] = pd.cut(wind_day['wind_dir'], bins=dir_bins, labels=dir_labels, include_lowest=True)
                    wind_day['direction'] = wind_day['direction'].astype(str)
                    wind_day['direction'] = wind_day['direction'].replace('N2', 'N')
                    wind_day['date'] = wind_day.index.date
                    grouped_w = wind_day.groupby('date')
                    rows_w = []
                    for dt, grp in grouped_w:
                        r = {
                            '날짜': dt,
                            '주풍향': grp['direction'].mode().iloc[0] if len(grp['direction'].mode()) > 0 else '-',
                            '예보 최대(m/s)': round(grp['wind_spd_north_fore'].max(), 1),
                            '예보 평균(m/s)': round(grp['wind_spd_north_fore'].mean(), 1),
                            '실측 최대(m/s)': round(grp['wind_spd_north'].max(), 1),
                            '실측 평균(m/s)': round(grp['wind_spd_north'].mean(), 1),
                        }
                        r['풍속 Bias(m/s)'] = round(r['예보 평균(m/s)'] - r['실측 평균(m/s)'], 2)
                        if has_wind_gen and 'real_wind_gen' in grp.columns:
                            r['예측 발전량(MW)'] = round(grp['est_wind_gen'].mean(), 1)
                            r['실측 발전량(MW)'] = round(grp['real_wind_gen'].mean(), 1)
                            r['발전량 오차(MW)'] = round(r['예측 발전량(MW)'] - r['실측 발전량(MW)'], 1)
                        rows_w.append(r)

                    daily_wind = pd.DataFrame(rows_w).sort_values('날짜', ascending=False)

                    # ── 기간 요약 ──
                    st.write("#### 기간 전체 요약")
                    ws = {
                        '분석 일수': len(daily_wind),
                        '실측 평균풍속(m/s)': round(daily_wind['실측 평균(m/s)'].mean(), 1),
                        '예보 평균풍속(m/s)': round(daily_wind['예보 평균(m/s)'].mean(), 1),
                        '풍속 Bias(m/s)': round(daily_wind['풍속 Bias(m/s)'].mean(), 2),
                        '풍속 MAE(m/s)': round(daily_wind['풍속 Bias(m/s)'].abs().mean(), 2),
                    }
                    if '실측 발전량(MW)' in daily_wind.columns:
                        ge = daily_wind['발전량 오차(MW)']
                        ws['발전량 Bias(MW)'] = round(ge.mean(), 1)
                        ws['발전량 MAE(MW)'] = round(ge.abs().mean(), 1)
                    st.dataframe(pd.DataFrame([ws]).style.format(precision=2), hide_index=True, width='stretch')
                    
                    # ── 풍향별 요약 ──
                    st.write("#### 풍향별 요약")
                    st.caption("8방위별로 풍속 예보 정확도와 발전량 오차를 비교합니다. 특정 풍향에서 예보 편차가 큰지 확인하세요.")
                    
                    dir_order = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
                    dir_summary_rows = []
                    for d in dir_order:
                        mask = daily_wind['주풍향'] == d
                        subset = daily_wind[mask]
                        if len(subset) == 0:
                            continue
                        dr = {
                            '풍향': d,
                            '일수': len(subset),
                            '실측 평균풍속(m/s)': round(subset['실측 평균(m/s)'].mean(), 1),
                            '예보 평균풍속(m/s)': round(subset['예보 평균(m/s)'].mean(), 1),
                            '풍속 Bias(m/s)': round(subset['풍속 Bias(m/s)'].mean(), 2),
                            '풍속 MAE(m/s)': round(subset['풍속 Bias(m/s)'].abs().mean(), 2),
                        }
                        if '실측 발전량(MW)' in subset.columns:
                            dr['실측 발전량(MW)'] = round(subset['실측 발전량(MW)'].mean(), 1)
                            dr['예측 발전량(MW)'] = round(subset['예측 발전량(MW)'].mean(), 1)
                            dr['발전량 오차(MW)'] = round(subset['발전량 오차(MW)'].mean(), 1)
                        dir_summary_rows.append(dr)
                    
                    if dir_summary_rows:
                        dir_summary_df = pd.DataFrame(dir_summary_rows)
                        
                        def _hl_dir_bias(row):
                            s = [''] * len(row)
                            if abs(row.get('풍속 Bias(m/s)', 0)) >= 1.5:
                                s = ['background-color: #fff3cd'] * len(row)
                            return s
                        
                        st.dataframe(
                            dir_summary_df.style.apply(_hl_dir_bias, axis=1).format(precision=2),
                            hide_index=True, width='stretch'
                        )
                    else:
                        st.info("풍향별 분석에 충분한 데이터가 없습니다.")

                    # ── 일별 상세 ──
                    st.write("#### 일별 상세")
                    st.caption("풍속을 과대 예보한 날(Bias > 0)일수록 발전량도 과대 예측되는 경향이 있습니다.")

                    disp_w = ['날짜', '주풍향', '예보 최대(m/s)', '예보 평균(m/s)',
                              '실측 최대(m/s)', '실측 평균(m/s)', '풍속 Bias(m/s)']
                    if '실측 발전량(MW)' in daily_wind.columns:
                        disp_w += ['예측 발전량(MW)', '실측 발전량(MW)', '발전량 오차(MW)']

                    def _hl_wind(row):
                        s = [''] * len(row)
                        if abs(row.get('풍속 Bias(m/s)', 0)) >= 2.0:
                            s = ['background-color: #fff3cd'] * len(row)
                        return s

                    st.dataframe(
                        daily_wind[disp_w].head(14).style.apply(_hl_wind, axis=1).format(precision=2),
                        hide_index=True, width='stretch'
                    )

    pass


# ==========================================
# Option E : 데이터 분석 (EDA)
# ==========================================
elif menu == "Option E : 데이터 분석 (EDA)" :
    st.subheader("🔍 데이터 분석 (EDA)")
    # 1단계: 피처 선택 (상단 expander)
    with st.expander("🎯 분석할 피처 선택", expanded=False):
        # 일단 전체 컬럼 목록을 가져오기 위해 최근 데이터를 살짝 조회
        sample_df = db.get_historical(
            (datetime.now().date() - timedelta(days=7)).strftime('%Y-%m-%d'),
            datetime.now().date().strftime('%Y-%m-%d')
        )
        
        if sample_df.empty:
            st.warning("DB에 데이터가 없습니다. [Option A : DB 관리]에서 데이터를 먼저 수집해 주세요.")
            selected_features = []
        else:
            if 'real_demand' in sample_df.columns and 'real_renew_gen' in sample_df.columns:
                sample_df['real_net_demand'] = sample_df['real_demand'] - sample_df['real_renew_gen']
            
            # ── forecast 데이터 병합하여 est_solar_gen, est_wind_gen 추가 ──
            sample_fore = db.get_forecast(
                (datetime.now().date() - timedelta(days=7)).strftime('%Y-%m-%d'),
                datetime.now().date().strftime('%Y-%m-%d')
            )
            if not sample_fore.empty:
                # est_solar_gen, est_wind_gen 계산
                if not pd.api.types.is_datetime64_any_dtype(sample_fore.index):
                    sample_fore.index = pd.to_datetime(sample_fore.index)
                if not pd.api.types.is_datetime64_any_dtype(sample_df.index):
                    sample_df.index = pd.to_datetime(sample_df.index)
                
                cap_solar_col = 'Solar_Capacity_Est' if 'Solar_Capacity_Est' in sample_fore.columns else None
                cap_wind_col = 'Wind_Capacity_Est' if 'Wind_Capacity_Est' in sample_fore.columns else None
                
                if 'est_Solar_Utilization' in sample_fore.columns and cap_solar_col:
                    sample_fore['est_solar_gen'] = (
                        pd.to_numeric(sample_fore['est_Solar_Utilization'], errors='coerce') 
                        * pd.to_numeric(sample_fore[cap_solar_col], errors='coerce')
                    )
                if 'est_Wind_Utilization' in sample_fore.columns and cap_wind_col:
                    sample_fore['est_wind_gen'] = (
                        pd.to_numeric(sample_fore['est_Wind_Utilization'], errors='coerce') 
                        * pd.to_numeric(sample_fore[cap_wind_col], errors='coerce')
                    )
                
                # sample_df에 예측 발전량 컬럼 병합
                for col in ['est_solar_gen', 'est_wind_gen']:
                    if col in sample_fore.columns:
                        sample_df[col] = sample_fore[col].reindex(sample_df.index)
            
            numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
            
            selected_features = st.multiselect(
                "분석할 피처를 선택하세요", 
                options=numeric_cols, 
                default=['real_demand', 'real_solar_gen', 'real_wind_gen', 'real_renew_gen', 'real_net_demand'] if 'real_demand' 
                in numeric_cols else numeric_cols[:2],
                label_visibility="collapsed"
            )

    # 2단계: 기간 선택 (하단 expander) — 탭 아래에 배치
    with st.expander("📅 조회 기간 설정", expanded=False):
        start_date, end_date = date_range_selector("eda", allow_future_days=0, default_option="1주")
        
    # 메인 컨텐츠: 탭 + 차트
    if not selected_features:
        st.info("👆 위에서 분석할 피처를 하나 이상 선택해 주세요.")
    else:
        tab1, tab2, tab3, tab4= st.tabs([
            "📈 시계열 데이터", 
            "📊 통계 요약", 
            "🔥 상관관계 히트맵", 
            "🌌 산점도",
        ])
        
        # 기간이 정해진 후 DB 조회
        df = db.get_historical(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if df.empty:
            st.warning(f"선택하신 기간({start_date} ~ {end_date})에 해당하는 데이터가 없습니다.")
        else:
            # ── forecast 병합: est_solar_gen, est_wind_gen 추가 ──
            if any(f in selected_features for f in ['est_solar_gen', 'est_wind_gen']):
                eda_fore = db.get_forecast(
                    start_date.strftime('%Y-%m-%d'), 
                    end_date.strftime('%Y-%m-%d')
                )
                if not eda_fore.empty:
                    if not pd.api.types.is_datetime64_any_dtype(eda_fore.index):
                        eda_fore.index = pd.to_datetime(eda_fore.index)
                    if not pd.api.types.is_datetime64_any_dtype(df.index):
                        df.index = pd.to_datetime(df.index)
                    
                    cap_s = 'Solar_Capacity_Est' if 'Solar_Capacity_Est' in eda_fore.columns else None
                    cap_w = 'Wind_Capacity_Est' if 'Wind_Capacity_Est' in eda_fore.columns else None
                    
                    if 'est_Solar_Utilization' in eda_fore.columns and cap_s:
                        eda_fore['est_solar_gen'] = (
                            pd.to_numeric(eda_fore['est_Solar_Utilization'], errors='coerce')
                            * pd.to_numeric(eda_fore[cap_s], errors='coerce')
                        )
                    if 'est_Wind_Utilization' in eda_fore.columns and cap_w:
                        eda_fore['est_wind_gen'] = (
                            pd.to_numeric(eda_fore['est_Wind_Utilization'], errors='coerce')
                            * pd.to_numeric(eda_fore[cap_w], errors='coerce')
                        )
                    
                    for col in ['est_solar_gen', 'est_wind_gen']:
                        if col in eda_fore.columns and col in selected_features:
                            df[col] = eda_fore[col].reindex(df.index)

            if 'real_demand' in df.columns and 'real_renew_gen' in df.columns:
                df['real_net_demand'] = df['real_demand'] - df['real_renew_gen']
            
            # 선택된 피처 중 실제 df에 존재하는 것만 필터
            valid_features = [f for f in selected_features if f in df.columns]
            if not valid_features:
                st.warning("선택한 피처가 조회된 데이터에 존재하지 않습니다.")
            else:
                analysis_df = df[valid_features].copy()
                with tab1:
                    normalize_eda = st.session_state.get("eda_normalize", False)
                    if normalize_eda:
                        plot_df = analysis_df.copy()
                        for col in valid_features:
                            cmin, cmax = plot_df[col].min(), plot_df[col].max()
                            if cmax - cmin > 0:
                                plot_df[col] = (plot_df[col] - cmin) / (cmax - cmin)
                            else:
                                plot_df[col] = 0.0
                        
                        fig = go.Figure()
                        for col in valid_features:
                            fig.add_trace(go.Scatter(
                                x=plot_df.index, y=plot_df[col],
                                mode='lines',
                                name=col,
                                customdata=analysis_df[col],
                                hovertemplate='정규화: %{y:.3f}<br>원본: %{customdata:,.2f}'
                            ))
                        fig.update_layout(
                            title="시계열 데이터 분석 (정규화)",
                            yaxis_title="정규화 (0~1)",
                        )
                    else:
                        fig = px.line(
                            analysis_df, 
                            x=analysis_df.index, 
                            y=valid_features,
                            title="시계열 데이터 분석"
                        )
                        fig.update_traces(hovertemplate='%{y:,.2f}')
                        fig.update_layout(yaxis_title="수치")
                    
                    fig.update_layout(
                        hovermode="x unified",
                        legend_title_text="선택된 피처",
                        xaxis_title="시간",
                        dragmode='zoom',
                        yaxis=dict(fixedrange=True)
                    )
                    st.plotly_chart(fig, width="stretch")
                    st.caption("💡 **Tip:** 차트를 드래그하면 X축만 확대됩니다. 더블클릭으로 원래 범위로 복귀합니다.")
                    st.checkbox("📐 정규화 (0~1 스케일)", value=False, key="eda_normalize")
                    

                with tab2:
                    st.subheader(f"통계 요약 ({start_date} ~ {end_date})")
                    st.dataframe(analysis_df.describe().style.format(precision=2), width="stretch")
                    
                with tab3:
                    if len(valid_features) < 2:
                        st.warning("상관관계를 분석하려면 최소 2개 이상의 피처를 선택해야 합니다.")
                    else:
                        corr_matrix = analysis_df.corr()
                        fig_corr = px.imshow(
                            corr_matrix,
                            text_auto=".2f",
                            aspect="auto",
                            color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1,
                            title="상관관계 행렬"
                        )
                        fig_corr.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_corr, width="stretch")

                with tab4:
                    if len(valid_features) < 2:
                        st.warning("산점도를 그리려면 위에서 최소 2개 이상의 피처를 선택해야 합니다.")
                    else:
                        sc_col1, sc_col2 = st.columns(2)
                        with sc_col1:
                            x_axis = st.selectbox("X축 피처 선택:", options=valid_features, index=0)
                        with sc_col2:
                            y_axis = st.selectbox("Y축 피처 선택:", options=valid_features, index=1 if len(valid_features) > 1 else 0)
                        
                        fig_scatter = px.scatter(
                            analysis_df,
                            x=x_axis,
                            y=y_axis,
                            opacity=0.5,
                            marginal_x="histogram",
                            marginal_y="histogram",
                            title=f"{x_axis} vs {y_axis} 산점도"
                        )
                        st.plotly_chart(fig_scatter, width="stretch")
                        st.info("20-24년도 자료에는 발전량 자료가 포함되어 있어 상관관계 분석 가능 합니다.")
                        

# ==========================================
# Option F : 시스템 안내
# ==========================================
elif menu == "Option F : 시스템 안내":
    render_system_info()