import os
import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.db_manager import init_db, load_range, load_weather_range
from utils import chart_warn
from utils import weather_motion

load_dotenv()   # CARTO_API_KEY (구름·바람 탭 배경 지도)

st.set_page_config(page_title="제주통제소 예측 대시보드", layout="wide")
init_db()

st.markdown("""
<style>
    header[data-testid="stHeader"] {
        background-color: #e0f8e0 !important;
    }
    header[data-testid="stHeader"]::before {
        content: "제주통제소 예측 대시보드";
        position: absolute;
        left: 80px;
        top: 15px;
        font-size: 20px;
        font-weight: 800;
        color: #2c3e50;
        z-index: 9999;
    }
    .block-container { padding-top: 3.0rem !important; }
    div[data-testid="stDateInput"] input { text-align: center; }

    /* 1920x1080 전체화면·반반모드(약 960px 폭) 둘 다에서 컨트롤 줄이 깨지지 않도록
       좁아지면 다음 줄로 자연스럽게 넘어가게 한다(고정 폭 대신 최소 폭만 보장). */
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        row-gap: 0.5rem;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        min-width: 150px;
    }
    @media (max-width: 1100px) {
        header[data-testid="stHeader"]::before { font-size: 17px; left: 60px; }
    }
</style>
""", unsafe_allow_html=True)

# (라벨, 컬럼 접미사, 색상, 기본 표시 여부) — 색상은 jeju_model pages/common.py COLOR 팔레트와 동일.
SERIES = [
    ("전력수요", "demand", "#2a78d6", True),
    ("순부하", "net_load", "#4a3aa7", True),
    ("신재생전체", "renew_gen", "#008300", False),
    ("풍력", "wind_gen", "#1baf7a", True),
    ("태양광", "solar_gen", "#eda100", True),
]

DAY_KEY = "selected_day"
if DAY_KEY not in st.session_state:
    st.session_state[DAY_KEY] = pd.Timestamp.now().normalize().date()


def _shift(delta):
    st.session_state[DAY_KEY] = st.session_state[DAY_KEY] + pd.Timedelta(days=delta)


SOLAR_MODEL_NAMES = {"patchtst": "PatchTST", "patchtst_bridge": "PatchTST(임시)", "lgbm": "LGBM"}


def solar_model_caption(df):
    """날짜별 태양광 모델을 이어지는 날끼리 묶어 '10-06 PatchTST(임시), 10-07~10-08 PatchTST' 형태로.
    jeju_model 이 모델을 기록하기 전(2026-10-06 이전) 예측은 기록이 없어 빠진다."""
    daily = df.dropna(subset=["solar_model"]).groupby(df["timestamp"].dt.date)["solar_model"].first()
    groups = []   # [첫 날, 마지막 날, 모델]
    for day, model in daily.items():
        if groups and groups[-1][2] == model and (day - groups[-1][1]).days == 1:
            groups[-1][1] = day
        else:
            groups.append([day, day, model])
    parts = []
    for first_day, last_day, model in groups:
        days = f"{first_day:%m-%d}" if first_day == last_day else f"{first_day:%m-%d}~{last_day:%m-%d}"
        parts.append(f"{days} {SOLAR_MODEL_NAMES.get(model, model)}")
    return ", ".join(parts)


# 구름·바람 탭은 열려 있을 때만 그린다(on_change="rerun" + .open) — 구름 프레임이 1MB 가까이 돼
# 예측 탭에서 버튼을 누를 때마다 지도까지 다시 보내지 않도록. 예측 탭은 항상 그려야
# 탭을 오가도 날짜·표시·경고 선택이 초기화되지 않는다.
forecast_tab, weather_tab = st.tabs(["예측", "구름·바람"], on_change="rerun")

with forecast_tab:
    col_prev, col_date, col_next, col_slider, col_series, col_warn = st.columns(
        [0.8, 1.6, 0.8, 2.2, 0.8, 0.8], vertical_alignment="center")
    col_prev.button("◀ 이전", on_click=_shift, args=(-1,), width="stretch")
    col_date.date_input("날짜", key=DAY_KEY, label_visibility="collapsed")
    col_next.button("다음 ▶", on_click=_shift, args=(1,), width="stretch")
    k = col_slider.slider("표시 기간(일)", 1, 5, 1, help="선택일부터 며칠치를 표시할지")
    with col_series.popover("표시", width="stretch"):
        chosen = {label: st.checkbox(label, value=default, key=f"series_{col}")
                  for label, col, _, default in SERIES}
    with col_warn.popover("경고", help="위험구간 음영 임계값 설정", width="stretch"):
        threshold_values = chart_warn.render_warning_threshold_inputs()
        if st.button("적용", key="warn_apply", type="primary", width="stretch"):
            chart_warn.commit_warning_thresholds(threshold_values)
            st.rerun()

    day = pd.Timestamp(st.session_state[DAY_KEY])
    start = day.strftime("%Y-%m-%d 00:00:00")
    end = (day + pd.Timedelta(days=k - 1)).strftime("%Y-%m-%d 23:00:00")
    df = load_range(start, end)

    fig = go.Figure()
    for label, col, color, _ in SERIES:
        if not chosen[label]:
            continue
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df[f"real_{col}"], name=f"{label}(실측)",
                                  mode="lines", line=dict(color=color, width=2)))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df[f"est_{col}"], name=f"{label}(예측)",
                                  mode="lines", line=dict(color=color, width=2, dash="dash")))
    # 위험구간 음영 — chart_warn 규약: DatetimeIndex 필수
    chart_warn.draw_warning_zones(fig, df.set_index("timestamp"))

    fig.update_layout(
        xaxis_title="시각", yaxis_title="MW",
        height=600, hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),          # 계열 범례 — 아래쪽
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, width="stretch")

    # 현재 화면에 표시 중인 예측이 언제 생성(발표)된 것인지 + 태양광을 어느 모델로 냈는지 표시.
    # 여러 날을 함께 보면 지평(horizon_d)별로 발표 시각이 다를 수 있어 최소~최대로 보여준다.
    forecast_bases = pd.to_datetime(df["base"].dropna().unique())
    if len(forecast_bases) == 0:
        base_text = "정보 없음"
    elif len(forecast_bases) == 1:
        base_text = f"{forecast_bases[0]:%Y-%m-%d %H:%M} 발표"
    else:
        base_text = (f"{forecast_bases.min():%Y-%m-%d %H:%M} ~ "
                     f"{forecast_bases.max():%Y-%m-%d %H:%M} 발표 (지평별로 발표 시각이 다름)")
    model_text = solar_model_caption(df)
    st.caption(f"예측 생성 시각: {base_text}" + (f" · 태양광 모델: {model_text}" if model_text else ""))

if weather_tab.open:
    with weather_tab:
        # 화면이 절반(1920 기준 약 960px)으로 좁아지면 차트·표 칸은 숨기고 지도만 전체 폭으로.
        # 칸에 key 를 붙여(st-key-...) 그 칸을 감싼 stColumn 을 CSS 로 찾는다.
        st.markdown("""
        <style>
            @media (max-width: 1100px) {
                div[data-testid="stColumn"]:has(.st-key-weather_trend) { display: none !important; }
                div[data-testid="stColumn"]:has(.st-key-weather_map) {
                    flex: 1 1 100% !important; width: 100% !important; max-width: 100% !important;
                }
            }
        </style>
        """, unsafe_allow_html=True)
        col_map, col_trend = st.columns([1, 1], gap="medium")

        with col_map, st.container(key="weather_map"):
            map_html = weather_motion.build_map_html()
            if map_html is None:
                st.info("구름·바람 자료가 아직 없습니다. 3시간마다 자동으로 받아옵니다.")
            else:
                components.html(map_html, height=780)

        with col_trend, st.container(key="weather_trend"):
            # 최근 24시간 실측 + 앞으로 24시간 예측 — 지도(구름·바람)와 발전량을 나란히 보기 위함
            now_hour = pd.Timestamp.now().floor("h")
            trend_start = now_hour - pd.Timedelta(hours=24)
            trend_end = now_hour + pd.Timedelta(hours=24)
            trend = load_range(trend_start.strftime("%Y-%m-%d %H:%M:%S"),
                               trend_end.strftime("%Y-%m-%d %H:%M:%S"))

            trend_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                      subplot_titles=("태양광", "풍력"))
            for row, (label, col, color) in enumerate(
                    [("태양광", "solar_gen", "#eda100"), ("풍력", "wind_gen", "#1baf7a")], start=1):
                trend_fig.add_trace(go.Scatter(x=trend["timestamp"], y=trend[f"real_{col}"],
                                               name=f"{label}(실측)", mode="lines",
                                               line=dict(color=color, width=2)), row=row, col=1)
                trend_fig.add_trace(go.Scatter(x=trend["timestamp"], y=trend[f"est_{col}"],
                                               name=f"{label}(예측)", mode="lines",
                                               line=dict(color=color, width=2, dash="dash")), row=row, col=1)
            trend_fig.add_vline(x=pd.Timestamp.now(), line=dict(color="#d03b3b", width=1, dash="dot"))
            trend_fig.update_yaxes(title_text="MW")
            trend_fig.update_layout(height=400, hovermode="x unified", showlegend=False,
                                    margin=dict(l=40, r=10, t=30, b=20))
            st.plotly_chart(trend_fig, width="stretch")

            # 시간별 기상 예보 표 — 예측에 실제로 들어간 입력(일사·강수 기상청 KIMG, 운량 JMA)
            zone_labels = {"서부(고산)": "west", "동부(성산)": "east", "남부(서귀포)": "south"}
            zone_label = st.segmented_control("지점", list(zone_labels), default="서부(고산)",
                                              key="weather_zone", label_visibility="collapsed")
            zone = zone_labels[zone_label or "서부(고산)"]
            table_start = now_hour - pd.Timedelta(hours=6)
            weather = load_weather_range(table_start.strftime("%Y-%m-%d %H:%M:%S"),
                                         trend_end.strftime("%Y-%m-%d %H:%M:%S"))
            weather_table = pd.DataFrame({
                "시각": weather["timestamp"].dt.strftime("%m-%d %H시"),
                "일사량(MJ/m²)": weather[f"radiation_{zone}"].round(2),
                "강수량(mm)": weather[f"rainfall_{zone}"].round(1),
                "운량(%)": (weather[f"total_cloud_{zone}"] * 100).round(0),
            })
            st.dataframe(weather_table, hide_index=True, height=300, width="stretch")
            st.caption("출처: 일사량·강수량 기상청, 운량 JMA")
