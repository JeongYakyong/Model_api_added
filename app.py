import os
import sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.db_manager import init_db, load_range
from utils import chart_warn

st.set_page_config(page_title="제주 전력수요 예측", layout="wide")
init_db()

st.markdown("""
<style>
    header[data-testid="stHeader"] {
        background-color: #e0f8e0 !important;
    }
    header[data-testid="stHeader"]::before {
        content: "제주 전력수요 예측";
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


col_prev, col_date, col_next, col_slider, col_series, col_warn = st.columns(
    [0.8, 1.6, 0.8, 2.2, 0.8, 0.8], vertical_alignment="center")
col_prev.button("◀ 이전", on_click=_shift, args=(-1,), width="stretch")
col_date.date_input("날짜", key=DAY_KEY, label_visibility="collapsed")
col_next.button("다음 ▶", on_click=_shift, args=(1,), width="stretch")
k = col_slider.slider("표시 기간(일)", 1, 5, 1, help="선택일부터 며칠치를 표시할지")
with col_series.popover("데이터 선택", width="stretch"):
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
