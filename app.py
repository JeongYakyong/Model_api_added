import os
import sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.db_manager import init_db, load_range

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

DAY_KEY = "selected_day"
if DAY_KEY not in st.session_state:
    st.session_state[DAY_KEY] = pd.Timestamp.now().normalize().date()


def _shift(delta):
    st.session_state[DAY_KEY] = st.session_state[DAY_KEY] + pd.Timedelta(days=delta)


col_prev, col_date, col_next, col_slider = st.columns([0.8, 1.6, 0.8, 2.5], vertical_alignment="center")
col_prev.button("◀ 이전", on_click=_shift, args=(-1,), width="stretch")
col_date.date_input("날짜", key=DAY_KEY, label_visibility="collapsed")
col_next.button("다음 ▶", on_click=_shift, args=(1,), width="stretch")
k = col_slider.slider("표시 기간(일)", 1, 2, 1, help="선택일부터 며칠치를 표시할지")

day = pd.Timestamp(st.session_state[DAY_KEY])
start = day.strftime("%Y-%m-%d 00:00:00")
end = (day + pd.Timedelta(days=k - 1)).strftime("%Y-%m-%d 23:00:00")
df = load_range(start, end)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["real_demand"], name="실측 수요",
                          mode="lines", line=dict(color="#2c3e50", width=2)))
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["est_demand"], name="예측 수요",
                          mode="lines", line=dict(color="#e67e22", width=2, dash="dash")))
fig.update_layout(
    xaxis_title="시각", yaxis_title="수요 (MW)",
    height=600, hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=20, t=40, b=40),
)
st.plotly_chart(fig, width="stretch")
