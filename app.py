import streamlit as st
import os
import logging
import warnings

warnings.filterwarnings("ignore", message=".*torch.classes.*")
logging.getLogger("torch").setLevel(logging.ERROR)

# ==========================================
# 페이지 설정 (반드시 최상단)
# ==========================================
st.set_page_config(
    page_title="제주통제소용 예측 대시보드",
    layout="wide",
    initial_sidebar_state="collapsed"   # lite에서는 사이드바 불필요, full에서 자동 확장됨
)

# ==========================================
# 공통 CSS
# ==========================================
st.markdown("""
<style>
    header[data-testid="stHeader"] {
        background-color: #e0f8e0 !important;
    }
    header[data-testid="stHeader"]::before {
        content: "🌱 제주통제소용 예측 대시보드";
        position: absolute;
        left: 80px;
        top: 15px;
        font-size: 20px;
        font-weight: 800;
        color: #2c3e50;
        z-index: 9999;
    }
    .block-container {
        padding-top: 3.0rem !important; # 여유공간 관련
    }
    div[data-testid="stDateInput"] input {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 비밀번호 인증
# ==========================================
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("  ")
    st.title("  ")
    password = st.text_input("비밀번호를 입력하세요", type="password")
    if password:
        if password == st.secrets["password"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("비밀번호가 틀렸습니다.")
    return False

if not check_password():
    st.stop()

# ==========================================
# 공유 리소스 로딩 (인증 후 1회만)
# ==========================================
import torch
import joblib
import numpy as np

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "jeju_energy.db")

@st.cache_resource
def get_db():
    return JejuEnergyDB(DB_PATH)

@st.cache_resource
def load_assets():
    print("[1/6] load_assets 시작!")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("[2/6] 메타데이터 및 스케일러 로딩 중...")
    metadata = joblib.load('models/metadata.pkl')
    scaler_solar = joblib.load('models/MinMax_scaler_solar.pkl')
    scaler_wind = joblib.load('models/MinMax_scaler_wind.pkl')
    scalers = {'solar': scaler_solar, 'wind': scaler_wind}

    pred_len = metadata['PRED_LEN']

    print("[3/6] 모델 초기화 중...")
    solar_model = PatchTST_Weather_Model(
        num_features=len(metadata['features_solar']),
        seq_len=metadata['SEQ_LEN_SOLAR'],
        pred_len=pred_len,
        patch_len=24, stride=12,
        d_model=256, num_heads=4, num_layers=3, d_ff=1024, dropout=0.2
    ).to(device)

    wind_model = PatchTST_Weather_Model(
        num_features=len(metadata['features_wind']),
        seq_len=metadata['SEQ_LEN_WIND'],
        pred_len=pred_len,
        patch_len=12, stride=6,
        d_model=128, num_heads=4, num_layers=2, d_ff=256, dropout=0.3
    ).to(device)

    print("[4/6] 태양광 모델 가중치 로딩 중...")
    solar_model.load_state_dict(torch.load('models/best_patchtst_solar_model.pth', map_location=device))

    print("[5/6] 풍력 모델 가중치 로딩 중...")
    wind_model.load_state_dict(torch.load('models/best_patchtst_wind_model.pth', map_location=device))

    solar_model.eval()
    wind_model.eval()

    print("[6/6] load_assets 완료!")
    return solar_model, wind_model, scalers, metadata, device

# session_state에 공유 리소스 저장
if 'shared_db' not in st.session_state:
    st.session_state['shared_db'] = get_db()
if 'shared_assets' not in st.session_state:
    st.session_state['shared_assets'] = load_assets()

# ==========================================
# 페이지 라우팅
# ==========================================
full_page = st.Page("pages/full.py", title="정식 버전", icon="🔧")
lite_page = st.Page("pages/lite.py", title="경량 버전", icon="📱")

pg = st.navigation([lite_page, full_page])
pg.run()
