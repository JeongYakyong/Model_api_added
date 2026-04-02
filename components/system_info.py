"""
pages/system_info.py
Option F : 시스템 안내 — app.py에서 분리된 README 페이지
"""
import streamlit as st


def render_system_info():
    st.title("📖 시스템 안내")
    st.caption("제주 재생에너지 및 전력 순부하 예측 대시보드의 사용법과 구조를 안내합니다.")

    tab_disclaimer, tab_overview, tab_menu, tab_model, tab_setup = st.tabs([
        "⚠️ 면책 고지",
        "🌟 시스템 개요",
        "📋 메뉴별 사용법",
        "🤖 예측 모델 안내",
        "🚀 실행 및 환경 설정",
    ])

    # ─── Tab 0: 면책 고지 ───
    with tab_disclaimer:
        _render_disclaimer()

    # ─── Tab 1: 시스템 개요 ───
    with tab_overview:
        _render_overview()

    # ─── Tab 2: 메뉴별 사용법 ───
    with tab_menu:
        _render_menu_guide()

    # ─── Tab 3: 예측 모델 안내 ───
    with tab_model:
        _render_model_info()

    # ─── Tab 4: 실행 및 환경 설정 ───
    with tab_setup:
        _render_setup()


# ============================================================
# 내부 헬퍼 (탭별 렌더링)
# ============================================================

def _render_disclaimer():
    st.subheader("⚠️ 면책 고지 (Disclaimer)")

    st.error(
        "**본 대시보드는 참고용 보조 도구이며, "
        "운영에 관한 최종 의사결정의 근거로 사용할 수 없습니다.**"
    )

    st.markdown("---")

    st.write("#### 1. 시스템의 목적 및 용도")
    st.write(
        "본 시스템은 제주 지역의 재생에너지 발전량 및 전력 순부하를 "
        "**예측·분석하기 위한 참고용 도구**입니다.\n\n"
        "제공되는 모든 예측 결과, 시각화, 통계 수치는 실제 계통 운영, 급전 지시, "
        "설비 투자, 전력 거래 등 **실무 의사결정을 대체하지 않습니다.**"
    )

    st.write("#### 2. 예측 정확도의 한계")
    st.write(
        "본 시스템에 탑재된 AI 예측 모델(PatchTST)은 과거 데이터를 기반으로 "
        "학습된 통계적 모델입니다.\n\n"
        "다음과 같은 상황에서 예측 정확도가 크게 저하될 수 있습니다."
    )
    st.write(
        "- **출력제어, 발전 유지 등 전력거래소의 의사결정**\n"
        "- 전력 계통의 구조적 변화 (정기점검, 송전선 차단 등)\n"
        "- 급격한 기상 변화 (태풍, 집중호우, 폭설 등 이상 기후)\n"
        "- 학습 데이터에 포함되지 않은 계절적 패턴이나 설비 변경\n"
        "- 입력 데이터(실측/예보)의 결측, 지연, 오류\n"
    )

    st.write("#### 3. 데이터 출처 및 신뢰성")
    st.write(
        "본 시스템은 전력거래소(KPX) 및 기상청(KMA)의 공개 API를 통해 "
        "데이터를 수집합니다.\n\n"
        "해당 기관의 API 장애, 데이터 지연, 형식 변경, 수치 정정 등으로 인해 "
        "수집된 데이터가 불완전하거나 부정확할 수 있습니다.\n\n"
        "본 시스템은 **외부 데이터의 정확성을 보증하지 않습니다.**"
    )

    st.write("#### 4. 면책 조항")
    st.write("본 시스템은 다음 사항에 대하여 어떠한 법적 책임도 지지 않습니다.")
    st.write(
        "- 예측 결과의 부정확성으로 인해 발생한 직접적·간접적 손실\n"
        "- 시스템 장애, 데이터 유실, API 중단 등으로 인한 서비스 불가\n"
        "- 본 시스템의 출력을 근거로 한 의사결정에 따른 재정적·운영적 손해\n"
        "- 제3자 서비스(KPX, KMA API 등)의 변경 또는 중단으로 인한 영향"
    )

    st.write("#### 5. 사용자의 책임")
    st.write(
        "본 시스템을 사용하는 모든 사용자는 위 내용을 충분히 이해하고 "
        "동의한 것으로 간주합니다.\n\n"
        "예측 결과를 실무에 활용할 경우, 반드시 **자체 검증 절차를 거치고 "
        "전문 인력의 판단을 병행**하여야 합니다.\n\n"
        "본 시스템의 사용으로 인해 발생하는 모든 결과에 대한 책임은 "
        "사용자 본인에게 있습니다."
    )

    st.markdown("---")
    st.caption("본 고지는 시스템 최초 배포일 기준으로 작성되었으며, 사전 통보 없이 변경될 수 있습니다.")
    st.caption("작성자 : 김범준")


def _render_overview():
    st.subheader("제주 재생에너지 및 전력 순부하 예측 대시보드")
    st.write(
        "기상청(KMA) 예보 데이터와 전력거래소(KPX) 데이터를 활용하여, "
        "**제주 지역의 태양광/풍력 발전 가동률을 예측**하고\n\n"
        "**전력 순부하(Net Demand) 및 경제성 지표(SMP)**를 모니터링하는 "
        "AI 대시보드입니다."
    )

    st.markdown("---")

    st.write("### 핵심 워크플로우")
    st.write(
        "이 시스템은 **데이터 수집 → 분석 → 예측 → 시각화 → 검증**의 "
        "순환 구조로 설계되었습니다. "
        "각 단계는 사이드바 메뉴(Option A~E)에 대응합니다."
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown("**Option A : DB 관리**\n\n`데이터 수집`")
    col2.markdown("**B. EDA**\n\n`탐색·분석`")
    col3.markdown("**C. 예측**\n\n`모델 추론`")
    col4.markdown("**D. 시각화**\n\n`결과 확인`")
    col5.markdown("**E. 검증**\n\n`정확도 평가`")

    st.markdown("---")

    st.write("### 기술 스택")
    col_fe, col_be, col_ai, col_data = st.columns(4)
    col_fe.markdown("**Frontend**\n\nStreamlit, Plotly")
    col_be.markdown("**Backend**\n\nPython, SQLite3")
    col_ai.markdown("**AI/ML**\n\nPyTorch (PatchTST)\n\nscikit-learn")
    col_data.markdown("**Data**\n\nPandas, NumPy\n\npvlib")

    st.markdown("---")

    st.write("### 데이터 출처")
    st.caption(
        "실측 발전량 및 전력수요, SMP 가격은 **KPX(전력거래소)** API에서, "
        "기상 관측/예보 데이터는 **KMA(기상청)** API에서 수집합니다."
    )
    st.caption(
        "기상 데이터는 남동쪽 표선-남원 데이터를 기준으로 하되, "
        "풍속과 풍향은 한림-한경지역 데이터를 참고합니다."
    )
    st.write(
        "**KPX(전력거래소)**\n\n"
        "- [한국전력거래소_계통한계가격 및 수요예측(하루전 발전계획용)]"
        "(https://www.data.go.kr/data/15131225/openapi.do)\n\n"
        "- [대국민 전력수급현황 공유 시스템]"
        "(https://openapi.kpx.or.kr/smp_day_avg.do)\n\n"
        "**KMA(기상청)**\n\n"
        "- [기상청_지상(종관, ASOS) 일자료 조회서비스]"
        "(https://www.data.go.kr/data/15057210/openapi.do)\n\n"
        "- [한국형수치예보모델(KIM) 자료 조회]"
        "(https://apihub.kma.go.kr/)"
    )


def _render_menu_guide():
    st.subheader("메뉴별 사용법")
    st.caption("각 메뉴를 펼쳐서 상세 사용법을 확인하세요.")

    with st.expander("**Option A : DB 관리**", expanded=False):
        st.write("#### Data Status")
        st.write(
            "DB에 저장된 전체 실측 데이터의 무결성을 점검합니다.\n\n"
            "시계열 누락(빠진 시간대), 컬럼별 결측치, 주요 컬럼의 불완전 행을 "
            "3가지 관점에서 검사하며, API로 채워지지 않는 결측이라면 \n\n"
            "**시간 비례 보간(최대 3건 연속)**을 적용할 수 있습니다."
        )
        st.write("#### API 데이터 수집")
        st.write(
            "시작일/종료일을 지정하여 KPX(발전량, SMP) 및 KMA(기상) 데이터를 "
            "수집합니다.\n\n"
            "실측 데이터는 최대 30일, Forecast 데이터는 과거 3일 ~ 미래 1일 "
            "범위로 수집 가능합니다."
        )
        st.info("💡 Forecast 자료는 전날 23시에 업로드됩니다. 매일 자정 이후 수집을 권장합니다.")
        st.write("#### 데이터 조회")
        st.write("실측/Forecast 테이블을 기간별로 조회할 수 있습니다.")
        st.write("#### CSV 업로드")
        st.write(
            "초기 셋팅이나 DB 복구 시, 과거 CSV 파일을 일괄 적재합니다. "
            "업로드 시 `timestamp` 컬럼이 반드시 포함되어야 하며, \n\n"
            "파생 변수(Solar_Capacity_Est 등)가 자동 계산됩니다."
        )
        st.caption("⚠️ LNG, HVDC, 기력 발전량은 API 실시간 수집이 불가합니다. 전력거래소 CSV를 별도 다운로드해 주세요.")
        st.caption("LNG, HVDC, 기력 발전량 데이터는 현재 20.01.01 - 24.12.31 까지 준비되어 있습니다.")

    with st.expander("**Option B : 데이터 분석 (EDA)**", expanded=False):
        st.write(
            "DB에 저장된 실측 데이터를 대상으로 탐색적 데이터 분석을 수행합니다.\n\n "
            "상단에서 분석할 피처를 선택하고, 하단에서 조회 기간을 설정하면 "
            "4개 탭에서 각각 다른 시각화를 확인할 수 있습니다."
        )
        st.write(
            "- **시계열 데이터**: 선택한 피처의 시간 흐름 그래프 (드래그로 X축 확대 가능)\n"
            "- **통계 요약**: describe() 기반의 기초 통계량\n"
            "- **상관관계 히트맵**: 피처 간 상관계수 행렬 (최소 2개 피처 필요)\n"
            "- **산점도**: X/Y축 피처를 선택하여 분포 확인 (히스토그램 포함)"
        )

    with st.expander("**Option C : 발전량 예측**", expanded=False):
        st.write("#### 예측 흐름")
        st.write(
            "1. 예측 대상 날짜를 선택합니다.\n"
            "2. 시스템이 자동으로 입력 데이터 상태를 점검합니다 "
            "(과거 실측 336시간 + 미래 예보 24시간).\n"
            "3. 데이터가 충분하면 [예측 실행] 버튼으로 모델을 구동합니다.\n"
            "4. 예측된 가동률이 Forecast 테이블에 저장됩니다."
        )
        st.warning(
            "⚠️ 과거 실측 336시간 + 미래 예보 24시간이 모두 채워져 있어야 "
            "예측이 가능합니다.\n\n"
            "부족하면 상세 안내에 따라 [Option A : DB 관리]에서 데이터를 보충해 주세요."
        )

    with st.expander("**Option D : 예측 결과 시각화**", expanded=False):
        st.write(
            "Option C에서 예측한 결과를 바탕으로 발전량, 순부하(Net Demand), "
            "SMP 등을 시각화합니다."
        )
        st.write("#### 경고 구간 설정")
        st.write(
            "[데이터 선택 / 경고 설정] 탭에서 경고 임계값을 조절할 수 있습니다. "
            "est_net_demand 기준으로 저발전/고발전 구간이 차트에 음영 처리됩니다."
        )
        st.write(
            "- **저발전 경고**: LNG 발전량이 지나치게 적은 구간 (기본 250MW 이하)\n"
            "- **고발전 경고**: LNG 발전량이 지나치게 높은 구간 (기본 750MW 이상)\n"
            "- **SMP 하한 경고**: 제주 SMP가 설정값 이하로 떨어지는 구간\n"
            "- **추가 경고**: 최저발전/최대발전 임계값을 별도 활성화 가능"
        )

    with st.expander("**Option E : 예측 정확도 검증**", expanded=False):
        st.write("모델의 예측 결과와 실제 발전량을 비교하여 정확도를 평가합니다.")
        st.write(
            "- **일간 비교**: 특정 날짜를 선택하여 24시간 단위로 실제 vs 예측 비교\n"
            "- **기간별 정확도 평가**: 선택한 기간 동안의 RMSE/MAE 산출 및 "
            "산점도(y=x 대각선 기준) 확인\n"
            "- **예보 Bias 분석**: 기상 예보의 체계적 편향을 분석"
        )
        st.write("#### 평가지표 해석")
        st.write(
            "- **RMSE** (Root Mean Squared Error): 큰 오차에 패널티를 부여하여 "
            "극단적 예측 실패를 감지\n"
            "- **MAE** (Mean Absolute Error): 실제 발전량과 평균적으로 몇 MW "
            "차이가 나는지 직관적으로 표현"
        )


def _render_model_info():
    st.subheader("예측 모델 구조")

    st.write("#### PatchTST + Weather Attention")
    st.write(
        "본 시스템의 예측 모델은 **PatchTST**(Patch Time Series Transformer) "
        "아키텍처에 **Weather Attention** 메커니즘을 결합한 구조입니다.\n\n"
        "시계열 데이터를 패치(patch) 단위로 분할하여 Transformer Encoder에 입력하고, "
        "미래 예보와 과거 기상 간의 어텐션을 통해 발전량을 예측합니다."
    )

    st.markdown("---")

    col_input, col_output = st.columns(2)
    with col_input:
        st.write("#### 입력 데이터")
        st.write(
            "- **태양광발전** : 과거 실측 336시간 태양광발전량 + 기상 관측치\n"
            "- **풍력발전** : 과거 실측 72시간 풍력발전량 + 기상 관측치\n"
            "- **공통** : 미래 기상 예보 24시간 (1일)"
        )
    with col_output:
        st.write("#### 출력 데이터")
        st.write(
            "- **태양광 가동률** (est_Solar_Utilization): 0~1 범위\n"
            "- **풍력 가동률** (est_Wind_Utilization): 0~1 범위\n"
            "- 가동률 × 설비용량 = 예측 발전량(MW)"
        )

    st.markdown("---")

    # ✅ 모델 스펙: load_assets() 실제 값과 일치하도록 수정
    st.write("#### 모델 상세 스펙")

    spec_col1, spec_col2 = st.columns(2)
    with spec_col1:
        st.write("**태양광 모델**")
        st.write(
            "- seq_len: 336, pred_len: 24\n"
            "- d_model: 128\n"
            "- num_heads: 4, num_layers: 3\n"
            "- d_ff: 512\n"
            "- patch_len: 24, stride: 12\n"
            "- dropout: 0.2"
        )
    with spec_col2:
        st.write("**풍력 모델**")
        st.write(
            "- seq_len: 72, pred_len: 24\n"
            "- d_model: 128\n"
            "- num_heads: 4, num_layers: 2\n"
            "- d_ff: 256\n"
            "- patch_len: 12, stride: 6\n"
            "- dropout: 0.3"
        )

    st.caption("두 모델 모두 Scaler는 MinMax Scaler를 사용하였습니다.")

    st.markdown("---")

    st.write("#### 핵심 구성 요소")

    with st.expander("Patch Embedding + Positional Encoding"):
        st.write(
            "입력 시계열(336시간 및 72시간)을 patch_len 크기의 패치로 분할합니다. \n\n"
            "각 패치는 Linear 레이어를 통해 d_model 차원으로 임베딩되고, "
            "학습 가능한 Positional Encoding이 더해져 시간 순서 정보를 유지합니다."
        )

    with st.expander("Transformer Encoder"):
        st.write(
            "Multi-Head Self-Attention과 Feed-Forward Network로 구성된 "
            "Encoder Layer를 여러 층 쌓아 패치 간의 시간적 의존성을 학습합니다.\n\n "
            "norm_first=True (Pre-Norm) 구조를 사용하여 학습 안정성을 확보했습니다."
        )

    with st.expander("Weather Attention (핵심 차별점)"):
        st.write(
            "미래 기상 예보(24시간)를 Query로, 과거 기상 패치를 Key로 사용하여 "
            "\"미래 날씨와 가장 유사했던 과거 구간\"을 찾아냅니다. \n\n"
            "해당 구간의 Transformer 출력(발전 패턴)을 가중 합산하여 "
            "미래 발전량 예측의 컨텍스트로 활용합니다."
        )

    with st.expander("Regressor (최종 예측)"):
        st.write(
            "Weather Attention의 출력과 미래 기상 예보 벡터를 결합하여 "
            "2-Layer MLP(LeakyReLU + Dropout)로 최종 24시간 가동률을 출력합니다."
        )


def _render_setup():
    st.subheader("실행 및 환경 설정")

    st.write("#### 1. 실행")
    st.write("내부 서버 구동의 어려움으로 외부 서버로 접속하고 있습니다.")
    st.write("접속링크는 매번 달라질 수 있습니다.")

    st.markdown("---")

    st.write("#### 2. 프로젝트 구조")
    st.code(
        "jeju_energy_project/\n"
        "├── app.py                          # 메인 Streamlit 실행 파일\n"
        "├── requirements.txt                # 패키지 의존성\n"
        "├── .env                            # API Key (보안, git 미추적)\n"
        "├── database/\n"
        "│   └── jeju_energy.db              # SQLite 데이터베이스\n"
        "├── models/\n"
        "│   ├── best_patchtst_solar_model.pth   # 태양광 모델 가중치\n"
        "│   ├── best_patchtst_wind_model.pth    # 풍력 모델 가중치\n"
        "│   ├── metadata.pkl                    # 모델 메타데이터\n"
        "│   ├── architecture.py                 # 모델 클래스 정의\n"
        "│   └── MinMax_scaler_*.pkl             # 스케일러\n"
        "├── pages/\n"
        "│   └── system_info.py             # 시스템 안내 (Option F)\n"
        "└── utils/\n"
        "    ├── api_fetchers.py             # KMA/KPX API 수집 모듈\n"
        "    ├── data_pipeline.py            # 전처리 및 추론 파이프라인\n"
        "    ├── chart_helpers.py            # 공통 차트/헬퍼 함수\n"
        "    └── db_manager.py               # DB 연결 및 쿼리 관리",
        language="text",
    )

    st.markdown("---")

    st.write("#### 3. 초기 데이터 셋업 순서")
    st.write("처음 실행 시 DB가 비어있으므로 아래 순서로 데이터를 채워야 합니다.")
    st.write(
        "1. **[Option A : DB 관리 → CSV 업로드]** 에서 과거 CSV 파일을 적재하거나\n"
        "2. **[Option A : DB 관리 → API 데이터 수집]** 에서 실측 데이터(최소 14일분)를 수집\n"
        "3. **[Option A : DB 관리 → API 데이터 수집]** 에서 Forecast 데이터(대상일)를 수집\n"
        "4. **[Option A : DB 관리 → Data Status]** 에서 결측치 확인 및 보간\n"
        "5. **[Option C : 발전량 예측]** 에서 예측 실행"
    )
    st.info("💡 이후 운영 시에는 매일 자정 이후 Forecast를 수집하고 예측을 실행하는 루틴을 권장합니다.")