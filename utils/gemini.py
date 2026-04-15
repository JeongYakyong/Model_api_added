import os
import json
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


# ==========================================
# 브리핑 저장/로드
# ==========================================
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

def render_briefing_expander(df, warn_low, warn_high, smp_threshold, vis_date,
                              btn_key="btn_ai_briefing", title="✨ AI 예측 브리핑"):
    """AI 예측 브리핑 expander를 렌더링합니다."""
    with st.expander(title, expanded=True):
        date_key = str(vis_date)

        if 'lite_briefings_storage' not in st.session_state:
            st.session_state['lite_briefings_storage'] = load_briefings_from_file()

        saved_briefing = st.session_state['lite_briefings_storage'].get(date_key)

        if st.button("AI 브리핑 생성 / 갱신", key=btn_key):
            with st.spinner("AI가 데이터를 분석하고 있습니다..."):
                briefing_text = generate_energy_narrative(
                    df=df,
                    warn_low=warn_low,
                    warn_high=warn_high,
                    smp_threshold=smp_threshold
                )
                save_briefing_to_file(date_key, briefing_text)
                st.session_state['lite_briefings_storage'][date_key] = briefing_text
                st.rerun()

        if saved_briefing:
            st.markdown(saved_briefing)
        else:
            st.caption("위 버튼을 눌러 해당 날짜의 브리핑을 생성하세요. 생성된 내용은 로컬에 자동 저장됩니다.")


def _time_block_summary(df, col):
    """시간대별 블록 평균 요약 (심야/오전/오후/야간)"""
    blocks = {
        '심야(00-06)': (0, 5),
        '오전(06-12)': (6, 11),
        '오후(12-18)': (12, 17),
        '야간(18-23)': (18, 23),
    }
    if col not in df.columns:
        return "데이터 없음"
    parts = []
    for label, (h0, h1) in blocks.items():
        sub = df[(df.index.hour >= h0) & (df.index.hour <= h1)]
        if sub.empty or sub[col].isna().all():
            continue
        parts.append(f"{label} {sub[col].mean():.1f}")
    return " → ".join(parts) if parts else "데이터 없음"


def _detect_risks(df, warn_low, warn_high, smp_threshold):
    """임계치 위반 시간대 추출 — LLM 대신 코드에서 확정"""
    events = []

    if 'est_net_demand' in df.columns:
        low = df[df['est_net_demand'] < warn_low]
        if not low.empty:
            events.append(
                f"저부하: {low.index.hour.min():02d}~{low.index.hour.max():02d}시, "
                f"최저 {low['est_net_demand'].min():.0f}MW ({len(low)}시간)"
            )
        high = df[df['est_net_demand'] > warn_high]
        if not high.empty:
            events.append(
                f"고부하: {high.index.hour.min():02d}~{high.index.hour.max():02d}시, "
                f"최대 {high['est_net_demand'].max():.0f}MW ({len(high)}시간)"
            )

    if 'smp_jeju' in df.columns:
        smp_low = df[df['smp_jeju'] <= smp_threshold]
        if not smp_low.empty:
            events.append(
                f"SMP 하락: {smp_low.index.hour.min():02d}~{smp_low.index.hour.max():02d}시, "
                f"최저 {smp_low['smp_jeju'].min():.1f}원 ({len(smp_low)}시간)"
            )

    return events


def generate_energy_narrative(df, warn_low, warn_high, smp_threshold):
    if not os.getenv("GEMINI_API_KEY"):
        return "Gemini API key is missing."

    try:
        # ── 기상 ──
        avg_cloud = df['total_cloud'].mean() if 'total_cloud' in df.columns else 0
        avg_wind = df['wind_spd'].mean() if 'wind_spd' in df.columns else 0
        avg_rain = df['rainfall'].mean() if 'rainfall' in df.columns else 0

        daytime = df[(df.index.hour >= 9) & (df.index.hour <= 16)]
        avg_solar = daytime['solar_rad'].mean() if not daytime.empty and 'solar_rad' in daytime.columns else 0

        # ── 이용률 ──
        avg_solar_util = daytime['est_Solar_Utilization'].mean() if not daytime.empty and 'est_Solar_Utilization' in daytime.columns else 0
        avg_wind_util = df['est_Wind_Utilization'].mean() if 'est_Wind_Utilization' in df.columns else 0

        # ── 순부하 & SMP ──
        min_net = df['est_net_demand'].min() if 'est_net_demand' in df.columns else 0
        max_net = df['est_net_demand'].max() if 'est_net_demand' in df.columns else 0

        if 'smp_jeju' in df.columns and not df['smp_jeju'].dropna().empty:
            smp_str = f"{df['smp_jeju'].min():.1f}"
        else:
            smp_str = "데이터 없음"

        # ── 시간대별 흐름 ──
        cloud_flow = _time_block_summary(df, 'total_cloud')
        rain_flow = _time_block_summary(df, 'rainfall')
        wind_flow = _time_block_summary(df, 'wind_spd')
        net_flow = _time_block_summary(df, 'est_net_demand')

 #       # ── 태양광 후처리 감지 (solar_rad 기반 재계산) ──
 #       SOLAR_RAD_THRESHOLD = 0.85
 #       SOLAR_CLIP_POWER = 2.0
 #       max_solar_rad = df['solar_rad'].max() if 'solar_rad' in df.columns else 999
 #       solar_clipped = max_solar_rad < SOLAR_RAD_THRESHOLD
 #       if solar_clipped:
 #           max_clip_pct = (max_solar_rad / SOLAR_RAD_THRESHOLD) ** SOLAR_CLIP_POWER * 100
 #           clip_info = (
 #               f"  ⚠ 저일사 후처리 적용: 일 최대 일사량 {max_solar_rad:.2f} MJ/m2 "
 #               f"(기준 {SOLAR_RAD_THRESHOLD} 미만), "
 #               f"태양광 이용률 최대 {max_clip_pct:.0f}%로 압축"
 #           )
 #       else:
 #           clip_info = ""

        # ── 리스크 감지 (코드 확정) ──
        risks = _detect_risks(df, warn_low, warn_high, smp_threshold)
        if solar_clipped:
            risks.append(
                f"저일사 후처리: 일 최대 일사량 {max_solar_rad:.2f} MJ/m2, "
                f"태양광 예측 이용률 최대 {max_clip_pct:.0f}%로 압축됨"
            )
        risk_str = "\n".join(f"  ⚠ {r}" for r in risks) if risks else "정상 범위"

    except Exception as e:
        return f"Data processing error: {e}"

    sys_instruct = (
        "당신은 제주 LNG터미널 운영 핵심 요약 전문가입니다. "
        "장황한 설명 없이 데이터 기반의 운영 지침을 4~5줄 내외로 요약하여 보고하십시오."
    )

    prompt = f"""[데이터]
- 기상: 전운량 {avg_cloud:.1f}, 일사 {avg_solar:.2f}, 풍속 {avg_wind:.1f}m/s, 강수 {avg_rain:.2f}mm
- 시간대별 전운량: {cloud_flow}
- 시간대별 강수: {rain_flow}
- 시간대별 풍속: {wind_flow}
- SMP: 최저 {smp_str}원
- 이용률: 태양광 {avg_solar_util:.2%}, 풍력 {avg_wind_util:.2%}
- 순부하: 최저 {min_net:.1f}MW, 최대 {max_net:.1f}MW
- 시간대별 순부하: {net_flow}
- 임계치: 저부하 {warn_low}MW, 고부하 {warn_high}MW, SMP {smp_threshold}원

[감지된 리스크]
{risk_str}

[작성 규칙]
1. 최대 5줄, 각 항목 '•' 기호 + 두 번 줄바꿈(\\n\\n) 개조식.
2. 첫 항목: 시간대별 기상 흐름 요약 ('오전 흐림→오후 비', '종일 맑고 바람 약함' 등 상태 위주).
3. 둘째 항목: 태양광/풍력 이용률 동향.
4. 강수가 0보다 크면 강수 시간대와 예측 정확도 저하 가능성 언급.
5. 셋째~넷째 항목: [감지된 리스크]에 기반한 주간(오전, 오후) LNG 운영 방향.
   - 저부하 → "순부하 감소로 LNG 발전 정지 가능성"
   -  SMP 하락 → "(-) SMP로 태양광 발전량 최대 예상"
   -  고부하 → "순부하 증가로 LNG 발전량 증가 예상"
   -  정상 → "LNG 발전 운영 안정적 유지 전망"
6. 다섯번째 항목 : [감지된 리스크]와 시간대별 순부하에 기반한 야간 LNG 운영 방향.
7. "~입니다", "~습니다" 경어체.
"""
#7. 여섯번째 항목 : 저일사 후처리가 적용된 경우에만 해당, 둘째 항목에 "저일사 후처리 적용(최대 N%)" 문구를 반드시 포함.

    try:
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_instruct,
                temperature=0.2,
            )
        )
        return response.text
    except Exception as e:
        return f"Generation failed: {e}"