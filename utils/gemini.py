import os
import json
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

from utils.chart_helpers import SMP_MIN_THRESHOLD

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

def render_briefing_expander(df, warn_low, warn_high, vis_date,
                              btn_key="btn_ai_briefing", title="✨ AI 예측 브리핑"):
    """AI 예측 브리핑 expander를 렌더링합니다."""
    with st.expander(title, expanded=True):
        date_key = str(vis_date)

        if 'lite_briefings_storage' not in st.session_state:
            st.session_state['lite_briefings_storage'] = load_briefings_from_file()

        saved_briefing = st.session_state['lite_briefings_storage'].get(date_key)

        col_btn, col_legend = st.columns([1, 2])
        with col_btn:
            do_generate = st.button("AI 브리핑 생성 / 갱신", key=btn_key)
        with col_legend:
            st.caption("시간대 구분 · 00-06 심야 · 06-12 오전 · 12-18 오후 · 18-23 야간")
        if do_generate:
            with st.spinner("AI가 데이터를 분석하고 있습니다..."):
                briefing_text = generate_energy_narrative(
                    df=df,
                    warn_low=warn_low,
                    warn_high=warn_high,
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


def _detect_risks(df, warn_low, warn_high):
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

        if st.session_state.get('warn_min_enabled', True):
            warn_min = st.session_state.get('warn_min', 100)
            cond = df['est_net_demand'] < warn_min
            if 'smp_jeju' in df.columns:
                cond = cond | (df['smp_jeju'] < SMP_MIN_THRESHOLD)
            mn = df[cond]
            if not mn.empty:
                events.append(
                    f"최저발전 경고: {mn.index.hour.min():02d}~{mn.index.hour.max():02d}시, "
                    f"최저 순부하 {mn['est_net_demand'].min():.0f}MW ({len(mn)}시간, "
                    f"임계 {warn_min}MW 또는 SMP<{SMP_MIN_THRESHOLD}원)"
                )

        if st.session_state.get('warn_overnight_enabled', True):
            warn_overnight = st.session_state.get('warn_overnight', 300)
            overnight = df[(df.index.hour < 6) & (df['est_net_demand'] < warn_overnight)]
            if not overnight.empty:
                events.append(
                    f"심야 저부하(00-06시): {overnight.index.hour.min():02d}~{overnight.index.hour.max():02d}시, "
                    f"최저 {overnight['est_net_demand'].min():.0f}MW ({len(overnight)}시간, 임계 {warn_overnight}MW)"
                )

    return events


def generate_energy_narrative(df, warn_low, warn_high):
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
        risks = _detect_risks(df, warn_low, warn_high)
        #if solar_clipped:
        #    risks.append(
        #        f"저일사 후처리: 일 최대 일사량 {max_solar_rad:.2f} MJ/m2, "
        #        f"태양광 예측 이용률 최대 {max_clip_pct:.0f}%로 압축됨"
        #    )
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
- 임계치: 저부하 {warn_low}MW, 고부하 {warn_high}MW, 최저발전 {st.session_state.get('warn_min', 100)}MW (또는 SMP<{SMP_MIN_THRESHOLD}원), 심야(00-06) {st.session_state.get('warn_overnight', 300)}MW
- 임계 교차 여부: 최저 순부하 {min_net:.1f}MW {"<" if min_net < warn_low else "≥"} 저부하 {warn_low}MW / 최대 순부하 {max_net:.1f}MW {">" if max_net > warn_high else "≤"} 고부하 {warn_high}MW

[감지된 리스크]
{risk_str}

[작성 규칙]
⚠ 절대 원칙: [감지된 리스크]에 명시된 이벤트만 언급하십시오. 거기에 없는 리스크(예: '고부하')를 평균·추세·증감으로부터 추론하거나 창작하지 마십시오. 해당 이벤트가 '정상 범위'면 반드시 '정상'으로만 서술하십시오.

1. 최대 5줄, 각 항목 '•' 기호 + 두 번 줄바꿈(\\n\\n) 개조식.
2. 첫 항목: 시간대별 기상 흐름 요약 ('오전 흐림→오후 비', '종일 맑고 바람 약함' 등 상태 위주).
3. 둘째 항목: 태양광/풍력 이용률 동향.
4. 강수가 0보다 크면 강수 시간대와 예측 정확도 저하 가능성 언급.
5. 셋째~넷째 항목: [감지된 리스크] 중 **주간(06-18시)** 시간대에 해당하는 이벤트에만 기반해 LNG 운영 방향을 작성. 규칙:
   - [감지된 리스크]에 '저부하' → "순부하 감소로 LNG 발전 정지 가능성"
   - [감지된 리스크]에 '최저발전 경고' → "(-) SMP 또는 저순부하로 LNG 정지·태양광 최대 발전 예상"
   - [감지된 리스크]에 '고부하' → "순부하 증가로 LNG 발전량 증가 예상"
   - 주간에 해당하는 위 이벤트가 **하나도 없으면** → "주간(오전·오후) 순부하 정상 범위, LNG 안정 운영 전망" (이 경우 반드시 한 줄로 작성하고 다른 리스크를 만들지 말 것).
6. 다섯번째 항목: [감지된 리스크] 중 **심야(00-06) 또는 야간(18-23)** 에 해당하는 이벤트 기반 LNG 운영 방향.
   - '심야 저부하' 감지 시 → "심야 LNG 정지/최소 출력 유지 검토 필요"
   - 심야·야간 해당 리스크가 없으면 → "심야·야간 순부하 정상 범위, LNG 안정 운영 전망"
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