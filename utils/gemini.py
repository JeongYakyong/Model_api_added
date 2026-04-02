import os
import math
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

def generate_energy_narrative(df, warn_low, warn_high, smp_threshold):
    if not os.getenv("GEMINI_API_KEY"):
        return "Gemini API key is missing."

    try:
        # 기상 및 부하 데이터 요약
        avg_cloud = df['total_cloud'].mean() if 'total_cloud' in df.columns else 0
        avg_wind = df['wind_spd'].mean() if 'wind_spd' in df.columns else 0
        avg_rain = df['rainfall'].mean() if 'rainfall' in df.columns else 0
        
        daytime_df = df[(df.index.hour >= 9) & (df.index.hour <= 16)]
        avg_solar = daytime_df['solar_rad'].mean() if not daytime_df.empty and 'solar_rad' in daytime_df.columns else 0

        # 이용률 데이터 추출
        avg_solar_util = daytime_df['est_Solar_Utilization'].mean() if not daytime_df.empty and 'est_Solar_Utilization' in daytime_df.columns else 0
        avg_wind_util = df['est_Wind_Utilization'].mean() if 'est_Wind_Utilization' in df.columns else 0

        min_net = df['est_net_demand'].min() if 'est_net_demand' in df.columns else 0
        max_net = df['est_net_demand'].max() if 'est_net_demand' in df.columns else 0

        # SMP 안전 추출 및 포맷팅 (NameError 방지)
        if 'smp_jeju' in df.columns and not df['smp_jeju'].dropna().empty:
            min_smp = df['smp_jeju'].min()
            smp_str = f"{min_smp:.1f}"
        else:
            min_smp = float('nan')
            smp_str = "데이터 없음"
        
    except Exception as e:
        return f"Data processing error: {e}"

    sys_instruct = (
        "당신은 제주 LNG터미널 운영 핵심 요약 전문가입니다. "
        "장황한 설명 없이 데이터 기반의 운영 지침을 4에서 5줄 내외로 요약하여 보고하십시오."
    )
    
    prompt = f"""
    [데이터]
    - 기상: 전운량 {avg_cloud:.1f}, 일사 {avg_solar:.2f}, 풍속 {avg_wind:.1f}m/s, 강수 {avg_rain:.2f}mm
    - SMP: 최저 {smp_str}원
    - 이용률: 태양광 {avg_solar_util:.2%}, 풍력 {avg_wind_util:.2%}
    - 순부하(Net Demand): 최저 {min_net:.1f}MW, 최대 {max_net:.1f}MW
    - 임계치: 저부하 기준 {warn_low}MW, 고부하 기준 {warn_high}MW, SMP 기준 {smp_threshold}원
    
    [작성 규칙]
    1. 전체 길이는 4~5줄로 제한하며, 반드시 '-' 또는 '•' 기호를 사용하는 글머리 기호(개조식) 형태로 작성하십시오. (숫자 번호 매기기 금지)
    2. 기상 정보는 '맑음/흐림', '비가 옴', '바람이 약함/강함', '일사 풍부/부족' 등 간단한 상태 위주로 첫 번째 항목에 기술하십시오.
    3. 태양광/풍력 이용률(Utilization) 동향을 두 번째 항목에 기술하십시오.
    4. 강수량({avg_rain:.2f}mm)이 0보다 큰 경우, 강수로 인하여 예측 정확도가 낮아질 수 있음을 추가로 기술하십시오.
    5. 순부하 및 SMP 데이터를 기반으로 한 LNG 발전 운영 방향을 세 번째 항목(필요시 네 번째까지)에 명시하십시오.
       - 최저 순부하가 저부하 임계치({warn_low}MW) 이하일 경우: "순부하 감소로 인해 LNG 발전의 정지 가능성이 높습니다."
       - SMP 가격이 기준치({smp_threshold}원) 이하일 경우: "(-) SMP로 친환경 발전량이 최대로 예상됩니다."
       - 최대 순부하가 고부하 임계치({warn_high}MW) 이상일 경우: "순부하 증가로 인해 LNG 발전량 증가가 예상됩니다."
       - 모든 데이터가 임계치 이내의 정상 범위일 경우: "LNG 발전 운영이 안정적으로 유지될 전망입니다."
    6. 명확한 설명 위주로 하되, "~입니다", "~습니다"와 같이 정중하고 부드러운 어조를 사용하십시오.
    """

    try:
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_instruct,
                temperature=0.2
            )
        )
        return response.text
    except Exception as e:
        return f"Generation failed: {e}"