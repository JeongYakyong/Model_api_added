# -*- coding: utf-8 -*-
"""위험구간 음영 — 순부하 경고 밴드(vrect)·임계값 편집.

jeju_model pages/chart_warn.py 를 이식(원래 이 프로젝트의 옛 chart_helpers.py 에서 jeju_model 로
건너갔던 로직이 다시 돌아온 것). SMP 관련 조건(음수가격 경보 등)은 이 앱에 SMP 데이터가 없어 제외.

★ 규약: draw_warning_zones 의 df 는 **DatetimeIndex** 여야 한다 (심야 판정에 df.index.hour 사용).
  timestamp 컬럼 프레임을 그대로 넘기면 조용히 오동작하므로 호출부에서 반드시
  ``df.set_index("timestamp")`` 후 전달할 것.

경고는 항상 **예측(est_net_load)** 기준이다 — 미래 구간도 판정할 수 있어야 하고, 과거 구간도
"그때 무엇이 경고됐었는가"를 보는 게 실측보다 유용하다(jeju_model 과 동일 관례).

우선순위 (높음→낮음): 최저발전(Min) > 최대발전(Max) > 심야 저부하(Overnight) > 저발전/고발전.
각 시각은 활성화된 가장 높은 우선순위의 경고 한 개만 음영 표시된다(상호배타).
기본 임계값(100/250/300/750/900MW)은 제주 순 부하 스케일 기준(jeju_model 과 동일).
"""
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

NET_COL = "est_net_load"

BAND_COLORS = {
    "min":       "#d03b3b",   # 최저발전 (가장 심각)
    "overnight": "#fab219",   # 심야 저부하
    "low":       "#fab219",   # 저발전 (심야와 같은 계열 — 우선순위 배타라 동시 표시 없음)
    "high":      "#2a78d6",   # 고발전
    "max":       "#8b5cf6",   # 최대발전 (기본 OFF)
}

_WARN_DEFAULTS = {
    'warn_low':               250,
    'warn_high':              750,
    'warn_min_enabled':       True,
    'warn_min':               100,
    'warn_max_enabled':       False,
    'warn_max':               900,
    'warn_overnight_enabled': True,
    'warn_overnight':         300,
}

OVERNIGHT_END_HOUR = 6   # 심야 경고 적용 구간: hour < OVERNIGHT_END_HOUR (00:00 ~ 05:59)


def init_warning_state():
    """페이지 진입 시 경고 임계값 session_state 기본값 초기화."""
    for k, v in _WARN_DEFAULTS.items():
        st.session_state.setdefault(k, v)


def _subtract_intervals(intervals, exclude):
    """intervals 목록에서 exclude 목록에 해당하는 구간을 잘라낸다."""
    result = []
    for s, e in intervals:
        remaining = [(s, e)]
        for xs, xe in sorted(exclude):
            clipped = []
            for rs, re in remaining:
                if xe <= rs or xs >= re:
                    clipped.append((rs, re))
                elif xs <= rs and xe >= re:
                    pass
                elif xs <= rs:
                    clipped.append((xe, re))
                elif xe >= re:
                    clipped.append((rs, xs))
                else:
                    clipped.append((rs, xs))
                    clipped.append((xe, re))
            remaining = clipped
        result.extend(remaining)
    return result


def draw_danger_zones(fig, df, condition_series, fill_color,
                      annotation_text=None, show_legend_label=None,
                      layer_pos="below", fill_opacity=0.15,
                      legend_ref='legend', padding_hours=1.0,
                      exclude_intervals=None):
    """Plotly figure에 위험 구간 음영(vrect)을 추가하는 헬퍼.

    반환값: 패딩 적용 후 병합된 구간 리스트 (상위 우선순위 구간 exclusion 전달용).
    df 는 DatetimeIndex 가정(연속 시간 그룹핑에 index 사용).
    """
    if not condition_series.any():
        return []

    danger_df = df[condition_series].copy()
    danger_df['group'] = (condition_series != condition_series.shift()).cumsum()
    danger_df['temp_time'] = danger_df.index

    danger_zones = danger_df.groupby('group').agg(
        start=('temp_time', 'min'),
        end=('temp_time', 'max')
    )

    pad = timedelta(hours=padding_hours)
    raw_intervals = sorted(
        [(row['start'] - pad, row['end'] + pad) for _, row in danger_zones.iterrows()]
    )
    merged = []
    for s, e in raw_intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    draw_ivs = _subtract_intervals(merged, exclude_intervals) if exclude_intervals else merged

    for start_time, end_time in draw_ivs:
        fig.add_vrect(
            x0=start_time, x1=end_time,
            fillcolor=fill_color, opacity=fill_opacity,
            layer=layer_pos, line_width=0,
        )

    if show_legend_label:
        trace_kwargs = dict(
            x=[None], y=[None], mode='markers',
            marker=dict(size=12, color=fill_color, symbol='square'),
            opacity=fill_opacity,
            name=show_legend_label, showlegend=True,
        )
        if legend_ref and legend_ref != 'legend':
            trace_kwargs['legend'] = legend_ref
        fig.add_trace(go.Scatter(**trace_kwargs))

    return merged


def draw_warning_zones(fig, df):
    """예측 차트에 경고 음영을 일괄 표시 (우선순위 기반 상호배타 마스크).

    ★ df 는 DatetimeIndex 프레임 — 호출부에서 set_index("timestamp") 필수(모듈 docstring 참고).

    - 최저발전 (기본 ON, 빨강)  : NET_COL < warn_min
    - 최대발전 (기본 OFF, 보라) : NET_COL > warn_max
    - 심야 저부하 (기본 ON, 금색): hour < 6  AND NET_COL < warn_overnight
    - 저발전 (기본 ON, 금색)    : NET_COL < warn_low
    - 고발전 (기본 ON, 파랑)    : NET_COL > warn_high
    """
    init_warning_state()

    if NET_COL not in df.columns:
        return

    nd = df[NET_COL]
    false_mask = pd.Series(False, index=df.index)

    low_raw = nd < st.session_state.get('warn_low', _WARN_DEFAULTS['warn_low'])
    high_raw = nd > st.session_state.get('warn_high', _WARN_DEFAULTS['warn_high'])

    min_raw = false_mask
    if st.session_state.get('warn_min_enabled', _WARN_DEFAULTS['warn_min_enabled']):
        min_raw = nd < st.session_state.get('warn_min', _WARN_DEFAULTS['warn_min'])

    max_raw = false_mask
    if st.session_state.get('warn_max_enabled', _WARN_DEFAULTS['warn_max_enabled']):
        max_raw = nd > st.session_state.get('warn_max', _WARN_DEFAULTS['warn_max'])

    overnight_raw = false_mask
    if st.session_state.get('warn_overnight_enabled', _WARN_DEFAULTS['warn_overnight_enabled']):
        hour_mask = pd.Series(df.index.hour < OVERNIGHT_END_HOUR, index=df.index)
        overnight_raw = hour_mask & (nd < st.session_state.get('warn_overnight',
                                                               _WARN_DEFAULTS['warn_overnight']))

    # ── 우선순위 배타 처리 ──
    min_mask = min_raw
    max_mask = max_raw & ~min_mask
    overnight_mask = overnight_raw & ~(min_mask | max_mask)
    priority_mask = min_mask | max_mask | overnight_mask
    low_mask = low_raw & ~priority_mask
    high_mask = high_raw & ~priority_mask

    opacity = 0.25
    min_ivs = draw_danger_zones(fig, df, min_mask, BAND_COLORS['min'],
                                show_legend_label='최저발전',
                                fill_opacity=opacity, legend_ref='legend2', padding_hours=1.0)
    overnight_ivs = draw_danger_zones(fig, df, overnight_mask, BAND_COLORS['overnight'],
                                      show_legend_label='심야 저부하',
                                      fill_opacity=opacity, legend_ref='legend2',
                                      exclude_intervals=min_ivs)
    draw_danger_zones(fig, df, low_mask, BAND_COLORS['low'],
                      show_legend_label='저발전',
                      fill_opacity=opacity, legend_ref='legend2',
                      exclude_intervals=min_ivs + overnight_ivs)
    max_ivs = draw_danger_zones(fig, df, max_mask, BAND_COLORS['max'],
                                show_legend_label='최대발전',
                                fill_opacity=opacity, legend_ref='legend2')
    draw_danger_zones(fig, df, high_mask, BAND_COLORS['high'],
                      show_legend_label='고발전',
                      fill_opacity=opacity, legend_ref='legend2',
                      exclude_intervals=max_ivs)

    if any(m.any() for m in [min_mask, low_mask, max_mask, high_mask, overnight_mask]):
        fig.update_layout(
            legend2=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
        )


def render_warning_threshold_inputs():
    """경고 임계값 입력 위젯 세트 (popover 안에서 호출).

    위젯에 key 를 붙이지 않고 **잠정값 dict 를 반환** — 호출측이 '적용' 버튼에서
    `commit_warning_thresholds(values)` 로 session_state 에 일괄 반영한다.
    """
    init_warning_state()

    w_low = st.number_input(
        "🟡 저발전 경고 (MW)",
        value=int(st.session_state.get('warn_low', _WARN_DEFAULTS['warn_low'])), step=10,
    )
    w_high = st.number_input(
        "🔵 고발전 경고 (MW)",
        value=int(st.session_state.get('warn_high', _WARN_DEFAULTS['warn_high'])), step=10,
    )

    w_overnight_on = st.checkbox(
        "🌙 심야 저부하 경고 활성화 (00-06시)",
        value=bool(st.session_state.get('warn_overnight_enabled',
                                        _WARN_DEFAULTS['warn_overnight_enabled'])),
    )
    w_overnight = int(st.session_state.get('warn_overnight', _WARN_DEFAULTS['warn_overnight']))
    if w_overnight_on:
        w_overnight = st.number_input("심야 순부하 임계값 (MW)", value=w_overnight, step=10)

    w_min_on = st.checkbox(
        "🔴 최저발전 경고 활성화",
        value=bool(st.session_state.get('warn_min_enabled', _WARN_DEFAULTS['warn_min_enabled'])),
    )
    w_min = int(st.session_state.get('warn_min', _WARN_DEFAULTS['warn_min']))
    if w_min_on:
        w_min = st.number_input("최저 순부하 임계값 (MW)", value=w_min, step=10)

    w_max_on = st.checkbox(
        "🟣 최대발전 경고 활성화",
        value=bool(st.session_state.get('warn_max_enabled', _WARN_DEFAULTS['warn_max_enabled'])),
    )
    w_max = int(st.session_state.get('warn_max', _WARN_DEFAULTS['warn_max']))
    if w_max_on:
        w_max = st.number_input("최대 순부하 임계값 (MW)", value=w_max, step=10)

    st.caption(f"경고 우선순위: 최저 > 최대 > 심야 > 저/고발전  \n\n"
               f"기준 = 예측 순부하(est_net_load). 심야 조건: 00-06시 중 순부하 < 임계값")

    return {
        'warn_low':               int(w_low),
        'warn_high':              int(w_high),
        'warn_min_enabled':       bool(w_min_on),
        'warn_min':               int(w_min),
        'warn_max_enabled':       bool(w_max_on),
        'warn_max':               int(w_max),
        'warn_overnight_enabled': bool(w_overnight_on),
        'warn_overnight':         int(w_overnight),
    }


def commit_warning_thresholds(values: dict):
    """`render_warning_threshold_inputs()` 가 반환한 dict 를 session_state 에 반영."""
    for k, v in values.items():
        st.session_state[k] = v
