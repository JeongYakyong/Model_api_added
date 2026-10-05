# -*- coding: utf-8 -*-
"""구름·바람 움직임 탭 — 위성 구름 프레임과 바람 격자를 지도 애니메이션으로 보여준다.

데이터는 collect_weather_motion.py(cron, 3시간마다)가 WEATHER_DIR 에 미리 받아 둔다.
  - 구름(관측): NASA GIBS 히마와리 적외선(Band13) 영상 → 구름만 남기고 투명 처리한 webp 프레임
  - 구름(예보): Open-Meteo JMA MSM 운량 격자 → 같은 모양의 webp 프레임
  - 바람: Open-Meteo JMA MSM 10m 바람 격자(시각별) → leaflet-velocity 입력 형식 JSON
지도는 "지난 몇 시간 위성 관측 → 앞으로의 예보"를 슬라이더 하나로 이어서 재생하고, 바람 입자도
화면 시각에 맞춰 바뀐다. 화면은 브라우저(Leaflet)가 그린다 — 서버는 파일만 읽어 HTML 에 넣어 줄 뿐이다.
"""
import base64
import json
import math
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEATHER_DIR = os.path.join(PROJECT_DIR, "database", "weather_motion")

# 표시 영역: 남한 전체(수도권~제주) + 서해·남해 약간. 중국·일본 쪽은 필요 없어 잘라낸다.
SOUTH_LAT, NORTH_LAT = 32.0, 39.0
WEST_LON, EAST_LON = 124.0, 131.0

# 위치 감을 잡기 위한 주요 도시 이름(배경 지도는 지명·경계 없는 깔끔한 것을 쓰므로 직접 찍는다)
CITY_LABELS = [
    ("서울", 37.57, 126.98), ("강릉", 37.75, 128.88), ("대전", 36.35, 127.38),
    ("대구", 35.87, 128.60), ("광주", 35.16, 126.85), ("부산", 35.18, 129.08),
    ("제주", 33.50, 126.53),
]

# 화면에 재생할 범위: 위성 관측 최근 몇 시간(몇 분 간격) + 예보 앞으로 몇 시간(1시간 간격)
ANIMATION_HOURS = 6
FRAME_INTERVAL_MINUTES = 30
FORECAST_HOURS = 24

KST = timezone(timedelta(hours=9))


def mercator_bbox():
    """표시 영역을 Web Mercator(EPSG:3857) 미터 좌표로 — 위성 영상을 지도와 정확히 겹치기 위해."""
    def to_mercator(lon, lat):
        x = lon * 20037508.34 / 180
        y = math.log(math.tan((90 + lat) * math.pi / 360)) * 6378137
        return x, y
    west_x, south_y = to_mercator(WEST_LON, SOUTH_LAT)
    east_x, north_y = to_mercator(EAST_LON, NORTH_LAT)
    return west_x, south_y, east_x, north_y


def _carto_key_query():
    """CARTO 타일 URL 뒤에 붙일 키 쿼리스트링(키가 없으면 빈 문자열 — 지도에 워터마크가 찍힌다).

    ★ 호출 시점에 os.getenv 한다 — 모듈 최상단에서 읽으면 load_dotenv() 순서에 따라 빈 값이
    굳어버린다(jeju_model weather_map_jeju._carto_tile_qs 와 같은 관례).
    """
    key = os.getenv("CARTO_API_KEY", "").strip()
    return f"?key={quote(key, safe='')}" if key else ""


def cloud_frame_path(observed_at_utc):
    return os.path.join(WEATHER_DIR, f"cloud_{observed_at_utc:%Y%m%d%H%M}.webp")


def forecast_cloud_frame_path(valid_at_utc):
    return os.path.join(WEATHER_DIR, f"fcloud_{valid_at_utc:%Y%m%d%H}.webp")


def wind_grid_path(valid_at_utc):
    return os.path.join(WEATHER_DIR, f"wind_{valid_at_utc:%Y%m%d%H}.json")


def _time_from_filename(filename, prefix, time_format):
    stem = os.path.splitext(filename)[0]
    return datetime.strptime(stem[len(prefix):], time_format).replace(tzinfo=timezone.utc)


def list_cloud_frames():
    """저장된 구름 프레임 [(관측시각 UTC, 경로)] — 오래된 것부터."""
    if not os.path.isdir(WEATHER_DIR):
        return []
    frames = [(_time_from_filename(name, "cloud_", "%Y%m%d%H%M"), os.path.join(WEATHER_DIR, name))
              for name in os.listdir(WEATHER_DIR) if name.startswith("cloud_")]
    return sorted(frames)


def list_forecast_cloud_frames():
    """저장된 구름 예보 프레임 [(유효시각 UTC, 경로)] — 오래된 것부터."""
    if not os.path.isdir(WEATHER_DIR):
        return []
    frames = [(_time_from_filename(name, "fcloud_", "%Y%m%d%H"), os.path.join(WEATHER_DIR, name))
              for name in os.listdir(WEATHER_DIR) if name.startswith("fcloud_")]
    return sorted(frames)


def list_wind_grids():
    """저장된 바람 격자 [(유효시각 UTC, 경로)] — 오래된 것부터."""
    if not os.path.isdir(WEATHER_DIR):
        return []
    grids = [(_time_from_filename(name, "wind_", "%Y%m%d%H"), os.path.join(WEATHER_DIR, name))
             for name in os.listdir(WEATHER_DIR) if name.startswith("wind_")]
    return sorted(grids)


def _encode_image(path):
    with open(path, "rb") as f:
        return "data:image/webp;base64," + base64.b64encode(f.read()).decode("ascii")


def build_map_html(height=720):
    """위성 관측(최근 ANIMATION_HOURS 시간) → 예보(앞으로 FORECAST_HOURS 시간)를 한 줄로 이은 지도 HTML.

    데이터가 하나도 없으면 None.
    """
    observed_frames = list_cloud_frames()
    newest_observed = observed_frames[-1][0] if observed_frames else datetime.now(timezone.utc)
    observed_frames = [(t, p) for t, p in observed_frames
                       if t > newest_observed - timedelta(hours=ANIMATION_HOURS)]
    # 예보는 위성 관측이 끝난 다음 시각부터 이어 붙인다
    forecast_limit = datetime.now(timezone.utc) + timedelta(hours=FORECAST_HOURS)
    forecast_frames = [(t, p) for t, p in list_forecast_cloud_frames()
                       if newest_observed < t <= forecast_limit]

    frame_entries = []
    for kind, frames in (("관측", observed_frames), ("예보", forecast_frames)):
        for valid_at, path in frames:
            nearest_hour = (valid_at + timedelta(minutes=30)).replace(minute=0)
            frame_entries.append({
                "label": f"{kind} {valid_at.astimezone(KST):%m-%d %H:%M}",
                "is_forecast": kind == "예보",
                "src": _encode_image(path),
                "wind": f"{nearest_hour:%Y%m%d%H}",
            })
    if not frame_entries:
        return None

    # 프레임들이 가리키는 시각의 바람 격자만 싣는다
    wanted_wind_keys = {entry["wind"] for entry in frame_entries}
    wind_grids = {}
    for valid_at, path in list_wind_grids():
        wind_key = f"{valid_at:%Y%m%d%H}"
        if wind_key in wanted_wind_keys:
            with open(path, encoding="utf-8") as f:
                wind_grids[wind_key] = json.load(f)

    html = MAP_TEMPLATE
    replacements = {
        "__HEIGHT__": str(height),
        "__FRAMES__": json.dumps(frame_entries, ensure_ascii=False),
        "__WIND_GRIDS__": json.dumps(wind_grids),
        "__BOUNDS__": json.dumps([[SOUTH_LAT, WEST_LON], [NORTH_LAT, EAST_LON]]),
        "__CITIES__": json.dumps(CITY_LABELS, ensure_ascii=False),
        "__TILE_KEY_QUERY__": _carto_key_query(),
    }
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)
    return html


MAP_TEMPLATE = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet-velocity@2.1.4/dist/leaflet-velocity.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/leaflet-velocity@2.1.4/dist/leaflet-velocity.min.js"></script>
<style>
  body { margin: 0; font-family: sans-serif; }
  #map { height: __HEIGHT__px; width: 100%; border-radius: 6px; }
  #controls { display: flex; align-items: center; gap: 10px; padding: 6px 2px; font-size: 14px; }
  #controls button { padding: 4px 12px; cursor: pointer; }
  /* 슬라이더 바탕을 관측(회색) | 예보(주황) 두 색으로 나눠, 지금 어느 구간인지 보이게 */
  #frame_slider { flex: 1; -webkit-appearance: none; appearance: none; height: 8px; border-radius: 4px; }
  #frame_slider::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px;
                                        border-radius: 50%; background: #37474f; cursor: pointer; }
  #frame_slider::-moz-range-thumb { width: 16px; height: 16px; border: none;
                                    border-radius: 50%; background: #37474f; cursor: pointer; }
  #frame_label { min-width: 130px; font-weight: 700; color: #37474f; }
  #frame_label.forecast { color: #e65100; }
  #map { background: #e8eef2; }
  /* 밝은 바탕에서 흰 구름이 묻히지 않도록 회색으로 */
  .cloud_frame { filter: brightness(0.62); }
  .city_label { background: none; border: none; box-shadow: none; padding: 0;
                color: #263238; font-size: 13px; font-weight: 700; text-shadow: 0 0 3px #fff, 0 0 3px #fff; }
  .city_label::before { display: none; }
</style>
<div id="controls">
  <button id="play_button">⏸ 정지</button>
  <input type="range" id="frame_slider" min="0" value="0">
  <span id="frame_label"></span>
  <label><input type="checkbox" id="show_clouds" checked> 구름</label>
  <label><input type="checkbox" id="show_wind" checked> 바람</label>
</div>
<div id="map"></div>
<script>
  const frames = __FRAMES__;
  const windGrids = __WIND_GRIDS__;
  const bounds = __BOUNDS__;

  // 남한 전체(수도권~제주)가 화면 절반 폭 지도에 꽉 차는 위치·배율
  const map = L.map("map", { minZoom: 6, maxZoom: 10, maxBounds: [[29, 119], [42, 136]] })
    .setView([35.7, 127.6], 7);
  // 지명·행정경계 없는 밝은 지도
  L.tileLayer("https://{s}.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}{r}.png__TILE_KEY_QUERY__", {
    attribution: "&copy; OpenStreetMap &copy; CARTO | 관측: NASA GIBS (Himawari AHI) | 예보·바람: JMA MSM via Open-Meteo (CC BY 4.0)",
    subdomains: "abcd",
  }).addTo(map);

  // 구름: 프레임마다 imageOverlay 를 미리 만들어 두고 투명도만 바꿔 깜빡임 없이 넘긴다.
  const cloudLayers = frames.map(frame =>
    L.imageOverlay(frame.src, bounds, { opacity: 0, interactive: false, className: "cloud_frame" }).addTo(map));
  const slider = document.getElementById("frame_slider");
  const frameLabel = document.getElementById("frame_label");
  const playButton = document.getElementById("play_button");
  const showClouds = document.getElementById("show_clouds");
  slider.max = Math.max(frames.length - 1, 0);
  const firstForecast = frames.findIndex(frame => frame.is_forecast);
  const forecastStartPercent = firstForecast < 0
    ? 100 : (firstForecast - 0.5) / Math.max(frames.length - 1, 1) * 100;
  slider.style.background = "linear-gradient(to right, #b0bec5 0 " + forecastStartPercent
    + "%, #ffb74d " + forecastStartPercent + "% 100%)";

  let currentFrame = 0;
  function showFrame(index) {
    cloudLayers.forEach((layer, i) =>
      layer.setOpacity(showClouds.checked && i === index ? 0.85 : 0));
    currentFrame = index;
    slider.value = index;
    frameLabel.textContent = frames[index].label;
    frameLabel.classList.toggle("forecast", frames[index].is_forecast);
    showWind(frames[index].wind);
  }

  // 도시 이름은 구름·바람 위에 보이도록 별도 pane 에 올린다
  map.createPane("city_pane").style.zIndex = 650;
  __CITIES__.forEach(([name, lat, lon]) => {
    L.circleMarker([lat, lon], { pane: "city_pane", radius: 3, color: "#263238", weight: 1, fillOpacity: 1 })
      .bindTooltip(name, { permanent: true, direction: "right", className: "city_label", pane: "city_pane" })
      .addTo(map);
  });

  // 바람: 성긴 격자를 입자로 흘려 보낸다(빠르기·방향만 감 잡는 용도). 화면 시각이 바뀌면 그 시각 격자로.
  const windLayer = L.velocityLayer({
    data: null,
    displayValues: true,
    displayOptions: {
      velocityType: "바람",
      position: "bottomleft",
      emptyString: "바람 자료 없음",
      speedUnit: "m/s",
      directionString: "풍향",
      speedString: "풍속",
    },
    minVelocity: 0,
    maxVelocity: 15,
    velocityScale: 0.008,
    particleMultiplier: 1 / 1500,
    lineWidth: 1.2,
    // 밝은 바탕·흰 구름 위에서도 보이게 진한 색 — 약하면 파랑, 강해질수록(15m/s 이상) 주황·빨강
    colorScale: ["#1565c0", "#1e88e5", "#f9a825", "#ef6c00", "#c62828"],
  }).addTo(map);
  let shownWindKey = null;
  function showWind(windKey) {
    if (windKey === shownWindKey || !windGrids[windKey]) return;
    windLayer.setData(windGrids[windKey]);
    shownWindKey = windKey;
  }
  document.getElementById("show_wind").onchange = event => {
    if (event.target.checked) windLayer.addTo(map); else map.removeLayer(windLayer);
  };

  showFrame(0);
  let playing = frames.length > 1;
  if (!playing) playButton.textContent = "▶ 재생";
  setInterval(() => {
    if (!playing) return;
    showFrame(currentFrame >= frames.length - 1 ? 0 : currentFrame + 1);
  }, 700);
  playButton.onclick = () => {
    playing = !playing;
    playButton.textContent = playing ? "⏸ 정지" : "▶ 재생";
  };
  slider.oninput = () => { playing = false; playButton.textContent = "▶ 재생"; showFrame(+slider.value); };
  showClouds.onchange = () => showFrame(currentFrame);
</script>
"""
