# -*- coding: utf-8 -*-
"""구름·바람 움직임 탭 — 위성 구름 프레임과 바람 격자를 지도 애니메이션으로 보여준다.

데이터는 collect_weather_motion.py(cron, 3시간마다)가 WEATHER_DIR 에 미리 받아 둔다.
  - 구름: NASA GIBS 히마와리 적외선(Band13) 영상 → 구름만 남기고 투명 처리한 webp 프레임
  - 바람: Open-Meteo 10m 바람 격자 → leaflet-velocity 입력 형식 JSON
화면은 브라우저(Leaflet)가 그린다 — 서버는 파일만 읽어 HTML 에 넣어 줄 뿐이다.
"""
import base64
import json
import math
import os
from datetime import datetime, timedelta, timezone

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEATHER_DIR = os.path.join(PROJECT_DIR, "database", "weather_motion")

# 표시 영역: 제주로 다가오는 구름이 보이도록 서해·남해·동중국해까지 넓게 잡는다.
SOUTH_LAT, NORTH_LAT = 30.0, 37.0
WEST_LON, EAST_LON = 122.0, 131.0

# 화면에 재생할 구름 프레임 범위(최근 몇 시간, 몇 분 간격)
ANIMATION_HOURS = 6
FRAME_INTERVAL_MINUTES = 30

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


def cloud_frame_path(observed_at_utc):
    return os.path.join(WEATHER_DIR, f"cloud_{observed_at_utc:%Y%m%d%H%M}.webp")


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


def list_wind_grids():
    """저장된 바람 격자 [(유효시각 UTC, 경로)] — 오래된 것부터."""
    if not os.path.isdir(WEATHER_DIR):
        return []
    grids = [(_time_from_filename(name, "wind_", "%Y%m%d%H"), os.path.join(WEATHER_DIR, name))
             for name in os.listdir(WEATHER_DIR) if name.startswith("wind_")]
    return sorted(grids)


def build_map_html(height=640):
    """최근 ANIMATION_HOURS 시간치 구름 프레임 + 현재 시각에 가장 가까운 바람 격자로 지도 HTML 생성.

    데이터가 하나도 없으면 None.
    """
    cloud_frames = list_cloud_frames()
    if cloud_frames:
        newest = cloud_frames[-1][0]
        cloud_frames = [(t, p) for t, p in cloud_frames
                        if t > newest - timedelta(hours=ANIMATION_HOURS)]

    wind_grids = list_wind_grids()
    wind_json = "null"
    wind_label = ""
    if wind_grids:
        now = datetime.now(timezone.utc)
        valid_at, path = min(wind_grids, key=lambda item: abs(item[0] - now))
        with open(path, encoding="utf-8") as f:
            wind_json = f.read()
        wind_label = f"{valid_at.astimezone(KST):%m-%d %H:%M} KST"

    if not cloud_frames and not wind_grids:
        return None

    frame_entries = []
    for observed_at, path in cloud_frames:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        frame_entries.append({
            "label": f"{observed_at.astimezone(KST):%m-%d %H:%M} KST",
            "src": f"data:image/webp;base64,{encoded}",
        })

    html = MAP_TEMPLATE
    replacements = {
        "__HEIGHT__": str(height),
        "__FRAMES__": json.dumps(frame_entries),
        "__WIND__": wind_json,
        "__WIND_LABEL__": wind_label,
        "__BOUNDS__": json.dumps([[SOUTH_LAT, WEST_LON], [NORTH_LAT, EAST_LON]]),
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
  #frame_slider { flex: 1; }
  #frame_label { min-width: 120px; font-weight: 600; }
  .dark_base_map { filter: invert(1) hue-rotate(180deg) grayscale(0.7) brightness(0.75); }
  #map { background: #1b1b1b; }
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
  const windData = __WIND__;
  const bounds = __BOUNDS__;

  // 제주가 가운데 오도록 — 위성·바람 영역(서해~남해)이 화면을 대부분 채우는 배율
  const map = L.map("map", { minZoom: 5, maxZoom: 10 }).setView([33.5, 126.5], 7);
  // OSM 지도를 CSS 로 어둡게 뒤집어 쓴다 — 흰 구름과 밝은 바람 입자가 잘 보이도록(윈디와 비슷한 느낌)
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap | 구름: NASA GIBS (Himawari AHI) | 바람: Open-Meteo (CC BY 4.0)",
    className: "dark_base_map",
  }).addTo(map);

  // 구름: 프레임마다 imageOverlay 를 미리 만들어 두고 투명도만 바꿔 깜빡임 없이 넘긴다.
  const cloudLayers = frames.map(frame =>
    L.imageOverlay(frame.src, bounds, { opacity: 0, interactive: false }).addTo(map));
  const slider = document.getElementById("frame_slider");
  const frameLabel = document.getElementById("frame_label");
  const playButton = document.getElementById("play_button");
  const showClouds = document.getElementById("show_clouds");
  slider.max = Math.max(frames.length - 1, 0);

  let currentFrame = frames.length - 1;
  function showFrame(index) {
    cloudLayers.forEach((layer, i) =>
      layer.setOpacity(showClouds.checked && i === index ? 0.85 : 0));
    currentFrame = index;
    slider.value = index;
    frameLabel.textContent = frames.length ? "구름 " + frames[index].label : "구름 자료 없음";
  }
  showFrame(Math.max(currentFrame, 0));

  let playing = frames.length > 1;
  setInterval(() => {
    if (!playing || frames.length < 2) return;
    showFrame(currentFrame >= frames.length - 1 ? 0 : currentFrame + 1);
  }, 700);
  playButton.onclick = () => {
    playing = !playing;
    playButton.textContent = playing ? "⏸ 정지" : "▶ 재생";
  };
  if (!playing) playButton.textContent = "▶ 재생";
  slider.oninput = () => { playing = false; playButton.textContent = "▶ 재생"; showFrame(+slider.value); };
  showClouds.onchange = () => showFrame(currentFrame);

  // 바람: 성긴 격자를 입자로 흘려 보낸다(빠르기·방향만 감 잡는 용도).
  let windLayer = null;
  if (windData) {
    windLayer = L.velocityLayer({
      data: windData,
      displayValues: true,
      displayOptions: {
        velocityType: "바람 (__WIND_LABEL__)",
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
      // 흰 구름과 겹치지 않게 하늘색 계열 → 강해질수록(15m/s 이상) 노랑·주황
      colorScale: ["#4fc3f7", "#81d4fa", "#ffe082", "#ffb300", "#ff7043"],
    }).addTo(map);
  }
  document.getElementById("show_wind").onchange = event => {
    if (!windLayer) return;
    if (event.target.checked) windLayer.addTo(map); else map.removeLayer(windLayer);
  };
</script>
"""
