"""collect_weather_motion.py — 구름·바람 움직임 탭 자료 수집 cron 진입점.

    python collect_weather_motion.py              # 위성 관측만 (cron 3시간마다)
    python collect_weather_motion.py --forecast   # 예보까지 (cron 하루 2번, 01:00·13:00 KST)

1) 구름: NASA GIBS 히마와리 적외선(Band13) 영상을 30분 간격으로 받아, 구름만 남기고 나머지는
   투명하게 바꾼 webp 프레임으로 저장한다(이미 받은 시각은 건너뜀). GIBS 는 약 1시간 늦게 올라온다.
2) 예보: Open-Meteo 의 JMA MSM 모델(일본 기상청 5km)에서 0.25도 격자로 운량·10m 바람을 받는다.
   - 운량 → 앞으로 FORECAST_FETCH_HOURS 시간의 시각별 구름 예보 프레임(webp). 위성 프레임 뒤에 이어 재생된다.
   - 바람 → 지난 몇 시간 + 앞으로의 시각별 JSON(0.5도로 솎아서). 화면 시각에 맞춰 입자가 바뀐다.
   새로 받을 때마다 최신 예보로 덮어쓴다. JMA 를 쓰는 이유: jeju_model 태양광 모델 운량 입력과 같은 출처.
   ★ Open-Meteo 는 지점 하나를 호출 1회로 센다(841점 = 841회, 무료 하루 1만 회). 그래서 예보는
   하루 2번만 받는다 → 하루 약 1,700회. JMA MSM 은 00·12 UTC 실행만 78시간까지 길게 내고,
   Open-Meteo 게시까지 약 3시간 반이 걸리므로 01:00·13:00 KST 에 받으면 각각 그 실행을 받는다.
3) 보관: RETENTION_HOURS 보다 오래된 파일은 지운다 — 디스크가 계속 늘지 않도록.
"""
import sys
import os
import io
import json
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.weather_motion import (
    WEATHER_DIR, SOUTH_LAT, NORTH_LAT, WEST_LON, EAST_LON,
    ANIMATION_HOURS, FRAME_INTERVAL_MINUTES,
    mercator_bbox, cloud_frame_path, forecast_cloud_frame_path, wind_grid_path,
    list_cloud_frames, list_forecast_cloud_frames, list_wind_grids,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('collect_weather_motion')

RETENTION_HOURS = 48
# 예보 1회 수신분의 길이 — 다음 수신(12시간 뒤)이 한 번 실패해도 앞으로 24시간이 비지 않도록 넉넉히
FORECAST_FETCH_HOURS = 48

GIBS_WMS_URL = "https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi"
GIBS_LAYER = "Himawari_AHI_Band13_Clean_Infrared"
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 790   # 표시 영역의 Web Mercator 가로세로 비율에 맞춤

# 적외선 밝기(0~255) → 구름 투명도. 지표·바다(따뜻함)는 어둡게(약 80~100) 찍히므로 투명하게 지우고,
# 차가운 구름 꼭대기일수록 밝아지므로 진하게 남긴다.
CLEAR_SKY_BRIGHTNESS = 110
THICK_CLOUD_BRIGHTNESS = 190

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_GRID_STEP = 0.25   # 도 — 구름 예보 격자(29x29=841점, 한 번 요청에 약 3초)
WIND_GRID_THINNING = 2      # 바람은 격자를 한 칸씩 건너 0.5도로 — 윈디처럼 촘촘할 필요 없음
FORECAST_IMAGE_WIDTH, FORECAST_IMAGE_HEIGHT = 320, 395   # 격자가 성기므로 작게 만들고 브라우저가 키운다

# 운량(%) → 구름 투명도. 옅은 운량(10% 이하)은 지우고 90% 이상이면 가장 진하게.
FORECAST_CLEAR_PERCENT = 10
FORECAST_OVERCAST_PERCENT = 90


def fetch_cloud_frame(observed_at_utc):
    """해당 시각 위성 영상을 받아 구름만 남긴 RGBA 이미지로. 아직 안 올라온 시각이면 None."""
    west_x, south_y, east_x, north_y = mercator_bbox()
    response = requests.get(GIBS_WMS_URL, params={
        "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0",
        "LAYERS": GIBS_LAYER, "CRS": "EPSG:3857",
        "BBOX": f"{west_x},{south_y},{east_x},{north_y}",
        "WIDTH": IMAGE_WIDTH, "HEIGHT": IMAGE_HEIGHT, "FORMAT": "image/png",
        "TIME": observed_at_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, timeout=60)
    response.raise_for_status()
    if not response.headers.get("content-type", "").startswith("image/"):
        return None
    satellite_image = Image.open(io.BytesIO(response.content)).convert("RGBA")
    if satellite_image.getchannel("A").getextrema()[1] == 0:   # 완전 투명 = 그 시각 영상 없음
        return None

    brightness = satellite_image.convert("L")
    scale = 255 / (THICK_CLOUD_BRIGHTNESS - CLEAR_SKY_BRIGHTNESS)
    cloud_alpha = brightness.point(
        lambda value: max(0, min(255, int((value - CLEAR_SKY_BRIGHTNESS) * scale))))
    cloud_image = Image.new("RGBA", satellite_image.size, (255, 255, 255, 0))
    cloud_image.putalpha(cloud_alpha)
    return cloud_image


def collect_cloud_frames():
    now = datetime.now(timezone.utc)
    newest_slot = now.replace(minute=now.minute - now.minute % FRAME_INTERVAL_MINUTES,
                              second=0, microsecond=0)
    # 표시 범위 + 1시간(GIBS 지연분)만큼 거슬러 올라가며 빠진 프레임을 채운다.
    slot_count = (ANIMATION_HOURS + 1) * 60 // FRAME_INTERVAL_MINUTES
    saved = 0
    for step in range(slot_count + 1):
        observed_at = newest_slot - timedelta(minutes=FRAME_INTERVAL_MINUTES * step)
        path = cloud_frame_path(observed_at)
        if os.path.exists(path):
            continue
        try:
            cloud_image = fetch_cloud_frame(observed_at)
        except requests.RequestException as error:
            logger.warning(f"구름 {observed_at:%m-%d %H:%M}Z 받기 실패: {error}")
            continue
        if cloud_image is None:
            continue
        cloud_image.save(path, "WEBP", quality=70)
        saved += 1
    logger.info(f"구름 프레임 {saved}장 새로 저장")


def _forecast_grid_axes():
    latitudes = [NORTH_LAT - FORECAST_GRID_STEP * row
                 for row in range(round((NORTH_LAT - SOUTH_LAT) / FORECAST_GRID_STEP) + 1)]
    longitudes = [WEST_LON + FORECAST_GRID_STEP * column
                  for column in range(round((EAST_LON - WEST_LON) / FORECAST_GRID_STEP) + 1)]
    return latitudes, longitudes


def _cloud_cover_to_image(cloud_cover_grid):
    """위경도 격자(북→남, 서→동)의 운량(%)을 지도(Web Mercator)에 맞는 구름 이미지로.

    위도는 메르카토르에서 간격이 고르지 않으므로, 이미지의 각 행이 가리키는 위도를 구해 격자를
    보간한다(그냥 늘리면 남북으로 조금씩 어긋난다).
    """
    west_x, south_y, east_x, north_y = mercator_bbox()
    row_mercator_y = north_y - (np.arange(FORECAST_IMAGE_HEIGHT) + 0.5) / FORECAST_IMAGE_HEIGHT * (north_y - south_y)
    row_latitude = np.degrees(2 * np.arctan(np.exp(row_mercator_y / 6378137)) - np.pi / 2)
    row_position = (NORTH_LAT - row_latitude) / FORECAST_GRID_STEP
    column_longitude = WEST_LON + (np.arange(FORECAST_IMAGE_WIDTH) + 0.5) / FORECAST_IMAGE_WIDTH * (EAST_LON - WEST_LON)
    column_position = (column_longitude - WEST_LON) / FORECAST_GRID_STEP

    grid_rows = np.arange(cloud_cover_grid.shape[0])
    grid_columns = np.arange(cloud_cover_grid.shape[1])
    by_row = np.array([np.interp(row_position, grid_rows, cloud_cover_grid[:, column])
                       for column in grid_columns]).T                    # (이미지 행, 격자 열)
    cloud_cover = np.array([np.interp(column_position, grid_columns, row) for row in by_row])

    alpha = np.clip((cloud_cover - FORECAST_CLEAR_PERCENT)
                    / (FORECAST_OVERCAST_PERCENT - FORECAST_CLEAR_PERCENT), 0, 1) * 255
    cloud_image = Image.new("RGBA", (FORECAST_IMAGE_WIDTH, FORECAST_IMAGE_HEIGHT), (255, 255, 255, 0))
    cloud_image.putalpha(Image.fromarray(alpha.astype("uint8")))
    return cloud_image


def _wind_records(speeds, directions, latitudes, longitudes, valid_at):
    """leaflet-velocity 입력 형식(grib2json 과 동일): u 성분, v 성분 두 레코드."""
    speeds = np.nan_to_num(speeds)[::WIND_GRID_THINNING, ::WIND_GRID_THINNING]
    directions = np.radians(np.nan_to_num(directions)[::WIND_GRID_THINNING, ::WIND_GRID_THINNING])
    # 풍향은 "불어오는" 방향(도) → 불어가는 방향 성분으로 바꾼다.
    eastward_wind = np.round(-speeds * np.sin(directions), 2)
    northward_wind = np.round(-speeds * np.cos(directions), 2)
    step = FORECAST_GRID_STEP * WIND_GRID_THINNING
    header = {
        "nx": eastward_wind.shape[1], "ny": eastward_wind.shape[0],
        "lo1": WEST_LON, "la1": NORTH_LAT,
        "lo2": longitudes[::WIND_GRID_THINNING][-1], "la2": latitudes[::WIND_GRID_THINNING][-1],
        "dx": step, "dy": step,
        "parameterCategory": 2, "refTime": valid_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return [
        {"header": {**header, "parameterNumber": 2, "parameterNumberName": "eastward_wind"},
         "data": eastward_wind.ravel().tolist()},
        {"header": {**header, "parameterNumber": 3, "parameterNumberName": "northward_wind"},
         "data": northward_wind.ravel().tolist()},
    ]


def collect_forecast_grids():
    latitudes, longitudes = _forecast_grid_axes()
    # Open-Meteo 다지점 요청: 위도·경도 목록을 같은 순서로 나란히 보낸다(북→남, 서→동).
    # 841점이라 주소가 길어져 POST 로 보낸다.
    response = requests.post(OPEN_METEO_URL, data={
        "latitude": ",".join(f"{lat:.2f}" for lat in latitudes for _ in longitudes),
        "longitude": ",".join(f"{lon:.2f}" for _ in latitudes for lon in longitudes),
        "hourly": "cloud_cover,wind_speed_10m,wind_direction_10m",
        "models": "jma_msm",
        "wind_speed_unit": "ms",
        "timezone": "GMT",
        "past_hours": ANIMATION_HOURS + 1,      # 위성 프레임 구간의 바람
        "forecast_hours": FORECAST_FETCH_HOURS + 1,
    }, timeout=120)
    response.raise_for_status()
    points = response.json()

    def as_grid(variable, hour_index):
        values = [point["hourly"][variable][hour_index] for point in points]
        return np.array(values, dtype=float).reshape(len(latitudes), len(longitudes))

    current_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    hour_labels = points[0]["hourly"]["time"]
    cloud_saved = 0
    for hour_index, hour_label in enumerate(hour_labels):
        valid_at = datetime.strptime(hour_label, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
        wind = _wind_records(as_grid("wind_speed_10m", hour_index),
                             as_grid("wind_direction_10m", hour_index), latitudes, longitudes, valid_at)
        with open(wind_grid_path(valid_at), "w", encoding="utf-8") as f:
            json.dump(wind, f)
        if valid_at >= current_hour:   # 지난 시각은 위성 관측이 있으므로 구름 예보는 앞으로만
            cloud_cover = np.nan_to_num(as_grid("cloud_cover", hour_index))
            _cloud_cover_to_image(cloud_cover).save(forecast_cloud_frame_path(valid_at), "WEBP", quality=70)
            cloud_saved += 1
    logger.info(f"바람 {len(hour_labels)}시각 · 구름 예보 {cloud_saved}시각 저장 ({len(points)}지점)")


def delete_old_files():
    oldest_kept = datetime.now(timezone.utc) - timedelta(hours=RETENTION_HOURS)
    deleted = 0
    for saved_at, path in list_cloud_frames() + list_forecast_cloud_frames() + list_wind_grids():
        if saved_at < oldest_kept:
            os.remove(path)
            deleted += 1
    logger.info(f"{RETENTION_HOURS}시간 지난 파일 {deleted}개 삭제")


def main():
    os.makedirs(WEATHER_DIR, exist_ok=True)
    steps = [collect_cloud_frames]
    if "--forecast" in sys.argv:
        steps.append(collect_forecast_grids)
    failed = False
    for step in steps:
        try:
            step()
        except Exception:
            logger.exception(f"{step.__name__} 실패")
            failed = True
    delete_old_files()
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
