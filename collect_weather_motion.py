"""collect_weather_motion.py — 구름·바람 움직임 탭 자료 수집 cron 진입점 (3시간마다).

    python collect_weather_motion.py

1) 구름: NASA GIBS 히마와리 적외선(Band13) 영상을 30분 간격으로 받아, 구름만 남기고 나머지는
   투명하게 바꾼 webp 프레임으로 저장한다(이미 받은 시각은 건너뜀). GIBS 는 약 1시간 늦게 올라온다.
2) 바람: Open-Meteo 에서 0.5도 간격 10m 바람 격자를 받아 시각별 JSON 으로 저장한다.
3) 보관: RETENTION_HOURS 보다 오래된 파일은 지운다 — 디스크가 계속 늘지 않도록.
"""
import sys
import os
import io
import json
import math
import logging
from datetime import datetime, timedelta, timezone

import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.weather_motion import (
    WEATHER_DIR, SOUTH_LAT, NORTH_LAT, WEST_LON, EAST_LON,
    ANIMATION_HOURS, FRAME_INTERVAL_MINUTES,
    mercator_bbox, cloud_frame_path, wind_grid_path, list_cloud_frames, list_wind_grids,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('collect_weather_motion')

RETENTION_HOURS = 48

GIBS_WMS_URL = "https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi"
GIBS_LAYER = "Himawari_AHI_Band13_Clean_Infrared"
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 790   # 표시 영역의 Web Mercator 가로세로 비율에 맞춤

# 적외선 밝기(0~255) → 구름 투명도. 지표·바다(따뜻함)는 어둡게(약 80~100) 찍히므로 투명하게 지우고,
# 차가운 구름 꼭대기일수록 밝아지므로 진하게 남긴다.
CLEAR_SKY_BRIGHTNESS = 110
THICK_CLOUD_BRIGHTNESS = 190

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
WIND_GRID_STEP = 0.5     # 도 — 윈디처럼 촘촘할 필요 없음(입자는 격자 사이를 보간해 흐른다)
WIND_HOURS_AHEAD = 3     # 다음 수집(3시간 뒤)까지 쓸 시각들


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


def collect_wind_grids():
    latitudes = [NORTH_LAT - WIND_GRID_STEP * row
                 for row in range(int((NORTH_LAT - SOUTH_LAT) / WIND_GRID_STEP) + 1)]
    longitudes = [WEST_LON + WIND_GRID_STEP * column
                  for column in range(int((EAST_LON - WEST_LON) / WIND_GRID_STEP) + 1)]
    # Open-Meteo 다지점 요청: 위도·경도 목록을 같은 순서로 나란히 보낸다(북→남, 서→동).
    point_latitudes = [lat for lat in latitudes for _ in longitudes]
    point_longitudes = [lon for _ in latitudes for lon in longitudes]

    response = requests.get(OPEN_METEO_URL, params={
        "latitude": ",".join(f"{lat:.2f}" for lat in point_latitudes),
        "longitude": ",".join(f"{lon:.2f}" for lon in point_longitudes),
        "hourly": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms",
        "timezone": "GMT",
        "past_hours": 1,
        "forecast_hours": WIND_HOURS_AHEAD + 1,
    }, timeout=60)
    response.raise_for_status()
    points = response.json()

    hour_labels = points[0]["hourly"]["time"]
    for hour_index, hour_label in enumerate(hour_labels):
        valid_at = datetime.strptime(hour_label, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
        eastward_wind, northward_wind = [], []
        for point in points:
            speed = point["hourly"]["wind_speed_10m"][hour_index] or 0.0
            direction = point["hourly"]["wind_direction_10m"][hour_index] or 0.0
            # 풍향은 "불어오는" 방향(도) → 불어가는 방향 성분으로 바꾼다.
            eastward_wind.append(round(-speed * math.sin(math.radians(direction)), 2))
            northward_wind.append(round(-speed * math.cos(math.radians(direction)), 2))

        # leaflet-velocity 입력 형식(grib2json 과 동일): u 성분, v 성분 두 레코드
        header = {
            "nx": len(longitudes), "ny": len(latitudes),
            "lo1": WEST_LON, "la1": NORTH_LAT, "lo2": EAST_LON, "la2": SOUTH_LAT,
            "dx": WIND_GRID_STEP, "dy": WIND_GRID_STEP,
            "parameterCategory": 2, "refTime": valid_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        velocity_records = [
            {"header": {**header, "parameterNumber": 2, "parameterNumberName": "eastward_wind"},
             "data": eastward_wind},
            {"header": {**header, "parameterNumber": 3, "parameterNumberName": "northward_wind"},
             "data": northward_wind},
        ]
        with open(wind_grid_path(valid_at), "w", encoding="utf-8") as f:
            json.dump(velocity_records, f)
    logger.info(f"바람 격자 {len(hour_labels)}시각 저장 ({len(points)}지점)")


def delete_old_files():
    oldest_kept = datetime.now(timezone.utc) - timedelta(hours=RETENTION_HOURS)
    deleted = 0
    for saved_at, path in list_cloud_frames() + list_wind_grids():
        if saved_at < oldest_kept:
            os.remove(path)
            deleted += 1
    logger.info(f"{RETENTION_HOURS}시간 지난 파일 {deleted}개 삭제")


def main():
    os.makedirs(WEATHER_DIR, exist_ok=True)
    failed = False
    for step in (collect_cloud_frames, collect_wind_grids):
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
