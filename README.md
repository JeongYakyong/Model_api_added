# Model_api_added — 제주 전력수요 예측 뷰어

`jeju_model` 이 매일 자동으로 만드는 전력수요 예측(내일·모레)을 실측과 함께 plot 으로 보여주는
초경량 Streamlit 앱. 서버 상세 배치는 `CLAUDE.md` 참고.

## 로컬 실행

```bash
pip install -r requirements.txt
cp .env.example .env   # JEJU_MODEL_DB_PATH 를 jeju_model/data/input_data_jeju.db 경로로 수정
python collect_actual_demand.py   # 최초 1회 — 실측 수요 채우기
python sync_forecast.py           # 최초 1회 — jeju_model 예측 동기화
streamlit run app.py
```

## crontab 가이드 (서버)

```cron
# 실측 수요 수집 — 매시 5분
5 * * * * cd /home/kimjourvanne/Model_api_added && venv/bin/python collect_actual_demand.py >> logs/cron.log 2>&1

# jeju_model 예측 동기화 — 그쪽 파이프라인(00:20/08:00 KST) 완료 후
40 0 * * * cd /home/kimjourvanne/Model_api_added && venv/bin/python sync_forecast.py >> logs/cron.log 2>&1
20 8 * * * cd /home/kimjourvanne/Model_api_added && venv/bin/python sync_forecast.py >> logs/cron.log 2>&1
```

## 운영 노트

- systemd 서비스 `jeju.service` (port 8503, `/jeju2`)가 그대로 이 `app.py` 를 서빙한다 — 코드
  갱신 후에는 프로세스를 재기동해야 반영된다(`Restart=always` 라 프로세스를 죽이기만 해도 된다).
- `database/jeju_energy.db` 는 git 제외 대상. 새 서버에 배포하면 첫 cron 실행 전까지는 빈 화면이다.
