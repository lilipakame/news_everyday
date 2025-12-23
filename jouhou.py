import os
import sys
import requests
import time
import httpx
from datetime import datetime
from importlib import import_module
from openai import APIConnectionError, APITimeoutError, OpenAI
from zoneinfo import ZoneInfo

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
prompt = os.environ["PROMPT"]

# JSTの今日
JST = ZoneInfo("Asia/Tokyo")
now = datetime.now(JST)
today_str = now.strftime("%Y-%m-%d")

# 日本の営業日判定（jpholiday があれば祝日も考慮、無ければ土日だけ）
def is_jp_holiday(date_obj):
    try:
        jpholiday = import_module("jpholiday")
        return jpholiday.is_holiday(date_obj)
    except Exception:
        return False

is_weekday = now.weekday() < 5            # Mon=0 ... Sun=6
is_holiday = is_jp_holiday(now.date())    # 祝日（jpholidayが無ければ常にFalse）
is_business_day = is_weekday and not is_holiday

if not is_business_day:
    print(f"本日は休日のためスキップ: {today_str} JST (weekday={now.weekday()}, jp_holiday={is_holiday})")
    sys.exit(0)

client = OpenAI(api_key=OPENAI_API_KEY, timeout=httpx.Timeout(1000.0))

request_body = {
    "model": "gpt-5.2",
    "reasoning": {"effort": "low"},
    "tools": [{"type": "web_search"}],
    "input": [
        {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
    ],
}

def run_with_retries(create_fn, *, max_attempts=3, base_wait=2.0):
    if max_attempts < 2:
        raise ValueError(
            "max_attempts must allow multiple retries (>=2) to satisfy retry policy"
        )
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return create_fn()
        except (APIConnectionError, APITimeoutError) as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            wait_time = base_wait * (2 ** (attempt - 1))
            print(f"OpenAI API call failed ({type(exc).__name__}: {exc}). Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
    raise last_error


resp = run_with_retries(lambda: client.responses.create(**request_body))

# Responses API のテキスト抽出（最初の出力を取得）
gpt_text = (getattr(resp, "output_text", "") or "").strip()

r = requests.post(WEBHOOK_URL, json={"content": gpt_text})

if r.status_code == 204:
    print("送信成功！")
else:
    print(f"送信失敗: {r.status_code} / {r.text}")
