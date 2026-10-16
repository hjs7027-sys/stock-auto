# ============================================================
# 국내 주식 초보자용 스윙매매 분석 시스템 (텔레그램 자동 알림용)
# ============================================================

import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import FinanceDataReader as fdr
from datetime import datetime, time
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")

# 1. 사용자 설정
CAPITAL = 5_000_000
RISK_PER_TRADE = 0.005
MAX_POSITION_RATIO = 0.25
MIN_TURNOVER = 1_000_000_000
MIN_VOLUME_RATIO = 1.2
MIN_RSI = 45
MAX_RSI = 70
ATR_MULTIPLIER = 2.0
TARGET_R_MULTIPLE = 1.5
MAX_STOP_PCT = 8.0
MIN_STOP_PCT = 1.0
MAX_RETURN20 = 25.0

now = datetime.now(ZoneInfo("Asia/Seoul"))

# 2. 텔레그램 메시지 전송 함수
def send_telegram_message(text):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("텔레그램 토큰이나 CHAT_ID가 설정되지 않았습니다.")
        return
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }
    requests.post(url, data=payload)

# 3. 보조지표 함수
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)
    return rsi

def calculate_atr(data, period=14):
    prev_close = data["Close"].shift(1)
    tr = pd.concat([
        data["High"] - data["Low"],
        (data["High"] - prev_close).abs(),
        (data["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

# 4. 시가총액 상위 100위 자동 수집 및 분석
krx_df = fdr.StockListing('KRX')
top100_df = krx_df[krx_df['Market'].isin(['KOSPI', 'KOSDAQ'])].sort_values('Marcap', ascending=False).head(100)

stocks = {}
for idx, row in top100_df.iterrows():
    yf_code = f"{row['Code']}.KS" if row['Market'] == 'KOSPI' else f"{row['Code']}.KQ"
    stocks[yf_code] = row['Name']

results = []

for ticker, name in stocks.items():
    try:
        data = yf.download(ticker, period="9mo", interval="1d", auto_adjust=True, progress=False, threads=False, multi_level_index=False, timeout=20)
        if data is None or data.empty: continue
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        data = data.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
        data = data[(data["Close"] > 0) & (data["Volume"] >= 0)]

        last_date = pd.Timestamp(data.index[-1]).date()
        if now.time() < time(15, 40) and last_date == now.date():
            data = data.iloc[:-1]

        if len(data) < 80: continue

        data["MA20"] = data["Close"].rolling(20).mean()
        data["MA60"] = data["Close"].rolling(60).mean()
        data["MA20_PREV5"] = data["MA20"].shift(5)
        data["VOL20"] = data["Volume"].rolling(20).mean()
        data["TURNOVER"] = data["Close"] * data["Volume"]
        data["TURNOVER20"] = data["TURNOVER"].rolling(20).mean()
        data["RSI14"] = calculate_rsi(data["Close"], 14)
        data["ATR14"] = calculate_atr(data, 14)
        data["RETURN20"] = data["Close"].pct_change(20) * 100

        today = data.iloc[-1]
        
        close = float(today["Close"])
        atr = float(today["ATR14"])
        avg_volume = float(today["VOL20"])
        avg_turnover = float(today["TURNOVER20"])
        rsi = float(today["RSI14"])
        return20 = float(today["RETURN20"])

        reference_price = close
        stop_price = reference_price - ATR_MULTIPLIER * atr
        stop_pct = (reference_price - stop_price) / reference_price * 100
        risk_per_share = reference_price - stop_price
        target_price = reference_price + TARGET_R_MULTIPLE * risk_per_share
        volume_ratio = float(today["Volume"]) / avg_volume

        trend_ok = (close > float(today["MA20"]) > float(today["MA60"]))
        ma_rising = (float(today["MA20"]) > float(today["MA20_PREV5"]))
        volume_ok = volume_ratio >= MIN_VOLUME_RATIO
        turnover_ok = avg_turnover >= MIN_TURNOVER
        rsi_ok = MIN_RSI <= rsi <= MAX_RSI
        stop_width_ok = MIN_STOP_PCT <= stop_pct <= MAX_STOP_PCT
        overheated = return20 >= MAX_RETURN20

        passed = all([trend_ok, ma_rising, volume_ok, turnover_ok, rsi_ok, stop_width_ok, not overheated])

        risk_budget = CAPITAL * RISK_PER_TRADE
        risk_shares = int(risk_budget // risk_per_share) if risk_per_share > 0 else 0
        capital_shares = int((CAPITAL * MAX_POSITION_RATIO) // reference_price)
        shares = max(0, min(risk_shares, capital_shares))

        planned_loss = shares * risk_per_share
        investment_amount = shares * reference_price

        results.append({
            "종목명": name, "기준가": round(reference_price, 0),
            "RSI14": round(rsi, 1), "거래량배수": round(volume_ratio, 2),
            "손절검토가": round(stop_price, 0), "목표검토가": round(target_price, 0),
            "예상수량": shares, "예상투자금": round(investment_amount, 0),
            "계획손실": round(planned_loss, 0), "후보통과": passed
        })
    except Exception as e:
        continue

# 5. 최종 결과 정리 및 텔레그램 전송
if results:
    df = pd.DataFrame(results)
    candidates = df[(df["후보통과"]) & (df["예상수량"] > 0)].sort_values(["거래량배수", "RSI14"], ascending=[False, True])

    if candidates.empty:
        msg = "🎯 *오늘의 스윙매매 분석 결과*\n\n조건을 완벽히 만족하는 종목이 없습니다.\n현금을 보유하며 관망하세요."
    else:
        msg = f"🎯 *오늘의 스윙매매 합격 종목 ({len(candidates)}개)*\n\n"
        for _, row in candidates.iterrows():
            msg += f"🔥 *{row['종목명']}*\n"
            msg += f"• 매수기준가: {row['기준가']:,.0f}원\n"
            msg += f"• 예상투자금: {row['예상투자금']:,.0f}원 ({row['예상수량']}주)\n"
            msg += f"• 목표/손절: {row['목표검토가']:,.0f}원 / {row['손절검토가']:,.0f}원\n"
            msg += f"• 계획손실금: {row['계획손실']:,.0f}원\n\n"
    
    send_telegram_message(msg)
