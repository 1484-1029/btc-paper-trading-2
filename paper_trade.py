import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# ===================================================
# 設定
# ===================================================
SHORT_WINDOW = 5
LONG_WINDOW = 20
TICKER = "BTC-USD"
TRANSACTION_COST = 0.001
LOG_FILE = "btc_sma_log.csv"

# ===================================================
# 1. 最新データの取得
# ===================================================
data = yf.download(TICKER, period="60d", interval="1d", progress=False)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data['SMA_short'] = data['Close'].rolling(window=SHORT_WINDOW).mean()
data['SMA_long'] = data['Close'].rolling(window=LONG_WINDOW).mean()
data = data.dropna()

# 最新の状態
latest = data.iloc[-1]
current_price = float(latest['Close'])
current_sma_short = float(latest['SMA_short'])
current_sma_long = float(latest['SMA_long'])
current_signal = 1 if current_sma_short > current_sma_long else 0

# ===================================================
# 2. 過去ログの読み込みと前回シグナルの確認
# ===================================================
if os.path.exists(LOG_FILE):
    log_df = pd.read_csv(LOG_FILE)
    previous_signal = int(log_df.iloc[-1]['signal']) if len(log_df) > 0 else current_signal
else:
    log_df = pd.DataFrame(columns=[
        'timestamp', 'price', 'sma_short', 'sma_long', 'signal', 'event'
    ])
    previous_signal = current_signal

# ===================================================
# 3. イベント判定
# ===================================================
if previous_signal == 0 and current_signal == 1:
    event = "BUY"
elif previous_signal == 1 and current_signal == 0:
    event = "SELL"
else:
    event = "HOLD"

# ===================================================
# 4. ログに追記して保存
# ===================================================
new_row = pd.DataFrame([{
    'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
    'price': round(current_price, 2),
    'sma_short': round(current_sma_short, 2),
    'sma_long': round(current_sma_long, 2),
    'signal': current_signal,
    'event': event
}])

log_df = pd.concat([log_df, new_row], ignore_index=True)
log_df.to_csv(LOG_FILE, index=False)

# ===================================================
# 5. 結果出力(GitHub Actionsのログに表示される)
# ===================================================
print("=" * 50)
print(f"実行時刻(UTC) : {new_row.iloc[0]['timestamp']}")
print(f"現在価格      : {current_price:,.2f} USD")
print(f"SMA{SHORT_WINDOW}          : {current_sma_short:,.2f}")
print(f"SMA{LONG_WINDOW}         : {current_sma_long:,.2f}")
print(f"シグナル      : {'ロング' if current_signal == 1 else 'フラット'}")
print(f"イベント      : {event}")
print(f"記録件数      : {len(log_df)}件")
print("=" * 50)
