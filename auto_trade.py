# ===================================================
# BTC自動売買スクリプト(bitbank) 改善版
# ポジション管理・損切り・Gmail通知追加
# ===================================================
import os
import time
import hmac
import hashlib
import requests
import json
import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
import warnings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
warnings.filterwarnings('ignore')

API_KEY            = os.environ.get('BITBANK_API_KEY')
API_SECRET         = os.environ.get('BITBANK_API_SECRET')
GMAIL_ADDRESS      = os.environ.get('GMAIL_ADDRESS')
GMAIL_APP_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD')
PAIR               = 'btc_jpy'

# ===================================================
# 設定値
# ===================================================
STOP_LOSS_PCT  = 0.05  # 損切りライン: 買値から-5%で強制売却
ORDER_RATIO    = 0.5   # 買い時: JPY残高の50%を使う
MIN_ORDER_JPY  = 1000  # 最低注文金額
POSITION_FILE  = 'position.json'

# ===================================================
# Gmail通知
# ===================================================
def send_notification(subject, body):
    try:
        msg = MIMEMultipart()
        msg['From']    = GMAIL_ADDRESS
        msg['To']      = GMAIL_ADDRESS
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.send_message(msg)
        print(f"通知送信完了: {subject}")
    except Exception as e:
        print(f"通知送信失敗: {e}")

# ===================================================
# 指標計算(pandas-ta不使用)
# ===================================================
def calc_sma(series, window):
    return series.rolling(window=window).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def calc_bbands(series, window=20):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return sma + 2 * std, sma - 2 * std

def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calc_stoch(high, low, close, k=14, d=3):
    low_k   = low.rolling(window=k).min()
    high_k  = high.rolling(window=k).max()
    stoch_k = 100 * (close - low_k) / (high_k - low_k)
    stoch_d = stoch_k.rolling(window=d).mean()
    return stoch_k, stoch_d

# ===================================================
# bitbank API
# ===================================================
def make_signature(path, nonce, body=''):
    message = nonce + body if body else nonce + path
    return hmac.new(
        API_SECRET.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def get_headers(path, nonce, body=''):
    return {
        'ACCESS-KEY': API_KEY,
        'ACCESS-NONCE': nonce,
        'ACCESS-SIGNATURE': make_signature(path, nonce, body),
        'Content-Type': 'application/json'
    }

def get_assets():
    path = '/v1/user/assets'
    nonce = str(int(time.time() * 1000))
    r = requests.get('https://api.bitbank.cc' + path, headers=get_headers(path, nonce))
    return r.json()

def get_btc_price():
    r = requests.get('https://public.bitbank.cc/btc_jpy/ticker')
    return float(r.json()['data']['last'])

def place_order(side, btc_amount):
    path = '/v1/user/spot/order'
    nonce = str(int(time.time() * 1000))
    body = json.dumps({
        'pair': PAIR,
        'amount': str(round(btc_amount, 4)),
        'side': side,
        'type': 'market',
        'post_only': False
    })
    r = requests.post(
        'https://api.bitbank.cc' + path,
        headers=get_headers(path, nonce, body),
        data=body
    )
    return r.json()

# ===================================================
# ポジション管理
# ===================================================
def load_position():
    if os.path.exists(POSITION_FILE):
        with open(POSITION_FILE, 'r') as f:
            return json.load(f)
    return {"position": "none", "entry_price": 0, "entry_date": "", "amount_btc": 0}

def save_position(position, entry_price, entry_date, amount_btc):
    data = {
        "position": position,
        "entry_price": entry_price,
        "entry_date": entry_date,
        "amount_btc": amount_btc
    }
    with open(POSITION_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"ポジション保存: {data}")

# ===================================================
# LightGBMシグナル生成
# ===================================================
def get_signal():
    df = yf.download('BTC-USD', period='5y', interval='1d', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['SMA5']  = calc_sma(df['Close'], 5)
    df['SMA20'] = calc_sma(df['Close'], 20)
    df['SMA5_20_ratio'] = df['SMA5'] / df['SMA20']
    df['RSI14'] = calc_rsi(df['Close'], 14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = calc_macd(df['Close'])

    bb_upper, bb_lower = calc_bbands(df['Close'], 20)
    df['BB_width']    = (bb_upper - bb_lower) / df['Close']
    df['BB_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)

    df['ATR14'] = calc_atr(df['High'], df['Low'], df['Close'], 14)

    for lag in [1, 2, 3, 5]:
        df[f'return_lag{lag}'] = df['Close'].pct_change(lag)

    df['weekday']       = df.index.dayofweek
    df['volume_change'] = df['Volume'].pct_change()
    df['volume_ratio']  = df['Volume'] / df['Volume'].rolling(5).mean()
    df['body']          = (df['Close'] - df['Open']) / df['Open']
    df['upper_shadow']  = (df['High'] - df[['Close','Open']].max(axis=1)) / df['Open']
    df['lower_shadow']  = (df[['Close','Open']].min(axis=1) - df['Low']) / df['Open']

    df['stoch_k'], df['stoch_d'] = calc_stoch(df['High'], df['Low'], df['Close'])
    df['volatility5']      = df['Close'].pct_change().rolling(5).std()
    df['volatility10']     = df['Close'].pct_change().rolling(10).std()
    df['high20']           = df['High'].rolling(20).max()
    df['low20']            = df['Low'].rolling(20).min()
    df['dist_from_high20'] = (df['Close'] - df['high20']) / df['high20']
    df['dist_from_low20']  = (df['Close'] - df['low20'])  / df['low20']

    FEATURE_COLS = [
        'SMA5_20_ratio', 'RSI14', 'MACD', 'MACD_signal', 'MACD_hist',
        'BB_width', 'BB_position', 'ATR14',
        'return_lag1', 'return_lag2', 'return_lag3', 'return_lag5',
        'weekday', 'volume_change', 'volume_ratio',
        'body', 'upper_shadow', 'lower_shadow',
        'stoch_k', 'stoch_d',
        'volatility5', 'volatility10',
        'dist_from_high20', 'dist_from_low20'
    ]

    df['target'] = (df['Close'].shift(-3) > df['Close']).astype(int)
    df = df.dropna()

    X = df[FEATURE_COLS]
    y = df['target']

    model = lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.01, max_depth=2,
        num_leaves=3, min_child_samples=50, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbose=-1
    )
    model.fit(X.iloc[:-3], y.iloc[:-3])
    return model.predict_proba(X.iloc[[-1]])[0][1]

# ===================================================
# メイン処理
# ===================================================
now   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
today = datetime.now().strftime('%Y-%m-%d')
print(f"実行時刻: {now}")
print("=" * 50)

# 残高と価格を取得
assets = get_assets()
jpy_balance = 0
btc_balance = 0
for asset in assets['data']['assets']:
    if asset['asset'] == 'jpy':
        jpy_balance = float(asset['free_amount'])
    if asset['asset'] == 'btc':
        btc_balance = float(asset['free_amount'])

btc_price = get_btc_price()
print(f"JPY残高       : {jpy_balance:,.0f}円")
print(f"BTC残高       : {btc_balance:.6f} BTC")
print(f"現在のBTC価格 : {btc_price:,.0f}円")

# ポジション読み込み
pos = load_position()
print(f"現在のポジション: {pos['position']}")
if pos['position'] == 'long':
    pnl_pct = (btc_price - pos['entry_price']) / pos['entry_price'] * 100
    print(f"エントリー価格  : {pos['entry_price']:,.0f}円")
    print(f"含み損益       : {pnl_pct:+.2f}%")

print("=" * 50)

# ===================================================
# 損切りチェック(最優先)
# ===================================================
if pos['position'] == 'long' and pos['entry_price'] > 0:
    pnl_pct = (btc_price - pos['entry_price']) / pos['entry_price']
    if pnl_pct <= -STOP_LOSS_PCT:
        print(f"🔴 損切り発動: {pnl_pct*100:.2f}% 下落 → 強制売却")
        if btc_balance >= 0.0001:
            result = place_order('sell', btc_balance)
            print(f"損切り注文結果: {result}")
            if result.get('success') == 1:
                save_position('none', 0, '', 0)
                send_notification(
                    '🔴 BTC損切り発動',
                    f'損切りが発動しました。\n'
                    f'売却価格: {btc_price:,.0f}円\n'
                    f'損益: {pnl_pct*100:.2f}%\n'
                    f'実行時刻: {now}'
                )
        else:
            print("BTC残高なし、ポジションリセット")
            save_position('none', 0, '', 0)
        exit()

# ===================================================
# シグナル取得
# ===================================================
print("シグナル計算中...")
proba = get_signal()
print(f"上昇確率: {proba:.2%}")

# ===================================================
# 売買判断
# ===================================================
if proba >= 0.5:
    # 買いシグナル
    if pos['position'] == 'none':
        if jpy_balance >= MIN_ORDER_JPY:
            order_jpy  = jpy_balance * ORDER_RATIO
            btc_amount = round(order_jpy / btc_price, 4)
            if btc_amount >= 0.0001:
                print(f"買いシグナル(新規): {order_jpy:,.0f}円分({btc_amount:.6f}BTC)を購入")
                result = place_order('buy', btc_amount)
                print(f"注文結果: {result}")
                if result.get('success') == 1:
                    save_position('long', btc_price, today, btc_amount)
                    send_notification(
                        '🟢 BTC買い注文完了',
                        f'買い注文が成立しました。\n'
                        f'購入価格: {btc_price:,.0f}円\n'
                        f'購入数量: {btc_amount:.6f}BTC\n'
                        f'購入金額: {order_jpy:,.0f}円\n'
                        f'実行時刻: {now}'
                    )
            else:
                print(f"注文数量が最小値未満のためスキップ")
        else:
            print(f"残高不足のためスキップ({jpy_balance:.0f}円)")
    else:
        print(f"買いシグナルだが、すでにロング保有中 → 追加購入しない")

else:
    # 売りシグナル
    if pos['position'] == 'long':
        if btc_balance >= 0.0001:
            print(f"売りシグナル(決済): BTC({btc_balance:.6f}BTC)を売却")
            result = place_order('sell', btc_balance)
            print(f"注文結果: {result}")
            if result.get('success') == 1:
                save_position('none', 0, '', 0)
                send_notification(
                    '🔵 BTC売り注文完了',
                    f'売り注文が成立しました。\n'
                    f'売却価格: {btc_price:,.0f}円\n'
                    f'売却数量: {btc_balance:.6f}BTC\n'
                    f'実行時刻: {now}'
                )
        else:
            print(f"BTC残高なし、ポジションリセット")
            save_position('none', 0, '', 0)
    else:
        print(f"売りシグナルだがポジションなし → 何もしない")

print("=" * 50)
print("完了")
