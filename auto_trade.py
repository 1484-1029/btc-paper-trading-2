# ===================================================
# BTC自動売買スクリプト(bitbank) pandas-ta不使用版
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
warnings.filterwarnings('ignore')

API_KEY    = os.environ.get('BITBANK_API_KEY')
API_SECRET = os.environ.get('BITBANK_API_SECRET')
PAIR       = 'btc_jpy'

# ===================================================
# 指標計算(pandas-taを使わず自前で実装)
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
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, lower

def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calc_stoch(high, low, close, k=14, d=3):
    low_k  = low.rolling(window=k).min()
    high_k = high.rolling(window=k).max()
    stoch_k = 100 * (close - low_k) / (high_k - low_k)
    stoch_d = stoch_k.rolling(window=d).mean()
    return stoch_k, stoch_d

# ===================================================
# bitbank API関連の関数
# ===================================================
def make_signature(path, nonce, body=''):
    # GETの場合: nonce + path
    # POSTの場合: nonce + body(JSON文字列)
    if body:
        message = nonce + body
    else:
        message = nonce + path
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
    r = requests.get(
        'https://api.bitbank.cc' + path,
        headers=get_headers(path, nonce)
    )
    return r.json()

def get_btc_price():
    r = requests.get('https://public.bitbank.cc/btc_jpy/ticker')
    data = r.json()
    return float(data['data']['last'])

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
# LightGBMによるシグナル生成
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

    X_train  = X.iloc[:-3]
    y_train  = y.iloc[:-3]
    X_latest = X.iloc[[-1]]

    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=2,
        num_leaves=3,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_latest)[0][1]
    return proba

# ===================================================
# メイン処理
# ===================================================
print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

assets = get_assets()
jpy_balance = 0
btc_balance = 0

for asset in assets['data']['assets']:
    if asset['asset'] == 'jpy':
        jpy_balance = float(asset['free_amount'])
    if asset['asset'] == 'btc':
        btc_balance = float(asset['free_amount'])

print(f"JPY残高: {jpy_balance:,.0f}円")
print(f"BTC残高: {btc_balance:.6f} BTC")

btc_price = get_btc_price()
print(f"現在のBTC価格: {btc_price:,.0f}円")

print("シグナル計算中...")
proba = get_signal()
print(f"上昇確率: {proba:.2%}")

MIN_ORDER_JPY = 1000

if proba >= 0.5:
    if jpy_balance >= MIN_ORDER_JPY:
        order_jpy = jpy_balance * 0.5
        btc_amount = round(order_jpy / btc_price, 4)
        # 最小注文数量チェック
        if btc_amount >= 0.0001:
            print(f"買いシグナル: {order_jpy:,.0f}円分({btc_amount:.6f}BTC)を購入")
            result = place_order('buy', btc_amount)
            print(f"注文結果: {result}")
        else:
            print(f"注文数量が最小値未満({btc_amount:.6f}BTC < 0.0001BTC)")
    else:
        print(f"買いシグナルだが残高不足({jpy_balance:.0f}円)")

print("完了")
