# ===================================================
# BTC自動売買スクリプト(bitbank)
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
import pandas_ta as ta
import lightgbm as lgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

API_KEY    = os.environ.get('BITBANK_API_KEY')
API_SECRET = os.environ.get('BITBANK_API_SECRET')
PAIR       = 'btc_jpy'

# ===================================================
# bitbank API関連の関数
# ===================================================
def make_signature(path, nonce, body=''):
    message = nonce + path + body
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

def place_order(side, amount_jpy, price):
    path = '/v1/user/spot/order'
    nonce = str(int(time.time() * 1000))

    # 購入数量(BTC)を計算(小数点4桁まで)
    btc_amount = round(amount_jpy / price, 4)

    body = json.dumps({
        'pair': PAIR,
        'amount': str(btc_amount),
        'side': side,        # 'buy' or 'sell'
        'type': 'market',    # 成行注文
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

    df['SMA5']  = ta.sma(df['Close'], length=5)
    df['SMA20'] = ta.sma(df['Close'], length=20)
    df['SMA5_20_ratio'] = df['SMA5'] / df['SMA20']
    df['RSI14'] = ta.rsi(df['Close'], length=14)

    macd = ta.macd(df['Close'])
    df['MACD']        = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['MACD_hist']   = macd['MACDh_12_26_9']

    bb = ta.bbands(df['Close'], length=20)
    df['BB_width']    = (bb['BBU_20_2.0_2.0'] - bb['BBL_20_2.0_2.0']) / df['Close']
    df['BB_position'] = (df['Close'] - bb['BBL_20_2.0_2.0']) / (bb['BBU_20_2.0_2.0'] - bb['BBL_20_2.0_2.0'])

    df['ATR14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    for lag in [1, 2, 3, 5]:
        df[f'return_lag{lag}'] = df['Close'].pct_change(lag)

    df['weekday']       = df.index.dayofweek
    df['volume_change'] = df['Volume'].pct_change()
    df['volume_ratio']  = df['Volume'] / df['Volume'].rolling(5).mean()
    df['body']          = (df['Close'] - df['Open']) / df['Open']
    df['upper_shadow']  = (df['High'] - df[['Close','Open']].max(axis=1)) / df['Open']
    df['lower_shadow']  = (df[['Close','Open']].min(axis=1) - df['Low']) / df['Open']

    stoch = ta.stoch(df['High'], df['Low'], df['Close'])
    df['stoch_k'] = stoch['STOCHk_14_3_3']
    df['stoch_d'] = stoch['STOCHd_14_3_3']

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

    # 直近データ以外で学習
    X_train = X.iloc[:-3]
    y_train = y.iloc[:-3]
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

# 現在の残高を取得
assets = get_assets()
jpy_balance  = 0
btc_balance  = 0

for asset in assets['data']['assets']:
    if asset['asset'] == 'jpy':
        jpy_balance = float(asset['free_amount'])
    if asset['asset'] == 'btc':
        btc_balance = float(asset['free_amount'])

print(f"JPY残高: {jpy_balance:,.0f}円")
print(f"BTC残高: {btc_balance:.6f} BTC")

# 現在のBTC価格を取得
btc_price = get_btc_price()
print(f"現在のBTC価格: {btc_price:,.0f}円")

# シグナルを取得
print("シグナル計算中...")
proba = get_signal()
print(f"上昇確率: {proba:.2%}")

# ===================================================
# 売買判断
# ===================================================
MIN_ORDER_JPY = 1000  # 最低注文金額

if proba >= 0.5:
    # 買いシグナル
    if jpy_balance >= MIN_ORDER_JPY:
        # JPY残高の50%だけ買う(リスク管理)
        order_jpy = jpy_balance * 0.5
        print(f"買いシグナル: {order_jpy:,.0f}円分のBTCを購入")
        result = place_order('buy', order_jpy, btc_price)
        print(f"注文結果: {result}")
    else:
        print(f"買いシグナルだが残高不足({jpy_balance:.0f}円)")
else:
    # 売りシグナル(BTCを持っていれば売る)
    if btc_balance * btc_price >= MIN_ORDER_JPY:
        print(f"売りシグナル: BTC全量({btc_balance:.6f}BTC)を売却")
        sell_amount = btc_balance
        result = place_order('sell', sell_amount * btc_price, btc_price)
        print(f"注文結果: {result}")
    else:
        print(f"売りシグナルだがBTC残高なし")

print("完了")
