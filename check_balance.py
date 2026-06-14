# ===================================================
# bitbank 残高確認スクリプト
# APIキーが正しく設定されているかの確認用
# ===================================================
import os
import time
import hmac
import hashlib
import requests
import json

API_KEY    = os.environ.get('BITBANK_API_KEY')
API_SECRET = os.environ.get('BITBANK_API_SECRET')

def get_balance():
    endpoint = 'https://api.bitbank.cc'
    path = '/v1/user/assets'
    nonce = str(int(time.time() * 1000))
    message = nonce + path
    signature = hmac.new(
        API_SECRET.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    headers = {
        'ACCESS-KEY': API_KEY,
        'ACCESS-NONCE': nonce,
        'ACCESS-SIGNATURE': signature,
        'Content-Type': 'application/json'
    }

    response = requests.get(endpoint + path, headers=headers)
    return response.json()

result = get_balance()

if result.get('success') == 1:
    print("✅ API接続成功")
    assets = result['data']['assets']
    for asset in assets:
        free = float(asset['free_amount'])
        if free > 0:
            print(f"  {asset['asset']}: {free}")
else:
    print("❌ API接続失敗")
    print(result)
