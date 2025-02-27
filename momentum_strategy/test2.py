import time
import hmac
import hashlib
import requests
import base64
import json

api_key = "66ba4dd1d3e67a000108330f"
api_secret = "c5465410-7eca-43a4-86ac-c63c0e9b8da5"
api_passphrase = "buhyabuhna" 

# KuCoin API URL

def generate_headers(endpoint, method="GET", body=""):
    now = str(int(time.time() * 1000))  # KuCoin requires timestamp in milliseconds
    str_to_sign = now + method + endpoint + body
    signature = base64.b64encode(hmac.new(api_secret.encode(), str_to_sign.encode(), hashlib.sha256).digest()).decode()
    passphrase = base64.b64encode(hmac.new(api_secret.encode(), api_passphrase.encode(), hashlib.sha256).digest()).decode()
    
    headers = {
        "kc-api-key": api_key,
        "kc-api-sign": signature,
        "kc-api-timestamp": now,
        "kc-api-passphrase": passphrase,
        "kc-api-key-version": "2",
        "content-type": "application/json"
    }
    return headers

def get_subaccount_details(sub_user_id):
    base_url = "https://api.kucoin.com"
    endpoint = f"/api/v1/sub-accounts/{sub_user_id}"
    url = base_url + endpoint
    headers = generate_headers(endpoint)
    response = requests.get(url, headers=headers)
    return response.json()

def place_market_order(sub_user_id, symbol, side, size):
    base_url = "https://api.kucoin.com"
    endpoint = "/api/v1/orders"
    url = base_url + endpoint
    body = json.dumps({
        "clientOid": str(int(time.time() * 1000)),
        "side": side,
        "symbol": symbol,
        "type": "market",
        "size": size,
        "subUserId": sub_user_id
    })
    headers = generate_headers(endpoint, method="POST", body=body)
    response = requests.post(url, headers=headers, data=body)
    return response.json()

# Replace with your actual sub_user_id
sub_user_id = "676fd1779377d60001a15a92"
subaccount_details = get_subaccount_details(sub_user_id)
print(json.dumps(subaccount_details, indent=4))

# # Example usage for placing buy and sell orders
# symbol = "XRP-USDT"
# buy_size = "1"



# buy_order = place_market_order(sub_user_id, symbol, "buy", buy_size)
# print("Buy Order:", json.dumps(buy_order, indent=4))

# sell_order = place_order(sub_user_id, symbol, "sell", sell_price, sell_size)
# print("Sell Order:", json.dumps(sell_order, indent=4))