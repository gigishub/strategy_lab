import websocket
import json
import logging
from datetime import datetime
import threading
import queue
import time
import requests
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConnectionMetrics:
    message_count: int = 0
    error_count: int = 0
    last_message_at: float = 0

class KucoinCandlestickWebSocket:
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.api_url = "https://api.kucoin.com"
        self.ws = None
        self.running = False
        self.data_queue = queue.Queue(maxsize=1000)
        self.metrics = ConnectionMetrics()
        self.token_timestamp = 0
        self.token_refresh_interval = 23 * 3600  # Refresh 1 hour before expiry
        
    def get_token(self) -> str:
        response = requests.post(f"{self.api_url}/api/v1/bullet-public")
        response.raise_for_status()
        return response.json()['data']['token']
        
    def start(self):
        token = self.get_token()
        self.token_timestamp = time.time()
        self.start_connection(token)

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if data.get('type') == 'message' and 'data' in data:
                self.metrics.message_count += 1
                self.metrics.last_message_at = time.time()
                
                candlestick_data = data['data']
                candlestick_data['time_received'] = datetime.now().isoformat()
                
                try:
                    self.data_queue.put_nowait(candlestick_data)
                except queue.Full:
                    logger.warning("Queue full, dropping message")
                    
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.metrics.error_count += 1

    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")
        self.metrics.error_count += 1

    def on_open(self, ws):
        subscribe_message = {
            "type": "subscribe",
            "topic": f"/market/candles:{self.symbol}_{self.timeframe}",
            "privateChannel": False,
            "response": True
        }
        ws.send(json.dumps(subscribe_message))
        
    def start_connection(self, token):
        self.ws = websocket.WebSocketApp(
            f"wss://ws-api-spot.kucoin.com/?token={token}",
            on_message=self.on_message,
            on_error=self.on_error,
            on_open=self.on_open
        )
        
        self.running = True
        ws_thread = threading.Thread(
            target=self.ws.run_forever,
            kwargs={'ping_interval': 20, 'ping_timeout': 10}
        )
        ws_thread.daemon = True
        ws_thread.start()

    def refresh_token_inline(self):
        """Attempt to refresh token without disconnecting"""
        try:
            new_token = self.get_token()
            refresh_message = {
                "type": "update_token",
                "newToken": new_token
            }
            self.ws.send(json.dumps(refresh_message))
            self.token_timestamp = time.time()
            logger.info("Token refreshed inline successfully")
            return True
        except Exception as e:
            logger.error(f"Inline token refresh failed: {e}")
            return False

    def refresh_connection(self):
        """Refresh connection with new token"""
        if time.time() - self.token_timestamp >= self.token_refresh_interval:
            # Try inline refresh first
            if not self.refresh_token_inline():
                # Fall back to full reconnection if inline refresh fails
                logger.info("Falling back to full reconnection")
                self.stop()
                self.start()

    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()
        logger.info(f"Connection closed. Messages: {self.metrics.message_count}, Errors: {self.metrics.error_count}")

    def get_data(self, timeout=None):
        self.refresh_connection()
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None

def main():
    ws_client = KucoinCandlestickWebSocket("BTC-USDT", "1min")
    try:
        ws_client.start()
        while True:
            data = ws_client.get_data(timeout=1.0)
            if data:
                print(json.dumps(data, indent=2))
    except KeyboardInterrupt:
        ws_client.stop()

if __name__ == "__main__":
    main()