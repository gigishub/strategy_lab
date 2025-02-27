import ccxt

# API credentials
api_key = '67bdadee817fd6000188774f'
api_secret = '6b4f04e7-9a9e-4ce8-b8e4-301caf4efaf1'
password = 'checkcheck'
sub_user_id = '676fd1779377d60001a15a92'

# Initialize the exchange
exchange = ccxt.kucoin({
    'apiKey': api_key,
    'secret': api_secret,
    'password': password,

})


# Define the symbol and amount
symbol = 'XRP/USDT'
amount = 1

# Create a market buy order
order = exchange.create_market_buy_order(symbol, amount)

# Print the order details
print(order)