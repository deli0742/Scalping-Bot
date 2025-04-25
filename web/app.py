from flask import Flask, jsonify, request
from core.trading_engine import TradingEngine
import threading

app = Flask(__name__)
bot = None
bot_thread = None

@app.route('/start', methods=['POST'])
def start_bot():
    global bot, bot_thread
    
    config = {
        'symbol': request.json.get('symbol', 'BTC/USDT'),
        'risk': request.json.get('risk', 1.0),
        'strategy': request.json.get('strategy', 'hybrid')
    }
    
    bot = TradingEngine(API_KEY, SECRET_KEY)
    bot_thread = threading.Thread(target=bot.run, kwargs=config)
    bot_thread.start()
    
    return jsonify({'status': 'running'})

@app.route('/analytics')
def get_analytics():
    return jsonify({
        'orderbook': bot.get_orderbook_analysis(),
        'macro': bot.get_macro_analysis(),
        'signals': bot.get_current_signals()
    })