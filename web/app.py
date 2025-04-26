import sys
import os
# core/ klasörünü modül yolu olarak ekle (trading_engine modülünü bulabilmesi için)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from flask import Flask, jsonify, request, render_template
from core.trading_engine import TradingEngine
from datetime import datetime
import threading

# .env dosyasını oku
load_dotenv()
API_KEY = os.getenv('API_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
if not API_KEY or not SECRET_KEY:
    raise RuntimeError('API_KEY veya SECRET_KEY tanımlı değil! .env dosyasını kontrol et.')

app = Flask(__name__, static_folder='static', template_folder='templates')
bot = None
bot_thread = None

# --- WEB ARAYÜZÜ ---
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/')
def home():
    return render_template('index.html')

# --- API: Botu Başlat ---
@app.route('/api/start-bot', methods=['POST'])
def api_start_bot():
    global bot, bot_thread
    data = request.get_json(force=True) if request.is_json else request.form.to_dict()
    symbol = data.get('symbol', 'BTC/USDT')
    try:
        bot = TradingEngine(API_KEY, SECRET_KEY)
        bot_thread = threading.Thread(
            target=bot.run,
            kwargs={'symbol': symbol},
            daemon=True
        )
        bot_thread.start()
        return jsonify({'success': True})
    except Exception as e:
        app.logger.exception('Bot başlatılırken hata')
        return jsonify({'success': False, 'error': str(e)}), 500

# --- API: Botu Durdur ---
@app.route('/api/stop-bot', methods=['POST'])
def api_stop_bot():
    global bot
    if bot:
        try:
            bot.stop()
        except Exception:
            pass
        bot = None
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Bot çalışmıyor'}), 400

# --- API: Piyasa Verisi ---
@app.route('/api/market-data', methods=['GET'])
def api_market_data():
    symbol = request.args.get('symbol', 'BTC/USDT')
    if not bot:
        return jsonify({'error': 'Bot çalışmıyor'}), 400
    try:
        # Core sinyaller ve makro skoru
        sig = bot.get_current_signals(symbol)
        macro_score = bot.get_macro_analysis(symbol)['score']

        # OHLCV verisi (son 50 dakika)
        ohlcv = bot.exchange.fetch_ohlcv(symbol, '1m', limit=50)
        labels = [datetime.fromtimestamp(c[0]/1000).strftime('%H:%M') for c in ohlcv]
        prices = [c[4] for c in ohlcv]

        # Chart data
        chartData = {
            'labels': labels,
            'prices': prices,
            'buySignals': [None] * len(labels),
            'sellSignals': [None] * len(labels)
        }

        # Dağılım verisi: likidite vs makro
        liq = bot.get_orderbook_analysis(symbol)['liquidity']
        distributionData = {
            'liquidity': liq,
            'macro': macro_score
        }

        # lastSignal'i stringe çevir
        last_sig = 'BUY' if sig['signal'] > 0.7 else 'SELL' if sig['signal'] < 0.3 else 'NEUTRAL'

        return jsonify({
            'price': sig['current_price'],
            'change24h': sig['price_change'],
            'lastSignal': last_sig,
            'macroScore': macro_score,
            'chartData': chartData,
            'distributionData': distributionData,
            'recentTrades': sig.get('recent_trades', [])
        })
    except Exception as e:
        app.logger.exception('api_market_data error')
        return jsonify({'error': 'Sunucu hatası', 'details': str(e)}), 500

# --- Uygulamayı Başlat ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
