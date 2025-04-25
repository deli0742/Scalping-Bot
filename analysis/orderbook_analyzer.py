class OrderbookAnalyzer:
    def calculate_liquidity(self, orderbook, depth_levels=[0.1, 0.5, 1.0]):
        """
        Price'ın %0.1, %0.5 ve %1 uzağındaki likiditeyi analiz eder
        """
        current_price = (orderbook['bids'][0][0] + orderbook['asks'][0][0])/2
        results = {}
        
        for level in depth_levels:
            price_range = current_price * level
            bid_vol = sum([x[1] for x in orderbook['bids'] if x[0] >= current_price - price_range])
            ask_vol = sum([x[1] for x in orderbook['asks'] if x[0] <= current_price + price_range])
            
            results[f'bid_{level}%'] = bid_vol
            results[f'ask_{level}%'] = ask_vol
        
        return results