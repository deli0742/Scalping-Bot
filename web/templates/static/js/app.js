$(document).ready(function() {
    // Global variables
    let currentSymbol = 'BTC/USDT';
    let priceChart = null;
    let distributionChart = null;
    
    // Initialize
    updateMarketData();
    setInterval(updateMarketData, 5000);
    
    // Event listeners
    $('#symbol-select').change(function() {
        currentSymbol = $(this).val();
        updateMarketData();
    });
    
    $('#start-bot').click(startBot);
    $('#stop-bot').click(stopBot);
    
    // Functions
    function updateMarketData() {
        $.get(`/api/market-data?symbol=${currentSymbol}`, function(data) {
            // Update price display
            $('#current-price').text(`$${data.price.toFixed(2)}`);
            $('#price-change').text(`${data.change24h.toFixed(2)}%`);
            $('#price-change').toggleClass('text-success', data.change24h >= 0);
            $('#price-change').toggleClass('text-danger', data.change24h < 0);
            
            // Update signal
            $('#last-signal').text(data.lastSignal);
            $('#last-signal').removeClass('text-success text-danger text-warning');
            if(data.lastSignal === 'BUY') {
                $('#last-signal').addClass('text-success');
            } else if(data.lastSignal === 'SELL') {
                $('#last-signal').addClass('text-danger');
            } else {
                $('#last-signal').addClass('text-warning');
            }
            
            // Update macro score
            $('#macro-score').text(data.macroScore.toFixed(2));
            
            // Update charts
            updatePriceChart(data.chartData);
            updateDistributionChart(data.distributionData);
            
            // Update trades table
            updateTradesTable(data.recentTrades);
        });
    }
    
    function startBot() {
        $('#start-bot').prop('disabled', true).html('<span class="spinner-border spinner-border-sm" role="status"></span> Başlatılıyor...');
        
        $.post('/api/start-bot', { symbol: currentSymbol }, function(response) {
            if(response.success) {
                $('#bot-status').removeClass('bg-danger').addClass('bg-success').text('ÇALIŞIYOR');
                $('#start-bot').prop('disabled', true);
                $('#stop-bot').prop('disabled', false);
            }
            $('#start-bot').prop('disabled', false).html('<i class="bi bi-play-fill"></i> Başlat');
        });
    }
    
    function stopBot() {
        $('#stop-bot').prop('disabled', true).html('<span class="spinner-border spinner-border-sm" role="status"></span> Durduruluyor...');
        
        $.post('/api/stop-bot', function(response) {
            if(response.success) {
                $('#bot-status').removeClass('bg-success').addClass('bg-danger').text('DURDURULDU');
                $('#start-bot').prop('disabled', false);
                $('#stop-bot').prop('disabled', true);
            }
            $('#stop-bot').prop('disabled', false).html('<i class="bi bi-stop-fill"></i> Durdur');
        });
    }
    
    function updatePriceChart(data) {
        const ctx = document.getElementById('price-chart').getContext('2d');
        
        if(priceChart) {
            priceChart.destroy();
        }
        
        priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [
                    {
                        label: 'Fiyat',
                        data: data.prices,
                        borderColor: '#3498db',
                        tension: 0.1,
                        fill: false
                    },
                    {
                        label: 'Al Sinyalleri',
                        data: data.buySignals,
                        borderColor: '#2ecc71',
                        backgroundColor: 'rgba(46, 204, 113, 0.1)',
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        pointStyle: 'triangle',
                        showLine: false
                    },
                    {
                        label: 'Sat Sinyalleri',
                        data: data.sellSignals,
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        pointStyle: 'triangle',
                        showLine: false
                    }
                ]
            },
            options: {
                responsive: true,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            }
        });
    }
    
    function updateTradesTable(trades) {
        let html = '';
        
        trades.forEach(trade => {
            const typeClass = trade.side === 'buy' ? 'text-success' : 'text-danger';
            const typeIcon = trade.side === 'buy' ? 'bi-arrow-up-circle' : 'bi-arrow-down-circle';
            
            html += `
                <tr>
                    <td>${new Date(trade.timestamp).toLocaleString()}</td>
                    <td>${trade.symbol}</td>
                    <td class="${typeClass}"><i class="bi ${typeIcon}"></i> ${trade.side.toUpperCase()}</td>
                    <td>${parseFloat(trade.price).toFixed(2)}</td>
                    <td>${parseFloat(trade.amount).toFixed(4)}</td>
                    <td>${(parseFloat(trade.price) * parseFloat(trade.amount)).toFixed(2)}</td>
                    <td><span class="badge bg-success">Tamamlandı</span></td>
                </tr>
            `;
        });
        
        $('#trades-table').html(html);
    }
});