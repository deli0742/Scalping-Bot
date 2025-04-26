$(document).ready(function() {
  // Polyfill for crypto.randomUUID()
  if (!crypto.randomUUID) {
    crypto.randomUUID = function() {
      return ([1e7] + -1e3 + -4e3 + -8e3 + -1e11)
        .replace(/[018]/g, c =>
          (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> (c / 4)).toString(16)
        );
    };
  }

  let currentSymbol = 'BTC/USDT';
  let priceChart = null;
  let distributionChart = null;
  let botRunning = false;
  let marketInterval = null;

  function updatePriceChart(data) {
    const canvas = document.getElementById('price-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (priceChart) priceChart.destroy();
    priceChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.labels,
        datasets: [
          { label: 'Fiyat', data: data.prices, tension: 0.1, fill: false },
          { label: 'Al Sinyalleri', data: data.buySignals, borderColor: '#e74c3c', backgroundColor: '#f1948a', pointRadius: 6, pointStyle: 'triangle', showLine: false },
          { label: 'Sat Sinyalleri', data: data.sellSignals, borderColor: '#f39c12', backgroundColor: '#f9e79f', pointRadius: 6, pointStyle: 'triangle', showLine: false }
        ]
      },
      options: {
        responsive: true,
        interaction: { mode: 'index', intersect: false },
        scales: { x: {}, y: { beginAtZero: false } }
      }
    });
  }

  function updateDistributionChart(data) {
    const canvas = document.getElementById('distribution-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (distributionChart) distributionChart.destroy();
    distributionChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: Object.keys(data),
        datasets: [{ data: Object.values(data), backgroundColor: ['#3498db', '#e74c3c'] }]
      },
      options: {
        responsive: true,
        plugins: { legend: { position: 'bottom' } }
      }
    });
  }

  function updateTradesTable(trades) {
    if (!Array.isArray(trades)) return;
    let html = '';
    trades.forEach(t => {
      html += `<tr>` +
        `<td>${new Date(t.timestamp).toLocaleString()}</td>` +
        `<td>${t.symbol}</td>` +
        `<td class="${t.side === 'buy' ? 'text-success' : 'text-danger'}">${t.side.toUpperCase()}</td>` +
        `<td>${parseFloat(t.price).toFixed(2)}</td>` +
        `<td>${parseFloat(t.amount).toFixed(4)}</td>` +
        `<td>${(parseFloat(t.price) * parseFloat(t.amount)).toFixed(2)}</td>` +
        `<td><span class="badge bg-success">Tamamlandı</span></td>` +
      `</tr>`;
    });
    $('#trades-table').html(html);
  }

  function fetchAndUpdate() {
    if (!botRunning) return;
    $.get(`/api/market-data?symbol=${encodeURIComponent(currentSymbol)}`)
      .done(data => {
        // Stats
        $('#current-price').text(`$${data.price.toFixed(2)}`);
        $('#price-change')
          .text(`${data.change24h.toFixed(2)}%`)
          .toggleClass('text-success', data.change24h >= 0)
          .toggleClass('text-danger', data.change24h < 0);
        $('#last-signal')
          .text(data.lastSignal)
          .removeClass('text-success text-danger text-warning')
          .addClass(
            data.lastSignal === 'BUY' ? 'text-success' :
            data.lastSignal === 'SELL' ? 'text-danger' :
            'text-warning'
          );
        $('#macro-score').text(data.macroScore.toFixed(2));

                        // Charts
        if (data.chartData && Array.isArray(data.chartData.labels)) {
          // Insert last signal marker
          const len = data.chartData.labels.length;
          data.chartData.buySignals = data.chartData.labels.map((_, i) =>
            i === len - 1 && data.lastSignal.toString().toUpperCase().includes('BUY') ? data.chartData.prices[i] : null
          );
          data.chartData.sellSignals = data.chartData.labels.map((_, i) =>
            i === len - 1 && data.lastSignal.toString().toUpperCase().includes('SELL') ? data.chartData.prices[i] : null
          );
          updatePriceChart(data.chartData);
        }
        // Distribution data chart
        if (data.distributionData) {
          updateDistributionChart(data.distributionData);
        }

        // Trades
        updateTradesTable(data.recentTrades);
      })
      .fail(() => console.error('Market data fetch error'));  
  }

  // Start Bot
  $('#start-bot').click(function() {
    const btn = $(this);
    btn.prop('disabled', true).html('<span class="spinner-border spinner-border-sm"></span> Başlatılıyor...');
    $.ajax({
      url: '/api/start-bot',
      method: 'POST',
      contentType: 'application/json',
      data: JSON.stringify({ symbol: currentSymbol }),
      success(res) {
        if (res.success) {
          botRunning = true;
          $('#bot-status').removeClass('bg-danger').addClass('bg-success').text('ÇALIŞIYOR');
          $('#stop-bot').prop('disabled', false);
          fetchAndUpdate();
          marketInterval = setInterval(fetchAndUpdate, 5000);
        } else {
          alert('Bot başlatılamadı: ' + (res.error || ''));
        }
      },
      error(xhr) { alert('Sunucu hatası: ' + xhr.responseText); },
      complete() { btn.prop('disabled', false).html('<i class="bi bi-play-fill"></i> Başlat'); }
    });
  });

  // Stop Bot
  $('#stop-bot').click(function() {
    const btn = $(this);
    btn.prop('disabled', true).html('<span class="spinner-border spinner-border-sm"></span> Durduruluyor...');
    $.post('/api/stop-bot')
      .done(res => {
        if (res.success) {
          botRunning = false;
          $('#bot-status').removeClass('bg-success').addClass('bg-danger').text('DURDURULDU');
          $('#start-bot').prop('disabled', false);
          clearInterval(marketInterval);
        } else {
          alert('Bot durdurulamadı: ' + (res.error || ''));
        }
      })
      .fail(() => alert('Sunucu hatası'))
      .always(() => btn.prop('disabled', false).html('<i class="bi bi-stop-fill"></i> Durdur'));
  });

  // Symbol change
  $('#symbol-select').change(function() {
    currentSymbol = $(this).val();
    if (botRunning) fetchAndUpdate();
  });
});
