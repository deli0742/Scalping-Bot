$(document).ready(function() {
    $('#login-form').submit(function(e) {
        e.preventDefault();
        
        const apiKey = $('#api-key').val();
        const apiSecret = $('#api-secret').val();
        const symbol = $('#symbol').val();
        const timeframe = $('#timeframe').val();
        
        // Simple validation
        if(!apiKey || !apiSecret) {
            alert('Lütfen API bilgilerini giriniz');
            return;
        }
        
        // Save to session and redirect
        $.post('/login', {
            api_key: apiKey,
            secret_key: apiSecret,
            symbol: symbol,
            timeframe: timeframe
        }, function() {
            window.location.href = '/';
        }).fail(function() {
            alert('Giriş başarısız. API bilgilerini kontrol edin.');
        });
    });
});